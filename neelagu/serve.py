"""Neelagu <-> TradingAgents HTTP sidecar.

Neelagu (the Node brain) can hand a per-instrument decision to this process over HTTP instead of using its own
in-brain swarm. This wraps the TradingAgents LangGraph pipeline behind two endpoints:

    GET  /health   -> liveness + whether a model key is set (Neelagu shows a "connected" dot)
    POST /decide   -> run the full desk for ONE instrument, return a Neelagu-schema verdict

The matching client on the Neelagu side is brain/src/adapters/tradingagents.js. The whole path is OFF unless the
owner sets TRADINGAGENTS_URL in Neelagu AND turns on the per-account "Research Desk" toggle, so this sidecar being
up changes nothing on its own.

Run it:
    pip install -e .            # TradingAgents itself (from the repo root)
    pip install -r neelagu/requirements.txt
    export GROQ_API_KEY=...     # or OPENAI_API_KEY / ANTHROPIC_API_KEY, matching TA_LLM_PROVIDER
    export TA_LLM_PROVIDER=groq TA_DEEP_MODEL=openai/gpt-oss-120b TA_QUICK_MODEL=openai/gpt-oss-20b
    uvicorn neelagu.serve:app --host 0.0.0.0 --port 8787
Then in Neelagu set TRADINGAGENTS_URL=http://<host>:8787 and flip Research Desk on.

NOTE — the multi-agent graph is several LLM calls per instrument; the free Groq tier cannot carry it (proved in
testing). A paid model key is required to actually run /decide. /health works without one.
"""

import copy
import os
import logging

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from .verdict import format_verdict, signal_to_dir

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("neelagu.serve")

# Neelagu asset class -> TradingAgents asset_type. The patched context.py (asset-class-awareness) reads asset_type
# in ("commodity","index","forex") directly and resolves everything else from the ticker shape, so metals/energy
# both map to "commodity" and the fundamentals agent is told "no company financials".
_CLASS_MAP = {
    "metal": "commodity", "metals": "commodity",
    "energy": "commodity", "commodity": "commodity", "commodities": "commodity",
    "crypto": "crypto", "cryptocurrency": "crypto",
    "index": "index", "indices": "index",
    "forex": "forex", "fx": "forex",
}


def _asset_type(cls: str) -> str:
    return _CLASS_MAP.get((cls or "").strip().lower(), "stock")


def _build_config() -> dict:
    """Config for TradingAgents from env, falling back to its shipped defaults."""
    from tradingagents.default_config import DEFAULT_CONFIG

    c = copy.deepcopy(DEFAULT_CONFIG)
    prov = os.environ.get("TA_LLM_PROVIDER")
    if prov:
        c["llm_provider"] = prov
    if os.environ.get("TA_BACKEND_URL"):
        c["backend_url"] = os.environ["TA_BACKEND_URL"]
    if os.environ.get("TA_DEEP_MODEL"):
        c["deep_think_llm"] = os.environ["TA_DEEP_MODEL"]
    if os.environ.get("TA_QUICK_MODEL"):
        c["quick_think_llm"] = os.environ["TA_QUICK_MODEL"]
    # keep a desk run lean: one debate round each side unless the operator widens it
    c["max_debate_rounds"] = int(os.environ.get("TA_DEBATE_ROUNDS", c.get("max_debate_rounds", 1)))
    c["max_risk_discuss_rounds"] = int(os.environ.get("TA_RISK_ROUNDS", c.get("max_risk_discuss_rounds", 1)))
    return c


def _has_key() -> bool:
    return bool(
        os.environ.get("GROQ_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
    )


# Build the graph lazily and reuse it: constructing it loads the LLM clients, which need a key. /health must work
# without one, so we never build at import time.
_GRAPH = None


def _graph():
    global _GRAPH
    if _GRAPH is None:
        from tradingagents.graph.trading_graph import TradingAgentsGraph

        _GRAPH = TradingAgentsGraph(debug=False, config=_build_config())
    return _GRAPH


app = FastAPI(title="Neelagu research desk", version="0.1.0")

_TOKEN = os.environ.get("TRADINGAGENTS_TOKEN", "").strip()


def _check_auth(authorization: str | None):
    if not _TOKEN:
        return  # no shared secret configured -> open (fine on a private VPS/localhost)
    if authorization != "Bearer " + _TOKEN:
        raise HTTPException(status_code=401, detail="bad token")


class DecideBody(BaseModel):
    symbol: str
    assetClass: str | None = None
    price: float | None = None
    date: str | None = None
    candles: list | None = None          # optional: Neelagu's own SPOT OHLCV (see "spot data" note in README)
    equity: float | None = None
    positions: list | None = None
    goal: dict | None = None
    minConfidence: float | None = None
    model: str | None = None


@app.get("/health")
def health():
    return {"ok": True, "service": "neelagu-research-desk", "keySet": _has_key()}


@app.post("/decide")
async def decide(body: DecideBody, authorization: str | None = Header(default=None)):
    _check_auth(authorization)
    if not _has_key():
        raise HTTPException(status_code=503, detail="no model key set on the desk")

    import datetime

    date = body.date or datetime.date.today().isoformat()
    asset_type = _asset_type(body.assetClass)
    seed = {
        "symbol": body.symbol,
        "assetClass": body.assetClass,
        "price": body.price,
        "goal": body.goal,
    }

    # TODO (spot data): body.candles carries Neelagu's real SPOT OHLCV. The clean wiring is a custom data vendor
    # that serves these instead of the Yahoo futures proxy (GC=F etc.), which matters for leveraged spot CFDs
    # (basis + the monthly roll gap). Until that vendor is wired the desk fetches its own data for the resolved
    # proxy; the price we pass in seed still grounds the final verdict on the spot price actually traded.

    try:
        graph = _graph()
        # propagate() is blocking (a full graph of LLM calls) -> off the event loop
        final_state, signal = await run_in_threadpool(
            graph.propagate, body.symbol, date, asset_type
        )
    except Exception as e:  # never leak a stack to the brain; a null-ish reply = "no reading this scan"
        log.exception("desk run failed for %s", body.symbol)
        raise HTTPException(status_code=502, detail="desk run failed: " + str(e)[:200])

    llm = getattr(graph, "quick_thinking_llm", None)
    verdict = format_verdict(llm, final_state, signal, seed)

    agents = []
    for key, name in (
        ("fundamentals_report", "fundamentals"),
        ("sentiment_report", "sentiment"),
        ("news_report", "news"),
        ("market_report", "technical"),
    ):
        txt = final_state.get(key)
        if isinstance(txt, str) and txt.strip():
            agents.append({"id": name, "reason": txt.strip()[:400]})

    return {
        "verdict": verdict,
        "decision": signal,
        "dir": signal_to_dir(signal),
        "agents": agents,
        "assetType": asset_type,
    }

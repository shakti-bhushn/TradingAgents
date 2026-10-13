"""Turn a TradingAgents run into Neelagu's verdict schema.

TradingAgents ends a run with a 5-tier *rating* (Buy / Overweight / Hold / Underweight / Sell) and a set of
prose reports (analysts, bull/bear debate, trader plan, risk-manager decision). Neelagu's execution layer, by
contrast, expects a compact JSON *verdict*:

    {dir, confidence, reason, stopPct, tpPct, riskPct, entry, entryPct}

(dir: +1 long / -1 short / 0 flat; stopPct/tpPct: distance from entry in %; riskPct: share of equity to risk;
entry: "now" | "pullback".) That is the SAME schema Neelagu's own in-brain swarm emits, so once we produce it
here every downstream gate on the Neelagu side (account breaker, goal halt, portfolio ceiling, sizing) clamps it
identically — the desk only changes WHO forms the opinion.

The rating fixes direction. A single structured LLM call then reads the desk's own reports and commits the
levels and size consistent with that direction and the caller's growth goal. If that call fails or returns
unparseable JSON we fall back to a conservative deterministic verdict from the rating alone, so a run never
crashes the desk — it just produces a safe, minimal call that Neelagu can still gate.
"""

import json
import re

# TradingAgents' five-tier rating -> Neelagu direction. Hold / REVIEW (unreadable) are no-trades.
_LONG = {"buy", "overweight"}
_SHORT = {"sell", "underweight"}


def signal_to_dir(signal: str) -> int:
    s = (signal or "").strip().lower()
    if s in _LONG:
        return 1
    if s in _SHORT:
        return -1
    return 0


def _rating_base_confidence(signal: str) -> float:
    """A conviction floor implied by the tier itself (strong call vs leaning call)."""
    s = (signal or "").strip().lower()
    if s in {"buy", "sell"}:
        return 0.72
    if s in {"overweight", "underweight"}:
        return 0.58
    return 0.0


_VERDICT_SYS = (
    "You are the lead trader on a desk. Your analysts have finished; their committed call is given as RATING, and "
    "the full reasoning is in the reports below. Convert that into ONE executable order as STRICT JSON, nothing "
    "else — no prose, no code fence. Keys:\n"
    '  "dir": 1 for long, -1 for short, 0 to stand aside (MUST match the RATING: Buy/Overweight=1, '
    "Sell/Underweight=-1, Hold=0),\n"
    '  "confidence": 0..1, your honest conviction in THIS order given the reports,\n'
    '  "stopPct": stop-loss distance from entry as a percent (e.g. 1.5 means 1.5%),\n'
    '  "tpPct": take-profit distance from entry as a percent,\n'
    '  "riskPct": the share of account equity to risk on this one trade as a percent — size it to the GOAL and '
    "your conviction (press when the edge is clean and the goal is behind pace; go light when it is thin),\n"
    '  "entry": "now" to take it at market, or "pullback" to wait for a better price,\n'
    '  "entryPct": if "pullback", how far away in percent (else omit),\n'
    '  "reason": one sentence, the core thesis.\n'
    "If dir is 0, only dir and reason are required. Keep SL/TP consistent with a sane reward:risk. "
    "Return only the JSON object."
)


def _extract_json(text: str) -> dict | None:
    if not isinstance(text, str):
        return None
    # tolerate a stray code fence or leading prose: grab the first {...} balanced-ish block
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _fallback_verdict(signal: str, reason: str) -> dict:
    """A safe minimal verdict from the rating alone, used only when the formatter call fails."""
    d = signal_to_dir(signal)
    if d == 0:
        return {"dir": 0, "reason": reason or "desk: no committed direction"}
    return {
        "dir": d,
        "confidence": _rating_base_confidence(signal) or 0.55,
        "reason": reason or ("desk rating " + str(signal)),
        # conservative defaults; Neelagu still sizes/gates these. Documented as a fallback, not the primary path.
        "stopPct": 1.5,
        "tpPct": 3.0,
        "riskPct": 2.0,
        "entry": "now",
    }


def _reports_blob(final_state: dict, goal, max_chars: int = 9000) -> str:
    """Stitch the desk's reports into one context block for the formatter, newest/most-decisive last."""
    parts = []
    g = goal or {}
    if g:
        parts.append(
            "GROWTH GOAL — progress {p}, target x{t}, days left {d}. Size with this urgency in mind.".format(
                p=g.get("progress"), t=g.get("targetMult"), d=g.get("daysLeft")
            )
        )
    for key, label in (
        ("fundamentals_report", "FUNDAMENTALS"),
        ("sentiment_report", "SENTIMENT"),
        ("news_report", "NEWS"),
        ("market_report", "TECHNICALS"),
        ("investment_plan", "BULL/BEAR DEBATE OUTCOME"),
        ("trader_investment_plan", "TRADER PLAN"),
        ("final_trade_decision", "RISK-MANAGER FINAL DECISION"),
    ):
        v = final_state.get(key)
        if isinstance(v, str) and v.strip():
            parts.append(label + ":\n" + v.strip())
    blob = "\n\n".join(parts)
    return blob[-max_chars:] if len(blob) > max_chars else blob


def format_verdict(llm, final_state: dict, signal: str, seed: dict) -> dict:
    """Produce a Neelagu verdict dict from a finished TradingAgents run. Never raises."""
    reason_hint = ""
    fd = final_state.get("final_trade_decision")
    if isinstance(fd, str):
        reason_hint = fd.strip().split("\n")[0][:200]

    if llm is None:
        return _fallback_verdict(signal, reason_hint)

    user = (
        "RATING: {sig}\nINSTRUMENT: {sym} (asset class: {cls}), live price {price}.\n\n{blob}".format(
            sig=signal,
            sym=seed.get("symbol"),
            cls=seed.get("assetClass"),
            price=seed.get("price"),
            blob=_reports_blob(final_state, seed.get("goal")),
        )
    )
    try:
        from langchain_core.messages import SystemMessage, HumanMessage

        resp = llm.invoke([SystemMessage(content=_VERDICT_SYS), HumanMessage(content=user)])
        obj = _extract_json(getattr(resp, "content", resp))
        if not obj:
            return _fallback_verdict(signal, reason_hint)
        # keep the rating as the source of truth for direction; the formatter fills conviction + levels + size
        obj["dir"] = signal_to_dir(signal)
        if "reason" not in obj and reason_hint:
            obj["reason"] = reason_hint
        return obj
    except Exception:
        return _fallback_verdict(signal, reason_hint)

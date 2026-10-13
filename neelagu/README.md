# Neelagu research-desk sidecar

This folder is the glue that lets **Neelagu** (a Node.js MT5 trading brain) use **TradingAgents** as its
decision engine. It wraps the LangGraph pipeline (analysts → bull/bear debate → trader → risk team → portfolio
manager) behind a tiny HTTP service that speaks Neelagu's verdict schema.

It is **not** part of upstream TradingAgents — it's Neelagu-specific integration code, kept on this fork.

## How the two sides connect

```
Neelagu auto-loop (Node)                         this sidecar (Python)
  computeDecision(symbol)                            POST /decide
   └─ if "Research Desk" ON  ──HTTP──▶  run TradingAgentsGraph.propagate(ticker, date, asset_type)
      and TRADINGAGENTS_URL set            └─ rating (Buy/Sell/Hold) + analyst/debate/risk reports
                                           └─ format_verdict(): one LLM call → Neelagu verdict JSON
   ◀──────────────────────────────────  { verdict:{dir,confidence,stopPct,tpPct,riskPct,entry}, agents:[...] }
   └─ validateAiVerdict() → same gates as the in-brain swarm
      (account breaker, goal halt, portfolio ceiling, sizing) → MT5 order
```

The client on the Neelagu side is `brain/src/adapters/tradingagents.js`. Both must be true for any of this to
run: `TRADINGAGENTS_URL` set in Neelagu **and** the per-account **Research Desk** toggle on (AI System page).
With the URL unset the whole path is inert — Neelagu never calls out, the in-brain swarm runs exactly as before.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness + `keySet` (is a model key present). Works with no key — drives Neelagu's "connected" dot. |
| POST | `/decide` | run the desk for one instrument, return a Neelagu-schema verdict. Needs a model key (503 otherwise). |

`/decide` request body: `{symbol, assetClass, price, date?, candles?, equity?, positions?, goal?}`.
Response: `{verdict:{dir,confidence,reason,stopPct,tpPct,riskPct,entry,entryPct?}, decision, dir, agents, assetType}`.

Asset-class mapping (Neelagu → TradingAgents `asset_type`): metal/energy → `commodity`, crypto → `crypto`,
index → `index`, forex → `forex`, else → `stock`. Combined with the asset-class-awareness patch, gold/oil/indices/
FX no longer get treated as a company with earnings.

## Run it

```bash
# from the repo root
pip install -e .
pip install -r neelagu/requirements.txt

# a model key + which provider/models to use
export GROQ_API_KEY=...            # or OPENAI_API_KEY / ANTHROPIC_API_KEY
export TA_LLM_PROVIDER=groq
export TA_DEEP_MODEL=openai/gpt-oss-120b
export TA_QUICK_MODEL=openai/gpt-oss-20b
# optional: lock the endpoint with a shared secret (set the same value as TRADINGAGENTS_TOKEN in Neelagu)
export TRADINGAGENTS_TOKEN=some-long-random-string

uvicorn neelagu.serve:app --host 0.0.0.0 --port 8787
```

Then in Neelagu (its `.env` / environment): `TRADINGAGENTS_URL=http://<host>:8787` (and the same
`TRADINGAGENTS_TOKEN` if you set one), restart the brain, and turn **Research Desk** on for the account.

Environment knobs: `TA_LLM_PROVIDER`, `TA_BACKEND_URL`, `TA_DEEP_MODEL`, `TA_QUICK_MODEL`,
`TA_DEBATE_ROUNDS` (default 1), `TA_RISK_ROUNDS` (default 1), `TRADINGAGENTS_TOKEN`.

## Honest limitations (read before trusting it with size)

- **A paid model is required.** The multi-agent graph is several LLM calls per instrument; the free Groq tier
  cannot carry it (it hits the per-minute token wall mid-run — proved in testing). `/health` works without a key;
  `/decide` returns 503 until one is set.
- **Spot vs futures data is not fully wired yet.** `/decide` accepts `candles` (Neelagu's real spot OHLCV), but
  the clean fix — a custom data vendor that serves those instead of the Yahoo *futures* proxy (`GC=F`, `CL=F`, …)
  — is still a TODO in `serve.py`. For leveraged spot CFDs the basis and the monthly roll gap matter, so until
  that vendor is wired, treat the desk's exact levels as proxy-derived. The live spot `price` passed in the
  request still grounds the final verdict.
- **No backtest claim.** This is built for forward-testing on a demo account. Past fundamentals/geopolitics don't
  transfer, so a backtest here would mislead. Real validation = watching it trade the demo.
- **The rating fixes direction; one extra LLM call fills levels + size.** If that call fails it falls back to a
  conservative deterministic verdict (dir from the rating, modest SL/TP/risk) so a run never crashes the brain —
  Neelagu still gates and sizes whatever comes back.

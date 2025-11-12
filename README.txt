GOLD BOT v7 — Adaptive Multi-Strategy
====================================

What’s new
----------
- Looser starts for testing; tightens adaptively based on live + logged data
- Adds two strategies: Opening Range Breakout (ORB) and Fair Value Gaps (FVG) alongside Break & Retest (BRRT)
- Rich logs: 
  * trade_log.csv: includes strategy, thresholds, trend, notes
  * signal_log.csv: every strategy decision each tick with diagnostics
  * market_snapshot.csv: heartbeat snapshots (close, ATR, BODY, EMA, spread, tokens)
- Exploratory probes: optional low-risk entries when gates fail to speed up learning

Quick start
-----------
1) Put `gold_bot_full_v7_adaptive_multistrat.py` in your working folder and run with Python that has MetaTrader5 installed.
2) Ensure MT5 is open/logged into the correct account, and symbol `XAUUSDm` exists (adjust SYMBOL if needed).
3) During first 24h TEST_MODE is on by default.
   - More tokens, faster refills, looser ATR/BODY, exploratory signals enabled (low risk).
   - After this window, the bot naturally tightens via thresholds and teacher adjustments.
4) Strategy toggles are at the top: `STRATEGIES_ENABLED = ["BRRT","ORB","FVG"]`.

Key files
---------
- trade_log.csv         — final record of trades (now with richer fields)
- signal_log.csv        — tick-by-tick strategy diagnostics for analytics
- market_snapshot.csv   — heartbeat metrics for environment tracking
- gold_bot_state.json   — runtime/adaptive state
- teacher_state.json    — longer-term learning state

Important knobs
---------------
- TEST_MODE, EXPLORATORY_SIGNAL_RATIO, EXPLORATORY_RISK_FACTOR
- ATR/BODY pctl bases and auto-tighten range
- ORB session definitions (ORB_SESSIONS), FVG params
- HARD_RISK_ZAR_CAP (always-on kill-switch & sizing cap)
- MIN_SL_ATR_X to avoid overly tight stops

Analytics ideas
---------------
- Use signal_log.csv to compute per-strategy WR, avg R, and best/worst micro-sessions (2h buckets).
- Join market_snapshot.csv to see how spread/ATR/body affected outcomes.
- Use trade_log.csv to plot equity curve in ZAR and in R units.

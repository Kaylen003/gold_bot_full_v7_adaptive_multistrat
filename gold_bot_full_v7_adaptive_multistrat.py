# -*- coding: utf-8 -*-
"""
Gold Bot (XAUUSDm) — Adaptive Multi-Strategy (v7)
- Starts loose for testing, then adapts/tightens from data
- Strategies: Break & Retest (BRRT), Opening Range Breakout (ORB), Fair Value Gap (FVG)
- Rich logging for learning (trades + signals + snapshots)
- Cash-aware risk, hard kill switch, staged trailing, loss cooldown
- Africa/Johannesburg timezone by default
"""

import os
import sys
import time
import math
import json
import csv
import warnings
from datetime import datetime, timedelta, timezone

import pytz
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from pandas.errors import ParserWarning

warnings.filterwarnings("ignore", category=ParserWarning)

# =================== OPTIONAL: DIRECT LOGIN (leave None to use terminal session) ===================
MT5_PATH = None
ACCOUNT_LOGIN = None
ACCOUNT_PASSWORD = None
ACCOUNT_SERVER = None

# =================== USER CONFIG ===================

SYMBOL = "XAUUSDm"
TIMEFRAME = mt5.TIMEFRAME_M5
TZ = pytz.timezone("Africa/Johannesburg")

# Test/Explore mode — allows more trades early so we can learn quickly
TEST_MODE = False
TEST_MODE_EXPIRES_HOURS = 24         # after first run; flips to False automatically in state once elapsed
EXPLORATORY_SIGNAL_RATIO = 0.33      # allow ~33% of otherwise-failed gates to pass as probe trades
EXPLORATORY_RISK_FACTOR = 0.35       # scale risk for probe trades (fraction of normal target risk)

# Risk and caps (ZAR)
RISK_PCT_PER_TRADE = 0.0015          # 0.15% of balance as base R
HARD_RISK_ZAR_CAP = 125.0            # hard cap per trade in ZAR (kill-switch & sizing target)
DAILY_R_CAP_R = -3.0                 # stop for the day after net -3R
DAILY_MAX_LOSSES = 4                 # allow one extra loss during testing

# Token bucket — looser defaults for testing; adapts intra-day based on performance
TOKEN_BUCKET_CAPACITY = 10
TOKEN_BUCKET_REFILL_SEC = 60         # ~60s

# ---- Portfolio guards (multi-position controls) ----
MAX_OPEN_TRADES        = 3           # hard cap per symbol
PER_STRATEGY_MAX       = {"BRRT": 1, "ORB": 1, "FVG": 2}  # per-strategy cap
FLOATING_RISK_CAP_ZAR  = 3 * HARD_RISK_ZAR_CAP  # worst-case total if all SLs hit

# Loss-lock window
LOSSLOCK_LOOKBACK_MIN = 60
LOSSLOCK_MAX_TRADES = 8
LOSSLOCK_NET_R_THRESHOLD = -2.5
LOSSLOCK_MAX_CONSEC_LOSSES = 4
PROBE_TRADES_ON_LOCK = 1
MIN_PROBE_RISK_FACTOR = 0.5

# ATR / Body thresholds (percentile-driven)
ATR_PERIOD = 14
ATR_LOOKBACK_CANDLES = 500
ATR_PCTL_BASE = 14     # start looser
BODY_PCTL_BASE = 14
AUTO_TIGHTEN_WAIT_MIN = 5
AUTO_TIGHTEN_STEP = 12
AUTO_TIGHTEN_MAX = 70
AUTO_TIGHTEN_MIN = 8

# --- Entry looseners / tolerances ---
BRRT_TOL_ATR_X = 0.18     # a bit looser
BRRT_RETEST_BARS = 10

# --- ATR/BODY "grace" multipliers ---
_ATR_GRACE  = 0.96
_BODY_GRACE = 0.92

# --- Trend-continuation SCOUT (OPTIONAL) ---
ALLOW_TREND_SCOUT = True
TREND_SCOUT_LOOKBACK = 20
TREND_SCOUT_BODY_MIN_MULT = 0.75
TREND_SCOUT_ATR_MIN_MULT  = 0.75

# Micro-session bias
SESSION_BIAS = True
SESSION_WINDOW_HRS = 2
SESSION_LOOSEN_PCTL = -6
SESSION_TIGHTEN_PCTL = +6

# Trailing / BE
BE_TRIGGER_R = 0.80
LOCK_AT_R = 2.00
LOCK_TO_R = 0.60
CHAND_START_R = 2.00
CHAND_ATR_X_BASE = 6.0
CHAND_ATR_X_MID = 4.0
CHAND_ATR_X_TIGHT = 2.5

# Take-profit mode
TP_MODE = "RUNNER"                  # "RUNNER" or "FIXED_ZAR"
FIXED_ZAR_TP = 120.0

PARTIAL_TP_ZAR      = 400.0  # bank half once floating PnL >= +400 ZAR
PARTIAL_TP_FRACTION = 0.50
HARD_TP_ZAR         = 700.0  # close full at +700 ZAR no matter what

TRAIL_START_CONFIRM_BARS = 1  # 1 closed bar confirmation

# Optional: tighten TP on momentum fade (only if TP_MODE != "RUNNER")
DYN_TP_TIGHTEN_ENABLE = True
DYN_TP_MIN_R          = 0.80
DYN_TP_GIVEBACK_R     = 0.50
DYN_TP_KEEP_R_AHEAD   = 0.20

# Misc
SLIPPAGE_POINTS = 20
LOG_FILE = "trade_log.csv"
SIGNAL_LOG_FILE = "signal_log.csv"
SNAPSHOT_FILE = "market_snapshot.csv"
STATE_FILE = "gold_bot_state.json"
POLL_SEC = 2
MIN_CANDLE_HISTORY = max(ATR_LOOKBACK_CANDLES, 800)

# Separate cadences for scanning vs managing (in-trade)
SIGNAL_POLL_SEC = 2
MANAGE_POLL_SEC = 1

# Debug / heartbeat
DEBUG = True
HEARTBEAT_EVERY_SEC = 45

# ===== Learning / Analytics =====
LEARN_WINDOW_TRADES   = 250   # lookback window for stats
LEARN_MIN_TRADES      = 30    # don't adapt until we have at least this many
LEARN_FILE            = "learn_state.json"
ADAPT_ENABLE          = True  # turn on simple auto-nudges

# ---- Post-trade diagnostics / reports ----
LOSS_REPORT_FILE       = "loss_report.json"
DAILY_REPORT_FILE      = "daily_report.json"
REPORT_TOP_CLUSTERS    = 6
REPORT_MIN_SAMPLES     = 5

# ---- Outcome-weighted gates (learn from our trades) ----
OUTCOME_WEIGHTED_ENABLE = True
OUTCOME_WEIGHTED_BLEND  = 0.30   # 0..1
OUTCOME_WEIGHTED_BINS   = 10     # deciles
OUTCOME_MIN_SAMPLES_BIN = 6
OUTCOME_LOOKBACK_TRADES = 250

# ---- Meta rate-control from rolling WR ----
META_RATE_ENABLE        = True
META_RATE_WINDOW_TRADES = 120
META_RATE_WR_TIGHT      = 0.40
META_RATE_WR_OK         = 0.50
META_RATE_WR_STRONG     = 0.60
META_RATE_MIN_REFILL_S  = 8 * 60
META_RATE_MAX_REFILL_S  = 45 * 60
META_RATE_MAX_CAP       = 4

# Adaptation knobs (small, safe nudges)
NUDGE_BE_TRIGGER_STEP = 0.05
NUDGE_BE_OFFSET_STEP  = 0.02
NUDGE_GIVEBACK_STEP   = 0.05
NUDGE_LOCK_AT_STEP    = 0.05
STRAT_REORDER_ENABLE  = True

# Diagnostics for signal gating
DIAG_SIGNALS = True

# ---- Quick-start clamp controls ----
STARTUP_NEW_BARS = 60
ATR_CLAMP_MULT  = 1.05
BODY_CLAMP_MULT = 1.12

# Spread & SL guards
MAX_SPREAD_POINTS = 250
MIN_SL_ATR_X = 0.55

# Trading windows (local TZ)
TRADE_WINDOWS = [("00:00","23:59")]

# Loss cooldown (after each loss)
LOSS_COOLDOWN_SEC = 12 * 60 if TEST_MODE else 20 * 60

# Strategy toggles
STRATEGIES_ENABLED = ["BRRT", "ORB", "FVG"]

# Soft gating by ATR×Body “green zones” (learned daily)
PREFERRED_BANDS = {}  # e.g., {"BRRT":{"atr":["Q3","Q4"],"body":["Q1","Q2"]}, ...}

# Opening Range config (local time)
ORB_SESSIONS = [
	# (name, start, minutes, min_atr_pctl, with_retest)
	("LDN_AM", "09:00", 30, 15, True),
	("NY_AM" , "15:30", 30, 15, True),
]
ORB_MAX_AGE_MIN = 240     # after range window, signals valid up to 4 hours

# FVG config
FVG_LOOKBACK = 250
FVG_MIN_SIZE_ATR_X = 0.20
FVG_VALID_FOR_BARS = 90
FVG_RETOUCH_TOL_ATR_X = 0.45
FVG_WITH_TREND_ONLY = True

# ---- RUNTIME PROFILE SWITCH (flip to PROD when going live) ----
PROFILE = "TEST"   # "TEST" (learning) or "PROD" (real money)

def apply_profile(name: str):
	"""Central place to tighten/relax knobs when moving to production."""
	global TEST_MODE, EXPLORATORY_SIGNAL_RATIO, EXPLORATORY_RISK_FACTOR
	global RISK_PCT_PER_TRADE, HARD_RISK_ZAR_CAP, DAILY_R_CAP_R, DAILY_MAX_LOSSES
	global TOKEN_BUCKET_CAPACITY, TOKEN_BUCKET_REFILL_SEC
	global MAX_OPEN_TRADES, PER_STRATEGY_MAX, FLOATING_RISK_CAP_ZAR
	global LOSS_COOLDOWN_SEC
	global ATR_PCTL_BASE, BODY_PCTL_BASE
	global STRATEGIES_ENABLED
	global DYN_TP_TIGHTEN_ENABLE

	if name.upper() == "PROD":
		# disable probes
		TEST_MODE = False
		EXPLORATORY_SIGNAL_RATIO = 0.0
		EXPLORATORY_RISK_FACTOR  = 1.0

		# risk & caps
		RISK_PCT_PER_TRADE = 0.0010
		HARD_RISK_ZAR_CAP  = 100.0
		DAILY_R_CAP_R      = -2.0
		DAILY_MAX_LOSSES   = 3

		# throttling
		TOKEN_BUCKET_CAPACITY = 2
		TOKEN_BUCKET_REFILL_SEC = 30 * 60

		# portfolio guards
		MAX_OPEN_TRADES = 2
		PER_STRATEGY_MAX.update({"BRRT": 1, "ORB": 1, "FVG": 1})
		FLOATING_RISK_CAP_ZAR = 2 * HARD_RISK_ZAR_CAP

		# cool-down
		LOSS_COOLDOWN_SEC = 20 * 60

		# a bit tighter starting percentiles
		ATR_PCTL_BASE  = max(ATR_PCTL_BASE, 22)
		BODY_PCTL_BASE = max(BODY_PCTL_BASE, 22)

		DYN_TP_TIGHTEN_ENABLE = True
	else:
		# TEST defaults already set above
		pass

# Apply the chosen profile immediately
apply_profile(PROFILE)

# =================== CSV / LOGGING UTILITIES ===================

EXPECTED_COLS = [
	"timestamp","symbol","strategy","direction","entry","sl","tp",
	"risk_zar","R","lot","ticket","exit_price","profit_zar",
	"max_fav_R","max_adverse_R","reason_exit",
	"atr","atr_th","body","body_th","ema_trend","notes"
]

SIGNAL_COLS = [
	"timestamp","symbol","strategy","dir","status","reason",
	"atr","atr_th","body","body_th","ema50_gt_200","ema_slope50","ema_slope200",
	"price","swH","swL","orb_name","orb_start","orb_end","fvg_gap","fvg_age","exploratory"
]

SNAPSHOT_COLS = [
	"timestamp","symbol","close","atr","body","ema50","ema200","trend","spread_pts","tokens","losslock","paused"
]

def _quote(s: str) -> str:
	s = str(s).replace('"', '""')
	return f'"{s}"'

def ensure_csv_header(path: str, cols: list[str]):
	if not os.path.exists(path):
		with open(path, "w", encoding="utf-8", newline="") as f:
			w = csv.writer(f, lineterminator="\n")
			w.writerow(cols)
			f.flush(); os.fsync(f.fileno())

def sanitize_trade_log_inplace(path: str, expected_cols: list[str] = EXPECTED_COLS):
	"""
	Repairs the header and row lengths if needed.
	IMPORTANT: uses real newlines '\\n' (not escaped).
	"""
	if not os.path.exists(path):
		return
	try:
		with open(path, "r", encoding="utf-8", newline="") as f:
			lines = f.read().splitlines()
		if not lines:
			return
		header = lines[0].split(",")
		if header != expected_cols:
			lines[0] = ",".join(expected_cols)

		fixed = [lines[0]]
		exp = len(expected_cols)
		for line in lines[1:]:
			parts = line.split(",")
			if len(parts) == exp:
				fixed.append(line); continue
			if len(parts) > exp:
				# too many commas → quote the overflowing tail into the last column
				head = parts[:exp-1]
				tail = ",".join(parts[exp-1:])
				head.append('"' + tail.replace('"', '""') + '"')
				fixed.append(",".join(head))
			else:
				# pad short rows
				parts += [""] * (exp - len(parts))
				fixed.append(",".join(parts))

		tmp = path + ".fixed.tmp"
		with open(tmp, "w", encoding="utf-8", newline="") as f:
			f.write("\n".join(fixed) + "\n")
			f.flush(); os.fsync(f.fileno())
		os.replace(tmp, path)
	except Exception as e:
		if DEBUG:
			print(f"⚠️ trade_log sanitize skipped: {e}")

def log_trade(row_dict):
	ensure_csv_header(LOG_FILE, EXPECTED_COLS)
	with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(
			f, fieldnames=EXPECTED_COLS, quoting=csv.QUOTE_MINIMAL, escapechar="\\", extrasaction="ignore"
		)
		if "reason_exit" in row_dict and row_dict["reason_exit"] is None:
			row_dict["reason_exit"] = ""
		writer.writerow(row_dict)
		f.flush(); os.fsync(f.fileno())

def log_signal(row):
	ensure_csv_header(SIGNAL_LOG_FILE, SIGNAL_COLS)
	with open(SIGNAL_LOG_FILE, "a", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=SIGNAL_COLS)
		writer.writerow(row)
		f.flush(); os.fsync(f.fileno())

def log_snapshot(row):
	ensure_csv_header(SNAPSHOT_FILE, SNAPSHOT_COLS)
	with open(SNAPSHOT_FILE, "a", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=SNAPSHOT_COLS)
		writer.writerow(row)
		f.flush(); os.fsync(f.fileno())

def read_trades_df():
	sanitize_trade_log_inplace(LOG_FILE, EXPECTED_COLS)
	if not os.path.exists(LOG_FILE):
		return pd.DataFrame(columns=EXPECTED_COLS)
	try:
		df = pd.read_csv(LOG_FILE, engine="python", on_bad_lines="skip")
		for c in EXPECTED_COLS:
			if c not in df.columns:
				df[c] = np.nan
		return df[EXPECTED_COLS]
	except Exception as e:
		if DEBUG:
			print(f"⚠️ trade_log read failed: {e} — returning empty frame")
		return pd.DataFrame(columns=EXPECTED_COLS)

def last_trades_window(df, minutes=60, max_n=LOSSLOCK_MAX_TRADES):
	if df.empty:
		return df
	try:
		df2 = df.copy()
		df2["timestamp"] = to_localized(df2["timestamp"])
		cutoff = pd.Timestamp.now(tz=TZ) - pd.Timedelta(minutes=minutes)
		win = df2[df2["timestamp"] >= cutoff].tail(max_n)
		return win
	except Exception:
		return pd.DataFrame(columns=df.columns)
# =================== LEARNING / ANALYTICS ===================

def _safe_float(x, default=0.0):
	try:
		v = float(x)
		if not (v == v):  # NaN
			return default
		return v
	except Exception:
		return default

def read_learn_state():
	if not os.path.exists(LEARN_FILE):
		return {}
	try:
		with open(LEARN_FILE, "r", encoding="utf-8") as f:
			return json.load(f)
	except Exception:
		return {}

def write_learn_state(obj: dict):
	try:
		with open(LEARN_FILE, "w", encoding="utf-8") as f:
			json.dump(obj, f, indent=2)
	except Exception:
		pass

def _mk_flags(row: pd.Series):
	"""Derive normalized flags used for 'false positive' fingerprints."""
	atr  = _safe_float(row.get("atr"), 0.0)
	atr_th = _safe_float(row.get("atr_th"), 0.0)
	body = _safe_float(row.get("body"), 0.0)
	body_th = _safe_float(row.get("body_th"), 0.0)
	ema_trend = str(row.get("ema_trend", "")).upper()
	# Relative strengths
	atr_rel  = atr / max(atr_th, 1e-9)
	body_rel = body / max(body_th, 1e-9)
	# Flags
	weak_atr  = atr_rel < 1.0
	weak_body = body_rel < 1.0
	flat_trend = (ema_trend == "FLAT")
	up_trend   = (ema_trend == "UP")
	down_trend = (ema_trend == "DOWN")
	return {
		"atr_rel": atr_rel,
		"body_rel": body_rel,
		"weak_atr": weak_atr,
		"weak_body": weak_body,
		"flat_trend": flat_trend,
		"up_trend": up_trend,
		"down_trend": down_trend
	}

def _fingerprints(df: pd.DataFrame):
	"""Return conditions that are overrepresented in losing trades vs winners."""
	if df.empty:
		return {}
	d = df.copy().tail(LEARN_WINDOW_TRADES)
	d["R"] = pd.to_numeric(d["R"], errors="coerce").fillna(0.0)
	d["is_win"] = d["R"] > 0.0
	# Build flags per row
	flags = d.apply(_mk_flags, axis=1, result_type="reduce")
	# Attach back
	d = pd.concat([d.reset_index(drop=True), pd.DataFrame(list(flags))], axis=1)
	losers = d[~d["is_win"]]
	winners = d[d["is_win"]]
	def rate(group, col):
		if group.empty: return 0.0
		return float(group[col].mean())
	# Overrep = loss_rate - win_rate for boolean flags; and medians for continuous
	bool_cols = ["weak_atr","weak_body","flat_trend","up_trend","down_trend"]
	cont_cols = ["atr_rel","body_rel"]
	overrep = {}
	for c in bool_cols:
		overrep[c] = rate(losers, c) - rate(winners, c)
	med = {}
	for c in cont_cols:
		med[f"{c}_loss_med"] = float(losers[c].median()) if not losers.empty else float("nan")
		med[f"{c}_win_med"]  = float(winners[c].median()) if not winners.empty else float("nan")
		med[f"{c}_delta_med"]= (med[f"{c}_loss_med"] - med[f"{c}_win_med"]) if (med[f"{c}_loss_med"]==med[f"{c}_loss_med"] and med[f"{c}_win_med"]==med[f"{c}_win_med"]) else float("nan")
	# Strategy skew
	strat_exp = {s: float(losers[d["strategy"]==s]["R"].mean()) - float(winners[d["strategy"]==s]["R"].mean())
				 for s in ["BRRT","ORB","FVG"]}
	return {"overrep_flags": overrep, "medians": med, "strat_skew": strat_exp}

def _session_block_from_ts(ts_series: pd.Series, tz=TZ, width_hours=2):
	t = pd.to_datetime(ts_series, errors="coerce", utc=True).dt.tz_convert(tz)
	hh = (t.dt.hour // width_hours) * width_hours
	return (hh.astype(int).astype(str).str.zfill(2) + ":00-"
			+ (hh + width_hours).astype(int).astype(str).str.zfill(2) + ":00")

def _q_label(values: pd.Series, x: float):
	if values is None or values.empty or not np.isfinite(x): return "NA"
	q25, q50, q75 = np.nanpercentile(values.dropna(), [25, 50, 75])
	if x < q25: return "Q1"
	if x < q50: return "Q2"
	if x < q75: return "Q3"
	return "Q4"

def build_feature_frame(trades_df: pd.DataFrame) -> pd.DataFrame:
	"""
	Returns a small feature set to study losing patterns.
	Uses columns you already log: strategy, R, atr, atr_th, body, body_th, ema_trend, timestamp.
	"""
	if trades_df is None or trades_df.empty:
		return pd.DataFrame()

	d = trades_df.copy().tail(LEARN_WINDOW_TRADES)
	# Coerce numerics
	for c in ("R","atr","atr_th","body","body_th","risk_zar","profit_zar","max_fav_R","max_adverse_R"):
		d[c] = pd.to_numeric(d.get(c, 0), errors="coerce")
	# Derived features
	d["atr_rel"]  = d["atr"]  / d["atr_th"].replace(0, np.nan)
	d["body_rel"] = d["body"] / d["body_th"].replace(0, np.nan)
	d["atr_rel"]  = d["atr_rel"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
	d["body_rel"] = d["body_rel"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

	# Session bucket
	d["session_block"] = _session_block_from_ts(d["timestamp"])
	# Quartile labels at entry context
	atr_vals, body_vals = d["atr"].copy(), d["body"].copy()
	d["atr_q"]  = [ _q_label(atr_vals,  x) for x in d["atr"].values ]
	d["body_q"] = [ _q_label(body_vals, x) for x in d["body"].values ]
	# Outcome
	d["is_win"] = d["R"] > 0.0
	# Safe string columns
	d["strategy"]  = d["strategy"].astype(str).str.upper()
	d["ema_trend"] = d["ema_trend"].astype(str).str.upper()
	return d

def analyze_trade_history(log_path=LOG_FILE,
						  out_json=LOSS_REPORT_FILE,
						  top_n=REPORT_TOP_CLUSTERS,
						  min_samples=REPORT_MIN_SAMPLES):
	"""
	Cluster simple contexts and surface *where losses concentrate*:
	  - by (strategy, session_block)
	  - by (strategy, ema_trend)
	  - by (strategy, atr_q, body_q)
	Writes a compact JSON report you can open while the bot runs.
	"""
	df = read_trades_df()
	if df.empty:
		return None

	F = build_feature_frame(df)
	if F.empty or len(F) < LEARN_MIN_TRADES:
		# still produce a small report so you can inspect during ramp-up
		rep = {"note": "insufficient samples yet", "n": int(len(F))}
		try:
			json.dump(rep, open(out_json, "w"), indent=2)
		except Exception:
			pass
		return rep

	# Helpers
	def agg(block):
		n  = len(block)
		wr = float((block["R"] > 0).mean()) if n else 0.0
		expR = float(np.nanmean(block["R"])) if n else 0.0
		pnl = float(np.nansum(block.get("profit_zar", 0)))
		return {"n": n, "wr": round(wr, 3), "expR": round(expR, 3), "pnl": round(pnl, 2)}

	sections = {}

	# 1) strategy × session
	g1 = (F.groupby(["strategy","session_block"]).apply(agg).reset_index())
	g1["score"] = g1["expR"]  # simplest: most negative expectancy is worst
	worst_1 = g1.sort_values("score").query("n >= @min_samples").head(top_n).to_dict(orient="records")
	sections["strategy_session"] = worst_1

	# 2) strategy × ema_trend
	g2 = (F.groupby(["strategy","ema_trend"]).apply(agg).reset_index())
	g2["score"] = g2["expR"]
	worst_2 = g2.sort_values("score").query("n >= @min_samples").head(top_n).to_dict(orient="records")
	sections["strategy_trend"] = worst_2

	# 3) strategy × ATR/Body quartiles
	g3 = (F.groupby(["strategy","atr_q","body_q"]).apply(agg).reset_index())
	g3["score"] = g3["expR"]
	worst_3 = g3.sort_values("score").query("n >= @min_samples").head(top_n).to_dict(orient="records")
	sections["strategy_atr_body_quartiles"] = worst_3

	# Global summary
	overall = {
		"n": int(len(F)),
		"wr": round(float((F["R"] > 0).mean()), 3),
		"expR": round(float(F["R"].mean()), 3)
	}
	report = {"overall": overall, "clusters": sections}

	try:
		json.dump(report, open(out_json, "w"), indent=2)
	except Exception:
		pass

	return report

def daily_structural_update():
	"""
	Heavyweight once-a-day pass:
	  - refresh loss report (bigger window)
	  - compute outcome-weighted percentile targets (already supported)
	  - derive and persist preferred bands / session bias into learn_state for visibility
	"""
	df = read_trades_df()
	rep = analyze_trade_history(out_json=DAILY_REPORT_FILE)  # large view
	# keep using your existing "learn_update()" and "derive_adaptive_knobs_from_last24h"
	st = read_learn_state()
	st["last_daily_run"] = ts_str()
	st["last_loss_report"] = rep or {}
	write_learn_state(st)

def compute_stats(df: pd.DataFrame):
	"""Overall and per-strategy stats from the last LEARN_WINDOW_TRADES."""
	if df.empty:
		return {}

	d = df.copy().tail(LEARN_WINDOW_TRADES)
	d["R"]             = pd.to_numeric(d.get("R", 0), errors="coerce").fillna(0.0)
	d["max_fav_R"]     = pd.to_numeric(d.get("max_fav_R", 0), errors="coerce").fillna(0.0)
	d["max_adverse_R"] = pd.to_numeric(d.get("max_adverse_R", 0), errors="coerce").fillna(0.0)
	d["strategy"]      = d.get("strategy", "").astype(str).str.upper()

	def agg(group):
		r   = group["R"]
		mfe = group["max_fav_R"]
		mae = group["max_adverse_R"]
		n   = len(group)
		wr  = float((r > 0).mean()) if n else 0.0
		expR = float(r.mean()) if n else 0.0
		be_rate = float((r.between(-0.05, 0.05)).mean())
		giveback_med = float((mfe - r).median())
		return {
			"n": n, "wr": wr, "expR": expR, "be_rate": be_rate,
			"mfe_med": float(mfe.median()), "mae_med": float(mae.median()),
			"giveback_med": giveback_med
		}

	stats = {"overall": agg(d)}
	for strat, g in d.groupby("strategy"):
		stats[strat] = agg(g)

	stats["fingerprints"] = _fingerprints(d)
	return stats

def _bin_index_from_quantiles(vals: pd.Series, bins: int, x: float) -> int:
	"""Return bin index 0..bins-1 based on quantiles of 'vals' that include x."""
	if vals.empty or not np.isfinite(x):
		return -1
	qs = np.linspace(0.0, 1.0, bins + 1)
	cuts = np.quantile(vals.dropna().values, qs)
	# ensure strictly increasing to avoid duplicates
	cuts = np.unique(cuts)
	if len(cuts) <= 2:
		return -1
	idx = np.searchsorted(cuts, x, side="right") - 1
	idx = max(0, min(idx, len(cuts) - 2))
	return int(idx)

def outcome_weighted_percentile_targets(df_trades: pd.DataFrame,
										bins: int = OUTCOME_WEIGHTED_BINS,
										min_samples: int = OUTCOME_MIN_SAMPLES_BIN):
	"""
	From historical trades, find ATR and BODY bins (deciles) with best expectancy,
	then translate to target percentiles (bin center: e.g., bin#*10+5).
	Returns (target_atr_pctl, target_body_pctl) or (None, None) if insufficient data.
	"""
	if df_trades is None or df_trades.empty:
		return None, None

	d = df_trades.copy().tail(OUTCOME_LOOKBACK_TRADES)
	# numeric coerce
	for c in ("atr","body","R"):
		d[c] = pd.to_numeric(d.get(c, 0), errors="coerce")
	d = d.dropna(subset=["atr","body","R"])
	if len(d) < LEARN_MIN_TRADES:
		return None, None

	atr_vals  = d["atr"].astype(float)
	body_vals = d["body"].astype(float)

	# assign bins
	d["atr_bin"]  = [ _bin_index_from_quantiles(atr_vals,  bins, x) for x in d["atr"].values ]
	d["body_bin"] = [ _bin_index_from_quantiles(body_vals, bins, x) for x in d["body"].values ]

	d = d[(d["atr_bin"] >= 0) & (d["body_bin"] >= 0)]
	if d.empty:
		return None, None

	# expectancy per bin
	gb_atr  = d.groupby("atr_bin")["R"].agg(["count","mean"]).reset_index()
	gb_body = d.groupby("body_bin")["R"].agg(["count","mean"]).reset_index()

	gb_atr  = gb_atr[gb_atr["count"]  >= min_samples]
	gb_body = gb_body[gb_body["count"] >= min_samples]

	if gb_atr.empty or gb_body.empty:
		return None, None

	best_atr_bin  = int(gb_atr.sort_values("mean", ascending=False).iloc[0]["atr_bin"])
	best_body_bin = int(gb_body.sort_values("mean", ascending=False).iloc[0]["body_bin"])

	# translate bin -> percentile target as bin mid-point
	bin_width = 100.0 / float(bins)
	targ_atr_pctl  = best_atr_bin  * bin_width + (bin_width / 2.0)
	targ_body_pctl = best_body_bin * bin_width + (bin_width / 2.0)

	# clamp to your allowed range
	targ_atr_pctl  = int(np.clip(targ_atr_pctl,  AUTO_TIGHTEN_MIN, AUTO_TIGHTEN_MAX))
	targ_body_pctl = int(np.clip(targ_body_pctl, AUTO_TIGHTEN_MIN, AUTO_TIGHTEN_MAX))

	return targ_atr_pctl, targ_body_pctl

def adapt_parameters_from_stats(stats: dict):
	
"""Small, safe automatic nudges based on recent performance."""
	global BE_TRIGGER_R, LOCK_AT_R
	try:
		ov = stats.get("overall", {})
		n   = int(ov.get("n", 0))
		if n < LEARN_MIN_TRADES:
			return
		be_rate = float(ov.get("be_rate", 0.0))
		giveback = float(ov.get("giveback_med", 0.0))  # in R
		changed = []

		# 1) Too many flat exits → protect earlier
		if be_rate >= 0.35:
			new_be = max(0.25, BE_TRIGGER_R - NUDGE_BE_TRIGGER_STEP)
			if abs(new_be - BE_TRIGGER_R) >= 1e-9:
				BE_TRIGGER_R = new_be
				changed.append(f"BE_TRIGGER_R->{BE_TRIGGER_R:.2f}")

		# 2) Large givebacks on winners → lock earlier
		if giveback >= 0.60:
			new_lock = max(0.60, LOCK_AT_R - NUDGE_LOCK_AT_STEP)
			if abs(new_lock - LOCK_AT_R) >= 1e-9:
				LOCK_AT_R = new_lock
				changed.append(f"LOCK_AT_R->{LOCK_AT_R:.2f}")

		# 3) Strategy reorder by expectancy (optional)
		if STRAT_REORDER_ENABLE:
			exp = {k: v.get("expR", -999) for k, v in stats.items() if k in ("BRRT","ORB","FVG")}
			if exp:
				order = [k for k,_ in sorted(exp.items(), key=lambda kv: kv[1], reverse=True)]
				enabled = [s for s in STRATEGIES_ENABLED if s in order]
				tail = [s for s in STRATEGIES_ENABLED if s not in order]
				new_enabled = enabled + tail
				if new_enabled != STRATEGIES_ENABLED:
					STRATEGIES_ENABLED[:] = new_enabled
					changed.append(f"STRAT_ORDER->{','.join(STRATEGIES_ENABLED)}")

		if changed:
			print("🧠 Adapt:", " | ".join(changed))

		# Print false-positive fingerprints (human friendly)
		fp = stats.get("fingerprints", {})
		over = fp.get("overrep_flags", {})
		med = fp.get("medians", {})
		print("🔎 False-Positive clues:",
			  f"weak_atrΔ={over.get('weak_atr',0.0):+.2f} weak_bodyΔ={over.get('weak_body',0.0):+.2f} flat_trendΔ={over.get('flat_trend',0.0):+.2f} | ",
			  f"atr_rel med(loss-win)={med.get('atr_rel_delta_med', float('nan')):+.2f} body_rel med(loss-win)={med.get('body_rel_delta_med', float('nan')):+.2f}")

	except Exception as e:
		if DEBUG:
			print(f"⚠️ adapt_parameters_from_stats error: {e}")
# =================== ADAPTIVE DERIVATION FROM LAST 24H ===================

def derive_adaptive_knobs_from_last24h(df_trades: pd.DataFrame | None = None):
	"""Summarize last 24h trades to auto-tighten ATR/body thresholds and session bias."""
	global ATR_PCTL_BASE, BODY_PCTL_BASE, PREFERRED_BANDS

	if df_trades is None:
		df_trades = read_trades_df()

	if df_trades.empty:
		return

	df = df_trades.copy()
	df["timestamp"] = to_localized(df["timestamp"])
	recent = df[df["timestamp"] >= (pd.Timestamp.now(tz=TZ) - pd.Timedelta(hours=24))]
	if recent.empty:
		return

	# Filter numeric
	recent["atr"] = pd.to_numeric(recent.get("atr", 0), errors="coerce")
	recent["body"] = pd.to_numeric(recent.get("body", 0), errors="coerce")
	recent["R"] = pd.to_numeric(recent.get("R", 0), errors="coerce")
	recent = recent.dropna(subset=["R"])

	# Weighted quartile selection by R
	def weighted_pctl(values, weights, pct):
		try:
			s = np.argsort(values)
			cumw = np.cumsum(np.array(weights)[s])
			return float(values[s][np.searchsorted(cumw, pct * cumw[-1])])
		except Exception:
			return np.nan

	# compute percentile targets via outcomes
	targ_atr, targ_body = outcome_weighted_percentile_targets(recent)
	if targ_atr and targ_body:
		ATR_PCTL_BASE = int(round((1 - OUTCOME_WEIGHTED_BLEND) * ATR_PCTL_BASE + OUTCOME_WEIGHTED_BLEND * targ_atr))
		BODY_PCTL_BASE = int(round((1 - OUTCOME_WEIGHTED_BLEND) * BODY_PCTL_BASE + OUTCOME_WEIGHTED_BLEND * targ_body))

	# micro-session bias
	if SESSION_BIAS:
		recent["session_block"] = _session_block_from_ts(recent["timestamp"])
		g = recent.groupby("session_block")["R"].mean().reset_index()
		if not g.empty:
			best = g.sort_values("R", ascending=False).head(1)["session_block"].values[0]
			worst = g.sort_values("R", ascending=True).head(1)["session_block"].values[0]
			PREFERRED_BANDS["best_block"] = best
			PREFERRED_BANDS["worst_block"] = worst

	print(f"🧭 Reorder STRATS -> {','.join(STRATEGIES_ENABLED)}")
	print(f"adapt-24h | ATR_PCTL_BASE={ATR_PCTL_BASE} BODY_PCTL_BASE={BODY_PCTL_BASE} | PREF_BANDS={PREFERRED_BANDS}")


# =================== DAILY RESET / TEST MODE ===================
def daily_reset_if_needed(st):
	"""Resets counters and performs daily heavy update once per day."""
	today = jh_now().strftime("%Y-%m-%d")
	if st.get("day") != today:
		st["day"] = today
		st["token_bucket"] = TOKEN_BUCKET_CAPACITY
		st["last_refill"] = jh_now().timestamp()
		st["no_trade_since"] = jh_now().timestamp()
		st["losslock"] = False
		st["probe_trades_left"] = PROBE_TRADES_ON_LOCK
		st["daily_paused"] = False
		st["session_stats"] = {}
		print(f"♻️ Daily reset @ {ts_str()} | tokens={st['token_bucket']} paused={st['daily_paused']}")
		derive_adaptive_knobs_from_last24h()

def maybe_flip_off_test_mode(state):
	"""Disable TEST_MODE after a defined number of hours."""
	global TEST_MODE
	if not TEST_MODE:
		return False
	first = state.get("first_run_ts", jh_now().timestamp())
	if (jh_now().timestamp() - first) >= TEST_MODE_EXPIRES_HOURS * 3600:
		TEST_MODE = False
		print("🔁 TEST_MODE disabled — normal constraints active.")
		return True
	return False


# =================== CORE RUNTIME UTILITIES ===================

def jh_now():
	return datetime.now(TZ)

def ts_str():
	return jh_now().strftime("%Y-%m-%d %H:%M:%S")

def to_localized(ts_series):
	return pd.to_datetime(ts_series, errors="coerce").dt.tz_localize("UTC").dt.tz_convert(TZ)

def connect_mt5():
	"""Initialize MT5 connection."""
	print("🔌 Connecting to MetaTrader 5...")
	if not mt5.initialize():
		raise RuntimeError("MetaTrader5 initialization failed")
	acc = mt5.account_info()
	if acc:
		print(f"✅ Connected | Account: {acc.login} | Currency: {acc.currency} | Balance: {acc.balance}")
	else:
		print("⚠️ Connected but no account info available.")
	return acc

def shutdown_mt5():
	"""Graceful disconnect."""
	try:
		mt5.shutdown()
	except Exception:
		pass
	print("👋 MT5 disconnected.")


# =================== HEARTBEAT ===================


def heartbeat(*args, **kwargs):
	"""
	Accepts either:
	  heartbeat(state)
	or:
	  heartbeat(df, atr_series, body_series, atr_th, body_th, state, eff_mult)
	"""
	try:
		# Rich signature
		if len(args) >= 6 and hasattr(args[0], "__len__"):
			df, atr_series, body_series, atr_th, body_th, state = args[:6]
			eff_mult = args[6] if len(args) >= 7 else 1.0

			last_close = float(df["close"].iloc[-1])
			last_time  = pd.to_datetime(df["time"].iloc[-1])
			last_atr   = float(atr_series.iloc[-1])
			last_body  = float(body_series.iloc[-1])

			ema_up, ema_down, ema50, ema200, *_ = ema_trend_info(df)
			trend = "UP" if ema_up else ("DOWN" if ema_down else "FLAT")
			try:
				swH = float(df["high"].tail(20).max())
				swL = float(df["low"].tail(20).min())
			except Exception:
				swH, swL = float("nan"), float("nan")

			print(
				f"⏱️ HB {ts_str()} | last={last_time:%H:%M} close={last_close:.2f} | "
				f"ATR={last_atr:.2f} (th {atr_th:.2f}) | BODY={last_body:.2f} (th {body_th:.2f}) | "
				f"trend={trend} | tokens={state.get('token_bucket')} | losslock={state.get('losslock')} "
				f"| paused={state.get('daily_paused')} | eff_mult={eff_mult:.2f} | swH={swH:.2f} swL={swL:.2f}"
			)
			return

		# Simple signature
		state = args[0] if args else kwargs.get("state", {})
		now = ts_str()
		tokens   = state.get("token_bucket", 0) if isinstance(state, dict) else "?"
		paused   = state.get("daily_paused", False) if isinstance(state, dict) else "?"
		losslock = state.get("losslock", False) if isinstance(state, dict) else "?"
		print(f"⏱️ HB {now} | paused={paused} losslock={losslock} tokens={tokens}")
	except Exception as e:
		print(f"HB error: {e}")


def learn_update():
	"""Triggered after trades or daily reset to refresh teacher."""
	df = read_trades_df()
	if df.empty:
		return
	stats = compute_stats(df)
	if not stats:
		return
	adapt_parameters_from_stats(stats)
	st = read_learn_state()
	st["last_learn_update"] = ts_str()
	st["recent_stats"] = stats
	write_learn_state(st)
	print("🧠 Teacher state updated.")


# =================== POST-TRADE LOGGING ===================

def summarize_last_loss():
	"""When a trade loses, print diagnostic context."""
	df = read_trades_df()
	if df.empty:
		return
	last = df.tail(1).iloc[0]
	R = _safe_float(last.get("R", 0))
	if R >= 0:
		return
	strat = str(last.get("strategy", "")).upper()
	trend = str(last.get("ema_trend", "")).upper()
	atr = _safe_float(last.get("atr"))
	atr_th = _safe_float(last.get("atr_th"))
	body = _safe_float(last.get("body"))
	body_th = _safe_float(last.get("body_th"))
	atr_rel = atr / max(atr_th, 1e-9)
	body_rel = body / max(body_th, 1e-9)
	print(f"🧩 LOSS CONTEXT | strat={strat} trend={trend} "
		  f"ATRrel={atr_rel:.2f} BODYrel={body_rel:.2f} | note='{last.get('notes','')}'")

def finalize_trade_log():
	"""Called when a trade closes — triggers analysis + learning."""
	try:
		learn_update()
	except Exception:
		pass
	try:
		analyze_trade_history(out_json=LOSS_REPORT_FILE)
		summarize_last_loss()
	except Exception:
		pass


# =================== MAIN LOOP ===================

def load_state():
	"""Initial runtime state dictionary."""
	st = {
		"day": None,
		"token_bucket": TOKEN_BUCKET_CAPACITY,
		"last_refill": jh_now().timestamp(),
		"no_trade_since": jh_now().timestamp(),
		"losslock": False,
		"probe_trades_left": PROBE_TRADES_ON_LOCK,
		"session_stats": {},
		"daily_paused": False,
		"startup_active": True,
		"startup_new_bars_left": STARTUP_NEW_BARS,
		"last_bar_time_str": "",
		"last_loss_time": 0.0,
		"first_run_ts": jh_now().timestamp(),
		"last_daily_analysis_day": "",
	}
	return st

def save_state(st, fn=STATE_FILE):
	try:
		with open(fn, "w", encoding="utf-8") as f:
			json.dump(st, f, indent=2)
	except Exception:
		pass

def main():
	
	global BE_TRIGGER_R, LOCK_AT_R, STRATEGIES_ENABLED, TOKEN_BUCKET_CAPACITY, TOKEN_BUCKET_REFILL_SEC
print("🧠 Teacher state loaded.")
	acc = connect_mt5()
	print(f"⚙️ MODE={'RUNNER' if not TEST_MODE else 'TEST'} | LEARN=ON | STRATS={','.join(STRATEGIES_ENABLED)}")
	state = load_state()
	last_hb = time.time()

	while True:
		try:
			# Token bucket refill
			now_ts = time.time()
			if now_ts - state.get("last_refill", 0) >= TOKEN_BUCKET_REFILL_SEC:
				state["token_bucket"] = min(TOKEN_BUCKET_CAPACITY, state.get("token_bucket", 0) + 1)
				state["last_refill"] = now_ts
				save_state(state)
				if DEBUG:
					print(f"🔄 RateLimiter: +1 token(s) → {state['token_bucket']}")

			# daily reset
			daily_reset_if_needed(state)
			maybe_flip_off_test_mode(state)

			# placeholder for signal scanning
			time.sleep(POLL_SEC)

			if time.time() - last_hb >= HEARTBEAT_EVERY_SEC:
				heartbeat(state)
				last_hb = time.time()

		except KeyboardInterrupt:
			print("🛑 User interrupt — exiting bot.")
			break
		except Exception as e:
			if DEBUG:
				print(f"⚠️ main loop error: {e}")
			time.sleep(5)

	shutdown_mt5()
	print("🏁 Bot terminated gracefully.")
# =================== REPORTING & CLEANUP UTILITIES ===================

def summarize_day():
	"""Summarize today's performance and store it in summary_today.json."""
	df = read_trades_df()
	if df.empty:
		return
	df["timestamp"] = to_localized(df["timestamp"])
	today = pd.Timestamp.now(tz=TZ).date()
	df = df[df["timestamp"].dt.date == today]
	if df.empty:
		return

	total_pnl = float(pd.to_numeric(df["profit_zar"], errors="coerce").fillna(0).sum())
	wins = (pd.to_numeric(df["R"], errors="coerce") > 0).sum()
	losses = (pd.to_numeric(df["R"], errors="coerce") < 0).sum()
	wr = wins / max(1, wins + losses)
	rep = {
		"date": today.strftime("%Y-%m-%d"),
		"total_pnl_zar": round(total_pnl, 2),
		"trades": int(len(df)),
		"wins": int(wins),
		"losses": int(losses),
		"wr": round(wr, 3),
	}
	try:
		with open("summary_today.json", "w", encoding="utf-8") as f:
			json.dump(rep, f, indent=2)
	except Exception:
		pass
	print(f"🧾 Daily Summary: {rep}")


def cleanup_old_logs(days=14):
	"""Rotate or truncate log files older than X days to keep them light."""
	for path in [LOG_FILE, SIGNAL_LOG_FILE, SNAPSHOT_FILE]:
		if not os.path.exists(path):
			continue
		try:
			df = pd.read_csv(path)
			if "timestamp" not in df.columns:
				continue
			df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
			cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
			df2 = df[df["timestamp"] >= cutoff]
			tmp = path + ".tmp"
			df2.to_csv(tmp, index=False)
			os.replace(tmp, path)
			print(f"🧹 Cleaned {path}, kept {len(df2)} rows")
		except Exception as e:
			print(f"⚠️ cleanup {path}: {e}")


# =================== MANUAL CLI HELPERS ===================

def manual_close_all(symbol=SYMBOL):
	"""Close all open positions for the symbol immediately."""
	try:
		positions = mt5.positions_get(symbol=symbol)
		if not positions:
			print("No open positions.")
			return
		for p in positions:
			side = "BUY" if p.type == mt5.POSITION_TYPE_BUY else "SELL"
			vol = p.volume
			tick = mt5.symbol_info_tick(symbol)
			price = tick.bid if side == "BUY" else tick.ask
			close_req = {
				"action": mt5.TRADE_ACTION_DEAL,
				"symbol": symbol,
				"volume": vol,
				"type": mt5.ORDER_TYPE_SELL if side == "BUY" else mt5.ORDER_TYPE_BUY,
				"position": p.ticket,
				"price": price,
				"deviation": 50,
				"magic": 0,
				"comment": "manual_close_all",
			}
			mt5.order_send(close_req)
		print(f"✅ Closed {len(positions)} positions on {symbol}")
	except Exception as e:
		print(f"⚠️ manual_close_all error: {e}")


def print_open_positions(symbol=SYMBOL):
	"""Print a quick summary of open positions."""
	positions = mt5.positions_get(symbol=symbol)
	if not positions:
		print("No open positions.")
		return
	print(f"=== OPEN POSITIONS ({len(positions)}) ===")
	for p in positions:
		side = "BUY" if p.type == mt5.POSITION_TYPE_BUY else "SELL"
		print(f"{p.ticket}: {side} {p.volume:.2f} @ {p.price_open:.2f} "
			  f"SL={p.sl:.2f} TP={p.tp:.2f} PnL={p.profit:.2f}")


def manual_learn_refresh():
	"""Force a learning update manually."""
	try:
		learn_update()
	except Exception as e:
		print(f"manual learn error: {e}")


# =================== SAFE EXIT HANDLER ===================

def graceful_shutdown():
	"""Called on KeyboardInterrupt or fatal error to tidy up before exit."""
	try:
		summarize_day()
		cleanup_old_logs()
	except Exception:
		pass
	finally:
		shutdown_mt5()
		print("🛑 MT5 disconnected and files flushed — safe exit complete.")


# ========== OPTIONAL COMMAND-LINE ENTRY POINT ==========

if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description="Gold Bot v7 Adaptive Multi-Strategy")
	parser.add_argument("--summary", action="store_true", help="Print daily summary and exit")
	parser.add_argument("--cleanup", action="store_true", help="Cleanup old logs then exit")
	parser.add_argument("--manual-close", action="store_true", help="Close all open positions and exit")
	parser.add_argument("--learn-refresh", action="store_true", help="Force a learning refresh and exit")
	parser.add_argument("--show-open", action="store_true", help="Show open positions and exit")
	args = parser.parse_args()

	if args.summary:
		summarize_day(); sys.exit(0)
	if args.cleanup:
		cleanup_old_logs(); sys.exit(0)
	if args.manual_close:
		manual_close_all(); sys.exit(0)
	if args.learn_refresh:
		manual_learn_refresh(); sys.exit(0)
	if args.show_open:
		print_open_positions(); sys.exit(0)

	try:
		main()
	except KeyboardInterrupt:
		graceful_shutdown()
	except Exception as e:
		print(f"FATAL: {e}")
		graceful_shutdown()
# =================== STRATEGY UTILITIES ===================

def ema_trend_info(df):
	"""
	Return EMA trend flags and diagnostics:
	  ema_up / ema_down, ema50, ema200, slope50(5-bar), slope200(5-bar)
	"""
	ema50  = df["close"].ewm(span=50, adjust=False).mean()
	ema200 = df["close"].ewm(span=200, adjust=False).mean()
	slope50  = float(ema50.iloc[-1]  - ema50.iloc[-5])
	slope200 = float(ema200.iloc[-1] - ema200.iloc[-5])
	ema_up   = bool(ema50.iloc[-1] > ema200.iloc[-1] and slope50 >= 0 and slope200 >= -1e-9)
	ema_down = bool(ema50.iloc[-1] < ema200.iloc[-1] and slope50 <= 0 and slope200 <=  1e-9)
	return ema_up, ema_down, float(ema50.iloc[-1]), float(ema200.iloc[-1]), slope50, slope200


def candle_body(df):
	"""Absolute candle body size series."""
	return (df["close"] - df["open"]).abs()


def swing_levels(df, lookback=20):
	"""
	Lightweight swing hi/lo: previous bar's rolling extrema over a window.
	Use df.iloc[:-1] (exclude current building bar) for stability.
	"""
	if len(df) < lookback + 2:
		return None, None
	highs = df["high"].rolling(lookback).max()
	lows  = df["low"].rolling(lookback).min()
	sw_high = float(highs.iloc[-2])
	sw_low  = float(lows.iloc[-2])
	return sw_high, sw_low


# =================== BRRT (Break & Retest) ===================

def signal_break_retest(df, atr_series, atr_th, body_th):
	"""
	Break & Retest:
	  • Require ATR/BODY >= thresholds (with a small grace)
	  • Require EMA trend alignment
	  • Recent break of swing level + retest rejection within tolerance
	  • Trigger when price reclaims/loses the level after the retest
	"""
	if len(df) < 210:
		return {"status":"fail","reason":"data_short","dir":None}

	last_atr  = float(atr_series.iloc[-1])
	last_body = float(candle_body(df).iloc[-1])

	# gates + grace
	atr_ok  = (last_atr  >= atr_th)  or (last_atr  >= atr_th  * _ATR_GRACE)
	body_ok = (last_body >= body_th) or (last_body >= body_th * _BODY_GRACE)
	if not (atr_ok and body_ok):
		return {"status":"fail","reason":"gate_atr_body","dir":None}

	ema_up, ema_down, ema50, ema200, slope50, slope200 = ema_trend_info(df)
	if not (ema_up or ema_down):
		return {"status":"fail","reason":"ema_gate","dir":None}

	sw_high, sw_low = swing_levels(df.iloc[:-1])
	if sw_high is None or sw_low is None:
		# fallback
		sw_high = float(df["high"].rolling(20).max().iloc[-2])
		sw_low  = float(df["low"].rolling(20).min().iloc[-2])

	tol = last_atr * BRRT_TOL_ATR_X
	recent = df.iloc[-4:]  # last 3 completed + current
	def rejection_ok(bar, direction):
		o, h, l, c = float(bar["open"]), float(bar["high"]), float(bar["low"]), float(bar["close"])
		rng = max(h - l, 1e-9)
		if direction == "long":
			# lower-wick rejection + not huge upper wick
			return ((min(o,c) - l)/rng >= 0.32) and ((h - max(o,c))/rng <= 0.68)
		else:
			# upper-wick rejection + not huge lower wick
			return ((h - max(o,c))/rng >= 0.32) and ((min(o,c) - l)/rng <= 0.68)

	# LONG setup
	if ema_up:
		# 1) Break above swing high, preceded by a close not far above it
		broke_up = any(
			(recent["high"].iloc[i] > sw_high + tol and df["close"].iloc[-2 - i] <= sw_high + tol)
			for i in range(1, 4)
		)
		if broke_up:
			# 2) Retest: look back N bars for a touch near the level with rejection
			lows_recent = df.iloc[-BRRT_RETEST_BARS:]
			touch_idx = None
			for k in range(len(lows_recent)):
				if abs(float(lows_recent["low"].iloc[k]) - sw_high) <= tol:
					touch_idx = k
			if touch_idx is None:
				return {"status":"hold","reason":"await_retest","dir":"long"}
			rej_bar = lows_recent.iloc[touch_idx]
			if not rejection_ok(rej_bar, "long"):
				return {"status":"fail","reason":"no_rejection","dir":"long"}

			last_close = float(df["close"].iloc[-1])
			if last_close <= sw_high:  # must reclaim high after retest
				return {"status":"fail","reason":"no_trigger","dir":"long"}

			entry = last_close
			last_low = float(df["low"].iloc[-1])
			sl = min(sw_high - last_atr*0.9, last_low - last_atr*0.6)
			if sl < entry:
				return {
					"status":"ok","dir":"long","entry":entry,"sl":sl,
					"reason":f"BRRT up swH={sw_high:.2f} tol={tol:.2f}",
					"diag":{"atr":last_atr,"atr_th":atr_th,"body":last_body,"body_th":body_th,
							"ema50":ema50,"ema200":ema200,"slope50":slope50,"slope200":slope200,
							"swH":sw_high,"swL":sw_low}
				}

	# SHORT setup
	if ema_down:
		broke_dn = any(
			(recent["low"].iloc[i] < sw_low - tol and df["close"].iloc[-2 - i] >= sw_low - tol)
			for i in range(1, 4)
		)
		if broke_dn:
			highs_recent = df.iloc[-BRRT_RETEST_BARS:]
			touch_idx = None
			for k in range(len(highs_recent)):
				if abs(float(highs_recent["high"].iloc[k]) - sw_low) <= tol:
					touch_idx = k
			if touch_idx is None:
				return {"status":"hold","reason":"await_retest","dir":"short"}
			rej_bar = highs_recent.iloc[touch_idx]
			if not rejection_ok(rej_bar, "short"):
				return {"status":"fail","reason":"no_rejection","dir":"short"}

			last_close = float(df["close"].iloc[-1])
			if last_close >= sw_low:
				return {"status":"fail","reason":"no_trigger","dir":"short"}

			entry = last_close
			last_high = float(df["high"].iloc[-1])
			sl = max(sw_low + last_atr*0.9, last_high + last_atr*0.6)
			if sl > entry:
				return {
					"status":"ok","dir":"short","entry":entry,"sl":sl,
					"reason":f"BRRT dn swL={sw_low:.2f} tol={tol:.2f}",
					"diag":{"atr":last_atr,"atr_th":atr_th,"body":last_body,"body_th":body_th,
							"ema50":ema50,"ema200":ema200,"slope50":slope50,"slope200":slope200,
							"swH":sw_high,"swL":sw_low}
				}

	return {"status":"fail","reason":"no_signal","dir":None}


# =================== ORB (Opening Range Breakout) ===================

def get_orb_ranges(df):
	"""
	Compute configured session opening ranges for the current local day.
	Returns {name: (start_dt, end_dt, hi, lo)} for sessions that have enough bars.
	"""
	out = {}
	now = jh_now()
	day = now.strftime("%Y-%m-%d")
	for name, start_s, minutes, min_pctl, with_retest in ORB_SESSIONS:
		start_dt = TZ.localize(datetime.strptime(f"{day} {start_s}", "%Y-%m-%d %H:%M"))
		end_dt = start_dt + timedelta(minutes=minutes)
		win = df[(df["time"] >= start_dt) & (df["time"] <= end_dt)]
		if len(win) >= 2:
			out[name] = (start_dt, end_dt, float(win["high"].max()), float(win["low"].min()))
	return out


def signal_orb(df, atr_series, atr_th, body_th):
	"""
	Opening Range Breakout with optional retest and EMA alignment.
	Signals are valid after the ORB window closes, up to ORB_MAX_AGE_MIN minutes.
	"""
	if len(df) < 120:
		return {"status":"fail","reason":"data_short","dir":None}

	last_atr  = float(atr_series.iloc[-1])
	last_body = float(candle_body(df).iloc[-1])

	# a bit looser than BRRT; ORB tends to be more impulsive
	atr_ok  = (last_atr  >= max(atr_th*0.85, 1e-9))
	body_ok = (last_body >= max(body_th*0.85, 1e-9))
	if not (atr_ok and body_ok):
		return {"status":"fail","reason":"gate_atr_body","dir":None}

	ema_up, ema_down, ema50, ema200, slope50, slope200 = ema_trend_info(df)
	now = jh_now()
	ranges = get_orb_ranges(df)
	if not ranges:
		return {"status":"fail","reason":"no_orb_window","dir":None}

	last_close = float(df["close"].iloc[-1])
	for name, (start_dt, end_dt, hi, lo) in ranges.items():
		# Valid only after the window closed, and within age limit
		if now <= end_dt or now - end_dt > timedelta(minutes=ORB_MAX_AGE_MIN):
			continue

		# Long breakout
		if last_close > hi and (ema_up or not FVG_WITH_TREND_ONLY):
			look = df[df["time"] >= end_dt].tail(12)
			retested = any(abs(x - hi) <= last_atr*0.15 for x in look["low"].values) if len(look) else True
			sl = lo
			# Ensure SL distance reasonable (≥ MIN_SL_ATR_X)
			if last_close - sl >= last_atr*MIN_SL_ATR_X:
				return {
					"status":"ok","dir":"long","entry":last_close,"sl":sl,
					"reason":f"ORB[{name}] brk {hi:.2f} (ret={retested})",
					"diag":{"atr":last_atr,"atr_th":atr_th,"body":last_body,"body_th":body_th,
							"ema50":ema50,"ema200":ema200,"slope50":slope50,"slope200":slope200,
							"orb_name":name,"orb_start":start_dt.strftime('%H:%M'),
							"orb_end":end_dt.strftime('%H:%M'),"swH":hi,"swL":lo}
				}

		# Short breakout
		if last_close < lo and (ema_down or not FVG_WITH_TREND_ONLY):
			look = df[df["time"] >= end_dt].tail(12)
			retested = any(abs(x - lo) <= last_atr*0.15 for x in look["high"].values) if len(look) else True
			sl = hi
			if sl - last_close >= last_atr*MIN_SL_ATR_X:
				return {
					"status":"ok","dir":"short","entry":last_close,"sl":sl,
					"reason":f"ORB[{name}] brk {lo:.2f} (ret={retested})",
					"diag":{"atr":last_atr,"atr_th":atr_th,"body":last_body,"body_th":body_th,
							"ema50":ema50,"ema200":ema200,"slope50":slope50,"slope200":slope200,
							"orb_name":name,"orb_start":start_dt.strftime('%H:%M'),
							"orb_end":end_dt.strftime('%H:%M'),"swH":hi,"swL":lo}
				}

	return {"status":"fail","reason":"no_signal","dir":None}


# =================== FVG (Fair Value Gap Retouch) ===================

def detect_fvg(df):
	"""
	Scan for the most recent fresh three-bar FVG:
	  • Bullish: h0 < l2
	  • Bearish: l0 > h2
	Returns dict: {"type": "bull"/"bear", "top": ..., "bot": ..., "age_bars": int, "size": float}
	"""
	if len(df) < 10:
		return None
	atr = compute_atr(df).iloc[-1]
	max_age = FVG_VALID_FOR_BARS
	for i in range(3, min(max_age + 3, len(df))):
		i0 = -i; i1 = i0 + 1; i2 = i0 + 2
		h0, l0 = float(df["high"].iloc[i0]), float(df["low"].iloc[i0])
		h1, l1 = float(df["high"].iloc[i1]), float(df["low"].iloc[i1])
		h2, l2 = float(df["high"].iloc[i2]), float(df["low"].iloc[i2])

		# Bullish gap
		if h0 < l2:
			size = l2 - h0
			if size >= max(float(atr)*FVG_MIN_SIZE_ATR_X, 1e-9):
				top = l2; bot = h0
				age = len(df) - (i2)
				return {"type":"bull","top":float(top),"bot":float(bot),"age_bars":int(age),"size":float(size)}
		# Bearish gap
		if l0 > h2:
			size = l0 - h2
			if size >= max(float(atr)*FVG_MIN_SIZE_ATR_X, 1e-9):
				top = l0; bot = h2
				age = len(df) - (i2)
				return {"type":"bear","top":float(top),"bot":float(bot),"age_bars":int(age),"size":float(size)}
	return None


def signal_fvg(df, atr_series, body_th, atr_th):
	"""
	FVG retouch with optional EMA alignment.
	Entry on retouch or reclaim past the gap edge; SL beyond the other edge ± ATR buffer.
	"""
	if len(df) < 120:
		return {"status":"fail","reason":"data_short","dir":None}

	last_atr  = float(atr_series.iloc[-1])
	last_body = float(candle_body(df).iloc[-1])

	atr_ok  = (last_atr  >= atr_th*0.8)
	body_ok = (last_body >= body_th*0.8)
	if not (atr_ok and body_ok):
		return {"status":"fail","reason":"gate_atr_body","dir":None}

	ema_up, ema_down, ema50, ema200, slope50, slope200 = ema_trend_info(df)
	if FVG_WITH_TREND_ONLY and not (ema_up or ema_down):
		return {"status":"fail","reason":"ema_gate","dir":None}

	fvg = detect_fvg(df.tail(FVG_LOOKBACK))
	if not fvg:
		return {"status":"fail","reason":"no_fvg","dir":None}

	last_close = float(df["close"].iloc[-1])
	tol = last_atr * FVG_RETOUCH_TOL_ATR_X

	if fvg["type"] == "bull" and (ema_up or not FVG_WITH_TREND_ONLY):
		# prefer retouch of top, or reclaim above top
		if abs(last_close - fvg["top"]) <= tol or last_close > fvg["top"]:
			sl = fvg["bot"] - last_atr*0.4
			if last_close - sl >= last_atr*MIN_SL_ATR_X:
				return {
					"status":"ok","dir":"long","entry":last_close,"sl":sl,
					"reason":f"FVG bull retouch (age {fvg['age_bars']} bars)",
					"diag":{"atr":last_atr,"atr_th":atr_th,"body":last_body,"body_th":body_th,
							"ema50":ema50,"ema200":ema200,"slope50":slope50,"slope200":slope200,
							"fvg_gap": f"{fvg['bot']:.2f}-{fvg['top']:.2f}", "fvg_age": fvg["age_bars"]}

				}

	if fvg["type"] == "bear" and (ema_down or not FVG_WITH_TREND_ONLY):
		if abs(last_close - fvg["bot"]) <= tol or last_close < fvg["bot"]:
			sl = fvg["top"] + last_atr*0.4
			if sl - last_close >= last_atr*MIN_SL_ATR_X:
				return {
					"status":"ok","dir":"short","entry":last_close,"sl":sl,
					"reason":f"FVG bear retouch (age {fvg['age_bars']} bars)",
					"diag":{"atr":last_atr,"atr_th":atr_th,"body":last_body,"body_th":body_th,
							"ema50":ema50,"ema200":ema200,"slope50":slope50,"slope200":slope200,
							"fvg_gap": f"{fvg['bot']:.2f}-{fvg['top']:.2f}", "fvg_age": fvg["age_bars"]}

				}

	return {"status":"fail","reason":"no_signal","dir":None}


# =================== SIGNAL CHOOSER ===================

def choose_signal(df, atr_series, atr_th, body_th):
	"""
	Run enabled strategies in order and pick first 'ok'.
	When none qualify and TEST_MODE is on, allow an exploratory probe with reduced risk.
	"""
	strat_map = {
		"BRRT": signal_break_retest,
		"ORB":  signal_orb,
		"FVG":  signal_fvg,
	}
	results = []

	# Per-strategy soft band enforcement (learned preferred ATR×Body zones)
	bodies = candle_body(df)

	for s in STRATEGIES_ENABLED:
		atr_th_s, body_th_s = enforce_preferred_bands(s, df, atr_series, bodies, atr_th, body_th)
		res = strat_map[s](df, atr_series, atr_th_s, body_th_s)
		results.append((s, res))

		# Diagnostics log
		diag = res.get("diag", {}) if isinstance(res, dict) else {}
		ema_up, ema_down, ema50, ema200, slope50, slope200 = ema_trend_info(df)
		log_signal({
			"timestamp": ts_str(),
			"symbol": SYMBOL,
			"strategy": s,
			"dir": res.get("dir"),
			"status": res.get("status"),
			"reason": res.get("reason"),
			"atr": f"{diag.get('atr', float(atr_series.iloc[-1])):.2f}",
			"atr_th": f"{diag.get('atr_th', atr_th_s):.2f}",
			"body": f"{diag.get('body', float(bodies.iloc[-1])):.2f}",
			"body_th": f"{body_th_s:.2f}",
			"ema50_gt_200": ema50 > ema200,
			"ema_slope50": f"{slope50:.4f}",
			"ema_slope200": f"{slope200:.4f}",
			"price": f"{float(df['close'].iloc[-1]):.2f}",
			"swH": f"{diag.get('swH','')}",
			"swL": f"{diag.get('swL','')}",
			"orb_name": diag.get("orb_name",""),
			"orb_start": diag.get("orb_start",""),
			"orb_end": diag.get("orb_end",""),
			"fvg_gap": diag.get("fvg_gap",""),
			"fvg_age": diag.get("fvg_age",""),
			"exploratory": False
		})

		if res.get("status") == "ok":
			return s, res, False

	# Exploratory probe (TEST_MODE)
	if TEST_MODE and results and np.random.rand() < EXPLORATORY_SIGNAL_RATIO:
		priority = {"ORB":0, "FVG":1, "BRRT":2}
		results.sort(key=lambda x: priority.get(x[0], 9))
		for s, r in results:
			if r.get("dir"):
				last_close = float(df["close"].iloc[-1]); last_atr = float(atr_series.iloc[-1])
				if r["dir"] == "long":
					entry = last_close
					sl = entry - max(last_atr*MIN_SL_ATR_X, last_atr*0.7)
				else:
					entry = last_close
					sl = entry + max(last_atr*MIN_SL_ATR_X, last_atr*0.7)
				r2 = {"status":"ok","dir":r["dir"],"entry":entry,"sl":sl,"reason":f"EXPLORATORY[{s}] {r.get('reason','gate_bypass')}"}
				return s, r2, True

	return None, None, False
# =================== RISK & ORDER SIZING ===================

def order_loss_for(symbol, direction, entry, sl, volume):
	"""
	Use broker engine to compute ZAR loss if SL is hit.
	Returns NEGATIVE for a loss (MetaTrader5 semantics), we flip where needed.
	"""
	typ = mt5.ORDER_TYPE_BUY if direction == "long" else mt5.ORDER_TYPE_SELL
	p = mt5.order_calc_profit(typ, symbol, volume, entry, sl)
	if p is None:
		raise RuntimeError(f"order_calc_profit failed: {mt5.last_error()}")
	if math.isnan(p) or math.isinf(p):
		raise RuntimeError("order_calc_profit returned NaN/Inf")
	return float(p)  # MT5 returns negative value for loss


def _safe_loss_abs(symbol, direction, entry, sl, volume, default=float("inf")):
	"""
	Return absolute ZAR loss (positive) if SL hits, or default on error.
	"""
	try:
		val = order_loss_for(symbol, direction, entry, sl, volume)
		return -val if val < 0 else 0.0  # absolute loss in ZAR
	except Exception:
		return default


def normalize_lot(lot, info):
	"""
	Round lot to broker step and respect min/max.
	"""
	step = info.volume_step if info.volume_step > 0 else 0.01
	lot = max(lot, info.volume_min)
	lot = min(lot, info.volume_max)
	steps = round(lot / step)
	decimals = max(0, min(3, -int(round(math.log10(step))) if step < 1 else 0))
	return round(steps * step, decimals)


def calc_lot_for_risk_zar(info, direction, entry, sl, balance,
						  losslock=False, risk_scale: float = 1.0):
	"""
	Binary-search volume to target ZAR risk, honoring HARD_RISK_ZAR_CAP and broker limits.
	Returns (lot, expected_loss_abs, mode_str).
	"""
	# target risk
	risk_unit   = balance * RISK_PCT_PER_TRADE
	target_risk = min(risk_unit, HARD_RISK_ZAR_CAP)
	if losslock:
		target_risk *= MIN_PROBE_RISK_FACTOR
	target_risk *= max(0.1, float(risk_scale))

	vol_min, vol_max = info.volume_min, info.volume_max
	step = info.volume_step if info.volume_step > 0 else 0.01

	# If min-lot already exceeds HARD cap → skip
	loss_min = _safe_loss_abs(SYMBOL, direction, entry, sl, vol_min)
	if not math.isfinite(loss_min) or loss_min <= 0:
		return 0.0, 0.0, "SIZING_QUOTE_INVALID"
	if loss_min > HARD_RISK_ZAR_CAP + 1e-9:
		return 0.0, 0.0, "MIN_EXCEEDS_CAP"

	# Estimate loss per 1.0 lot (or per min lot if 1.0 out of range)
	base_vol = 1.0 if (vol_min <= 1.0 <= vol_max) else vol_min
	loss_at_base = _safe_loss_abs(SYMBOL, direction, entry, sl, base_vol)
	if not math.isfinite(loss_at_base) or loss_at_base <= 0:
		return 0.0, 0.0, "SIZING_QUOTE_INVALID"

	# Cap the useful upper bound for search
	max_useful = target_risk / (loss_at_base / base_vol) * 1.2
	hi_cap = max(vol_min, min(vol_max, max_useful))

	lo, hi = vol_min, hi_cap
	best_v = vol_min

	for _ in range(24):
		mid = round((lo + hi) / 2.0 / step) * step
		mid = max(vol_min, min(hi_cap, mid))
		loss_abs = _safe_loss_abs(SYMBOL, direction, entry, sl, mid)
		if not math.isfinite(loss_abs) or loss_abs <= 0:
			hi = max(lo, mid - step)
		elif loss_abs > target_risk:
			hi = mid - step
		else:
			best_v = mid
			lo = mid + step
		if hi < lo:
			break

	lot = normalize_lot(best_v, info)
	actual_loss = _safe_loss_abs(SYMBOL, direction, entry, sl, lot, default=target_risk)

	# Sanity: if way below target risk but very large lot ⇒ suspicious quote
	if actual_loss < target_risk * 0.15 and lot > vol_min * 5:
		return 0.0, 0.0, "SIZING_SUSPECT"

	return lot, actual_loss, "OK"


# =================== ORDERING & EXECUTION ===================

def current_price(symbol, side_for_entry: str):
	"""
	Return best current price for the given side: 'buy'→ask, 'sell'→bid, else mid.
	"""
	tick = mt5.symbol_info_tick(symbol)
	if tick is None:
		raise RuntimeError("No tick available")
	if side_for_entry == "buy":
		return float(tick.ask)
	elif side_for_entry == "sell":
		return float(tick.bid)
	else:
		return float((tick.bid + tick.ask) / 2.0)


def normalize_price(price, digits):
	"""Floor to broker digits to avoid 'invalid price'."""
	if price is None:
		return None
	factor = 10 ** digits
	return math.floor(price * factor) / factor


def _send_with_filling(req):
	"""
	Try RETURN → IOC → FOK fill policies to maximize success without slippage explosions.
	"""
	req["type_filling"] = mt5.ORDER_FILLING_RETURN
	r = mt5.order_send(req)
	if r is not None and r.retcode == mt5.TRADE_RETCODE_DONE:
		return r

	if r is None or r.retcode in (10030, mt5.TRADE_RETCODE_INVALID_FILL):
		req["type_filling"] = mt5.ORDER_FILLING_IOC
		r2 = mt5.order_send(req)
		if r2 is not None and r2.retcode == mt5.TRADE_RETCODE_DONE:
			return r2
		req["type_filling"] = mt5.ORDER_FILLING_FOK
		r3 = mt5.order_send(req)
		return r3
	return r


def send_order(symbol, direction, entry, sl, tp, lot, info, comment_tag="GOLD_BOT_V7"):
	"""
	Market order with SL/TP set server-side. Returns (ticket, response).
	"""
	digits = info.digits
	entry = normalize_price(entry, digits)
	sl    = normalize_price(sl, digits)
	tp    = normalize_price(tp, digits) if tp else 0.0

	req = {
		"action": mt5.TRADE_ACTION_DEAL,
		"symbol": symbol,
		"volume": lot,
		"type": mt5.ORDER_TYPE_BUY if direction == "long" else mt5.ORDER_TYPE_SELL,
		"price": entry,
		"sl": sl,
		"tp": tp if tp else 0.0,
		"deviation": SLIPPAGE_POINTS,
		"magic": 20251015,
		"comment": comment_tag,
		"type_time": mt5.ORDER_TIME_GTC,
	}
	r = _send_with_filling(req)
	if r is None:
		raise RuntimeError(f"order_send None: {mt5.last_error()}")
	if r.retcode != mt5.TRADE_RETCODE_DONE:
		raise RuntimeError(f"order_send failed: {r.retcode} {r._asdict()}")
	return r.order, r


def modify_sl_tp(ticket, sl=None, tp=None):
	"""
	Modify SL/TP of an open position by ticket. Returns True on success.
	"""
	pos = get_position_by_ticket(ticket)
	if not pos:
		return False
	info = mt5.symbol_info(pos.symbol)
	digits = info.digits if info else 2
	new_sl = normalize_price(sl, digits) if sl is not None else pos.sl
	new_tp = normalize_price(tp, digits) if tp is not None else pos.tp
	req = {
		"action": mt5.TRADE_ACTION_SLTP,
		"position": pos.ticket,
		"sl": new_sl,
		"tp": new_tp,
		"symbol": pos.symbol
	}
	r = _send_with_filling(req)
	return (r is not None) and (r.retcode in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED))


def get_position_by_ticket(ticket):
	"""
	Fetch a single open position by ticket id, or None.
	"""
	positions = mt5.positions_get()
	if positions is None:
		return None
	for p in positions:
		if p.ticket == ticket:
			return p
	return None


def get_open_positions_count(symbol=SYMBOL):
	positions = mt5.positions_get(symbol=symbol)
	return 0 if positions is None else len(positions)


def chandelier_stop(direction, atr, factor, last_close):
	"""
	Chandelier trailing stop level relative to last price and ATR*factor.
	"""
	if direction == "long":
		return last_close - atr * factor
	else:
		return last_close + atr * factor


def floating_profit_zar(ticket):
	"""
	Current floating P/L in account currency (ZAR) for a position ticket.
	"""
	pos = get_position_by_ticket(ticket)
	if not pos:
		return 0.0
	return float(pos.profit)


# =================== PARTIAL CLOSE HELPERS ===================

def close_position_market(ticket):
	"""
	Close full position at market.
	"""
	pos = get_position_by_ticket(ticket)
	if not pos:
		return
	side = "sell" if pos.type == mt5.POSITION_TYPE_BUY else "buy"
	price = current_price(pos.symbol, side)
	req = {
		"action": mt5.TRADE_ACTION_DEAL,
		"position": pos.ticket,
		"symbol": pos.symbol,
		"volume": pos.volume,
		"type": mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
		"price": price,
		"deviation": SLIPPAGE_POINTS,
		"magic": 20251015,
		"comment": "GOLD_BOT_CLOSE",
	}
	r = _send_with_filling(req)
	if r is None:
		print(f"⚠️ close_position: order_send None {mt5.last_error()}")
	elif r.retcode != mt5.TRADE_RETCODE_DONE:
		print(f"⚠️ close_position retcode={r.retcode} {r._asdict()}")


def close_position_partial_market(ticket, fraction: float):
	"""
	Close fraction (0<fraction<1) of position volume at market.
	Returns True on success.
	"""
	pos = get_position_by_ticket(ticket)
	if not pos or not (0.0 < fraction < 1.0):
		return False

	info = mt5.symbol_info(pos.symbol)
	if not info:
		return False

	vol_min = info.volume_min
	step    = info.volume_step if info.volume_step > 0 else 0.01

	target = pos.volume * float(fraction)
	steps = round(target / step)
	volume_to_close = max(vol_min, steps * step)
	# don't overshoot; leave at least min lot if possible
	if pos.volume - volume_to_close < vol_min:
		volume_to_close = max(vol_min, pos.volume - vol_min)

	if volume_to_close <= 0:
		return False

	side = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
	px_side = "sell" if pos.type == mt5.POSITION_TYPE_BUY else "buy"
	price = current_price(pos.symbol, px_side)

	req = {
		"action": mt5.TRADE_ACTION_DEAL,
		"position": pos.ticket,
		"symbol": pos.symbol,
		"volume": volume_to_close,
		"type": side,
		"price": price,
		"deviation": SLIPPAGE_POINTS,
		"magic": 20251015,
		"comment": "GOLD_BOT_PARTIAL",
	}
	r = _send_with_filling(req)
	if r is None:
		print(f"⚠️ partial_close: order_send None {mt5.last_error()}")
		return False
	if r.retcode != mt5.TRADE_RETCODE_DONE:
		print(f"⚠️ partial_close retcode={r.retcode} {r._asdict()}")
		return False
	print(f"🧩 Partial closed {volume_to_close} lot(s).")
	return True
# =================== POSITION MANAGEMENT & FINALIZATION ===================

def finalize_trade_log(ticket, exit_reason, max_fav_R, max_adv_R, entry, risk_zar):
	"""
	Safely finalize a trade row: write exit fields, profit_zar, and R.
	Uses atomic write (tmp + replace) and real newlines.
	"""
	time.sleep(0.8)
	df = read_trades_df()
	if df.empty:
		return
	idx = df[df["ticket"] == str(ticket)].index
	if len(idx) == 0:
		return
	i = idx[-1]

	profit_zar = 0.0
	exit_price = float("nan")
	try:
		fr = datetime.now(timezone.utc) - timedelta(days=5)
		to = datetime.now(timezone.utc)
		deals = mt5.history_deals_get(fr, to)
		if deals:
			for d in deals:
				if d.position_id == ticket:
					profit_zar += float(d.profit or 0.0)
					exit_price = float(d.price or float("nan"))
	except Exception:
		pass

	R = (profit_zar / risk_zar) if risk_zar and risk_zar > 0 else 0.0

	df.loc[i, "exit_price"]    = f"{exit_price:.2f}" if exit_price == exit_price else ""
	df.loc[i, "profit_zar"]    = f"{profit_zar:.2f}"
	df.loc[i, "R"]             = f"{R:.2f}"
	df.loc[i, "max_fav_R"]     = f"{max_fav_R:.2f}"
	df.loc[i, "max_adverse_R"] = f"{max_adv_R:.2f}"
	df.loc[i, "reason_exit"]   = exit_reason

	tmp = LOG_FILE + ".tmp"
	try:
		with open(tmp, "w", encoding="utf-8", newline="") as f:
			df.to_csv(f, index=False, lineterminator="\n")
			f.flush(); os.fsync(f.fileno())
		os.replace(tmp, LOG_FILE)
	except Exception as e:
		print(f"⚠️ trade_log write failed: {e}")

	print(f"🧾 Closed: profit={profit_zar:.2f} ZAR | R={R:.2f} | reason={exit_reason}")

	# Kick a learn update if available
	try:
		learn_update()
	except Exception:
		pass


def summarize_last_loss():
	"""
	Print a compact context summary for the last loss to STDOUT.
	"""
	df = read_trades_df()
	if df.empty:
		return
	last = df.tail(1).iloc[0]
	try:
		R = float(last.get("R", 0))
	except Exception:
		R = 0.0
	if R >= 0:
		return
	strat = str(last.get("strategy", "")).upper()
	trend = str(last.get("ema_trend", "")).upper()
	atr    = _safe_float(last.get("atr"), 0.0)
	atr_th = _safe_float(last.get("atr_th"), 0.0)
	body   = _safe_float(last.get("body"), 0.0)
	body_th= _safe_float(last.get("body_th"), 0.0)
	atr_rel  = atr / max(atr_th, 1e-9)
	body_rel = body / max(body_th, 1e-9)
	sess = _session_block_from_ts(pd.Series([last.get("timestamp")])).iloc[0]
	print(
		f"🧩 LOSS CONTEXT: strat={strat} session={sess} trend={trend} "
		f"atr_rel={atr_rel:.2f} body_rel={body_rel:.2f} notes='{last.get('notes','')}'"
	)


def get_latest_atr(symbol, timeframe, period=ATR_PERIOD, history=200):
	"""
	Convenience for management loop: fetch recent candles and ATR.
	"""
	df = get_rates(symbol, timeframe, history)
	atr = compute_atr(df, period=period)
	return float(atr.iloc[-1]), df


def manage_position(ticket, direction, entry, sl_init, lot, risk_zar, state):
	"""
	Position manager:
	  • BE at +0.8R (BE_TRIGGER_R)
	  • Lock at +2R → SL to +0.6R (LOCK_AT_R / LOCK_TO_R)
	  • Chandelier trail AFTER >= TRAIL_START_CONFIRM_BARS closed bars with R ≥ CHAND_START_R
	  • Dynamic ATR trailing factors: 6.0 (2.0–2.5R), 4.0 (2.5–3.5R), 2.5 (≥3.5R)
	  • One-time partial at +PARTIAL_TP_ZAR (e.g., 400 ZAR)
	  • Hard full TP at +HARD_TP_ZAR (e.g., 700 ZAR)
	  • Always-on hard floating kill-switch at -HARD_RISK_ZAR_CAP
	"""
	pos = get_position_by_ticket(ticket)
	if not pos:
		print("⚠️ manage_position: position not found.")
		return

	max_fav_R = 0.0
	max_adv_R = 0.0
	be_done = False
	lock_done = False
	chand_started = False
	partial_done = False

	px_per_R = abs(entry - sl_init)  # price move equivalent to 1R

	confirm_counter = 0
	confirm_ready = False

	while True:
		# refresh position
		pos = get_position_by_ticket(ticket)
		if not pos:
			print("ℹ️ Position closed (external SL/TP or manual). Finalizing…")
			finalize_trade_log(ticket, "EXTERNAL_EXIT", max_fav_R, max_adv_R, entry, risk_zar)
			break

		# HARD kill-switch in ZAR (floating)
		flp = floating_profit_zar(ticket)
		if flp <= -HARD_RISK_ZAR_CAP:
			print(f"🛑 HARD Kill-Switch: floating={flp:.2f} ZAR <= -{HARD_RISK_ZAR_CAP:.2f} — closing.")
			close_position_market(ticket)
			finalize_trade_log(ticket, "HARD_KILL_SWITCH", max_fav_R, max_adv_R, entry, risk_zar)
			summarize_last_loss()
			break

		# Hard ZAR take-profit
		if flp >= HARD_TP_ZAR:
			print(f"🏁 HARD ZAR TP: +{flp:.2f} ZAR >= {HARD_TP_ZAR:.2f} — closing full.")
			close_position_market(ticket)
			finalize_trade_log(ticket, "HARD_ZAR_TP", max_fav_R, max_adv_R, entry, risk_zar)
			break

		# One-time partial bank
		if (not partial_done) and flp >= PARTIAL_TP_ZAR:
			if close_position_partial_market(ticket, PARTIAL_TP_FRACTION):
				partial_done = True
				pos = get_position_by_ticket(ticket)
				print(f"💾 Banked partial at +{flp:.2f} ZAR (fraction={PARTIAL_TP_FRACTION:.2f}).")

		# Indicator refresh for trailing
		try:
			last_atr, df = get_latest_atr(SYMBOL, TIMEFRAME, period=ATR_PERIOD, history=210)
		except Exception as e:
			print(f"ATR refresh error: {e}")
			time.sleep(POLL_SEC)
			continue

		tick = mt5.symbol_info_tick(SYMBOL)
		if tick is None:
			time.sleep(POLL_SEC)
			continue
		last_close = tick.bid if direction == "long" else tick.ask

		# Compute R-progress
		move = (last_close - entry) if direction == "long" else (entry - last_close)
		Rnow = (move / px_per_R) if px_per_R > 0 else 0.0
		max_fav_R = max(max_fav_R, Rnow)
		max_adv_R = min(max_adv_R, Rnow)

		# Confirmation based on CLOSED bars
		try:
			closed_close = float(df["close"].iloc[-2])  # last CLOSED bar close
			closed_move = (closed_close - entry) if direction == "long" else (entry - closed_close)
			closed_R = (closed_move / px_per_R) if px_per_R > 0 else 0.0
		except Exception:
			closed_R = 0.0

		if not confirm_ready:
			if closed_R >= CHAND_START_R:
				confirm_counter += 1
				if confirm_counter >= max(1, int(TRAIL_START_CONFIRM_BARS)):
					confirm_ready = True
					print(f"✅ Trail start confirmed: {confirm_counter} closed bar(s) ≥ {CHAND_START_R:.2f}R.")
			else:
				confirm_counter = 0

		# Optional dynamic TP tightening (only if TP_MODE != "RUNNER")
		if DYN_TP_TIGHTEN_ENABLE and TP_MODE != "RUNNER" and max_fav_R >= DYN_TP_MIN_R:
			if (max_fav_R - Rnow) >= DYN_TP_GIVEBACK_R:
				pos2 = get_position_by_ticket(ticket)
				if pos2 and pos2.tp:
					sign = 1 if direction == "long" else -1
					new_tp = entry + sign * (DYN_TP_KEEP_R_AHEAD * px_per_R)
					if (direction == "long" and new_tp < pos2.tp) or (direction == "short" and new_tp > pos2.tp):
						if modify_sl_tp(ticket, tp=new_tp):
							print(
								f"🎛️ Tighten TP -> {new_tp:.2f} after giveback "
								f"(max {max_fav_R:.2f}R → now {Rnow:.2f}R)"
							)

		# Move to BE at +BE_TRIGGER_R
		if (not be_done) and Rnow >= BE_TRIGGER_R:
			if modify_sl_tp(ticket, sl=entry, tp=None):
				be_done = True
				print(f"🟢 BE set at {entry:.2f} (+{BE_TRIGGER_R}R)")

		# Lock profits at +LOCK_AT_R → SL to +LOCK_TO_R
		if (not lock_done) and Rnow >= LOCK_AT_R:
			lock_price = entry + (LOCK_TO_R * px_per_R) if direction == "long" else entry - (LOCK_TO_R * px_per_R)
			if modify_sl_tp(ticket, sl=lock_price, tp=None):
				lock_done = True
				print(f"🔒 Lock: SL -> {lock_price:.2f} (+{LOCK_TO_R}R) at {Rnow:.2f}R")

		# Start chandelier trail only after confirmation
		if confirm_ready:
			chand_started = True

		if chand_started:
			if Rnow >= 3.5:
				chand_factor = CHAND_ATR_X_TIGHT
			elif Rnow >= 2.5:
				chand_factor = CHAND_ATR_X_MID
			else:
				chand_factor = CHAND_ATR_X_BASE

			new_sl = chandelier_stop(direction, last_atr, chand_factor, last_close)

			# Never move SL backward; respect BE/lock floors
			floor_price = sl_init
			if be_done:
				floor_price = max(floor_price, entry) if direction == "long" else min(floor_price, entry)
			if lock_done:
				lock_floor = entry + LOCK_TO_R * px_per_R if direction == "long" else entry - LOCK_TO_R * px_per_R
				floor_price = max(floor_price, lock_floor) if direction == "long" else min(floor_price, lock_floor)

			if direction == "long":
				new_sl = max(new_sl, floor_price)
			else:
				new_sl = min(new_sl, floor_price)

			pos2 = get_position_by_ticket(ticket)
			cur_sl = pos2.sl if pos2 else None

			improve = (cur_sl is None) or \
					  (direction == "long" and new_sl > cur_sl + 1e-10) or \
					  (direction == "short" and new_sl < cur_sl - 1e-10)

			if improve and modify_sl_tp(ticket, sl=new_sl, tp=None):
				print(f"📈 Trail SL -> {new_sl:.2f} | R≈{Rnow:.2f} | ATR={last_atr:.2f}×{chand_factor}")

		time.sleep(MANAGE_POLL_SEC)
# =================== LOSS-LOCK & DAILY LIMITS ===================

LOSSLOCK_LOOKBACK_MIN      = 60
LOSSLOCK_MAX_TRADES        = 8
LOSSLOCK_NET_R_THRESHOLD   = -2.5
LOSSLOCK_MAX_CONSEC_LOSSES = 4
PROBE_TRADES_ON_LOCK       = 1
MIN_PROBE_RISK_FACTOR      = 0.5

DAILY_R_CAP_R    = -3.0
DAILY_MAX_LOSSES = 4

def last_trades_window(df, minutes=60, max_n=LOSSLOCK_MAX_TRADES):
	"""
	Return last N trades within the lookback minutes (local TZ).
	"""
	if df is None or df.empty:
		return pd.DataFrame(columns=EXPECTED_COLS)
	try:
		d = df.copy()
		d["timestamp"] = to_localized(d["timestamp"])
		cutoff = pd.Timestamp.now(tz=TZ) - pd.Timedelta(minutes=minutes)
		return d[d["timestamp"] >= cutoff].tail(max_n)
	except Exception:
		return pd.DataFrame(columns=df.columns)

def evaluate_losslock(state):
	"""
	Set/clear state['losslock'] based on recent net-R and consecutive losses.
	When engaging, also load PROBE_TRADES_ON_LOCK to allow small test entries.
	"""
	df = read_trades_df()
	win = last_trades_window(df, minutes=LOSSLOCK_LOOKBACK_MIN, max_n=LOSSLOCK_MAX_TRADES)
	if win.empty:
		state["losslock"] = False
		return

	def to_float(x):
		try:
			return float(x)
		except Exception:
			return 0.0

	# Compute recent R values (fallback from profit_zar/risk_zar if R missing)
	R_vals = []
	for _, r in win.iterrows():
		if pd.notna(r.get("R", np.nan)):
			R_vals.append(to_float(r["R"]))
		else:
			rz = to_float(r.get("risk_zar", 0.0))
			pz = to_float(r.get("profit_zar", 0.0))
			R_vals.append(pz / rz if rz > 0 else 0.0)

	netR = float(np.nansum(R_vals))
	consec_losses = 0
	for r in reversed(R_vals):
		if r <= -0.5:
			consec_losses += 1
		else:
			break

	lock = (netR <= LOSSLOCK_NET_R_THRESHOLD) or (consec_losses >= LOSSLOCK_MAX_CONSEC_LOSSES)

	if lock and not state.get("losslock", False):
		state["losslock"] = True
		state["probe_trades_left"] = PROBE_TRADES_ON_LOCK
		print(f"🔒 LOSS-LOCK engaged: netR={netR:.2f}, consec_losses={consec_losses}")
	elif (not lock) and state.get("losslock", False):
		state["losslock"] = False
		print("🔓 LOSS-LOCK cleared by window conditions")

def evaluate_daily_limits(state):
	"""
	Pause for the day if total net-R falls below the limit or too many losses occur.
	"""
	df = read_trades_df()
	if df.empty:
		return
	today = pd.Timestamp.now(tz=TZ).date()
	d = df.copy()
	d["timestamp"] = to_localized(d["timestamp"])
	d = d[d["timestamp"].dt.date == today]

	rr = []
	for _, r in d.iterrows():
		try:
			rz = float(r.get("risk_zar", 0.0))
			pz = float(r.get("profit_zar", 0.0))
			if rz > 0:
				rr.append(pz / rz)
		except Exception:
			pass
	netR = float(np.nansum(rr))
	losses = sum(1 for x in rr if x < 0)

	if (netR <= DAILY_R_CAP_R) or (losses >= DAILY_MAX_LOSSES):
		state["daily_paused"] = True
		print(f"⛔ Daily limit reached: netR={netR:.2f}, losses={losses}. Pausing for the day.")


# =================== TEACHER (ONLINE LEARNER) ===================

class Teacher:
	"""
	Lightweight online learner that nudges entry thresholds via:
	  • Directional WR bias (long vs short)
	  • Strategy WR (BRRT/ORB/FVG)
	  • ATR/body 'gain' biases (relative to thresholds when winners happen)
	Produces (eff_mult, atr_nudge, body_nudge, strategy_wr) advice each loop.
	"""
	def __init__(self, state_file="teacher_state.json"):
		self.state_file = state_file
		self.state = {
			"seen": 0,
			"ema_bias": 0.0,
			"atr_gain_bias": 0.0,
			"body_gain_bias": 0.0,
			"wr_short": 0.0,
			"wr_long": 0.0,
			"last_update": None,
			"strategy_wr": {"BRRT": 0.0, "ORB": 0.0, "FVG": 0.0},
		}

	def load(self):
		if os.path.exists(self.state_file):
			try:
				with open(self.state_file, "r", encoding="utf-8") as f:
					s = json.load(f)
				if isinstance(s, dict):
					self.state.update(s)
				print("🧠 Teacher state loaded.")
			except Exception:
				pass

	def save(self):
		try:
			with open(self.state_file, "w", encoding="utf-8") as f:
				json.dump(self.state, f, indent=2)
		except Exception:
			pass

	def bootstrap_from_log(self, df: pd.DataFrame, tail_n: int = 300):
		"""
		Initialize rough win-rates from existing log so early advice isn't blind.
		"""
		if df is None or df.empty:
			return
		d = df.copy().tail(tail_n)
		to_f = lambda s: pd.to_numeric(s, errors="coerce").fillna(0.0)
		R = to_f(d["R"])
		strat = d["strategy"].astype(str)
		direction = d["direction"].astype(str).str.lower()

		def wr(mask):
			r = R[mask]
			return float((r > 0).mean()) if r.size else 0.0

		for sname in ["BRRT", "ORB", "FVG"]:
			self.state["strategy_wr"][sname] = wr(strat.eq(sname))
		self.state["wr_long"] = wr(direction.eq("long"))
		self.state["wr_short"] = wr(direction.eq("short"))
		self.state["seen"] = int(self.state.get("seen", 0)) + len(d)
		self.state["last_update"] = ts_str()
		self.save()
		print("🧠 Teacher bootstrap complete.")

	def learn_from_trade(self, strategy: str, direction: str, R_value: float,
						 atr_rel: float, body_rel: float, ema_up: bool):
		"""
		Update internal moving averages from one completed trade.
		"""
		lr = 0.05  # learning rate
		self.state["seen"] = int(self.state.get("seen", 0)) + 1

		# Directional WR
		key = "wr_long" if direction == "long" else "wr_short"
		self.state[key] = float((1 - lr) * self.state[key] + lr * (R_value > 0))

		# Strategy WR
		wr0 = self.state["strategy_wr"].get(strategy, 0.0)
		self.state["strategy_wr"][strategy] = float((1 - lr) * wr0 + lr * (R_value > 0))

		# ATR/body gain biases: positive when winning above threshold; negative when losing above threshold
		self.state["atr_gain_bias"] = float(
			np.clip((1 - lr) * self.state.get("atr_gain_bias", 0.0) + lr * (atr_rel - 1.0) * (1 if R_value > 0 else -1),
					-0.25, 0.25)
		)
		self.state["body_gain_bias"] = float(
			np.clip((1 - lr) * self.state.get("body_gain_bias", 0.0) + lr * (body_rel - 1.0) * (1 if R_value > 0 else -1),
					-0.25, 0.25)
		)

		# Optional EMA bias (currently unused for gates, kept for future)
		ema_bias = self.state.get("ema_bias", 0.0)
		ema_bias = float(np.clip((1 - lr) * ema_bias + lr * (0.05 if ema_up else -0.05), -0.25, 0.25))
		self.state["ema_bias"] = ema_bias

		self.state["last_update"] = ts_str()
		self.save()

	def advise(self):
		"""
		Return:
		  eff_mult   -> multiplier for ATR/body thresholds (0.85..1.25)
		  atr_nudge  -> +/- integer percentile shift from ATR gain bias (-10..+10)
		  body_nudge -> +/- integer percentile shift from BODY gain bias (-10..+10)
		  strat_wr   -> dict of per-strategy WR estimates
		"""
		long_wr  = float(self.state.get("wr_long", 0.0))
		short_wr = float(self.state.get("wr_short", 0.0))
		edge = max(long_wr, short_wr) - min(long_wr, short_wr)
		eff_mult = float(np.clip(1.0 + 0.15 * (edge - 0.30), 0.85, 1.25))

		atr_nudge  = int(np.clip(-20 * float(self.state.get("atr_gain_bias", 0.0)),  -10, 10))
		body_nudge = int(np.clip(-20 * float(self.state.get("body_gain_bias", 0.0)), -10, 10))

		return eff_mult, atr_nudge, body_nudge, dict(self.state.get("strategy_wr", {}))
# =================== STATE & RATE LIMITER ===================

STATE_FILE = "gold_bot_state.json"

def load_state():
	"""
	Load or initialize persistent runtime state for intraday controls.
	"""
	st = {
		"day": None,
		"token_bucket": TOKEN_BUCKET_CAPACITY,
		"last_refill": jh_now().timestamp(),
		"no_trade_since": jh_now().timestamp(),
		"losslock": False,
		"probe_trades_left": PROBE_TRADES_ON_LOCK,
		"session_stats": {},
		"daily_paused": False,
		"startup_active": True,
		"startup_new_bars_left": STARTUP_NEW_BARS,
		"last_bar_time_str": "",
		"last_loss_time": 0.0,
		"first_run_ts": jh_now().timestamp(),
		"last_daily_analysis_day": "",
	}
	try:
		if os.path.exists(STATE_FILE):
			with open(STATE_FILE, "r", encoding="utf-8") as f:
				loaded = json.load(f)
				if isinstance(loaded, dict):
					st.update(loaded)
	except Exception as e:
		if DEBUG:
			print(f"⚠️ state load warning: {e} — using defaults")

	# ensure required keys present
	for k, v in {
		"session_stats": {},
		"token_bucket": TOKEN_BUCKET_CAPACITY,
		"daily_paused": False,
		"startup_active": True,
		"startup_new_bars_left": STARTUP_NEW_BARS,
		"last_bar_time_str": "",
		"last_loss_time": 0.0,
	}.items():
		if k not in st:
			st[k] = v
	return st

def save_state(st: dict):
	try:
		with open(STATE_FILE, "w", encoding="utf-8") as f:
			json.dump(st, f, indent=2)
	except Exception as e:
		if DEBUG:
			print(f"⚠️ state save warning: {e}")

def daily_reset_if_needed(st: dict):
	"""
	Reset intraday counters at the local calendar day boundary.
	Also trigger a once-a-day heavy analysis pass.
	"""
	today = jh_now().strftime("%Y-%m-%d")
	if st.get("day") != today:
		st["day"] = today
		st["token_bucket"] = st.get("token_bucket", TOKEN_BUCKET_CAPACITY)
		st["last_refill"] = jh_now().timestamp()
		st["no_trade_since"] = jh_now().timestamp()
		st["losslock"] = False
		st["probe_trades_left"] = PROBE_TRADES_ON_LOCK
		st["daily_paused"] = False
		if "session_stats" not in st or not isinstance(st["session_stats"], dict):
			st["session_stats"] = {}
		print(f"♻️ Daily reset @ {ts_str()}: tokens={st['token_bucket']} paused={st['daily_paused']}")

	# heavy daily analysis (once per calendar day)
	today_str = jh_now().strftime("%Y-%m-%d")
	if st.get("last_daily_analysis_day", "") != today_str:
		try:
			daily_structural_update()  # builds DAILY_REPORT_FILE, loss clusters, etc.
			st["last_daily_analysis_day"] = today_str
			save_state(st)
			print(f"📚 Daily structural analysis done @ {ts_str()}")
		except Exception as e:
			if DEBUG:
				print(f"daily_structural_update error: {e}")

def refill_tokens_if_needed(st: dict):
	"""
	Discrete token-bucket refill. Adds 1 token per TOKEN_BUCKET_REFILL_SEC,
	up to TOKEN_BUCKET_CAPACITY.
	"""
	now_ts = jh_now().timestamp()
	elapsed = now_ts - st["last_refill"]
	if elapsed >= TOKEN_BUCKET_REFILL_SEC:
		tokens_to_add = int(elapsed // TOKEN_BUCKET_REFILL_SEC)
		st["token_bucket"] = min(TOKEN_BUCKET_CAPACITY, st["token_bucket"] + tokens_to_add)
		st["last_refill"] = now_ts
		if tokens_to_add > 0:
			print(f"🔄 RateLimiter: +{tokens_to_add} token(s) -> {st['token_bucket']}")

def _rolling_wr_from_trades(df: pd.DataFrame, window_trades: int = META_RATE_WINDOW_TRADES) -> float:
	if df is None or df.empty:
		return 0.0
	d = df.copy().tail(window_trades)
	R = pd.to_numeric(d.get("R", 0), errors="coerce").fillna(0.0)
	return float((R > 0).mean()) if len(R) else 0.0

def apply_meta_rate_policy(state: dict):
	"""
	Soft auto-tune of TOKEN_BUCKET_* based on rolling win-rate over recent trades.
	Works as a layer on top of profile defaults and intraday guardrails.
	"""
	if not META_RATE_ENABLE:
		return

	df = read_trades_df()
	wr = _rolling_wr_from_trades(df, META_RATE_WINDOW_TRADES)

	base_cap    = globals().get("TOKEN_BUCKET_CAPACITY", 2)
	base_refill = globals().get("TOKEN_BUCKET_REFILL_SEC", 30 * 60)

	if wr <= META_RATE_WR_TIGHT:
		cap = max(1, min(base_cap, 2))
		refill = max(base_refill, META_RATE_MAX_REFILL_S)
	elif wr >= META_RATE_WR_STRONG:
		cap = min(META_RATE_MAX_CAP, max(base_cap, 3))
		refill = min(base_refill, META_RATE_MIN_REFILL_S)
	elif wr >= META_RATE_WR_OK:
		cap = min(META_RATE_MAX_CAP, max(base_cap, 2))
		refill = base_refill
	else:
		cap = max(1, base_cap)
		refill = base_refill

	globals()["TOKEN_BUCKET_CAPACITY"] = int(cap)
	globals()["TOKEN_BUCKET_REFILL_SEC"] = int(refill)

def consume_token(st: dict) -> bool:
	"""
	Consume one token for a new entry attempt. Return False if none available.
	"""
	if st["token_bucket"] <= 0:
		return False
	st["token_bucket"] -= 1
	print(f"⏳ RateLimiter: consume 1 -> {st['token_bucket']} left")
	return True


# =================== HEARTBEAT & DIAGNOSTICS ===================

SNAPSHOT_FILE = "market_snapshot.csv"
SNAPSHOT_COLS = [
	"timestamp","symbol","close","atr","body",
	"ema50","ema200","trend","spread_pts",
	"tokens","losslock","paused"
]

def log_snapshot(row: dict):
	ensure_csv_header(SNAPSHOT_FILE, SNAPSHOT_COLS)
	with open(SNAPSHOT_FILE, "a", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=SNAPSHOT_COLS)
		writer.writerow(row)
		f.flush(); os.fsync(f.fileno())


def heartbeat(*args, **kwargs):
	"""
	Accepts either:
	  heartbeat(state)
	or:
	  heartbeat(df, atr_series, body_series, atr_th, body_th, state, eff_mult)
	"""
	try:
		# Rich signature
		if len(args) >= 6 and hasattr(args[0], "__len__"):
			df, atr_series, body_series, atr_th, body_th, state = args[:6]
			eff_mult = args[6] if len(args) >= 7 else 1.0

			last_close = float(df["close"].iloc[-1])
			last_time  = pd.to_datetime(df["time"].iloc[-1])
			last_atr   = float(atr_series.iloc[-1])
			last_body  = float(body_series.iloc[-1])

			ema_up, ema_down, ema50, ema200, *_ = ema_trend_info(df)
			trend = "UP" if ema_up else ("DOWN" if ema_down else "FLAT")
			try:
				swH = float(df["high"].tail(20).max())
				swL = float(df["low"].tail(20).min())
			except Exception:
				swH, swL = float("nan"), float("nan")

			print(
				f"⏱️ HB {ts_str()} | last={last_time:%H:%M} close={last_close:.2f} | "
				f"ATR={last_atr:.2f} (th {atr_th:.2f}) | BODY={last_body:.2f} (th {body_th:.2f}) | "
				f"trend={trend} | tokens={state.get('token_bucket')} | losslock={state.get('losslock')} "
				f"| paused={state.get('daily_paused')} | eff_mult={eff_mult:.2f} | swH={swH:.2f} swL={swL:.2f}"
			)
			return

		# Simple signature
		state = args[0] if args else kwargs.get("state", {})
		now = ts_str()
		tokens   = state.get("token_bucket", 0) if isinstance(state, dict) else "?"
		paused   = state.get("daily_paused", False) if isinstance(state, dict) else "?"
		losslock = state.get("losslock", False) if isinstance(state, dict) else "?"
		print(f"⏱️ HB {now} | paused={paused} losslock={losslock} tokens={tokens}")
	except Exception as e:
		print(f"HB error: {e}")


def debug_gate_snapshot(state, atr_pctl, body_pctl, atr_th, body_th):
	print(
		"🧪 GATE SNAPSHOT | "
		f"paused={state.get('daily_paused')} losslock={state.get('losslock')} "
		f"tokens={state.get('token_bucket')} "
		f"pctl(atr,body)=({atr_pctl},{body_pctl}) "
		f"th(atr,body)=({atr_th:.2f},{body_th:.2f})"
	)
# =================== ORDER HELPERS ===================

SLIPPAGE_POINTS = 20

def current_price(symbol, side_for_entry: str):
	tick = mt5.symbol_info_tick(symbol)
	if tick is None:
		raise RuntimeError("No tick available")
	if side_for_entry == "buy":
		return float(tick.ask)
	elif side_for_entry == "sell":
		return float(tick.bid)
	return float((tick.bid + tick.ask) / 2.0)

def _send_with_filling(req: dict):
	req["type_filling"] = mt5.ORDER_FILLING_RETURN
	r = mt5.order_send(req)
	if r is not None and r.retcode == mt5.TRADE_RETCODE_DONE:
		return r
	if r is None or r.retcode in (10030, mt5.TRADE_RETCODE_INVALID_FILL):
		req["type_filling"] = mt5.ORDER_FILLING_IOC
		r2 = mt5.order_send(req)
		if r2 is not None and r2.retcode == mt5.TRADE_RETCODE_DONE:
			return r2
		req["type_filling"] = mt5.ORDER_FILLING_FOK
		r3 = mt5.order_send(req)
		return r3
	return r

def send_order(symbol, direction, entry, sl, tp, lot, info, comment_tag="GOLD_BOT_V7"):
	digits = info.digits
	entry = normalize_price(entry, digits)
	sl    = normalize_price(sl, digits)
	tp    = normalize_price(tp, digits) if tp else 0.0

	req = {
		"action": mt5.TRADE_ACTION_DEAL,
		"symbol": symbol,
		"volume": float(lot),
		"type": mt5.ORDER_TYPE_BUY if direction == "long" else mt5.ORDER_TYPE_SELL,
		"price": entry,
		"sl": sl,
		"tp": tp if tp else 0.0,
		"deviation": SLIPPAGE_POINTS,
		"magic": 20251015,
		"comment": comment_tag,  # strategy tag goes here
		"type_time": mt5.ORDER_TIME_GTC,
	}
	r = _send_with_filling(req)
	if r is None:
		raise RuntimeError(f"order_send None: {mt5.last_error()}")
	if r.retcode != mt5.TRADE_RETCODE_DONE:
		raise RuntimeError(f"order_send failed: {r.retcode} {r._asdict()}")
	return r.order, r

def modify_sl_tp(ticket, sl=None, tp=None):
	pos = get_position_by_ticket(ticket)
	if not pos:
		return False
	info = mt5.symbol_info(pos.symbol)
	digits = info.digits if info else 2
	new_sl = normalize_price(sl, digits) if sl is not None else pos.sl
	new_tp = normalize_price(tp, digits) if tp is not None else pos.tp
	req = {
		"action": mt5.TRADE_ACTION_SLTP,
		"position": pos.ticket,
		"sl": new_sl,
		"tp": new_tp,
		"symbol": pos.symbol
	}
	r = _send_with_filling(req)
	return (r is not None) and (r.retcode in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED))

def get_position_by_ticket(ticket):
	positions = mt5.positions_get()
	if positions is None:
		return None
	for p in positions:
		if p.ticket == ticket:
			return p
	return None

def get_open_positions_count(symbol=SYMBOL):
	positions = mt5.positions_get(symbol=symbol)
	return 0 if positions is None else len(positions)

def chandelier_stop(direction, atr, factor, last_close):
	if direction == "long":
		return last_close - atr * factor
	else:
		return last_close + atr * factor

def floating_profit_zar(ticket):
	pos = get_position_by_ticket(ticket)
	if not pos:
		return 0.0
	return float(pos.profit)

def close_position_market(ticket):
	pos = get_position_by_ticket(ticket)
	if not pos:
		return
	side = "sell" if pos.type == mt5.POSITION_TYPE_BUY else "buy"
	price = current_price(pos.symbol, side)
	req = {
		"action": mt5.TRADE_ACTION_DEAL,
		"position": pos.ticket,
		"symbol": pos.symbol,
		"volume": pos.volume,
		"type": mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
		"price": price,
		"deviation": SLIPPAGE_POINTS,
		"magic": 20251015,
		"comment": "GOLD_BOT_CLOSE",
	}
	r = _send_with_filling(req)
	if r is None:
		print(f"⚠️ close_position: order_send None {mt5.last_error()}")
	elif r.retcode != mt5.TRADE_RETCODE_DONE:
		print(f"⚠️ close_position retcode={r.retcode} {r._asdict()}")

def close_position_partial_market(ticket, fraction: float):
	"""
	Market-close a fraction (0 < fraction < 1) of the current position volume.
	Respects broker min volume and step.
	"""
	pos = get_position_by_ticket(ticket)
	if not pos or not (0.0 < fraction < 1.0):
		return False
	info = mt5.symbol_info(pos.symbol)
	if not info:
		return False

	vol_min = info.volume_min
	step    = info.volume_step if info.volume_step > 0 else 0.01
	target  = pos.volume * float(fraction)

	steps = round(target / step)
	volume_to_close = max(vol_min, steps * step)
	volume_to_close = min(volume_to_close, pos.volume - vol_min) if pos.volume > vol_min else vol_min
	if volume_to_close <= 0:
		return False

	side = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
	px_side = "sell" if pos.type == mt5.POSITION_TYPE_BUY else "buy"
	price = current_price(pos.symbol, px_side)

	req = {
		"action": mt5.TRADE_ACTION_DEAL,
		"position": pos.ticket,
		"symbol": pos.symbol,
		"volume": volume_to_close,
		"type": side,
		"price": price,
		"deviation": SLIPPAGE_POINTS,
		"magic": 20251015,
		"comment": "GOLD_BOT_PARTIAL",
	}
	r = _send_with_filling(req)
	if r is None:
		print(f"⚠️ partial_close: order_send None {mt5.last_error()}")
		return False
	if r.retcode != mt5.TRADE_RETCODE_DONE:
		print(f"⚠️ partial_close retcode={r.retcode} {r._asdict()}")
		return False
	print(f"🧩 Partial closed {volume_to_close} lot(s).")
	return True


# =================== POSITION MANAGEMENT ===================

BE_TRIGGER_R   = 0.80
LOCK_AT_R      = 2.00
LOCK_TO_R      = 0.60
CHAND_START_R  = 2.00
CHAND_ATR_X_BASE = 6.0
CHAND_ATR_X_MID  = 4.0
CHAND_ATR_X_TIGHT= 2.5

TP_MODE = "RUNNER"     # or "FIXED_ZAR"
PARTIAL_TP_ZAR      = 400.0
PARTIAL_TP_FRACTION = 0.50
HARD_TP_ZAR         = 700.0
TRAIL_START_CONFIRM_BARS = 1
HARD_RISK_ZAR_CAP   = 125.0

def finalize_trade_log(ticket, exit_reason, max_fav_R, max_adv_R, entry, risk_zar):
	"""
	Safely finalize last trade row (profit, R, max swings, exit_price).
	"""
	time.sleep(0.8)
	df = read_trades_df()
	if df.empty:
		return
	idx = df[df["ticket"] == str(ticket)].index
	if len(idx) == 0:
		return
	i = idx[-1]

	profit_zar = 0.0
	exit_price = float("nan")
	try:
		fr = datetime.now(timezone.utc) - timedelta(days=5)
		to = datetime.now(timezone.utc)
		deals = mt5.history_deals_get(fr, to)
		if deals:
			for d in deals:
				if d.position_id == ticket:
					profit_zar += float(d.profit or 0.0)
					exit_price = float(d.price or float("nan"))
	except Exception:
		pass

	R = (profit_zar / risk_zar) if risk_zar and risk_zar > 0 else 0.0

	df.loc[i, "exit_price"]    = f"{exit_price:.2f}" if exit_price == exit_price else ""
	df.loc[i, "profit_zar"]    = f"{profit_zar:.2f}"
	df.loc[i, "R"]             = f"{R:.2f}"
	df.loc[i, "max_fav_R"]     = f"{max_fav_R:.2f}"
	df.loc[i, "max_adverse_R"] = f"{max_adv_R:.2f}"
	df.loc[i, "reason_exit"]   = exit_reason

	tmp = LOG_FILE + ".tmp"
	try:
		with open(tmp, "w", encoding="utf-8", newline="") as f:
			df.to_csv(f, index=False, lineterminator="\n")
			f.flush(); os.fsync(f.fileno())
		os.replace(tmp, LOG_FILE)
	except Exception as e:
		print(f"⚠️ trade_log write failed: {e}")

	if R < 0:
		try:
			st = load_state()
			st["last_loss_time"] = time.time()
			save_state(st)
		except Exception:
			pass

	print(f"🧾 Closed: profit={profit_zar:.2f} ZAR | R={R:.2f} | reason={exit_reason}")

	try:
		learn_update()
	except Exception:
		pass

def manage_position(ticket, direction, entry, sl_init, lot, risk_zar, state):
	"""
	BE at +0.8R → lock at +2R (to +0.6R) → chandelier trailing after 1 closed bar ≥ +2R.
	Also: one-time partial at +400 ZAR, hard TP at +700 ZAR, hard kill at -HARD_RISK_ZAR_CAP.
	"""
	pos = get_position_by_ticket(ticket)
	if not pos:
		print("⚠️ manage_position: position not found.")
		return

	max_fav_R = 0.0
	max_adv_R = 0.0
	be_done = False
	chand_started = False
	lock_done = False
	partial_done = False

	px_per_R = abs(entry - sl_init)
	confirm_counter = 0
	confirm_ready = False

	while True:
		pos = get_position_by_ticket(ticket)
		if not pos:
			print("ℹ️ Position closed (external SL/TP or manual). Finalizing…")
			finalize_trade_log(ticket, "EXTERNAL_EXIT", max_fav_R, max_adv_R, entry, risk_zar)
			break

		# Hard kill-switch (floating PnL)
		flp = floating_profit_zar(ticket)
		if flp <= -HARD_RISK_ZAR_CAP:
			print(f"🛑 HARD Kill-Switch: floating={flp:.2f} ZAR <= -{HARD_RISK_ZAR_CAP:.2f} — closing.")
			close_position_market(ticket)
			finalize_trade_log(ticket, "HARD_KILL_SWITCH", max_fav_R, max_adv_R, entry, risk_zar)
			break

		# ZAR-based exits
		if flp >= HARD_TP_ZAR:
			print(f"🏁 HARD ZAR TP: +{flp:.2f} ZAR >= {HARD_TP_ZAR:.2f} — closing full.")
			close_position_market(ticket)
			finalize_trade_log(ticket, "HARD_ZAR_TP", max_fav_R, max_adv_R, entry, risk_zar)
			break

		if (not partial_done) and flp >= PARTIAL_TP_ZAR:
			if close_position_partial_market(ticket, PARTIAL_TP_FRACTION):
				partial_done = True
				pos = get_position_by_ticket(ticket)
				print(f"💾 Banked partial at +{flp:.2f} ZAR (fraction={PARTIAL_TP_FRACTION:.2f}).")

		# Refresh ATR & recent data
		try:
			last_atr, df = get_latest_atr(SYMBOL, TIMEFRAME, period=ATR_PERIOD, history=210)
		except Exception as e:
			print(f"ATR refresh error: {e}")
			time.sleep(MANAGE_POLL_SEC); continue

		tick = mt5.symbol_info_tick(SYMBOL)
		if tick is None:
			time.sleep(MANAGE_POLL_SEC); continue
		last_close = tick.bid if direction == "long" else tick.ask

		move = (last_close - entry) if direction == "long" else (entry - last_close)
		Rnow = (move / px_per_R) if px_per_R > 0 else 0.0
		max_fav_R = max(max_fav_R, Rnow)
		max_adv_R = min(max_adv_R, Rnow)

		# confirmation: need 1 closed bar ≥ +2R before enabling chand trailing
		try:
			closed_close = float(df["close"].iloc[-2])
			closed_move = (closed_close - entry) if direction == "long" else (entry - closed_close)
			closed_R = (closed_move / px_per_R) if px_per_R > 0 else 0.0
		except Exception:
			closed_R = 0.0

		if not confirm_ready:
			if closed_R >= CHAND_START_R:
				confirm_counter += 1
				if confirm_counter >= max(1, int(TRAIL_START_CONFIRM_BARS)):
					confirm_ready = True
					print(f"✅ Trail start confirmed: {confirm_counter} closed bar(s) ≥ {CHAND_START_R:.2f}R.")
			else:
				confirm_counter = 0

		# Dynamic TP tighten (only for FIXED_ZAR modes)
		if DYN_TP_TIGHTEN_ENABLE and TP_MODE != "RUNNER" and max_fav_R >= DYN_TP_MIN_R:
			if (max_fav_R - Rnow) >= DYN_TP_GIVEBACK_R:
				pos2 = get_position_by_ticket(ticket)
				if pos2 and pos2.tp:
					sign = 1 if direction == "long" else -1
					new_tp = entry + sign * (DYN_TP_KEEP_R_AHEAD * px_per_R)
					if (direction == "long" and new_tp < pos2.tp) or (direction == "short" and new_tp > pos2.tp):
						if modify_sl_tp(ticket, tp=new_tp):
							print(f"🎛️ Tighten TP -> {new_tp:.2f} after giveback (max {max_fav_R:.2f}R → now {Rnow:.2f}R)")

		# Move to BE at +0.8R
		if not be_done and Rnow >= BE_TRIGGER_R:
			if modify_sl_tp(ticket, sl=entry, tp=None):
				be_done = True
				print(f"🟢 BE set at {entry:.2f} (+{BE_TRIGGER_R}R)")

		# Lock profits at +2R → SL to +0.6R
		if not lock_done and Rnow >= LOCK_AT_R:
			lock_price = entry + (LOCK_TO_R * px_per_R) if direction == "long" else entry - (LOCK_TO_R * px_per_R)
			if modify_sl_tp(ticket, sl=lock_price, tp=None):
				lock_done = True
				print(f"🔒 Lock: SL -> {lock_price:.2f} (+{LOCK_TO_R}R) at {Rnow:.2f}R")

		# Start chandelier trailing after confirmation
		if confirm_ready:
			if Rnow >= 3.5:   chand_factor = CHAND_ATR_X_TIGHT
			elif Rnow >= 2.5: chand_factor = CHAND_ATR_X_MID
			else:             chand_factor = CHAND_ATR_X_BASE

			new_sl = chandelier_stop(direction, last_atr, chand_factor, last_close)

			# never move SL backward; respect BE/lock floors
			floor_price = sl_init
			if be_done:
				floor_price = max(floor_price, entry) if direction == "long" else min(floor_price, entry)
			if lock_done:
				lock_floor = entry + LOCK_TO_R*px_per_R if direction=="long" else entry - LOCK_TO_R*px_per_R
				floor_price = max(floor_price, lock_floor) if direction=="long" else min(floor_price, lock_floor)

			if direction == "long":
				new_sl = max(new_sl, floor_price)
			else:
				new_sl = min(new_sl, floor_price)

			pos2 = get_position_by_ticket(ticket)
			cur_sl = pos2.sl if pos2 else None
			improve = (cur_sl is None) or \
					  (direction == "long" and new_sl > cur_sl + 1e-10) or \
					  (direction == "short" and new_sl < cur_sl - 1e-10)

			if improve and modify_sl_tp(ticket, sl=new_sl, tp=None):
				print(f"📈 Trail SL -> {new_sl:.2f} | R≈{Rnow:.2f} | ATR={last_atr:.2f}×{chand_factor}")

		time.sleep(MANAGE_POLL_SEC)
def summarize_last_loss():
	df = read_trades_df()
	if df.empty:
		return
	last = df.tail(1).iloc[0]
	try:
		R = float(last.get("R", 0))
	except Exception:
		R = 0.0
	if R >= 0:
		return
	strat = str(last.get("strategy", "")).upper()
	trend = str(last.get("ema_trend", "")).upper()
	atr   = _safe_float(last.get("atr"), 0.0)
	atr_th= _safe_float(last.get("atr_th"), 0.0)
	body  = _safe_float(last.get("body"), 0.0)
	body_th=_safe_float(last.get("body_th"), 0.0)
	atr_rel  = atr / max(atr_th, 1e-9)
	body_rel = body / max(body_th, 1e-9)
	sess = _session_block_from_ts(pd.Series([last.get("timestamp")])).iloc[0]
	print(
		f"🧩 LOSS CONTEXT: strat={strat} session={sess} trend={trend} "
		f"atr_rel={atr_rel:.2f} body_rel={body_rel:.2f} "
		f"notes='{last.get('notes','')}'"
	)


# =================== LOSS-LOCK / DAILY LIMITS ===================

LOSSLOCK_LOOKBACK_MIN      = 60
LOSSLOCK_MAX_TRADES        = 8
LOSSLOCK_NET_R_THRESHOLD   = -2.5
LOSSLOCK_MAX_CONSEC_LOSSES = 4
PROBE_TRADES_ON_LOCK       = 1
MIN_PROBE_RISK_FACTOR      = 0.5

DAILY_R_CAP_R      = -3.0
DAILY_MAX_LOSSES   = 4

def last_trades_window(df, minutes=60, max_n=LOSSLOCK_MAX_TRADES):
	if df.empty:
		return df
	try:
		d2 = df.copy()
		d2["timestamp"] = to_localized(d2["timestamp"])
		cutoff = pd.Timestamp.now(tz=TZ) - pd.Timedelta(minutes=minutes)
		return d2[d2["timestamp"] >= cutoff].tail(max_n)
	except Exception:
		return pd.DataFrame(columns=df.columns)

def evaluate_losslock(state):
	df = read_trades_df()
	win = last_trades_window(df, minutes=LOSSLOCK_LOOKBACK_MIN, max_n=LOSSLOCK_MAX_TRADES)
	if win.empty:
		state["losslock"] = False
		return

	def to_float(x):
		try:
			return float(x)
		except Exception:
			return 0.0

	R_vals = []
	for _, r in win.iterrows():
		if "R" in r and pd.notna(r["R"]):
			R_vals.append(to_float(r["R"]))
		elif "profit_zar" in r and "risk_zar" in r:
			rz = to_float(r["risk_zar"])
			pz = to_float(r["profit_zar"])
			R_vals.append(pz / rz if rz > 0 else 0.0)
		else:
			R_vals.append(0.0)

	netR = float(np.nansum(R_vals))
	consec_losses = 0
	for r in reversed(R_vals):
		if r <= -0.5:
			consec_losses += 1
		else:
			break

	lock = (netR <= LOSSLOCK_NET_R_THRESHOLD) or (consec_losses >= LOSSLOCK_MAX_CONSEC_LOSSES)
	if lock and not state.get("losslock", False):
		state["losslock"] = True
		state["probe_trades_left"] = PROBE_TRADES_ON_LOCK
		print(f"🔒 LOSS-LOCK engaged: netR={netR:.2f}, consec_losses={consec_losses}")
	elif not lock and state.get("losslock", False):
		state["losslock"] = False
		print("🔓 LOSS-LOCK cleared by window conditions")

def today_netR_losses():
	df = read_trades_df()
	if df.empty:
		return 0.0, 0
	d = df.copy()
	d["timestamp"] = to_localized(d["timestamp"])
	today = pd.Timestamp.now(tz=TZ).date()
	d = d[d["timestamp"].dt.date == today]
	r = pd.to_numeric(d["R"], errors="coerce").fillna(0.0)
	return float(r.sum()), int((r < 0).sum())

def evaluate_daily_limits(state):
	df = read_trades_df()
	if df.empty:
		return
	today = pd.Timestamp.now(tz=TZ).date()
	d = df.copy()
	d["timestamp"] = to_localized(d["timestamp"])
	d = d[d["timestamp"].dt.date == today]
	rr = []
	for _, r in d.iterrows():
		try:
			rz = float(r.get("risk_zar", 0) or 0.0)
			pz = float(r.get("profit_zar", 0) or 0.0)
			rr.append(pz / rz if rz > 0 else 0.0)
		except Exception:
			pass
	netR = float(np.nansum(rr))
	losses = sum(1 for x in rr if x < 0)
	if (netR <= DAILY_R_CAP_R) or (losses >= DAILY_MAX_LOSSES):
		state["daily_paused"] = True
		print(f"⛔ Daily limit reached: netR={netR:.2f}, losses={losses}. Pausing for the day.")


# =================== LEARNING “TEACHER” ===================

class Teacher:
	"""
	Lightweight on-line learner that:
	  • Tracks WR by direction and by strategy
	  • Produces small nudges (percentile deltas) to tighten/relax gates
	  • Exposes advise() for main loop to consume
	"""
	def __init__(self, state_file="teacher_state.json"):
		self.state_file = state_file
		self.state = {
			"seen": 0,
			"ema_bias": 0.0,
			"atr_gain_bias": 0.0,
			"body_gain_bias": 0.0,
			"wr_short": 0.0,
			"wr_long": 0.0,
			"last_update": None,
			"strategy_wr": {"BRRT": 0.0, "ORB": 0.0, "FVG": 0.0},
		}

	def load(self):
		if os.path.exists(self.state_file):
			try:
				with open(self.state_file, "r", encoding="utf-8") as f:
					s = json.load(f)
				if isinstance(s, dict):
					self.state.update(s)
				print("🧠 Teacher state loaded.")
			except Exception:
				pass

	def save(self):
		try:
			with open(self.state_file, "w", encoding="utf-8") as f:
				json.dump(self.state, f, indent=2)
		except Exception:
			pass

	def bootstrap_from_log(self, df: pd.DataFrame, tail_n: int = 300):
		if df.empty:
			return
		d = df.copy().tail(tail_n)
		to_f = lambda s: pd.to_numeric(s, errors="coerce").fillna(0.0)
		R = to_f(d["R"])
		strat = d["strategy"].astype(str)
		def wr(mask):
			r = R[mask]
			return float((r > 0).mean()) if r.size else 0.0
		for sname in ["BRRT", "ORB", "FVG"]:
			self.state["strategy_wr"][sname] = wr(strat.eq(sname))
		self.state["wr_long"] = wr(d["direction"].astype(str).str.lower().eq("long"))
		self.state["wr_short"] = wr(d["direction"].astype(str).str.lower().eq("short"))
		self.state["seen"] = int(self.state.get("seen", 0)) + len(d)
		self.state["last_update"] = ts_str()
		self.save()
		print("🧠 Teacher bootstrap complete.")

	def learn_from_trade(self, strategy: str, direction: str,
						 R_value: float, atr_rel: float, body_rel: float, ema_up: bool):
		lr = 0.05
		self.state["seen"] = int(self.state.get("seen", 0)) + 1
		key = "wr_long" if direction == "long" else "wr_short"
		self.state[key] = float((1 - lr) * self.state[key] + lr * (R_value > 0))
		# strategy-specific WR
		wr0 = self.state["strategy_wr"].get(strategy, 0.0)
		self.state["strategy_wr"][strategy] = float((1 - lr) * wr0 + lr * (R_value > 0))
		# nudges from relative thresholds
		self.state["atr_gain_bias"]  = float(np.clip(
			(1 - lr) * self.state.get("atr_gain_bias", 0.0) + lr * (atr_rel - 1.0) * (1 if R_value > 0 else -1),
			-0.25, 0.25
		))
		self.state["body_gain_bias"] = float(np.clip(
			(1 - lr) * self.state.get("body_gain_bias", 0.0) + lr * (body_rel - 1.0) * (1 if R_value > 0 else -1),
			-0.25, 0.25
		))
		self.state["last_update"] = ts_str()
		self.save()

	def advise(self):
		"""
		Returns:
		  eff_mult   : multiplier applied to ATR/BODY thresholds (1±~0.25)
		  atr_nudge  : integer percentile shift for ATR threshold
		  body_nudge : integer percentile shift for BODY threshold
		  strat_wr   : dict of recent per-strategy WR (for reorder hints)
		"""
		long_wr = self.state.get("wr_long", 0.0)
		short_wr = self.state.get("wr_short", 0.0)
		edge = max(long_wr, short_wr) - min(long_wr, short_wr)
		eff_mult = float(np.clip(1.0 + 0.15 * (edge - 0.3), 0.85, 1.25))
		atr_nudge  = int(np.clip(-20 * self.state.get("atr_gain_bias", 0.0), -10, 10))
		body_nudge = int(np.clip(-20 * self.state.get("body_gain_bias", 0.0), -10, 10))
		return eff_mult, atr_nudge, body_nudge, self.state.get("strategy_wr", {})
# =================== STATE & RATE LIMITER ===================

STATE_FILE = "gold_bot_state.json"

def load_state():
	st = {
		"day": None,
		"token_bucket": TOKEN_BUCKET_CAPACITY,
		"last_refill": jh_now().timestamp(),
		"no_trade_since": jh_now().timestamp(),
		"losslock": False,
		"probe_trades_left": PROBE_TRADES_ON_LOCK,
		"session_stats": {},
		"daily_paused": False,
		"startup_active": True,
		"startup_new_bars_left": STARTUP_NEW_BARS,
		"last_bar_time_str": "",
		"last_loss_time": 0.0,
		"first_run_ts": jh_now().timestamp(),
		"last_daily_analysis_day": "",
	}
	try:
		if os.path.exists(STATE_FILE):
			with open(STATE_FILE, "r", encoding="utf-8") as f:
				loaded = json.load(f)
				if isinstance(loaded, dict):
					st.update(loaded)
	except Exception as e:
		if DEBUG:
			print(f"⚠️ state load warning: {e} — using defaults")
	# ensure required keys exist
	defaults = {
		"session_stats": {},
		"token_bucket": TOKEN_BUCKET_CAPACITY,
		"daily_paused": False,
		"startup_active": True,
		"startup_new_bars_left": STARTUP_NEW_BARS,
		"last_bar_time_str": "",
		"last_loss_time": 0.0,
	}
	for k, v in defaults.items():
		if k not in st:
			st[k] = v
	return st

def save_state(st):
	try:
		with open(STATE_FILE, "w", encoding="utf-8") as f:
			json.dump(st, f, indent=2)
	except Exception as e:
		if DEBUG:
			print(f"⚠️ state save warning: {e}")

def daily_structural_update():
	"""
	Heavy once-a-day analysis:
	  • refresh loss clusters (DAILY_REPORT_FILE)
	  • write learn_state timestamps
	"""
	df = read_trades_df()
	rep = analyze_trade_history(out_json=DAILY_REPORT_FILE)
	st = read_learn_state()
	st["last_daily_run"] = ts_str()
	st["last_loss_report"] = rep or {}
	write_learn_state(st)

def daily_reset_if_needed(st):
	today = jh_now().strftime("%Y-%m-%d")
	if st.get("day") != today:
		st["day"] = today
		st["token_bucket"] = st.get("token_bucket", TOKEN_BUCKET_CAPACITY)
		st["last_refill"] = jh_now().timestamp()
		st["no_trade_since"] = jh_now().timestamp()
		st["losslock"] = False
		st["probe_trades_left"] = PROBE_TRADES_ON_LOCK
		st["daily_paused"] = False
		if "session_stats" not in st or not isinstance(st["session_stats"], dict):
			st["session_stats"] = {}
		print(f"♻️ Daily reset @ {ts_str()}: tokens={st['token_bucket']} paused={st['daily_paused']}")

	# run daily heavy pass once per calendar day
	today_str = jh_now().strftime("%Y-%m-%d")
	if st.get("last_daily_analysis_day", "") != today_str:
		try:
			daily_structural_update()
			st["last_daily_analysis_day"] = today_str
			save_state(st)
			print(f"📚 Daily structural analysis done @ {ts_str()}")
		except Exception as e:
			if DEBUG:
				print(f"daily_structural_update error: {e}")

def refill_tokens_if_needed(st):
	now_ts = jh_now().timestamp()
	elapsed = now_ts - st["last_refill"]
	if elapsed >= TOKEN_BUCKET_REFILL_SEC:
		tokens_to_add = int(elapsed // TOKEN_BUCKET_REFILL_SEC)
		st["token_bucket"] = min(TOKEN_BUCKET_CAPACITY, st["token_bucket"] + tokens_to_add)
		st["last_refill"] = now_ts
		if tokens_to_add > 0:
			print(f"🔄 RateLimiter: +{tokens_to_add} token(s) -> {st['token_bucket']}")

def _rolling_wr_from_trades(df: pd.DataFrame, window_trades: int = META_RATE_WINDOW_TRADES) -> float:
	if df is None or df.empty:
		return 0.0
	d = df.copy().tail(window_trades)
	R = pd.to_numeric(d.get("R", 0), errors="coerce").fillna(0.0)
	return float((R > 0).mean()) if len(R) else 0.0

def apply_meta_rate_policy(state):
	"""
	Adjust TOKEN_BUCKET_* based on rolling win-rate; soft override around baseline.
	"""
	if not META_RATE_ENABLE:
		return

	df = read_trades_df()
	wr = _rolling_wr_from_trades(df, META_RATE_WINDOW_TRADES)

	base_cap    = globals().get("TOKEN_BUCKET_CAPACITY", 2)
	base_refill = globals().get("TOKEN_BUCKET_REFILL_SEC", 30 * 60)

	if wr <= META_RATE_WR_TIGHT:
		cap = max(1, min(base_cap, 2))
		refill = max(base_refill, META_RATE_MAX_REFILL_S)
	elif wr >= META_RATE_WR_STRONG:
		cap = min(META_RATE_MAX_CAP, max(base_cap, 3))
		refill = min(base_refill, META_RATE_MIN_REFILL_S)
	elif wr >= META_RATE_WR_OK:
		cap = min(META_RATE_MAX_CAP, max(base_cap, 2))
		refill = base_refill
	else:
		cap = max(1, base_cap)
		refill = base_refill

	globals()["TOKEN_BUCKET_CAPACITY"]  = int(cap)
	globals()["TOKEN_BUCKET_REFILL_SEC"] = int(refill)

def consume_token(st):
	if st["token_bucket"] <= 0:
		return False
	st["token_bucket"] -= 1
	print(f"⏳ RateLimiter: consume 1 -> {st['token_bucket']} left")
	return True


# =================== HEARTBEAT / DIAGNOSTICS ===================

HEARTBEAT_EVERY_SEC = 45


def heartbeat(*args, **kwargs):
	"""
	Accepts either:
	  heartbeat(state)
	or:
	  heartbeat(df, atr_series, body_series, atr_th, body_th, state, eff_mult)
	"""
	try:
		# Rich signature
		if len(args) >= 6 and hasattr(args[0], "__len__"):
			df, atr_series, body_series, atr_th, body_th, state = args[:6]
			eff_mult = args[6] if len(args) >= 7 else 1.0

			last_close = float(df["close"].iloc[-1])
			last_time  = pd.to_datetime(df["time"].iloc[-1])
			last_atr   = float(atr_series.iloc[-1])
			last_body  = float(body_series.iloc[-1])

			ema_up, ema_down, ema50, ema200, *_ = ema_trend_info(df)
			trend = "UP" if ema_up else ("DOWN" if ema_down else "FLAT")
			try:
				swH = float(df["high"].tail(20).max())
				swL = float(df["low"].tail(20).min())
			except Exception:
				swH, swL = float("nan"), float("nan")

			print(
				f"⏱️ HB {ts_str()} | last={last_time:%H:%M} close={last_close:.2f} | "
				f"ATR={last_atr:.2f} (th {atr_th:.2f}) | BODY={last_body:.2f} (th {body_th:.2f}) | "
				f"trend={trend} | tokens={state.get('token_bucket')} | losslock={state.get('losslock')} "
				f"| paused={state.get('daily_paused')} | eff_mult={eff_mult:.2f} | swH={swH:.2f} swL={swL:.2f}"
			)
			return

		# Simple signature
		state = args[0] if args else kwargs.get("state", {})
		now = ts_str()
		tokens   = state.get("token_bucket", 0) if isinstance(state, dict) else "?"
		paused   = state.get("daily_paused", False) if isinstance(state, dict) else "?"
		losslock = state.get("losslock", False) if isinstance(state, dict) else "?"
		print(f"⏱️ HB {now} | paused={paused} losslock={losslock} tokens={tokens}")
	except Exception as e:
		print(f"HB error: {e}")


def debug_gate_snapshot(state, atr_pctl, body_pctl, atr_th, body_th):
	print(
		"🧪 GATE SNAPSHOT | "
		f"paused={state.get('daily_paused')} losslock={state.get('losslock')} "
		f"tokens={state.get('token_bucket')} "
		f"pctl(atr,body)=({atr_pctl},{body_pctl}) "
		f"th(atr,body)=({atr_th:.2f},{body_th:.2f})"
	)


# =================== RUNTIME HELPERS ===================

def within_trade_window(dt, windows=TRADE_WINDOWS):
	t = dt.strftime("%H:%M")
	for start, end in windows:
		if start <= t <= end:
			return True
	return False

def open_positions(symbol=SYMBOL):
	return list(mt5.positions_get(symbol=symbol) or [])

def positions_by_strategy(symbol=SYMBOL):
	out = {"BRRT": [], "ORB": [], "FVG": []}
	ps = open_positions(symbol)
	for p in ps:
		tag = (p.comment or "").upper()
		if tag in out:
			out[tag].append(p)
	return out

def worst_case_floating_loss_zar(symbol=SYMBOL):
	tot = 0.0
	for p in open_positions(symbol):
		side = "long" if p.type == mt5.POSITION_TYPE_BUY else "short"
		if p.sl:
			try:
				loss = -mt5.order_calc_profit(
					mt5.ORDER_TYPE_BUY if side == "long" else mt5.ORDER_TYPE_SELL,
					p.symbol, p.volume, p.price_open, p.sl
				)
				if loss and loss > 0:
					tot += loss
			except Exception:
				pass
	return float(tot)

def maybe_flip_off_test_mode(state):
	global TEST_MODE
	if not TEST_MODE:
		return False
	first = state.get("first_run_ts", jh_now().timestamp())
	if (jh_now().timestamp() - first) >= TEST_MODE_EXPIRES_HOURS * 3600:
		TEST_MODE = False
		print("🔁 TEST_MODE disabled after window — running with normal constraints.")
		return True
	return False


# =================== MAIN LOOP (part 1: setup) ===================

def main():
	
	global BE_TRIGGER_R, LOCK_AT_R, STRATEGIES_ENABLED, TOKEN_BUCKET_CAPACITY, TOKEN_BUCKET_REFILL_SEC
global BE_TRIGGER_R, LOCK_AT_R

	ensure_csv_header(LOG_FILE, EXPECTED_COLS)
	ensure_csv_header(SIGNAL_LOG_FILE, SIGNAL_COLS)
	ensure_csv_header(SNAPSHOT_FILE, SNAPSHOT_COLS)

	teacher = Teacher()
	teacher.load()
	df_boot = read_trades_df()
	if not df_boot.empty and int(teacher.state.get("seen", 0)) == 0:
		teacher.bootstrap_from_log(df_boot)

	acc = ensure_mt5()
	info = symbol_prepare(SYMBOL)
	state = load_state()
	daily_reset_if_needed(state)

	print(f"⚙️ MODE={'RUNNER' if TP_MODE=='RUNNER' else 'FIXED_ZAR'} | TEACHER=ON | STRATS={','.join(STRATEGIES_ENABLED)} | TEST_MODE={TEST_MODE}")
	last_hb = 0.0

	try:
		while True:
			try:
				# auto end test mode after window
				if maybe_flip_off_test_mode(state):
					pass

				# trading hours
				if not within_trade_window(jh_now()):
					time.sleep(POLL_SEC)
					continue

				daily_reset_if_needed(state)

				# spread guard
				if not spread_ok(SYMBOL):
					if DEBUG:
						print("🧱 Spread too wide — skip.")
					time.sleep(POLL_SEC)
					continue

				# loss cooldown
				netR_today, losses_today = today_netR_losses()
				if losses_today > 0 and (time.time() - state.get("last_loss_time", 0)) < LOSS_COOLDOWN_SEC:
					if DEBUG:
						print("🧊 Loss cooldown active — skipping.")
					time.sleep(POLL_SEC)
					continue

				# evaluate losslock & daily limits
				evaluate_losslock(state)
				evaluate_daily_limits(state)
				if state.get("daily_paused", False):
					time.sleep(POLL_SEC)
					continue

				# adaptive token day-policy
				if (netR_today <= -1.5) or (losses_today >= 2):
					globals()["TOKEN_BUCKET_CAPACITY"] = 1
					globals()["TOKEN_BUCKET_REFILL_SEC"] = 60 * 60
				else:
					globals()["TOKEN_BUCKET_CAPACITY"] = 3 if TEST_MODE else 2
					globals()["TOKEN_BUCKET_REFILL_SEC"] = 20 * 60 if TEST_MODE else 30 * 60

				# meta-rate override
				try:
					apply_meta_rate_policy(state)
				except Exception as e:
					if DEBUG:
						print(f"meta-rate error: {e}")

				refill_tokens_if_needed(state)
				if state["token_bucket"] <= 0:
					idle_since_min = (jh_now().timestamp() - state.get("no_trade_since", jh_now().timestamp())) / 60.0
					if idle_since_min >= 3:
						state["token_bucket"] = 1

				# fetch market data
				df = get_rates(SYMBOL, TIMEFRAME, MIN_CANDLE_HISTORY)
				atr_series = compute_atr(df)
				body_series = candle_body(df)

				# startup clamp tracker for closed bars
				try:
					bar_time_str = str(df["time"].iloc[-2])
				except Exception:
					bar_time_str = ""
				if bar_time_str and bar_time_str != state.get("last_bar_time_str", ""):
					state["last_bar_time_str"] = bar_time_str
					if state.get("startup_active", True) and state.get("startup_new_bars_left", 0) > 0:
						state["startup_new_bars_left"] -= 1
						if state["startup_new_bars_left"] <= 0:
							state["startup_active"] = False

				# teacher advice (nudges)
				eff_mult, nud_atr, nud_body, strat_wr = teacher.advise()

				# dynamic percentiles
				thresholds_obj = Thresholds()
				atr_pctl, body_pctl = thresholds_obj.auto_adjust(state)
				atr_pctl  = int(np.clip(atr_pctl  + nud_atr,  AUTO_TIGHTEN_MIN, AUTO_TIGHTEN_MAX))
				body_pctl = int(np.clip(body_pctl + nud_body, AUTO_TIGHTEN_MIN, AUTO_TIGHTEN_MAX))

				# outcome-weighted pull toward historically better bins
				if OUTCOME_WEIGHTED_ENABLE:
					try:
						df_hist = read_trades_df()
						t_atr, t_body = outcome_weighted_percentile_targets(df_hist)
						if t_atr is not None and t_body is not None:
							atr_pctl  = int(round((1.0 - OUTCOME_WEIGHTED_BLEND) * atr_pctl  + OUTCOME_WEIGHTED_BLEND * t_atr))
							body_pctl = int(round((1.0 - OUTCOME_WEIGHTED_BLEND) * body_pctl + OUTCOME_WEIGHTED_BLEND * t_body))
					except Exception as e:
						if DEBUG:
							print(f"⚠️ outcome-weighted calc error: {e}")

				# idle relax or tighten-on-bad
				if (netR_today > -1.5 and losses_today < 2):
					idle_min = (jh_now().timestamp() - state["no_trade_since"]) / 60.0
					if idle_min >= 60:
						atr_pctl  = max(10, atr_pctl - 6)
						body_pctl = max(10, body_pctl - 6)
				else:
					atr_pctl  = min(45, atr_pctl + 10)
					body_pctl = min(45, body_pctl + 10)

				atr_th  = percentile_threshold(atr_series.tail(ATR_LOOKBACK_CANDLES),  atr_pctl)  or float(atr_series.iloc[-1])
				body_th = percentile_threshold(body_series.tail(ATR_LOOKBACK_CANDLES), body_pctl) or float(body_series.iloc[-1])
				atr_th *= eff_mult
				body_th *= eff_mult

				if DEBUG:
					debug_gate_snapshot(state, atr_pctl, body_pctl, atr_th, body_th)

				# (main loop continues in next block…)
				# save interim state, print heartbeat
				now_ts = time.time()
				if DEBUG and now_ts - last_hb >= HEARTBEAT_EVERY_SEC:
					heartbeat(df, atr_series, body_series, atr_th, body_th, state, eff_mult)
					last_hb = now_ts

				# Persist state regularly
				save_state(state)

			except KeyboardInterrupt:
				print("User abort.")
				break
			except Exception as e:
				print(f"⚠️ Loop error: {e}")
				time.sleep(POLL_SEC)

			time.sleep(POLL_SEC)
	finally:
		shutdown_mt5()
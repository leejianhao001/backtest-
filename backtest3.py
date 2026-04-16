"""
backtest.py — Polymarket BTC Up/Down Bot  (v16 — regime-aware strategy)
=======================================================================
Run with:
    pip install aiohttp pandas numpy scipy
    python backtest.py

HOW TO CLEAR CACHE
──────────────────────────────────────────────────────────────
  rm -f candles_15m.csv && python backtest.py

HOW TO SET DATE RANGE
──────────────────────────────────────────────────────────────
  Edit START_DATE and END_DATE at the top of this file.
  Set both to None for the most recent 90 days.

THE CORE INSIGHT
──────────────────────────────────────────────────────────────
v15 proved that VWAP + ORB works in trending markets but fails
in sideways (2023: -15.5%) and extreme crash (2021: -28.8%)
markets. Instead of using one strategy for all conditions, v16
automatically detects the market regime and applies the best
strategy for that specific environment.

HOW REGIME DETECTION WORKS
──────────────────────────────────────────────────────────────
Every candle, the bot looks at three things:

  1. ADX (Average Directional Index) — measures trend STRENGTH.
     Not direction, just how strongly price is moving in any
     direction. ADX > 25 = trending. ADX < 20 = sideways.

  2. 50-candle price change — measures overall direction.
     > +3% over 50 candles = bullish trend
     < -3% over 50 candles = bearish trend
     Between -3% and +3% = sideways

  3. Volatility ratio — compares recent ATR to longer-term ATR.
     High vol ratio = markets are moving more than usual = trending
     Low vol ratio = quiet market = sideways

These three combine into one regime label per candle:
  TRENDING_BULL  — ADX high + price rising
  TRENDING_BEAR  — ADX high + price falling
  SIDEWAYS       — ADX low + price flat
  CRASH          — ADX very high + price falling very fast

THE THREE STRATEGIES
──────────────────────────────────────────────────────────────
STRATEGY 1 — TRENDING (bull or bear): VWAP + ORB
  Kept exactly from v15. Proven 57-71% win rate.
  Price deviates from VWAP AND session open confirms direction.
  Both signals must agree. Full Kelly.

STRATEGY 2 — SIDEWAYS: Pure VWAP Mean Reversion
  In sideways markets, ORB rarely fires cleanly.
  Instead: price deviates from VWAP by threshold + RSI confirms
  overextension (RSI > 65 to fade, RSI < 35 to buy).
  No ORB required — just VWAP + RSI agreement.
  Half Kelly (more uncertain, but still tradeable).

STRATEGY 3 — CRASH: Momentum Following (short bias)
  During crashes, mean reversion kills you — price keeps going
  down. Instead follow the momentum:
  - Session open breaks DOWN through previous day's low → SHORT
  - Volume spike confirms the move (3× average)
  - RSI must be below 45 (already oversold direction confirmed)
  - Never go LONG during a crash regime (too dangerous)
  Quarter Kelly (high risk environment, small bets only).

RISK
──────────────────────────────────────────────────────────────
  80% DD → permanent stop for this session.
  Only put in money you are okay losing 80% of.
"""

import asyncio
import aiohttp
import time
import math
import csv
import os
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from dataclasses import dataclass, field
from itertools import product


# ══════════════════════════════════════════════════════════════
# DATE RANGE
# ══════════════════════════════════════════════════════════════

START_DATE = "2023-01-10"
END_DATE   = "2023-04-10"
# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════

TIMEFRAMES = {
    "15m": {
        "interval":    "15m",
        "cache_file":  "candles_15m.csv",
        "label":       "15-minute markets",
        "trades_file": "trades_15m.csv",
        "equity_file": "equity_15m.csv",
        "mc_file":     "mc_15m.csv",
        "grid_file":   "grid_15m.csv",
    },
}

CONFIG = {
    # ── Data ──────────────────────────────────────────────────
    "symbol":              "BTCUSDT",
    "cache_max_age_hours": 24,

    # ── Regime detection ──────────────────────────────────────
    # ADX period — measures trend strength (not direction)
    "adx_period":          14,
    # ADX thresholds
    "adx_trending":        25,    # above = trending market
    "adx_crash":           40,    # above + fast drop = crash
    # Price change over this many candles to determine direction
    "regime_lookback":     50,    # ~12.5 hours on 15m
    # % change thresholds for direction
    "regime_bull_thresh":  0.03,  # +3% = bullish trend
    "regime_bear_thresh": -0.03,  # -3% = bearish trend
    "regime_crash_thresh":-0.08,  # -8% = crash (fast drop)

    # ── Strategy 1: VWAP + ORB (trending markets) ─────────────
    "vwap_dev_threshold":  0.003,
    "vwap_min_candles":    2,
    "orb_session_hours":   [8, 13, 20],

    # ── Strategy 2: BB + VWAP + RSI (sideways markets) ───────
    "sideways_bb_period":  20,     # Bollinger Band period
    "sideways_bb_std":     2.0,    # standard deviations for BB
    "sideways_vwap_dev":   0.001,  # VWAP deviation threshold
    "sideways_rsi_high":   60,     # RSI above = overbought → short
    "sideways_rsi_low":    40,     # RSI below = oversold → long
    "sideways_rsi_period": 14,
    # Minimum ATR as % of price — skip if market too quiet
    "sideways_min_atr_pct": 0.002,
    "sideways_atr_period":  14,
    # VWAP slope filter
    "sideways_vwap_slope_candles": 50,
    "sideways_vwap_slope_thresh":  0.0,

    # ── Strategy 3: Slow bleed (gradual downtrend) ────────────
    # New regime: ADX moderate + VWAP declining + price falling
    # Different from CRASH (fast) — this is a slow grind down
    # Only SHORT trades, momentum following
    "bleed_adx_min":       18,    # moderate ADX — some trend
    "bleed_adx_max":       35,    # not a full crash
    "bleed_vwap_slope_candles": 50,
    "bleed_vwap_slope_thresh": -0.005,  # VWAP must be declining
    "bleed_rsi_max":       48,    # RSI must be bearish
    "bleed_body_ratio_min": 0.5,  # candle body must be bearish

    # ── Strategy 4: Momentum short (crash markets) ────────────
    "crash_vol_mult":      2.5,   # volume must be 2.5× average
    "crash_vol_window":    20,
    "crash_rsi_max":       45,
    "crash_rsi_period":    14,
    "crash_body_ratio_min": 0.6,  # strong bearish candle required
    "crash_prev_low_lookback": 96,

    # ── Grid search ───────────────────────────────────────────
    "grid_vwap_dev":       [0.001, 0.002, 0.003, 0.005],
    "grid_sideways_dev":   [0.001, 0.002, 0.003],
    "grid_min_atr_pct":    [0.002, 0.003, 0.004],
    "grid_adx_trending":   [20, 25, 30],
    "grid_min_trades":     8,

    # ── Bankroll & sizing ─────────────────────────────────────
    "bankroll":         100.0,
    "kelly_fraction":   0.5,
    "max_position_pct": 0.05,
    "min_bet_usdc":     1.0,
    "min_win_prob":     0.51,

    # ── Realism ───────────────────────────────────────────────
    "polymarket_fee":  0.02,
    "slippage_pct":    0.005,

    # ── DD halt ───────────────────────────────────────────────
    "dd_halt_pct": 0.80,

    # ── Walk-forward ──────────────────────────────────────────
    "train_pct":    0.70,
    "n_wf_windows": 4,

    # ── Monte Carlo ───────────────────────────────────────────
    "mc_simulations": 1000,
    "mc_min_trades":  20,
}


# ══════════════════════════════════════════════════════════════
# DATE RANGE HELPERS
# ══════════════════════════════════════════════════════════════

def get_time_range():
    if END_DATE:
        end_dt = datetime.strptime(END_DATE, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        end_dt = datetime.now(timezone.utc)
    end_ms = int(end_dt.timestamp() * 1000)
    if START_DATE:
        start_dt = datetime.strptime(START_DATE, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        start_dt = datetime.fromtimestamp(
            end_dt.timestamp() - 90 * 24 * 3600, tz=timezone.utc)
    start_ms = int(start_dt.timestamp() * 1000)
    days = (end_dt - start_dt).days
    return start_ms, end_ms, days, start_dt, end_dt


# ══════════════════════════════════════════════════════════════
# DATA FETCHING
# ══════════════════════════════════════════════════════════════

async def fetch_candles(interval: str, cache_file: str,
                        session: aiohttp.ClientSession,
                        start_ms: int, end_ms: int,
                        days: int) -> pd.DataFrame:
    max_age = CONFIG["cache_max_age_hours"] * 3600
    if os.path.exists(cache_file):
        age = time.time() - os.path.getmtime(cache_file)
        if age < max_age:
            df = pd.read_csv(cache_file, parse_dates=["time"])
            if all(c in df.columns for c in ["high","low","open"]):
                print(f"  Cache fresh — {cache_file} ({len(df):,} candles)")
                return df
            print(f"  Cache missing columns — re-downloading...")
        else:
            print(f"  Cache stale — refreshing...")

    mins_pc = int(interval.replace("h","").replace("m","")) * \
              (60 if "h" in interval else 1)
    total_needed = days * 24 * 60 // mins_pc
    print(f"  Fetching {days}d of {interval} ({total_needed:,} candles)...")

    all_raw      = []
    cursor_start = start_ms

    while len(all_raw) < total_needed:
        params = {
            "symbol":    CONFIG["symbol"],
            "interval":  interval,
            "limit":     1000,
            "startTime": cursor_start,
            "endTime":   end_ms,
        }
        async with session.get(
            "https://api.binance.com/api/v3/klines",
            params=params,
            timeout=aiohttp.ClientTimeout(total=15),
        ) as r:
            if r.status != 200:
                print(f"\n  Binance {r.status} — stopping.")
                break
            chunk = await r.json()
        if not chunk:
            break
        all_raw += chunk
        latest = int(chunk[-1][0])
        if latest >= end_ms:
            break
        cursor_start = latest + 1
        print(f"    {len(all_raw):>6}/{total_needed}...", end="\r")
        await asyncio.sleep(0.2)

    seen, deduped = set(), []
    for c in all_raw:
        if c[0] not in seen:
            seen.add(c[0]); deduped.append(c)
    all_raw = sorted(deduped, key=lambda x: x[0])
    print(f"\n  Fetched {len(all_raw):,} candles.")

    df = pd.DataFrame(all_raw, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","qav","trades","tbbav","tbqav","ignore",
    ])
    for col in ["open","high","low","close","volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["time"] = pd.to_datetime(df["open_time"].astype(int), unit="ms", utc=True)
    df = df[["time","open","high","low","close","volume"]].dropna().reset_index(drop=True)
    df.to_csv(cache_file, index=False)
    print(f"  Saved → {cache_file}")
    return df


async def fetch_all_data(start_ms, end_ms, days) -> dict:
    print("\n  Downloading data...\n")
    async with aiohttp.ClientSession() as session:
        df15, = await asyncio.gather(
            fetch_candles("15m", "candles_15m.csv", session, start_ms, end_ms, days),
        )
    return {"15m": df15}


# ══════════════════════════════════════════════════════════════
# REGIME DETECTION
# ══════════════════════════════════════════════════════════════

def compute_adx(highs: np.ndarray, lows: np.ndarray,
                closes: np.ndarray, period: int) -> float:
    """
    Average Directional Index — measures trend STRENGTH not direction.
    ADX > 25 = trending (strong move in any direction)
    ADX < 20 = sideways (choppy, no clear trend)
    ADX > 40 = very strong trend (possible crash or breakout)
    """
    if len(closes) < period * 2 + 1:
        return 0.0

    h = highs[-(period * 2):]
    l = lows[-(period * 2):]
    c = closes[-(period * 2):]

    tr_list, pdm_list, ndm_list = [], [], []
    for j in range(1, len(c)):
        tr  = max(h[j]-l[j], abs(h[j]-c[j-1]), abs(l[j]-c[j-1]))
        pdm = max(h[j]-h[j-1], 0) if h[j]-h[j-1] > l[j-1]-l[j] else 0
        ndm = max(l[j-1]-l[j], 0) if l[j-1]-l[j] > h[j]-h[j-1] else 0
        tr_list.append(tr)
        pdm_list.append(pdm)
        ndm_list.append(ndm)

    tr_arr  = np.array(tr_list[-period:])
    pdm_arr = np.array(pdm_list[-period:])
    ndm_arr = np.array(ndm_list[-period:])

    atr  = tr_arr.mean()
    if atr == 0:
        return 0.0

    pdi  = 100 * pdm_arr.mean() / atr
    ndi  = 100 * ndm_arr.mean() / atr
    dx   = 100 * abs(pdi - ndi) / (pdi + ndi) if (pdi + ndi) > 0 else 0.0
    return round(dx, 1)


def detect_regime(i: int, df: pd.DataFrame, cfg: dict) -> str:
    """
    Detects market regime at candle i.

    Returns one of:
      TRENDING_BULL  — strong uptrend, ADX high, price rising
      TRENDING_BEAR  — strong downtrend, ADX high, price falling
      CRASH          — very fast drop, ADX very high, big price move
      SLOW_BLEED     — gradual decline, moderate ADX, VWAP declining
      SIDEWAYS       — low ADX, price flat, no clear direction

    SLOW_BLEED is the key new addition — catches the 2021-style
    crash where price grinds down over weeks without triggering
    the CRASH threshold on any single candle.
    """
    lb = max(cfg["adx_period"] * 2 + 5, cfg["regime_lookback"] + 5,
             cfg.get("bleed_vwap_slope_candles", 50) + 5)
    if i < lb:
        return "SIDEWAYS"

    sl     = slice(max(0, i - lb), i + 1)
    highs  = df["high"].iloc[sl].values
    lows   = df["low"].iloc[sl].values
    closes = df["close"].iloc[sl].values

    adx = compute_adx(highs, lows, closes, cfg["adx_period"])

    lookback   = cfg["regime_lookback"]
    price_now  = float(closes[-1])
    price_then = float(closes[-lookback]) if len(closes) >= lookback else float(closes[0])
    price_chg  = (price_now - price_then) / price_then if price_then != 0 else 0

    # CRASH: very fast drop with high ADX
    if price_chg <= cfg["regime_crash_thresh"] and adx >= cfg["adx_crash"]:
        return "CRASH"

    # TRENDING: strong directional move
    if adx >= cfg["adx_trending"]:
        if price_chg >= cfg["regime_bull_thresh"]:
            return "TRENDING_BULL"
        elif price_chg <= cfg["regime_bear_thresh"]:
            return "TRENDING_BEAR"

    # SLOW_BLEED: moderate ADX + VWAP declining + price falling
    # This catches gradual downtrends disguised as sideways
    bleed_adx_min = cfg.get("bleed_adx_min", 18)
    bleed_adx_max = cfg.get("bleed_adx_max", 35)
    if bleed_adx_min <= adx <= bleed_adx_max and price_chg < -0.01:
        # Check VWAP slope
        vwap_slope_lb = cfg.get("bleed_vwap_slope_candles", 50)
        if i >= vwap_slope_lb:
            # Compute mini VWAP slope from closes as proxy
            recent_closes   = closes[-vwap_slope_lb:]
            first_half_mean = recent_closes[:vwap_slope_lb//2].mean()
            second_half_mean= recent_closes[vwap_slope_lb//2:].mean()
            slope = (second_half_mean - first_half_mean) / first_half_mean if first_half_mean > 0 else 0
            if slope < cfg.get("bleed_vwap_slope_thresh", -0.005):
                return "SLOW_BLEED"

    return "SIDEWAYS"


def compute_regime_series(df: pd.DataFrame, cfg: dict) -> list:
    """Pre-computes regime for every candle. Used for reporting."""
    regimes = []
    for i in range(len(df)):
        regimes.append(detect_regime(i, df, cfg))
    return regimes


# ══════════════════════════════════════════════════════════════
# VWAP
# ══════════════════════════════════════════════════════════════

def compute_vwap_series(df: pd.DataFrame) -> np.ndarray:
    typical = (df["high"] + df["low"] + df["close"]) / 3
    vol     = df["volume"]
    dates   = df["time"].dt.date
    vwap    = np.zeros(len(df))
    cum_tpv = 0.0
    cum_vol = 0.0
    prev_d  = None
    for i in range(len(df)):
        d = dates.iloc[i]
        if d != prev_d:
            cum_tpv = 0.0
            cum_vol = 0.0
            prev_d  = d
        cum_tpv += typical.iloc[i] * vol.iloc[i]
        cum_vol  += vol.iloc[i]
        vwap[i]  = cum_tpv / cum_vol if cum_vol > 0 else float(typical.iloc[i])
    return vwap


def compute_candles_since_reset(df: pd.DataFrame) -> np.ndarray:
    dates  = df["time"].dt.date
    counts = np.zeros(len(df), dtype=int)
    count  = 0
    prev_d = None
    for i in range(len(df)):
        d = dates.iloc[i]
        if d != prev_d:
            count  = 0
            prev_d = d
        counts[i] = count
        count += 1
    return counts


# ══════════════════════════════════════════════════════════════
# INDICATORS
# ══════════════════════════════════════════════════════════════

def compute_rsi(closes: np.ndarray, period: int) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas   = np.diff(closes[-(period + 1):])
    gains    = np.where(deltas > 0, deltas,  0.0)
    losses   = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = gains.mean()
    avg_loss = losses.mean()
    if avg_loss == 0:
        return 100.0
    return round(100.0 - (100.0 / (1.0 + avg_gain / avg_loss)), 2)


def build_orb_signals(df: pd.DataFrame, session_hours: list) -> dict:
    signals = {}
    hours   = df["time"].dt.hour.values
    highs   = df["high"].values
    lows    = df["low"].values
    closes  = df["close"].values
    for i in range(len(df) - 1):
        if hours[i] in session_hours:
            rh = highs[i]; rl = lows[i]
            if rh == rl:
                continue
            nc = closes[i + 1]
            if nc > rh:
                signals[i + 1] = "up"
            elif nc < rl:
                signals[i + 1] = "down"
    return signals


# ══════════════════════════════════════════════════════════════
# THREE STRATEGIES
# ══════════════════════════════════════════════════════════════

def signal_trending(i: int, df: pd.DataFrame,
                    vwap: np.ndarray, csr: np.ndarray,
                    orb_signals: dict, cfg: dict) -> tuple:
    """
    STRATEGY 1 — TRENDING markets.
    VWAP deviation + ORB both must agree.
    Proven 57-71% win rate in v15.
    Returns (direction, confidence, kelly_mult, label)
    """
    if csr[i] < cfg["vwap_min_candles"]:
        return None, 0.0, 0.0, ""

    price    = float(df["close"].iloc[i])
    open_val = float(df["open"].iloc[i])
    vwap_val = float(vwap[i])
    if vwap_val == 0:
        return None, 0.0, 0.0, ""

    dev      = (price - vwap_val) / vwap_val
    thresh   = cfg["vwap_dev_threshold"]
    candle_d = "up" if price > open_val else ("down" if price < open_val else "neutral")
    orb_sig  = orb_signals.get(i)

    # VWAP reversion direction
    if dev > thresh and candle_d == "down":
        vwap_dir = "down"
    elif dev < -thresh and candle_d == "up":
        vwap_dir = "up"
    else:
        return None, 0.0, 0.0, ""

    # ORB must agree
    if orb_sig != vwap_dir:
        return None, 0.0, 0.0, ""

    dev_str    = min(abs(dev) / thresh / 2.0, 1.0)
    confidence = round(min(dev_str * 0.6 + 0.3, 1.0), 3)
    return vwap_dir, confidence, 1.0, "TREND_VWAP+ORB"


def compute_bb(closes: np.ndarray, period: int, n_std: float):
    """Bollinger Bands: returns (upper, middle, lower)."""
    if len(closes) < period:
        mid = closes[-1]
        return mid, mid, mid
    window = closes[-period:]
    mid    = window.mean()
    std    = window.std()
    return mid + n_std * std, mid, mid - n_std * std


def signal_sideways(i: int, df: pd.DataFrame,
                    vwap: np.ndarray, csr: np.ndarray,
                    cfg: dict) -> tuple:
    """
    STRATEGY 2 — SIDEWAYS: Bollinger Band + VWAP + RSI confluence.

    Three independent signals must agree — this is the key
    improvement over pure VWAP reversion:

    SHORT signal (all three must fire):
      1. Price touches or exceeds upper Bollinger Band (2σ above mean)
      2. Price is above VWAP by threshold (two measures say overextended)
      3. RSI above 60 (momentum confirming overbought)

    LONG signal (all three must fire):
      1. Price touches or falls below lower Bollinger Band
      2. Price is below VWAP by threshold
      3. RSI below 40 (momentum confirming oversold)
      4. VWAP must NOT be declining (slope filter — no longs in downtrend)

    Why Bollinger Bands work in sideways:
    In trending markets, price rides the upper/lower band.
    In sideways markets, price BOUNCES off the bands.
    The BB period (20) captures the current "normal range".
    Touching the band = 2 standard deviations from normal = extreme.
    """
    if csr[i] < cfg["vwap_min_candles"]:
        return None, 0.0, 0.0, ""

    bb_period = cfg.get("sideways_bb_period", 20)
    lb        = max(bb_period + 2, cfg["sideways_rsi_period"] + 2,
                    cfg.get("sideways_atr_period", 14) + 2)

    closes_sl = df["close"].iloc[max(0, i-lb): i+1].values
    highs_sl  = df["high"].iloc[max(0, i-lb): i+1].values
    lows_sl   = df["low"].iloc[max(0, i-lb): i+1].values

    if len(closes_sl) < bb_period:
        return None, 0.0, 0.0, ""

    price    = float(closes_sl[-1])
    open_val = float(df["open"].iloc[i])
    vwap_val = float(vwap[i])
    if vwap_val == 0:
        return None, 0.0, 0.0, ""

    # ── ATR volatility filter ──────────────────────────────
    atr_period = cfg.get("sideways_atr_period", 14)
    true_ranges = []
    for j in range(1, len(closes_sl)):
        tr = max(highs_sl[j]-lows_sl[j],
                 abs(highs_sl[j]-closes_sl[j-1]),
                 abs(lows_sl[j]-closes_sl[j-1]))
        true_ranges.append(tr)
    atr     = np.mean(true_ranges[-atr_period:]) if true_ranges else 0
    atr_pct = atr / vwap_val if vwap_val > 0 else 0
    if atr_pct < cfg.get("sideways_min_atr_pct", 0.002):
        return None, 0.0, 0.0, ""

    # ── Bollinger Bands ────────────────────────────────────
    bb_upper, bb_mid, bb_lower = compute_bb(
        closes_sl, bb_period, cfg.get("sideways_bb_std", 2.0))

    # ── VWAP deviation ────────────────────────────────────
    dev    = (price - vwap_val) / vwap_val
    thresh = cfg["sideways_vwap_dev"]

    # ── RSI ───────────────────────────────────────────────
    rsi = compute_rsi(closes_sl, cfg["sideways_rsi_period"])

    # ── VWAP slope (block longs in downtrend) ─────────────
    slope_lb      = cfg.get("sideways_vwap_slope_candles", 50)
    vwap_prev_idx = max(0, i - slope_lb)
    vwap_prev     = float(vwap[vwap_prev_idx])
    vwap_slope    = (vwap_val - vwap_prev) / vwap_prev if vwap_prev > 0 else 0
    vwap_declining = vwap_slope < cfg.get("sideways_vwap_slope_thresh", 0.0)

    # ── SHORT: BB upper + above VWAP + RSI overbought ─────
    at_upper = price >= bb_upper
    if at_upper and dev > thresh and rsi > cfg["sideways_rsi_high"]:
        bb_str  = min((price - bb_upper) / max(bb_upper - bb_mid, 0.001) + 1, 2.0) / 2
        dev_str = min(abs(dev) / thresh / 2.0, 1.0)
        rsi_str = min((rsi - cfg["sideways_rsi_high"]) / 10.0, 1.0)
        conf    = round(min(bb_str * 0.35 + dev_str * 0.35 + rsi_str * 0.2 + 0.1, 1.0), 3)
        return "down", conf, 0.5, "SIDE_BB+VWAP+RSI"

    # ── LONG: BB lower + below VWAP + RSI oversold ────────
    at_lower = price <= bb_lower
    if at_lower and dev < -thresh and rsi < cfg["sideways_rsi_low"] and not vwap_declining:
        bb_str  = min((bb_lower - price) / max(bb_mid - bb_lower, 0.001) + 1, 2.0) / 2
        dev_str = min(abs(dev) / thresh / 2.0, 1.0)
        rsi_str = min((cfg["sideways_rsi_low"] - rsi) / 10.0, 1.0)
        conf    = round(min(bb_str * 0.35 + dev_str * 0.35 + rsi_str * 0.2 + 0.1, 1.0), 3)
        return "up", conf, 0.5, "SIDE_BB+VWAP+RSI"

    return None, 0.0, 0.0, ""


def signal_crash(i: int, df: pd.DataFrame,
                 cfg: dict) -> tuple:
    """
    STRATEGY 3 — CRASH markets.
    During crashes, mean reversion kills you. Follow momentum DOWN.

    Entry: price breaks below previous session's low with
    volume spike (3× average) + RSI confirms downtrend.
    Only SHORT trades — never long during a crash.
    Quarter Kelly — very high risk environment.
    """
    vol_win = cfg["crash_vol_window"]
    lb      = max(vol_win + 2, cfg["crash_prev_low_lookback"] + 2,
                  cfg["crash_rsi_period"] + 2)
    if i < lb:
        return None, 0.0, 0.0, ""

    closes  = df["close"].iloc[max(0, i-lb): i+1].values
    volumes = df["volume"].iloc[max(0, i-lb): i+1].values
    lows    = df["low"].iloc[max(0, i-lb): i+1].values

    # Volume spike check
    avg_vol  = volumes[-(vol_win+2):-2].mean()
    curr_vol = volumes[-1]
    if avg_vol == 0 or curr_vol < cfg["crash_vol_mult"] * avg_vol:
        return None, 0.0, 0.0, ""

    # RSI must confirm downtrend
    rsi = compute_rsi(closes, cfg["crash_rsi_period"])
    if rsi >= cfg["crash_rsi_max"]:
        return None, 0.0, 0.0, ""

    # Price must break below recent low (momentum confirmation)
    lookback_lows = lows[-cfg["crash_prev_low_lookback"]:-1]
    if len(lookback_lows) == 0:
        return None, 0.0, 0.0, ""
    prev_low  = float(lookback_lows.min())
    curr_close = float(closes[-1])

    if curr_close >= prev_low:
        return None, 0.0, 0.0, ""   # didn't break below — skip

    # Confidence based on how far below previous low
    breakdown_pct = (prev_low - curr_close) / prev_low
    vol_str       = min(curr_vol / avg_vol / cfg["crash_vol_mult"], 2.0) / 2.0
    rsi_str       = min((cfg["crash_rsi_max"] - rsi) / cfg["crash_rsi_max"], 1.0)
    confidence    = round(min(breakdown_pct * 10 + vol_str * 0.3 + rsi_str * 0.3, 1.0), 3)

    return "down", confidence, 0.25, "CRASH_MOMENTUM"


def signal_slow_bleed(i: int, df: pd.DataFrame, cfg: dict) -> tuple:
    """
    STRATEGY 4 — SLOW BLEED (gradual downtrend).

    This catches the 2021-style crash where BTC dropped from
    $58k to $30k over 3 months — not in one day, but grinding
    down slowly. Each individual candle looked like sideways.

    Only SHORT trades. Follow the momentum DOWN.

    Entry conditions:
    1. Current candle is bearish (close < open)
    2. Candle body ratio > 0.5 (strong directional candle,
       not just a doji or spinning top)
       Body ratio = |close-open| / (high-low)
    3. RSI below 48 — confirms bearish momentum
    4. Price making new N-period low (trend continuation)
    5. Volume not collapsing (not an exhaustion move)

    Quarter Kelly — cautious sizing in uncertain regime.
    """
    lb = max(cfg.get("crash_rsi_period", 14) + 2,
             cfg.get("crash_vol_window", 20) + 2,
             cfg.get("crash_prev_low_lookback", 96) + 2)
    if i < lb:
        return None, 0.0, 0.0, ""

    closes  = df["close"].iloc[max(0,i-lb): i+1].values
    highs   = df["high"].iloc[max(0,i-lb): i+1].values
    lows    = df["low"].iloc[max(0,i-lb): i+1].values
    volumes = df["volume"].iloc[max(0,i-lb): i+1].values
    opens   = df["open"].iloc[max(0,i-lb): i+1].values

    price    = float(closes[-1])
    open_val = float(opens[-1])
    high_val = float(highs[-1])
    low_val  = float(lows[-1])

    # Must be a bearish candle
    if price >= open_val:
        return None, 0.0, 0.0, ""

    # Body ratio — how much of the candle range was the body
    candle_range = high_val - low_val
    if candle_range == 0:
        return None, 0.0, 0.0, ""
    body_ratio = abs(price - open_val) / candle_range
    if body_ratio < cfg.get("bleed_body_ratio_min", 0.5):
        return None, 0.0, 0.0, ""   # weak candle — skip

    # RSI must confirm bearish
    rsi = compute_rsi(closes, cfg.get("crash_rsi_period", 14))
    if rsi >= cfg.get("bleed_rsi_max", 48):
        return None, 0.0, 0.0, ""

    # Price must be making new lows
    lookback_lows = lows[-cfg.get("crash_prev_low_lookback", 96):-1]
    if len(lookback_lows) == 0 or price >= float(lookback_lows.min()):
        return None, 0.0, 0.0, ""

    # Volume must not be collapsing (avoid exhaustion shorts)
    vol_win = cfg.get("crash_vol_window", 20)
    avg_vol = volumes[-(vol_win+2):-2].mean()
    curr_vol = volumes[-1]
    if avg_vol > 0 and curr_vol < 0.5 * avg_vol:
        return None, 0.0, 0.0, ""   # volume collapsing = exhaustion

    # Confidence
    body_str = min((body_ratio - 0.5) / 0.5, 1.0)
    rsi_str  = min((cfg.get("bleed_rsi_max", 48) - rsi) / 20.0, 1.0)
    conf     = round(min(body_str * 0.4 + rsi_str * 0.4 + 0.1, 1.0), 3)

    return "down", conf, 0.25, "BLEED_MOMENTUM"


# ══════════════════════════════════════════════════════════════
# MAIN SIGNAL ROUTER
# ══════════════════════════════════════════════════════════════

def compute_signal(i: int, df: pd.DataFrame,
                   vwap: np.ndarray, csr: np.ndarray,
                   orb_signals: dict, regime: str,
                   cfg: dict) -> tuple:
    """
    Routes to the correct strategy based on detected regime.
    Returns (direction, confidence, kelly_mult, strategy_label)
    """
    if regime in ("TRENDING_BULL", "TRENDING_BEAR"):
        return signal_trending(i, df, vwap, csr, orb_signals, cfg)
    elif regime == "CRASH":
        return signal_crash(i, df, cfg)
    elif regime == "SLOW_BLEED":
        return signal_slow_bleed(i, df, cfg)
    else:  # SIDEWAYS
        return signal_sideways(i, df, vwap, csr, cfg)


# ══════════════════════════════════════════════════════════════
# MARKET PRICE
# ══════════════════════════════════════════════════════════════

def simulate_market_price(i: int) -> float:
    rng   = np.random.default_rng(seed=i % (2**31))
    noise = rng.uniform(-0.02, 0.02)
    return round(float(np.clip(0.50 + noise, 0.47, 0.53)), 3)


# ══════════════════════════════════════════════════════════════
# POSITION SIZING
# ══════════════════════════════════════════════════════════════

def kelly_bet(win_prob: float, market_price: float,
              bankroll: float, direction: str,
              kelly_mult: float = 1.0) -> float:
    if win_prob < CONFIG["min_win_prob"]:
        return 0.0
    slipped = min(market_price + CONFIG["slippage_pct"], 0.95)
    p_win   = (1.0 - win_prob) if direction == "down" else win_prob
    if not (0 < slipped < 1):
        return 0.0
    b   = (1.0 / slipped) - 1
    f   = max((p_win * b - (1 - p_win)) / b, 0.0)
    bet = min(CONFIG["kelly_fraction"] * f * bankroll * kelly_mult,
              CONFIG["max_position_pct"] * bankroll)
    return round(max(bet, 0.0), 2)


# ══════════════════════════════════════════════════════════════
# STATISTICS
# ══════════════════════════════════════════════════════════════

def sharpe(returns: list, tpd: float) -> float:
    if len(returns) < 2:
        return 0.0
    arr = np.array(returns)
    std = arr.std()
    if std == 0:
        return 0.0
    return round((arr.mean() / std) * math.sqrt(max(tpd, 0.01) * 252), 2)


def max_drawdown(equity: list) -> float:
    peak, dd = equity[0], 0.0
    for v in equity:
        peak = max(peak, v)
        dd   = max(dd, (peak - v) / peak)
    return round(dd * 100, 2)


def rolling_win_rate(trades: list, window: int = 30) -> list:
    outcomes = [1 if t["outcome"] == "WIN" else 0 for t in trades]
    rates = []
    for i in range(len(outcomes)):
        chunk = outcomes[max(0, i - window + 1): i + 1]
        rates.append(round(sum(chunk) / len(chunk), 3))
    return rates


# ══════════════════════════════════════════════════════════════
# SIMULATION
# ══════════════════════════════════════════════════════════════

@dataclass
class SimResult:
    trades:         list = field(default_factory=list)
    equity:         list = field(default_factory=list)
    trade_returns:  list = field(default_factory=list)
    skipped_signal: int  = 0
    skipped_kelly:  int  = 0
    skipped_halted: int  = 0
    halt_count:     int  = 0
    attempted:      int  = 0

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        return sum(1 for t in self.trades if t["outcome"] == "WIN") / len(self.trades)

    @property
    def total_pnl(self) -> float:
        return (self.equity[-1] - CONFIG["bankroll"]) if self.equity else 0.0


def run_simulation(df: pd.DataFrame, vwap: np.ndarray,
                   csr: np.ndarray, start_idx: int,
                   end_idx: int, cfg: dict) -> SimResult:
    result        = SimResult()
    bankroll      = CONFIG["bankroll"]
    peak_bankroll = bankroll
    result.equity.append(bankroll)
    halt_forever  = False
    orb_signals   = build_orb_signals(df, cfg["orb_session_hours"])
    min_i         = max(start_idx, start_idx + 60)

    for i in range(min_i, end_idx - 1):
        result.attempted += 1

        if bankroll > peak_bankroll:
            peak_bankroll = bankroll

        if halt_forever:
            result.skipped_halted += 1
            continue

        dd_now = (peak_bankroll - bankroll) / peak_bankroll if peak_bankroll > 0 else 0.0
        if dd_now >= cfg.get("dd_halt_pct", 0.80):
            halt_forever = True
            result.halt_count += 1
            result.skipped_halted += 1
            continue

        regime = detect_regime(i, df, cfg)
        direction, confidence, kelly_mult, strat_label = compute_signal(
            i, df, vwap, csr, orb_signals, regime, cfg)

        if direction is None:
            result.skipped_signal += 1
            continue

        win_prob       = 0.52 + confidence * 0.08
        market_price   = simulate_market_price(i)
        entry_bankroll = bankroll

        bet = kelly_bet(win_prob, market_price, bankroll, direction, kelly_mult)
        if bet < CONFIG["min_bet_usdc"]:
            result.skipped_kelly += 1
            continue

        entry_close = float(df["close"].iloc[i])
        exit_close  = float(df["close"].iloc[i + 1])
        won = (exit_close > entry_close) if direction == "up" else \
              (exit_close < entry_close)

        slipped = min(market_price + CONFIG["slippage_pct"], 0.95)
        pnl     = (bet * ((1.0 / slipped) - 1.0) * (1 - CONFIG["polymarket_fee"])
                   if won else -bet)

        bankroll = max(round(bankroll + pnl, 2), 0.0)
        result.equity.append(bankroll)
        result.trade_returns.append(pnl / entry_bankroll)

        result.trades.append({
            "timestamp":    df["time"].iloc[i].strftime("%Y-%m-%d %H:%M UTC"),
            "regime":       regime,
            "strategy":     strat_label,
            "direction":    direction.upper(),
            "confidence":   round(confidence, 3),
            "kelly_mult":   kelly_mult,
            "market_price": market_price,
            "win_prob":     round(win_prob, 3),
            "entry_close":  round(entry_close, 2),
            "exit_close":   round(exit_close, 2),
            "bet_usdc":     round(bet, 2),
            "outcome":      "WIN" if won else "LOSS",
            "pnl":          round(pnl, 2),
            "bankroll":     bankroll,
        })

    return result


# ══════════════════════════════════════════════════════════════
# GRID SEARCH
# ══════════════════════════════════════════════════════════════

def grid_search(df: pd.DataFrame, vwap: np.ndarray,
                csr: np.ndarray, split_idx: int,
                grid_file: str, days_train: float) -> dict:
    combos = list(product(
        CONFIG["grid_vwap_dev"],
        CONFIG["grid_sideways_dev"],
        CONFIG["grid_adx_trending"],
        CONFIG["grid_min_atr_pct"],
    ))
    print(f"  Grid search: {len(combos)} combinations "
          f"(≥{CONFIG['grid_min_trades']} trades required)...")

    rows = []
    for vwap_dev, side_dev, adx_thresh, min_atr in combos:
        cfg = {
            **CONFIG,
            "vwap_dev_threshold":  vwap_dev,
            "sideways_vwap_dev":   side_dev,
            "adx_trending":        adx_thresh,
            "sideways_min_atr_pct": min_atr,
        }
        res = run_simulation(df, vwap, csr, 0, split_idx, cfg)
        n   = len(res.trades)
        if n < CONFIG["grid_min_trades"]:
            continue

        tpd        = n / max(days_train, 1)
        raw_sharpe = sharpe(res.trade_returns, tpd)
        score      = raw_sharpe * math.sqrt(n / 100.0)

        rows.append({
            "vwap_dev":   vwap_dev,
            "side_dev":   side_dev,
            "adx_thresh": adx_thresh,
            "min_atr":    min_atr,
            "trades":     n,
            "win_rate":   round(res.win_rate * 100, 1),
            "pnl":        round(res.total_pnl, 2),
            "sharpe":     raw_sharpe,
            "score":      round(score, 3),
        })

    if not rows:
        print(f"  ⚠️  No combo reached {CONFIG['grid_min_trades']} trades.\n")
        return CONFIG

    rdf  = pd.DataFrame(rows).sort_values("score", ascending=False)
    rdf.to_csv(grid_file, index=False)
    best = rdf.iloc[0].to_dict()

    print(f"  Best: vwap_dev={best['vwap_dev']*100:.2f}%  "
          f"side_dev={best['side_dev']*100:.2f}%  "
          f"adx_thresh={int(best['adx_thresh'])}")
    print(f"  Trades={int(best['trades'])}  WR={best['win_rate']}%  "
          f"score={best['score']}")
    print(f"  Grid saved → {grid_file}")

    return {
        **CONFIG,
        "vwap_dev_threshold":  best["vwap_dev"],
        "sideways_vwap_dev":   best["side_dev"],
        "adx_trending":        int(best["adx_thresh"]),
        "sideways_min_atr_pct": best["min_atr"],
    }


# ══════════════════════════════════════════════════════════════
# ROLLING WALK-FORWARD
# ══════════════════════════════════════════════════════════════

def rolling_walk_forward(df: pd.DataFrame, vwap: np.ndarray,
                         csr: np.ndarray, best_cfg: dict) -> list:
    n          = len(df)
    train_size = int(n * CONFIG["train_pct"])
    test_frac  = (1 - CONFIG["train_pct"]) / CONFIG["n_wf_windows"]
    test_size  = int(n * test_frac)
    windows    = []
    for w in range(CONFIG["n_wf_windows"]):
        ts = train_size + w * test_size
        te = min(ts + test_size, n - 1)
        if ts >= n - 1 or te <= ts:
            break
        res      = run_simulation(df, vwap, csr, ts, te, best_cfg)
        n_trades = len(res.trades)
        start_dt = df["time"].iloc[ts].strftime("%b %d")
        end_dt   = df["time"].iloc[te - 1].strftime("%b %d")
        windows.append({
            "window":   w + 1,
            "period":   f"{start_dt} → {end_dt}",
            "trades":   n_trades,
            "win_rate": res.win_rate,
            "pnl":      res.total_pnl,
            "reliable": n_trades >= 10,
        })
    return windows


def print_rolling_wf(windows: list) -> None:
    W = 62
    print("=" * W)
    print("  ROLLING WALK-FORWARD  (out-of-sample windows)")
    print("=" * W)
    print(f"  {'#':>2}  {'Period':<18} {'Trades':>7} "
          f"{'WinRate':>8} {'P&L':>8}  {'Status'}")
    print(f"  {'-'*2}  {'-'*18} {'-'*7} {'-'*8} {'-'*8}  {'-'*10}")
    profitable = 0
    reliable   = 0
    for w in windows:
        if not w["reliable"]:
            wr_str, pnl_str, status = "  n/a  ", "  n/a  ", "⚠️  <10 trades"
        else:
            wr_str  = f"{w['win_rate']*100:.1f}%"
            pnl_str = f"${w['pnl']:+.2f}"
            reliable += 1
            if w["win_rate"] >= 0.54:
                status = "✅ edge"; profitable += 1
            elif w["win_rate"] >= 0.50:
                status = "➖ marginal"
            else:
                status = "❌ negative"
        print(f"  {w['window']:>2}  {w['period']:<18} {w['trades']:>7} "
              f"{wr_str:>8} {pnl_str:>8}  {status}")
    print()
    if reliable > 0:
        pct = profitable / reliable * 100
        print(f"  {profitable}/{reliable} windows showed edge ({pct:.0f}%)")
        if pct >= 75:
            print("  ✅  Consistent.\n")
        elif pct >= 50:
            print("  ⚠️   Mixed.\n")
        else:
            print("  ❌  Failing.\n")
    else:
        print("  ⚠️  All windows <10 trades.\n")


# ══════════════════════════════════════════════════════════════
# MONTE CARLO
# ══════════════════════════════════════════════════════════════

def monte_carlo(trades: list) -> dict:
    if not trades:
        return {}
    trade_rets = [t["pnl"] / max(t["bankroll"] - t["pnl"], 1.0) for t in trades]
    rng    = np.random.default_rng(seed=42)
    start  = CONFIG["bankroll"]
    finals, dds, ruins = [], [], 0
    for _ in range(CONFIG["mc_simulations"]):
        shuffled = rng.permutation(trade_rets)
        bal, equity, ruined = start, [start], False
        for ret in shuffled:
            bal = max(round(bal * (1.0 + ret), 2), 0.0)
            equity.append(bal)
            if bal < 1.0:
                ruined = True
        finals.append(equity[-1])
        dds.append(max_drawdown(equity))
        if ruined:
            ruins += 1
    arr    = np.array(finals)
    dd_arr = np.array(dds)
    n      = CONFIG["mc_simulations"]
    actual = CONFIG["bankroll"] + sum(t["pnl"] for t in trades)
    mc_med = float(np.median(arr))
    mc_p90 = float(np.percentile(arr, 90))
    mc_p10 = float(np.percentile(arr, 10))
    spread = mc_p90 - mc_p10
    luck   = round(max(min((actual - mc_med) / spread * 100, 100.0), -100.0), 1) \
             if spread > 0 else 0.0
    return {
        "n_trades":      len(trades),
        "too_few":       len(trades) < CONFIG["mc_min_trades"],
        "ruin_pct":      round(ruins / n * 100, 1),
        "median_final":  round(mc_med, 2),
        "p10_final":     round(mc_p10, 2),
        "p90_final":     round(mc_p90, 2),
        "median_dd_pct": round(float(np.median(dd_arr)), 1),
        "kelly_luck":    luck,
        "all_finals":    arr.tolist(),
    }


def print_monte_carlo(mc: dict) -> None:
    if not mc:
        print("  Monte Carlo: no trades.\n")
        return
    start = CONFIG["bankroll"]
    W     = 62
    print("=" * W)
    print("  MONTE CARLO  —  1000 random reshuffles of trade order")
    print("=" * W)
    if mc.get("too_few"):
        print(f"  ⚠️  Only {mc['n_trades']} trades — MC approximate.\n")
    print(f"  Starting bankroll : ${start:.2f}")
    print(f"  Median outcome    : ${mc['median_final']:.2f}  "
          f"({(mc['median_final']-start)/start*100:+.1f}%)")
    print(f"  Best 10%          : ${mc['p90_final']:.2f}+  ← lucky")
    print(f"  Worst 10%         : ${mc['p10_final']:.2f}   ← unlucky")
    print(f"  Median drawdown   : {mc['median_dd_pct']}%")
    ruin_note = "✅" if mc["ruin_pct"] < 5 else "⚠️"
    print(f"  Ruin probability  : {mc['ruin_pct']}%  [{ruin_note}]")
    luck = mc["kelly_luck"]
    bar_width = 32
    centre    = bar_width // 2
    pos       = max(0, min(bar_width-1, int((luck+100)/200*bar_width)))
    bar       = ["-"] * bar_width
    bar[centre] = "|"; bar[pos] = "█"
    print(f"  Kelly luck score  : {luck:+.1f}%")
    print(f"  [{''.join(bar)}]")
    print(f"   -100% (unlucky)   0%   +100% (lucky)")
    print()
    if luck > 60:
        ll, ln = "🎰 NOT IDEAL", "Very high luck — won't repeat reliably."
    elif luck > 20:
        ll, ln = "⚠️  MODERATE",  "Some luck — result slightly inflated."
    elif luck >= -20:
        ll, ln = "✅ IDEAL",       "Minimal luck — reflects real performance."
    elif luck >= -60:
        ll, ln = "⚠️  MODERATE",  "Some bad luck — may be better than shown."
    else:
        ll, ln = "💀 NOT IDEAL",  "Very unlucky — don't judge on this alone."
    print(f"  Luck rating       : {ll}")
    print(f"                      {ln}")
    print()
    med_pct = (mc["median_final"] - start) / start * 100
    p10_pct = (mc["p10_final"]    - start) / start * 100
    p90_pct = (mc["p90_final"]    - start) / start * 100
    print(f"  WHAT GOOD NUMBERS LOOK LIKE")
    print(f"  {'─'*56}")
    print(f"  Median >= 0%          : {med_pct:+.1f}%  {'✅' if med_pct >= 0 else '❌'}")
    print(f"  Worst 10% >= -30%     : {p10_pct:+.1f}%  {'✅' if p10_pct >= -30 else '❌'}")
    print(f"  Kelly luck -20 to +20 : {luck:+.1f}%   {'✅' if -20 <= luck <= 20 else '❌'}")
    print(f"  Ruin prob < 5%        : {mc['ruin_pct']}%    {'✅' if mc['ruin_pct'] < 5 else '❌'}")
    print()
    print(f"  ACCOUNT IMPACT (real money estimates)")
    print(f"  {'─'*56}")
    print(f"  {'Amount':<8} {'Median':>10} {'Bad luck':>10} {'Good luck':>10}")
    print(f"  {'─'*8} {'─'*10} {'─'*10} {'─'*10}")
    for amt in [50, 100, 500]:
        print(f"  ${amt:<7} ${amt*(1+med_pct/100):>9.0f} "
              f"${amt*(1+p10_pct/100):>9.0f} "
              f"${amt*(1+p90_pct/100):>9.0f}")
    print()
    print(f"  RESET STRATEGY")
    print(f"  {'─'*56}")
    if med_pct >= 10:
        t, f_, lt = "+30% from start", "Every 2-3 weeks", \
            "⚠️  Not sustainable long term — reset frequently"
    elif med_pct >= 0:
        t, f_, lt = "+20% from start", "Every 3-4 weeks", \
            "❌ Not sustainable long term — short sessions only"
    else:
        t, f_, lt = "Not recommended — median negative", "N/A", \
            "❌ Do not run long term — treat as one-off gamble"
    print(f"  Reset when : {t}")
    print(f"  How often  : {f_}")
    print(f"  Long term  : {lt}")
    print()


# ══════════════════════════════════════════════════════════════
# PRINT RESULT
# ══════════════════════════════════════════════════════════════

def print_result(label: str, result: SimResult, days: float,
                 flag_low: bool = False) -> None:
    trades = result.trades
    if not trades:
        print(f"  [{label}] No trades generated.\n")
        return

    total = len(trades)
    wins  = sum(1 for t in trades if t["outcome"] == "WIN")
    wr    = wins / total * 100
    pnl   = result.total_pnl
    roi   = pnl / CONFIG["bankroll"] * 100
    tpd   = total / max(days, 1)
    sh    = sharpe(result.trade_returns, tpd)
    dd    = max_drawdown(result.equity)

    # Strategy breakdown
    for strat in ["TREND_VWAP+ORB", "SIDE_VWAP+RSI", "CRASH_MOMENTUM"]:
        st = [t for t in trades if t.get("strategy") == strat]
        if st:
            sw = sum(1 for t in st if t["outcome"] == "WIN")
            print(f"") if strat == "TREND_VWAP+ORB" else None

    W = 62
    print("=" * W)
    print(f"  {label}")
    print("=" * W)

    # Regime distribution
    regime_counts = {}
    for t in trades:
        r = t.get("regime", "?")
        regime_counts[r] = regime_counts.get(r, 0) + 1
    regime_str = "  ".join(
        f"{k.replace('TRENDING_BULL','T.BULL').replace('TRENDING_BEAR','T.BEAR').replace('SLOW_BLEED','S.BLEED')}:{v}"
        for k,v in sorted(regime_counts.items()))
    print(f"  Regime mix : {regime_str}")

    low_flag = "  ⚠️  low" if (flag_low and total < CONFIG["mc_min_trades"]) else ""
    print(f"  Trades/day : {tpd:.1f}  (total {total}){low_flag}")

    wr_note = "✅ edge" if wr >= 54 else ("➖" if wr >= 50 else "❌")
    print(f"  Win rate   : {wr:.1f}%  ({wins}W / {total-wins}L)  [{wr_note}]")

    pnl_note = "✅" if pnl > 0 and wr >= 54 else ("⚠️  Kelly luck" if pnl > 0 else "❌")
    print(f"  P&L        : ${pnl:+.2f}  (ROI {roi:+.1f}%)  [{pnl_note}]")

    sh_note = "✅" if sh > 1.0 else ("➖" if sh > 0 else "❌")
    print(f"  Sharpe     : {sh}  [{sh_note}]")

    dd_note = "✅" if dd < 20 else ("➖" if dd < 40 else "❌")
    print(f"  Drawdown   : {dd}%  [{dd_note}]")

    print(f"  Strategy breakdown:")
    for strat, label_s, kelly_s in [
        ("TREND_VWAP+ORB",   "Trending (VWAP+ORB)",    "1.0×"),
        ("SIDE_BB+VWAP+RSI", "Sideways (BB+VWAP+RSI)", "0.5×"),
        ("BLEED_MOMENTUM",   "Slow bleed (Momentum↓)", "0.25×"),
        ("CRASH_MOMENTUM",   "Crash (Momentum↓)",      "0.25×"),
    ]:
        st = [t for t in trades if t.get("strategy") == strat]
        if st:
            sw  = sum(1 for t in st if t["outcome"] == "WIN")
            swr = sw / len(st) * 100
            print(f"    {label_s:<24}: {len(st):>4} trades  WR {swr:.1f}%  ({kelly_s} Kelly)")

    if result.halt_count > 0:
        print(f"  DD halt    : 80% permanent halt triggered")
    pct = result.skipped_signal / max(result.attempted, 1) * 100
    print(f"  Filtered   : {result.skipped_signal:,} ({pct:.0f}%) candles")
    print()


# ══════════════════════════════════════════════════════════════
# CSV EXPORT
# ══════════════════════════════════════════════════════════════

def save_csvs(result: SimResult, mc: dict, tf: dict) -> None:
    if result.trades:
        with open(tf["trades_file"], "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=result.trades[0].keys())
            w.writeheader()
            w.writerows(result.trades)
    rwr = rolling_win_rate(result.trades)
    with open(tf["equity_file"], "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trade_num", "equity", "rolling_win_rate_30"])
        for idx, (eq, rw) in enumerate(zip(result.equity[1:], rwr)):
            w.writerow([idx + 1, eq, rw])
    if mc and mc.get("all_finals"):
        with open(tf["mc_file"], "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sim_num", "final_bankroll", "roi_pct"])
            s = CONFIG["bankroll"]
            for idx, final in enumerate(mc["all_finals"]):
                w.writerow([idx+1, round(final,2), round((final-s)/s*100,1)])
    print(f"  Saved: {tf['trades_file']}  |  "
          f"{tf['equity_file']}  |  {tf['mc_file']}\n")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

async def run_backtest() -> None:
    W = 62
    start_ms, end_ms, days, start_dt, end_dt = get_time_range()

    print(f"\n{'═'*W}")
    print(f"  HOW TO CLEAR CACHE")
    print(f"{'═'*W}")
    print(f"  rm -f candles_15m.csv && python backtest.py")
    print(f"{'═'*W}")
    print(f"\n  HOW TO CHANGE DATE RANGE")
    print(f"{'═'*W}")
    print(f"  Edit START_DATE and END_DATE at the top of this file.")
    print(f"{'═'*W}")
    print(f"\n  Testing: {start_dt.strftime('%b %d %Y')} → "
          f"{end_dt.strftime('%b %d %Y')}  ({days} days)\n")

    data = await fetch_all_data(start_ms, end_ms, days)
    df15 = data["15m"]

    if len(df15) < 100:
        print("Not enough data.\n")
        return

    print("  Computing VWAP and regime series...")
    vwap = compute_vwap_series(df15)
    csr  = compute_candles_since_reset(df15)

    # Show regime distribution for this period
    regime_series = compute_regime_series(df15, CONFIG)
    from collections import Counter
    rc = Counter(regime_series)
    total_c = len(regime_series)
    print(f"  Regime distribution for this period:")
    for regime, count in sorted(rc.items()):
        print(f"    {regime:<20}: {count:>5} candles ({count/total_c*100:.0f}%)")
    print()

    tf   = TIMEFRAMES["15m"]
    days_actual = days

    print(f"\n{'═'*W}")
    print(f"  TIMEFRAME: 15-MINUTE MARKETS")
    print(f"  v16: Auto regime detection → 3 strategies")
    print(f"  Trending → VWAP+ORB | Sideways → VWAP+RSI | Crash → Momentum↓")
    print(f"  Halt: 80% DD → permanent stop")
    print(f"{'═'*W}\n")

    split      = int(len(df15) * CONFIG["train_pct"])
    days_train = days_actual * CONFIG["train_pct"]
    days_test  = days_actual * (1 - CONFIG["train_pct"])

    print("── Grid Search ───────────────────────────────────────\n")
    best_cfg = grid_search(df15, vwap, csr, split, tf["grid_file"], days_train)
    print()

    print("── Walk-Forward: Single Split ────────────────────────\n")
    train = run_simulation(df15, vwap, csr, 0,     split,     best_cfg)
    test  = run_simulation(df15, vwap, csr, split, len(df15), best_cfg)
    print_result(f"In-sample  ({int(days_train)}d)", train, days_train)
    print_result(f"Out-of-sample ({int(days_test)}d)", test, days_test, flag_low=True)

    if train.trades and test.trades and len(test.trades) >= 5:
        gap = train.win_rate - test.win_rate
        if gap > 0.08:
            print(f"  ⚠️  WR drops {gap*100:.1f}pp OOS — some overfit.\n")
        elif test.win_rate >= 0.54:
            print(f"  ✅  WR holds OOS ({test.win_rate*100:.1f}%) — promising!\n")
        else:
            print(f"  ➖  WR borderline OOS ({test.win_rate*100:.1f}%).\n")

    print("── Walk-Forward: Rolling Windows ─────────────────────\n")
    windows = rolling_walk_forward(df15, vwap, csr, best_cfg)
    print_rolling_wf(windows)

    print("── Full Period Baseline ───────────────────────────────\n")
    base = run_simulation(df15, vwap, csr, 0, len(df15), best_cfg)
    print_result(f"Full {days_actual}d  |  15m", base, days_actual)

    print("── Monte Carlo ───────────────────────────────────────\n")
    mc = monte_carlo(base.trades)
    print_monte_carlo(mc)

    save_csvs(base, mc, tf)

    reliable = [w for w in windows if w["reliable"]]
    wf_cons  = (sum(1 for w in reliable if w["win_rate"] >= 0.54)
                / max(len(reliable), 1))

    print(f"\n{'═'*W}")
    print(f"  FINAL RESULT  —  v16 (regime-aware)")
    print(f"  Period: {start_dt.strftime('%b %d %Y')} → {end_dt.strftime('%b %d %Y')}")
    print(f"{'═'*W}\n")
    print(f"  Trades     : {len(base.trades)}  ({len(base.trades)/days_actual:.1f}/day)")
    print(f"  Win rate   : {base.win_rate*100:.1f}%")
    print(f"  P&L        : ${base.total_pnl:+.2f}  "
          f"({base.total_pnl/CONFIG['bankroll']*100:+.1f}%)")
    print(f"  OOS WR     : {test.win_rate*100:.1f}%")
    print(f"  WF consist : {wf_cons*100:.0f}%")
    print(f"  Kelly luck : {mc.get('kelly_luck', 0):+.1f}%")
    print()

    wr  = base.win_rate * 100
    oos = test.win_rate * 100

    if wr >= 54 and oos >= 52 and wf_cons >= 0.75:
        print(f"  ✅  PROMISING — paper trade 2+ weeks before going live.")
    elif wr >= 51 and oos >= 50:
        print(f"  ⚠️   MARGINAL — test more periods.")
    else:
        print(f"  ❌  NO EDGE on this period.")
    print()
    print(f"  ⚠️  REMINDER: Only put in money you are okay losing 80% of.\n")


if __name__ == "__main__":
    asyncio.run(run_backtest())
"""
backtest.py — Polymarket BTC Up/Down Bot  (v12 — genuine edge attempt)
======================================================================
Run with:
    pip install aiohttp pandas numpy scipy
    python backtest.py

WHY EVERY PREVIOUS VERSION FAILED
──────────────────────────────────────────────────────────────
Every version used lagging price indicators (RSI, MACD, EMA,
Bollinger Bands). These are priced into the market by the time
they signal. Win rate was stuck at 46-51% — below break-even.

The fake +1895% results were caused by a bug in simulate_market_price
that nudged YES prices based on momentum, creating artificial profits.
With the bug fixed, the true performance was -81%. Honest but bad.

WHAT v12 DOES DIFFERENTLY
──────────────────────────────────────────────────────────────
Five genuinely different signal sources — none are pure price
derivatives. Each has a structural reason to predict direction:

SIGNAL 1 — Funding Rate (market positioning)
  Binance perpetual futures require traders to pay/receive
  funding every 8 hours. High positive funding = market is
  heavily long = vulnerable to squeeze DOWN. Negative funding
  = heavily short = vulnerable to squeeze UP.
  This is free on Binance and measures REAL positions, not price.
  Fetched from: /fapi/v1/fundingRate endpoint

SIGNAL 2 — Liquidation Cascade Detection
  When leveraged traders get liquidated, their positions are
  force-closed causing sharp price moves. After the cascade
  ends (spike in volume + sharp price move), forced sellers/
  buyers are gone and price tends to MEAN REVERT.
  Detected via: volume spike + price reversal pattern

SIGNAL 3 — Bid-Ask Spread Proxy
  (high - low) / close = normalised candle range.
  Low range = tight spread = market makers confident = TREND continues.
  High range = wide spread = uncertainty = MEAN REVERSION likely.
  This tells you whether to follow momentum or fade it.

SIGNAL 4 — Time of Day Filter
  BTC behaves differently by hour:
  - Asian session (00:00-08:00 UTC): low volume, choppy, avoid
  - London open (08:00-10:00 UTC): directional, good for trend
  - US pre-market (13:00-15:00 UTC): high vol, good for momentum
  - US close (20:00-22:00 UTC): trend exhaustion, mean reversion
  Only trade during statistically higher-probability windows.

SIGNAL 5 — Multi-Timeframe Momentum (4h + 1h + 15m)
  All three timeframes must point the same direction.
  4h = macro trend, 1h = intermediate, 15m = entry timing.
  When all three agree, win rate on that setup is higher because
  you're trading WITH momentum at every relevant timeframe.

HOW TO CLEAR CACHE
──────────────────────────────────────────────────────────────
  rm -f candles_15m.csv candles_1h.csv candles_4h.csv \\
         funding_rate.csv && python backtest.py

WHAT GOOD RESULTS LOOK LIKE
──────────────────────────────────────────────────────────────
  Win rate       ≥ 54%   buffer above 53% break-even after fee
  OOS win rate   ≥ 52%   holds up on data never seen before
  WF consistency ≥ 75%   3 of 4 rolling windows profitable
  Trades/day     ≥ 1     enough for meaningful statistics
  Kelly luck     -20% to +20%  result not driven by luck
"""

import asyncio
import aiohttp
import time
import math
import csv
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from itertools import product
from datetime import datetime, timezone


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
    "lookback_days":       180,
    "cache_max_age_hours": 24,

    # ── Signal 1: Funding rate ─────────────────────────────────
    # Funding rate above this = market too long = DOWN bias
    # Funding rate below this = market too short = UP bias
    # Binance funding is paid every 8h, expressed as % per period
    "funding_bull_thresh": -0.0001,  # below this = short squeeze potential → UP
    "funding_bear_thresh":  0.0001,  # above this = long squeeze potential → DOWN

    # ── Signal 2: Liquidation cascade ─────────────────────────
    # Volume spike multiplier to detect cascade
    # Price reversal threshold after spike
    "liq_vol_spike":     2.0,    # volume must be 2× average
    "liq_reversal_pct":  0.003,  # price must reverse 0.3% after spike

    # ── Signal 3: Bid-ask spread proxy ────────────────────────
    # Candle range / close price
    # Below low_thresh = tight = trend continuation likely
    # Above high_thresh = wide = mean reversion likely
    "spread_low_thresh":  0.003,  # below = tight market, follow trend
    "spread_high_thresh": 0.008,  # above = volatile, fade the move

    # ── Signal 4: Time of day filter ──────────────────────────
    # UTC hours considered high-probability for trading
    # Format: list of (start_hour, end_hour) tuples
    # London open + US session only — skip Asian low-vol hours
    "trade_hours": [
        (6,  13),   # London open + early US overlap
        (13, 19),   # US session full coverage
        (20, 23),   # US afternoon + evening
    ],

    # ── Signal 5: Multi-timeframe momentum ────────────────────
    # All three timeframes must agree on direction
    # EMA period used on each timeframe
    "mtf_ema_period": 20,
    "mtf_slope_min":  0.00003,   # minimum slope to count as trending

    # ── RSI as light confirmation ──────────────────────────────
    "rsi_period":     14,
    "rsi_oversold":   45,
    "rsi_overbought": 55,

    # ── ROC minimum ───────────────────────────────────────────
    "roc_threshold": 0.0003,

    # ── Grid search ───────────────────────────────────────────
    "grid_funding_thresh": [0.00005, 0.0001, 0.0002],
    "grid_liq_vol_spike":  [1.5, 2.0, 2.5],
    "grid_mtf_slope":      [0.00002, 0.00003, 0.00005],
    "grid_min_trades":     20,   # lowered to get more combos qualifying

    # ── Bankroll & sizing ─────────────────────────────────────
    "bankroll":         100.0,
    "kelly_fraction":   0.5,
    "max_position_pct": 0.03,   # reduced 5%→3% to protect bankroll
    "min_bet_usdc":     1.0,
    "min_win_prob":     0.52,

    # ── Realism ───────────────────────────────────────────────
    "polymarket_fee":  0.02,
    "slippage_pct":    0.005,

    # ── Three-tier DD halt ────────────────────────────────────
    "dd_tier1_pct":   0.20,
    "dd_tier1_hours": 1,
    "dd_tier2_pct":   0.50,
    "dd_tier2_hours": 4,
    "dd_tier3_pct":   0.80,

    # ── Walk-forward ──────────────────────────────────────────
    "train_pct":    0.70,
    "n_wf_windows": 4,

    # ── Monte Carlo ───────────────────────────────────────────
    "mc_simulations": 1000,
    "mc_min_trades":  30,
}


# ══════════════════════════════════════════════════════════════
# DATA FETCHING
# ══════════════════════════════════════════════════════════════

async def fetch_candles(interval: str, cache_file: str,
                        session: aiohttp.ClientSession) -> pd.DataFrame:
    """Fetch OHLCV candles. Reuses session for efficiency."""
    max_age = CONFIG["cache_max_age_hours"] * 3600

    if os.path.exists(cache_file):
        age = time.time() - os.path.getmtime(cache_file)
        if age < max_age:
            df = pd.read_csv(cache_file, parse_dates=["time"])
            if "high" in df.columns and "low" in df.columns:
                print(f"  Cache fresh — {cache_file} ({len(df):,} candles)")
                return df
            print(f"  Cache missing columns — re-downloading {cache_file}...")
        else:
            print(f"  Cache stale — refreshing {cache_file}...")

    # Work out candle count needed
    if "h" in interval:
        mins_pc = int(interval.replace("h", "")) * 60
    else:
        mins_pc = int(interval.replace("m", ""))
    total_needed = CONFIG["lookback_days"] * 24 * 60 // mins_pc

    print(f"  Fetching {CONFIG['lookback_days']}d of {interval} "
          f"({total_needed:,} candles)...")

    all_raw  = []
    end_time = int(time.time() * 1000)

    while len(all_raw) < total_needed:
        params = {
            "symbol":   CONFIG["symbol"],
            "interval": interval,
            "limit":    1000,
            "endTime":  end_time,
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
        all_raw  = chunk + all_raw
        end_time = chunk[0][0] - 1
        print(f"    {len(all_raw):>6}/{total_needed}...", end="\r")
        await asyncio.sleep(0.2)

    print(f"\n  Fetched {len(all_raw):,} candles.")

    df = pd.DataFrame(all_raw, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "qav", "trades", "tbbav", "tbqav", "ignore",
    ])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["time"] = pd.to_datetime(df["open_time"].astype(int), unit="ms", utc=True)
    df = df[["time", "open", "high", "low", "close", "volume"]].dropna().reset_index(drop=True)
    df.to_csv(cache_file, index=False)
    print(f"  Saved → {cache_file}")
    return df


async def fetch_funding_rates(session: aiohttp.ClientSession) -> pd.DataFrame:
    """
    Fetches BTC perpetual futures funding rate history from Binance.
    Funding rate is paid every 8 hours.
    Positive = longs pay shorts (market is net long / bullish)
    Negative = shorts pay longs (market is net short / bearish)

    Uses /fapi/v1/fundingRate endpoint (futures API).
    """
    cache_file = "funding_rate.csv"
    max_age    = CONFIG["cache_max_age_hours"] * 3600

    if os.path.exists(cache_file):
        age = time.time() - os.path.getmtime(cache_file)
        if age < max_age:
            df = pd.read_csv(cache_file, parse_dates=["time"])
            # Ensure time column is timezone-aware datetime
            if not pd.api.types.is_datetime64_any_dtype(df["time"]):
                df["time"] = pd.to_datetime(df["time"], utc=True)
            elif df["time"].dt.tz is None:
                df["time"] = df["time"].dt.tz_localize("UTC")
            print(f"  Cache fresh — {cache_file} ({len(df):,} funding periods)")
            return df
        print("  Refreshing funding rate data...")

    # Need ~180 days × 3 funding periods per day = ~540 periods
    total_needed = CONFIG["lookback_days"] * 3
    print(f"  Fetching {CONFIG['lookback_days']}d of funding rates "
          f"(~{total_needed} periods)...")

    all_raw  = []
    end_time = int(time.time() * 1000)

    while len(all_raw) < total_needed:
        params = {
            "symbol":  CONFIG["symbol"],
            "limit":   1000,
            "endTime": end_time,
        }
        try:
            async with session.get(
                "https://fapi.binance.com/fapi/v1/fundingRate",
                params=params,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as r:
                if r.status != 200:
                    print(f"\n  Funding rate API {r.status} — using zeros.")
                    break
                chunk = await r.json()
        except Exception:
            print("  Funding rate fetch failed — using zeros.")
            break
        if not chunk:
            break
        all_raw  = chunk + all_raw
        end_time = int(chunk[0]["fundingTime"]) - 1
        print(f"    {len(all_raw):>4}/{total_needed}...", end="\r")
        await asyncio.sleep(0.2)

    print(f"\n  Fetched {len(all_raw)} funding periods.")

    if not all_raw:
        # Return empty df with correct structure — signals will be neutral
        return pd.DataFrame(columns=["time", "funding_rate"])

    df = pd.DataFrame(all_raw)
    df["time"]         = pd.to_datetime(df["fundingTime"].astype(int), unit="ms", utc=True)
    df["funding_rate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    df = df[["time", "funding_rate"]].dropna().sort_values("time").reset_index(drop=True)
    df.to_csv(cache_file, index=False)
    print(f"  Saved → {cache_file}")
    return df


async def fetch_all_data() -> dict:
    """Fetches all required data in one async session."""
    print("\n  Downloading all data sources...\n")
    async with aiohttp.ClientSession() as session:
        df15, df1h, df4h, df_funding = await asyncio.gather(
            fetch_candles("15m", "candles_15m.csv", session),
            fetch_candles("1h",  "candles_1h.csv",  session),
            fetch_candles("4h",  "candles_4h.csv",  session),
            fetch_funding_rates(session),
        )
    return {
        "15m":     df15,
        "1h":      df1h,
        "4h":      df4h,
        "funding": df_funding,
    }


# ══════════════════════════════════════════════════════════════
# SIGNAL HELPERS
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


def get_funding_signal(candle_time: pd.Timestamp,
                       df_funding: pd.DataFrame, cfg: dict) -> str:
    """
    SIGNAL 1 — Funding rate direction.

    Finds the most recent funding rate before the candle time.
    High positive → market overleveraged long → DOWN signal
    High negative → market overleveraged short → UP signal
    Near zero → balanced → NEUTRAL
    """
    if df_funding.empty:
        return "neutral"

    mask = df_funding["time"] <= candle_time
    if not mask.any():
        return "neutral"

    rate = float(df_funding.loc[mask, "funding_rate"].iloc[-1])

    bull_thresh = cfg.get("funding_bull_thresh", -0.0001)
    bear_thresh = cfg.get("funding_bear_thresh",  0.0001)

    if rate < bull_thresh:
        return "up"      # market too short → squeeze up likely
    elif rate > bear_thresh:
        return "down"    # market too long → squeeze down likely
    return "neutral"


def get_liquidation_signal(closes: np.ndarray, volumes: np.ndarray,
                            cfg: dict) -> str:
    """
    SIGNAL 2 — Liquidation cascade detection.

    Detects: sudden volume spike (2× average) combined with a
    sharp price move. After the cascade, price tends to reverse
    as forced sellers/buyers are exhausted.

    Returns direction of the REVERSAL (opposite of the spike move).
    """
    liq_spike = cfg.get("liq_vol_spike", 2.0)
    rev_pct   = cfg.get("liq_reversal_pct", 0.003)
    vol_win   = 20

    if len(closes) < vol_win + 3 or len(volumes) < vol_win + 3:
        return "neutral"

    avg_vol     = volumes[-(vol_win + 2):-2].mean()
    spike_vol   = volumes[-2]   # previous candle
    current_vol = volumes[-1]

    if avg_vol == 0 or spike_vol < liq_spike * avg_vol:
        return "neutral"   # no volume spike

    # Was there a sharp price move on the spike candle?
    spike_move = (closes[-2] - closes[-3]) / closes[-3] if closes[-3] != 0 else 0
    if abs(spike_move) < rev_pct:
        return "neutral"   # move not sharp enough

    # Current candle shows reversal (volume cooling, price turning)
    if current_vol < spike_vol and spike_move > 0:
        return "down"   # spike was up → reversal is down
    elif current_vol < spike_vol and spike_move < 0:
        return "up"     # spike was down → reversal is up
    return "neutral"


def get_spread_proxy_signal(highs: np.ndarray, lows: np.ndarray,
                             closes: np.ndarray, direction: str,
                             cfg: dict) -> str:
    """
    SIGNAL 3 — Bid-ask spread proxy.

    Normalised range = (high - low) / close
    Low range  → tight spread → trend CONTINUATION more likely
    High range → wide spread  → MEAN REVERSION more likely

    Returns whether the spread proxy CONFIRMS or OPPOSES direction.
    """
    if len(closes) < 2:
        return "neutral"

    norm_range = (float(highs[-1]) - float(lows[-1])) / float(closes[-1])
    low_t  = cfg.get("spread_low_thresh",  0.003)
    high_t = cfg.get("spread_high_thresh", 0.008)

    roc = (closes[-1] - closes[-2]) / closes[-2] if closes[-2] != 0 else 0

    if norm_range < low_t:
        # Tight spread — follow the momentum direction
        return "up" if roc > 0 else "down"
    elif norm_range > high_t:
        # Wide spread — fade the move (mean reversion)
        return "down" if roc > 0 else "up"
    return "neutral"


def is_good_trading_hour(candle_time: pd.Timestamp, cfg: dict) -> bool:
    """
    SIGNAL 4 — Time of day filter.

    Only trade during historically higher-probability UTC hours:
      07:00-12:00  London open + early US overlap
      13:00-17:00  US pre-market and open
      20:00-22:00  US afternoon momentum window

    Skip Asian session (22:00-07:00 UTC) — low volume, choppy,
    signals fire but don't follow through.
    """
    hour = candle_time.hour
    for start, end in cfg.get("trade_hours", [(0, 24)]):
        if start <= hour < end:
            return True
    return False


def get_mtf_direction(closes_4h: np.ndarray, closes_1h: np.ndarray,
                       closes_15m: np.ndarray, cfg: dict) -> str:
    """
    SIGNAL 5 — Multi-timeframe momentum agreement.

    Calculates EMA slope on each of the three timeframes.
    All three must slope in the same direction above min threshold.

    Returns "up", "down", or "neutral" (if any disagree or flat).
    """
    period   = cfg.get("mtf_ema_period", 20)
    slope_min = cfg.get("mtf_slope_min", 0.00003)

    def ema_slope(closes: np.ndarray) -> float:
        if len(closes) < period + 6:
            return 0.0
        s       = pd.Series(closes)
        ema     = s.ewm(span=period, adjust=False).mean()
        current = ema.iloc[-1]
        prev    = ema.iloc[-5]
        return (current - prev) / prev if prev != 0 else 0.0

    s4h  = ema_slope(closes_4h)
    s1h  = ema_slope(closes_1h)
    s15m = ema_slope(closes_15m)

    all_up   = s4h > slope_min  and s1h > slope_min  and s15m > slope_min
    all_down = s4h < -slope_min and s1h < -slope_min and s15m < -slope_min

    if all_up:
        return "up"
    elif all_down:
        return "down"
    return "neutral"


# ══════════════════════════════════════════════════════════════
# MAIN SIGNAL AGGREGATOR
# ══════════════════════════════════════════════════════════════

def compute_signal(
    closes_15m:  np.ndarray,
    highs_15m:   np.ndarray,
    lows_15m:    np.ndarray,
    volumes_15m: np.ndarray,
    closes_1h:   np.ndarray,
    closes_4h:   np.ndarray,
    candle_time: pd.Timestamp,
    df_funding:  pd.DataFrame,
    cfg:         dict,
) -> tuple:
    """
    Returns (direction, confidence) or (None, 0.0).

    Voting system — 5 signals:
      S1 Funding rate    — must not oppose direction
      S2 Liquidation     — optional confirmation (+confidence)
      S3 Spread proxy    — must confirm direction
      S4 Time filter     — must be in good trading window
      S5 MTF momentum    — must agree on direction (hard gate)

    Plus RSI + ROC as basic sanity filters.

    Minimum to trade: S4 (time) + S5 (MTF) + S3 (spread) must pass.
    S1 (funding) must not actively oppose.
    S2 (liquidation) adds confidence bonus when it agrees.
    """
    # ── S4: Time filter (cheapest — check first) ────────────
    if not is_good_trading_hour(candle_time, cfg):
        return None, 0.0

    # ── Minimum data check ──────────────────────────────────
    min_len = max(cfg["mtf_ema_period"] + 10, cfg["rsi_period"] + 2, 25)
    if len(closes_15m) < min_len:
        return None, 0.0

    # ── RSI sanity filter ───────────────────────────────────
    rsi = compute_rsi(closes_15m, cfg["rsi_period"])

    # ── ROC direction ───────────────────────────────────────
    if len(closes_15m) < 2:
        return None, 0.0
    roc = (closes_15m[-1] - closes_15m[-2]) / closes_15m[-2]
    if abs(roc) < cfg["roc_threshold"]:
        return None, 0.0
    roc_dir = "up" if roc > 0 else "down"

    # ── S5: Multi-timeframe momentum (hard gate) ────────────
    mtf_dir = get_mtf_direction(closes_4h, closes_1h, closes_15m, cfg)
    if mtf_dir == "neutral":
        return None, 0.0   # timeframes don't agree — skip

    # ROC must agree with MTF direction
    if roc_dir != mtf_dir:
        return None, 0.0

    direction = mtf_dir

    # RSI must not be extreme in wrong direction
    if direction == "up"   and rsi > cfg["rsi_overbought"]:
        return None, 0.0
    if direction == "down" and rsi < cfg["rsi_oversold"]:
        return None, 0.0

    # ── S3: Spread proxy ────────────────────────────────────
    spread_sig = get_spread_proxy_signal(
        highs_15m, lows_15m, closes_15m, direction, cfg)
    if spread_sig != direction:
        return None, 0.0   # spread proxy disagrees — skip

    # ── S1: Funding rate ────────────────────────────────────
    funding_sig = get_funding_signal(candle_time, df_funding, cfg)
    if funding_sig != "neutral" and funding_sig != direction:
        return None, 0.0   # funding rate actively opposing — skip

    # ── S2: Liquidation cascade (confidence bonus) ──────────
    liq_sig        = get_liquidation_signal(closes_15m, volumes_15m, cfg)
    liq_confirms   = (liq_sig == direction)

    # ── Confidence scoring ──────────────────────────────────
    # Base: how strong is the ROC?
    roc_strength = min(abs(roc) / cfg["roc_threshold"] / 5.0, 1.0)

    # MTF bonus: how steep are the slopes?
    mtf_bonus = 0.2

    # Funding bonus: if funding actively confirms direction
    funding_bonus = 0.15 if funding_sig == direction else 0.0

    # Liquidation bonus: confirmed reversal signal
    liq_bonus = 0.15 if liq_confirms else 0.0

    confidence = round(
        min(roc_strength * 0.5 + mtf_bonus + funding_bonus + liq_bonus, 1.0),
        3)

    return direction, confidence


# ══════════════════════════════════════════════════════════════
# MARKET PRICE SIMULATION (honest — fixed at 0.50)
# ══════════════════════════════════════════════════════════════

def simulate_market_price(roc: float) -> float:
    """
    Fixed at 0.50 ± tiny noise. Real Polymarket BTC markets
    stay near 0.50 — the bug of nudging by ROC is removed.
    """
    rng   = np.random.default_rng(seed=int(abs(roc) * 1e9) % (2 ** 31))
    noise = rng.uniform(-0.02, 0.02)
    return round(float(np.clip(0.50 + noise, 0.47, 0.53)), 3)


# ══════════════════════════════════════════════════════════════
# POSITION SIZING
# ══════════════════════════════════════════════════════════════

def kelly_bet(win_prob_yes: float, market_price: float,
              bankroll: float, direction: str) -> float:
    if win_prob_yes < CONFIG["min_win_prob"]:
        return 0.0
    slipped = min(market_price + CONFIG["slippage_pct"], 0.95)
    p_win   = (1.0 - win_prob_yes) if direction == "down" else win_prob_yes
    if not (0 < slipped < 1):
        return 0.0
    b   = (1.0 / slipped) - 1
    f   = max((p_win * b - (1 - p_win)) / b, 0.0)
    bet = min(CONFIG["kelly_fraction"] * f * bankroll,
              CONFIG["max_position_pct"] * bankroll)
    return round(max(bet, 0.0), 2)


# ══════════════════════════════════════════════════════════════
# STATISTICS
# ══════════════════════════════════════════════════════════════

def sharpe(returns: list, trades_per_day: float) -> float:
    if len(returns) < 2:
        return 0.0
    arr = np.array(returns)
    std = arr.std()
    if std == 0:
        return 0.0
    return round((arr.mean() / std) * math.sqrt(max(trades_per_day, 0.01) * 252), 2)


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
# SIMULATION LOOP
# ══════════════════════════════════════════════════════════════

@dataclass
class SimResult:
    trades:         list  = field(default_factory=list)
    equity:         list  = field(default_factory=list)
    trade_returns:  list  = field(default_factory=list)
    skipped_signal: int   = 0
    skipped_kelly:  int   = 0
    skipped_halted: int   = 0
    halt_t1:        int   = 0
    halt_t2:        int   = 0
    halt_t3:        int   = 0
    attempted:      int   = 0

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        return sum(1 for t in self.trades if t["outcome"] == "WIN") / len(self.trades)

    @property
    def total_pnl(self) -> float:
        return (self.equity[-1] - CONFIG["bankroll"]) if self.equity else 0.0


def run_simulation(df15: pd.DataFrame, df1h: pd.DataFrame,
                   df4h: pd.DataFrame, df_funding: pd.DataFrame,
                   start_idx: int, end_idx: int,
                   cfg: dict, tf_label: str = "") -> SimResult:
    result        = SimResult()
    bankroll      = CONFIG["bankroll"]
    peak_bankroll = bankroll
    result.equity.append(bankroll)

    t1_candles = int(cfg.get("dd_tier1_hours", 1) * 60 / 15)
    t2_candles = int(cfg.get("dd_tier2_hours", 4) * 60 / 15)
    halt_until = -1

    # Pre-index 1h and 4h by timestamp for fast lookup
    h1_times   = df1h["time"].values
    h1_closes  = df1h["close"].values
    h4_times   = df4h["time"].values
    h4_closes  = df4h["close"].values

    min_i = max(start_idx, start_idx + cfg["mtf_ema_period"] + 15)

    for i in range(min_i, end_idx - 1):
        result.attempted += 1

        if bankroll > peak_bankroll:
            peak_bankroll = bankroll

        if halt_until > 0 and i >= halt_until:
            peak_bankroll = bankroll
            halt_until    = -1

        if halt_until > 0 and i < halt_until:
            result.skipped_halted += 1
            continue

        dd_now = (peak_bankroll - bankroll) / peak_bankroll if peak_bankroll > 0 else 0.0

        if dd_now >= cfg.get("dd_tier3_pct", 0.80):
            result.halt_t3 += 1
            result.skipped_halted += 1
            halt_until = end_idx + 1
            continue

        if dd_now >= cfg.get("dd_tier2_pct", 0.50):
            result.halt_t2 += 1
            result.skipped_halted += 1
            halt_until = i + t2_candles
            continue

        if dd_now >= cfg.get("dd_tier1_pct", 0.20):
            result.halt_t1 += 1
            result.skipped_halted += 1
            halt_until = i + t1_candles
            continue

        # Slice 15m lookback
        lb      = 60
        sl      = slice(max(0, i - lb), i + 1)
        closes  = df15["close"].iloc[sl].values
        highs   = df15["high"].iloc[sl].values
        lows    = df15["low"].iloc[sl].values
        volumes = df15["volume"].iloc[sl].values
        roc     = (closes[-1] - closes[-2]) / closes[-2] if len(closes) >= 2 else 0.0

        # Get aligned 1h and 4h closes
        curr_ts  = df15["time"].iloc[i]
        h1_mask  = h1_times <= curr_ts.to_datetime64()
        h4_mask  = h4_times <= curr_ts.to_datetime64()
        cl_1h    = h1_closes[h1_mask][-50:] if h1_mask.sum() > 0 else np.array([])
        cl_4h    = h4_closes[h4_mask][-30:] if h4_mask.sum() > 0 else np.array([])

        direction, confidence = compute_signal(
            closes, highs, lows, volumes,
            cl_1h, cl_4h, curr_ts, df_funding, cfg)

        if direction is None:
            result.skipped_signal += 1
            continue

        win_prob_yes = 0.52 + confidence * 0.08
        market_price = simulate_market_price(roc)
        entry_bankroll = bankroll

        bet = kelly_bet(win_prob_yes, market_price, bankroll, direction)
        if bet < CONFIG["min_bet_usdc"]:
            result.skipped_kelly += 1
            continue

        entry_close = float(df15["close"].iloc[i])
        exit_close  = float(df15["close"].iloc[i + 1])
        won = (exit_close > entry_close) if direction == "up" else \
              (exit_close < entry_close)

        slipped = min(market_price + CONFIG["slippage_pct"], 0.95)
        pnl     = (bet * ((1.0 / slipped) - 1.0) * (1 - CONFIG["polymarket_fee"])
                   if won else -bet)

        bankroll = max(round(bankroll + pnl, 2), 0.0)
        result.equity.append(bankroll)
        result.trade_returns.append(pnl / entry_bankroll)

        result.trades.append({
            "timestamp":    curr_ts.strftime("%Y-%m-%d %H:%M UTC"),
            "direction":    direction.upper(),
            "confidence":   round(confidence, 3),
            "market_price": market_price,
            "win_prob":     round(win_prob_yes, 3),
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

def grid_search(df15: pd.DataFrame, df1h: pd.DataFrame,
                df4h: pd.DataFrame, df_funding: pd.DataFrame,
                split_idx: int, grid_file: str,
                days_train: float) -> dict:
    combos = list(product(
        CONFIG["grid_funding_thresh"],
        CONFIG["grid_liq_vol_spike"],
        CONFIG["grid_mtf_slope"],
    ))
    print(f"  Grid search: {len(combos)} combinations "
          f"(≥{CONFIG['grid_min_trades']} trades required)...")

    rows = []
    for f_thresh, liq_spike, mtf_slope in combos:
        cfg = {
            **CONFIG,
            "funding_bull_thresh": -f_thresh,
            "funding_bear_thresh":  f_thresh,
            "liq_vol_spike":        liq_spike,
            "mtf_slope_min":        mtf_slope,
        }
        res = run_simulation(df15, df1h, df4h, df_funding,
                             0, split_idx, cfg)
        n   = len(res.trades)
        if n < CONFIG["grid_min_trades"]:
            continue

        tpd        = n / max(days_train, 1)
        raw_sharpe = sharpe(res.trade_returns, tpd)
        score      = raw_sharpe * math.sqrt(n / 100.0)

        rows.append({
            "funding_thresh": f_thresh,
            "liq_vol_spike":  liq_spike,
            "mtf_slope_min":  mtf_slope,
            "trades":         n,
            "win_rate_pct":   round(res.win_rate * 100, 1),
            "pnl":            round(res.total_pnl, 2),
            "raw_sharpe":     raw_sharpe,
            "score":          round(score, 3),
        })

    if not rows:
        print(f"  ⚠️  No combo reached {CONFIG['grid_min_trades']} trades.")
        print(f"     The time filter + MTF gate may be very selective.")
        print(f"     Try adding more hours to trade_hours in CONFIG.\n")
        return CONFIG

    rdf  = pd.DataFrame(rows).sort_values("score", ascending=False)
    rdf.to_csv(grid_file, index=False)
    best = rdf.iloc[0].to_dict()

    print(f"  Best: funding_thresh=±{best['funding_thresh']}  "
          f"liq_spike={best['liq_vol_spike']}×  "
          f"mtf_slope={best['mtf_slope_min']}")
    print(f"  Trades={int(best['trades'])}  WR={best['win_rate_pct']}%  "
          f"score={best['score']}")
    print(f"  Grid saved → {grid_file}")

    return {
        **CONFIG,
        "funding_bull_thresh": -best["funding_thresh"],
        "funding_bear_thresh":  best["funding_thresh"],
        "liq_vol_spike":        best["liq_vol_spike"],
        "mtf_slope_min":        best["mtf_slope_min"],
    }


# ══════════════════════════════════════════════════════════════
# ROLLING WALK-FORWARD
# ══════════════════════════════════════════════════════════════

def rolling_walk_forward(df15: pd.DataFrame, df1h: pd.DataFrame,
                         df4h: pd.DataFrame, df_funding: pd.DataFrame,
                         best_cfg: dict) -> list:
    n          = len(df15)
    train_size = int(n * CONFIG["train_pct"])
    test_frac  = (1 - CONFIG["train_pct"]) / CONFIG["n_wf_windows"]
    test_size  = int(n * test_frac)
    windows    = []

    for w in range(CONFIG["n_wf_windows"]):
        ts = train_size + w * test_size
        te = min(ts + test_size, n - 1)
        if ts >= n - 1 or te <= ts:
            break

        res      = run_simulation(df15, df1h, df4h, df_funding,
                                  ts, te, best_cfg)
        n_trades = len(res.trades)
        reliable = n_trades >= 10

        start_dt = df15["time"].iloc[ts].strftime("%b %d")
        end_dt   = df15["time"].iloc[te - 1].strftime("%b %d")

        windows.append({
            "window":   w + 1,
            "period":   f"{start_dt} → {end_dt}",
            "trades":   n_trades,
            "win_rate": res.win_rate,
            "pnl":      res.total_pnl,
            "reliable": reliable,
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
            print("  ✅  Consistent — promising signal.\n")
        elif pct >= 50:
            print("  ⚠️   Some windows fail — keep testing.\n")
        else:
            print("  ❌  Fails most windows — signals need work.\n")
    else:
        print("  ⚠️  All windows <10 trades — need more data or looser filters.\n")


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
        bal      = start
        equity   = [bal]
        ruined   = False
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

    actual_final = CONFIG["bankroll"] + sum(t["pnl"] for t in trades)
    mc_median    = float(np.median(arr))
    mc_p90       = float(np.percentile(arr, 90))
    mc_p10       = float(np.percentile(arr, 10))
    mc_spread    = mc_p90 - mc_p10
    kelly_luck   = round(
        max(min((actual_final - mc_median) / mc_spread * 100, 100.0), -100.0), 1
    ) if mc_spread > 0 else 0.0

    return {
        "n_trades":      len(trades),
        "too_few":       len(trades) < CONFIG["mc_min_trades"],
        "ruin_pct":      round(ruins / n * 100, 1),
        "median_final":  round(mc_median, 2),
        "p10_final":     round(mc_p10, 2),
        "p90_final":     round(mc_p90, 2),
        "median_dd_pct": round(float(np.median(dd_arr)), 1),
        "kelly_luck":    kelly_luck,
        "actual_final":  round(actual_final, 2),
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
    if luck > 60:
        luck_label, luck_note = "🎰 NOT IDEAL", "Very high luck — won't repeat reliably."
    elif luck > 20:
        luck_label, luck_note = "⚠️  MODERATE",  "Some luck — result slightly inflated."
    elif luck >= -20:
        luck_label, luck_note = "✅ IDEAL",       "Minimal luck — reflects real performance."
    elif luck >= -60:
        luck_label, luck_note = "⚠️  MODERATE",  "Some bad luck — may be better than it looks."
    else:
        luck_label, luck_note = "💀 NOT IDEAL",  "Very unlucky — strategy may be underrated."

    print(f"  Kelly luck score  : {luck:+.1f}%  [{luck_label}]")
    print(f"                      {luck_note}")
    print(f"                    → Ideal: -20% to +20%  |  "
          f"Moderate: ±20-60%  |  Not ideal: ±60%+")
    print()


# ══════════════════════════════════════════════════════════════
# PRINT RESULT
# ══════════════════════════════════════════════════════════════

def print_result(label: str, result: SimResult, days: float,
                 flag_low: bool = False) -> None:
    trades = result.trades
    if not trades:
        print(f"  [{label}] No trades generated.")
        print(f"  Time filter or MTF gate may be very selective.")
        print(f"  Try adding hours to trade_hours or lowering mtf_slope_min.\n")
        return

    total = len(trades)
    wins  = sum(1 for t in trades if t["outcome"] == "WIN")
    wr    = wins / total * 100
    pnl   = result.total_pnl
    roi   = pnl / CONFIG["bankroll"] * 100
    tpd   = total / max(days, 1)
    sh    = sharpe(result.trade_returns, tpd)
    dd    = max_drawdown(result.equity)

    W = 62
    print("=" * W)
    print(f"  {label}")
    print("=" * W)

    low_flag = "  ⚠️  low" if (flag_low and total < CONFIG["mc_min_trades"]) else ""
    print(f"  Trades/day : {tpd:.1f}  (total {total}){low_flag}")

    wr_note = "✅ edge" if wr >= 54 else ("➖" if wr >= 50 else "❌ no edge")
    print(f"  Win rate   : {wr:.1f}%  ({wins}W / {total-wins}L)  [{wr_note}]")
    print(f"             → Need >54% for real edge after 2% fee.")

    pnl_note = "✅" if pnl > 0 and wr >= 54 else ("⚠️  Kelly luck" if pnl > 0 else "❌")
    print(f"  P&L        : ${pnl:+.2f}  (ROI {roi:+.1f}%)  [{pnl_note}]")

    sh_note = "✅" if sh > 1.0 else ("➖" if sh > 0 else "❌")
    print(f"  Sharpe     : {sh}  [{sh_note}]")

    dd_note = "✅" if dd < 20 else ("➖" if dd < 40 else "❌")
    print(f"  Drawdown   : {dd}%  [{dd_note}]")

    halts = result.halt_t1 + result.halt_t2 + result.halt_t3
    if halts > 0:
        print(f"  DD halts   : T1(20%)×{result.halt_t1}  "
              f"T2(50%)×{result.halt_t2}  "
              f"T3(80%)×{result.halt_t3}  "
              f"({result.skipped_halted} candles paused)")

    pct_skip = result.skipped_signal / max(result.attempted, 1) * 100
    print(f"  Filtered   : {result.skipped_signal:,} ({pct_skip:.0f}%) candles")
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
                w.writerow([idx + 1, round(final, 2),
                            round((final - s) / s * 100, 1)])

    print(f"  Saved: {tf['trades_file']}  |  "
          f"{tf['equity_file']}  |  {tf['mc_file']}\n")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

async def run_backtest() -> None:
    W = 62
    print(f"\n{'═'*W}")
    print(f"  HOW TO CLEAR CACHE")
    print(f"{'═'*W}")
    print(f"  rm -f candles_15m.csv candles_1h.csv candles_4h.csv \\")
    print(f"         funding_rate.csv && python backtest.py")
    print(f"{'═'*W}\n")

    # Fetch all data sources
    data = await fetch_all_data()
    df15     = data["15m"]
    df1h     = data["1h"]
    df4h     = data["4h"]
    df_fund  = data["funding"]

    if len(df15) < 300:
        print("Not enough 15m data.\n")
        return

    print(f"\n  Data loaded:")
    print(f"    15m candles : {len(df15):,}")
    print(f"    1h  candles : {len(df1h):,}")
    print(f"    4h  candles : {len(df4h):,}")
    print(f"    Funding pts : {len(df_fund):,}")
    print()

    tf  = TIMEFRAMES["15m"]
    days = CONFIG["lookback_days"]

    print(f"\n{'═'*W}")
    print(f"  TIMEFRAME: 15-MINUTE MARKETS")
    print(f"  Signals: Funding + Liquidation + Spread + Time + 4h/1h/15m MTF")
    print(f"  Halts: 20%→1h  |  50%→4h  |  80%→permanent")
    print(f"{'═'*W}\n")

    split      = int(len(df15) * CONFIG["train_pct"])
    days_train = days * CONFIG["train_pct"]
    days_test  = days * (1 - CONFIG["train_pct"])

    print("── Grid Search (in-sample only) ──────────────────────\n")
    best_cfg = grid_search(df15, df1h, df4h, df_fund,
                           split, tf["grid_file"], days_train)
    print()

    print("── Walk-Forward: Single Split ────────────────────────\n")
    train = run_simulation(df15, df1h, df4h, df_fund,
                           0, split, best_cfg, "15m")
    test  = run_simulation(df15, df1h, df4h, df_fund,
                           split, len(df15), best_cfg, "15m")
    print_result(f"In-sample  ({int(days_train)}d)", train, days_train)
    print_result(f"Out-of-sample ({int(days_test)}d)", test, days_test, flag_low=True)

    if train.trades and test.trades and len(test.trades) >= 10:
        gap = train.win_rate - test.win_rate
        if gap > 0.08:
            print(f"  ⚠️  WR drops {gap*100:.1f}pp OOS — some overfit.\n")
        elif test.win_rate >= 0.54:
            print(f"  ✅  WR holds OOS ({test.win_rate*100:.1f}%) — promising!\n")
        else:
            print(f"  ➖  WR borderline OOS ({test.win_rate*100:.1f}%).\n")

    print("── Walk-Forward: Rolling Windows ─────────────────────\n")
    windows = rolling_walk_forward(df15, df1h, df4h, df_fund, best_cfg)
    print_rolling_wf(windows)

    print("── Full Period Baseline ───────────────────────────────\n")
    base = run_simulation(df15, df1h, df4h, df_fund,
                          0, len(df15), best_cfg, "15m")
    print_result(f"Full {days}d  |  15m", base, days)

    print("── Monte Carlo ───────────────────────────────────────\n")
    mc = monte_carlo(base.trades)
    print_monte_carlo(mc)

    save_csvs(base, mc, tf)

    reliable = [w for w in windows if w["reliable"]]
    wf_cons  = (sum(1 for w in reliable if w["win_rate"] >= 0.54)
                / max(len(reliable), 1))

    print(f"\n{'═'*W}")
    print(f"  FINAL RESULT  —  v12 (genuine edge attempt)")
    print(f"{'═'*W}\n")
    print(f"  Trades     : {len(base.trades)}  ({len(base.trades)/days:.1f}/day)")
    print(f"  Win rate   : {base.win_rate*100:.1f}%")
    print(f"  P&L        : ${base.total_pnl:+.2f}  ({base.total_pnl/CONFIG['bankroll']*100:+.1f}%)")
    print(f"  OOS WR     : {test.win_rate*100:.1f}%")
    print(f"  WF consist : {wf_cons*100:.0f}%")
    print(f"  Kelly luck : {mc.get('kelly_luck', 0):+.1f}%")
    print()

    wr  = base.win_rate * 100
    oos = test.win_rate * 100

    if wr >= 54 and oos >= 52 and wf_cons >= 0.75:
        print(f"  ✅  PROMISING — OOS win rate and consistency both pass.")
        print(f"      Paper trade for 2+ weeks before going live.")
    elif wr >= 51 and oos >= 50:
        print(f"  ⚠️   MARGINAL — signals showing some edge but not enough.")
        print(f"      Keep testing on fresh data daily.")
    else:
        print(f"  ❌  NO EDGE — win rate below break-even on fresh data.")
        print(f"      Clear cache and retest. If consistent, signals need rework.")
    print()


if __name__ == "__main__":
    asyncio.run(run_backtest())
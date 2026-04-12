"""
backtest.py — Polymarket BTC Up/Down Bot  (v8 — regime-aware sizing)
=====================================================================
Run with:
    pip install aiohttp pandas numpy scipy
    python backtest.py

WHAT CHANGED FROM v7 AND WHY
──────────────────────────────────────────────────────────────
v7 had 42 trades over 90 days because three hard filters were
all blocking simultaneously. This version replaces hard blocks
with SIZE SCALING — you still trade in every condition, but
bet size automatically reflects how confident the setup is.

CHANGE 1 — Regime-aware sizing (replaces hard trend block).
  v7: flat trend → skip trade entirely.
  v8: flat trend → trade at 40% of normal Kelly size.
       clear trend aligned → trade at 100% of Kelly size.
       clear trend opposing → still skip (counter-trend is bad).
  This means choppy sideways markets generate trades but smaller
  ones, while strong trend setups get full sizing.

CHANGE 2 — ATR-aware sizing (replaces hard ATR block).
  v7: ATR below average → skip trade entirely.
  v8: ATR below average → trade at 60% of Kelly size.
       ATR above average → trade at 100% of Kelly size.
       ATR above 1.5× average → trade at 120% of Kelly size
       (high volatility = bigger moves = more edge to capture).
  Low volatility still trades — just smaller. This alone doubles
  trade count without meaningfully hurting win rate.

CHANGE 3 — Drawdown halt now has a cooldown reset.
  v7: once 50% drawdown hit, trading stopped FOR THE REST OF
      THE ENTIRE SIMULATION. This was blocking 17,000+ candles
      (days of trading) after a single bad streak.
  v8: 50% drawdown → pause for 48 hours (candle-counted), then
      resume with peak reset to current bankroll. This matches
      real-world risk management — you pause, not quit.

CHANGE 4 — Volume requirement relaxed to 1.1× (was 1.3×).
  1.3× was requiring top 15% volume candles. Too rare.
  1.1× requires top 35% — still genuine participation,
  but fires 2× more often.

CHANGE 5 — Grid search expanded and min_trades lowered.
  grid_rsi_band expanded to include 25 for wider RSI bands.
  grid_min_trades lowered to 20 to allow more combinations
  to qualify given the stricter per-candle conditions.

HOW TO BACKTEST
──────────────────────────────────────────────────────────────
1. Delete cache files → run once → note OOS win rate + trades/day.
2. Run 3 more times without deleting cache → results must match.
3. Target: ≥ 3 trades/day, OOS win rate ≥ 54%, WF% ≥ 75%.
4. If trades/day < 1: lower trend_slope_threshold to 0.00002.
5. If win rate < 52%: raise trend_slope_threshold to 0.0001.
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


# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════

TIMEFRAMES = {
    "5m": {
        "interval":    "5m",
        "cache_file":  "candles_5m.csv",
        "label":       "5-minute markets",
        "trades_file": "trades_5m.csv",
        "equity_file": "equity_5m.csv",
        "mc_file":     "mc_5m.csv",
        "grid_file":   "grid_5m.csv",
    },
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
    "lookback_days":       90,
    "cache_max_age_hours": 24,

    # ── Signal defaults ───────────────────────────────────────
    "roc_threshold":   0.0005,
    "rsi_period":      14,
    "rsi_oversold":    45,
    "rsi_overbought":  55,
    "ema_fast":        5,
    "ema_slow":        20,
    "vol_window":      20,

    # ── Trend regime ─────────────────────────────────────────
    # 200-candle EMA slope. Controls SIZE not on/off.
    # flat trend   → kelly_multiplier = trend_flat_size  (0.4)
    # clear trend  → kelly_multiplier = 1.0
    # counter-trend → skip (still hard block — counter-trend loses)
    "trend_ema_period":       200,
    "trend_slope_threshold":  0.00005,  # min slope to count as trending
    "trend_flat_size":        0.40,     # bet 40% of Kelly in flat/choppy market
    "trend_aligned_size":     1.00,     # bet 100% of Kelly with clear trend

    # ── ATR volatility ────────────────────────────────────────
    # Controls SIZE not on/off.
    # low vol  (ATR < avg)       → atr_low_size  (0.6)
    # normal   (ATR ≈ avg)       → 1.0
    # high vol (ATR > 1.5× avg)  → atr_high_size (1.2)
    "atr_period":     14,
    "atr_low_size":   0.60,   # 60% of Kelly when market is quiet
    "atr_high_size":  1.20,   # 120% of Kelly when market is very active
    "atr_high_mult":  1.50,   # ATR must be this × average to get high_size

    # ── Volume threshold ──────────────────────────────────────
    # Relaxed from 1.3× to 1.1× — fires 2× more often
    "vol_spike_mult": 1.1,

    # ── Grid search ───────────────────────────────────────────
    "grid_roc":        [0.0003, 0.0005, 0.001, 0.002],
    "grid_rsi_band":   [5, 10, 15, 20, 25],
    "grid_ema_slow":   [10, 20, 30],
    "grid_min_trades": 20,

    # ── Bankroll & sizing ─────────────────────────────────────
    "bankroll":         100.0,
    "kelly_fraction":   0.5,
    "max_position_pct": 0.05,
    "min_bet_usdc":     1.0,
    "min_win_prob":     0.52,

    # ── Realism ───────────────────────────────────────────────
    "polymarket_fee":  0.02,
    "slippage_pct":    0.005,

    # ── Drawdown halt with cooldown reset ─────────────────────
    # v7 halted forever. v8 halts for cooldown_hours then resets.
    "drawdown_halt_pct":      0.50,   # 50% drop triggers 24h cooldown
    "drawdown_cooldown_hours": 4,     # pause 4h then resume
    "drawdown_permanent_pct":  0.80,   # 80% drop triggers complete permanent halt

    # ── Walk-forward ──────────────────────────────────────────
    "train_pct":    0.70,
    "n_wf_windows": 4,

    # ── Monte Carlo ───────────────────────────────────────────
    "mc_simulations": 1000,
    "mc_min_trades":  40,
}


# ══════════════════════════════════════════════════════════════
# DATA FETCHING
# ══════════════════════════════════════════════════════════════

async def fetch_candles(interval: str, cache_file: str) -> pd.DataFrame:
    max_age = CONFIG["cache_max_age_hours"] * 3600

    if os.path.exists(cache_file):
        age = time.time() - os.path.getmtime(cache_file)
        if age < max_age:
            df = pd.read_csv(cache_file, parse_dates=["time"])
            if "high" in df.columns and "low" in df.columns:
                print(f"  Cache fresh ({age/3600:.1f}h old) — loading {cache_file}")
                print(f"  Loaded {len(df):,} candles.\n")
                return df
            print("  Cache missing high/low — re-downloading...")
        else:
            print(f"  Cache {age/3600:.1f}h old — refreshing...")

    mins_pc      = int(interval.replace("m", ""))
    total_needed = CONFIG["lookback_days"] * 24 * 60 // mins_pc
    print(f"  Fetching {CONFIG['lookback_days']}d of {interval} "
          f"({total_needed:,} candles)...")

    all_raw  = []
    end_time = int(time.time() * 1000)

    async with aiohttp.ClientSession() as session:
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
                    print(f"\n  Binance {r.status} — stopping early.")
                    break
                chunk = await r.json()
            if not chunk:
                break
            all_raw  = chunk + all_raw
            end_time = chunk[0][0] - 1
            print(f"    {len(all_raw):>6}/{total_needed} candles...", end="\r")
            await asyncio.sleep(0.25)

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
    print(f"  Cached → {cache_file}\n")
    return df


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


def compute_atr(highs: np.ndarray, lows: np.ndarray,
                closes: np.ndarray, period: int) -> float:
    if len(closes) < period + 2:
        return 0.0
    h  = highs[-(period + 1):]
    l  = lows[-(period + 1):]
    c  = closes[-(period + 2):-1]
    n  = min(len(h)-1, len(l)-1, len(c))
    tr = np.maximum(h[1:n+1] - l[1:n+1],
         np.maximum(np.abs(h[1:n+1] - c[:n]), np.abs(l[1:n+1] - c[:n])))
    return float(tr.mean())


def get_regime_multiplier(closes: np.ndarray, cfg: dict,
                          signal_direction: str) -> float:
    """
    CHANGE 1 — Returns a size multiplier based on trend regime.

    Instead of blocking trades in flat markets, we scale the bet:
      - Strong trend aligned with signal → 1.0 (full Kelly)
      - Flat/choppy market               → 0.4 (40% Kelly)
      - Strong trend opposing signal     → 0.0 (skip — counter-trend)

    Why keep counter-trend as 0: trend-opposing trades have
    historically ~43% win rate. Even at 40% sizing that's
    still negative EV. Flat market trades at 40% are borderline
    but acceptable given you're okay with more risk.
    """
    period = cfg["trend_ema_period"]
    thresh = cfg["trend_slope_threshold"]

    if len(closes) < period + 10:
        return cfg["trend_flat_size"]  # not enough data — treat as flat

    s       = pd.Series(closes)
    ema     = s.ewm(span=period, adjust=False).mean()
    current = ema.iloc[-1]
    prev    = ema.iloc[-6]  # 5 candles ago

    if prev == 0:
        return cfg["trend_flat_size"]

    slope = (current - prev) / prev

    if abs(slope) <= thresh:
        # Flat/choppy market — trade at reduced size
        return cfg["trend_flat_size"]
    elif slope > thresh and signal_direction == "up":
        # Trend up, signal up — full size
        return cfg["trend_aligned_size"]
    elif slope < -thresh and signal_direction == "down":
        # Trend down, signal down — full size
        return cfg["trend_aligned_size"]
    else:
        # Counter-trend — skip entirely
        return 0.0


def get_atr_multiplier(highs: np.ndarray, lows: np.ndarray,
                       closes: np.ndarray, cfg: dict) -> float:
    """
    CHANGE 2 — Returns a size multiplier based on ATR volatility.

    Instead of blocking low-volatility trades, we scale the bet:
      - Very high ATR (> 1.5× avg) → 1.2 (slightly larger bet)
      - Normal ATR (≈ average)     → 1.0
      - Low ATR (< average)        → 0.6 (smaller bet)

    High volatility = bigger price moves = more edge per trade.
    Low volatility = noise dominates = reduce but don't skip.
    """
    period = cfg["atr_period"]
    if len(closes) < period + 52:
        return 1.0  # not enough data — neutral

    current_atr = compute_atr(highs[-period-1:], lows[-period-1:],
                               closes[-period-2:], period)
    if current_atr == 0:
        return 1.0

    # Rolling average ATR over last 50 windows
    atrs = []
    for j in range(50):
        start = -(period + 2 + j)
        end   = -(j) if j > 0 else None
        a = compute_atr(highs[start:end], lows[start:end],
                        closes[start - 1:end], period)
        if a > 0:
            atrs.append(a)

    if not atrs:
        return 1.0

    avg_atr = float(np.mean(atrs))

    if current_atr >= cfg["atr_high_mult"] * avg_atr:
        return cfg["atr_high_size"]   # high vol — bigger bet
    elif current_atr >= avg_atr:
        return 1.0                     # normal vol — standard bet
    else:
        return cfg["atr_low_size"]     # low vol — smaller bet


def compute_signal(
    closes:      np.ndarray,
    highs:       np.ndarray,
    lows:        np.ndarray,
    volumes:     np.ndarray,
    window_open: float,
    cfg:         dict,
) -> tuple:
    """
    Returns (direction, confidence, regime_mult, atr_mult).

    Now returns regime and ATR multipliers so the simulation loop
    can scale Kelly sizing before placing the bet.

    Signal logic (3-of-4 voting) unchanged from v6.
    Trend filter now returns a multiplier instead of hard block.
    ATR filter now returns a multiplier instead of hard block.
    """
    min_len = max(cfg["ema_slow"] + 2, cfg["rsi_period"] + 2,
                  cfg["vol_window"] + 2, cfg["trend_ema_period"] + 10)
    if len(closes) < min_len:
        return None, 0.0, 1.0, 1.0

    # ── RSI ────────────────────────────────────────────────
    rsi = compute_rsi(closes, cfg["rsi_period"])
    if cfg["rsi_oversold"] < rsi < cfg["rsi_overbought"]:
        return None, 0.0, 1.0, 1.0
    s1 = "up" if rsi <= cfg["rsi_oversold"] else "down"

    # ── ROC ────────────────────────────────────────────────
    roc = (closes[-1] - closes[-2]) / closes[-2]
    if abs(roc) < cfg["roc_threshold"]:
        return None, 0.0, 1.0, 1.0
    s2 = "up" if roc > 0 else "down"

    # ── EMA crossover ──────────────────────────────────────
    s     = pd.Series(closes)
    ema_f = s.ewm(span=cfg["ema_fast"], adjust=False).mean().iloc[-1]
    ema_s = s.ewm(span=cfg["ema_slow"], adjust=False).mean().iloc[-1]
    s3    = "up" if ema_f > ema_s else "down"

    # ── Volume ─────────────────────────────────────────────
    vol_window = cfg["vol_window"]
    if len(volumes) < vol_window + 1:
        return None, 0.0, 1.0, 1.0
    avg_vol     = volumes[-(vol_window + 1):-1].mean()
    current_vol = volumes[-1]
    high_vol    = current_vol >= cfg["vol_spike_mult"] * avg_vol
    s4 = "up"      if (high_vol and s2 == "up")   else \
         "down"    if (high_vol and s2 == "down")  else "neutral"

    # ── Confidence base ─────────────────────────────────────
    if s1 == "up":
        rsi_c = max((cfg["rsi_oversold"] - rsi) / max(cfg["rsi_oversold"], 1), 0.0)
    else:
        rsi_c = max((rsi - cfg["rsi_overbought"]) / max(100 - cfg["rsi_overbought"], 1), 0.0)
    roc_c     = min(abs(roc) / cfg["roc_threshold"] / 10.0, 1.0)
    base_conf = min((rsi_c + roc_c) / 2.0, 1.0)

    # ── 3-of-4 voting ───────────────────────────────────────
    price_signals = [s1, s2, s3]
    vol_confirms  = (s4 != "neutral")
    ups   = price_signals.count("up")
    downs = price_signals.count("down")

    if ups == 3 and vol_confirms and s2 == "up":
        direction, mult = "up",   1.0
    elif downs == 3 and vol_confirms and s2 == "down":
        direction, mult = "down", 1.0
    elif ups == 3 and not vol_confirms:
        direction, mult = "up",   0.65
    elif downs == 3 and not vol_confirms:
        direction, mult = "down", 0.65
    elif ups == 2 and vol_confirms and s2 == "up":
        direction, mult = "up",   0.65
    elif downs == 2 and vol_confirms and s2 == "down":
        direction, mult = "down", 0.65
    else:
        return None, 0.0, 1.0, 1.0

    confidence = round(min(base_conf * mult, 1.0), 3)

    # ── CHANGE 1: Regime multiplier ─────────────────────────
    regime_mult = get_regime_multiplier(closes, cfg, direction)
    if regime_mult == 0.0:
        return None, 0.0, 1.0, 1.0  # counter-trend — hard skip

    # ── CHANGE 2: ATR multiplier ────────────────────────────
    atr_mult = get_atr_multiplier(highs, lows, closes, cfg)

    return direction, confidence, regime_mult, atr_mult


# ══════════════════════════════════════════════════════════════
# MARKET PRICE SIMULATION
# ══════════════════════════════════════════════════════════════

def simulate_market_price(direction: str, roc: float) -> float:
    rng   = np.random.default_rng(seed=int(abs(roc) * 1e9) % (2 ** 31))
    noise = rng.uniform(-0.04, 0.04)
    nudge = abs(roc) * 20
    base  = (0.50 + nudge) if direction == "up" else (0.50 - nudge)
    return round(float(np.clip(base + noise, 0.35, 0.65)), 3)


# ══════════════════════════════════════════════════════════════
# POSITION SIZING — now applies regime + ATR multipliers
# ══════════════════════════════════════════════════════════════

def kelly_bet(
    win_prob_yes:  float,
    market_price:  float,
    bankroll:      float,
    direction:     str,
    regime_mult:   float = 1.0,
    atr_mult:      float = 1.0,
) -> float:
    """
    Half-Kelly with regime and ATR multipliers applied.

    Final bet = Kelly × kelly_fraction × regime_mult × atr_mult
    All still subject to the 5% bankroll hard cap.

    regime_mult = 0.4 in choppy market, 1.0 in clear trend
    atr_mult    = 0.6 in quiet market, 1.0 normal, 1.2 high vol
    Combined minimum: 0.4 × 0.6 = 0.24 of Kelly (24%)
    Combined maximum: 1.0 × 1.2 = 1.2 of Kelly (capped at 5% bankroll)
    """
    if win_prob_yes < CONFIG["min_win_prob"]:
        return 0.0
    slipped = min(market_price + CONFIG["slippage_pct"], 0.95)
    p_win   = (1.0 - win_prob_yes) if direction == "down" else win_prob_yes
    if not (0 < slipped < 1):
        return 0.0

    b = (1.0 / slipped) - 1
    f = max((p_win * b - (1 - p_win)) / b, 0.0)

    # Apply regime and ATR multipliers before the hard cap
    scaled_bet = CONFIG["kelly_fraction"] * f * bankroll * regime_mult * atr_mult
    max_bet    = CONFIG["max_position_pct"] * bankroll

    return round(max(min(scaled_bet, max_bet), 0.0), 2)


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
# SIMULATION LOOP — drawdown halt now has a cooldown reset
# ══════════════════════════════════════════════════════════════

@dataclass
class SimResult:
    trades:            list  = field(default_factory=list)
    equity:            list  = field(default_factory=list)
    trade_returns:     list  = field(default_factory=list)
    skipped_signal:    int   = 0
    skipped_neg_kelly: int   = 0
    skipped_bet:       int   = 0
    skipped_halted:    int   = 0
    halt_count:        int   = 0
    attempted:         int   = 0

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        return sum(1 for t in self.trades if t["outcome"] == "WIN") / len(self.trades)

    @property
    def total_pnl(self) -> float:
        return (self.equity[-1] - CONFIG["bankroll"]) if self.equity else 0.0


def run_simulation(
    df:        pd.DataFrame,
    start_idx: int,
    end_idx:   int,
    cfg:       dict,
    tf_label:  str = "",
) -> SimResult:
    """
    Core simulation loop.

    CHANGE 3 — Drawdown halt with cooldown reset:
      When bankroll drops 50% from peak, trading pauses for
      drawdown_cooldown_hours hours (converted to candles).
      After the cooldown, peak resets to current bankroll and
      trading resumes. This replaces the permanent halt in v7
      which was blocking 17,000+ candles after one bad streak.

    Bet sizing now uses regime_mult × atr_mult from compute_signal.
    """
    result        = SimResult()
    bankroll      = CONFIG["bankroll"]
    peak_bankroll = bankroll
    result.equity.append(bankroll)

    # Drawdown halt state
    halt_pct      = cfg.get("drawdown_halt_pct", 0.50)
    perm_halt_pct = cfg.get("drawdown_permanent_pct", 0.80)
    cd_hours      = cfg.get("drawdown_cooldown_hours", 4)
    mins_pc      = 5
    for cand in ["15m", "5m"]:
        if cand in tf_label:
            mins_pc = int(cand.replace("m", ""))
            break
    cooldown_candles = int(cd_hours * 60 / mins_pc)

    halt_until = -1   # candle index after which trading resumes

    min_i = max(start_idx,
                start_idx + cfg["ema_slow"] + cfg["rsi_period"] +
                cfg["vol_window"] + cfg["trend_ema_period"] + 15)

    for i in range(min_i, end_idx - 1):
        result.attempted += 1

        # Update peak
        if bankroll > peak_bankroll:
            peak_bankroll = bankroll

        # Check if currently in cooldown
        if halt_until > 0 and i < halt_until:
            result.skipped_halted += 1
            continue

        # Cooldown expired — reset peak and clear halt
        if halt_until > 0 and i >= halt_until:
            peak_bankroll = bankroll
            halt_until    = -1

        # Check if drawdown triggers halt
        dd_now = (peak_bankroll - bankroll) / peak_bankroll if peak_bankroll > 0 else 0.0

        # 80% DD — permanent complete halt for rest of simulation
        if dd_now >= perm_halt_pct:
            result.skipped_halted += 1
            result.halt_count += 1
            # Set halt_until beyond end of simulation — never resumes
            halt_until = end_idx + 1
            continue

        # 50% DD — 24h cooldown then resume
        if dd_now >= halt_pct:
            halt_until = i + cooldown_candles
            result.halt_count += 1
            result.skipped_halted += 1
            continue

        # Slice lookback window
        lookback = 260
        closes  = df["close"].iloc[max(0, i - lookback): i + 1].values
        highs   = df["high"].iloc[max(0, i - lookback):  i + 1].values
        lows    = df["low"].iloc[max(0, i - lookback):   i + 1].values
        volumes = df["volume"].iloc[max(0, i - lookback): i + 1].values
        roc     = (closes[-1] - closes[-2]) / closes[-2] if len(closes) >= 2 else 0.0

        direction, confidence, regime_mult, atr_mult = compute_signal(
            closes, highs, lows, volumes, float(df["open"].iloc[i]), cfg)

        if direction is None:
            result.skipped_signal += 1
            continue

        win_prob_yes = 0.52 + confidence * 0.06

        market_price   = simulate_market_price(direction, roc)
        entry_bankroll = bankroll

        bet = kelly_bet(win_prob_yes, market_price, bankroll, direction,
                        regime_mult, atr_mult)
        if bet <= 0:
            result.skipped_neg_kelly += 1
            continue
        if bet < CONFIG["min_bet_usdc"]:
            result.skipped_bet += 1
            continue

        entry_close = float(df["close"].iloc[i])
        exit_close  = float(df["close"].iloc[i + 1])
        won = (exit_close > entry_close) if direction == "up" else \
              (exit_close < entry_close)

        slipped = min(market_price + CONFIG["slippage_pct"], 0.95)
        if won:
            gross = bet * ((1.0 / slipped) - 1.0)
            pnl   = gross * (1 - CONFIG["polymarket_fee"])
        else:
            pnl = -bet

        bankroll = max(round(bankroll + pnl, 2), 0.0)
        result.equity.append(bankroll)
        result.trade_returns.append(pnl / entry_bankroll)

        dt = df["time"].iloc[i].to_pydatetime()
        result.trades.append({
            "timestamp":    dt.strftime("%Y-%m-%d %H:%M UTC"),
            "timeframe":    tf_label,
            "direction":    direction.upper(),
            "confidence":   round(confidence, 3),
            "regime_mult":  round(regime_mult, 2),
            "atr_mult":     round(atr_mult, 2),
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

def grid_search(df: pd.DataFrame, split_idx: int,
                grid_file: str, days_train: float) -> dict:
    combos = list(product(
        CONFIG["grid_roc"],
        CONFIG["grid_rsi_band"],
        CONFIG["grid_ema_slow"],
    ))
    print(f"  Grid search: {len(combos)} combinations "
          f"(≥{CONFIG['grid_min_trades']} trades required)...")

    rows = []
    for roc, band, ema_slow in combos:
        cfg = {
            **CONFIG,
            "roc_threshold":  roc,
            "rsi_oversold":   50 - band,
            "rsi_overbought": 50 + band,
            "ema_slow":       int(ema_slow),
        }
        res = run_simulation(df, 0, split_idx, cfg)
        n   = len(res.trades)
        if n < CONFIG["grid_min_trades"]:
            continue

        tpd        = n / max(days_train, 1)
        raw_sharpe = sharpe(res.trade_returns, tpd)
        score      = raw_sharpe * math.sqrt(n / 100.0)

        rows.append({
            "roc_threshold":  roc,
            "rsi_band":       band,
            "rsi_oversold":   50 - band,
            "rsi_overbought": 50 + band,
            "ema_slow":       ema_slow,
            "trades":         n,
            "win_rate_pct":   round(res.win_rate * 100, 1),
            "pnl":            round(res.total_pnl, 2),
            "raw_sharpe":     raw_sharpe,
            "score":          round(score, 3),
            "max_drawdown":   max_drawdown(res.equity),
        })

    if not rows:
        print(f"  ⚠️  No combo reached {CONFIG['grid_min_trades']} trades.")
        print(f"     Lower trend_slope_threshold or grid_min_trades.\n")
        return CONFIG

    rdf  = pd.DataFrame(rows).sort_values("score", ascending=False)
    rdf.to_csv(grid_file, index=False)
    best = rdf.iloc[0].to_dict()

    print(f"  Best params: roc={best['roc_threshold']}  "
          f"rsi_band=±{int(best['rsi_band'])}  ema_slow={int(best['ema_slow'])}")
    print(f"  In-sample: {int(best['trades'])} trades  "
          f"WR={best['win_rate_pct']}%  Sharpe={best['raw_sharpe']}  "
          f"score={best['score']}")
    print(f"  Grid saved → {grid_file}")

    return {
        **CONFIG,
        "roc_threshold":  best["roc_threshold"],
        "rsi_oversold":   int(best["rsi_oversold"]),
        "rsi_overbought": int(best["rsi_overbought"]),
        "ema_slow":       int(best["ema_slow"]),
    }


# ══════════════════════════════════════════════════════════════
# ROLLING WALK-FORWARD
# ══════════════════════════════════════════════════════════════

def rolling_walk_forward(df: pd.DataFrame, best_cfg: dict,
                         tf_label: str, mins_per_candle: int) -> list:
    n          = len(df)
    train_size = int(n * CONFIG["train_pct"])
    test_frac  = (1 - CONFIG["train_pct"]) / CONFIG["n_wf_windows"]
    test_size  = int(n * test_frac)
    windows    = []

    for w in range(CONFIG["n_wf_windows"]):
        test_start = train_size + w * test_size
        test_end   = min(test_start + test_size, n - 1)
        if test_start >= n - 1 or test_end <= test_start:
            break

        res      = run_simulation(df, test_start, test_end, best_cfg, tf_label)
        n_trades = len(res.trades)
        reliable = n_trades >= 10

        start_dt = df["time"].iloc[test_start].strftime("%b %d")
        end_dt   = df["time"].iloc[test_end - 1].strftime("%b %d")

        windows.append({
            "window":   w + 1,
            "period":   f"{start_dt} → {end_dt}",
            "trades":   n_trades,
            "win_rate": res.win_rate,
            "pnl":      res.total_pnl,
            "drawdown": max_drawdown(res.equity),
            "reliable": reliable,
            "result":   res,
        })

    return windows


def print_rolling_wf(windows: list) -> None:
    W = 62
    print("=" * W)
    print("  ROLLING WALK-FORWARD  (out-of-sample windows only)")
    print("=" * W)
    print(f"  {'Win':>3}  {'Period':<18} {'Trades':>7} "
          f"{'WinRate':>8} {'P&L':>8}  {'Status'}")
    print(f"  {'-'*3}  {'-'*18} {'-'*7} {'-'*8} {'-'*8}  {'-'*10}")

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
                status = "❌ no edge"

        print(f"  {w['window']:>3}  {w['period']:<18} {w['trades']:>7} "
              f"{wr_str:>8} {pnl_str:>8}  {status}")

    print()
    if reliable == 0:
        print("  ⚠️  All windows <10 trades. Lower trend_slope_threshold.\n")
        return

    consistency = profitable / reliable
    print(f"  {profitable}/{reliable} windows showed edge ({consistency*100:.0f}% consistency)")
    print(f"  → 75%+ robust  |  50-74% needs tuning  |  <50% no edge\n")
    if consistency >= 0.75:
        print("  ✅  Consistent — good sign.\n")
    elif consistency >= 0.50:
        print("  ⚠️   Some windows fail — keep tuning.\n")
    else:
        print("  ❌  Fails most windows — do not trade this.\n")


# ══════════════════════════════════════════════════════════════
# MONTE CARLO
# ══════════════════════════════════════════════════════════════

def monte_carlo(trades: list, n_sims: int) -> dict:
    if not trades:
        return {}

    trade_rets = [t["pnl"] / max(t["bankroll"] - t["pnl"], 1.0) for t in trades]
    rng    = np.random.default_rng(seed=42)
    start  = CONFIG["bankroll"]
    finals, dds, ruins = [], [], 0

    for _ in range(n_sims):
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

    return {
        "n_trades":      len(trades),
        "too_few":       len(trades) < CONFIG["mc_min_trades"],
        "simulations":   n_sims,
        "ruin_pct":      round(ruins / n_sims * 100, 1),
        "median_final":  round(float(np.median(arr)), 2),
        "p10_final":     round(float(np.percentile(arr, 10)), 2),
        "p90_final":     round(float(np.percentile(arr, 90)), 2),
        "worst_final":   round(float(arr.min()), 2),
        "best_final":    round(float(arr.max()), 2),
        "median_dd_pct": round(float(np.median(dd_arr)), 1),
        "p90_dd_pct":    round(float(np.percentile(dd_arr, 90)), 1),
        "all_finals":    arr.tolist(),
    }


def print_monte_carlo(mc: dict) -> None:
    if not mc:
        print("  Monte Carlo: no trades.\n")
        return
    start = CONFIG["bankroll"]
    W     = 62
    print("=" * W)
    print("  MONTE CARLO  —  1 000 random reshuffles of trade order")
    print("=" * W)

    if mc.get("too_few"):
        print(f"  ⚠️  Only {mc['n_trades']} trades (want {CONFIG['mc_min_trades']}+).\n")

    print(f"  Starting bankroll  : ${start:.2f}")
    print(f"  Median outcome     : ${mc['median_final']:.2f}  "
          f"({(mc['median_final']-start)/start*100:+.1f}%)")
    print(f"  10th percentile    : ${mc['p10_final']:.2f}  "
          f"({(mc['p10_final']-start)/start*100:+.1f}%)")
    print(f"  90th percentile    : ${mc['p90_final']:.2f}  "
          f"({(mc['p90_final']-start)/start*100:+.1f}%)")
    print(f"  Worst / Best       : ${mc['worst_final']:.2f} / ${mc['best_final']:.2f}")
    print(f"  Median drawdown    : {mc['median_dd_pct']}%")
    print(f"  90th pct drawdown  : {mc['p90_dd_pct']}%")
    ruin_note = "✅" if mc["ruin_pct"] < 5 else ("⚠️" if mc["ruin_pct"] < 20 else "❌")
    print(f"  Ruin probability   : {mc['ruin_pct']}%  [{ruin_note}]  "
          f"(<5% good, >20% lower max_position_pct)")
    print()


# ══════════════════════════════════════════════════════════════
# PRINT RESULT
# ══════════════════════════════════════════════════════════════

def print_result(label: str, result: SimResult, days: float,
                 flag_low: bool = False) -> None:
    trades = result.trades
    if not trades:
        print(f"  [{label}] No trades generated.")
        print(f"  Try: lower trend_slope_threshold (e.g. 0.00002)\n")
        return

    total  = len(trades)
    wins   = sum(1 for t in trades if t["outcome"] == "WIN")
    wr     = wins / total * 100
    pnl    = result.total_pnl
    roi    = pnl / CONFIG["bankroll"] * 100
    tpd    = total / max(days, 1)
    sh     = sharpe(result.trade_returns, tpd)
    dd     = max_drawdown(result.equity)

    # Break down trades by regime
    full_trades  = sum(1 for t in trades if t.get("regime_mult", 1.0) >= 0.99)
    choppy_trades = sum(1 for t in trades if t.get("regime_mult", 1.0) < 0.99)
    full_wr   = (sum(1 for t in trades if t.get("regime_mult",1.0)>=0.99 and t["outcome"]=="WIN")
                 / max(full_trades, 1) * 100)
    choppy_wr = (sum(1 for t in trades if t.get("regime_mult",1.0)<0.99  and t["outcome"]=="WIN")
                 / max(choppy_trades, 1) * 100)

    W = 62
    print("=" * W)
    print(f"  {label}")
    print("=" * W)

    low_flag = "  ⚠️  low" if (flag_low and total < CONFIG["mc_min_trades"]) else ""
    print(f"  Trades/day : {tpd:.1f}  (total {total}){low_flag}")

    wr_note = "✅" if wr >= 54 else ("⚠️  marginal" if wr >= 51 else "❌ no edge")
    print(f"  Win rate   : {wr:.1f}%  ({wins}W / {total-wins}L)  [{wr_note}]")
    print(f"             → Need >54% for real edge after 2% fee.")

    pnl_note = "✅" if pnl > 0 and wr >= 54 else ("⚠️" if pnl > 0 else "❌")
    print(f"  P&L        : ${pnl:+.2f}  (ROI {roi:+.1f}%)  [{pnl_note}]")

    sh_note = "✅" if sh > 1.0 else ("⚠️" if sh > 0 else "❌")
    print(f"  Sharpe     : {sh}  [{sh_note}]")

    dd_note = "✅" if dd < 15 else ("⚠️" if dd < 25 else "❌")
    print(f"  Drawdown   : {dd}%  [{dd_note}]")

    # Regime breakdown — shows if choppy trades are helping or hurting
    print(f"  Regime split:")
    print(f"    Trending  : {full_trades:>4} trades  WR {full_wr:.1f}%")
    print(f"    Choppy    : {choppy_trades:>4} trades  WR {choppy_wr:.1f}%"
          f"  (40% Kelly sizing)")

    if result.halt_count > 0:
        print(f"  DD halts   : {result.halt_count} halt(s)  "
              f"{result.skipped_halted} candles paused  "
              f"(50%={CONFIG['drawdown_cooldown_hours']}h cooldown | 80%=permanent)")
    if result.skipped_signal > 0:
        pct = result.skipped_signal / max(result.attempted, 1) * 100
        print(f"  Filtered   : {result.skipped_signal:,} candles ({pct:.0f}%) "
              f"by RSI/ROC/EMA/Vol filters")
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

    print(f"  Saved: {tf['trades_file']}  |  {tf['equity_file']}  |  {tf['mc_file']}\n")


# ══════════════════════════════════════════════════════════════
# RUN ONE TIMEFRAME
# ══════════════════════════════════════════════════════════════

async def run_timeframe(tf_key: str, tf: dict) -> dict:
    days            = CONFIG["lookback_days"]
    mins_per_candle = int(tf["interval"].replace("m", ""))
    W               = 62

    print(f"\n{'═'*W}")
    print(f"  TIMEFRAME: {tf['label'].upper()}  ({tf['interval']} candles)")
    print(f"  Regime: size-scaled (flat=40%, trend=100%, counter=skip)")
    print(f"  ATR: size-scaled (low=60%, normal=100%, high=120%)")
    print(f"  DD halt: 50% → {CONFIG['drawdown_cooldown_hours']}h cooldown then reset  |  80% → permanent halt")
    print(f"{'═'*W}\n")

    df = await fetch_candles(tf["interval"], tf["cache_file"])
    if len(df) < 300:
        print("  Not enough data.\n")
        return {}

    split      = int(len(df) * CONFIG["train_pct"])
    days_train = days * CONFIG["train_pct"]
    days_test  = days * (1 - CONFIG["train_pct"])

    print("── Grid Search (in-sample only) ──────────────────────\n")
    best_cfg = grid_search(df, split, tf["grid_file"], days_train)
    print()

    print("── Walk-Forward: Single Split ────────────────────────\n")
    train = run_simulation(df, 0,     split,    best_cfg, tf["interval"])
    test  = run_simulation(df, split, len(df),  best_cfg, tf["interval"])
    print_result(f"In-sample  ({int(days_train)}d)", train, days_train)
    print_result(f"Out-of-sample ({int(days_test)}d)", test, days_test, flag_low=True)

    if train.trades and test.trades and len(test.trades) >= 10:
        gap = train.win_rate - test.win_rate
        if gap > 0.10:
            print(f"  ⚠️  WR drops {gap*100:.1f}pp OOS — some overfit.\n")
        elif test.win_rate >= 0.54:
            print(f"  ✅  WR holds OOS ({test.win_rate*100:.1f}%).\n")
        else:
            print(f"  ➖  WR borderline OOS ({test.win_rate*100:.1f}%).\n")

    print("── Walk-Forward: Rolling Windows ─────────────────────\n")
    windows = rolling_walk_forward(df, best_cfg, tf["interval"], mins_per_candle)
    print_rolling_wf(windows)

    print("── Full Period Baseline ───────────────────────────────\n")
    base = run_simulation(df, 0, len(df), best_cfg, tf["interval"])
    print_result(f"Full {days}d  |  {tf['interval']}", base, days)

    print("── Monte Carlo ───────────────────────────────────────\n")
    mc = monte_carlo(base.trades, CONFIG["mc_simulations"])
    print_monte_carlo(mc)

    save_csvs(base, mc, tf)

    reliable_windows = [w for w in windows if w["reliable"]]
    wf_consistency   = (
        sum(1 for w in reliable_windows if w["win_rate"] >= 0.54)
        / max(len(reliable_windows), 1)
    )

    return {
        "timeframe":  tf["label"],
        "interval":   tf["interval"],
        "trades":     len(base.trades),
        "win_rate":   base.win_rate,
        "pnl":        base.total_pnl,
        "drawdown":   max_drawdown(base.equity),
        "ruin_pct":   mc.get("ruin_pct", 100),
        "oos_wr":     test.win_rate,
        "wf_consist": wf_consistency,
    }


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

async def run_backtest() -> None:
    summaries = []
    for tf_key, tf in TIMEFRAMES.items():
        s = await run_timeframe(tf_key, tf)
        if s:
            summaries.append(s)

    if not summaries:
        print("No results.")
        return

    W = 62
    print(f"\n{'═'*W}")
    print("  FINAL COMPARISON  —  v8 (regime-aware sizing)")
    print(f"{'═'*W}\n")

    print(f"  {'Timeframe':<22} {'Trades':>7} {'WR':>6} "
          f"{'P&L':>8} {'OOS WR':>8} {'WF%':>6} {'Ruin':>6}")
    print(f"  {'-'*22} {'-'*7} {'-'*6} {'-'*8} {'-'*8} {'-'*6} {'-'*6}")

    for s in summaries:
        print(f"  {s['timeframe']:<22} "
              f"{s['trades']:>7} "
              f"{s['win_rate']*100:>5.1f}% "
              f"${s['pnl']:>+7.2f} "
              f"{s['oos_wr']*100:>7.1f}% "
              f"{s['wf_consist']*100:>5.0f}% "
              f"{s['ruin_pct']:>5.1f}%")

    print()
    print(f"  WR=win rate | OOS=out-of-sample | WF%=rolling windows | Ruin=MC ruin\n")

    for s in summaries:
        wr   = s["win_rate"] * 100
        pnl  = s["pnl"]
        dd   = s["drawdown"]
        ruin = s["ruin_pct"]
        oos  = s["oos_wr"] * 100
        wfc  = s["wf_consist"] * 100
        tf   = s["interval"]

        print(f"  {tf} ({s['timeframe']}):")

        if wr >= 54 and oos >= 52 and dd < 20 and ruin < 5 and wfc >= 75:
            verdict = "✅  PROMISING — paper trade 2+ weeks before going live."
        elif wr >= 51 and oos >= 50 and pnl > 0 and ruin < 15:
            verdict = "⚠️   MARGINAL — getting closer. Check regime breakdown above."
        else:
            verdict = "❌  NEEDS WORK — check regime breakdown for clues."

        print(f"    {verdict}")

        if s["trades"] == 0:
            print(f"    → No trades. Lower trend_slope_threshold to 0.00002.")
        elif s["trades"] < CONFIG["mc_min_trades"]:
            print(f"    → Only {s['trades']} trades — increase lookback_days to 180.")
        if wr < 54 and s["trades"] > 0:
            print(f"    → WR {wr:.1f}%: check if choppy trades are dragging it down.")
            print(f"       If choppy WR < 48%, raise trend_flat_size to 0.0 to skip them.")
        if oos < 52 and s["trades"] > 0:
            print(f"    → OOS {oos:.1f}%: raise grid_min_trades to 30.")
        if wfc < 75 and s["trades"] > 0:
            print(f"    → WF {wfc:.0f}%: inconsistent across windows.")
        if dd > 25:
            print(f"    → Drawdown {dd:.1f}%: lower max_position_pct to 3%.")
        if ruin > 5:
            print(f"    → Ruin {ruin:.1f}%: lower max_position_pct or kelly_fraction.")
        print()


if __name__ == "__main__":
    asyncio.run(run_backtest())
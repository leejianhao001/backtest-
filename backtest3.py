"""
backtest.py — Polymarket BTC Up/Down Bot  (v11 — Kelly luck mode)
=================================================================
Run with:
    pip install aiohttp pandas numpy scipy
    python backtest.py

WHAT THIS IS
──────────────────────────────────────────────────────────────
This bot does not have a proven statistical edge. Win rate sits
around 48-52% which is below the 53% break-even after fees.

What it DOES have:
  - Kelly sizing that compounds aggressively on winning streaks
  - Tight drawdown protection that limits max loss to ~20%
  - High trade frequency to maximise chances of a lucky run
  - Both 5m and 15m markets for more opportunities

Think of it like a slot machine with a stop-loss. You might
hit a lucky streak that compounds into big returns. The DD
halts mean you can't lose everything in one bad run.

RISK SETTINGS
──────────────────────────────────────────────────────────────
  20% DD → 1h cooldown then resume    (short pause, try again)
  50% DD → 4h cooldown then resume    (bigger pause)
  80% DD → permanent halt             (done for this session)

Max you can lose per session: ~80% of starting bankroll.
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

    # ── Signals ───────────────────────────────────────────────
    # Kept simple — more signals firing = more trades = more
    # chances for Kelly to compound on lucky streaks.
    "roc_threshold":   0.0003,
    "rsi_period":      14,
    "rsi_oversold":    45,
    "rsi_overbought":  55,
    "ema_fast":        5,
    "ema_slow":        20,
    "vol_window":      20,
    "vol_spike_mult":  1.0,   # just needs to be above average

    # ── Grid search ───────────────────────────────────────────
    "grid_roc":        [0.0001, 0.0003, 0.0005, 0.001],
    "grid_rsi_band":   [5, 10, 15, 20],
    "grid_ema_slow":   [10, 20, 30],
    "grid_min_trades": 30,

    # ── Bankroll & sizing ─────────────────────────────────────
    "bankroll":         100.0,
    "kelly_fraction":   0.5,      # half Kelly
    "max_position_pct": 0.05,     # max 5% per trade
    "min_bet_usdc":     1.0,
    "min_win_prob":     0.51,     # low floor — more trades

    # ── Realism ───────────────────────────────────────────────
    "polymarket_fee":  0.02,
    "slippage_pct":    0.005,

    # ── THREE-TIER drawdown protection ────────────────────────
    # Tier 1: 20% DD → short pause, reset and try again
    # Tier 2: 50% DD → longer pause, reset and try again
    # Tier 3: 80% DD → permanent stop for this session
    "dd_tier1_pct":        0.20,   # 20% triggers short cooldown
    "dd_tier1_hours":      1,      # 1 hour cooldown
    "dd_tier2_pct":        0.50,   # 50% triggers long cooldown
    "dd_tier2_hours":      4,      # 4 hour cooldown
    "dd_tier3_pct":        0.80,   # 80% triggers permanent halt

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
                print(f"  Cache fresh ({age/3600:.1f}h old) — {cache_file} "
                      f"({len(df):,} candles)")
                return df
            print("  Cache missing columns — re-downloading...")
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
                    print(f"\n  Binance {r.status} — stopping.")
                    break
                chunk = await r.json()
            if not chunk:
                break
            all_raw  = chunk + all_raw
            end_time = chunk[0][0] - 1
            print(f"    {len(all_raw):>6}/{total_needed}...", end="\r")
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
    print(f"  Saved → {cache_file}\n")
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


def compute_signal(closes: np.ndarray, volumes: np.ndarray,
                   window_open: float, cfg: dict) -> tuple:
    """
    Simple 3-of-4 voting: RSI + ROC + EMA + Volume.
    Designed for high trade frequency rather than selectivity.
    """
    min_len = max(cfg["ema_slow"] + 2, cfg["rsi_period"] + 2,
                  cfg["vol_window"] + 2)
    if len(closes) < min_len:
        return None, 0.0

    rsi = compute_rsi(closes, cfg["rsi_period"])
    if cfg["rsi_oversold"] < rsi < cfg["rsi_overbought"]:
        return None, 0.0
    s1 = "up" if rsi <= cfg["rsi_oversold"] else "down"

    roc = (closes[-1] - closes[-2]) / closes[-2]
    if abs(roc) < cfg["roc_threshold"]:
        return None, 0.0
    s2 = "up" if roc > 0 else "down"

    s     = pd.Series(closes)
    ema_f = s.ewm(span=cfg["ema_fast"], adjust=False).mean().iloc[-1]
    ema_s = s.ewm(span=cfg["ema_slow"], adjust=False).mean().iloc[-1]
    s3    = "up" if ema_f > ema_s else "down"

    vol_win  = cfg["vol_window"]
    avg_vol  = volumes[-(vol_win + 1):-1].mean() if len(volumes) >= vol_win + 1 else 0
    high_vol = volumes[-1] >= cfg["vol_spike_mult"] * avg_vol if avg_vol > 0 else False
    s4       = s2 if high_vol else "neutral"

    if s1 == "up":
        rsi_c = max((cfg["rsi_oversold"] - rsi) / max(cfg["rsi_oversold"], 1), 0.0)
    else:
        rsi_c = max((rsi - cfg["rsi_overbought"]) / max(100 - cfg["rsi_overbought"], 1), 0.0)
    roc_c     = min(abs(roc) / cfg["roc_threshold"] / 10.0, 1.0)
    base_conf = min((rsi_c + roc_c) / 2.0, 1.0)

    ups   = [s1, s2, s3].count("up")
    downs = [s1, s2, s3].count("down")
    vol_confirms = (s4 != "neutral")

    if ups == 3 and vol_confirms and s2 == "up":
        return "up",   round(base_conf * 1.0, 3)
    if downs == 3 and vol_confirms and s2 == "down":
        return "down", round(base_conf * 1.0, 3)
    if ups == 3 and not vol_confirms:
        return "up",   round(base_conf * 0.65, 3)
    if downs == 3 and not vol_confirms:
        return "down", round(base_conf * 0.65, 3)
    if ups == 2 and vol_confirms and s2 == "up":
        return "up",   round(base_conf * 0.65, 3)
    if downs == 2 and vol_confirms and s2 == "down":
        return "down", round(base_conf * 0.65, 3)

    return None, 0.0


# ══════════════════════════════════════════════════════════════
# MARKET PRICE & SIZING
# ══════════════════════════════════════════════════════════════

def simulate_market_price(direction: str, roc: float) -> float:
    rng   = np.random.default_rng(seed=int(abs(roc) * 1e9) % (2 ** 31))
    noise = rng.uniform(-0.04, 0.04)
    nudge = abs(roc) * 20
    base  = (0.50 + nudge) if direction == "up" else (0.50 - nudge)
    return round(float(np.clip(base + noise, 0.35, 0.65)), 3)


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
# SIMULATION — three-tier DD halt
# ══════════════════════════════════════════════════════════════

@dataclass
class SimResult:
    trades:         list  = field(default_factory=list)
    equity:         list  = field(default_factory=list)
    trade_returns:  list  = field(default_factory=list)
    skipped_signal: int   = 0
    skipped_kelly:  int   = 0
    skipped_halted: int   = 0
    halt_t1:        int   = 0   # tier 1 halts (20% DD)
    halt_t2:        int   = 0   # tier 2 halts (50% DD)
    halt_t3:        int   = 0   # tier 3 halts (80% DD permanent)
    attempted:      int   = 0

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        return sum(1 for t in self.trades if t["outcome"] == "WIN") / len(self.trades)

    @property
    def total_pnl(self) -> float:
        return (self.equity[-1] - CONFIG["bankroll"]) if self.equity else 0.0


def run_simulation(df: pd.DataFrame, start_idx: int, end_idx: int,
                   cfg: dict, tf_label: str = "") -> SimResult:
    """
    Three-tier drawdown protection:
      20% DD → 1h cooldown, peak resets on resume
      50% DD → 4h cooldown, peak resets on resume
      80% DD → permanent halt, session over
    """
    result        = SimResult()
    bankroll      = CONFIG["bankroll"]
    peak_bankroll = bankroll
    result.equity.append(bankroll)

    mins_pc = int(tf_label.replace("m", "")) if tf_label.replace("m", "").isdigit() else 15
    t1_candles = int(cfg.get("dd_tier1_hours", 1) * 60 / mins_pc)
    t2_candles = int(cfg.get("dd_tier2_hours", 4) * 60 / mins_pc)

    halt_until = -1

    min_i = max(start_idx,
                start_idx + cfg["ema_slow"] + cfg["rsi_period"] + cfg["vol_window"] + 5)

    for i in range(min_i, end_idx - 1):
        result.attempted += 1

        if bankroll > peak_bankroll:
            peak_bankroll = bankroll

        # Cooldown expired — reset peak and resume
        if halt_until > 0 and i >= halt_until:
            peak_bankroll = bankroll
            halt_until    = -1

        # Still in cooldown
        if halt_until > 0 and i < halt_until:
            result.skipped_halted += 1
            continue

        dd_now = (peak_bankroll - bankroll) / peak_bankroll if peak_bankroll > 0 else 0.0

        # Tier 3: 80% DD — permanent halt
        if dd_now >= cfg.get("dd_tier3_pct", 0.80):
            result.halt_t3 += 1
            result.skipped_halted += 1
            halt_until = end_idx + 1
            continue

        # Tier 2: 50% DD — 4h cooldown
        if dd_now >= cfg.get("dd_tier2_pct", 0.50):
            result.halt_t2 += 1
            result.skipped_halted += 1
            halt_until = i + t2_candles
            continue

        # Tier 1: 20% DD — 1h cooldown
        if dd_now >= cfg.get("dd_tier1_pct", 0.20):
            result.halt_t1 += 1
            result.skipped_halted += 1
            halt_until = i + t1_candles
            continue

        closes  = df["close"].iloc[max(0, i - 50): i + 1].values
        volumes = df["volume"].iloc[max(0, i - 50): i + 1].values
        roc     = (closes[-1] - closes[-2]) / closes[-2] if len(closes) >= 2 else 0.0

        direction, confidence = compute_signal(
            closes, volumes, float(df["open"].iloc[i]), cfg)

        if direction is None:
            result.skipped_signal += 1
            continue

        win_prob_yes = 0.52 + confidence * 0.08

        market_price   = simulate_market_price(direction, roc)
        entry_bankroll = bankroll

        bet = kelly_bet(win_prob_yes, market_price, bankroll, direction)
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
            "timeframe":    tf_label,
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
        print(f"  ⚠️  No combo reached {CONFIG['grid_min_trades']} trades.\n")
        return CONFIG

    rdf  = pd.DataFrame(rows).sort_values("score", ascending=False)
    rdf.to_csv(grid_file, index=False)
    best = rdf.iloc[0].to_dict()

    print(f"  Best: roc={best['roc_threshold']}  "
          f"rsi_band=±{int(best['rsi_band'])}  "
          f"ema_slow={int(best['ema_slow'])}")
    print(f"  Trades={int(best['trades'])}  "
          f"WR={best['win_rate_pct']}%  score={best['score']}")
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
                         tf_label: str) -> list:
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

        res      = run_simulation(df, ts, te, best_cfg, tf_label)
        n_trades = len(res.trades)
        reliable = n_trades >= 10

        start_dt = df["time"].iloc[ts].strftime("%b %d")
        end_dt   = df["time"].iloc[te - 1].strftime("%b %d")

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
            if w["win_rate"] >= 0.53:
                status = "✅ positive"; profitable += 1
            elif w["win_rate"] >= 0.48:
                status = "➖ marginal"
            else:
                status = "❌ negative"

        print(f"  {w['window']:>2}  {w['period']:<18} {w['trades']:>7} "
              f"{wr_str:>8} {pnl_str:>8}  {status}")

    print()
    if reliable > 0:
        pct = profitable / reliable * 100
        print(f"  {profitable}/{reliable} windows positive ({pct:.0f}%)\n")


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
        "ruin_pct":      round(ruins / n_sims * 100, 1),
        "median_final":  round(float(np.median(arr)), 2),
        "p10_final":     round(float(np.percentile(arr, 10)), 2),
        "p90_final":     round(float(np.percentile(arr, 90)), 2),
        "median_dd_pct": round(float(np.median(dd_arr)), 1),
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
    print("  Shows the range of possible outcomes from luck alone.")
    print("=" * W)
    if mc.get("too_few"):
        print(f"  ⚠️  Only {mc['n_trades']} trades — MC is approximate.\n")
    print(f"  Starting bankroll : ${start:.2f}")
    print(f"  Median outcome    : ${mc['median_final']:.2f}  "
          f"({(mc['median_final']-start)/start*100:+.1f}%)")
    print(f"  Best 10% of runs  : ${mc['p90_final']:.2f}+  "
          f"← lucky scenario")
    print(f"  Worst 10% of runs : ${mc['p10_final']:.2f}  "
          f"← unlucky scenario")
    print(f"  Median drawdown   : {mc['median_dd_pct']}%")
    ruin_note = "✅" if mc["ruin_pct"] < 5 else "⚠️"
    print(f"  Ruin probability  : {mc['ruin_pct']}%  [{ruin_note}]")
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

    W = 62
    print("=" * W)
    print(f"  {label}")
    print("=" * W)

    low_flag = "  ⚠️  low" if (flag_low and total < CONFIG["mc_min_trades"]) else ""
    print(f"  Trades/day : {tpd:.1f}  (total {total}){low_flag}")

    wr_note = "✅" if wr >= 53 else ("➖" if wr >= 50 else "❌")
    print(f"  Win rate   : {wr:.1f}%  ({wins}W / {total-wins}L)  [{wr_note}]")

    pnl_note = "✅" if pnl > 0 else "❌"
    print(f"  P&L        : ${pnl:+.2f}  (ROI {roi:+.1f}%)  [{pnl_note}]")

    sh_note = "✅" if sh > 1.0 else ("➖" if sh > 0 else "❌")
    print(f"  Sharpe     : {sh}  [{sh_note}]")

    dd_note = "✅" if dd < 20 else ("➖" if dd < 30 else "❌")
    print(f"  Drawdown   : {dd}%  [{dd_note}]")

    # DD halt summary
    halts = result.halt_t1 + result.halt_t2 + result.halt_t3
    if halts > 0:
        print(f"  DD halts   : T1(20%)×{result.halt_t1}  "
              f"T2(50%)×{result.halt_t2}  "
              f"T3(80%)×{result.halt_t3}  "
              f"({result.skipped_halted} candles paused)")
    else:
        print(f"  DD halts   : none")
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
# RUN ONE TIMEFRAME
# ══════════════════════════════════════════════════════════════

async def run_timeframe(tf_key: str, tf: dict) -> dict:
    days = CONFIG["lookback_days"]
    W    = 62

    print(f"\n{'═'*W}")
    print(f"  TIMEFRAME: {tf['label'].upper()}  ({tf['interval']} candles)")
    print(f"  Mode: Kelly luck  |  Halts: 20%→1h  50%→4h  80%→permanent")
    print(f"{'═'*W}\n")

    df = await fetch_candles(tf["interval"], tf["cache_file"])
    if len(df) < 200:
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

    print("── Walk-Forward: Rolling Windows ─────────────────────\n")
    windows = rolling_walk_forward(df, best_cfg, tf["interval"])
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
        sum(1 for w in reliable_windows if w["win_rate"] >= 0.53)
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
        "mc":         mc,
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
    print("  FINAL SUMMARY  —  v11 Kelly Luck Mode")
    print(f"  Halts: 20%DD→1h  |  50%DD→4h  |  80%DD→permanent stop")
    print(f"{'═'*W}\n")

    print(f"  {'Timeframe':<22} {'Trades':>7} {'WR':>6} "
          f"{'P&L':>8} {'OOS WR':>8} {'Ruin':>6}")
    print(f"  {'-'*22} {'-'*7} {'-'*6} {'-'*8} {'-'*8} {'-'*6}")

    for s in summaries:
        print(f"  {s['timeframe']:<22} "
              f"{s['trades']:>7} "
              f"{s['win_rate']*100:>5.1f}% "
              f"${s['pnl']:>+7.2f} "
              f"{s['oos_wr']*100:>7.1f}% "
              f"{s['ruin_pct']:>5.1f}%")

    print()
    print(f"  ⚠️  REMINDER: This strategy relies on Kelly luck,")
    print(f"  not a proven statistical edge. Treat it as a gamble")
    print(f"  with a maximum loss of ~80% of your starting bankroll.\n")

    for s in summaries:
        mc  = s.get("mc", {})
        start = CONFIG["bankroll"]
        print(f"  {s['interval']} possible outcomes from Monte Carlo:")
        if mc:
            print(f"    Lucky run  : ${mc.get('p90_final', 0):.2f}+  "
                  f"({(mc.get('p90_final',start)-start)/start*100:+.0f}%)")
            print(f"    Median run : ${mc.get('median_final', 0):.2f}  "
                  f"({(mc.get('median_final',start)-start)/start*100:+.0f}%)")
            print(f"    Unlucky run: ${mc.get('p10_final', 0):.2f}  "
                  f"({(mc.get('p10_final',start)-start)/start*100:+.0f}%)")
        print()


if __name__ == "__main__":
    asyncio.run(run_backtest())
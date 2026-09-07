"""Strategy library. Every strategy returns a signal frame aligned 1:1 with the
input bars. A row at bar i means: *bar i has closed and these are the orders*.
The engine fills them at bar i+1's open. Nothing here may read bar i+1.
"""
import numpy as np
import pandas as pd
import indicators as ta

SIG_COLS = ["side", "sl", "tp", "ref_price", "max_hold_bars", "tag",
            "risk_mult", "atr", "be_at_r", "trail_atr", "trail_start_r", "regime"]


def blank(df, tag=""):
    s = pd.DataFrame(index=df.index)
    s["side"] = 0
    s["sl"] = np.nan; s["tp"] = np.nan
    s["ref_price"] = df.close
    s["max_hold_bars"] = 0
    s["tag"] = tag
    s["risk_mult"] = 1.0
    s["atr"] = np.nan
    s["be_at_r"] = 0.0
    s["trail_atr"] = 0.0
    s["trail_start_r"] = 1.0
    s["regime"] = ""
    return s


# ══════════════════════════════════════════════════════════════════════
#  LEGACY — an exact re-implementation of what the current repo does,
#  used only to measure the honest baseline. Do not "improve" this.
# ══════════════════════════════════════════════════════════════════════

def legacy_smc(df, lookback=36, tp_atr=2.0, sl_atr=1.5,
               max_hold_bars=96, vol_mult=1.25, trend_filter=True):
    s = blank(df, "LEGACY_SMC")
    a = ta.atr(df, 14)
    vol20 = df.volume.rolling(20, min_periods=20).mean()
    e200 = ta.ema(df.close, 200)
    sw_lo = df.low.shift(1).rolling(lookback, min_periods=lookback).min()
    sw_hi = df.high.shift(1).rolling(lookback, min_periods=lookback).max()

    long_sweep = (df.low < sw_lo) & (df.close > sw_lo)
    short_sweep = (df.high > sw_hi) & (df.close < sw_hi)
    volok = df.volume >= vol_mult * vol20
    long_sweep &= volok; short_sweep &= volok
    if trend_filter:
        long_sweep &= df.close >= e200
        short_sweep &= df.close <= e200

    side = np.where(long_sweep, 1, np.where(short_sweep, -1, 0))
    a_f = a.fillna(df.close * 0.01)
    s["side"] = side
    s["atr"] = a_f
    s["sl"] = np.where(side > 0, df.close - sl_atr * a_f,
                       df.close + sl_atr * a_f)
    s["tp"] = np.where(side > 0, df.close + tp_atr * a_f,
                       df.close - tp_atr * a_f)
    s["max_hold_bars"] = max_hold_bars
    s.loc[a.isna() | vol20.isna() | e200.isna(), "side"] = 0
    return s


LEGACY_CCI = {
    "BTC": (2.0, -135, 2.2, 140), "ETH": (2.0, -130, 2.3, 135),
    "SOL": (2.2, -140, 2.4, 145), "LINK": (2.0, -130, 2.2, 130),
    "NEAR": (2.2, -140, 2.3, 140), "HYPE": (2.5, -150, 2.5, 150),
    "PEPE": (2.5, -150, 2.5, 150), "WIF": (2.5, -150, 2.5, 150),
    "FET": (2.3, -140, 2.3, 140),
}


def legacy_cci_bb(df, symbol, tp_atr=2.0, sl_atr=1.5, max_hold_bars=48):
    s = blank(df, "LEGACY_CCI_BB")
    lo_k, lo_th, hi_k, hi_th = LEGACY_CCI.get(symbol, (2.0, -134, 2.0, 134))
    c = ta.cci(df, 14)
    e100 = ta.ema(df.close, 100)
    a = ta.atr(df, 14)
    _, _, lower = ta.bbands(df.close, 20, lo_k)
    upper, _, _ = ta.bbands(df.close, 20, hi_k)

    lng = (c <= lo_th) & (df.close < lower) & (df.close >= e100)
    sht = (c >= hi_th) & (df.close > upper) & (df.close <= e100)
    side = np.where(lng, 1, np.where(sht, -1, 0))
    a_f = a.fillna(df.close * 0.01)
    s["side"] = side; s["atr"] = a_f
    s["sl"] = np.where(side > 0, df.close - sl_atr * a_f, df.close + sl_atr * a_f)
    s["tp"] = np.where(side > 0, df.close + tp_atr * a_f, df.close - tp_atr * a_f)
    s["max_hold_bars"] = max_hold_bars
    s.loc[a.isna() | e100.isna() | lower.isna(), "side"] = 0
    return s


# ══════════════════════════════════════════════════════════════════════
#  V2 — the rebuilt strategies
# ══════════════════════════════════════════════════════════════════════

def add_regime(df, ltf_index=None, htf="4h"):
    """Causal regime label from the asset's own higher timeframe.
    trend  : |EMA50-EMA200|/ATR large and efficiency ratio high
    chop   : efficiency ratio low
    """
    h = ta.resample(df, htf) if htf else df
    e50 = ta.ema(h.close, 50)
    e200 = ta.ema(h.close, 200)
    er = ta.efficiency_ratio(h.close, 20)
    slope = e50.diff(5) / h.close
    lab = pd.Series("chop", index=h.index, dtype=object)
    lab[(e50 > e200) & (er > 0.30) & (slope > 0)] = "bull"
    lab[(e50 < e200) & (er > 0.30) & (slope < 0)] = "bear"
    idx = ltf_index if ltf_index is not None else df.index
    return ta.align_htf(lab, idx).fillna("chop"), ta.align_htf(er, idx)


def sweep_reclaim_v2(df, symbol="", lookback=48, htf="4h",
                     sl_atr=1.1, tp_atr=2.6, max_hold_bars=64,
                     vol_mult=1.3, min_atr_pct=0.0015, max_atr_pct=0.06,
                     be_at_r=1.0, trail_atr=2.0, trail_start_r=1.5,
                     require_reclaim_body=True, allow_regimes=("bull", "bear", "chop"),
                     with_trend_only=True, adx_max=60, session_filter=False):
    """Liquidity sweep + reclaim, rebuilt.

    Changes vs legacy:
      * stop sits beyond the SWEEP WICK (structure), not a fixed ATR from close
      * asymmetric R:R >= 2 with a break-even move and an ATR trail
      * volatility band filter: skips dead tape and blow-off tape
      * regime gate from the asset's own 4h structure
      * short and long are evaluated independently, no long-priority bug
    """
    s = blank(df, "SWEEP_V2")
    a = ta.atr(df, 14)
    atr_pct = a / df.close
    vol20 = df.volume.rolling(20, min_periods=20).mean()
    e200 = ta.ema(df.close, 200)
    adx14, _, _ = ta.adx(df, 14)
    sw_hi, sw_lo = ta.donchian(df, lookback)
    reg, er = add_regime(df, df.index, htf)

    rng = (df.high - df.low).replace(0, np.nan)
    body_lo = (df.close - df.low) / rng      # closes near the high -> buyers
    body_hi = (df.high - df.close) / rng

    volok = df.volume >= vol_mult * vol20
    volband = (atr_pct >= min_atr_pct) & (atr_pct <= max_atr_pct)
    calm = adx14 <= adx_max

    lng = (df.low < sw_lo) & (df.close > sw_lo) & volok & volband & calm
    sht = (df.high > sw_hi) & (df.close < sw_hi) & volok & volband & calm
    if require_reclaim_body:
        lng &= body_lo >= 0.55
        sht &= body_hi >= 0.55
    if with_trend_only:
        lng &= df.close >= e200
        sht &= df.close <= e200
    okreg = reg.isin(allow_regimes)
    lng &= okreg; sht &= okreg

    # stop beyond the sweep wick with a small ATR pad, floored at 0.6 ATR
    pad = 0.25 * a
    sl_long = np.minimum(df.low - pad, df.close - 0.6 * a)
    sl_short = np.maximum(df.high + pad, df.close + 0.6 * a)
    # ...and capped so one trade can never risk an absurd distance
    sl_long = np.maximum(sl_long, df.close - sl_atr * 2.5 * a)
    sl_short = np.minimum(sl_short, df.close + sl_atr * 2.5 * a)

    side = np.where(lng & ~sht, 1, np.where(sht & ~lng, -1, 0))
    risk_long = df.close - sl_long
    risk_short = sl_short - df.close

    s["side"] = side
    s["atr"] = a
    s["sl"] = np.where(side > 0, sl_long, sl_short)
    s["tp"] = np.where(side > 0, df.close + tp_atr / sl_atr * risk_long,
                       df.close - tp_atr / sl_atr * risk_short)
    s["max_hold_bars"] = max_hold_bars
    s["be_at_r"] = be_at_r
    s["trail_atr"] = trail_atr
    s["trail_start_r"] = trail_start_r
    s["regime"] = reg.values
    bad = a.isna() | vol20.isna() | e200.isna() | sw_lo.isna() | adx14.isna()
    s.loc[bad, "side"] = 0
    return s


def trend_pullback_v2(df, symbol="", htf="4h", ema_fast=21, ema_slow=55,
                      pull_atr=0.5, sl_atr=1.3, tp_atr=3.0, max_hold_bars=96,
                      adx_min=20, be_at_r=1.0, trail_atr=2.5, trail_start_r=1.5,
                      min_atr_pct=0.0015, max_atr_pct=0.06):
    """Buy the pullback inside an established trend. Positive-skew engine:
    low win rate, large payoff — the structural opposite of the legacy scalper,
    which is what makes the pair of them worth running together."""
    s = blank(df, "PULLBACK_V2")
    a = ta.atr(df, 14); atr_pct = a / df.close
    ef = ta.ema(df.close, ema_fast); es = ta.ema(df.close, ema_slow)
    e200 = ta.ema(df.close, 200)
    adx14, pdi, mdi = ta.adx(df, 14)
    reg, er = add_regime(df, df.index, htf)
    rsi = ta.rsi(df.close, 14)

    up = (ef > es) & (df.close > e200) & (adx14 >= adx_min) & (reg == "bull")
    dn = (ef < es) & (df.close < e200) & (adx14 >= adx_min) & (reg == "bear")
    volband = (atr_pct >= min_atr_pct) & (atr_pct <= max_atr_pct)

    # pullback: price dipped to/below fast EMA this bar but closed back above it
    touch_up = (df.low <= ef + pull_atr * a) & (df.close > ef) & (rsi > 40)
    touch_dn = (df.high >= ef - pull_atr * a) & (df.close < ef) & (rsi < 60)

    lng = up & touch_up & volband
    sht = dn & touch_dn & volband
    side = np.where(lng & ~sht, 1, np.where(sht & ~lng, -1, 0))

    s["side"] = side; s["atr"] = a
    s["sl"] = np.where(side > 0, df.close - sl_atr * a, df.close + sl_atr * a)
    s["tp"] = np.where(side > 0, df.close + tp_atr * a, df.close - tp_atr * a)
    s["max_hold_bars"] = max_hold_bars
    s["be_at_r"] = be_at_r; s["trail_atr"] = trail_atr
    s["trail_start_r"] = trail_start_r
    s["regime"] = reg.values
    s.loc[a.isna() | e200.isna() | adx14.isna(), "side"] = 0
    return s


def mean_reversion_v2(df, symbol="", htf="4h", bb_n=20, bb_k=2.5, z_n=100,
                      sl_atr=1.5, tp_atr=2.0, max_hold_bars=24,
                      adx_max=22, rsi_lo=22, rsi_hi=78,
                      min_atr_pct=0.0012, max_atr_pct=0.05,
                      be_at_r=0.0, trail_atr=0.0, only_chop=True):
    """Fade statistical extremes, but ONLY in a range regime and ONLY with a
    hard stop. This is the legacy CCI_BB idea with the two things it was
    missing: a regime gate and a target that clears the fee drag."""
    s = blank(df, "MR_V2")
    a = ta.atr(df, 14); atr_pct = a / df.close
    upper, mid, lower = ta.bbands(df.close, bb_n, bb_k)
    adx14, _, _ = ta.adx(df, 14)
    rsi = ta.rsi(df.close, 14)
    reg, er = add_regime(df, df.index, htf)
    vol20 = df.volume.rolling(20, min_periods=20).mean()

    base = ((adx14 <= adx_max) & (atr_pct >= min_atr_pct)
            & (atr_pct <= max_atr_pct) & (df.volume >= vol20))
    if only_chop:
        base &= (reg == "chop")

    lng = base & (df.close < lower) & (rsi <= rsi_lo)
    sht = base & (df.close > upper) & (rsi >= rsi_hi)
    side = np.where(lng & ~sht, 1, np.where(sht & ~lng, -1, 0))

    # target the band midline, floored so it clears round-trip cost
    tgt_long = np.maximum(mid, df.close + 1.2 * a)
    tgt_short = np.minimum(mid, df.close - 1.2 * a)
    s["side"] = side; s["atr"] = a
    s["sl"] = np.where(side > 0, df.close - sl_atr * a, df.close + sl_atr * a)
    s["tp"] = np.where(side > 0, tgt_long, tgt_short)
    s["max_hold_bars"] = max_hold_bars
    s["be_at_r"] = be_at_r; s["trail_atr"] = trail_atr
    s["regime"] = reg.values
    s.loc[a.isna() | lower.isna() | adx14.isna(), "side"] = 0
    return s


def combine(*frames):
    """Merge several strategies on one symbol. First non-zero wins, in order."""
    out = frames[0].copy()
    for f in frames[1:]:
        take = (out.side == 0) & (f.side != 0)
        for c in SIG_COLS:
            out.loc[take, c] = f.loc[take, c]
    return out


def donchian_trend_v2(df, symbol="", entry_n=48, exit_n=24, atr_n=14,
                      sl_atr=2.5, trail_atr=3.0, trail_start_r=0.5,
                      tp_atr=0.0, max_hold_bars=0, adx_min=15,
                      ema_filter=200, min_atr_pct=0.001, max_atr_pct=0.08,
                      long_only=False, be_at_r=0.0, htf=None):
    """Classic breakout trend-following: buy the n-bar high, ride it with an
    ATR trail, no fixed profit target.

    This is the structural opposite of everything the old bot did. It expects a
    LOW win rate (30-40%) and makes its money from a small number of very large
    winners. It is also cheap to run: few trades, long holds, so fees are a
    small fraction of gross edge instead of a quarter of it.
    """
    s = blank(df, "DONCHIAN_V2")
    a = ta.atr(df, atr_n)
    atr_pct = a / df.close
    hi, lo = ta.donchian(df, entry_n)
    adx14, _, _ = ta.adx(df, 14)
    ef = ta.ema(df.close, ema_filter) if ema_filter else None

    volband = (atr_pct >= min_atr_pct) & (atr_pct <= max_atr_pct)
    strong = adx14 >= adx_min

    lng = (df.close > hi) & volband & strong
    sht = (df.close < lo) & volband & strong
    if ef is not None:
        lng &= df.close > ef
        sht &= df.close < ef
    if long_only:
        sht &= False

    side = np.where(lng & ~sht, 1, np.where(sht & ~lng, -1, 0))
    s["side"] = side
    s["atr"] = a
    s["sl"] = np.where(side > 0, df.close - sl_atr * a, df.close + sl_atr * a)
    # no fixed target by default: the trail is the exit. A far target keeps the
    # engine's level checks well-formed without ever realistically binding.
    far = (tp_atr if tp_atr else 40.0) * a
    s["tp"] = np.where(side > 0, df.close + far, df.close - far)
    s["max_hold_bars"] = max_hold_bars
    s["be_at_r"] = be_at_r
    s["trail_atr"] = trail_atr
    s["trail_start_r"] = trail_start_r
    if htf:
        reg, _ = add_regime(df, df.index, htf)
        s["regime"] = reg.values
    s.loc[a.isna() | hi.isna() | adx14.isna() |
          (ef.isna() if ef is not None else False), "side"] = 0
    return s


def vol_breakout_v2(df, symbol="", squeeze_n=96, squeeze_q=0.25, entry_n=24,
                    sl_atr=2.0, trail_atr=2.5, trail_start_r=0.5,
                    max_hold_bars=0, ema_filter=200, min_atr_pct=0.001,
                    be_at_r=0.0, htf=None):
    """Breakout, but only out of a genuine volatility contraction. Bollinger
    bandwidth in its own lowest quartile over `squeeze_n` bars, then an n-bar
    range break. Fewer, higher-quality breakouts than plain Donchian."""
    s = blank(df, "SQUEEZE_V2")
    a = ta.atr(df, 14)
    bw = ta.bb_width(df.close, 20, 2.0)
    thr = bw.rolling(squeeze_n, min_periods=squeeze_n).quantile(squeeze_q)
    hi, lo = ta.donchian(df, entry_n)
    ef = ta.ema(df.close, ema_filter) if ema_filter else None
    squeezed = bw.shift(1) <= thr.shift(1)
    volok = (a / df.close) >= min_atr_pct

    lng = squeezed & (df.close > hi) & volok
    sht = squeezed & (df.close < lo) & volok
    if ef is not None:
        lng &= df.close > ef
        sht &= df.close < ef
    side = np.where(lng & ~sht, 1, np.where(sht & ~lng, -1, 0))
    s["side"] = side; s["atr"] = a
    s["sl"] = np.where(side > 0, df.close - sl_atr * a, df.close + sl_atr * a)
    far = 40.0 * a
    s["tp"] = np.where(side > 0, df.close + far, df.close - far)
    s["max_hold_bars"] = max_hold_bars
    s["be_at_r"] = be_at_r
    s["trail_atr"] = trail_atr; s["trail_start_r"] = trail_start_r
    if htf:
        reg, _ = add_regime(df, df.index, htf)
        s["regime"] = reg.values
    s.loc[a.isna() | thr.isna() | hi.isna() |
          (ef.isna() if ef is not None else False), "side"] = 0
    return s


def btc_trend_filter(btc_df, index, n=100, ret_n=30, thresh=0.0):
    """+1 when BTC daily closes above its n-day EMA and is up over ret_n days,
    -1 when the mirror holds, 0 otherwise. Shifted so it is knowable at the
    start of the day it is applied to."""
    d = ta.resample(btc_df, "1D")
    e = ta.ema(d.close, n)
    r = d.close.pct_change(ret_n)
    f = pd.Series(0.0, index=d.index)
    f[(d.close > e) & (r > thresh)] = 1.0
    f[(d.close < e) & (r < -thresh)] = -1.0
    return ta.align_htf(f, index).fillna(0.0)


def apply_market_filter(sig, mfilter, mode="align"):
    """Zero out signals that fight the market-wide trend."""
    s = sig.copy()
    if mode == "align":
        bad = (s.side * mfilter.reindex(s.index).fillna(0.0)) <= 0
    elif mode == "long_only_bull":
        bad = ~((s.side > 0) & (mfilter.reindex(s.index) > 0))
    else:
        return s
    s.loc[bad, "side"] = 0
    return s

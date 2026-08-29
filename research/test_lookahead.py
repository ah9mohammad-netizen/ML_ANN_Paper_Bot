"""Look-ahead / repainting self-test.

For a strategy to be honest, the signal row at bar i must be IDENTICAL whether
it was computed on data[0:i+1] or on the full history. Anything that fails this
is reading the future (shift(-n), whole-series .mean(), resample without a
shift, etc). This is the same idea as freqtrade's `lookahead-analysis`.
"""
import warnings, sys
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import okx_data as od
import strategies as st

CHECK = ["side", "sl", "tp", "ref_price", "atr"]


def test(fn, df, name, cuts=25, tail=400, seed=3):
    rng = np.random.default_rng(seed)
    full = fn(df)
    n = len(df)
    pts = rng.integers(max(2000, n - tail * 12), n - 2, size=cuts)
    bad = {}
    for i in sorted(set(pts.tolist())):
        part = fn(df.iloc[: i + 1])
        a, b = part.iloc[i], full.iloc[i]
        for c in CHECK:
            x, y = a[c], b[c]
            if pd.isna(x) and pd.isna(y):
                continue
            if isinstance(x, (int, float, np.floating)) and isinstance(y, (int, float, np.floating)):
                if abs(float(x) - float(y)) > max(1e-9, abs(float(y)) * 1e-9):
                    bad[c] = bad.get(c, 0) + 1
            elif x != y:
                bad[c] = bad.get(c, 0) + 1
    ok = not bad
    print(f"  {'PASS' if ok else 'FAIL'}  {name:22s} "
          f"({len(set(pts.tolist()))} cut points)" + ("" if ok else f"  -> {bad}"))
    return ok


def main():
    df = od.get("BTC", "1h", "2024-01-01")
    print(f"look-ahead test on BTC 1h, {len(df)} bars\n")
    allok = True
    allok &= test(lambda d: st.legacy_smc(d), df, "legacy_smc")
    allok &= test(lambda d: st.legacy_cci_bb(d, "BTC"), df, "legacy_cci_bb")
    allok &= test(lambda d: st.sweep_reclaim_v2(d), df, "sweep_reclaim_v2")
    allok &= test(lambda d: st.trend_pullback_v2(d), df, "trend_pullback_v2")
    allok &= test(lambda d: st.mean_reversion_v2(d), df, "mean_reversion_v2")
    allok &= test(lambda d: st.donchian_trend_v2(d), df, "donchian_trend_v2")
    allok &= test(lambda d: st.vol_breakout_v2(d), df, "vol_breakout_v2")
    print("\n", "ALL CLEAN" if allok else "LOOK-AHEAD DETECTED")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())

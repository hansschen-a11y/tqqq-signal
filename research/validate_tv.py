#!/usr/bin/env python3
"""
tv=0.20 描述性驗證（pre-registered）
=====================================
目的：部署中的 Variant A 用 target_vol=0.20，但 final_strategy.py 只驗證過 0.30，
Variant B walk-forward 格子 [0.30..0.60] 也從未包含 0.20。
本腳本用「與 final_strategy.py 逐字相同的引擎數學」補上 0.20 的歷史描述性驗證。

── 判準（跑之前已鎖定，不得事後改）──
  PASS 條件（沿用 final_strategy.py 原始三目標 + 已鎖定的 bootstrap 準則）：
    G1. Total Return > SPY B&H
    G2. Sharpe > SPY B&H
    G3. MDD > -50%
    G4. Stationary block bootstrap (mean block=63, n=2000)：
        Sharpe(tv=0.20) − Sharpe(SPY) 的 95% CI 下界 > 0
  資訊項（不判 pass/fail）：
    vs tv=0.30 / 0.25 的 ΔCAGR、ΔMDD、ΔSharpe、逐年表

用法（GitHub Actions 一次性 job 或本機）：
  python validate_tv.py            # 預設 2015-01-02 起
"""

import numpy as np
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# ─── 與 final_strategy.py 完全相同的常數 ───
RF_ANNUAL = 0.045
RF_DAILY = (1 + RF_ANNUAL) ** (1 / 252) - 1
SLIPPAGE = 0.0005
START = '2015-01-02'

TV_GRID = [0.20, 0.25, 0.30]          # 0.20=部署值(待驗)；0.30=已驗基準；0.25=中間參考
BOOT_N = 2000
BOOT_MEAN_BLOCK = 63
SEED = 42


def download_data():
    """與 final_strategy.download_data 相同邏輯（只留需要的 tickers）。"""
    data = yf.download(['SPY', 'QQQ', 'TQQQ'], start='2010-01-01',
                       auto_adjust=True, progress=False)
    closes = data['Close'].copy()
    closes = closes.ffill().dropna()
    return closes


def run_variant_a(tqqq_ret, rv20, target_vol):
    """逐字對應 final_strategy.run_variant_a 的數學。"""
    daily_rets = []
    prev_pos = 1.0
    for i in range(1, len(tqqq_ret)):
        vol = rv20.iloc[i - 1]                     # LAGGED
        if np.isnan(vol) or vol <= 0:
            pos = 1.0
        else:
            pos = min(1.0, max(0.0, target_vol / vol))
        slip = abs(pos - prev_pos) * SLIPPAGE
        r = pos * tqqq_ret[i] + (1 - pos) * RF_DAILY - slip
        daily_rets.append(r)
        prev_pos = pos
    return np.array(daily_rets)


def calc_metrics(dr):
    """逐字對應 final_strategy.calc_metrics。"""
    dr = np.asarray(dr, dtype=float)
    eq = np.cumprod(1 + dr)
    total = eq[-1] - 1
    n_y = len(dr) / 252
    cagr = (1 + total) ** (1 / n_y) - 1
    sharpe = (np.mean(dr) - RF_DAILY) / np.std(dr) * np.sqrt(252) if np.std(dr) > 0 else 0
    mdd = (eq / np.maximum.accumulate(eq) - 1).min()
    return {'total': total, 'cagr': cagr, 'sharpe': sharpe, 'mdd': mdd}


def sharpe_of(dr):
    dr = np.asarray(dr)
    return (np.mean(dr) - RF_DAILY) / np.std(dr) * np.sqrt(252) if np.std(dr) > 0 else 0.0


def stationary_bootstrap_sharpe_diff(strat, bench, n_boot=BOOT_N,
                                     mean_block=BOOT_MEAN_BLOCK, seed=SEED):
    """
    Politis–Romano stationary bootstrap，成對重抽（同一組索引同時抽 strat 與 bench），
    回傳 Sharpe(strat)-Sharpe(bench) 的 bootstrap 分布 (2.5%, 50%, 97.5%)。
    """
    rng = np.random.default_rng(seed)
    n = len(strat)
    p = 1.0 / mean_block
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.empty(n, dtype=np.int64)
        pos = rng.integers(0, n)
        for t in range(n):
            if t > 0 and rng.random() >= p:
                pos = (pos + 1) % n
            else:
                if t > 0:
                    pos = rng.integers(0, n)
            idx[t] = pos
        diffs[b] = sharpe_of(strat[idx]) - sharpe_of(bench[idx])
    return np.percentile(diffs, [2.5, 50.0, 97.5])


def yearly_table(dr, dates, label):
    df = pd.DataFrame({'ret': dr}, index=dates)
    print(f"\n  逐年表：{label}")
    print(f"  {'Year':>6} {'Return':>10} {'Sharpe':>8} {'MaxDD':>8}")
    print(f"  {'-' * 36}")
    for year in sorted(df.index.year.unique()):
        yr = df[df.index.year == year]['ret']
        eq = np.cumprod(1 + yr.values)
        ret = eq[-1] - 1
        sh = sharpe_of(yr.values)
        dd = (eq / np.maximum.accumulate(eq) - 1).min()
        print(f"  {year:>6} {ret:>10.1%} {sh:>8.2f} {dd:>8.1%}")


def main():
    print("=" * 64)
    print("  tv=0.20 描述性驗證（判準已 pre-register，見檔頭）")
    print("=" * 64)
    print("\n下載資料中...")
    closes = download_data()
    c = closes[closes.index >= START].copy()
    rets = c.pct_change().iloc[1:]
    print(f"回測區間：{rets.index[0].date()} → {rets.index[-1].date()}"
          f"（{len(rets)} 交易日）")

    # 與 final_strategy.compute_indicators 相同：RV20 = simple-return rolling std × √252
    rv20 = rets['TQQQ'].rolling(20).std() * np.sqrt(252)
    tqqq_ret = rets['TQQQ'].values
    dates = rets.index[1:]

    spy_dr = rets['SPY'].values[1:]          # 對齊策略天數（策略從 i=1 起算）
    spy_m = calc_metrics(spy_dr)

    results = {}
    for tv in TV_GRID:
        dr = run_variant_a(tqqq_ret, rv20, tv)
        results[tv] = {'dr': dr, 'm': calc_metrics(dr)}

    # ── 總表 ──
    print(f"\n  {'策略':<22} {'Total':>9} {'CAGR':>7} {'Sharpe':>7} {'MDD':>8}")
    print(f"  {'-' * 58}")
    print(f"  {'SPY B&H':<22} {spy_m['total']:>9.0%} {spy_m['cagr']:>7.1%} "
          f"{spy_m['sharpe']:>7.3f} {spy_m['mdd']:>8.1%}")
    for tv in TV_GRID:
        m = results[tv]['m']
        tag = '← 部署值(待驗)' if tv == 0.20 else ('← 已驗基準' if tv == 0.30 else '')
        print(f"  {'VolTgt tv=%.2f' % tv:<22} {m['total']:>9.0%} {m['cagr']:>7.1%} "
              f"{m['sharpe']:>7.3f} {m['mdd']:>8.1%}  {tag}")

    # ── 判準檢核（tv=0.20）──
    m20 = results[0.20]['m']
    print("\n" + "=" * 64)
    print("  判準檢核：tv=0.20")
    print("=" * 64)
    g1 = m20['total'] > spy_m['total']
    g2 = m20['sharpe'] > spy_m['sharpe']
    g3 = m20['mdd'] > -0.50
    print(f"  G1 Total > SPY：   {m20['total']:.0%} vs {spy_m['total']:.0%}"
          f"  → {'PASS' if g1 else 'FAIL'}")
    print(f"  G2 Sharpe > SPY：  {m20['sharpe']:.3f} vs {spy_m['sharpe']:.3f}"
          f"  → {'PASS' if g2 else 'FAIL'}")
    print(f"  G3 MDD > -50%：    {m20['mdd']:.1%}"
          f"  → {'PASS' if g3 else 'FAIL'}")

    print(f"\n  G4 Bootstrap（stationary, mean block={BOOT_MEAN_BLOCK}, "
          f"n={BOOT_N}, seed={SEED}）計算中...")
    lo, med, hi = stationary_bootstrap_sharpe_diff(results[0.20]['dr'], spy_dr)
    g4 = lo > 0
    print(f"     Sharpe(0.20)−Sharpe(SPY) 95% CI = [{lo:+.3f}, {hi:+.3f}]"
          f"（中位 {med:+.3f}）→ {'PASS' if g4 else 'FAIL'}（下界>0）")

    n_pass = sum([g1, g2, g3, g4])
    print(f"\n  ★ 總結：{n_pass}/4 PASS "
          f"{'→ GO：tv=0.20 通過描述性驗證' if n_pass == 4 else '→ NO-GO 或需討論'}")

    # ── 對照實驗（v2 追加；明確標註為 post-hoc，用於解讀 G4，不改判準本身）──
    print("\n" + "=" * 64)
    print("  對照實驗（post-hoc，v2 追加）")
    print("=" * 64)
    print("  C1：已驗基準 tv=0.30 是否也過不了同一個 G4？")
    lo30, med30, hi30 = stationary_bootstrap_sharpe_diff(results[0.30]['dr'], spy_dr)
    print(f"     Sharpe(0.30)−Sharpe(SPY) 95% CI = [{lo30:+.3f}, {hi30:+.3f}]"
          f"（中位 {med30:+.3f}）→ {'PASS' if lo30 > 0 else 'FAIL'}")
    if lo30 <= 0:
        print("     → 已驗基準同樣 FAIL：G4 反映的是 11.5 年資料的統計檢定力上限，")
        print("       非 tv=0.20 特有的缺陷。")
    print("\n  C2：0.20 與 0.30 的 Sharpe 差可否與 0 區分？（成對比較）")
    lo23, med23, hi23 = stationary_bootstrap_sharpe_diff(
        results[0.20]['dr'], results[0.30]['dr'])
    print(f"     Sharpe(0.20)−Sharpe(0.30) 95% CI = [{lo23:+.3f}, {hi23:+.3f}]"
          f"（中位 {med23:+.3f}）")
    if lo23 <= 0 <= hi23:
        print("     → CI 跨 0：兩者風險調整報酬統計上不可區分，")
        print("       tv 的選擇是純風險偏好，量化上成立。")

    # ── 資訊項：0.20 相對 0.30 的代價 ──
    m30 = results[0.30]['m']
    print("\n" + "=" * 64)
    print("  資訊項：tv=0.20 相對 tv=0.30 的代價（風險偏好的價格）")
    print("=" * 64)
    print(f"  ΔCAGR   {m20['cagr'] - m30['cagr']:>+8.1%}   （犧牲的年化報酬）")
    print(f"  ΔMDD    {m20['mdd'] - m30['mdd']:>+8.1%}   （改善的最大回檔，正=較淺）")
    print(f"  ΔSharpe {m20['sharpe'] - m30['sharpe']:>+8.3f}")

    yearly_table(results[0.20]['dr'], dates, 'VolTgt tv=0.20')
    yearly_table(results[0.30]['dr'], dates, 'VolTgt tv=0.30')

    # ── 輸出 CSV ──
    rows = [{'strategy': 'SPY_BH', **spy_m}]
    for tv in TV_GRID:
        rows.append({'strategy': f'voltgt_{tv:.2f}', **results[tv]['m']})
    out = pd.DataFrame(rows)
    out.to_csv('validate_tv_results.csv', index=False, float_format='%.4f')
    print("\n  已輸出 validate_tv_results.csv")


if __name__ == '__main__':
    main()

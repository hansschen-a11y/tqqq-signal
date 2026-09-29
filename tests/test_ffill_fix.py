"""重播 2026-09-28 事件：yfinance 9/28 列 QQQ/TQQQ NaN（VIX 有值），第二來源有真 9/28。"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tqqq_signal as ts

rng = np.random.default_rng(7)
n = 260
idx = pd.bdate_range(end='2026-09-25', periods=n)
q_ret = rng.normal(0.0008, 0.005, n)
# 植入真實事件：倒數第 5 個交易日 (9/21) QQQ +2.77%
q_ret[-5] = 0.0277
qqq = 600 * np.cumprod(1 + q_ret)
tqqq = 60 * np.cumprod(1 + 3 * q_ret + rng.normal(0, 0.002, n))
vix = np.full(n, 18.0)
raw = pd.DataFrame({'QQQ': qqq, 'TQQQ': tqqq, 'VIX': vix}, index=idx)

# yfinance 回傳：多一列 9/28，主 ticker NaN、VIX 有值
yf_raw = raw.copy()
yf_raw.loc[pd.Timestamp('2026-09-28')] = [np.nan, np.nan, 19.0]
yf_raw.columns = ['QQQ', 'TQQQ', '^VIX']

# 第二來源：含真 9/28（QQQ -0.92%）
r28 = -0.0092
ref_q = pd.concat([raw['QQQ'], pd.Series([raw['QQQ'].iloc[-1] * (1 + r28)], index=[pd.Timestamp('2026-09-28')])])
ref_t = pd.concat([raw['TQQQ'], pd.Series([raw['TQQQ'].iloc[-1] * (1 + 3 * r28)], index=[pd.Timestamp('2026-09-28')])])
ref_v = pd.Series(vix, index=idx)

def fake_ref(days=45, symbol='TQQQ', timeout=12):
    return {'TQQQ': ref_t, 'QQQ': ref_q, 'VIX': ref_v}[symbol].tail(days)
ts.fetch_reference_closes = fake_ref
ts._us_today = lambda: pd.Timestamp('2026-09-28').date()

# ── 舊行為（ffill）對照 ──
old = yf_raw.copy(); old.columns = ['QQQ', 'TQQQ', 'VIX']; old = old.ffill().dropna()
print('舊 fetch_data 最後日期:', old.index[-1].date(), ' TQQQ 最後兩筆相同:', old['TQQQ'].iloc[-1] == old['TQQQ'].iloc[-2])

# ── 新行為 ──
new = ts.clean_closes(yf_raw)
print('新 clean_closes 最後日期:', new.index[-1].date())
sig = ts.compute_tqqq_signal(new, {})
assert 'error' not in sig, sig
print('date:', sig['date'], '| backfilled:', sig['backfilled_dates'], '| status:', sig['data_status'])
print('RV20 %.1f%% 倉位 %d%% | worst %s tracks_qqq=%s single_point=%s' % (
    sig['rv20'], sig['position_pct'], sig['dq_worst_date'],
    sig['dq_worst_date_tracks_qqq'], sig['dq_single_point_dominated']))
print('shadow:', sig['shadow_decision'])
assert sig['date'] == '2026-09-28'
assert sig['backfilled_dates'] == ['2026-09-28']
assert sig['dq_worst_date'] == '2026-09-21' and sig['dq_worst_date_tracks_qqq'] is True
assert not sig['dq_single_point_dominated']
assert not (sig['shadow_decision'] and sig['shadow_decision']['action'] == 'use_rv20_drop1')

# ── 情境 2：yfinance 自己回傳重複收盤（真 ffill 尾巴）→ drop_ffill_tail 要砍 ──
dup = new.copy(); dup.loc[pd.Timestamp('2026-09-28')] = dup.iloc[-1].values
sig2 = ts.compute_tqqq_signal(dup, {})
print('情境2 ffill_dropped:', sig2['ffill_dropped_dates'], '| backfilled:', sig2['backfilled_dates'], '| date:', sig2['date'])
assert sig2['ffill_dropped_dates'] == ['2026-09-28'] and sig2['backfilled_dates'] == ['2026-09-28']

# ── 情境 3：真髒資料（TQQQ 單日 -15% 而 QQQ 正常）→ 仍要被判 corrected ──
dirty = new.copy(); dirty.iloc[-3, dirty.columns.get_loc('TQQQ')] *= 0.85
sig3 = ts.compute_tqqq_signal(dirty, {})
print('情境3 verdict:', sig3['review_verdict'], '| status:', sig3['data_status'], '| bad:', list(sig3['review_bad_dates']))
assert sig3['review_verdict'] == 'corrected' and sig3['data_status'] == 'ok_corrected'

print('\n' + ts.format_message(sig, sig['date']))
print('\nALL PASS')

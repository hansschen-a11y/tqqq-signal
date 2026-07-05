#!/usr/bin/env python3
"""
tqqq_signal.py 最小回歸測試組（無網路、全 mock、<10s）
======================================================
在 CI（push 時 + 每日訊號前）跑，任何一項失敗 exit 1 → 擋下廣播。
涵蓋：語法/函式清單、config 預設、乾淨日、單日髒自動修正、
hard_reject 暫停、stale 整組 backfill 一致性、canary、訊息格式化。
"""
import ast
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "tqqq_signal.py")

PASS = 0
def ok(name, cond, detail=""):
    global PASS
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if cond:
        PASS += 1
    else:
        raise AssertionError(f"{name}: {detail}")


# ── 1. 語法 + 函式清單（防誤刪 fetch_data 之類的歷史事故）──
src_text = open(SRC).read()
tree = ast.parse(src_text)
funcs = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
REQUIRED = {
    "load_config", "_us_today", "realized_vol", "data_quality_report",
    "freshness_check", "_halt_payload", "fetch_reference_closes",
    "_fetch_twelvedata", "_fetch_stooq", "auto_review", "backfill_frame",
    "compute_shadow_decision", "fetch_data", "compute_tqqq_signal",
    "format_message", "send_line_message", "upload_to_github", "main",
}
ok("語法可解析", True)
missing = REQUIRED - funcs
ok("關鍵函式齊全", not missing, f"缺 {missing}")

# ── 2. 載入模組 + config 預設 ──
spec = importlib.util.spec_from_file_location("t", SRC)
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)
ok("target_vol=0.20", abs(t.TQQQ_TARGET_VOL - 0.20) < 1e-9, f"got {t.TQQQ_TARGET_VOL}")
ok("csp_delta=-0.20", abs(t.CSP_TARGET_DELTA + 0.20) < 1e-9, f"got {t.CSP_TARGET_DELTA}")
ok("AUTO_CORRECT 預設 True", t.AUTO_CORRECT is True)
ok("AUTO_DECIDE 預設 False", t.AUTO_DECIDE is False)
ok("MAX_BACKFILL_DAYS=2", t.MAX_BACKFILL_DAYS == 2, f"got {t.MAX_BACKFILL_DAYS}")

# ── 共用 mock 資料 ──
N = 260
TODAY = pd.Timestamp(t._us_today())
COLS = pd.MultiIndex.from_product([["Close"], ["QQQ", "TQQQ", "^VIX"]])
QRET = np.random.RandomState(4).normal(0.0003, 0.011, N + 5)
IDX_FULL = pd.bdate_range(end=TODAY, periods=N + 5)
Q_FULL = pd.Series(500 * np.cumprod(1 + QRET), index=IDX_FULL)
T_FULL = pd.Series(60 * np.cumprod(1 + 3 * QRET), index=IDX_FULL)
V_FULL = pd.Series(np.full(N + 5, 16.5), index=IDX_FULL)

def frame_from(idx, tqqq_override=None):
    q = Q_FULL[idx].values
    tq = T_FULL[idx].values if tqqq_override is None else tqqq_override
    return pd.DataFrame(np.column_stack([q, tq, np.full(len(idx), 16.5)]),
                        index=idx, columns=COLS)

def refs(symbol="TQQQ", days=45, timeout=12):
    return {"TQQQ": T_FULL, "QQQ": Q_FULL, "VIX": V_FULL}.get(symbol)

# ── 3. 乾淨日：ok + canary ok ──
idx = IDX_FULL[5:]
t.yf.download = lambda *a, **k: frame_from(idx)
t.fetch_reference_closes = refs
sig = t.compute_tqqq_signal(t.fetch_data(), {})
ok("乾淨日 data_status=ok", sig["data_status"] == "ok", sig["data_status"])
ok("乾淨日 canary=ok", sig.get("ref_canary") == "ok", sig.get("ref_canary"))
ok("倉位在 (0,100]", 0 < sig["position_pct"] <= 100, sig["position_pct"])
m = t.format_message(sig, sig["date"])
ok("乾淨日訊息無 canary 警示", "🩺" not in m)

# ── 4. canary：第二來源掛掉 → unavailable，但訊號照發 ──
t.fetch_reference_closes = lambda *a, **k: None
sig = t.compute_tqqq_signal(t.fetch_data(), {})
ok("備援掛 → canary=unavailable", sig.get("ref_canary") == "unavailable",
   sig.get("ref_canary"))
ok("備援掛 → 訊號照發（不 halt）", "error" not in sig)
m = t.format_message(sig, sig["date"])
ok("備援掛 → 訊息含 🩺 警示", "🩺" in m)
t.fetch_reference_closes = refs

# ── 5. 單日髒 → auto_correct 修正 ──
tq_dirty = T_FULL[idx].values.copy()
tq_dirty[-6] = tq_dirty[-7] * 1.20
t.yf.download = lambda *a, **k: frame_from(idx, tq_dirty)
sig = t.compute_tqqq_signal(t.fetch_data(), {})
ok("單日髒 → ok_corrected", sig["data_status"] == "ok_corrected", sig["data_status"])

# ── 6. hard_reject（真崩盤、兩邊一致）→ AUTO_DECIDE=False 暫停＋影子決策 ──
tq_crash = T_FULL[idx].values.copy()
tq_crash[-5] = tq_crash[-6] * 1.40
crash_ref = pd.Series(tq_crash, index=idx)
t.yf.download = lambda *a, **k: frame_from(idx, tq_crash)
t.fetch_reference_closes = (
    lambda symbol="TQQQ", days=45, timeout=12:
    crash_ref if symbol == "TQQQ" else refs(symbol))
sig = t.compute_tqqq_signal(t.fetch_data(), {})
ok("hard_reject → halted", "error" in sig and sig.get("data_status") == "halted",
   str(sig.get("data_status")))
ok("halted 附影子決策", sig.get("shadow_decision") is not None)
t.fetch_reference_closes = refs

# ── 7. stale 2 天 → 整組 backfill，日期一致 ──
idx_stale = IDX_FULL[:-2]
t.yf.download = lambda *a, **k: frame_from(idx_stale)
sig = t.compute_tqqq_signal(t.fetch_data(), {})
ok("stale → ok_backfilled", sig["data_status"] == "ok_backfilled", sig["data_status"])
ok("backfill 後 date=第二來源最新日", sig["date"] == sig["ref_latest_date"],
   f"{sig['date']} vs {sig['ref_latest_date']}")
m = t.format_message(sig, sig["date"])
ok("backfill 訊息標題日期一致", sig["date"] in m)

# ── 8. stale 5 天 → 超上限不補 → needs_review ──
idx_stale5 = IDX_FULL[:-5]
t.yf.download = lambda *a, **k: frame_from(idx_stale5)
sig = t.compute_tqqq_signal(t.fetch_data(), {})
ok("超上限 → 不補 + needs_review",
   sig["backfill_capped"] and sig["data_status"] == "needs_review",
   f"capped={sig['backfill_capped']} status={sig['data_status']}")

print(f"\n全部 {PASS} 項通過 ✅")

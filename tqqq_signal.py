#!/usr/bin/env python3
"""
TQQQ 每日訊號 — Variant A (Vol Targeting) + CSP 建議
====================================================
GitHub Actions 每日自動執行：
  1. 計算訊號
  2. LINE Messaging API 廣播給所有好友
  3. 上傳 JSON 到 GitHub repo（供 Claude 即時讀取）

CSP strike 使用 Black-Scholes 計算 delta -0.35 的 put strike。

環境變數（GitHub Secrets）：
  LINE_CHANNEL_ACCESS_TOKEN  — LINE Developers 的 Channel access token
  GH_PAT                     — GitHub Personal Access Token (contents write)
  GITHUB_REPO                — 自動帶入 (github.repository)

用法：
  python tqqq_signal.py                    # 印出訊號
  python tqqq_signal.py --line             # 推送 LINE
  python tqqq_signal.py --upload           # 上傳 JSON 到 GitHub
  python tqqq_signal.py --line --upload    # 兩個都做
"""

import argparse
import datetime
import json
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import pandas as pd
import yfinance as yf

# ═══════════════════════════════════════════════════════════
# 設定
# ═══════════════════════════════════════════════════════════

LINE_CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "")

GITHUB_TOKEN = os.environ.get("GH_PAT", "")
GITHUB_REPO  = os.environ.get("GITHUB_REPO", "")
GITHUB_PATH  = "data/latest_signal.json"

# ── 從 config.json 讀取可變參數 ──
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

def load_config():
    defaults = {
        "target_vol": 0.20,          # FIX: 原本 0.25，與實際部署值不符（silent failure 風險）
        "csp_target_delta": -0.20,   # FIX: 原本 -0.35，與實際部署值不符
        "csp_expiry_days": 30,
        "iv_premium_mult": 1.15,
        "tqqq_iv_mult": 3.2,
        "min_iv": 0.55,
        "rf_annual": 0.045,
        "dq_warn_thr": 0.20,         # 單日|報酬|警戒閾值：超過就標記＋在訊息附註
        "dq_reject_thr": 0.30,       # 單日|log報酬|硬拒絕閾值(約±35%簡單報酬)：暫停推播
        "dq_track_tol": 0.03,        # TQQQ vs 3×QQQ 單日追蹤誤差警戒(3pp，warn-only)
        "auto_review": True,         # warn/hard_reject/stale 時，自動抓第二來源(Stooq)覆核
        "auto_correct": True,        # 資料層自動：單日髒資料自動修正、stale 自動用 Stooq 補最新日
                                      #   （屬資料清洗不改策略；不放心可設 False 只報告不動）
        "auto_decide": False,        # 決策層：ambiguous/hard_reject 是否『真的自動處置』
                                      #   預設 False = 只輸出 dry-run 影子決策供觀察，不執行
        "ref_tol": 0.03,             # 第二來源比對容忍(單日報酬差 >3pp 視為不一致)
        "max_backfill_days": 2,      # stale 自動補最新日的上限；超過視為 yfinance 嚴重故障 → needs_review
    }
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                cfg = json.load(f)
            defaults.update(cfg)
        except Exception as e:
            print(f"⚠️  讀取 config.json 失敗，使用預設值: {e}")
    return defaults

_cfg = load_config()
TQQQ_TARGET_VOL    = _cfg["target_vol"]
CSP_TARGET_DELTA   = _cfg["csp_target_delta"]
CSP_EXPIRY_DAYS    = _cfg["csp_expiry_days"]
IV_PREMIUM_MULT    = _cfg["iv_premium_mult"]
TQQQ_IV_MULT       = _cfg["tqqq_iv_mult"]
MIN_IV             = _cfg["min_iv"]
RF_ANNUAL          = _cfg["rf_annual"]
DQ_WARN_THR        = _cfg.get("dq_warn_thr", 0.20)
DQ_REJECT_THR      = _cfg.get("dq_reject_thr", 0.30)
DQ_TRACK_TOL       = _cfg.get("dq_track_tol", 0.03)
AUTO_REVIEW        = _cfg.get("auto_review", True)
AUTO_CORRECT       = _cfg.get("auto_correct", True)
AUTO_DECIDE        = _cfg.get("auto_decide", False)
REF_TOL            = _cfg.get("ref_tol", 0.03)
MAX_BACKFILL_DAYS  = _cfg.get("max_backfill_days", 2)

STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tqqq_state.json")


def _us_today():
    """美東(NYSE 所在)今天日期，避免 GitHub Actions 用 UTC 造成的日期時差誤報。"""
    try:
        from zoneinfo import ZoneInfo
        return datetime.datetime.now(ZoneInfo("America/New_York")).date()
    except Exception:
        return datetime.date.today()


# ═══════════════════════════════════════════════════════════
# Black-Scholes 定價引擎
# ═══════════════════════════════════════════════════════════

def bs_d1(S, K, T, r, sigma):
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

def bs_price(S, K, T, r, sigma, option_type='put'):
    if T <= 1e-8 or sigma <= 1e-8:
        if option_type == 'put':
            return max(0.0, K - S)
        return max(0.0, S - K)
    d1 = bs_d1(S, K, T, r, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'put':
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_delta(S, K, T, r, sigma, option_type='put'):
    if T <= 1e-8 or sigma <= 1e-8:
        if option_type == 'put':
            return -1.0 if S < K else 0.0
        return 1.0 if S > K else 0.0
    d1 = bs_d1(S, K, T, r, sigma)
    if option_type == 'put':
        return norm.cdf(d1) - 1.0
    return norm.cdf(d1)

def find_strike_for_delta(S, T, r, sigma, target_delta):
    """找 put strike 使得 BS delta = target_delta（負值）。"""
    if T <= 1e-8:
        return S
    def objective(K):
        return bs_delta(S, K, T, r, sigma, 'put') - target_delta
    try:
        return brentq(objective, S * 0.3, S * 1.5, xtol=0.01)
    except Exception:
        return round(S * 0.90, 2)  # fallback ~10% OTM


# ═══════════════════════════════════════════════════════════
# IV 估算
# ═══════════════════════════════════════════════════════════

def estimate_iv(rv20, vix=None):
    """
    估算 TQQQ implied volatility。
    方法：RV20 × IV premium multiplier，VIX × 3.2 做下限。
    """
    iv = rv20 * IV_PREMIUM_MULT
    if vix is not None and not np.isnan(vix):
        vix_floor = (vix / 100.0) * TQQQ_IV_MULT
        iv = max(iv, vix_floor)
    iv = max(iv, MIN_IV)
    return iv


# ═══════════════════════════════════════════════════════════
# 已實現波動率 + 資料品質檢查
# ═══════════════════════════════════════════════════════════

def realized_vol(prices, window=20, ann=252, use_log=True):
    """
    年化已實現波動率。
    預設：對數報酬 ln(P_t/P_{t-1})、樣本標準差(ddof=1)、×√252。
    （與 Barchart 對數定義一致；年化因子統一用 252，模型內部才不會各說各話。）
    """
    p = prices.dropna()
    if len(p) < window + 1:
        return float('nan')
    if use_log:
        r = np.log(p / p.shift(1)).dropna()
    else:
        r = p.pct_change().dropna()
    if len(r) < window:
        return float('nan')
    return float(r.iloc[-window:].std(ddof=1) * np.sqrt(ann))


def data_quality_report(tqqq, qqq=None, warn_thr=DQ_WARN_THR,
                        reject_thr=DQ_REJECT_THR, track_tol=DQ_TRACK_TOL):
    """
    資料品質健檢。回傳 flags 供訊號決定是否照常推播。

    四道防線（全部 warn-only，唯 hard_reject 會暫停推播）：
      1) 硬離群 (hard_reject) —— 單日 |log 報酬| > reject_thr(預設 0.30，約 ±35%
         簡單報酬)。TQQQ 真實單日極端史上約 -20% 上下；超過此值幾乎必為髒資料
         (yfinance 壞 print / 調整瑕疵)，直接暫停推播。
      2) 單點綁架 (single_point_dominated) —— 留一法：把 |報酬| 最大的那一天拿掉
         後重算 RV20，若原始 RV20 > 去一日版 × 1.25，代表整個 RV20 被『一天』撐起來。
         真實的連續多日高波動不會因少一天就崩掉，故不誤殺。
      3) 期限結構參考 (inconsistent_curve) —— RV20 同時 > RV9 且 > RV50 一截，輔助佐證。
      4) 追蹤誤差 (tracking_error) —— TQQQ 單日報酬應 ≈ 3×QQQ 單日報酬(每日重置)。
         若某日 |TQQQ_ret − 3×QQQ_ret| > track_tol(預設 3pp)，代表 TQQQ 那天的收盤價
         與底層 QQQ 對不上 —— 可抓 ffill 假 0%、調整因子多日漂移、單點髒 print。
         乾淨資料實測日追蹤誤差 <1pp，故 3pp 幾乎不誤報。需傳入 qqq 才會啟用。
    """
    logr = np.log(tqqq / tqqq.shift(1)).dropna()
    last20 = logr.iloc[-20:]
    if len(last20) < 20:
        return {"rv9": None, "rv20": None, "rv50": None, "max_abs_ret": None,
                "worst_date": None, "outliers": {}, "rv20_drop1": None,
                "hard_reject": False, "single_point_dominated": False,
                "inconsistent_curve": False, "track_max_resid": None,
                "track_n_bad": 0, "track_worst_date": None, "warn": False}

    abs20 = last20.abs()
    max_abs = float(abs20.max())
    worst_date = abs20.idxmax()

    rv9  = realized_vol(tqqq, 9)
    rv20 = realized_vol(tqqq, 20)
    rv50 = realized_vol(tqqq, 50)

    # 留一法：拿掉最大絕對值那天，用剩下 19 天算 RV20
    kept = last20.drop(worst_date)
    rv20_drop1 = float(kept.std(ddof=1) * np.sqrt(252))
    single_point_dominated = bool(rv20_drop1 > 0 and rv20 > rv20_drop1 * 1.25)

    inconsistent = bool(
        not np.isnan(rv9) and not np.isnan(rv50)
        and rv20 > rv9 * 1.10 and rv20 > rv50 * 1.30
    )

    outliers = last20[abs20 > warn_thr]
    hard_reject = bool(max_abs > reject_thr)

    # ── 追蹤誤差檢查：TQQQ 單日報酬 ≈ 3×QQQ 單日報酬 ──
    track_max_resid = None
    track_n_bad = 0
    track_worst_date = None
    if qqq is not None:
        qr = qqq.pct_change()
        tr = tqqq.pct_change()
        pair = pd.concat([qr, tr], axis=1, keys=['q', 't']).dropna().iloc[-20:]
        if len(pair) >= 5:
            resid = (pair['t'] - 3.0 * pair['q']).abs()
            track_max_resid = round(float(resid.max()), 4)
            bad = resid[resid > track_tol]
            track_n_bad = int(len(bad))
            if track_n_bad > 0:
                track_worst_date = resid.idxmax().strftime('%Y-%m-%d')

    tracking_flag = bool(track_n_bad > 0)
    warn = bool(single_point_dominated or inconsistent
                or len(outliers) > 0 or tracking_flag)

    return {
        "rv9":  round(rv9 * 100, 1)  if not np.isnan(rv9)  else None,
        "rv20": round(rv20 * 100, 1) if not np.isnan(rv20) else None,
        "rv50": round(rv50 * 100, 1) if not np.isnan(rv50) else None,
        "rv20_drop1": round(rv20_drop1 * 100, 1),
        "max_abs_ret": round(max_abs, 4),
        "worst_date": worst_date.strftime('%Y-%m-%d'),
        "outliers": {d.strftime('%Y-%m-%d'): round(float(v), 4) for d, v in outliers.items()},
        "hard_reject": hard_reject,
        "single_point_dominated": single_point_dominated,
        "inconsistent_curve": inconsistent,
        "track_max_resid": track_max_resid,
        "track_n_bad": track_n_bad,
        "track_worst_date": track_worst_date,
        "warn": warn,
    }


def freshness_check(closes, main_tickers=('QQQ', 'TQQQ'), max_business_gap=2):
    """
    資料新鮮度檢查（warn-only → needs_review）。補 tracking check 抓不到的兩種情況：
      - stale：最新交易日距今超過 max_business_gap 個營業日 → 疑資料源多日延遲/中斷。
      - ffill_tail：任一主 ticker 最新收盤與前一日『完全相同』→ ffill 指紋
        (TQQQ 幾乎不可能真的 0.0000% 日報酬)。
    局限：單一交易日的整體延遲需 NYSE 行事曆才能精準判定，此處用營業日 gap 近似（backlog）。
    """
    info = {"stale": False, "ffill_tail": [], "latest_date": None, "busday_gap": None}
    if closes is None or len(closes) < 2:
        return info
    latest = closes.index[-1]
    info["latest_date"] = latest.strftime("%Y-%m-%d")
    try:
        gap = int(np.busday_count(latest.date(), _us_today()))
        info["busday_gap"] = gap
        info["stale"] = bool(gap >= max_business_gap)
    except Exception:
        pass
    for tk in main_tickers:
        if tk in closes.columns and len(closes[tk]) >= 2:
            if float(closes[tk].iloc[-1]) == float(closes[tk].iloc[-2]):
                info["ffill_tail"].append(tk)
    return info


def _halt_payload(msg, review=None, date=None):
    """暫停推播時的結構化回傳：確保 error payload 也帶固定的 data_status/review 欄位。"""
    review = review or {"verdict": "skipped", "detail": "", "bad_dates": {}}
    return {
        "error": msg,
        "date": date,
        "data_status": "halted",
        "review_verdict": review.get("verdict", "skipped"),
        "review_detail": review.get("detail", ""),
        "review_bad_dates": review.get("bad_dates", {}),
        "updated_at": datetime.datetime.utcnow().isoformat() + "Z",
    }


# ═══════════════════════════════════════════════════════════
# 第二來源覆核（自動化人工複核）
# ═══════════════════════════════════════════════════════════

# 第二來源符號對照（中性符號 → 各供應商符號）
_REF_SYMBOLS = {
    "TQQQ": {"td": "TQQQ", "stooq": "tqqq.us"},
    "QQQ":  {"td": "QQQ",  "stooq": "qqq.us"},
    "VIX":  {"td": "VIX",  "stooq": "^vix"},
}


def _fetch_twelvedata(symbol, days, timeout):
    """
    Twelve Data /time_series → pd.Series(close) 或 None。
    需環境變數 TWELVEDATA_API_KEY；沒設 key、額度用盡、symbol 不支援都回 None。
    """
    key = os.environ.get("TWELVEDATA_API_KEY", "")
    if not key:
        return None
    try:
        import requests
        td_sym = _REF_SYMBOLS.get(symbol, {}).get("td", symbol)
        outputsize = min(5000, int(days * 2 + 10))
        resp = requests.get(
            "https://api.twelvedata.com/time_series",
            params={"symbol": td_sym, "interval": "1day",
                    "outputsize": outputsize, "apikey": key, "format": "JSON"},
            timeout=timeout)
        if resp.status_code != 200:
            return None
        j = resp.json()
        if not isinstance(j, dict) or j.get("status") == "error" or "values" not in j:
            return None                      # 額度用盡/symbol 不支援 → 退回 fallback
        vals = j["values"]
        if not vals or len(vals) < 10:
            return None
        closes = {pd.to_datetime(v["datetime"]): float(v["close"])
                  for v in vals if v.get("close") not in (None, "")}
        s = pd.Series(closes).sort_index().dropna()
        return s if len(s) >= 10 else None
    except Exception as e:
        print(f"⚠️  Twelve Data 抓取失敗，改用 fallback：{e}")
        return None


def _fetch_stooq(symbol, days, timeout):
    """Stooq CSV → pd.Series(close) 或 None（fallback；runner 上常被擋）。"""
    try:
        import requests, io
        stq = _REF_SYMBOLS.get(symbol, {}).get("stooq", symbol)
        end = _us_today()
        start = end - datetime.timedelta(days=int(days * 2 + 10))
        url = (f"https://stooq.com/q/d/l/?s={stq}"
               f"&d1={start.strftime('%Y%m%d')}&d2={end.strftime('%Y%m%d')}&i=d")
        resp = requests.get(url, timeout=timeout,
                            headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code != 200 or "Date" not in resp.text[:200]:
            return None
        df = pd.read_csv(io.StringIO(resp.text))
        if "Close" not in df.columns or "Date" not in df.columns or len(df) < 10:
            return None
        s = pd.Series(df["Close"].values,
                      index=pd.to_datetime(df["Date"])).sort_index().dropna()
        return s if len(s) >= 10 else None
    except Exception as e:
        print(f"⚠️  Stooq 抓取失敗：{e}")
        return None


def fetch_reference_closes(days=45, symbol="TQQQ", timeout=12):
    """
    第二獨立來源（yfinance 之外）日收盤，供交叉驗證與 stale backfill。
    來源順序：Twelve Data（有 TWELVEDATA_API_KEY 時，雲端可用）→ Stooq（fallback）。
    任何失敗都回 None，絕不拋例外；全失敗時 auto_review 判 'unavailable' → 退回 warn-only。
    """
    s = _fetch_twelvedata(symbol, days, timeout)
    if s is not None:
        return s
    return _fetch_stooq(symbol, days, timeout)


def auto_review(tqqq_yf, qqq, dq, ref_closes=None, tol=REF_TOL,
                track_tol=DQ_TRACK_TOL, max_fix_days=2):
    """
    自動覆核：warn/hard_reject 觸發時，用第二來源逐日比對，判定真波動還是髒資料。

    回傳 dict:
      verdict: 'clean'      —— 20 天全部與第二來源一致 → RV 為真，不動倉位
               'corrected'  —— 1~2 天『同時與 Stooq 分歧 且 偏離 3×QQQ』(yfinance 髒)
                               → 用第二來源修正該日，附 corrected_tqqq
               'ambiguous'  —— 多日不一致 / 只與 Stooq 分歧但仍貼 3×QQQ(疑調整基準差異)
                               → 不自動修，保留 warn 交人工
               'unavailable'—— 第二來源抓不到 → 保留 warn（等同原行為）
      detail / corrected_tqqq / bad_dates

    第 4 點防呆：只修『Stooq 分歧 ∩ 偏離 3×QQQ』的交集。拆股/配息調整基準不同造成的
    假分歧，yfinance 那天仍會貼 3×QQQ(拆股不改經濟報酬)，故不會被誤修。
    """
    if ref_closes is None:
        ref_closes = fetch_reference_closes()
    if ref_closes is None or len(ref_closes) < 10:
        return {"verdict": "unavailable", "detail": "第二來源不可用",
                "corrected_tqqq": None, "bad_dates": {}}

    yf_ret = tqqq_yf.pct_change()
    ref_ret = ref_closes.pct_change()
    q_ret = qqq.pct_change() if qqq is not None else None

    pair = pd.concat([yf_ret, ref_ret], axis=1, keys=["yf", "ref"]).dropna().iloc[-20:]
    if len(pair) < 10:
        return {"verdict": "unavailable", "detail": "第二來源重疊天數不足",
                "corrected_tqqq": None, "bad_dates": {}}

    diff = (pair["yf"] - pair["ref"]).abs()
    ref_bad = diff[diff > tol]                       # 與 Stooq 分歧的日子

    if len(ref_bad) == 0:
        return {"verdict": "clean",
                "detail": f"第二來源逐日一致(最大日報酬差 {diff.max()*100:.1f}pp)",
                "corrected_tqqq": None, "bad_dates": {}}

    # 交集：同時偏離 3×QQQ 才算 yfinance 真髒（排除調整基準假分歧）
    if q_ret is not None:
        track_resid = (pair["yf"] - 3.0 * q_ret.reindex(pair.index)).abs()
        correctable = ref_bad.index[track_resid.reindex(ref_bad.index) > track_tol]
    else:
        correctable = ref_bad.index

    bad_dates = {d.strftime("%Y-%m-%d"): (round(float(pair["yf"][d]), 4),
                                          round(float(pair["ref"][d]), 4))
                 for d in ref_bad.index}

    if len(correctable) == 0:
        return {"verdict": "ambiguous",
                "detail": (f"{len(ref_bad)} 天與第二來源分歧，但 yfinance 仍貼 3×QQQ"
                           f"(疑調整基準差異，非髒資料)，不自動修正"),
                "corrected_tqqq": None, "bad_dates": bad_dates}

    if len(correctable) > max_fix_days:
        return {"verdict": "ambiguous",
                "detail": f"{len(correctable)} 天疑髒(疑整段偏移)，超過修正上限，不自動修正",
                "corrected_tqqq": None, "bad_dates": bad_dates}

    # 單日/雙日髒資料：用第二來源的日報酬修正 yfinance 該日收盤（base 無關）
    corrected = tqqq_yf.copy()
    for d in correctable:
        pos = corrected.index.get_loc(d)
        if pos <= 0:
            continue
        prev_close = corrected.iloc[pos - 1]
        corrected.iloc[pos] = prev_close * (1.0 + float(ref_ret[d]))
    fixed = [d.strftime("%Y-%m-%d") for d in correctable]
    return {"verdict": "corrected",
            "detail": f"{len(correctable)} 天同時偏離 Stooq 與 3×QQQ，已用第二來源修正：{fixed}",
            "corrected_tqqq": corrected, "bad_dates": bad_dates}


def backfill_frame(closes, refs, max_days=2):
    """
    資料層：yfinance 延遲時，用第二來源補齊 closes 尾端缺的交易日（整個 frame，
    非只 TQQQ），使 date / qqq_price / sma200 / regime / tqqq / rv20 全部一致。
    refs: {col: ref_close_series 或 None}。以各欄 ref 的日報酬 chain 到 frame 現值（base 無關）。
    規則：
      - 缺的交易日數 > max_days → 不補（yfinance 疑嚴重故障，交 needs_review）。
      - 主 ticker(QQQ, TQQQ) 任一日補不了 → 整段放棄（避免半套 backfill 造成不一致）。
      - VIX 無 ref → carry forward（僅影響 iv_est floor，屬進階項），記入 carried。
    回傳 (extended_closes, added_dates, carried_cols(list), capped(bool))
    """
    added, carried, capped = [], set(), False
    if closes is None or len(closes) < 1:
        return closes, added, sorted(carried), capped
    last = closes.index[-1]
    future = set()
    for s in refs.values():
        if s is not None and len(s):
            future |= {d for d in s.index if d > last}
    if not future:
        return closes, added, sorted(carried), capped
    missing = sorted(future)
    if len(missing) > max_days:
        return closes, added, sorted(carried), True     # 超過上限，不補
    ext = closes.copy()
    for d in missing:
        row = {}
        ok = True
        for col in closes.columns:
            s = refs.get(col)
            prev = float(ext[col].iloc[-1])
            if s is not None and d in s.index:
                loc = s.index.get_loc(d)
                if loc >= 1:
                    r = float(s.iloc[loc] / s.iloc[loc - 1] - 1.0)
                    row[col] = prev * (1.0 + r)
                    continue
            if col in ("QQQ", "TQQQ"):     # 主 ticker 補不了 → 放棄整段
                ok = False
                break
            row[col] = prev                # VIX 等：carry forward
            carried.add(col)
        if not ok:
            return closes, [], [], False
        ext.loc[d] = pd.Series(row)
        added.append(d.strftime("%Y-%m-%d"))
    return ext.sort_index(), added, sorted(carried), capped


def compute_shadow_decision(tqqq, dq, review):
    """
    決策層 dry-run：對 ambiguous / hard_reject 這種目前『交給人』的情況，
    計算『若全自動會怎麼處置』的影子決策供觀察，但預設不執行（AUTO_DECIDE 控制）。
    回傳 dict(action, rv20, position_pct, detail) 或 None。
    """
    v = review.get("verdict")
    action = None
    shadow_rv20 = None
    detail = ""
    if dq["hard_reject"]:
        if v == "clean":
            action = "proceed_de_risk"
            shadow_rv20 = realized_vol(tqqq, 20)
            detail = "第二來源確認真實極端波動 → 全自動會照發 de-risk 訊號"
        elif v == "corrected" and review.get("corrected_tqqq") is not None:
            action = "use_stooq_corrected"
            shadow_rv20 = realized_vol(review["corrected_tqqq"], 20)
            detail = "全自動會用第二來源修正後重算"
        else:
            action = "use_rv20_drop1"
            shadow_rv20 = (dq["rv20_drop1"] / 100.0) if dq.get("rv20_drop1") else None
            detail = "無法交叉驗證 → 全自動會用剔除離群後 RV20(rv20_drop1)"
    elif v == "ambiguous":
        action = "use_rv20_drop1"
        shadow_rv20 = (dq["rv20_drop1"] / 100.0) if dq.get("rv20_drop1") else None
        detail = "多日分歧無法自動判定 → 全自動會用剔除離群後 RV20(rv20_drop1)"
    if action is None or not shadow_rv20 or shadow_rv20 <= 0:
        return None
    pos = min(1.0, max(0.0, TQQQ_TARGET_VOL / shadow_rv20))
    return {"action": action, "rv20": round(shadow_rv20 * 100, 1),
            "position_pct": round(pos * 100), "detail": detail}


# ═══════════════════════════════════════════════════════════
# 資料
# ═══════════════════════════════════════════════════════════

def fetch_data(retries=3):
    import time
    end = _us_today() + datetime.timedelta(days=1)
    start = end - datetime.timedelta(days=400)
    tickers = ['QQQ', 'TQQQ', '^VIX']
    for attempt in range(retries):
        try:
            data = yf.download(tickers, start=start.strftime('%Y-%m-%d'),
                               end=end.strftime('%Y-%m-%d'),
                               auto_adjust=True, progress=False)
            if isinstance(data.columns, pd.MultiIndex):
                closes = data['Close']
            else:
                closes = data
            closes.columns = [c.replace('^', '') for c in closes.columns]
            closes = closes.ffill().dropna()
            if len(closes) > 0 and 'TQQQ' in closes.columns:
                return closes
            print(f"⚠️  資料不完整，重試 {attempt+1}/{retries}...")
        except Exception as e:
            print(f"⚠️  下載失敗（{e}），重試 {attempt+1}/{retries}...")
        time.sleep(5)
    raise RuntimeError("無法下載資料，已重試 3 次")


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_state(state):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2, default=str)


# ═══════════════════════════════════════════════════════════
# TQQQ 訊號（Variant A + BS-based CSP）
# ═══════════════════════════════════════════════════════════

def compute_tqqq_signal(closes, state):
    qqq = closes['QQQ']
    tqqq = closes['TQQQ']
    sma200 = qqq.rolling(200).mean()

    current_price = qqq.iloc[-1]
    current_sma = sma200.iloc[-1]
    if pd.isna(current_sma):
        return _halt_payload("SMA200 資料不足")

    above = current_price > current_sma

    # ── 資料新鮮度檢查（stale / ffill 尾巴）──
    fresh = freshness_check(closes)
    stale_flag = bool(fresh["stale"] or len(fresh["ffill_tail"]) > 0)

    # ── 資料品質健檢（log/√252 一致；含 3×QQQ 追蹤檢查） ──
    dq = data_quality_report(tqqq, qqq)

    # 第二來源：每天都抓（canary）。出事日拿來覆核/backfill；平常日驗證備援活著且一致，
    # 避免「災難日才第一次測試備援」。抓不到不影響訊號（warn-only）。
    need_review = bool(dq["warn"] or dq["hard_reject"] or stale_flag)
    ref_closes = fetch_reference_closes() if AUTO_REVIEW else None
    yf_latest = tqqq.index[-1].strftime('%Y-%m-%d')
    ref_latest = (ref_closes.index[-1].strftime('%Y-%m-%d')
                  if ref_closes is not None and len(ref_closes) else None)

    # ── canary 健康判定（純診斷，不改任何決策）──
    if ref_closes is None:
        canary = "unavailable"          # 備援掛了：今天就知道，不是等災難日
    else:
        try:
            common = tqqq.index.intersection(ref_closes.index)
            if len(common) >= 2:
                d_last, d_prev = common[-1], common[-2]
                r_yf = float(tqqq.loc[d_last] / tqqq.loc[d_prev] - 1.0)
                r_rf = float(ref_closes.loc[d_last] / ref_closes.loc[d_prev] - 1.0)
                canary = "ok" if abs(r_yf - r_rf) <= REF_TOL else "mismatch"
            else:
                canary = "no_overlap"
        except Exception:
            canary = "error"

    # ── 資料層①：stale 且 Stooq 有更新 → 自動補最新日（整個 frame；auto_correct 控制）──
    backfilled = []
    bf_carried = []
    bf_capped = False
    if stale_flag and AUTO_CORRECT and ref_closes is not None:
        refs = {"TQQQ": ref_closes,
                "QQQ": fetch_reference_closes(symbol="QQQ"),
                "VIX": fetch_reference_closes(symbol="VIX")}
        closes_bf, backfilled, bf_carried, bf_capped = backfill_frame(
            closes, refs, MAX_BACKFILL_DAYS)
        if backfilled:
            closes = closes_bf
            qqq = closes['QQQ']
            tqqq = closes['TQQQ']
            sma200 = qqq.rolling(200).mean()
            current_price = qqq.iloc[-1]
            current_sma = sma200.iloc[-1]
            above = current_price > current_sma
            dq = data_quality_report(tqqq, qqq)             # 補後重新健檢
            fresh = freshness_check(closes)                 # 補後重新評估新鮮度
            stale_flag = bool(fresh["stale"] or len(fresh["ffill_tail"]) > 0)

    # ── 自動覆核：warn / hard_reject / 新鮮度 都用第二來源自我核對 ──
    review = {"verdict": "skipped", "detail": "", "bad_dates": {}}
    correction_applied = False
    if need_review and AUTO_REVIEW:
        review = auto_review(tqqq, qqq, dq, ref_closes=ref_closes)
        # ── 資料層②：單日髒資料自動修正（auto_correct 控制）──
        if review["verdict"] == "corrected" and AUTO_CORRECT \
                and review["corrected_tqqq"] is not None:
            tqqq = review["corrected_tqqq"]
            dq = data_quality_report(tqqq, qqq)
            correction_applied = True

    # ── 決策層：對 ambiguous / hard_reject 算『影子決策』(dry-run) ──
    shadow = compute_shadow_decision(tqqq, dq, review)
    decision_applied = False
    override_rv20 = None
    if AUTO_DECIDE and shadow is not None \
            and (dq["hard_reject"] or review["verdict"] == "ambiguous"):
        override_rv20 = shadow["rv20"] / 100.0     # 決策層 live：套用影子決策
        decision_applied = True

    # ── hard_reject 且未套用決策 → 暫停，附 Stooq 覆核結果與影子決策 ──
    if dq["hard_reject"] and not decision_applied:
        v = review["verdict"]
        note = {"clean": "第二來源確認該日為真實極端波動(兩邊一致)，",
                "corrected": "第二來源顯示 yfinance 該日疑有誤(auto_correct 關閉未修)，",
                "ambiguous": f"第二來源覆核：{review['detail']}，",
                "unavailable": "第二來源不可用無法交叉驗證，"}.get(v, "")
        msg = (f"資料異常：偵測到單日報酬 {dq['max_abs_ret']:+.1%} "
               f"@ {dq['worst_date']}。{note}今日暫停推播，請確認後手動處理。")
        payload = _halt_payload(msg, review, date=closes.index[-1].strftime('%Y-%m-%d'))
        payload["shadow_decision"] = shadow
        payload["ref_latest_date"] = ref_latest
        payload["yf_latest_date"] = yf_latest
        return payload

    # ── 權威資料狀態（下游/JSON 讀這個，不要只讀 dq_warn）──
    if decision_applied:
        data_status = "ok_auto_decided"
    elif correction_applied:
        data_status = "ok_corrected"
    elif dq["warn"] and review["verdict"] == "clean":
        data_status = "ok"
    elif dq["warn"] or dq["hard_reject"]:
        data_status = "needs_review"
    elif backfilled:
        data_status = "ok_backfilled"
    else:
        data_status = "ok"
    # backfill 後 stale_flag 已重算：若仍 stale（含未補成、capped、QQQ ffill）→ needs_review
    if (stale_flag or bf_capped) and data_status in (
            "ok", "ok_corrected", "ok_backfilled", "ok_auto_decided"):
        data_status = "needs_review"

    rv20 = override_rv20 if override_rv20 is not None else realized_vol(tqqq, 20)
    tqqq_price = float(tqqq.iloc[-1])
    vix = float(closes['VIX'].iloc[-1]) if 'VIX' in closes.columns else None

    # Variant A 倉位比例
    raw_pos = min(1.0, max(0.0, TQQQ_TARGET_VOL / rv20)) if rv20 > 0 else 1.0
    position = raw_pos
    cash_pct = 1 - position

    # IV 估算 + BS delta-based CSP strike
    iv = estimate_iv(rv20, vix)
    T = CSP_EXPIRY_DAYS / 365.0
    csp_strike = round(find_strike_for_delta(tqqq_price, T, RF_ANNUAL, iv, CSP_TARGET_DELTA), 2)
    csp_premium = round(bs_price(tqqq_price, csp_strike, T, RF_ANNUAL, iv, 'put'), 2)
    csp_delta = round(bs_delta(tqqq_price, csp_strike, T, RF_ANNUAL, iv, 'put'), 3)
    csp_otm_pct = round((1 - csp_strike / tqqq_price) * 100, 1)
    csp_margin_2x = round(csp_strike * 2 * 100, 0)  # 2x 覆蓋：每張所需閒置現金

    regime = "🟢 牛市" if above else "🔴 熊市"

    return {
        "date": closes.index[-1].strftime('%Y-%m-%d'),
        "regime": regime,
        "asset": "TQQQ",
        "position_pct": round(position * 100),
        "cash_pct": round(cash_pct * 100),
        "tqqq_price": round(tqqq_price, 2),
        "qqq_price": round(float(current_price), 2),
        "sma200": round(float(current_sma), 2),
        "qqq_vs_sma": round((current_price / current_sma - 1) * 100, 2),
        "rv20": round(float(rv20 * 100), 1),
        "rv9": dq["rv9"],
        "rv50": dq["rv50"],
        "dq_warn": dq["warn"],
        "dq_single_point_dominated": dq["single_point_dominated"],
        "dq_inconsistent_curve": dq["inconsistent_curve"],
        "dq_max_abs_ret": dq["max_abs_ret"],
        "dq_rv20_drop1": dq["rv20_drop1"],
        "dq_track_max_resid": dq["track_max_resid"],
        "dq_track_n_bad": dq["track_n_bad"],
        "dq_track_worst_date": dq["track_worst_date"],
        "review_verdict": review["verdict"],
        "review_detail": review["detail"],
        "review_bad_dates": review["bad_dates"],
        "data_status": data_status,
        "backfilled_dates": backfilled,
        "backfill_carried": bf_carried,
        "backfill_capped": bf_capped,
        "ref_latest_date": ref_latest,
        "yf_latest_date": yf_latest,
        "ref_canary": canary,
        "shadow_decision": shadow,
        "decision_applied": decision_applied,
        "dq_stale": bool(fresh["stale"]),
        "dq_ffill_tail": fresh["ffill_tail"],
        "data_latest_date": fresh["latest_date"],
        "data_busday_gap": fresh["busday_gap"],
        "target_vol": TQQQ_TARGET_VOL,
        "iv_est": round(float(iv * 100), 1),
        "vix": round(float(vix), 1) if vix else None,
        "csp_strike": csp_strike,
        "csp_delta": csp_delta,
        "csp_otm_pct": csp_otm_pct,
        "csp_premium": csp_premium,
        "csp_margin_2x": csp_margin_2x,
        "csp_expiry_days": CSP_EXPIRY_DAYS,
        "updated_at": datetime.datetime.utcnow().isoformat() + "Z",
    }


# ═══════════════════════════════════════════════════════════
# 訊息格式化
# ═══════════════════════════════════════════════════════════

def format_message(sig, today):
    if "error" in sig:
        return f"⚠️ {sig['error']}"

    tv_pct = int(sig['target_vol'] * 100)

    msg = f"{'━' * 28}\n"
    msg += f"📊 TQQQ 每日訊號 — {today}\n"
    msg += f"{'━' * 28}\n\n"

    msg += f"🇺🇸 TQQQ Variant A Vol Targeting\n"
    msg += f"{sig['regime']}（僅供參考，不影響倉位）\n"
    msg += f"\nTQQQ ${sig['tqqq_price']} ｜ RV20 {sig['rv20']:.0f}%\n"
    if sig.get('ref_canary') in ('unavailable', 'mismatch', 'error'):
        note = {"unavailable": "第二來源今日抓不到（備援失效，出事日將無交叉驗證）",
                "mismatch": "第二來源與 yfinance 最新日報酬不一致",
                "error": "第二來源 canary 檢查出錯"}[sig['ref_canary']]
        msg += f"🩺 {note}，請留意\n"
    if sig.get('backfilled_dates'):
        carry = f"（{'/'.join(sig['backfill_carried'])} 無第二來源，仍以 yfinance 舊值計）" if sig.get('backfill_carried') else ""
        msg += f"🔧 yfinance 延遲，已用第二來源補整組資料最新日：{'、'.join(sig['backfilled_dates'])}{carry}\n"
    elif sig.get('backfill_capped'):
        msg += f"⚠️ yfinance 延遲超過 {sig.get('data_busday_gap','?')} 營業日（超過自動補值上限），未自動補，請人工複核\n"
    if sig.get('dq_stale') or sig.get('dq_ffill_tail'):
        bits = []
        if sig.get('dq_stale'):
            bits.append(f"最新資料 {sig.get('data_latest_date')}（距今 {sig.get('data_busday_gap')} 營業日）")
        if sig.get('dq_ffill_tail'):
            bits.append(f"{'/'.join(sig['dq_ffill_tail'])} 最新值疑為 ffill")
        if sig.get('ref_latest_date') and sig.get('yf_latest_date') \
                and sig['ref_latest_date'] > sig['yf_latest_date'] and not sig.get('backfilled_dates'):
            bits.append(f"yfinance 最新 {sig['yf_latest_date']} vs 第二來源最新 {sig['ref_latest_date']}（yfinance 延遲）")
        rv = sig.get('review_verdict', 'skipped')
        rnote = {"clean": "，第二來源重疊日一致", "corrected": "，已依第二來源修正",
                 "ambiguous": "，第二來源覆核不確定", "unavailable": "，第二來源不可用"}.get(rv, "")
        if not sig.get('backfilled_dates'):
            msg += f"⚠️ 資料新鮮度提醒：{'，'.join(bits)}{rnote}，請人工複核\n"
    status = sig.get('data_status', 'ok')
    verdict = sig.get('review_verdict', 'skipped')
    parts = []
    if sig.get('dq_single_point_dominated') and sig.get('dq_rv20_drop1') is not None:
        parts.append(f"剔除單日離群後 RV20≈{sig['dq_rv20_drop1']:.0f}%")
    if sig.get('rv9') is not None:
        parts.append(f"RV9 {sig['rv9']:.0f}%/RV50 {sig['rv50']:.0f}%")
    if sig.get('dq_max_abs_ret'):
        parts.append(f"單日最大 {sig['dq_max_abs_ret']*100:+.0f}%")
    if sig.get('dq_track_n_bad'):
        parts.append(f"{sig['dq_track_n_bad']}天偏離3×QQQ")
    detail = "，".join(parts)

    if status == 'ok_corrected':
        bad = "、".join(sig.get('review_bad_dates', {}).keys())
        msg += f"🔧 已自動修正髒資料（{bad}），依第二來源重算：RV20 {sig['rv20']:.0f}%、倉位 {sig['position_pct']}%\n"
    elif status == 'ok_auto_decided':
        sd = sig.get('shadow_decision') or {}
        msg += f"🤖 已自動處置（{sd.get('action','')}）：{sd.get('detail','')}\n"
    elif sig.get('dq_warn'):
        if verdict == 'clean':
            msg += f"✅ 已自動覆核（{detail}）：第二來源逐日一致，RV 為真波動，倉位照跑\n"
        elif verdict == 'corrected':   # 判定為髒但 auto_correct 關閉，未修正
            bad = "、".join(sig.get('review_bad_dates', {}).keys())
            msg += f"⚠️ 偵測到疑似髒資料（{bad}），auto_correct 關閉未自動修正，請人工複核\n"
        elif verdict in ('ambiguous', 'unavailable'):
            why = "第二來源不可用" if verdict == 'unavailable' else "無法自動判定"
            msg += f"⚠️ 資料品質提醒（{detail}）：{why}，請人工複核收盤價\n"
        else:  # skipped（auto_review 關閉）
            msg += f"⚠️ 資料品質提醒：RV20 疑被異常日灌高（{detail}），請人工複核收盤價\n"
    # 決策層 dry-run：顯示『若全自動會怎麼做』(不影響當前倉位)
    sd = sig.get('shadow_decision')
    if sd and not sig.get('decision_applied'):
        msg += f"🔬 dry-run（僅供觀察，未執行）：若全自動→{sd['action']}，RV20 {sd['rv20']:.0f}%、倉位 {sd['position_pct']}%\n"
    msg += f"QQQ ${sig['qqq_price']} vs SMA200 ${sig['sma200']}（{sig['qqq_vs_sma']:+.1f}%）\n"

    msg += f"\n🎯 建議倉位：\n"
    msg += f"  TQQQ {sig['position_pct']}% ／ 現金 {sig['cash_pct']}%\n"
    if sig['cash_pct'] > 5:
        msg += f"  （現金建議 parking 在 BOXX）\n"
    msg += f"  （公式：{tv_pct}% ÷ {sig['rv20']:.0f}% = {sig['position_pct']}%）\n"

    if sig['cash_pct'] > 5:
        msg += f"\n💰 Sell Put 建議（{sig['csp_expiry_days']}天到期）：\n"
        msg += f"  Strike ${sig['csp_strike']}（delta {sig['csp_delta']:.2f}，OTM {sig['csp_otm_pct']:.0f}%）\n"
        msg += f"  預估權利金 ~${sig['csp_premium']}/股\n"
        msg += f"  IV {sig['iv_est']:.0f}%"
        if sig.get('vix'):
            msg += f"（VIX {sig['vix']:.0f}）"
        msg += f"\n  閒置現金每 ${sig['csp_margin_2x']:,.0f} 賣一張（2x 覆蓋）\n"

    msg += f"\n{'━' * 28}"
    return msg


# ═══════════════════════════════════════════════════════════
# LINE Messaging API 廣播
# ═══════════════════════════════════════════════════════════

def send_line_message(msg):
    import requests

    if not LINE_CHANNEL_ACCESS_TOKEN:
        print("⚠️  LINE_CHANNEL_ACCESS_TOKEN 未設定")
        return False

    try:
        resp = requests.post(
            "https://api.line.me/v2/bot/message/broadcast",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}",
            },
            json={
                "messages": [
                    {"type": "text", "text": msg}
                ],
            },
            timeout=10,
        )
        if resp.status_code == 200:
            return True
        else:
            print(f"⚠️  LINE API 回應: {resp.status_code} {resp.text}")
            return False
    except Exception as e:
        print(f"❌ LINE API 錯誤: {e}")
        return False


# ═══════════════════════════════════════════════════════════
# 上傳 JSON 到 GitHub（供 Claude 讀取）
# ═══════════════════════════════════════════════════════════

def upload_to_github(sig):
    import requests
    import base64

    if not GITHUB_TOKEN or not GITHUB_REPO:
        print("⚠️  GH_PAT 或 GITHUB_REPO 未設定")
        return False

    api_url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{GITHUB_PATH}"
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
    }

    sha = None
    try:
        resp = requests.get(api_url, headers=headers, timeout=10)
        if resp.status_code == 200:
            sha = resp.json().get("sha")
    except Exception:
        pass

    content = json.dumps(sig, indent=2, ensure_ascii=False)
    content_b64 = base64.b64encode(content.encode('utf-8')).decode('utf-8')

    payload = {
        "message": f"Update TQQQ signal {sig.get('date', 'unknown')}",
        "content": content_b64,
    }
    if sha:
        payload["sha"] = sha

    try:
        resp = requests.put(api_url, headers=headers, json=payload, timeout=15)
        if resp.status_code in (200, 201):
            raw_url = f"https://raw.githubusercontent.com/{GITHUB_REPO}/main/{GITHUB_PATH}"
            print(f"✅ GitHub 上傳成功: {raw_url}")
            return True
        else:
            print(f"⚠️  GitHub API 回應: {resp.status_code} {resp.text[:200]}")
            return False
    except Exception as e:
        print(f"❌ GitHub 上傳錯誤: {e}")
        return False


# ═══════════════════════════════════════════════════════════
# 主程式
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='TQQQ Daily Signal (Variant A)')
    parser.add_argument('--line', action='store_true', help='推送 LINE')
    parser.add_argument('--upload', action='store_true', help='上傳 JSON 到 GitHub')
    parser.add_argument('--json', action='store_true', help='輸出 JSON')
    args = parser.parse_args()

    print("拉取資料中...")
    closes = fetch_data()
    today = closes.index[-1].strftime('%Y-%m-%d')
    print(f"資料截至：{today}")

    state = load_state()

    print("\n計算 TQQQ 訊號...")
    sig = compute_tqqq_signal(closes, state)

    save_state(state)

    msg = format_message(sig, sig.get('date', today))   # backfill 日以補後日期為準
    print(msg)

    if args.json:
        print("\n" + json.dumps(sig, indent=2, ensure_ascii=False, default=str))

    if args.line:
        if send_line_message(msg):
            print("\n✅ LINE 已發送")
        else:
            print("\n❌ LINE 發送失敗")

    if args.upload:
        upload_to_github(sig)


if __name__ == '__main__':
    main()

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

STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tqqq_state.json")


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


# ═══════════════════════════════════════════════════════════
# 資料
# ═══════════════════════════════════════════════════════════

def fetch_data(retries=3):
    import time
    end = datetime.date.today() + datetime.timedelta(days=1)
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
        return {"error": "SMA200 資料不足"}

    above = current_price > current_sma

    # ── 資料品質健檢（log/√252 一致；含 3×QQQ 追蹤檢查） ──
    dq = data_quality_report(tqqq, qqq)
    if dq["hard_reject"]:
        return {"error": (f"資料異常：偵測到單日報酬 {dq['max_abs_ret']:+.1%} "
                          f"@ {dq['worst_date']}（疑似髒資料），今日暫停推播訊號。"
                          f"請檢查 yfinance 收盤價後手動重跑。")}

    rv20 = realized_vol(tqqq, 20)   # log/√252，與健檢同一套定義
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
    if sig.get('dq_warn'):
        parts = []
        if sig.get('dq_single_point_dominated') and sig.get('dq_rv20_drop1') is not None:
            parts.append(f"剔除單日離群後 RV20≈{sig['dq_rv20_drop1']:.0f}%")
        if sig.get('rv9') is not None:
            parts.append(f"RV9 {sig['rv9']:.0f}%/RV50 {sig['rv50']:.0f}%")
        if sig.get('dq_max_abs_ret'):
            parts.append(f"單日最大 {sig['dq_max_abs_ret']*100:+.0f}%")
        if sig.get('dq_track_n_bad'):
            parts.append(f"{sig['dq_track_n_bad']}天偏離3×QQQ(最差@{sig.get('dq_track_worst_date')})")
        detail = "，".join(parts)
        msg += f"⚠️ 資料品質提醒：RV20 疑被單一異常日灌高（{detail}），請人工複核收盤價\n"
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

    msg = format_message(sig, today)
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

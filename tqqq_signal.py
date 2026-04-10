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
        "target_vol": 0.25,
        "csp_target_delta": -0.35,
        "csp_expiry_days": 30,
        "iv_premium_mult": 1.15,
        "tqqq_iv_mult": 3.2,
        "min_iv": 0.55,
        "rf_annual": 0.045,
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
# 資料
# ═══════════════════════════════════════════════════════════

def fetch_data():
    end = datetime.date.today() + datetime.timedelta(days=1)
    start = end - datetime.timedelta(days=400)
    tickers = ['QQQ', 'TQQQ', '^VIX']
    data = yf.download(tickers, start=start.strftime('%Y-%m-%d'),
                       end=end.strftime('%Y-%m-%d'),
                       auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        closes = data['Close']
    else:
        closes = data
    closes.columns = [c.replace('^', '') for c in closes.columns]
    closes = closes.ffill().dropna()
    return closes


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
    tqqq_ret = tqqq.pct_change().dropna()
    rv20 = tqqq_ret.iloc[-20:].std() * np.sqrt(252)
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

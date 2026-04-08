#!/usr/bin/env python3
"""
TQQQ 每日訊號 — Variant A (Vol Targeting) + CSP 建議
====================================================
GitHub Actions 每日自動執行：
  1. 計算訊號
  2. LINE Messaging API 推播
  3. 上傳 JSON 到 GitHub repo（供 Claude 即時讀取）

環境變數（GitHub Secrets）：
  LINE_CHANNEL_ACCESS_TOKEN  — LINE Developers 的 Channel access token
  LINE_USER_ID               — Basic settings 裡的 Your user ID (U 開頭)
  GH_PAT                     — GitHub Personal Access Token (contents write)
  GITHUB_REPO                — 自動帶入 (github.repository)
  TOTAL_CAPITAL_USD           — 總資金，預設 155000

用法：
  python tqqq_signal.py                    # 印出訊號
  python tqqq_signal.py --line             # 推送 LINE
  python tqqq_signal.py --upload           # 上傳 JSON 到 GitHub
  python tqqq_signal.py --line --upload    # 兩個都做
  python tqqq_signal.py --capital 155000   # 自訂資金
"""

import argparse
import datetime
import json
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf

# ═══════════════════════════════════════════════════════════
# 設定（全部從環境變數讀取，GitHub Secrets 管理）
# ═══════════════════════════════════════════════════════════

LINE_CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_USER_ID = os.environ.get("LINE_USER_ID", "")

GITHUB_TOKEN = os.environ.get("GH_PAT", "")
GITHUB_REPO  = os.environ.get("GITHUB_REPO", "")
GITHUB_PATH  = "data/latest_signal.json"

TOTAL_CAPITAL_USD = float(os.environ.get("TOTAL_CAPITAL_USD", "155000"))

TQQQ_TARGET_VOL = 0.30
TQQQ_CSP_OTM = 0.85
TQQQ_CSP_PREMIUM_EST = 0.027

STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tqqq_state.json")


# ═══════════════════════════════════════════════════════════
# 資料
# ═══════════════════════════════════════════════════════════

def fetch_data():
    end = datetime.date.today() + datetime.timedelta(days=1)
    start = end - datetime.timedelta(days=400)
    tickers = ['QQQ', 'TQQQ', 'BOXX', 'GLD']
    data = yf.download(tickers, start=start.strftime('%Y-%m-%d'),
                       end=end.strftime('%Y-%m-%d'),
                       auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        closes = data['Close']
    else:
        closes = data
    closes = closes.ffill().dropna()
    return closes


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return {"tqqq_last_asset": "TQQQ", "tqqq_days_above": 0}


def save_state(state):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2, default=str)


# ═══════════════════════════════════════════════════════════
# TQQQ 訊號（Variant A + CSP 建議）
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
    tqqq_price = tqqq.iloc[-1]

    raw_pos = min(1.0, max(0.0, TQQQ_TARGET_VOL / rv20)) if rv20 > 0 else 1.0
    position = raw_pos
    cash_pct = 1 - position

    shares = int((TOTAL_CAPITAL_USD * position) / tqqq_price) if tqqq_price > 0 else 0
    cash_for_put = TOTAL_CAPITAL_USD * cash_pct

    csp_strike = round(tqqq_price * TQQQ_CSP_OTM, 2)
    csp_contracts = int(cash_for_put / (csp_strike * 100)) if csp_strike > 0 else 0
    est_premium = round(tqqq_price * TQQQ_CSP_PREMIUM_EST * max(csp_contracts, 0), 0)

    mom_boxx = closes['BOXX'].iloc[-1] / closes['BOXX'].iloc[-63] - 1 if len(closes) >= 63 else 0
    mom_gld = closes['GLD'].iloc[-1] / closes['GLD'].iloc[-63] - 1 if len(closes) >= 63 else 0

    regime = "🟢 牛市" if above else "🔴 熊市"

    return {
        "date": closes.index[-1].strftime('%Y-%m-%d'),
        "regime": regime,
        "asset": "TQQQ",
        "position_pct": round(position * 100),
        "shares": shares,
        "cash_pct": round(cash_pct * 100),
        "tqqq_price": round(float(tqqq_price), 2),
        "qqq_price": round(float(current_price), 2),
        "sma200": round(float(current_sma), 2),
        "qqq_vs_sma": round((current_price / current_sma - 1) * 100, 2),
        "rv20": round(float(rv20 * 100), 1),
        "capital": TOTAL_CAPITAL_USD,
        "target_vol": TQQQ_TARGET_VOL,
        "csp_strike": csp_strike,
        "csp_contracts": csp_contracts,
        "csp_est_premium": float(est_premium),
        "mom_boxx": round(float(mom_boxx * 100), 2),
        "mom_gld": round(float(mom_gld * 100), 2),
        "updated_at": datetime.datetime.utcnow().isoformat() + "Z",
    }


# ═══════════════════════════════════════════════════════════
# 訊息格式化
# ═══════════════════════════════════════════════════════════

def format_message(sig, today):
    if "error" in sig:
        return f"⚠️ {sig['error']}"

    msg = f"{'━' * 28}\n"
    msg += f"📊 TQQQ 每日訊號 — {today}\n"
    msg += f"{'━' * 28}\n\n"

    msg += f"🇺🇸 TQQQ（Variant A Vol Targeting）\n"
    msg += f"{sig['regime']}（僅供參考，不影響倉位）\n"
    msg += f"QQQ ${sig['qqq_price']}（vs SMA200 ${sig['sma200']}，{sig['qqq_vs_sma']:+.1f}%）\n"
    msg += f"TQQQ 20日波動率：{sig['rv20']:.0f}%\n"

    msg += f"\n🎯 倉位配置（${sig['capital']:,.0f}）：\n"
    msg += f"  TQQQ {sig['position_pct']}%（{sig['shares']}股 × ${sig['tqqq_price']}）\n"
    msg += f"  現金 {sig['cash_pct']}%\n"

    if sig.get('csp_contracts', 0) > 0:
        msg += f"\n💰 CSP 建議：\n"
        msg += f"  Sell {sig['csp_contracts']}張 TQQQ Put\n"
        msg += f"  Strike ${sig['csp_strike']}（~15% OTM）\n"
        msg += f"  月到期，預估權利金 ~${sig['csp_est_premium']:.0f}\n"

    msg += f"\n📊 替代資產動量：\n"
    msg += f"  BOXX {sig['mom_boxx']:+.1f}% / GLD {sig['mom_gld']:+.1f}%\n"
    msg += f"\n{'━' * 28}"
    return msg


# ═══════════════════════════════════════════════════════════
# LINE Messaging API 推播（push message）
# ═══════════════════════════════════════════════════════════

def send_line_message(msg):
    """
    透過 LINE Messaging API push message 發送。
    文件：https://developers.line.biz/en/reference/messaging-api/#send-push-message
    免費方案每月 200 則，每日 1 則綽綽有餘。
    """
    import requests

    if not LINE_CHANNEL_ACCESS_TOKEN or not LINE_USER_ID:
        print("⚠️  LINE_CHANNEL_ACCESS_TOKEN 或 LINE_USER_ID 未設定")
        return False

    try:
        resp = requests.post(
            "https://api.line.me/v2/bot/message/push",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}",
            },
            json={
                "to": LINE_USER_ID,
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
    parser.add_argument('--capital', type=float, default=None, help='總資金 USD')
    parser.add_argument('--json', action='store_true', help='輸出 JSON')
    args = parser.parse_args()

    if args.capital:
        global TOTAL_CAPITAL_USD
        TOTAL_CAPITAL_USD = args.capital

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

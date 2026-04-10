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

# ── 從 config.json 讀取可變參數（金額、目標波動率等）──
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

def load_config():
    """讀取 config.json，找不到就用預設值。"""
    defaults = {"target_vol": 0.30, "csp_otm": 0.85}
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                cfg = json.load(f)
            defaults.update(cfg)
        except Exception as e:
            print(f"⚠️  讀取 config.json 失敗，使用預設值: {e}")
    return defaults

_cfg = load_config()
TQQQ_TARGET_VOL = _cfg["target_vol"]
TQQQ_CSP_OTM = _cfg["csp_otm"]
TQQQ_CSP_PREMIUM_EST = 0.027

STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tqqq_state.json")


# ═══════════════════════════════════════════════════════════
# 資料
# ═══════════════════════════════════════════════════════════

def fetch_data():
    end = datetime.date.today() + datetime.timedelta(days=1)
    start = end - datetime.timedelta(days=400)
    tickers = ['QQQ', 'TQQQ']
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

    # Variant A 倉位比例
    raw_pos = min(1.0, max(0.0, TQQQ_TARGET_VOL / rv20)) if rv20 > 0 else 1.0
    position = raw_pos
    cash_pct = 1 - position

    # CSP 建議（通用，不綁資金）
    csp_strike = round(float(tqqq_price * TQQQ_CSP_OTM), 2)
    csp_premium_per_share = round(float(tqqq_price * TQQQ_CSP_PREMIUM_EST), 2)
    # 每 $10,000 閒置現金可賣幾張
    csp_per_10k = int(10000 / (csp_strike * 100)) if csp_strike > 0 else 0

    regime = "🟢 牛市" if above else "🔴 熊市"

    return {
        "date": closes.index[-1].strftime('%Y-%m-%d'),
        "regime": regime,
        "asset": "TQQQ",
        "position_pct": round(position * 100),
        "cash_pct": round(cash_pct * 100),
        "tqqq_price": round(float(tqqq_price), 2),
        "qqq_price": round(float(current_price), 2),
        "sma200": round(float(current_sma), 2),
        "qqq_vs_sma": round((current_price / current_sma - 1) * 100, 2),
        "rv20": round(float(rv20 * 100), 1),
        "target_vol": TQQQ_TARGET_VOL,
        "csp_strike": csp_strike,
        "csp_premium_per_share": csp_premium_per_share,
        "csp_per_10k": csp_per_10k,
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

    msg += f"🇺🇸 TQQQ Variant A Vol Targeting\n"
    msg += f"{sig['regime']}（僅供參考，不影響倉位）\n"
    msg += f"\nTQQQ ${sig['tqqq_price']} ｜ RV20 {sig['rv20']:.0f}%\n"
    msg += f"QQQ ${sig['qqq_price']} vs SMA200 ${sig['sma200']}（{sig['qqq_vs_sma']:+.1f}%）\n"

    msg += f"\n🎯 建議倉位：\n"
    msg += f"  TQQQ {sig['position_pct']}% ／ 現金 {sig['cash_pct']}%\n"
    if sig['cash_pct'] > 5:
        msg += f"  （現金建議 parking 在 BOXX）\n"
    msg += f"  （公式：30% ÷ {sig['rv20']:.0f}% = {sig['position_pct']}%）\n"

    if sig['cash_pct'] > 5:
        msg += f"\n💰 閒置現金 Sell Put 建議：\n"
        msg += f"  Strike ${sig['csp_strike']}（~15% OTM）\n"
        msg += f"  預估權利金 ~${sig['csp_premium_per_share']}/股\n"
        msg += f"  每 $10,000 閒置現金 → {sig['csp_per_10k']} 張\n"

    msg += f"\n{'━' * 28}"
    return msg


# ═══════════════════════════════════════════════════════════
# LINE Messaging API 推播（push message）
# ═══════════════════════════════════════════════════════════

def send_line_message(msg):
    """
    透過 LINE Messaging API broadcast 發送給所有好友。
    文件：https://developers.line.biz/en/reference/messaging-api/#send-broadcast-message
    免費方案每月 200 則 broadcast，每日 1 則綽綽有餘。
    """
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

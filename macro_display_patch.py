# -*- coding: utf-8 -*-
"""
macro_display_patch.py — 把 izaax macro_regime.json 接進 daily_signal.py 推播（純顯示層）
放進 tqqq-signal repo 根目錄，macro_regime.json 也放 repo 根目錄。

用法（daily_signal.py 裡加兩行）:
    from macro_display_patch import macro_regime_line
    message += macro_regime_line()          # 加在 Telegram 訊息最尾端

鐵律:
  - 本模組只產生一行文字，不回傳任何可用於調整倉位的數值
  - regime 檔缺失/過期/壞掉一律靜默降級，絕不影響主訊號流程
"""
import json
import os
from datetime import datetime, timezone, timedelta

# 路徑解析順序：環境變數 MACRO_REGIME_PATH > 同目錄 macro_regime.json
# 在 Mac 本機跑的系統（Scan V4 等）建議直接指到 Izaxx 資料夾，永遠讀最新版：
#   export MACRO_REGIME_PATH="$HOME/Library/Mobile Documents/com~apple~CloudDocs/投資理財/03總經與學習資料/Izaxx/macro_regime.json"
# GitHub Actions 上跑的（tqqq-signal）把檔案 commit 進 repo 根目錄即可。
_IZAXX_DEFAULT = os.path.expanduser(
    "~/Library/Mobile Documents/com~apple~CloudDocs/投資理財/03總經與學習資料/Izaxx/macro_regime.json"
)
_CANDIDATES = [
    os.environ.get("MACRO_REGIME_PATH", ""),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "macro_regime.json"),
    _IZAXX_DEFAULT,
]
REGIME_PATH = next((c for c in _CANDIDATES if c and os.path.exists(c)), _IZAXX_DEFAULT)
STALE_DAYS = 45  # izaax 財訊雙週一篇；超過 45 天沒更新就標註過期


def macro_regime_line() -> str:
    """回傳一行總經定調文字（含換行前綴）；任何異常回傳空字串。"""
    try:
        with open(REGIME_PATH, encoding="utf-8") as f:
            r = json.load(f)
        cov = r.get("coverage_until", "?")
        stance = r.get("stance", "?")
        phase = r.get("cycle_phase", "").split("（")[0].replace("生產力循環・", "")
        flags = "、".join(
            f["flag"].split("（")[0].split("，")[0] for f in r.get("risk_flags", [])[:2]
        )
        # 過期檢查（用台灣時間）
        stale = ""
        try:
            cov_d = datetime.strptime(cov, "%Y-%m-%d").replace(tzinfo=timezone(timedelta(hours=8)))
            now = datetime.now(timezone(timedelta(hours=8)))
            if (now - cov_d).days > STALE_DAYS:
                stale = f"⚠️已{(now - cov_d).days}天未更新"
        except Exception:
            pass
        line = f"\n—\n🌐 總經(izaax): {stance}｜{phase}"
        if flags:
            line += f"｜旗標: {flags}"
        line += f"（至{cov}{stale}，僅供參考不計分）"
        return line
    except Exception:
        return ""  # 靜默降級


if __name__ == "__main__":
    print(macro_regime_line() or "(no regime file)")

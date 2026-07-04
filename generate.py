#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ray 投資 App — 每日資料產生器(在 GitHub Actions 上執行)
產出 docs/data.json 供手機網頁 App 讀取"""
import json, os, datetime, urllib.request

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")
TZ = datetime.timezone(datetime.timedelta(hours=8))  # 台北時間

WATCHLIST = {
    "2330.TW": "台積電", "2317.TW": "鴻海", "2308.TW": "台達電", "2454.TW": "聯發科",
    "2327.TW": "國巨", "1802.TW": "台玻", "6862.TW": "三集瑞-KY", "3701.TW": "大眾控",
    "4722.TW": "國精化", "6706.TW": "惠特", "2305.TW": "全友", "6442.TW": "光聖",
    "6515.TW": "穎崴", "6446.TW": "藥華藥",
    "3324.TWO": "雙鴻", "3491.TWO": "昇達科", "3585.TWO": "聯致", "6223.TWO": "旺矽",
    "6750.TWO": "泰創工程",
    "AAPL": "Apple", "NVDA": "NVIDIA", "TSLA": "Tesla", "META": "Meta",
}
INDICES = {"^TWII": "台股加權", "^GSPC": "S&P500", "^VIX": "VIX", "USDTWD=X": "美元兌台幣"}


def rsi14(closes):
    if len(closes) < 15: return None
    gains, losses = [], []
    for i in range(1, len(closes)):
        d = closes[i] - closes[i-1]
        gains.append(max(d, 0)); losses.append(max(-d, 0))
    ag = sum(gains[:14]) / 14; al = sum(losses[:14]) / 14
    for i in range(14, len(gains)):
        ag = (ag * 13 + gains[i]) / 14; al = (al * 13 + losses[i]) / 14
    return 100.0 if al == 0 else round(100 - 100 / (1 + ag / al), 1)


def fetch_all():
    import yfinance as yf
    rows = []
    symbols = list(INDICES) + list(WATCHLIST)
    data = yf.download(symbols, period="1y", interval="1d",
                       group_by="ticker", auto_adjust=True, progress=False, threads=True)
    for sym in symbols:
        name = INDICES.get(sym) or WATCHLIST.get(sym)
        is_index = sym in INDICES
        try:
            df = data[sym].dropna()
            c = df["Close"]
            if len(c) < 2: continue
            px, prev = float(c.iloc[-1]), float(c.iloc[-2])
            ma20 = float(c.tail(20).mean()); ma50 = float(c.tail(50).mean())
            ma200 = float(c.tail(200).mean())
            trend = "多頭" if px > ma50 and px > ma200 else ("空頭" if px < ma50 and px < ma200 else "整理")
            rows.append(dict(
                sym=sym, name=name, index=is_index,
                px=round(px, 2), pct=round((px / prev - 1) * 100, 2),
                ma20=round(ma20, 1), ma50=round(ma50, 1), ma200=round(ma200, 1),
                hi=round(float(c.max()), 1), lo=round(float(c.min()), 1),
                rsi=rsi14(list(c.tail(120))), trend=trend,
                dev50=round((px / ma50 - 1) * 100, 1),
                spark=[round(float(v), 2) for v in c.tail(30)],
            ))
        except Exception as e:
            rows.append(dict(sym=sym, name=name, index=is_index, err=str(e)[:60]))
    return rows


def gemini(prompt, key):
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{GEMINI_MODEL}:generateContent?key={key}")
    body = json.dumps({"contents": [{"parts": [{"text": prompt}]}],
                       "generationConfig": {"temperature": 0.4, "maxOutputTokens": 4096}}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)["candidates"][0]["content"]["parts"][0]["text"]


def build_prompt(now, rows, is_sat):
    block = "\n".join(
        f"{r['name']}({r['sym']}) 現價{r['px']} 日漲跌{r['pct']:+}% MA50 {r['ma50']} "
        f"MA200 {r['ma200']} 52週高{r['hi']}低{r['lo']} RSI {r['rsi']} 趨勢{r['trend']} 乖離50日{r['dev50']:+}%"
        for r in rows if not r.get("err"))
    today = now.strftime("%-m/%-d")
    wd = "一二三四五六日"[now.weekday()]
    mode = ("週六改週報:①本週回顧 ②下週展望與行動劇本 ③深度觀念(8句)" if is_sat else
            "七節:①今日速覽(3行) ②大盤趨勢與情緒 "
            "③綜合判讀+行動劇本(溫度計1-10、立場進攻/中性/防守、2-3條基於真實價位的若-則規則、否證條件) "
            "④持股重點(只挑5檔最需要注意的,各給支撐壓力與參考動作) "
            "⑤事件提醒 ⑥風險與部位(單一持股≤15%現金≥20%) ⑦今日新知(3句)")
    return (f"你是專業投資研究助理,為台灣企金背景的投資人 Ray 產出 {today}(週{wd}) 簡報。"
            f"繁體中文;純文字;「━━ 節名 ━━」分節;短行;價位必須由下方真實數據計算,嚴禁捏造;"
            f"穎崴與旺矽同屬測試介面視為同一部位;結尾標註(研究參考,非投資建議)。{mode}\n\n真實數據:\n{block}")


def main():
    now = datetime.datetime.now(TZ)
    key = os.environ.get("GEMINI_API_KEY", "")
    rows = fetch_all()
    briefing = ""
    if key and now.weekday() != 6:
        try:
            briefing = gemini(build_prompt(now, rows, now.weekday() == 5), key)
        except Exception as e:
            briefing = f"(今日 AI 判讀失敗:{str(e)[:100]}——以下為原始數據,App 內各卡片仍為即時計算)"
    os.makedirs("docs", exist_ok=True)
    hist_path = "docs/history.json"
    hist = json.load(open(hist_path)) if os.path.exists(hist_path) else []
    stamp = now.strftime("%Y-%m-%d")
    hist = [h for h in hist if h["date"] != stamp]
    hist.append({"date": stamp, "briefing": briefing})
    hist = hist[-30:]  # 保留30天
    json.dump(hist, open(hist_path, "w"), ensure_ascii=False)
    json.dump({"updated": now.strftime("%Y-%m-%d %H:%M"), "rows": rows, "briefing": briefing},
              open("docs/data.json", "w"), ensure_ascii=False)
    print("OK", stamp)


if __name__ == "__main__":
    main()

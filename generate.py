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
                       "generationConfig": {"temperature": 0.4, "maxOutputTokens": 8192}}).encode()
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
    mode = ("週六可在開頭加開【本週回顧】一節，其餘同平日" if is_sat else "") + \
        "務必輸出完整八節，每節標題獨立一行用「━━ 節名 ━━」，節與節之間空一行，內容全部條列一行一重點：" \
        "━━ 今日速覽 ━━(3-5行：大盤方向、今天最重要一件事、最值得看的一檔) " \
        "━━ 大盤與情緒 ━━(加權/S&P500/VIX/美元台幣逐項，標多空與乖離；VIX<15自滿15-20正常>25恐慌) " \
        "━━ 綜合判讀 ━━(溫度計1-10分+兩行理由｜立場:進攻/中性/防守｜2-3條基於真實價位的若-則行動規則｜否證條件) " \
        "━━ 全持股動態 ━━(下方每一檔自選股都要列不可省略，一行一檔：名稱 現價 漲跌%｜趨勢｜RSI｜支撐(50日線或近期低)｜壓力(52週高)｜參考:續抱/回檔加碼/減碼/觀察；乖離>20%標「勿追設移動停利」；穎崴與旺矽視為同一部位) " \
        "━━ 事件倒數 ━━(只列高度確信的例行時點如台股每月10日前公布上月營收、美股財報季月份；不確定日期寫「建議查證」，嚴禁編造) " \
        "━━ 潛力雷達 ━━(從下方數據找2-3個值得研究方向：強勢且乖離尚小、超跌接近支撐、趨勢轉強者，說邏輯與風險；嚴禁捏造不在數據中的消息) " \
        "━━ 風險與部位 ━━(組合八成集中台灣電子/AI鏈；單一持股≤15%、現金≥20%、乖離>20%移動停利) " \
        "━━ 今日新知 ━━(3-5句實用觀念，主題輪替)"
    return (f"你是專業投資研究助理,為台灣企金背景的投資人 Ray 產出 {today}(週{wd}) 簡報。"
            f"直接輸出簡報成品全文,嚴禁包含思考過程、草稿標記(如 Drafting、Section)、前言或任何後設文字,第一行必須是簡報標題;繁體中文;務必完整輸出全部段落不得中途停止或截斷;純文字;「━━ 節名 ━━」分節;短行;價位必須由下方真實數據計算,嚴禁捏造;"
            f"穎崴與旺矽同屬測試介面視為同一部位;結尾標註(研究參考,非投資建議)。{mode}\n\n真實數據:\n{block}")


def main():
    now = datetime.datetime.now(TZ)
    key = os.environ.get("GEMINI_API_KEY", "")
    rows = fetch_all()
    briefing = ""
    if key and now.weekday() != 6:
        try:
            briefing = gemini(build_prompt(now, rows, False), key)
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

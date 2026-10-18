"""Public news headlines (Google News RSS, zh-TW) with simple keyword tagging."""
import datetime as dt
import time
import urllib.parse
from email.utils import parsedate_to_datetime

import feedparser
import requests

from .config import UA

TAGS = {
    "法說會": ["法說", "法人說明會", "股東會"],
    "營收": ["營收"],
    "財報/獲利": ["財報", "EPS", "獲利", "毛利率", "季報", "淨利", "賺"],
    "除權息/股利": ["除息", "除權", "股利", "配息", "填息"],
    "訂單/產品": ["訂單", "接單", "出貨", "新品", "量產", "擴產", "認證", "打入"],
    "併購/投資": ["併購", "收購", "入股", "合併", "策略聯盟", "私募"],
    "處置/注意股": ["處置", "注意股", "警示", "分盤"],
    "增資/籌資": ["增資", "可轉債", "減資", "庫藏股"],
    "法人報告/評等": ["目標價", "調升", "調降", "評等", "外資報告", "升評", "降評", "喊進", "看好"],
    "利空事件": ["修正", "走跌", "崩", "下修", "衰退", "虧損", "裁員", "停工", "火災", "罰", "調查", "跌停", "重挫", "大跌", "砍"],
    "利多事件": ["漲停", "大漲", "噴", "創高", "新高", "飆"],
    "政策/總經": ["關稅", "制裁", "政策", "補貼", "出口管制", "央行", "Fed", "聯準會", "川普"],
    "AI/半導體題材": ["AI", "輝達", "NVIDIA", "CoWoS", "伺服器", "ASIC", "HBM", "先進封裝", "矽光子"],
}


def tag(title):
    return [k for k, kws in TAGS.items() if any(w.lower() in title.lower() for w in kws)]


def headlines(name, code, start: dt.date, end: dt.date, limit=8):
    """Headlines mentioning the stock published between start 00:00 and end 23:59 (Taiwan time)."""
    q = f'"{name}" {code} after:{(start - dt.timedelta(days=1)).isoformat()} before:{(end + dt.timedelta(days=1)).isoformat()}'
    url = "https://news.google.com/rss/search?" + urllib.parse.urlencode(
        {"q": q, "hl": "zh-TW", "gl": "TW", "ceid": "TW:zh-Hant"})
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=20)
        feed = feedparser.parse(r.content)
    except Exception:
        return None
    finally:
        time.sleep(1.2)
    out = []
    tw = dt.timezone(dt.timedelta(hours=8))
    for e in feed.entries:
        try:
            pub = parsedate_to_datetime(e.published).astimezone(tw)
        except Exception:
            continue
        if not (start <= pub.date() <= end):
            continue
        title = e.title
        if "爆料同學會" in title:  # skip forum chatter
            continue
        if name not in title and code not in title:  # skip market-wide lists that merely mention the stock
            continue
        src = e.get("source", {}).get("title", "") if isinstance(e.get("source"), dict) else ""
        out.append({"time": pub.strftime("%Y-%m-%d %H:%M"), "title": title, "source": src, "link": e.link,
                    "tags": tag(title)})
    out.sort(key=lambda x: x["time"], reverse=True)
    return out[:limit]

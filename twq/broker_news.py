"""券商/投顧研究報告的公開代理：Google News RSS 標題中的「目標價、調升/調降評等」提及次數（只能即時累積）。
Google News 的歷史查詢（after:/before:）回傳量極少（每週僅個位數），無法回補，所以這組資料只從上線日起逐日累積，
累積滿 120 個交易日前不進模型。"""
import datetime as dt
import re
import time
import urllib.parse

import requests

from . import db
from .config import UA

QUERIES = {"tp": "目標價 when:1d", "up": "調升評等 OR 上修目標價 OR 調高目標價 when:1d",
           "down": "調降評等 OR 下修目標價 OR 調降目標價 when:1d", "broker": "投顧 OR 外資報告 看好 when:1d"}
UP_KW = re.compile(r"調升|上修|調高|看好|買進|優於大盤|加碼")
DN_KW = re.compile(r"調降|下修|砍|中立|減碼|劣於大盤|賣出")


def collect(day=None):
    day = day or dt.date.today().isoformat()
    con = db.connect()
    con.execute("CREATE TABLE IF NOT EXISTS broker_news(date TEXT, code TEXT, n_tp INT, n_up INT, n_down INT, PRIMARY KEY(date, code))")
    names = [(c, n) for c, n in con.execute("SELECT code, name FROM stock_info") if n and len(n) >= 2]
    cnt = {}
    seen = set()
    for k, q in QUERIES.items():
        url = "https://news.google.com/rss/search?" + urllib.parse.urlencode({"q": q, "hl": "zh-TW", "gl": "TW", "ceid": "TW:zh-Hant"})
        try:
            t = requests.get(url, headers={"User-Agent": UA}, timeout=30).text
        except Exception as e:
            print("broker_news fetch failed", k, e)
            continue
        for title in re.findall(r"<item><title>(.*?)</title>", t, re.S):
            if title in seen:
                continue
            seen.add(title)
            for code, name in names:
                if name in title or f"({code})" in title or f"（{code}）" in title:
                    c = cnt.setdefault(code, [0, 0, 0])
                    c[0] += "目標價" in title
                    c[1] += bool(UP_KW.search(title))
                    c[2] += bool(DN_KW.search(title))
        time.sleep(2)
    con.executemany("INSERT OR REPLACE INTO broker_news VALUES(?,?,?,?,?)", [(day, k, *v) for k, v in cnt.items()])
    con.commit()
    con.close()
    print(f"broker_news {day}: {len(seen)} 則標題, {len(cnt)} 檔被提及")
    return cnt


if __name__ == "__main__":
    collect()

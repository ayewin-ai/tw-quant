"""ETF / index-rebalance / government-fund / event-calendar data (free, public) and features.

Sources
- ETF holdings: MoneyDJ full holdings page (daily, *current snapshot only* -> stored going forward; no free history).
- Index reviews (MSCI, FTSE TWSE Taiwan 50 -> 0050/006208, 0056, 00878, 00919, 00929, 00940, 00713):
  adds/deletes parsed from Google News zh-TW headlines in each review window (noisy; dated by publication time),
  rebalance day from the published review rule (MSCI: last business day of Feb/May/Aug/Nov; FTSE TW50: 3rd Friday
  of Mar/Jun/Sep/Dec; ETF index rules approximated, see INDEXES).
- 法說會: MOPS t100sb02_1 (historical, per month, listed + OTC).
- Major tech events / earnings: hand-maintained calendar (TECH_EVENTS); dates after 2026-06 are approximate.
- 國安基金 / 八大行庫 / 勞動基金: no free daily holdings or trade data. Only Google News counts (same method live and
  historical, so comparable) are used as a slow proxy.
"""
import datetime as dt
import re
import time
import urllib.parse

import numpy as np
import pandas as pd
import requests

from . import db

UA = {"User-Agent": "Mozilla/5.0"}
SCHEMA = """
CREATE TABLE IF NOT EXISTS etf_holdings(date TEXT, etf TEXT, code TEXT, weight REAL, shares REAL, PRIMARY KEY(date, etf, code));
CREATE TABLE IF NOT EXISTS conf_calls(code TEXT, date TEXT, info TEXT, PRIMARY KEY(code, date));
CREATE TABLE IF NOT EXISTS conf_months(ym TEXT, market TEXT, n INTEGER, PRIMARY KEY(ym, market));
CREATE TABLE IF NOT EXISTS index_events(ann_date TEXT, idx TEXT, code TEXT, action TEXT, rebal_date TEXT, title TEXT,
  PRIMARY KEY(idx, code, action, rebal_date));
CREATE TABLE IF NOT EXISTS index_windows(idx TEXT, rebal_date TEXT, n_titles INTEGER, PRIMARY KEY(idx, rebal_date));
CREATE TABLE IF NOT EXISTS nat_news(date TEXT, topic TEXT, n INTEGER, PRIMARY KEY(date, topic));
"""

ETFS = ["0050", "006208", "0056", "00878", "00919", "00929", "00940", "00713", "00881", "00692"]

# idx -> search query, review months, rule ('last' = last trading day of month, 'fri3' = 3rd Friday), flow weight
INDEXES = {
    "MSCI": ("MSCI 季度調整 台股 成分股", [2, 5, 8, 11], "last", 1.0),
    "0050": ("0050 成分股 納入 剔除", [3, 6, 9, 12], "fri3", 1.0),     # FTSE TWSE Taiwan 50 (0050 + 006208)
    "0056": ("0056 成分股 調整", [6, 12], "fri3", 0.6),
    "00878": ("00878 成分股 調整", [5, 11], "last", 0.7),
    "00919": ("00919 成分股 調整", [5, 12], "fri3", 0.6),            # approx rule
    "00929": ("00929 成分股 調整", [6], "fri3", 0.3),                # approx rule
    "00940": ("00940 成分股 調整", [5, 11], "last", 0.3),            # approx rule
    "00713": ("00713 成分股 調整", [6, 12], "fri3", 0.3),            # approx rule
}
ADD_KW = ["納入", "新增", "加入", "入列", "入選", "上榜", "增", "進"]
DEL_KW = ["剔除", "刪除", "移除", "出局", "出清", "踢", "刪", "剔", "出"]
SPEC_KW = ["候選", "可能", "有望", "估", "預測", "搶先", "點名", "機會", "猜", "倒數", "恐", "誰"]

# Major tech events (Taiwan calendar date the information hits Taiwan trading). kind -> affected industries.
TECH_EVENTS = [
    ("2024-07-18", "TSMC法說"), ("2024-10-17", "TSMC法說"), ("2025-01-16", "TSMC法說"), ("2025-04-17", "TSMC法說"),
    ("2025-07-17", "TSMC法說"), ("2025-10-16", "TSMC法說"), ("2026-01-15", "TSMC法說"), ("2026-04-16", "TSMC法說"),
    ("2026-07-16", "TSMC法說"), ("2026-10-15", "TSMC法說"),
    ("2024-08-29", "NVDA財報"), ("2024-11-21", "NVDA財報"), ("2025-02-27", "NVDA財報"), ("2025-05-29", "NVDA財報"),
    ("2025-08-28", "NVDA財報"), ("2025-11-20", "NVDA財報"), ("2026-02-26", "NVDA財報"), ("2026-05-21", "NVDA財報"),
    ("2026-08-27", "NVDA財報"),
    ("2024-08-02", "AAPL財報"), ("2024-11-01", "AAPL財報"), ("2025-01-31", "AAPL財報"), ("2025-05-02", "AAPL財報"),
    ("2025-08-01", "AAPL財報"), ("2025-10-31", "AAPL財報"), ("2026-01-30", "AAPL財報"), ("2026-05-01", "AAPL財報"),
    ("2026-07-31", "AAPL財報"),
    ("2024-09-10", "Apple發表會"), ("2025-09-10", "Apple發表會"), ("2026-09-09", "Apple發表會"),
    ("2025-06-10", "WWDC"), ("2026-06-09", "WWDC"),
    ("2025-03-19", "NVIDIA GTC"), ("2026-03-17", "NVIDIA GTC"),
    ("2024-06-04", "Computex"), ("2025-05-20", "Computex"), ("2026-06-02", "Computex"),
    ("2025-01-07", "CES"), ("2026-01-07", "CES"),
    ("2024-10-11", "Tesla發表"), ("2025-11-07", "Tesla股東會"),
]
NAT_TOPICS = {"國安基金": "國安基金 護盤", "八大行庫": "八大行庫 買超", "勞動基金": "(勞動基金 OR 勞退基金 OR 退撫基金) 台股"}


def init():
    db.init()
    con = db.connect()
    con.executescript(SCHEMA)
    con.commit()
    con.close()


# ---------------------------------------------------------------- fetchers
def etf_holdings(date: dt.date = None, etfs=ETFS):
    """Snapshot of full ETF holdings from MoneyDJ (current only)."""
    init()
    rows = []
    for e in etfs:
        try:
            r = requests.get(f"https://www.moneydj.com/ETF/X/Basic/Basic0007B.xdjhtm?etfid={e}.TW", headers=UA, timeout=30)
            r.encoding = "utf-8"
            t = re.sub(r"<[^>]+>", " ", r.text)
            m = re.search(r"資料日期：(\d{4})/(\d{2})/(\d{2})", t)
            if not m:
                continue
            d = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
            for code, w, sh in re.findall(r"\((\d{4,6})\.TW\)\s+([\d.]+)\s+([\d,]+)", t):
                rows.append((d, e, code, float(w), float(sh.replace(",", ""))))
        except requests.RequestException:
            pass
        time.sleep(1.0)
    return db.upsert("etf_holdings", ["date", "etf", "code", "weight", "shares"], rows)


def conf_calls(start: dt.date, end: dt.date, force_recent=True):
    """MOPS 法人說明會 schedule, month by month (ROC year)."""
    init()
    con = db.connect()
    have = {(a, b) for a, b in con.execute("SELECT ym, market FROM conf_months")}
    con.close()
    m = dt.date(start.year, start.month, 1)
    n_total = 0
    recent = (dt.date.today() - dt.timedelta(days=62)).strftime("%Y-%m")
    while m <= end:
        ym = m.strftime("%Y-%m")
        for mk in ("sii", "otc"):
            if (ym, mk) in have and not (force_recent and ym >= recent):
                continue
            data = {"encodeURIComponent": "1", "step": "1", "firstin": "1", "off": "1", "TYPEK": mk,
                    "year": str(m.year - 1911), "month": f"{m.month:02d}", "co_id": ""}
            for i in range(3):
                try:
                    r = requests.post("https://mopsov.twse.com.tw/mops/web/ajax_t100sb02_1", data=data, headers=UA, timeout=60)
                    if r.status_code == 200 and "公司代號" in r.text:
                        rows = _parse_conf(r.text)
                        db.upsert("conf_calls", ["code", "date", "info"], rows)
                        db.upsert("conf_months", ["ym", "market", "n"], [(ym, mk, len(rows))])
                        n_total += len(rows)
                        break
                except requests.RequestException:
                    pass
                time.sleep(10 * (i + 1))
            time.sleep(4)
        m = (m + dt.timedelta(days=32)).replace(day=1)
    return n_total


def _parse_conf(html):
    rows = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", html, flags=re.S):
        tds = [re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", td)).strip() for td in re.findall(r"<td[^>]*>(.*?)</td>", tr, flags=re.S)]
        if len(tds) < 5 or not re.fullmatch(r"\d{4}", tds[0]):
            continue
        m = re.match(r"(\d{2,3})/(\d{2})/(\d{2})", tds[2])
        if not m:
            continue
        d = dt.date(int(m.group(1)) + 1911, int(m.group(2)), int(m.group(3)))
        rows.append((tds[0], d.isoformat(), (tds[5] if len(tds) > 5 else "")[:200]))
    return rows


def rebal_dates(start: dt.date, end: dt.date, tdays=None):
    """Rule-based rebalance (effective-at-close) trading day per index."""
    tdays = sorted(tdays or [])
    out = []
    for idx, (_, months, rule, w) in INDEXES.items():
        for y in range(start.year, end.year + 1):
            for mo in months:
                first = dt.date(y, mo, 1)
                nxt = (first + dt.timedelta(days=32)).replace(day=1)
                if rule == "last":
                    d = nxt - dt.timedelta(days=1)
                    while d.weekday() >= 5:
                        d -= dt.timedelta(days=1)
                    cand = [t for t in tdays if first.isoformat() <= t < nxt.isoformat()]
                    if cand and cand[-1] <= d.isoformat():
                        d = dt.date.fromisoformat(cand[-1])
                else:
                    d = first + dt.timedelta(days=(4 - first.weekday()) % 7 + 14)
                    cand = [t for t in tdays if t <= d.isoformat() and t >= (d - dt.timedelta(days=6)).isoformat()]
                    if tdays and cand and d.isoformat() not in tdays and d.isoformat() <= tdays[-1]:
                        d = dt.date.fromisoformat(cand[-1])
                if start <= d <= end:
                    out.append((idx, d.isoformat(), w))
    return sorted(out, key=lambda x: x[1])


def _names():
    con = db.connect()
    df = pd.read_sql("SELECT code, name FROM stock_info", con)
    con.close()
    out = {}
    for c, n in df.itertuples(index=False):
        if not n or not re.fullmatch(r"\d{4}", str(c)):
            continue
        n = str(n).replace("*", "").strip()
        for v in {n, n.replace("-KY", ""), n.replace("投控", ""), n.replace("控股", "")}:
            if len(v) >= 2:
                out.setdefault(v, c)
    return out


def _parse_title(title, names):
    """Return list of (code, action) found in an index-change headline."""
    t = re.sub(r"\s+-\s+[^-]+$", "", title)  # drop " - Publisher"
    t = t.replace("＊", "").replace("*", "")
    found = []
    for name in sorted(names, key=len, reverse=True):
        for m in re.finditer(re.escape(name), t):
            p = m.start()
            if any(p >= a and p < b for a, b, _ in found):
                continue
            found.append((p, p + len(name), names[name]))
    out = []
    kws = [(m.start(), "add") for k in ADD_KW for m in re.finditer(k, t)] + \
          [(m.start(), "del") for k in DEL_KW for m in re.finditer(k, t)]
    for a, b, code in found:
        before = [(a - p, act) for p, act in kws if p < a and a - p <= 14]
        after = [(p - b, act) for p, act in kws if p >= b and p - b <= 3]
        cand = sorted(before) or sorted(after)
        if cand:
            out.append((code, cand[0][1]))
    return out


def index_events(start: dt.date, end: dt.date):
    """Backfill index add/delete events from news headlines around each rule-based rebalance day (resumable)."""
    import feedparser
    from email.utils import parsedate_to_datetime
    from . import features as F
    init()
    tw = dt.timezone(dt.timedelta(hours=8))
    names = _names()
    con = db.connect()
    have = {(a, b) for a, b in con.execute("SELECT idx, rebal_date FROM index_windows")}
    con.close()
    today = dt.date.today()
    for idx, rd, _w in rebal_dates(start, end + dt.timedelta(days=45), F.trading_days()):
        r_d = dt.date.fromisoformat(rd)
        a = r_d - dt.timedelta(days=35)
        if a > today:
            continue
        done_window = r_d + dt.timedelta(days=2) < today
        if (idx, rd) in have and done_window:
            continue
        q = f"{INDEXES[idx][0]} after:{a.isoformat()} before:{min(r_d, today + dt.timedelta(days=1)).isoformat()}"
        url = "https://news.google.com/rss/search?" + urllib.parse.urlencode({"q": q, "hl": "zh-TW", "gl": "TW", "ceid": "TW:zh-Hant"})
        try:
            f = feedparser.parse(requests.get(url, headers=UA, timeout=30).content)
        except requests.RequestException:
            continue
        rows = {}
        for e in f.entries:
            title = e.title
            if any(k in title for k in SPEC_KW):
                continue
            if idx != "MSCI" and idx not in title and not (idx == "0050" and ("台灣50" in title or "006208" in title)):
                continue
            if idx == "MSCI" and "MSCI" not in title.upper():
                continue
            try:
                d = parsedate_to_datetime(e.published).astimezone(tw).date().isoformat()
            except Exception:
                continue
            for code, act in _parse_title(title, names):
                k = (idx, code, act, rd)
                if k not in rows or d < rows[k][0]:
                    rows[k] = (d, title[:200])
        db.upsert("index_events", ["ann_date", "idx", "code", "action", "rebal_date", "title"],
                  [(v[0], k[0], k[1], k[2], k[3], v[1]) for k, v in rows.items()])
        if done_window:
            db.upsert("index_windows", ["idx", "rebal_date", "n_titles"], [(idx, rd, len(f.entries))])
        print("index_events", idx, rd, len(f.entries), "titles", len(rows), "events", flush=True)
        time.sleep(2.5)


def nat_news(start: dt.date, end: dt.date):
    """Daily Google News counts for government-fund topics (same method live and historical; resumable)."""
    import feedparser
    from email.utils import parsedate_to_datetime
    init()
    tw = dt.timezone(dt.timedelta(hours=8))
    con = db.connect()
    have = {(a, b) for a, b in con.execute("SELECT date, topic FROM nat_news")}
    con.close()
    d = end
    while d >= start:
        rows = []
        for topic, q in NAT_TOPICS.items():
            if (d.isoformat(), topic) in have and d < dt.date.today() - dt.timedelta(days=2):
                continue
            url = "https://news.google.com/rss/search?" + urllib.parse.urlencode(
                {"q": f"{q} after:{(d - dt.timedelta(days=1)).isoformat()} before:{(d + dt.timedelta(days=1)).isoformat()}",
                 "hl": "zh-TW", "gl": "TW", "ceid": "TW:zh-Hant"})
            for i in range(3):
                try:
                    r = requests.get(url, headers=UA, timeout=20)
                    if r.status_code == 200:
                        n = 0
                        for e in feedparser.parse(r.content).entries:
                            try:
                                n += parsedate_to_datetime(e.published).astimezone(tw).date() == d
                            except Exception:
                                pass
                        rows.append((d.isoformat(), topic, n))
                        break
                    time.sleep(30 * (i + 1))
                except requests.RequestException:
                    time.sleep(10)
            time.sleep(1.5)
        db.upsert("nat_news", ["date", "topic", "n"], rows)
        if d.day == 1:
            print("nat_news at", d, flush=True)
        d -= dt.timedelta(days=1)


def update(start: dt.date, end: dt.date, verbose=True):
    init()
    out = {}
    for name, fn in (("etf_holdings", lambda: etf_holdings()),
                     ("conf_calls", lambda: conf_calls(end - dt.timedelta(days=40), end + dt.timedelta(days=40))),
                     ("index_events", lambda: index_events(end - dt.timedelta(days=40), end)),
                     ("nat_news", lambda: nat_news(end - dt.timedelta(days=3), end))):
        try:
            out[name] = fn()
        except Exception as e:  # never break the daily pipeline
            out[name] = f"error {e}"
    if verbose:
        print("flows update:", out, flush=True)
    return out

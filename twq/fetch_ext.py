"""Extra free data: TAIFEX futures / options, world markets (Yahoo), event-risk (GDELT, USGS, Google News)."""
import csv
import datetime as dt
import io
import json
import time
import urllib.parse

import requests

from . import db
from .fetch import get, num, _session

TAIFEX = "https://www.taifex.com.tw/cht/3"
TAIFEX_SLEEP = 3.0

EXT_SCHEMA = """
CREATE TABLE IF NOT EXISTS taifex_tx(date TEXT, session TEXT, contract TEXT, open REAL, high REAL, low REAL,
  close REAL, change REAL, volume REAL, settle REAL, oi REAL, PRIMARY KEY(date, session));
CREATE TABLE IF NOT EXISTS taifex_inst(date TEXT, who TEXT, net_oi REAL, net_oi_amt REAL, net_trade REAL,
  PRIMARY KEY(date, who));
CREATE TABLE IF NOT EXISTS taifex_pcr(date TEXT PRIMARY KEY, pc_vol REAL, pc_oi REAL);
CREATE TABLE IF NOT EXISTS gdelt_daily(date TEXT, topic TEXT, volume REAL, tone REAL, PRIMARY KEY(date, topic));
CREATE TABLE IF NOT EXISTS quakes(id TEXT PRIMARY KEY, time_utc TEXT, mag REAL, lat REAL, lon REAL, place TEXT);
CREATE TABLE IF NOT EXISTS news_counts(date TEXT, topic TEXT, n INTEGER, PRIMARY KEY(date, topic));
"""

# Yahoo symbols -> our name. close_utc_hour ~ when the daily bar is final (UTC), used for alignment.
WORLD = {
    "^N225": "nikkei", "^KS11": "kospi", "^HSI": "hsi", "000001.SS": "shanghai", "^GDAXI": "dax",
    "^STOXX50E": "stoxx50", "ES=F": "es_fut", "NQ=F": "nq_fut", "TWD=X": "usdtwd", "^TNX": "us10y",
    "DX-Y.NYB": "dxy", "CL=F": "oil", "GC=F": "gold",
}

GDELT_TOPICS = {
    "taiwan_strait": '("Taiwan Strait" OR "PLA drills" OR "China military" Taiwan)',
    "war": '(war OR missile OR airstrike OR invasion)',
    "sanctions_tariff": '(sanctions OR tariff OR "export controls")',
    "disaster": '(earthquake OR typhoon OR flood OR tsunami OR wildfire)',
    "emergency": '(blackout OR "power outage" OR pandemic OR "state of emergency" OR cyberattack OR terrorist)',
}

NEWS_TOPICS = {  # Google News zh-TW keyword counts (live; historical counts are capped and unreliable)
    "地緣政治": "(共軍 OR 軍演 OR 台海 OR 戰爭 OR 封鎖)",
    "制裁關稅": "(制裁 OR 關稅 OR 出口管制)",
    "氣候災害": "(地震 OR 颱風 OR 水災 OR 豪雨 OR 乾旱)",
    "突發事件": "(停電 OR 爆炸 OR 火災 OR 疫情 OR 網路攻擊)",
}


def init():
    db.init()
    con = db.connect()
    con.executescript(EXT_SCHEMA)
    con.commit()
    con.close()


def _months(start, end):
    m = dt.date(start.year, start.month, 1)
    while m <= end:
        nxt = (m.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
        yield max(m, start), min(nxt - dt.timedelta(days=1), end)
        m = nxt


def _taifex_csv(path, data):
    for i in range(4):
        try:
            time.sleep(TAIFEX_SLEEP)
            r = _session("taifex").post(f"{TAIFEX}/{path}", data=data, timeout=(10, 60))
            if r.status_code == 200:
                txt = r.content.decode("big5", errors="ignore")
                if "," in txt[:200]:
                    return list(csv.reader(io.StringIO(txt)))
                return []
        except requests.RequestException:
            pass
        time.sleep(10 * (i + 1))
    return None


def taifex_tx(start, end):
    n = 0
    for a, b in _months(start, end):
        rows = _taifex_csv("futDataDown", {"down_type": "1", "commodity_id": "TX",
                                           "queryStartDate": a.strftime("%Y/%m/%d"),
                                           "queryEndDate": b.strftime("%Y/%m/%d")})
        if not rows:
            continue
        best = {}
        for r in rows[1:]:
            if len(r) < 18 or r[1].strip() != "TX":
                continue
            c = r[2].strip()
            if "/" in c or "W" in c:  # spreads / weekly
                continue
            d = r[0].replace("/", "-")
            sess = "day" if r[17].strip() == "一般" else "night"
            key = (d, sess)
            if key not in best or c < best[key][2]:  # nearest month
                best[key] = (d, sess, c, num(r[3]), num(r[4]), num(r[5]), num(r[6]), num(r[7]), num(r[9]),
                             num(r[10]), num(r[11]))
        n += db.upsert("taifex_tx", ["date", "session", "contract", "open", "high", "low", "close", "change",
                                     "volume", "settle", "oi"], list(best.values()))
    return n


def taifex_inst(start, end):
    n = 0
    for a, b in _months(start, end):
        rows = _taifex_csv("futContractsDateDown", {"queryStartDate": a.strftime("%Y/%m/%d"),
                                                    "queryEndDate": b.strftime("%Y/%m/%d"), "commodityId": "TXF"})
        if not rows:
            continue
        out = []
        for r in rows[1:]:
            if len(r) < 15 or "臺股期貨" not in r[1]:
                continue
            who = {"外資及陸資": "foreign", "投信": "trust", "自營商": "dealer"}.get(r[2].strip())
            if who:
                out.append((r[0].replace("/", "-"), who, num(r[13]), num(r[14]), num(r[7])))
        n += db.upsert("taifex_inst", ["date", "who", "net_oi", "net_oi_amt", "net_trade"], out)
    return n


def taifex_pcr(start, end):
    n = 0
    for a, b in _months(start, end):
        rows = _taifex_csv("pcRatioDown", {"queryStartDate": a.strftime("%Y/%m/%d"),
                                           "queryEndDate": b.strftime("%Y/%m/%d")})
        if not rows:
            continue
        out = [(r[0].replace("/", "-"), num(r[3]), num(r[6])) for r in rows[1:] if len(r) >= 7 and "/" in r[0]]
        n += db.upsert("taifex_pcr", ["date", "pc_vol", "pc_oi"], out)
    return n


def world(rng="3y"):
    rows = []
    for sym, name in WORLD.items():
        r = get(f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(sym)}", "yahoo", 1.5,
                {"range": rng, "interval": "1d"})
        try:
            res = r["chart"]["result"][0]
            ts, cl = res["timestamp"], res["indicators"]["quote"][0]["close"]
            off = res["meta"].get("gmtoffset", 0)
        except (TypeError, KeyError, IndexError):
            print("world: failed", sym)
            continue
        prev = None
        for t, c in zip(ts, cl):
            if c is None:
                continue
            d = dt.datetime.fromtimestamp(t + off, dt.timezone.utc).date().isoformat()
            rows.append((d, name, c, (c / prev - 1) if prev else None))
            prev = c
    return db.upsert("us_daily", ["date", "symbol", "close", "ret"], rows)


def gdelt(start, end, topics=None):
    """Daily article volume (fraction of all coverage) and average tone per topic. 1 request / 6 s."""
    n = 0
    for topic, q in (topics or GDELT_TOPICS).items():
        # chunk by ~6 months to keep daily resolution
        s = start
        while s <= end:
            e = min(s + dt.timedelta(days=180), end)
            vals = {}
            for mode in ("timelinevol", "timelinetone"):
                url = "https://api.gdeltproject.org/api/v2/doc/doc?" + urllib.parse.urlencode({
                    "query": q, "mode": mode, "format": "json", "timelinesmooth": 0,
                    "startdatetime": s.strftime("%Y%m%d000000"),
                    "enddatetime": (e + dt.timedelta(days=1)).strftime("%Y%m%d000000")})
                j = None
                for i in range(4):
                    time.sleep(6.5)
                    try:
                        r = requests.get(url, timeout=90)
                        j = r.json()
                        break
                    except (requests.RequestException, ValueError):
                        time.sleep(15 * (i + 1))
                if not j or not j.get("timeline"):
                    continue
                for p in j["timeline"][0]["data"]:
                    d = dt.datetime.strptime(p["date"][:8], "%Y%m%d").date().isoformat()
                    vals.setdefault(d, [None, None])[0 if mode == "timelinevol" else 1] = p["value"]
            rows = [(d, topic, v[0], v[1]) for d, v in vals.items()]
            n += db.upsert("gdelt_daily", ["date", "topic", "volume", "tone"], rows)
            print(f"gdelt {topic} {s}~{e}: {len(rows)}", flush=True)
            s = e + dt.timedelta(days=1)
    return n


def quakes(start, end):
    """USGS M>=5 earthquakes in/near Taiwan (lat 20-27, lon 118-124)."""
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    r = requests.get(url, params={"format": "geojson", "starttime": start.isoformat(),
                                  "endtime": (end + dt.timedelta(days=1)).isoformat(), "minmagnitude": 5,
                                  "minlatitude": 20, "maxlatitude": 27, "minlongitude": 118, "maxlongitude": 124},
                     timeout=60)
    rows = []
    for f in r.json().get("features", []):
        p, g = f["properties"], f["geometry"]["coordinates"]
        t = dt.datetime.fromtimestamp(p["time"] / 1000, dt.timezone.utc).isoformat(timespec="seconds")
        rows.append((f["id"], t, p["mag"], g[1], g[0], p.get("place")))
    return db.upsert("quakes", ["id", "time_utc", "mag", "lat", "lon", "place"], rows)


def news_counts(date: dt.date):
    """Google News zh-TW keyword counts for Taiwan date `date`.

    auto-fix 2026-10-21: the old live version used `when:1d` and counted every RSS entry, which
    almost always hits Google's ~100-item cap, while the backfill counts only items published on
    that Taiwan date via an after/before window. The mismatch made every live day look like a
    huge spike (Z≈4.8). Now the live collector uses exactly the backfill method and only for
    completed days (d-3..d-1, overwritten each run): counting day d itself while d+1 is still empty
    also inflates it toward the cap. Features use strict as-of (date < day), so nothing is lost."""
    return news_counts_hist(date - dt.timedelta(days=3), date - dt.timedelta(days=1), refresh=True)


def news_counts_hist(start: dt.date, end: dt.date, refresh=False):
    """Backfill daily zh-TW Google News counts per topic (capped at ~100 items/query, so a rough intensity proxy)."""
    import feedparser
    from email.utils import parsedate_to_datetime
    tw = dt.timezone(dt.timedelta(hours=8))
    con = db.connect()
    have = set() if refresh else {(r[0], r[1]) for r in con.execute("SELECT date, topic FROM news_counts")}
    con.close()
    total = 0
    d = end
    while d >= start:
        rows = []
        for topic, q in NEWS_TOPICS.items():
            if (d.isoformat(), topic) in have:
                continue
            url = "https://news.google.com/rss/search?" + urllib.parse.urlencode(
                {"q": f"{q} after:{(d - dt.timedelta(days=1)).isoformat()} before:{(d + dt.timedelta(days=1)).isoformat()}",
                 "hl": "zh-TW", "gl": "TW", "ceid": "TW:zh-Hant"})
            for i in range(3):
                try:
                    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
                    if r.status_code == 200:
                        f = feedparser.parse(r.content)
                        n = 0
                        for e in f.entries:
                            try:
                                if parsedate_to_datetime(e.published).astimezone(tw).date() == d:
                                    n += 1
                            except Exception:
                                pass
                        rows.append((d.isoformat(), topic, n))
                        break
                    time.sleep(30 * (i + 1))
                except requests.RequestException:
                    time.sleep(10)
            time.sleep(2.0)
        total += db.upsert("news_counts", ["date", "topic", "n"], rows) or 0
        if d.day == 1 and not refresh:
            print("news_counts_hist at", d, flush=True)
        d -= dt.timedelta(days=1)
    return total


def update(start: dt.date, end: dt.date, verbose=True, gdelt_on=True):
    init()
    out = {}
    for name, fn in (("taifex_tx", taifex_tx), ("taifex_inst", taifex_inst), ("taifex_pcr", taifex_pcr)):
        try:
            out[name] = fn(start, end)
        except Exception as e:  # never break the daily pipeline
            out[name] = f"error {e}"
    for name, fn in (("world", lambda: world("3y" if (end - start).days > 120 else "3mo")),
                     ("quakes", lambda: quakes(start - dt.timedelta(days=5), end)),
                     ("news_counts", lambda: news_counts(end))):
        try:
            out[name] = fn()
        except Exception as e:
            out[name] = f"error {e}"
    if gdelt_on:
        try:
            out["gdelt"] = gdelt_events(start - dt.timedelta(days=3), end)
        except Exception as e:
            out["gdelt"] = f"error {e}"
    if verbose:
        print("ext update:", json.dumps(out, ensure_ascii=False), flush=True)
    return out


def gdelt_events(start: dt.date, end: dt.date):
    """Backfillable geopolitical index from GDELT 1.0 daily event files (https://data.gdeltproject.org/events/).

    Per day (file date = events added that UTC day; file is published ~06:00 UTC next day):
      taiwan_strait: article share (per 10k) of China-Taiwan events or conflict events involving / located in Taiwan
      war:           article share of root codes 19 (fight) / 20 (mass violence)
      sanctions_tariff: article share of code 163x (embargo/sanctions) and 1056/1246 style economic coercion (16x)
      emergency:     article share of root 18 (assault) + 14 (protest)
    tone = average AvgTone of the taiwan_strait / war events.
    """
    import zipfile
    con = db.connect()
    have = {r[0] for r in con.execute("SELECT DISTINCT date FROM gdelt_daily WHERE topic='taiwan_strait'")}
    con.close()
    d = start
    while d <= end:
        ds = d.isoformat()
        if ds in have:
            d += dt.timedelta(days=1)
            continue
        url = f"https://data.gdeltproject.org/events/{d:%Y%m%d}.export.CSV.zip"
        try:
            r = requests.get(url, timeout=120)
        except requests.RequestException:
            r = None
        if r is None or r.status_code != 200:
            d += dt.timedelta(days=1)
            continue
        z = zipfile.ZipFile(io.BytesIO(r.content))
        tot = 0
        acc = {k: [0.0, 0.0, 0.0] for k in ("taiwan_strait", "war", "sanctions_tariff", "emergency")}  # arts, tone*arts
        with z.open(z.namelist()[0]) as f:
            for line in io.TextIOWrapper(f, encoding="utf-8", errors="ignore"):
                x = line.split("\t")
                if len(x) < 52:
                    continue
                try:
                    a = int(x[33] or 0)
                    tone = float(x[34] or 0)
                except ValueError:
                    continue
                tot += a
                c1, c2, code, root, quad, geo = x[7], x[17], x[26], x[28], x[29], x[51]
                hits = []
                if ({c1, c2} == {"CHN", "TWN"}) or ("TWN" in (c1, c2) and quad in ("3", "4")) or \
                        (geo == "TW" and root in ("15", "17", "18", "19", "20")):
                    hits.append("taiwan_strait")
                if root in ("19", "20"):
                    hits.append("war")
                if code.startswith("163") or code.startswith("172"):
                    hits.append("sanctions_tariff")
                if root in ("14", "18"):
                    hits.append("emergency")
                for h in hits:
                    acc[h][0] += a
                    acc[h][1] += tone * a
        rows = [(ds, k, v[0] / tot * 1e4 if tot else None, v[1] / v[0] if v[0] else None) for k, v in acc.items()]
        db.upsert("gdelt_daily", ["date", "topic", "volume", "tone"], rows)
        if d.day == 1:
            print("gdelt_events", ds, rows[0], flush=True)
        time.sleep(0.5)
        d += dt.timedelta(days=1)

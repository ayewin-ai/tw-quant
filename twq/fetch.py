"""Data ingestion from free public sources: TWSE, TPEx, MOPS, Yahoo (US overnight)."""
import datetime as dt
import io
import json
import re
import threading
import time

import pandas as pd
import requests

from . import db
from .config import TWSE_SLEEP, TPEX_SLEEP, MOPS_SLEEP, UA, US_SYMBOLS

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

_POOL = ThreadPoolExecutor(max_workers=16)
_tls = threading.local()


def _session(host):
    """One requests.Session per thread and host (Session is not thread-safe)."""
    key = "yahoo" if host == "yahoo" else "default"
    ss = getattr(_tls, "s", None)
    if ss is None:
        ss = _tls.s = {}
    if key not in ss:
        s = requests.Session()
        s.headers.update({"User-Agent": "Mozilla/5.0"} if key == "yahoo" else
                         {"User-Agent": UA, "Accept": "application/json,text/html,*/*"})
        ss[key] = s
    return ss[key]

_last = {}
_tl = threading.Lock()


def _throttle(host, gap):
    with _tl:
        t = _last.get(host, 0)
        wait = t + gap - time.time()
    if wait > 0:
        time.sleep(wait)
    with _tl:
        _last[host] = time.time()


def get(url, host, gap, params=None, retries=4, as_json=True, timeout=30):
    sess = _session(host)
    for i in range(retries):
        _throttle(host, gap)
        try:
            fut = _POOL.submit(sess.get, url, params=params, timeout=(10, timeout))
            try:
                r = fut.result(timeout=timeout + 20)  # hard deadline: the proxy sometimes stalls sockets
            except FuturesTimeout:
                _tls.s = None
                raise requests.Timeout("hard deadline exceeded")
            if r.status_code == 200:
                if as_json:
                    return r.json()
                return r
            if r.status_code in (403, 429, 503):
                print(f"  [{host}] HTTP {r.status_code}, backing off", flush=True)
                time.sleep((5 if host == "yahoo" else 30) * (i + 1))
        except (requests.RequestException, json.JSONDecodeError, ValueError):
            _tls.s = None  # drop possibly broken connection
            time.sleep(5 * (i + 1))
    return None


def num(x):
    if x is None:
        return None
    s = re.sub(r"<[^>]+>", "", str(x)).replace(",", "").strip()
    if s in ("", "--", "---", "----", "X", "除權", "除息", "除權息", "N/A"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def roc(d: dt.date):
    return f"{d.year - 1911}/{d.month:02d}/{d.day:02d}"


def ymd(d: dt.date):
    return d.strftime("%Y%m%d")


def is_stock_code(code):
    return bool(re.fullmatch(r"[1-9]\d{3}", code))


# ---------------------------------------------------------------- TWSE
def twse_prices(d: dt.date):
    j = get("https://www.twse.com.tw/exchangeReport/MI_INDEX", "twse", TWSE_SLEEP,
            {"response": "json", "date": ymd(d), "type": "ALLBUT0999"})
    if j is None:
        return None  # network failure
    if j.get("stat") != "OK" or not j.get("tables"):
        return 0  # no trading
    date = d.isoformat()
    rows, idx = [], []
    for t in j["tables"]:
        title = t.get("title") or ""
        if "每日收盤行情" in title:
            for r in t["data"]:
                code = r[0].strip()
                sign = re.sub(r"<[^>]+>", "", r[9]).strip()
                ch = num(r[10])
                close = num(r[8])
                if ch is not None:
                    ch = -ch if "-" in sign else ch
                ret = None
                if close is not None and ch is not None and (close - ch) > 0:
                    ret = ch / (close - ch)
                rows.append((date, code, r[1].strip(), "TWSE", num(r[5]), num(r[6]), num(r[7]), close,
                             num(r[2]), num(r[4]), num(r[3]), ch, ret, num(r[15]) if len(r) > 15 else None))
        if "價格指數(臺灣證券交易所)" in title:
            for r in t["data"]:
                if r[0].strip() in ("發行量加權股價指數",):
                    sign = re.sub(r"<[^>]+>", "", r[2]).strip()
                    pct = num(r[4])
                    if pct is not None and "-" in sign:
                        pct = -abs(pct)
                    idx.append((date, "TAIEX", num(r[1]), pct / 100 if pct is not None else None))
                if r[0].strip() == "電子類指數":
                    sign = re.sub(r"<[^>]+>", "", r[2]).strip()
                    pct = num(r[4])
                    if pct is not None and "-" in sign:
                        pct = -abs(pct)
                    idx.append((date, "TWSE_ELEC", num(r[1]), pct / 100 if pct is not None else None))
    db.upsert("prices", ["date", "code", "name", "market", "open", "high", "low", "close", "volume", "value",
                         "trades", "change", "ret", "pe"], rows)
    db.upsert("index_daily", ["date", "name", "close", "pct"], idx)
    return len(rows)


def twse_inst(d: dt.date):
    j = get("https://www.twse.com.tw/fund/T86", "twse", TWSE_SLEEP,
            {"response": "json", "date": ymd(d), "selectType": "ALLBUT0999"})
    if j is None:
        return None
    if j.get("stat") != "OK":
        return 0
    f = j["fields"]

    def fi(*keys):
        for i, name in enumerate(f):
            if all(k in name for k in keys):
                return i
        return None
    i_f1 = fi("外陸資買賣超股數")
    i_f2 = fi("外資自營商買賣超股數")
    i_t = fi("投信買賣超股數")
    i_d = fi("自營商買賣超股數")
    i_tot = fi("三大法人買賣超股數")
    rows = []
    for r in j["data"]:
        fn = (num(r[i_f1]) or 0) + ((num(r[i_f2]) or 0) if i_f2 is not None else 0)
        rows.append((d.isoformat(), r[0].strip(), fn, num(r[i_t]), num(r[i_d]), num(r[i_tot])))
    return db.upsert("inst", ["date", "code", "foreign_net", "trust_net", "dealer_net", "total_net"], rows)


def twse_margin(d: dt.date):
    j = get("https://www.twse.com.tw/exchangeReport/MI_MARGN", "twse", TWSE_SLEEP,
            {"response": "json", "date": ymd(d), "selectType": "STOCK"})
    if j is None:
        return None
    if j.get("stat") != "OK":
        return 0
    rows = []
    for t in j.get("tables", []):
        if "融資融券彙總" in (t.get("title") or ""):
            for r in t["data"]:
                code = r[0].strip()
                if not code or code == "\u3000":
                    continue
                rows.append((d.isoformat(), code, num(r[6]), num(r[12])))
    return db.upsert("margin", ["date", "code", "margin_bal", "short_bal"], rows)


def twse_exdiv(start: dt.date, end: dt.date):
    j = get("https://www.twse.com.tw/exchangeReport/TWT49U", "twse", TWSE_SLEEP,
            {"response": "json", "startDate": ymd(start), "endDate": ymd(end)})
    if not j or j.get("stat") != "OK":
        return 0
    rows = []
    for r in j["data"]:
        m = re.match(r"(\d+)年(\d+)月(\d+)日", r[0])
        if not m:
            continue
        date = dt.date(int(m[1]) + 1911, int(m[2]), int(m[3])).isoformat()
        rows.append((date, r[1].strip(), r[2].strip(), r[6].strip(), num(r[5])))
    return db.upsert("exdiv", ["date", "code", "name", "kind", "amount"], rows)


def twse_exdiv_upcoming():
    """除權除息預告表 (future ex-dates)."""
    j = get("https://www.twse.com.tw/exchangeReport/TWT48U", "twse", TWSE_SLEEP, {"response": "json"})
    if not j or j.get("stat") != "OK":
        return 0
    rows = []
    for r in j["data"]:
        m = re.match(r"(\d+)年(\d+)月(\d+)日", r[0])
        if not m:
            continue
        date = dt.date(int(m[1]) + 1911, int(m[2]), int(m[3])).isoformat()
        rows.append((date, r[1].strip(), r[2].strip(), r[3].strip(), num(r[7])))
    return db.upsert("exdiv", ["date", "code", "name", "kind", "amount"], rows)


def twse_holidays(year):
    j = get("https://www.twse.com.tw/holidaySchedule/holidaySchedule", "twse", TWSE_SLEEP,
            {"response": "json", "queryYear": str(year - 1911)})
    if not j or "data" not in j:
        return 0
    rows = [(r[0], r[1], r[2]) for r in j["data"]]
    return db.upsert("holidays", ["date", "name", "note"], rows)


# ---------------------------------------------------------------- TPEx
TPEX = "https://www.tpex.org.tw/www/zh-tw"


def tpex_prices(d: dt.date):
    j = get(f"{TPEX}/afterTrading/dailyQuotes", "tpex", TPEX_SLEEP, {"date": d.strftime("%Y/%m/%d"), "response": "json"})
    if j is None:
        return None
    tabs = j.get("tables") or []
    if not tabs or not tabs[0].get("data"):
        return 0
    if j.get("date") and j["date"] != ymd(d):
        return 0
    rows = []
    for r in tabs[0]["data"]:
        code = r[0].strip()
        if not re.fullmatch(r"\d{4}|00\d{2,4}[A-Z]?", code):
            continue  # skip warrants / convertible bonds
        close = num(r[2])
        ch = num(r[3])
        ret = None
        if close is not None and ch is not None and (close - ch) > 0:
            ret = ch / (close - ch)
        rows.append((d.isoformat(), code, r[1].strip(), "TPEx", num(r[4]), num(r[5]), num(r[6]), close,
                     num(r[8]), num(r[9]), num(r[10]), ch, ret, None))
    db.upsert("prices", ["date", "code", "name", "market", "open", "high", "low", "close", "volume", "value",
                         "trades", "change", "ret", "pe"], rows)
    return len(rows)


def tpex_inst(d: dt.date):
    j = get(f"{TPEX}/insti/dailyTrade", "tpex", TPEX_SLEEP,
            {"type": "Daily", "sect": "EW", "date": d.strftime("%Y/%m/%d"), "response": "json"})
    if j is None:
        return None
    tabs = j.get("tables") or []
    if not tabs or not tabs[0].get("data"):
        return 0
    rows = []
    for r in tabs[0]["data"]:
        if len(r) < 24:
            continue
        # [8..10] 外資及陸資合計, [11..13] 投信, [20..22] 自營商合計, [23] 三大法人合計
        rows.append((d.isoformat(), r[0].strip(), num(r[10]), num(r[13]), num(r[22]), num(r[23])))
    return db.upsert("inst", ["date", "code", "foreign_net", "trust_net", "dealer_net", "total_net"], rows)


def tpex_margin(d: dt.date):
    j = get(f"{TPEX}/margin/balance", "tpex", TPEX_SLEEP, {"date": d.strftime("%Y/%m/%d"), "response": "json"})
    if j is None:
        return None
    tabs = j.get("tables") or []
    if not tabs or not tabs[0].get("data"):
        return 0
    rows = [(d.isoformat(), r[0].strip(), num(r[6]), num(r[14])) for r in tabs[0]["data"]]
    return db.upsert("margin", ["date", "code", "margin_bal", "short_bal"], rows)


def tpex_index_month(d: dt.date):
    j = get(f"{TPEX}/indexInfo/inx", "tpex", TPEX_SLEEP, {"date": d.strftime("%Y/%m/01"), "response": "json"})
    if not j or not j.get("tables"):
        return 0
    rows = []
    prev = None
    data = j["tables"][0].get("data") or []
    for r in data:
        close = num(r[4])
        chg = num(r[5])
        pct = chg / (close - chg) if (close and chg is not None and close - chg) else None
        rows.append((r[0].replace("/", "-"), "TPEX", close, pct))
    return db.upsert("index_daily", ["date", "name", "close", "pct"], rows)


def tpex_exdiv(start: dt.date, end: dt.date):
    j = get(f"{TPEX}/bulletin/exDailyQ", "tpex", TPEX_SLEEP,
            {"startDate": start.strftime("%Y/%m/%d"), "endDate": end.strftime("%Y/%m/%d"), "response": "json"})
    if not j or not j.get("tables"):
        return 0
    rows = []
    for r in j["tables"][0].get("data") or []:
        y, m, dd = r[0].split("/")
        rows.append((dt.date(int(y) + 1911, int(m), int(dd)).isoformat(), r[1].strip(), r[2].strip(),
                     r[8].strip(), num(r[7])))
    return db.upsert("exdiv", ["date", "code", "name", "kind", "amount"], rows)


# ---------------------------------------------------------------- MOPS monthly revenue
def mops_revenue(year, month):
    """上市(sii)/上櫃(otc) 月營收彙總表. ym = calendar month of revenue."""
    out = []
    for mk in ("sii", "otc"):
        for suffix in ("_0", "_1"):  # _0 國內, _1 國外(KY)
            url = f"https://mopsov.twse.com.tw/nas/t21/{mk}/t21sc03_{year - 1911}_{month}{suffix}.html"
            r = get(url, "mops", MOPS_SLEEP, as_json=False, retries=2)
            if r is None:
                continue
            html = r.content.decode("big5", errors="ignore")
            try:
                tabs = pd.read_html(io.StringIO(html))
            except ValueError:
                continue
            for t in tabs:
                if t.shape[1] < 7:
                    continue
                cols = [" ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in t.columns]
                if not any("代號" in c.replace(" ", "") for c in cols):
                    continue
                t.columns = cols
                ci = [i for i, c in enumerate(cols) if "代號" in c.replace(" ", "")][0]
                for _, row in t.iterrows():
                    code = str(row.iloc[ci]).strip()
                    if not re.fullmatch(r"\d{4}", code):
                        continue
                    vals = [num(row.iloc[ci + k]) for k in range(2, 7)]
                    out.append((f"{year:04d}-{month:02d}", code, *vals))
    return db.upsert("revenue", ["ym", "code", "rev", "rev_last_month", "rev_last_year", "mom", "yoy"], out)


def openapi_revenue_latest():
    """Latest month revenue via TWSE / TPEx openapi (fast)."""
    rows = []
    for url, host in (("https://openapi.twse.com.tw/v1/opendata/t187ap05_L", "twse_open"),
                      ("https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap05_O", "tpex")):
        j = get(url, host, 1.0)
        if not j:
            continue
        for r in j:
            vals = list(r.values())
            # keys differ between TWSE (zh) and TPEx (en); use positional
            try:
                ym_roc = str(vals[1])
                y, m = int(ym_roc[:-2]) + 1911, int(ym_roc[-2:])
                rows.append((f"{y:04d}-{m:02d}", str(vals[2]).strip(), num(vals[5]), num(vals[6]), num(vals[7]),
                             num(vals[8]), num(vals[9])))
            except (ValueError, IndexError):
                continue
    return db.upsert("revenue", ["ym", "code", "rev", "rev_last_month", "rev_last_year", "mom", "yoy"], rows)


# ---------------------------------------------------------------- company info / industry
INDUSTRY = {"01": "水泥", "02": "食品", "03": "塑膠", "04": "紡織纖維", "05": "電機機械", "06": "電器電纜",
            "08": "玻璃陶瓷", "09": "造紙", "10": "鋼鐵", "11": "橡膠", "12": "汽車", "14": "建材營造",
            "15": "航運", "16": "觀光餐旅", "17": "金融保險", "18": "貿易百貨", "19": "綜合", "20": "其他",
            "21": "化學", "22": "生技醫療", "23": "油電燃氣", "24": "半導體", "25": "電腦及週邊",
            "26": "光電", "27": "通信網路", "28": "電子零組件", "29": "電子通路", "30": "資訊服務",
            "31": "其他電子", "32": "文化創意", "33": "農業科技", "34": "電子商務", "35": "綠能環保",
            "36": "數位雲端", "37": "運動休閒", "38": "居家生活", "80": "管理股票", "91": "存託憑證"}


def stock_info():
    rows = []
    j = get("https://openapi.twse.com.tw/v1/opendata/t187ap03_L", "twse_open", 1.0)
    for r in j or []:
        rows.append((r["公司代號"].strip(), r["公司簡稱"].strip(), "TWSE",
                     INDUSTRY.get(str(r["產業別"]).strip(), str(r["產業別"]).strip())))
    j = get("https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O", "tpex", 1.0)
    for r in j or []:
        rows.append((r["SecuritiesCompanyCode"].strip(), r["CompanyAbbreviation"].strip(), "TPEx",
                     INDUSTRY.get(str(r["SecuritiesIndustryCode"]).strip(), str(r["SecuritiesIndustryCode"]).strip())))
    return db.upsert("stock_info", ["code", "name", "market", "industry"], rows)


# ---------------------------------------------------------------- US overnight (Yahoo chart API)
def us_markets(rng="3y"):
    rows = []
    for sym, name in US_SYMBOLS.items():
        r = get(f"https://query1.finance.yahoo.com/v8/finance/chart/{requests.utils.quote(sym)}", "yahoo", 1.5,
                {"range": rng, "interval": "1d"})
        try:
            res = r["chart"]["result"][0]
            ts = res["timestamp"]
            cl = res["indicators"]["quote"][0]["close"]
            off = res["meta"].get("gmtoffset", -14400)
        except (TypeError, KeyError, IndexError):
            continue
        prev = None
        for t, c in zip(ts, cl):
            if c is None:
                continue
            d = dt.datetime.fromtimestamp(t + off, dt.timezone.utc).date().isoformat()
            rows.append((d, name, c, (c / prev - 1) if prev else None))
            prev = c
    return db.upsert("us_daily", ["date", "symbol", "close", "ret"], rows)


# ---------------------------------------------------------------- orchestration
def trading_day_candidates(start: dt.date, end: dt.date):
    """要抓資料的候選日：依官方行事曆（人事行政總處＋證交所＋臨時休市記錄）排除休市日。
    不使用『沒資料』來排除，避免資料尚未公布時不再重試；颱風假等事前未知的休市，抓不到就維持 pending 0。"""
    from . import calendar
    d = end
    out = []
    while d >= start:
        if calendar.official_closed(d) is None:
            out.append(d)
        d -= dt.timedelta(days=1)
    return out  # newest first


TWSE_JOBS = [("twse_prices", twse_prices), ("twse_inst", twse_inst), ("twse_margin", twse_margin)]
TPEX_JOBS = [("tpex_prices", tpex_prices), ("tpex_inst", tpex_inst), ("tpex_margin", tpex_margin)]


def _run_jobs(days, jobs, force=False, verbose=True):
    done = {name: db.fetched(name) for name, _ in jobs}
    for d in days:
        ds = d.isoformat()
        for name, fn in jobs:
            if not force and done[name].get(ds) in ("ok", "nodata"):
                continue
            # skip chip data if prices say no trading
            if name.endswith("_inst") or name.endswith("_margin"):
                pname = name.split("_")[0] + "_prices"
                if db.fetched(pname).get(ds) == "nodata":
                    db.log_fetch(ds, name, "nodata")
                    continue
            n = fn(d)
            status = "error" if n is None else ("nodata" if n == 0 else "ok")
            # Today's data may not be published yet -> do not cache 'nodata' for today/yesterday
            if status == "nodata" and (dt.date.today() - d).days <= 1 and not force:
                status = "pending"
            db.log_fetch(ds, name, status, n or 0)
            if verbose:
                print(f"[{dt.datetime.now():%H:%M:%S}] {name} {ds}: {status} {n}", flush=True)


def update(start: dt.date, end: dt.date, force=False, verbose=True, extras=True):
    db.init()
    for y in range(start.year, end.year + 1):
        twse_holidays(y)
    days = trading_day_candidates(start, end)
    if verbose:
        print(f"candidate trading days: {len(days)} ({start} ~ {end})", flush=True)
    t1 = threading.Thread(target=_run_jobs, args=(days, TWSE_JOBS, force, verbose))
    t2 = threading.Thread(target=_run_jobs, args=(days, TPEX_JOBS, force, verbose))
    t1.start(); t2.start(); t1.join(); t2.join()
    try:
        from . import calendar
        calendar._clear()  # 新抓到的資料日要讓日曆看得到
    except Exception:
        pass
    if extras:
        update_extras(start, end, verbose)


def update_extras(start: dt.date, end: dt.date, verbose=True, revenue_months=None):
    # TPEx index (monthly), ex-dividend, US markets, stock info, revenue
    m = dt.date(start.year, start.month, 1)
    con = db.connect()
    have_tpex = {r[0][:7] for r in con.execute("SELECT date FROM index_daily WHERE name='TPEX'")}
    have_rev = {r[0] for r in con.execute("SELECT DISTINCT ym FROM revenue")}
    con.close()
    months = []
    while m <= end:
        months.append(m)
        m = (m.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
    for m in months:
        if m.strftime("%Y-%m") not in have_tpex or m.strftime("%Y-%m") >= (end - dt.timedelta(days=35)).strftime("%Y-%m"):
            tpex_index_month(m)
    # ex-dividend in ~2-month chunks
    s = start
    while s <= end:
        e = min(s + dt.timedelta(days=60), end + dt.timedelta(days=30))
        twse_exdiv(s, min(e, end))
        tpex_exdiv(s, e)
        s = e + dt.timedelta(days=1)
    twse_exdiv_upcoming()
    if verbose:
        print("ex-dividend done", flush=True)
    us_markets("3y" if (end - start).days > 300 else "3mo")
    stock_info()
    # Monthly revenue: month m revenue is published by the 10th of month m+1
    for m in months[:-1] if revenue_months is None else revenue_months:
        ym = m.strftime("%Y-%m")
        if ym in have_rev and ym < (end - dt.timedelta(days=70)).strftime("%Y-%m"):
            continue
        n = mops_revenue(m.year, m.month)
        if verbose:
            print(f"revenue {ym}: {n}", flush=True)
    openapi_revenue_latest()
    if verbose:
        print("extras done", flush=True)

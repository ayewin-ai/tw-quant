"""台股交易日曆。

資料來源（皆為公開資料，離線可用快取）：
1. 行政院人事行政總處「政府行政機關辦公日曆表」（data.gov.tw 資料集），取自其公開鏡像
   ruyut/TaiwanCalendar（每日逐筆 isHoliday）。星期六日、國定假日、補假、補班日（補班日證交所不開市）。
2. 臺灣證券交易所「市場開休市日期」（openapi.twse.com.tw /holidaySchedule/holidaySchedule），
   含人事行政總處日曆沒有的市場休市（例如農曆春節前後「僅辦理結算交割」的日子）。
3. config/calendar_overrides.json：臨時休市（颱風假等）— 這種事沒辦法事先知道，官方日曆也不會預先列出，
   事後依公告補記；若來不及補記，則以「實際資料」為準（見下）。
4. 實際資料：DB 內該日有 TWSE 成交資料 → 一定是交易日；DB 資料區間內、平日、行事曆說開市但完全沒資料 → 視為臨時休市。

判斷順序（status()）：實際資料 > 週末 > 臨時休市(overrides) > 人事行政總處 > 證交所假日表 > 資料區間內無資料 > 預設開市。
快取：config/calendar/{dgpa,twse}_YYYY.json，超過 REFRESH_DAYS 天或不存在才重抓；抓不到就用舊快取；完全沒有就退回 DB holidays 表。
CLI：python -m twq.calendar [status [DATE]|next [DATE]|is-open [DATE]|upcoming [N]|refresh|check|retarget|add-closure DATE 原因 來源]
"""
import datetime as dt
import json
import sys
import time
import urllib.request
from functools import lru_cache
from pathlib import Path

from .config import ROOT, UA

CAL_DIR = ROOT / "config" / "calendar"
OVERRIDES = ROOT / "config" / "calendar_overrides.json"
REFRESH_DAYS = 90
DGPA_URLS = ["https://cdn.jsdelivr.net/gh/ruyut/TaiwanCalendar/data/{y}.json",
             "https://raw.githubusercontent.com/ruyut/TaiwanCalendar/master/data/{y}.json"]
TWSE_URL = "https://openapi.twse.com.tw/v1/holidaySchedule/holidaySchedule"
# 證交所假日表中這兩類名稱其實是「開市」的提示（開始交易日/最後交易日）
TWSE_OPEN_HINTS = ("開始交易", "最後交易")


def _get(url, timeout=25):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _fresh(p: Path):
    return p.exists() and (time.time() - p.stat().st_mtime) < REFRESH_DAYS * 86400


def _roc_to_iso(s):
    s = str(s).strip()
    y, m, d = int(s[:-4]), int(s[-4:-2]), int(s[-2:])
    return dt.date(y + 1911, m, d).isoformat()


def load_dgpa(year, refresh=False):
    """{iso_date: {'holiday': bool, 'desc': str}}，人事行政總處辦公日曆。"""
    CAL_DIR.mkdir(parents=True, exist_ok=True)
    p = CAL_DIR / f"dgpa_{year}.json"
    if refresh or not _fresh(p):
        for u in DGPA_URLS:
            try:
                raw = _get(u.format(y=year))
                out = {}
                for r in raw:
                    d = f"{r['date'][:4]}-{r['date'][4:6]}-{r['date'][6:]}"
                    out[d] = {"holiday": bool(r["isHoliday"]), "desc": r.get("description", "")}
                if len(out) >= 365:
                    p.write_text(json.dumps({"source": u.format(y=year), "fetched": dt.datetime.now().isoformat(timespec="seconds"),
                                             "days": out}, ensure_ascii=False, indent=0))
                    break
            except Exception as e:
                print(f"[calendar] DGPA {year} 抓取失敗 ({u}): {e}", file=sys.stderr)
    if p.exists():
        return json.loads(p.read_text())["days"]
    return None


def load_twse(year, refresh=False):
    """{iso_date: {'name','note','open_hint'}}，證交所休市日表（官方 openapi 一次回傳當年度整年）。"""
    CAL_DIR.mkdir(parents=True, exist_ok=True)
    p = CAL_DIR / f"twse_{year}.json"
    if refresh or not _fresh(p):
        try:
            raw = _get(TWSE_URL)
            by_year = {}
            for r in raw:
                d = _roc_to_iso(r["Date"])
                by_year.setdefault(int(d[:4]), {})[d] = {
                    "name": r["Name"].strip(), "note": (r.get("Description") or "").replace("<br>", " ").strip(),
                    "open_hint": any(k in r["Name"] for k in TWSE_OPEN_HINTS)}
            for y, days in by_year.items():  # 只覆蓋「有資料」的年度
                (CAL_DIR / f"twse_{y}.json").write_text(json.dumps(
                    {"source": TWSE_URL, "fetched": dt.datetime.now().isoformat(timespec="seconds"), "days": days},
                    ensure_ascii=False, indent=0))
        except Exception as e:
            print(f"[calendar] TWSE 假日表抓取失敗: {e}", file=sys.stderr)
    if p.exists():
        return json.loads(p.read_text())["days"]
    return None


def refresh(years=None):
    """重抓行事曆快取（每年至少一次；排程流程會自動檢查）。"""
    today = dt.date.today()
    years = years or [today.year, today.year + 1]
    out = {}
    for y in years:
        d = load_dgpa(y, refresh=True)
        t = load_twse(y, refresh=True)
        out[y] = {"dgpa_days": len(d) if d else 0, "twse_entries": len(t) if t else 0}
    _clear()
    return out


def ensure_fresh():
    """每日流程開頭呼叫：快取過期才抓，失敗不丟例外。"""
    try:
        y = dt.date.today().year
        for yy in (y, y + 1):
            load_dgpa(yy)
            load_twse(yy)
        _clear()
    except Exception as e:
        print("[calendar] ensure_fresh 失敗（沿用舊快取）:", e, file=sys.stderr)


def _overrides():
    if not OVERRIDES.exists():
        return {}, set()
    j = json.loads(OVERRIDES.read_text())
    return {c["date"]: c for c in j.get("closures", [])}, set(j.get("extra_open", []))


@lru_cache(maxsize=1)
def _db_info():
    """(有成交資料的日期集合, 最早, 最晚, DB holidays 表)"""
    from . import db
    con = db.connect()
    days = {r[0] for r in con.execute("SELECT DISTINCT date FROM prices WHERE market='TWSE'")}
    hol = {r[0]: r[1] for r in con.execute("SELECT date,name FROM holidays")}
    con.close()
    return days, (min(days) if days else None), (max(days) if days else None), hol


def _clear():
    _db_info.cache_clear()
    _year.cache_clear()


@lru_cache(maxsize=8)
def _year(y):
    return load_dgpa(y), load_twse(y)


def status(d, use_actual=True):
    """回傳 dict(open, why, source, firm)；firm=False 表示只是『預設開市』的推定（未來日期且無官方資訊）。"""
    if isinstance(d, str):
        d = dt.date.fromisoformat(d)
    ds = d.isoformat()
    days, lo, hi, dbhol = _db_info()
    ov_closed, ov_open = _overrides()
    if use_actual and ds in days:
        return dict(open=True, why="有實際成交資料", source="actual", firm=True)
    if d.weekday() >= 5:
        return dict(open=False, why="週末（補班日證交所也不開市）", source="weekend", firm=True)
    if ds in ov_open:
        return dict(open=True, why="overrides 指定開市", source="override", firm=True)
    if ds in ov_closed:
        return dict(open=False, why=ov_closed[ds]["reason"], source="override(臨時休市)", firm=True)
    dg, tw = _year(d.year)
    if dg and dg.get(ds, {}).get("holiday"):
        return dict(open=False, why=dg[ds]["desc"] or "政府機關放假", source="人事行政總處", firm=True)
    if tw and ds in tw and not tw[ds]["open_hint"]:
        return dict(open=False, why=tw[ds]["name"], source="證交所假日表", firm=True)
    if dg is None and tw is None and ds in dbhol and not any(k in dbhol[ds] for k in TWSE_OPEN_HINTS):
        return dict(open=False, why=dbhol[ds], source="DB holidays(離線備援)", firm=True)
    if use_actual and lo and hi and lo <= ds <= hi:
        return dict(open=False, why="行事曆為開市日，但資料區間內完全沒有成交資料→臨時休市（颱風假等）", source="推定(無資料)", firm=True)
    return dict(open=True, why="行事曆為開市日" if (dg or tw) else "無行事曆資料，預設開市",
                source="行事曆" if (dg or tw) else "預設", firm=bool(dg or tw))


def official_closed(d):
    """只看官方/已記錄資訊（週末、overrides、人事行政總處、證交所假日表），不看『有沒有資料』。回傳原因或 None。
    資料抓取用：避免把『資料還沒公布』誤判成休市而不再重試。"""
    if isinstance(d, str):
        d = dt.date.fromisoformat(d)
    ds = d.isoformat()
    if d.weekday() >= 5:
        return "週末"
    ov_closed, ov_open = _overrides()
    if ds in ov_open:
        return None
    if ds in ov_closed:
        return ov_closed[ds]["reason"]
    dg, tw = _year(d.year)
    if dg and dg.get(ds, {}).get("holiday"):
        return dg[ds]["desc"] or "政府機關放假"
    if tw and ds in tw and not tw[ds]["open_hint"]:
        return tw[ds]["name"]
    return None


def is_trading_day(d):
    return status(d)["open"]


def next_trading_day(d, n=1):
    if isinstance(d, str):
        d = dt.date.fromisoformat(d)
    x, k = d, 0
    for _ in range(400):
        x += dt.timedelta(days=1)
        if is_trading_day(x):
            k += 1
            if k >= n:
                return x
    return None


def prev_trading_day(d):
    if isinstance(d, str):
        d = dt.date.fromisoformat(d)
    x = d
    for _ in range(400):
        x -= dt.timedelta(days=1)
        if is_trading_day(x):
            return x
    return None


def upcoming_closures(start=None, days=45):
    """未來 days 天內的『平日休市』清單（週末不列）。"""
    start = start or dt.date.today()
    out = []
    for i in range(days + 1):
        x = start + dt.timedelta(days=i)
        if x.weekday() < 5:
            s = status(x)
            if not s["open"]:
                out.append({"date": x.isoformat(), "reason": s["why"], "source": s["source"]})
    return out


def cross_check(year):
    """人事行政總處 vs 證交所：平日休市的差異（供人工確認）。"""
    dg, tw = load_dgpa(year), load_twse(year)
    if not dg or not tw:
        return None
    dg_closed = {d for d, v in dg.items() if v["holiday"] and dt.date.fromisoformat(d).weekday() < 5}
    tw_closed = {d for d, v in tw.items() if not v["open_hint"] and dt.date.fromisoformat(d).weekday() < 5
                 and d.startswith(str(year))}
    return {"only_dgpa": sorted(dg_closed - tw_closed), "only_twse": sorted(tw_closed - dg_closed),
            "both": len(dg_closed & tw_closed)}


def retarget_predictions(verbose=True):
    """把『目標日其實休市』的收盤後預測改指向真正的下一交易日；盤前預測若目標日休市則標記作廢。
    只改 target_date 欄位並留下 target_date_original / note，不動任何預測內容。"""
    from .config import PRED_DIR
    changed = []
    for p in sorted(PRED_DIR.glob("*.json")):
        if p.name.startswith("_"):
            continue
        o = json.loads(p.read_text())
        t = o.get("target_date")
        if not t or is_trading_day(t):
            continue
        s = status(t)
        if p.stem.startswith("preopen_"):
            if o.get("void"):
                continue
            o["void"] = True
            o["void_reason"] = f"目標日 {t} 休市（{s['why']}），此盤前預測作廢，不列入檢討"
        else:
            nt = next_trading_day(dt.date.fromisoformat(o["pred_date"]))
            if nt is None or nt.isoformat() == t:
                continue
            o["target_date_original"] = t
            o["target_date"] = nt.isoformat()
            o["retarget_note"] = f"原目標日 {t} 休市（{s['why']}），改為下一個交易日 {nt.isoformat()}"
        p.write_text(json.dumps(o, ensure_ascii=False, indent=1))
        changed.append((p.name, t, o.get("target_date"), o.get("void", False)))
    if verbose:
        for c in changed:
            print("[calendar] 修正預測檔", c)
    return changed


def main(argv=None):
    a = (argv or sys.argv[1:]) or ["status"]
    cmd = a[0]
    d = dt.date.fromisoformat(a[1]) if len(a) > 1 and cmd in ("status", "next", "is-open", "add-closure") else dt.date.today()
    if cmd == "status":
        print(d, status(d))
    elif cmd == "next":
        print(next_trading_day(d))
    elif cmd == "upcoming":
        for r in upcoming_closures(days=int(a[1]) if len(a) > 1 else 60):
            print(r)
    elif cmd == "refresh":
        print(refresh())
    elif cmd == "check":
        for y in (d.year - 1, d.year, d.year + 1):
            print(y, cross_check(y))
    elif cmd == "is-open":  # shell 用：exit 0=開市
        sys.exit(0 if is_trading_day(d) else 1)
    elif cmd == "add-closure":  # python -m twq.calendar add-closure 2026-10-23 "原因" "來源"
        j = json.loads(OVERRIDES.read_text()) if OVERRIDES.exists() else {"closures": [], "extra_open": []}
        if not any(c["date"] == a[1] for c in j["closures"]):
            j["closures"].append({"date": a[1], "reason": a[2] if len(a) > 2 else "臨時休市",
                                  "source": a[3] if len(a) > 3 else ""})
            j["closures"].sort(key=lambda c: c["date"])
            OVERRIDES.write_text(json.dumps(j, ensure_ascii=False, indent=2))
        print("ok")
    elif cmd == "retarget":
        retarget_predictions()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()

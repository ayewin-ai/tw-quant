"""MOPS 彙總報表（綜合損益 t163sb04 / 資產負債 t163sb05）季資料。快取於 data/raw_fin/。
公告期限（point-in-time 可用日）：Q1→5/15, Q2→8/14, Q3→11/14, Q4(年報)→3/31(次年)。
"""
import io
import re
import time
from pathlib import Path

import pandas as pd
import requests

from . import db
from .config import DATA, UA

RAW = DATA / "raw_fin"
RAW.mkdir(exist_ok=True)
URL = "https://mopsov.twse.com.tw/mops/web/ajax_{rpt}"


def _get(rpt, typek, roc, season):
    p = RAW / f"{rpt}_{typek}_{roc}_{season:02d}.html"
    if p.exists() and p.stat().st_size > 50000:
        return p.read_text()
    for i in range(3):
        try:
            r = requests.post(URL.format(rpt=rpt), data={"encodeURIComponent": 1, "step": 1, "firstin": 1, "off": 1,
                              "isQuery": "Y", "TYPEK": typek, "year": roc, "season": f"{season:02d}"},
                              headers={"User-Agent": UA}, timeout=120)
            r.encoding = "utf-8"
            if len(r.text) > 50000:
                p.write_text(r.text)
                return r.text
            print("short response", rpt, typek, roc, season, len(r.text), flush=True)
        except Exception as e:
            print("err", rpt, typek, roc, season, e, flush=True)
        time.sleep(10 * (i + 1))
    return None


def _pick(cols, pats):
    for pat in pats:
        for c in cols:
            if re.search(pat, c):
                return c
    return None


FIELDS = {
    "sb04": {"revenue": [r"^營業收入$", r"^收益$", r"^利息淨收益$", r"淨收益"], "gross": [r"^營業毛利（毛損）淨額$", r"^營業毛利（毛損）$"],
             "op_income": [r"^營業利益"], "ni_parent": [r"^淨利（.*損）歸屬於母公司業主$"], "eps": [r"基本每股盈餘"]},
    "sb05": {"assets": [r"^資產總計$", r"^資產總額$"], "liab": [r"^負債總計$", r"^負債總額$"],
             "equity_parent": [r"^歸屬於母公司業主之權益合計$", r"^歸屬於母公司業主之權益$"], "equity": [r"^權益總計$", r"^權益總額$"],
             "cur_assets": [r"^流動資產$"], "cur_liab": [r"^流動負債$"], "cash": [r"^現金及約當現金$"]},
}


def parse(html, rpt):
    out = []
    for t in pd.read_html(io.StringIO(html)):
        if t.shape[1] < 5:
            continue
        t.columns = [str(c).replace(" ", "") for c in t.columns]
        code = _pick(t.columns, [r"公司代號"])
        if not code:
            continue
        d = pd.DataFrame({"code": t[code].astype(str).str.strip()})
        for k, pats in FIELDS[rpt].items():
            c = _pick(t.columns, pats)
            d[k] = pd.to_numeric(t[c], errors="coerce") if c else float("nan")
        out.append(d[d.code.str.fullmatch(r"\d{4}")])
    return pd.concat(out) if out else pd.DataFrame()


def init():
    con = db.connect()
    con.execute("""CREATE TABLE IF NOT EXISTS fin_q(code TEXT, year INT, season INT, revenue REAL, gross REAL,
      op_income REAL, ni_parent REAL, eps REAL, assets REAL, liab REAL, equity_parent REAL, equity REAL,
      cur_assets REAL, cur_liab REAL, cash REAL, PRIMARY KEY(code, year, season))""")
    con.commit()
    return con


def fetch(quarters):
    """quarters: list of (western_year, season). Income figures are cumulative YTD as published."""
    con = init()
    for y, s in quarters:
        roc = y - 1911
        parts = []
        for typek in ("sii", "otc"):
            a = _get("t163sb04", typek, roc, s); time.sleep(4)
            b = _get("t163sb05", typek, roc, s); time.sleep(4)
            if a is None or b is None:
                print("skip", y, s, typek, flush=True)
                continue
            pa, pb = parse(a, "sb04"), parse(b, "sb05")
            parts.append(pa.drop_duplicates("code").merge(pb.drop_duplicates("code"), on="code", how="outer"))
        if not parts:
            continue
        df = pd.concat(parts).drop_duplicates("code")
        df["year"], df["season"] = y, s
        cols = ["code", "year", "season", "revenue", "gross", "op_income", "ni_parent", "eps", "assets", "liab",
                "equity_parent", "equity", "cur_assets", "cur_liab", "cash"]
        con.executemany(f"INSERT OR REPLACE INTO fin_q({','.join(cols)}) VALUES({','.join('?' * len(cols))})",
                        df[cols].astype(object).where(df[cols].notna(), None).values.tolist())
        con.commit()
        print("fin", y, s, len(df), flush=True)
    con.close()


if __name__ == "__main__":
    import sys
    qs = [(y, s) for y in range(2023, 2027) for s in (1, 2, 3, 4) if (y, s) <= (2026, 2)]
    fetch(qs if len(sys.argv) < 2 else [tuple(map(int, a.split("Q"))) for a in sys.argv[1:]])


def refresh(today=None):
    """每日流程用：最新已過公告期限的一季若快取超過 7 天就重抓（公司陸續補件/更正）。"""
    import datetime as dt
    today = today or dt.date.today()
    cands = []
    for y in (today.year - 1, today.year):
        for s in (1, 2, 3, 4):
            dl = dt.date.fromisoformat((f"{y + 1}-" if s == 4 else f"{y}-") + DEADLINE_MD[s])
            if dl < today:
                cands.append((dl, y, s))
    _, y, s = max(cands)
    for p in RAW.glob(f"*_{y - 1911}_{s:02d}.html"):
        if time.time() - p.stat().st_mtime > 7 * 86400:
            p.unlink()
    fetch([(y, s)])


DEADLINE_MD = {1: "05-15", 2: "08-14", 3: "11-14", 4: "03-31"}

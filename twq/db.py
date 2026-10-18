import sqlite3
import threading
from .config import DB_PATH

_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS prices(
  date TEXT, code TEXT, name TEXT, market TEXT,
  open REAL, high REAL, low REAL, close REAL,
  volume REAL, value REAL, trades REAL, change REAL, ret REAL, pe REAL,
  PRIMARY KEY(date, code));
CREATE TABLE IF NOT EXISTS inst(
  date TEXT, code TEXT, foreign_net REAL, trust_net REAL, dealer_net REAL, total_net REAL,
  PRIMARY KEY(date, code));
CREATE TABLE IF NOT EXISTS margin(
  date TEXT, code TEXT, margin_bal REAL, short_bal REAL,
  PRIMARY KEY(date, code));
CREATE TABLE IF NOT EXISTS index_daily(
  date TEXT, name TEXT, close REAL, pct REAL,
  PRIMARY KEY(date, name));
CREATE TABLE IF NOT EXISTS revenue(
  ym TEXT, code TEXT, rev REAL, rev_last_month REAL, rev_last_year REAL, mom REAL, yoy REAL,
  PRIMARY KEY(ym, code));
CREATE TABLE IF NOT EXISTS exdiv(
  date TEXT, code TEXT, name TEXT, kind TEXT, amount REAL,
  PRIMARY KEY(date, code));
CREATE TABLE IF NOT EXISTS us_daily(
  date TEXT, symbol TEXT, close REAL, ret REAL,
  PRIMARY KEY(date, symbol));
CREATE TABLE IF NOT EXISTS stock_info(
  code TEXT PRIMARY KEY, name TEXT, market TEXT, industry TEXT);
CREATE TABLE IF NOT EXISTS holidays(
  date TEXT PRIMARY KEY, name TEXT, note TEXT);
CREATE TABLE IF NOT EXISTS fetch_log(
  date TEXT, source TEXT, status TEXT, rows INTEGER, ts TEXT DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY(date, source));
"""


def connect():
    con = sqlite3.connect(DB_PATH, timeout=120, check_same_thread=False)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    return con


def init():
    con = connect()
    con.executescript(SCHEMA)
    con.commit()
    con.close()


def upsert(table, cols, rows):
    if not rows:
        return 0
    q = f"INSERT OR REPLACE INTO {table}({','.join(cols)}) VALUES({','.join('?' * len(cols))})"
    with _lock:
        con = connect()
        con.executemany(q, rows)
        con.commit()
        con.close()
    return len(rows)


def log_fetch(date, source, status, rows=0):
    upsert("fetch_log", ["date", "source", "status", "rows"], [(date, source, status, rows)])


def fetched(source):
    con = connect()
    s = {r[0]: r[1] for r in con.execute("SELECT date,status FROM fetch_log WHERE source=?", (source,))}
    con.close()
    return s

"""CLI: python -m twq {update,backfill,validate,predict,review,daily,status}"""
import argparse
import datetime as dt
import sys
from zoneinfo import ZoneInfo

TW = ZoneInfo("Asia/Taipei")


def tw_today():
    return dt.datetime.now(TW).date()


def parse_date(s):
    return dt.date.fromisoformat(s) if s else None


def main(argv=None):
    p = argparse.ArgumentParser(prog="twq", description="台股量化判斷 + 每日檢討系統")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("backfill", help="回補歷史資料")
    a.add_argument("--start", default="2024-07-01")
    a.add_argument("--end", default=None)

    a = sub.add_parser("update", help="更新最近資料 (預設補到今天, 台灣時間)")
    a.add_argument("--days", type=int, default=10, help="往回檢查幾天")
    a.add_argument("--end", default=None)

    a = sub.add_parser("validate", help="walk-forward 驗證 + top-10 回測")
    a.add_argument("--start", default=None, help="測試期起日")

    a = sub.add_parser("predict", help="以 DATE 收盤資料預測下一交易日")
    a.add_argument("--date", default=None)
    a.add_argument("--no-update", action="store_true")
    a.add_argument("--top", type=int, default=None)

    a = sub.add_parser("review", help="檢討 DATE 那天產生的預測 (預設: 最近一個已有實際結果的預測)")
    a.add_argument("--date", default=None, help="預測產生日 (prediction date)")
    a.add_argument("--no-update", action="store_true")
    a.add_argument("--no-news", action="store_true")

    sub.add_parser("status", help="資料庫狀態")
    sub.add_parser("site", help="重建網站資料 docs/data/*.json")
    a = sub.add_parser("has-data", help="DATE 是否已有收盤資料 (exit 0=有, 1=無)")
    a.add_argument("--date", default=None)

    args = p.parse_args(argv)
    if args.cmd == "backfill":
        from . import fetch
        fetch.update(parse_date(args.start), parse_date(args.end) or tw_today())
    elif args.cmd == "update":
        from . import fetch
        end = parse_date(args.end) or tw_today()
        fetch.update(end - dt.timedelta(days=args.days), end)
    elif args.cmd == "validate":
        from . import model
        model.validate(args.start)
    elif args.cmd == "predict":
        from . import predict
        predict.run(parse_date(args.date), update=not args.no_update, top=args.top)
    elif args.cmd == "review":
        from . import review
        review.run(parse_date(args.date), update=not args.no_update, news=not args.no_news)
    elif args.cmd == "site":
        from . import site
        site.build()
    elif args.cmd == "has-data":
        from . import db
        d = (parse_date(args.date) or tw_today()).isoformat()
        con = db.connect()
        n = con.execute("SELECT COUNT(*) FROM prices WHERE date=? AND market='TWSE'", (d,)).fetchone()[0]
        m = con.execute("SELECT COUNT(*) FROM index_daily WHERE date=? AND name='TAIEX'", (d,)).fetchone()[0]
        print(f"{d}: TWSE rows={n}, TAIEX={'yes' if m else 'no'}")
        return 0 if (n > 0 and m > 0) else 1
    elif args.cmd == "status":
        from . import db
        con = db.connect()
        for t in ["prices", "inst", "margin", "index_daily", "revenue", "exdiv", "us_daily", "stock_info"]:
            n, lo, hi = con.execute(
                f"SELECT COUNT(*), MIN({'ym' if t=='revenue' else 'date' if t!='stock_info' else 'code'}),"
                f" MAX({'ym' if t=='revenue' else 'date' if t!='stock_info' else 'code'}) FROM {t}").fetchone()
            print(f"{t:12s} rows={n:>9} range={lo} ~ {hi}")
        print("trading days with prices:", con.execute("SELECT COUNT(DISTINCT date) FROM prices").fetchone()[0])
        for r in con.execute("SELECT source, status, COUNT(*) FROM fetch_log GROUP BY 1,2 ORDER BY 1,2"):
            print("  fetch_log", r)


if __name__ == "__main__":
    sys.exit(main())

#region Imports
from datetime import datetime, timedelta, timezone, date
from zoneinfo import ZoneInfo
from typing import Iterable, List, Dict, Any, Optional

from sqlalchemy import create_engine, select, func
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import insert as pg_insert

# Alpaca Market Data v2
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed
from alpaca.trading.requests import GetCalendarRequest
from alpaca.trading.client import TradingClient

from config import ConnectionParameters
from config import metadata, StockMaster, BarsDaily, BarsAftermarket10m
from config import BarsAftermarket4h, BarsPremarket4h
#endregion Imports

class Bars:
    """
    Service for fetching and storing daily OHLCV bars from Alpaca.
    """
    def __init__(self, feed: Optional[str] = DataFeed.SIP, batch_size: int = 200):
        """
        :param feed: Optional Alpaca feed (e.g., 'iex', 'sip').
        :param batch_size: How many symbols to request per call.
        """
        params = ConnectionParameters()
        self.engine = create_engine(params.get_db_url())
        self.Session = sessionmaker(bind=self.engine)

        # Alpaca Market Data client uses the same API keys
        creds = params.get_alpaca_params()
        self.data_client = StockHistoricalDataClient(creds["api_key"], creds["secret_key"])

        # Alpaca Trading client (calendar, account, orders, etc.)
        self.trading_client = TradingClient(creds["api_key"], creds["secret_key"])

        self.feed = feed
        self.batch_size = max(1, batch_size)

        # ensure tables exist
        metadata.create_all(self.engine)

    # ------------- Public API -------------

    def backfill_daily_last_two_years(self) -> None:
        """
        Fetches and stores last 2 years of daily bars for all active, tradable symbols.
        """
        session = self.Session()
        try:
            symbols = self._get_active_symbols(session)
            if not symbols:
                print("No active/tradable symbols found to backfill.")
                return

            end_dt = datetime.now(timezone.utc) - timedelta(days=1) # up to yesterday
            start_dt = end_dt - timedelta(days=730) # 2 years

            self._fetch_and_upsert_daily(session, symbols, start_dt, end_dt)
            session.commit()
            print(f"Backfill complete for {len(symbols)} symbols ({start_dt.date()} → {end_dt.date()}).")
        except Exception as e:
            session.rollback()
            raise
        finally:
            session.close()

    def sync_latest_daily(self) -> None:
        """
        Incrementally updates bars by fetching only missing days since the last stored bar per symbol.
        - If a symbol has no bars yet, it backfills last 2 years for that symbol.
        """
        session = self.Session()
        try:
            # Universe = active, tradable symbols
            symbols = self._get_active_symbols(session)
            if not symbols:
                print("No active/tradable symbols to sync.")
                return

            # Find last stored date per symbol (in one query)
            last_dates = self._get_last_dates_for_symbols(session, symbols)

            today_utc = datetime.now(timezone.utc).date() - timedelta(days=1) # up to yesterday

            # Partition: symbols with no data → backfill 2y; symbols with data → fetch from last+1
            need_backfill: List[str] = []
            ranged_requests: Dict[str, tuple[datetime, datetime]] = {}

            for sym in symbols:
                last_dt: Optional[date] = last_dates.get(sym)
                if last_dt is None:
                    need_backfill.append(sym)
                else:
                    start_date = last_dt + timedelta(days=1)
                    # Only fetch if there's at least one day to fill
                    if start_date <= today_utc:
                        ranged_requests[sym] = (
                            datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc),
                            datetime.now(timezone.utc)-timedelta(days=1),
                        )

            # Backfill group: last 2 years
            if need_backfill:
                end_dt = datetime.now(timezone.utc) - timedelta(days=1) # up to yesterday
                start_dt = end_dt - timedelta(days=730)
                self._fetch_and_upsert_daily(session, need_backfill, start_dt, end_dt)
                print(f"Backfilled {len(need_backfill)} symbols with last 2 years.")

            # Incremental ranges: to reduce requests, bucket symbols by same start/end (most will share 'today')
            if ranged_requests:
                # Simple approach: many will share the same end; we can group by start date string for batching
                buckets: Dict[str, List[str]] = {}
                for sym, (sdt, edt) in ranged_requests.items():
                    key = f"{sdt.date()}|{edt.date()}"
                    buckets.setdefault(key, []).append(sym)

                for key, bucket_syms in buckets.items():
                    s_str, e_str = key.split("|")
                    sdt = datetime.combine(datetime.fromisoformat(s_str).date(), datetime.min.time(), tzinfo=timezone.utc)
                    edt = datetime.combine(datetime.fromisoformat(e_str).date(), datetime.max.time(), tzinfo=timezone.utc)
                    self._fetch_and_upsert_daily(session, bucket_syms, sdt, edt)
                    print(f"Synced {len(bucket_syms)} symbols from {sdt.date()} to {edt.date()}.")

            session.commit()
            print("Incremental sync complete.")
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def populate_bars_aftermarket_10m(self):
        """
        Fetch the single aggregated 10-minute aftermarket candle (16:00–16:10 ET)
        for each open market day over the last 2 years until yesterday and store
        it into bars_aftermarket_10m.
        """
        session = self.Session()
        try:
            ny = ZoneInfo("America/New_York")
            end_local_date = datetime.now(timezone.utc).date() - timedelta(days=1)
            start_local_date = end_local_date - timedelta(days=730)

            # ✅ Query Alpaca for open market days only
            cal_req = GetCalendarRequest(start=start_local_date, end=end_local_date)
            open_days = [c.date for c in self.trading_client.get_calendar(cal_req)] # type: ignore

            symbols = self._get_active_symbols(session)

            # Process in batches to avoid URI too large
            for i in range(0, len(symbols), self.batch_size):
                batch = symbols[i : i + self.batch_size]

                for current in open_days:
                    rows_to_insert = []
                    ny_start = datetime(current.year, current.month, current.day, 16, 0, tzinfo=ny)
                    ny_end = ny_start + timedelta(minutes=9)
                    start_utc = ny_start.astimezone(timezone.utc)
                    end_utc = ny_end.astimezone(timezone.utc)

                    req = StockBarsRequest(
                        symbol_or_symbols=batch,
                        timeframe=TimeFrame(amount=10, unit=TimeFrameUnit.Minute),  # type: ignore
                        start=start_utc,
                        end=end_utc,
                        adjustment="raw",  # type: ignore
                        feed=self.feed,  # type: ignore
                    )
                    resp = self.data_client.get_stock_bars(req)

                    for sym in batch:
                        bars = resp.data.get(sym, [])  # type: ignore
                        for b in bars:
                            rows_to_insert.append({
                                "symbol": sym,
                                "session_date": ny_start.date(),
                                "open": _to_dec(b.open),
                                "high": _to_dec(b.high),
                                "low": _to_dec(b.low),
                                "close": _to_dec(b.close),
                                "vwap": _to_dec(getattr(b, "vwap", None)),
                                "volume": int(b.volume) if getattr(b, "volume", None) is not None else 0,
                                "trade_count": int(getattr(b, "trade_count", 0)) if getattr(b, "trade_count", None) is not None else None,
                                "time_utc": b.timestamp.replace(tzinfo=timezone.utc) if b.timestamp.tzinfo is None else b.timestamp.astimezone(timezone.utc),
                                "date_inserted": datetime.now(timezone.utc),
                            })

                    if rows_to_insert:
                        stmt = pg_insert(BarsAftermarket10m).values(rows_to_insert)
                        stmt = stmt.on_conflict_do_nothing(index_elements=["symbol", "time_utc"])
                        session.execute(stmt)
                        session.commit()
                        print(f"Inserted {len(rows_to_insert)} aftermarket 10m candles for {len(batch)} symbols in {start_utc.strftime("%Y-%m-%d")}.")
                    else:
                        print("No aftermarket 10m bars found in the requested range.")

        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def populate_bars_aftermarket_4h(self):
        """
        Fetch the single aggregated 4-hour aftermarket candle (16:00–20:00 ET)
        for each open market day over the last 2 years until yesterday and store
        it into bars_aftermarket_4h.
        """
        session = self.Session()
        try:
            ny = ZoneInfo("America/New_York")
            end_local_date = datetime.now(timezone.utc).date() - timedelta(days=1)
            start_local_date = end_local_date - timedelta(days=280)

            # ✅ Query Alpaca for open market days only
            cal_req = GetCalendarRequest(start=start_local_date, end=end_local_date)
            open_days = [c.date for c in self.trading_client.get_calendar(cal_req)] # type: ignore

            symbols = self._get_active_symbols(session)

            # Process in batches to avoid URI too large
            for i in range(10000, len(symbols), self.batch_size):
                batch = symbols[i : i + self.batch_size]

                for current in open_days:
                    rows_to_insert = []
                    ny_start = datetime(current.year, current.month, current.day, 16, 0, tzinfo=ny)
                    ny_end = ny_start + timedelta(hours=3, minutes=59)
                    start_utc = ny_start.astimezone(timezone.utc)
                    end_utc = ny_end.astimezone(timezone.utc)

                    req = StockBarsRequest(
                        symbol_or_symbols=batch,
                        timeframe=TimeFrame(amount=4, unit=TimeFrameUnit.Hour),  # type: ignore
                        start=start_utc,
                        end=end_utc,
                        adjustment="raw",  # type: ignore
                        feed=self.feed,  # type: ignore
                    )
                    resp = self.data_client.get_stock_bars(req)

                    for sym in batch:
                        bars = resp.data.get(sym, [])  # type: ignore
                        for b in bars:
                            rows_to_insert.append({
                                "symbol": sym,
                                "session_date": ny_start.date(),
                                "open": _to_dec(b.open),
                                "high": _to_dec(b.high),
                                "low": _to_dec(b.low),
                                "close": _to_dec(b.close),
                                "vwap": _to_dec(getattr(b, "vwap", None)),
                                "volume": int(b.volume) if getattr(b, "volume", None) is not None else 0,
                                "trade_count": int(getattr(b, "trade_count", 0)) if getattr(b, "trade_count", None) is not None else None,
                                "time_utc": b.timestamp.replace(tzinfo=timezone.utc) if b.timestamp.tzinfo is None else b.timestamp.astimezone(timezone.utc),
                                "date_inserted": datetime.now(timezone.utc),
                            })

                    if rows_to_insert:
                        stmt = pg_insert(BarsAftermarket4h).values(rows_to_insert)
                        stmt = stmt.on_conflict_do_nothing(index_elements=["symbol", "time_utc"])
                        session.execute(stmt)
                        session.commit()
                        print(f"Batch: {i}, inserted {len(rows_to_insert)} aftermarket 4h candles for {len(batch)} symbols in {start_utc.strftime("%Y-%m-%d")}.")
                    else:
                        print("No aftermarket 4h bars found in the requested range.")

        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def populate_bars_premarket_4h(self):
        """
        Fetch the single aggregated 4-hour premarket candle (03:00–07:00 ET)
        for each open market day over the last 2 years until yesterday and store
        it into bars_premarket_4h.
        """
        session = self.Session()
        try:
            ny = ZoneInfo("America/New_York")
            end_local_date = datetime.now(timezone.utc).date() #- timedelta(days=1)
            start_local_date = end_local_date - timedelta(days=730)

            # ✅ Query Alpaca for open market days only
            cal_req = GetCalendarRequest(start=start_local_date, end=end_local_date)
            open_days = [c.date for c in self.trading_client.get_calendar(cal_req)] # type: ignore

            symbols = self._get_active_symbols(session)

            # Process in batches to avoid URI too large
            for i in range(0, len(symbols), self.batch_size):
                batch = symbols[i : i + self.batch_size]

                for current in open_days:
                    rows_to_insert = []
                    ny_start = datetime(current.year, current.month, current.day, 3, 0, tzinfo=ny)
                    ny_end = ny_start + timedelta(hours=3, minutes=59)
                    start_utc = ny_start.astimezone(timezone.utc)
                    end_utc = ny_end.astimezone(timezone.utc)

                    req = StockBarsRequest(
                        symbol_or_symbols=batch,
                        timeframe=TimeFrame(amount=4, unit=TimeFrameUnit.Hour),  # type: ignore
                        start=start_utc,
                        end=end_utc,
                        adjustment="raw",  # type: ignore
                        feed=self.feed,  # type: ignore
                    )
                    resp = self.data_client.get_stock_bars(req)

                    for sym in batch:
                        bars = resp.data.get(sym, [])  # type: ignore
                        for b in bars:
                            rows_to_insert.append({
                                "symbol": sym,
                                "session_date": ny_start.date(),
                                "open": _to_dec(b.open),
                                "high": _to_dec(b.high),
                                "low": _to_dec(b.low),
                                "close": _to_dec(b.close),
                                "vwap": _to_dec(getattr(b, "vwap", None)),
                                "volume": int(b.volume) if getattr(b, "volume", None) is not None else 0,
                                "trade_count": int(getattr(b, "trade_count", 0)) if getattr(b, "trade_count", None) is not None else None,
                                "time_utc": b.timestamp.replace(tzinfo=timezone.utc) if b.timestamp.tzinfo is None else b.timestamp.astimezone(timezone.utc),
                                "date_inserted": datetime.now(timezone.utc),
                            })

                    if rows_to_insert:
                        stmt = pg_insert(BarsPremarket4h).values(rows_to_insert)
                        stmt = stmt.on_conflict_do_nothing(index_elements=["symbol", "time_utc"])
                        result = session.execute(stmt)
                        print(f"Batch: {(i/self.batch_size) + 1}, inserted {result.rowcount} of {len(rows_to_insert)} premarket 4h candles from {len(batch)} symbols in {start_utc.strftime("%Y-%m-%d")}.")
                    else:
                        print("No premarket 4h bars found in the requested range.")
                session.commit()
                print(f'Database commit after batch {(i/self.batch_size) + 1} complete.')

        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # ------------- Internals -------------

    def _get_active_symbols(self, session) -> List[str]:
        stmt = (
            select(StockMaster.c.symbol)
            .where(StockMaster.c.status == "active")
            .where(StockMaster.c.tradable == True)   # noqa: E712
        )
        return [row.symbol for row in session.execute(stmt).fetchall()]

    def _get_last_dates_for_symbols(self, session, symbols: Iterable[str]) -> Dict[str, Optional[date]]:
        """
        Returns {symbol: max(session_date)} for provided symbols; missing symbols not present in the table return None.
        """
        if not symbols:
            return {}

        stmt = (
            select(BarsDaily.c.symbol, func.max(BarsDaily.c.session_date))
            .where(BarsDaily.c.symbol.in_(list(symbols)))
            .group_by(BarsDaily.c.symbol)
        )
        result = {row.symbol: (row[1] if row[1] is not None else None) for row in session.execute(stmt).fetchall()}
        # Ensure all requested symbols are present in the dict
        for s in symbols:
            result.setdefault(s, None)
        return result

    def _fetch_and_upsert_daily(self, session, symbols: List[str], start_dt: datetime, end_dt: datetime) -> None:
        """
        Fetch bars in batches and upsert into DB.
        """
        if not symbols:
            return

        batch_count = 0
        for batch in self._chunks(symbols, self.batch_size):
            batch_count += 1
            req = StockBarsRequest(
                symbol_or_symbols=batch,
                timeframe=TimeFrame.Day, # type: ignore
                start=start_dt,
                end=end_dt,
                feed=self.feed  # type: ignore
            )
            resp = self.data_client.get_stock_bars(req)

            # The SDK returns a dict-like structure: resp.data[symbol] -> list[Bar]
            rows_to_insert: List[Dict[str, Any]] = []
            for sym in batch : # if only one symbol, still returns dict with one key
                bars = resp.data.get(sym, []) # type: ignore
                if not bars:
                    continue
                for b in bars:
                    # Attributes: b.timestamp, b.open, b.high, b.low, b.close, b.vwap, b.volume, b.trade_count
                    ts_utc: datetime = b.timestamp
                    sess_date = ts_utc.date()  # use UTC date for session
                    rows_to_insert.append({
                        "symbol": sym,
                        "session_date": sess_date,
                        "open": _to_dec(b.open),
                        "high": _to_dec(b.high),
                        "low": _to_dec(b.low),
                        "close": _to_dec(b.close),
                        "vwap": _to_dec(getattr(b, "vwap", None)),
                        "volume": int(b.volume) if getattr(b, "volume", None) is not None else 0,
                        "trade_count": int(getattr(b, "trade_count", 0)) if getattr(b, "trade_count", None) is not None else None,
                        "time_utc": ts_utc.replace(tzinfo=timezone.utc) if ts_utc.tzinfo is None else ts_utc.astimezone(timezone.utc),
                        "date_inserted": datetime.now(timezone.utc),
                    })
            try:
                if rows_to_insert:
                    # Upsert with ON CONFLICT DO NOTHING on (symbol, session_date)
                    stmt = pg_insert(BarsDaily).values(rows_to_insert)
                    # stmt = stmt.on_conflict_do_nothing(index_elements=["symbol", "session_date"])
                    session.execute(stmt)
                    print(f"Inserted {len(rows_to_insert)} bars for batch: {batch_count}")
            except Exception as e:
                print(f"Error inserting batch {batch_count}: {e}")

    @staticmethod
    def _chunks(seq: List[str], n: int) -> Iterable[List[str]]:
        for i in range(0, len(seq), n):
            yield seq[i:i + n]


def _to_dec(x) -> Optional[float]:
    if x is None:
        return None
    # Keep as float; PostgreSQL Numeric will coerce. If you prefer Decimal, convert here.
    return float(x)

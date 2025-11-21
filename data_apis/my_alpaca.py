import os
from zoneinfo import ZoneInfo
import pandas as pd

from config import ConnectionParameters
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.enums import DataFeed, Adjustment
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from data_apis import MyMarketIndexList
from data_apis import MyHelper


# ============================================================
#                    MyAlpacaStock
# ============================================================

class MyAlpacaStock:
    """
    Wrapper cleanly handling Alpaca API calls, local file loading,
    merging adjusted/unadjusted datasets, trimming data, and producing
    specialized event-based DataFrames.

    This class is intentionally refactored under Option B:
    clean code + SOLID principles while maintaining compatibility.
    """

    def __init__(self, symbol: str = "", feed: str = DataFeed.SIP):
        self._cfg = ConnectionParameters()
        self._client = StockHistoricalDataClient(
            self._cfg.ALPACA_API_KEY,
            self._cfg.ALPACA_SECRET_KEY
        )
        self._symbol = symbol
        self._feed = feed

        # state
        self._df: pd.DataFrame | None = None
        self._quantity: int | None = None
        self._unit: TimeFrameUnit | None = None

    # ------------------------------------------------------------
    # Internal Utilities
    # ------------------------------------------------------------

    def _to_ny(self, date_str: str) -> pd.Timestamp:
        """Convert YYYY-MM-DD string to NY timezone at 00:00:01."""
        return (
            pd.Timestamp(date_str)
            .tz_localize(ZoneInfo("America/New_York"))
            .floor("D")
            + pd.Timedelta(seconds=1)
        )

    def _to_ny_end(self, date_str: str) -> pd.Timestamp:
        """Convert YYYY-MM-DD string to NY timezone at 23:59:59."""
        return (
            pd.Timestamp(date_str)
            .tz_localize(ZoneInfo("America/New_York"))
            .ceil("D")
            - pd.Timedelta(seconds=1)
        )

    def _apply_standard_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert Alpaca MultiIndex -> flat schema with columns:
        ['symbol', 'timestamp', ...] and convert timestamp to NY.
        """
        df = df.copy()
        df.insert(0, "symbol", df.index.get_level_values("symbol"))
        df.insert(
            1,
            "timestamp",
            df.index.get_level_values("timestamp").tz_convert("America/New_York"),
        )
        return df.reset_index(drop=True)

    # ------------------------------------------------------------
    # API Queries
    # ------------------------------------------------------------

    def query_historical_data(
        self,
        start: str,
        end: str,
        quantity: int,
        unit: TimeFrameUnit = TimeFrameUnit.Minute,
        adjustment: Adjustment = Adjustment.ALL,
    ) -> None:
        """
        Query Alpaca API for historical bars.
        Saves standardized DataFrame internally.
        """

        self._quantity = quantity
        self._unit = unit

        start_ts = self._to_ny(start)
        end_ts = self._to_ny_end(end)

        request = StockBarsRequest(
            symbol_or_symbols=self._symbol,
            timeframe=TimeFrame(amount=quantity, unit=unit),
            start=start_ts,
            end=end_ts,
            adjustment=adjustment,
            feed=self._feed,
        )

        try:
            df = self._client.get_stock_bars(request).df
            self._df = self._apply_standard_schema(df) if not df.empty else pd.DataFrame()
        except Exception as exc:
            if "invalid symbol" in str(exc).lower():
                print(f"[WARN] Invalid symbol '{self._symbol}'. No data retrieved.")
                self._df = pd.DataFrame()
            else:
                raise exc

    # ------------------------------------------------------------
    # Load Local Files (Bronze Layer)
    # ------------------------------------------------------------

    def query_from_file_data(
        self,
        quantity: int,
        unit: TimeFrameUnit = TimeFrameUnit.Minute,
        adjustment: Adjustment = Adjustment.ALL,
    ) -> None:
        """
        Load historical bars from local CSV (bronze).
        """
        adj_folder = "adj_all" if adjustment == Adjustment.ALL else "adj_raw"
        path = (
            f"{self._cfg.PATH_TO_SAVE}{quantity}{unit.value}_history_{adj_folder}/"
            f"{self._symbol}.csv"
        )

        df = pd.read_csv(path, parse_dates=["timestamp"])
        df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

        self._df = df
        self._quantity = quantity
        self._unit = unit

    # ------------------------------------------------------------
    # DataFrame Operations
    # ------------------------------------------------------------

    def merge_raw_and_all_from_files(
        self,
        quantity: int,
        unit: TimeFrameUnit = TimeFrameUnit.Minute,
    ) -> None:
        """
        Merge adjusted ALL and adjusted RAW datasets with consistent schema.
        """
        self._quantity = quantity
        self._unit = unit

        # load ALL
        self.query_from_file_data(quantity, unit, adjustment=Adjustment.ALL)
        df_all = self.get_df().copy()

        # load RAW
        self.query_from_file_data(quantity, unit, adjustment=Adjustment.RAW)
        df_raw = self.get_df().copy()

        # merge on symbol+timestamp
        merged = pd.merge(
            df_all[["symbol", "timestamp", "close"]],
            df_raw,
            on=["symbol", "timestamp"],
            how="inner",
            suffixes=("_adj", ""),
        )
        self._df = merged

    def trim_df_by_date(self, start: str, end: str) -> None:
        """
        Filter dataframe between two boundary dates (YYYY-MM-DD).
        """
        if self._df is None:
            return

        start_ts = pd.Timestamp(start).tz_localize(ZoneInfo("America/New_York"))
        end_ts = pd.Timestamp(end).tz_localize(ZoneInfo("America/New_York"))

        mask = (self._df["timestamp"] >= start_ts) & (
            self._df["timestamp"] <= end_ts
        )
        self._df = self._df.loc[mask].reset_index(drop=True)

    # ------------------------------------------------------------
    # Event Builder Logic
    # ------------------------------------------------------------

    def build_event_df(self) -> pd.DataFrame:
        """
        Build trend-based event DataFrame using dynamic thresholds.
        (Refactor Option B applied: smaller blocks, cleaner logic)
        """

        if self._df is None or self._df.empty:
            return pd.DataFrame()

        df = self._df.reset_index(drop=True)

        # initial references
        symbol = df.loc[0, "symbol"]
        ref_price = float(df.loc[0, "open"])
        ref_factor = float(df.loc[0, "close_adj"]) / float(df.loc[0, "close"])

        start_time = df.loc[0, "timestamp"]
        low, high = ref_price, ref_price

        events = []

        for _, row in df.iterrows():
            factor_ratio = float(row["close_adj"]) / float(row["close"]) / ref_factor
            adj_high = row["high"] * factor_ratio
            adj_low = row["low"] * factor_ratio

            min_t, max_t = MyHelper.min_max_target(ref_price)

            # movement does not break threshold
            if not (adj_high >= max_t or adj_low <= min_t):
                high = max(high, row["high"])
                low = min(low, row["low"])
                continue

            # threshold broken → capture event
            events.append(
                self._capture_event(
                    symbol=symbol,
                    start_time=start_time,
                    end_time=row["timestamp"],
                    open_price=ref_price,
                    high=high,
                    low=low,
                    close=row["close"],
                    factor=factor_ratio,
                )
            )

            # reset reference values
            ref_price = row["close"]
            ref_factor = float(row["close_adj"]) / float(row["close"])
            start_time = row["timestamp"] + pd.Timedelta(minutes=1)
            low, high = ref_price, ref_price

        # final event for last candle
        last_row = df.iloc[-1]
        events.append(
            self._capture_event(
                symbol=symbol,
                start_time=start_time,
                end_time=last_row["timestamp"],
                open_price=ref_price,
                high=high,
                low=low,
                close=last_row["close"],
                factor=1.0,
            )
        )

        return pd.DataFrame(events)

    def _capture_event(
        self,
        symbol: str,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        open_price: float,
        high: float,
        low: float,
        close: float,
        factor: float,
    ) -> dict:
        """Small helper to build event record."""
        adj_close = close * factor
        pct_change = (adj_close - open_price) / open_price

        return {
            "symbol": symbol,
            "start_time": start_time,
            "end_time": end_time,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "close_adj": adj_close,
            "factor": factor,
            "pct_change": pct_change,
        }

    # ------------------------------------------------------------
    # Public Getter
    # ------------------------------------------------------------

    def get_df(self) -> pd.DataFrame:
        """Return current internal DataFrame (never None)."""
        return self._df if self._df is not None else pd.DataFrame()


# ============================================================
#                    MyAlpacaJob
# ============================================================

class MyAlpacaJob:
    """
    Job orchestrator: runs symbol loops, triggers downloads,
    merges, trims, and saves CSV outputs.

    Refactored under Option B:
    - clearer naming
    - smaller units
    - no deep structural changes
    """

    def __init__(
        self,
        start: str,
        end: str,
        quantity: int,
        unit: TimeFrameUnit,
        my_index_list: MyMarketIndexList | None = None,
        stock_list: list[str] | None = None,
        adjustment: Adjustment = Adjustment.ALL,
    ):
        self._cfg = ConnectionParameters()
        self._start = start
        self._end = end
        self._quantity = quantity
        self._unit = unit
        self._adjustment = adjustment

        self._index_list = my_index_list
        self._stock_list = stock_list or []

        if my_index_list is not None and not isinstance(
            my_index_list, MyMarketIndexList
        ):
            raise TypeError("my_index_list must be MyMarketIndexList")

    # ------------------------------------------------------------
    # Symbol Resolution
    # ------------------------------------------------------------

    def _resolve_symbols(self) -> list[str]:
        """Pick symbols based on provided stock_list, index list or folder."""
        if self._stock_list:
            return self._stock_list

        if self._index_list is not None:
            cons = self._index_list.get_selected_constituents()
            return cons["symbol"].unique().tolist()

        # fallback: read from local folder
        folder = (
            f"{self._cfg.PATH_TO_SAVE}"
            f"{self._quantity}{self._unit.value}_history_adj_{self._adjustment.value}/"
        )
        if not os.path.exists(folder):
            raise ValueError("No symbols available in local directory.")

        symbols = [f.replace(".csv", "") for f in os.listdir(folder)]
        return symbols

    # ------------------------------------------------------------
    # Save Single Symbol
    # ------------------------------------------------------------

    def _save_symbol(self, symbol: str, out_path: str) -> None:
        stock = MyAlpacaStock(symbol=symbol)
        stock.query_historical_data(
            start=self._start,
            end=self._end,
            quantity=self._quantity,
            unit=self._unit,
            adjustment=self._adjustment,
        )
        df = stock.get_df()

        if df.empty:
            print(f"[WARN] No data for {symbol}")
            return

        df.to_csv(f"{out_path}{symbol}.csv", index=False)
        print(f"[OK] Saved: {out_path}{symbol}.csv")

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------

    def save_to_folder(self, limit: int = 999999) -> None:
        """
        Download all symbols (up to limit) and write CSV to Bronze folders.
        """
        out_path = f"{self._cfg.PATH_TO_SAVE}{self._quantity}{self._unit.value}_history/"
        symbols = self._resolve_symbols()[:limit]

        os.makedirs(out_path, exist_ok=True)

        for symbol in symbols:
            # skip if already exists
            file_path = f"{out_path}{symbol}.csv"
            if os.path.exists(file_path):
                print(f"[SKIP] {symbol} already exists.")
                continue
            self._save_symbol(symbol, out_path)


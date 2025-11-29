from re import sub
from config import ConnectionParameters
from alpaca.data.enums import DataFeed, Adjustment
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.trading.client import TradingClient
from typing import Optional, List, Iterable
import pandas as pd
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

class Alpa:

    def __init__(self, feed: Optional[str] = DataFeed.SIP, batch_size: int = 200):
        """
        :param feed: Optional Alpaca feed (e.g., 'iex', 'sip').
        :param batch_size: How many symbols to request per call.
        """
        params = ConnectionParameters()

        # Alpaca Market Data client uses the same API keys
        creds = params.get_alpaca_params()
        self.shdc = StockHistoricalDataClient(creds["api_key"], creds["secret_key"])

        # Alpaca Trading client (calendar, account, orders, etc.)
        self.tc = TradingClient(creds["api_key"], creds["secret_key"])

        self.feed = feed
        self.batch_size = max(1, batch_size)

    def relative_exp_tr(self, alpha_soft: float = 0.01, alpha_peak: float = 0.1):

        if alpha_peak <= alpha_soft:
            raise ValueError('Value of soft alpha should be less than peak alpha')
        
        # Previous close per symbol
        self.dfm["prev_close"] = self.dfm.groupby("symbol")["close"].shift(1)
        self.dfm["prev_close"] = self.dfm.groupby("symbol")["prev_close"].transform(
            lambda s: s.fillna(self.dfm.loc[s.index, "open"])
        )

        # True range
        self.dfm["tr"] = self.dfm[["high", "low", "prev_close"]].bfill(axis=1).max(axis=1) - \
                self.dfm[["low", "prev_close"]].bfill(axis=1).min(axis=1)

        # Relative true range (percentage)
        self.dfm["rel_tr"] = self.dfm["tr"] / self.dfm["prev_close"]

        # Exponentially weighted average (like ATR but relative)
        self.dfm["exp_rel_tr_soft"] = (
            self.dfm.groupby("symbol")["rel_tr"]
            .transform(lambda x: x.ewm(adjust=False, alpha=alpha_soft).mean())
        )

        self.dfm["exp_rel_tr_peak"] = (
            self.dfm.groupby("symbol")["rel_tr"]
            .transform(lambda x: x.ewm(adjust=False, alpha=alpha_peak).mean())
        )

    def get_30min_bars(
        self,
        symbols: list[str],
        end: str,
        days: int = 365
    ):
        start_date = pd.Timestamp(end, tz="America/New_York") - pd.Timedelta(days=days)
        end_date = pd.Timestamp(end, tz="America/New_York")

        all_dfs = []
        for batch in self._chunks(symbols, self.batch_size):
            request = StockBarsRequest(
                symbol_or_symbols=batch,
                timeframe=TimeFrame(amount=30, unit=TimeFrameUnit.Minute),  # type: ignore
                start=start_date,
                end=end_date,
                adjustment=Adjustment.ALL,  # type: ignore
                feed=DataFeed.SIP,  # type: ignore
            )
            df_batch = self.shdc.get_stock_bars(request).df  # type: ignore
            all_dfs.append(df_batch)

        # --- Concatenate all results into one DataFrame ---
        self.df_30m = pd.concat(all_dfs).sort_index()
        del all_dfs
        # --- Adjust gaps and timezones ---
        self.df_30m = self.adjust_30min_tz_gaps(self.df_30m)

    def get_minute_bars(
        self,
        symbols: list[str],
        start: str,
        days: int = 1,
        minutes: int = 1
    ):
        start_date = pd.Timestamp(start, tz="America/New_York")
        end_date = start_date + pd.Timedelta(days=days)

        all_dfs = []
        for batch in self._chunks(symbols, self.batch_size):
            request = StockBarsRequest(
                symbol_or_symbols=batch,
                timeframe=TimeFrame(amount=minutes, unit=TimeFrameUnit.Minute),  # type: ignore
                start=start_date,
                end=end_date,
                adjustment=Adjustment.RAW,  # type: ignore
                feed=DataFeed.SIP,  # type: ignore
            )
            df_batch = self.shdc.get_stock_bars(request).df  # type: ignore
            all_dfs.append(df_batch)

        # --- Concatenate all results into one DataFrame ---
        self.dfm = pd.concat(all_dfs).sort_index()
        del all_dfs

        # --- Adjust gaps and timezones ---
        self.dfm = self.adjust_minutes_tz_gaps(self.dfm)
    
    def adjust_30min_tz_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        # --- Build full index for all symbols × full timeline ---
        symbols = df.index.get_level_values("symbol").unique()

        # --- Build full NY-localized timeline restricted to 04:00–20:00 ---
        start = df.index.get_level_values("timestamp").min().tz_convert("America/New_York").normalize()
        end   = df.index.get_level_values("timestamp").max().tz_convert("America/New_York").normalize() + pd.Timedelta(days=1)

        # full range at 30min only during weekdays, removing weekends
        times = pd.date_range(start, end, freq=f"30min", tz="America/New_York")
        times = times[times.dayofweek < 5]
        
        # filter only 04:00–20:00
        times = times[(times.time >= pd.to_datetime("04:00").time()) &
                    (times.time <= pd.to_datetime("19:59").time())]

        full_index = pd.MultiIndex.from_product([symbols, times],
                                                names=["symbol", "timestamp"])

        # --- Reindex to full grid (fills missing rows with NaN) ---
        df2 = df.reindex(full_index)
        del df
    
        # (Optional) Fill missing with 0 if required
        ffill_cols = ["open", "high", "low", "vwap"]
        zero_cols  = ["volume", "trade_count"]

        # --- Forward fill 'close' only ---
        df2["close"] = df2["close"].groupby(level="symbol").ffill().bfill()

        # --- Fill other OHLC fields from 'close' ---
        for col in ffill_cols:
            df2[col] = df2[col].fillna(df2["close"])

        # --- Fill zeros ---
        df2[zero_cols] = df2[zero_cols].fillna(0)

        # --- Reset index for plotting ---
        df2.reset_index(inplace=True)

        # --- Save back again to self.df ---
        return df2

    def adjust_minutes_tz_gaps(self, df: pd.DataFrame) ->  pd.DataFrame:
        # --- Build full index for all symbols × full timeline ---
        symbols = df.index.get_level_values("symbol").unique()

        # --- Build full NY-localized timeline restricted to 04:00–20:00 ---
        start = df.index.get_level_values("timestamp").min().tz_convert("America/New_York").normalize()
        end   = df.index.get_level_values("timestamp").max().tz_convert("America/New_York").normalize() + pd.Timedelta(days=1)

        # full range at 1min
        times = pd.date_range(start, end, freq=f"1min", tz="America/New_York")

        # filter only 04:00–20:00
        times = times[(times.time >= pd.to_datetime("04:00").time()) &
                    (times.time <= pd.to_datetime("19:59").time())]

        full_index = pd.MultiIndex.from_product([symbols, times],
                                                names=["symbol", "timestamp"])

        # --- Reindex to full grid (fills missing rows with NaN) ---
        df2 = df.reindex(full_index)
        del df
    
        # (Optional) Fill missing with 0 if required
        ffill_cols = ["open", "high", "low", "vwap"]
        zero_cols  = ["volume", "trade_count"]

        # --- Forward fill 'close' only ---
        df2["close"] = df2["close"].groupby(level="symbol").ffill().bfill()

        # --- Fill other OHLC fields from 'close' ---
        for col in ffill_cols:
            df2[col] = df2[col].fillna(df2["close"])

        # --- Fill zeros ---
        df2[zero_cols] = df2[zero_cols].fillna(0)

        # --- Reset index for plotting ---
        df2.reset_index(inplace=True)

        # --- Save back again to self.df ---
        return df2

    def adjust_daily_tz_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        # --- Build full index for all symbols × full timeline ---
        symbols = df.index.get_level_values("symbol").unique()

        # --- Build full NY-localized timeline and set time to premarket open ---
        start = df.index.get_level_values("timestamp").min().tz_convert("America/New_York")
        end   = df.index.get_level_values("timestamp").max().tz_convert("America/New_York")

        # full range at 1D
        times = pd.date_range(start, end, freq="1D", tz="America/New_York")

        full_index = pd.MultiIndex.from_product([symbols, times],
                                                names=["symbol", "timestamp"])

        # --- Reindex to full grid (fills missing rows with NaN) ---
        df2 = df.reindex(full_index)
        del df
    
        # (Optional) Fill missing with 0 if required
        ffill_cols = ["open", "high", "low", "vwap"]
        zero_cols  = ["volume", "trade_count"]

        # --- Forward fill 'close' only ---
        df2["close"] = df2["close"].groupby(level="symbol").ffill().bfill()

        # --- Fill other OHLC fields from 'close' ---
        for col in ffill_cols:
            df2[col] = df2[col].fillna(df2["close"])

        # --- Fill zeros ---
        df2[zero_cols] = df2[zero_cols].fillna(0)

        # --- Reset index for plotting ---
        df2.reset_index(inplace=True)

        # --- Save back again to self.df ---
        return df2

    def get_volume_symbols(
        self,
        symbols: list[str],
        date: str,
        min: float = 100_000,
        max: float = 10_000_000,
        days_range: int = 7
    ) -> list[str] :
        end_date = pd.Timestamp(date, tz="America/New_York")
        end_date = end_date - pd.Timedelta(days=1)
        start_date = end_date - pd.Timedelta(days=days_range)

        all_dfs = []
        for batch in self._chunks(symbols, self.batch_size):
            request = StockBarsRequest(
                symbol_or_symbols=batch,
                timeframe=TimeFrame(amount=1, unit=TimeFrameUnit.Day), # type: ignore
                start=start_date,
                end=end_date,
                adjustment=Adjustment.RAW,
                feed=DataFeed.SIP,
            )
            df_batch = self.shdc.get_stock_bars(request).df  # type: ignore
            all_dfs.append(df_batch)

        # --- Concatenate all results into one DataFrame ---
        df = pd.concat(all_dfs).sort_index()

        # --- Compute traded amount (volume × VWAP) ---
        df["amount_traded"] = df["volume"] * df["vwap"]

        # --- Group by symbol and check if ALL days are within range ---
        mask = df.groupby("symbol")["amount_traded"].apply(
            lambda x: (x.between(min, max)).all()
        )

        # mask is a boolean Series indexed by symbol
        symbols_in_range = mask[mask].index.tolist()

        return symbols_in_range
        
    def minutes_above(self, threshold_soft: float = 0.001, threshold_peak: float = 0.01):
        self.df_sym = self.dfm.groupby("symbol")
        self.df_sym = (
            self.df_sym
            .agg(
                soft_above=("exp_rel_tr_soft", lambda x: (x > threshold_soft).sum()),
                peak_above=("exp_rel_tr_peak", lambda x: (x > threshold_peak).sum()),
            )
            .reset_index()
        )

    def calculate_day_metrics(
        self,
        symbols: list[str],
        date: str,
        time_back: list[int] = [7,30,180]
    ):
        end_date = pd.Timestamp(date, tz="America/New_York") - pd.Timedelta(days=1)
        start_date = end_date - pd.Timedelta(days=max(time_back)-1)
        all_dfs = []
        for batch in self._chunks(symbols, self.batch_size):
            request = StockBarsRequest(
                symbol_or_symbols=batch,
                timeframe=TimeFrame(amount=1, unit=TimeFrameUnit.Day),  # type: ignore
                start=start_date,
                end=end_date,
                adjustment=Adjustment.ALL,
                feed=DataFeed.SIP,
            )
            all_dfs.append(self.shdc.get_stock_bars(request).df) # type: ignore

        # --- Concatenate all results into one DataFrame ---
        self.dfd_prev = pd.concat(all_dfs).sort_index()
        del all_dfs
        self.dfd_prev = self.adjust_daily_tz_gaps(df=self.dfd_prev)

        # --- Calculate metrics ---
        # Calculate return and variance of % daily changes over the period in dataframe by symbol dfs
        def calc_returns(x):
            out = {}
            for d in time_back:
                sub = x.tail(d)
                if len(sub) >= d:
                    ret = sub["close"].iloc[-1] / sub["open"].iloc[0] - 1
                    pct_values = sub["close"] / sub["open"].iloc[0]
                    std = pct_values.std()
                    out[f"return_{d}d"] = ret
                    out[f"std_{d}d"] = std
                else:
                    out[f"return_{d}d"] = None
                    out[f"std_{d}d"] = None
            return pd.Series(out)

        returns = self.dfd_prev.groupby("symbol").apply(calc_returns, include_groups=False).reset_index() # type: ignore
        returns = returns.drop(columns=[c for c in returns.columns if c.endswith("1d") and "std" in c])
        self.df_sym = pd.merge(self.df_sym, returns, on="symbol", how="left").reset_index(drop=True) # type: ignore

    def calculate_minute_metrics(
        self,
        symbols: list[str],
        date: str,
        time_back: list[int] = [60, 240]
    ):
        end_date = pd.Timestamp(date, tz="America/New_York")
        start_date = end_date - pd.Timedelta(days=1)  # beginning of previous day
        all_dfs = []
        for batch in self._chunks(symbols, self.batch_size):
            request = StockBarsRequest(
                symbol_or_symbols=batch,
                timeframe=TimeFrame.Minute,  # type: ignore
                start=start_date,
                end=end_date,
                adjustment=Adjustment.ALL,
                feed=DataFeed.SIP,
            )
            all_dfs.append(self.shdc.get_stock_bars(request).df)  # type: ignore

         # --- Concatenate all results into one DataFrame ---
        self.dfm_prev = pd.concat(all_dfs).sort_index()
        del all_dfs
         # --- Adjust gaps and timezones ---
        self.dfm_prev = self.adjust_minutes_tz_gaps(df=self.dfm_prev)

        def calc_metrics(x):
            out = {}
            for m in time_back:
                sub = x.tail(m)
                if len(sub) >= m:
                    ret = sub["close"].iloc[-1] / sub["open"].iloc[0] - 1
                    pct_values = sub["close"] / sub["open"].iloc[0]
                    std = pct_values.std()
                    out[f"return_{m}m"] = ret
                    out[f"std_{m}m"] = std
                else:
                    out[f"return_{m}m"] = None
                    out[f"std_{m}m"] = None
            return pd.Series(out)

        returns = self.dfm_prev.groupby("symbol").apply(calc_metrics, include_groups=False).reset_index() # type: ignore
        returns = returns.drop(columns=[c for c in returns.columns if c.endswith("1m") and "std" in c])
        self.df_sym = pd.merge(self.df_sym, returns, on="symbol", how="left").reset_index(drop=True)  # type: ignore

    @staticmethod
    def _chunks(seq: List[str], n: int) -> Iterable[List[str]]:
        for i in range(0, len(seq), n):
            yield seq[i:i + n]
import pandas as pd
import numpy as np
from zoneinfo import ZoneInfo
import datetime
from pandas.tseries.holiday import USFederalHolidayCalendar
import pandas_market_calendars as mcal


class MyStockFunctions:

    # ============================================================
    # ====================== HELPERS =============================
    # ============================================================

    @staticmethod
    def _to_ny(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure timestamp is tz-aware NY time."""
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["timestamp"] = df["timestamp"].dt.tz_convert("America/New_York")
        return df

    @staticmethod
    def _add_date(df: pd.DataFrame) -> pd.DataFrame:
        """Add date column based on NY timestamp."""
        df = df.copy()
        df = MyStockFunctions._to_ny(df)
        df["date"] = df["timestamp"].dt.date
        return df

    @staticmethod
    def _add_4am_timestamp(df: pd.DataFrame) -> pd.DataFrame:
        """Add timestamp aligned to 4:00am NY for daily output."""
        df = df.copy()
        df['timestamp'] = (
            pd.to_datetime(df['date'])
            .dt.tz_localize(ZoneInfo("America/New_York"))
            + pd.Timedelta(hours=4)
        )
        return df

    @staticmethod
    def _premarket(df: pd.DataFrame) -> pd.DataFrame:
        """Return only candles in premarket 04:00–09:30 NY."""
        df = MyStockFunctions._add_date(df)
        df["time"] = df["timestamp"].dt.time
        pre_start = datetime.time(4, 0)
        pre_end = datetime.time(9, 30)
        return df[(df["time"] >= pre_start) & (df["time"] < pre_end)]

    @staticmethod
    def _regular_hours(df: pd.DataFrame) -> pd.DataFrame:
        """Return only candles in regular session 09:30–16:00 NY."""
        df = MyStockFunctions._add_date(df)
        df["time"] = df["timestamp"].dt.time
        reg_start = datetime.time(9, 30)
        reg_end = datetime.time(16, 0)
        return df[(df["time"] >= reg_start) & (df["time"] < reg_end)]

    @staticmethod
    def _daily_last_close(df: pd.DataFrame) -> pd.DataFrame:
        """Return last close per date."""
        df = MyStockFunctions._add_date(df)
        return df.groupby("date")["close"].last().reset_index()

    @staticmethod
    def _daily_ohlc(df: pd.DataFrame) -> pd.DataFrame:
        """Return daily OHLC aggregated from intraday."""
        df = MyStockFunctions._add_date(df)
        return df.groupby("date").agg(
            day_open=("open", "first"),
            day_high=("high", "max"),
            day_low=("low", "min"),
            day_close=("close", "last"),
            symbol=("symbol", "first")
        ).reset_index()

    # ============================================================
    # ====================== FEATURES =============================
    # ============================================================

    @staticmethod
    def compute_prev_day_return(df):
        df = MyStockFunctions._add_date(df)
        daily_close = df.groupby(['symbol', 'date']).agg(
            close_daily=('close', 'last')
        ).reset_index()

        daily_close["prev_day_return"] = (
            daily_close.groupby("symbol")["close_daily"].pct_change().shift(1)
        )

        daily_close = daily_close.dropna(subset=["prev_day_return"])
        daily_close = MyStockFunctions._add_4am_timestamp(daily_close)
        return daily_close[['symbol', 'timestamp', 'prev_day_return']]

    @staticmethod
    def compute_std_return_last5d(df):
        df = MyStockFunctions._add_date(df)
        df = df.sort_values("timestamp")

        df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
        df.loc[df["date"] != df["date"].shift(1), "log_ret"] = np.nan

        daily = df.groupby("date").agg(
            daily_log_return=("log_ret", "sum"),
            symbol=("symbol", "first")
        ).reset_index()

        daily["std_return_last5d"] = (
            daily["daily_log_return"].rolling(5).std()
        )

        daily = MyStockFunctions._add_4am_timestamp(daily)
        return daily[["symbol", "timestamp", "std_return_last5d"]]

    @staticmethod
    def compute_range_rel_last1d(df):
        daily = MyStockFunctions._daily_ohlc(df)
        daily["range_rel"] = (
            (daily["day_high"] - daily["day_low"]) /
            daily["day_close"]
        )

        daily["range_rel_last1d"] = daily["range_rel"].shift(1)
        daily = MyStockFunctions._add_4am_timestamp(daily)
        return daily[["symbol", "timestamp", "range_rel_last1d"]]

    @staticmethod
    def compute_rvol_20d(df):
        df = MyStockFunctions._add_date(df)
        df["dollar_volume"] = df["volume"] * df["vwap"]

        daily = df.groupby("date").agg(
            day_dollar_volume=("dollar_volume", "sum"),
            symbol=("symbol", "first")
        ).reset_index()

        daily["dvol_ma20"] = daily["day_dollar_volume"].rolling(20).mean().shift(1)
        daily["rvol_20d"] = daily["day_dollar_volume"] / daily["dvol_ma20"]

        daily = MyStockFunctions._add_4am_timestamp(daily)
        return daily[["symbol", "timestamp", "rvol_20d"]]

    @staticmethod
    def compute_gap_pct(df):
        df = MyStockFunctions._add_date(df)
        df["time"] = df["timestamp"].dt.time

        all_days = (
            df.groupby("date")
            .agg(symbol=("symbol", "first"))
            .reset_index()
        )

        premarket = MyStockFunctions._premarket(df)
        pre = (
            premarket.groupby("date")
            .agg(pre_open=("open", "first"))
            .reset_index()
        )

        daily = all_days.merge(pre, on="date", how="left")
        daily["pre_open"] = daily["pre_open"].fillna(0)

        last_close = MyStockFunctions._daily_last_close(df)
        last_close = last_close.rename(columns={"close": "last_close"})

        daily = daily.merge(last_close, on="date", how="left")
        daily["prev_close"] = daily["last_close"].shift(1)

        daily["gap_pct"] = 0.0
        mask = (daily["prev_close"].notna()) & (daily["pre_open"] != 0)
        daily.loc[mask, "gap_pct"] = (
            (daily.loc[mask, "pre_open"] - daily.loc[mask, "prev_close"])
            / daily.loc[mask, "prev_close"]
        )

        daily = MyStockFunctions._add_4am_timestamp(daily)
        return daily[["symbol", "timestamp", "gap_pct"]]

    @staticmethod
    def compute_dist_max_20d(df):
        daily = (
            MyStockFunctions._daily_last_close(df)
            .rename(columns={"close": "close_daily"})
        )

        daily["max_20d_prev"] = daily["close_daily"].shift(1).rolling(20).max()
        daily["dist_max_20d"] = (
            (daily["close_daily"].shift(1) - daily["max_20d_prev"])
            / daily["max_20d_prev"]
        )

        daily = MyStockFunctions._add_4am_timestamp(daily)
        return daily[["symbol", "timestamp", "dist_max_20d"]]

    @staticmethod
    def compute_upper_wick_ratio_last1d(df):
        df = MyStockFunctions._add_date(df)
        df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
        df["candle_range"] = df["high"] - df["low"]

        df["upper_wick_ratio"] = np.where(
            df["candle_range"] == 0,
            0,
            df["upper_wick"] / df["candle_range"]
        )

        daily = df.groupby("date").agg(
            upper_wick_ratio_last1d=("upper_wick_ratio", "mean"),
            symbol=("symbol", "first")
        ).reset_index()

        daily["upper_wick_ratio_last1d"] = daily["upper_wick_ratio_last1d"].shift(1)
        daily = MyStockFunctions._add_4am_timestamp(daily)
        return daily[["symbol", "timestamp", "upper_wick_ratio_last1d"]]

    @staticmethod
    def compute_lower_wick_ratio_last1d(df):
        df = MyStockFunctions._add_date(df)
        df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
        df["candle_range"] = df["high"] - df["low"]

        df["lower_wick_ratio"] = np.where(
            df["candle_range"] == 0,
            0,
            df["lower_wick"] / df["candle_range"]
        )

        daily = df.groupby("date").agg(
            lower_wick_ratio_last1d=("lower_wick_ratio", "mean"),
            symbol=("symbol", "first")
        ).reset_index()

        daily["lower_wick_ratio_last1d"] = daily["lower_wick_ratio_last1d"].shift(1)
        daily = MyStockFunctions._add_4am_timestamp(daily)
        return daily[["symbol", "timestamp", "lower_wick_ratio_last1d"]]

    @staticmethod
    def compute_weekday(df):
        df = MyStockFunctions._add_date(df)
        df["weekday"] = pd.to_datetime(df["date"]).dt.day_name()

        out = df.groupby(["symbol", "date"])["weekday"].first().reset_index()
        out = MyStockFunctions._add_4am_timestamp(out)
        return out[["symbol", "timestamp", "weekday"]]

    @staticmethod
    def compute_weekday_cyclic(df):
        df = MyStockFunctions._add_date(df)
        df["weekday"] = pd.to_datetime(df["date"]).dt.weekday
        df = df[df["weekday"] < 5]

        df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 5)
        df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 5)

        out = df[["symbol", "date", "weekday_sin", "weekday_cos"]].drop_duplicates()
        out = MyStockFunctions._add_4am_timestamp(out)
        return out[["symbol", "timestamp", "weekday_sin", "weekday_cos"]]

    @staticmethod
    def compute_days_since_holiday_general(df):
        df = MyStockFunctions._add_date(df)
        df = df.sort_values("date").drop_duplicates(subset=["date", "symbol"])

        cal = USFederalHolidayCalendar()
        holidays = cal.holidays(
            start=df["date"].min(), end=df["date"].max()
        ).date

        holiday_dates = np.array(holidays)
        vals = []
        for d in df["date"]:
            idx = np.searchsorted(holiday_dates, d, side='right') - 1
            if idx >= 0:
                vals.append((d - holiday_dates[idx]).days)
            else:
                vals.append(np.nan)

        df["days_since_holiday"] = vals
        df = MyStockFunctions._add_4am_timestamp(df)
        return df[["symbol", "timestamp", "days_since_holiday"]]

    @staticmethod
    def compute_hhi_premarket_volume(df):
        pre = MyStockFunctions._premarket(df)

        def hhi(vols):
            total = vols.sum()
            if total == 0:
                return np.nan
            p = vols / total
            return (p ** 2).sum()

        hhi_daily = (
            pre.groupby(["symbol", "date"])["volume"].apply(hhi).reset_index()
        )
        hhi_daily = hhi_daily.rename(columns={"volume": "hhi_premarket_volume"})
        hhi_daily = MyStockFunctions._add_4am_timestamp(hhi_daily)
        return hhi_daily

    @staticmethod
    def compute_premarket_zscore(df, lookback=20):
        df = MyStockFunctions._add_date(df)
        symbol = df["symbol"].iloc[0]

        pre = MyStockFunctions._premarket(df)
        pm_daily = pre.groupby("date").agg(
            pm_open=("open", "first"),
            pm_last=("close", "last")
        ).reset_index()

        pm_daily["premarket_return"] = (
            (pm_daily["pm_last"] - pm_daily["pm_open"]) / pm_daily["pm_open"]
        )

        daily = MyStockFunctions._daily_last_close(df)
        daily["daily_return"] = daily["close"].pct_change()
        daily["mu_m"] = daily["daily_return"].rolling(lookback).mean()
        daily["sigma_m"] = daily["daily_return"].rolling(lookback).std()

        merged = pm_daily.merge(daily, on="date", how="left")
        merged["z_pm"] = (
            (merged["premarket_return"] - merged["mu_m"]) /
            merged["sigma_m"]
        )

        merged["symbol"] = symbol
        merged = MyStockFunctions._add_4am_timestamp(merged)
        return merged[["symbol", "timestamp", "z_pm"]]

    @staticmethod
    def compute_premarket_avg_trade_size_ratio(df, lookback=20, eps=0.01):
        df = MyStockFunctions._add_date(df)
        symbol = df["symbol"].iloc[0]

        pre = MyStockFunctions._premarket(df)
        if pre.empty:
            return pd.DataFrame(columns=["symbol", "timestamp", "R_PM_hist"])

        pm_daily = pre.groupby("date").agg(
            total_volume=("volume", "sum"),
            total_trades=("trade_count", "sum")
        ).reset_index()

        pm_daily["avg_trade_size"] = (
            pm_daily["total_volume"] / pm_daily["total_trades"]
        )
        pm_daily["median_hist"] = (
            pm_daily["avg_trade_size"].rolling(lookback).median()
        )

        pm_daily["R_PM_hist"] = np.log(
            (pm_daily["avg_trade_size"] + eps) /
            (pm_daily["median_hist"] + eps)
        )

        pm_daily["symbol"] = symbol
        pm_daily = MyStockFunctions._add_4am_timestamp(pm_daily)
        return pm_daily[["symbol", "timestamp", "R_PM_hist"]]

    @staticmethod
    def compute_premarket_vwap_return(df):
        df = MyStockFunctions._add_date(df)
        symbol = df["symbol"].iloc[0]

        pre = MyStockFunctions._premarket(df)

        pm_daily = pre.groupby("date").apply(
            lambda g: (
                np.sum(g["vwap"] * g["volume"]) / np.sum(g["volume"])
                if g["volume"].sum() > 0 else np.nan
            ),
            include_groups=False
        ).reset_index(name="vwap_pm")

        reg = MyStockFunctions._regular_hours(df)
        dc = reg.groupby("date")["close"].last().reset_index()
        dc["prev_close"] = dc["close"].shift(1)

        merged = pm_daily.merge(dc[["date", "prev_close"]], on="date", how="left")

        merged["PM_VWAP_Return"] = merged["vwap_pm"] / merged["prev_close"] - 1
        merged["symbol"] = symbol

        merged = MyStockFunctions._add_4am_timestamp(merged)
        return merged[["symbol", "timestamp", "PM_VWAP_Return"]]

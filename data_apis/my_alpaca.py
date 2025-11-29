import os
from zoneinfo import ZoneInfo
from config import ConnectionParameters
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.enums import DataFeed, Adjustment
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
import pandas as pd
from data_apis import MyMarketIndexList
from data_apis import MyHelper

class MyAlpacaStock:
    def __init__(self, feed: str = DataFeed.SIP, symbol: str = ""):
        self.__cp = ConnectionParameters()
        self.__shdc = StockHistoricalDataClient(self.__cp.ALPACA_API_KEY, self.__cp.ALPACA_SECRET_KEY)
        self.__feed = feed
        self.__symbol = symbol

    def build_event_df(self) -> pd.DataFrame:
        """
        Build event DataFrame where each row is a full trend (continuous movement in one direction)
        until an opposite movement surpasses the dynamic threshold given by MyHelper.min_max_target().
        Uses factor ratios between adjusted and raw close to neutralize splits/dividends.
        """

        events = []
        df30 = self.__df.reset_index(drop=True)
        factor_ratio = 1.0
        if df30.empty:
            return pd.DataFrame(events)

        symbol = df30.loc[0, "symbol"]
        reference_price = pd.to_numeric(df30.loc[0, "open"], errors="coerce")
        low = reference_price
        high = reference_price
        price_point_adj = reference_price
        start_time = df30.loc[0, "timestamp"]
        reference_factor = pd.to_numeric(df30.loc[0, "close_adj"], errors="coerce") / pd.to_numeric(df30.loc[0, "close"], errors="coerce")

        for _, row30 in df30.iterrows():
            candle_factor = pd.to_numeric(row30["close_adj"], errors="coerce") / pd.to_numeric(row30["close"], errors="coerce")
            factor_ratio = round(candle_factor / reference_factor, 3)
            adj_high_30 = row30["high"] * factor_ratio
            adj_low_30 = row30["low"] * factor_ratio

            min_target, max_target = MyHelper.min_max_target(reference_price)
            if not ((adj_high_30 >= max_target) or (adj_low_30 <= min_target)) :
                high = max(high, row30["high"])
                low = min(low, row30["low"])
                continue

            start_date = row30["timestamp"]
            end_date = start_date + pd.Timedelta(value=self.__quantity, unit=self.__unit.value) # type: ignore
            request_bars = StockBarsRequest(
                symbol_or_symbols=row30["symbol"],
                timeframe=TimeFrame(amount=1, unit=TimeFrameUnit.Minute), # type: ignore
                start=start_date,
                end=end_date,
                adjustment=Adjustment.RAW,
                feed=self.__feed, # type: ignore
            )
            df_min = self.__shdc.get_stock_bars(request_bars).df # type: ignore
            self.__change_df(df_min)

            for _, m in df_min.iterrows():
                price_point_adj = m["close"] * factor_ratio
                min_target, max_target = MyHelper.min_max_target(reference_price)
                hit_up = price_point_adj >= max_target
                hit_down = price_point_adj <= min_target
                low = min(low, m["low"])
                high = max(high, m["high"])
                if not (hit_up or hit_down):
                    continue
                
                if factor_ratio > 1:
                    high = round(high / factor_ratio, 3)
                elif factor_ratio < 1:
                    low = round(low * factor_ratio, 3)
                events.append({
                    "symbol": symbol,
                    "start_time": start_time,
                    "end_time": m["timestamp"],
                    "open": reference_price / factor_ratio,
                    "high": high,
                    "low": low,
                    "close": m["close"],
                    "close_adj": price_point_adj,
                    "factor": factor_ratio,
                    "pct_change": (price_point_adj - reference_price) / reference_price,
                })
                reference_price = m["close"]
                start_time = m["timestamp"] + pd.Timedelta(minutes=1)
                low = reference_price
                high = reference_price
                reference_factor = candle_factor
                factor_ratio = 1.0

        events.append({
            "symbol": symbol,
            "start_time": start_time,
            "end_time": df30.iloc[-1]["timestamp"],
            "open": reference_price,
            "high": high,
            "low": low,
            "close": df30.iloc[-1]["close"],
            "close_adj": df30.iloc[-1]["close"] * factor_ratio,
            "factor": factor_ratio,
            "pct_change": (df30.iloc[-1]["close"] * factor_ratio - reference_price) / reference_price,
        })

        return pd.DataFrame(events)

    def query_from_file_data(self, quantity: int, unit: str = TimeFrameUnit.Minute,
                             adjustment: Adjustment = Adjustment.ALL):
        #Read start and end date from file
        if adjustment == Adjustment.ALL:
            adj_path = "adj_all"
        elif adjustment == Adjustment.RAW:
            adj_path = "adj_raw"
        full_path = f"{self.__cp.PATH_TO_SAVE}{quantity}{unit.value}_history_{adj_path}/{self.__symbol}.csv" #type: ignore
        self.__df = pd.read_csv(full_path, parse_dates=['timestamp'])
        self.__df = self.__df.sort_values(by=["symbol", "timestamp"]).reset_index(drop=True)

    def query_historical_data(self, start: str, end: str, quantity: int, unit: str = TimeFrameUnit.Minute,
                              adjustment: Adjustment = Adjustment.ALL):
        # Convert string dates to pandas timestamps in New York timezone with start_date time equal to 00:00:01 AM and end_date time equal to 11:59:59 PM
        start_date = pd.Timestamp(start).tz_localize(ZoneInfo("America/New_York")).floor('D') + pd.Timedelta(seconds=1)
        end_date = pd.Timestamp(end).tz_localize(ZoneInfo("America/New_York")).ceil('D') - pd.Timedelta(seconds=1)
        # Create the request for stock bars
        request_bars =StockBarsRequest(
            symbol_or_symbols=self.__symbol,
            timeframe=TimeFrame(amount=quantity, unit=unit), # type: ignore
            start=start_date,
            end=end_date,
            adjustment=adjustment, # type: ignore
            feed=self.__feed, # type: ignore
        )
        # Execute the query
        try:
            self.__df = self.__shdc.get_stock_bars(request_bars).df # type: ignore
            if not self.__df.empty:
                self.__change_df(self.__df)
        except Exception as e:
            #If error contains text "invalid symbol", print message and set self.__df to None
            #otherwise, raise the exception
            if "invalid symbol" in str(e).lower():
                print(f"Error: Invalid symbol '{self.__symbol}'. No data retrieved.")
                self.__df = pd.DataFrame()
                return
            else:
                raise e
    
    def merge_raw_and_all_from_files(self, quantity: int, unit: str = TimeFrameUnit.Minute):
        #Load adjusted all data
        self.__quantity = quantity
        self.__unit = unit
        self.query_from_file_data(quantity=quantity, unit=unit, adjustment=Adjustment.ALL)
        df_all = self.get_df()
        #Load adjusted raw data
        self.query_from_file_data(quantity=quantity, unit=unit, adjustment=Adjustment.RAW)
        df_raw = self.get_df()
        #Merge df_all and df_raw using as keys symbol and timestamp bringing all columns from df_raw and only close column from df_all
        self.__df = pd.merge(df_all[['symbol', 'timestamp', 'close']], df_raw, on=['symbol', 'timestamp'], how='inner', suffixes=('_adj', ''))

    def trim_df_by_date(self, start: str, end: str):
        # Convert string dates to pandas timestamps in New York timezone
        start_date = pd.Timestamp(start).tz_localize(ZoneInfo("America/New_York"))
        end_date = pd.Timestamp(end).tz_localize(ZoneInfo("America/New_York"))
        # Trim the dataframe
        self.__df = self.__df[(self.__df['timestamp'] >= start_date) & (self.__df['timestamp'] <= end_date)].reset_index(drop=True)

    def get_df(self) -> pd.DataFrame:
        return self.__df
    
    def __change_df(self, df: pd.DataFrame):
        df.insert(0, 'symbol', df.index.get_level_values('symbol'))
        df.insert(1, 'timestamp', df.index.get_level_values('timestamp').tz_convert('America/New_York')) # type: ignore

class MyAlpacaJob:
    def __init__(self, start: str, end: str, quantity: int, unit: str, my_index_list=None,
                 stock_list: list[str] = [], adjustment: Adjustment = Adjustment.ALL):
        self.__cp = ConnectionParameters()
        if my_index_list is not None and not isinstance(my_index_list, MyMarketIndexList):
            raise TypeError("my_index_list parameter must be of the class MyMarketIndexList")
        self.__my_index_list = my_index_list
        self.__start = start
        self.__end = end
        self.__quantity = quantity
        self.__unit = unit
        self.__stock_list = stock_list
        self.__adjustment = adjustment

    def save_to_folder(self, limit: int):
        #Bring the index list data frame and turn it into a list of symbols without duplicates
        full_path = f"{self.__cp.PATH_TO_SAVE}{self.__quantity}{self.__unit.value}_history/" #type: ignore
        symbols = self.__get_list_of_symbols()[:limit]   
        for symbol in symbols:
            #check if symbol is already saved in folder
            try:
                already_saved = pd.read_csv(f"{full_path}{symbol}.csv")
                if not already_saved.empty:
                    print(f"Data for symbol {symbol} already exists in {full_path}{symbol}.csv. Skipping download.")
                    continue
            except FileNotFoundError:
                pass
            my_alpaca_stock = MyAlpacaStock(symbol=symbol)
            my_alpaca_stock.query_historical_data(start=self.__start, end=self.__end, quantity=self.__quantity,
                                                  unit=self.__unit, adjustment=self.__adjustment)
            df = my_alpaca_stock.get_df()
            if not df.empty:
                #Save Dataframe as CSV file in folder "alpaca_data" with filename as symbol.csv {self.__cp.PATH_TO_SAVE}
                df.to_csv(f"{full_path}{symbol}.csv", index=False)
                print(f"Saved data for symbol {symbol} to {full_path}{symbol}.csv")
            else:
                print(f"No data to save for symbol {symbol}.")

    def __get_list_of_symbols(self) -> list[str]:
        if self.__stock_list:
            return self.__stock_list
        elif not self.__my_index_list is None:
            historical_constituents = self.__my_index_list.get_selected_constituents()
            return historical_constituents['symbol'].unique().tolist()
        elif not self.__stock_list:
            full_path = f"{self.__cp.PATH_TO_SAVE}{self.__quantity}{self.__unit.value}_history_adj_{self.__adjustment.value}/" # type: ignore
            self.__stock_list = os.listdir(full_path)
            #Remove suffix .csv from each element of the list
            self.__stock_list = [symbol.replace('.csv', '') for symbol in self.__stock_list]            
            return self.__stock_list
        else:
            raise ValueError("No symbols found in the index constituents and no stock list provided.")
        
    def save_by_day_consolidated(self, size_indices_filename: str = "Size Indices.xlsx", sector_indices_filename: str = "Sector Indices.xlsx"):
        my_size_indices = MyMarketIndexList()
        my_sector_indices = MyMarketIndexList()
        my_size_indices.set_selected_indices_from_file(size_indices_filename)
        my_sector_indices.set_selected_indices_from_file(sector_indices_filename)
        
        #TEMPORAL FOR PROJECT AT UNIVERSITY INIT
        # sector_cons = my_sector_indices.get_selected_constituents("sector_constituents.csv")
        sector_cons = my_sector_indices.get_selected_constituents(filename="sector_constituents.csv", only_active=True, active_on_date="2016-01-04")
        sector_cons = (
            sector_cons
            .groupby('index_symbol')
            .sample(n=5, replace=False, random_state=75)
            .reset_index(drop=True)
        )
        #TEMPORAL FOR PROJECT AT UNIVERSITY END

        size_cons = my_size_indices.get_selected_constituents("size_constituents.csv")
        for symbol in self.__get_list_of_symbols():
            #TEMPORAL FOR PROJECT AT UNIVERSITY INIT
            if not symbol in sector_cons['symbol'].values:
                continue
            # if self.__get_list_of_symbols().index(symbol) >= 5:
            #     break
            #TEMPORAL FOR PROJECT AT UNIVERSITY END

            my_alpaca_stock = MyAlpacaStock(symbol=symbol)
            my_alpaca_stock.merge_raw_and_all_from_files(quantity=self.__quantity, unit=self.__unit)
            my_alpaca_stock.trim_df_by_date(start='2016-01-01', end='2025-12-31')
            df30 = my_alpaca_stock.get_df()
            #Create new dataframe by one day intervals from df30 with first open, last close, last close_adj,
            #higher high, lower low, sum of volume, sum of trade_count of the day
            df_day = pd.DataFrame()
            if df30.empty:
                continue
            df30['timestamp'] = pd.to_datetime(df30['timestamp'], utc=True).dt.tz_convert("America/New_York")
            df30['date'] = df30['timestamp'].dt.date
            #Turn timestamp back again to datetime with timezone America/New_York at 4:00 am and drop date column
            df30['timestamp'] = pd.to_datetime(df30['date']).dt.tz_localize(ZoneInfo("America/New_York")) + pd.Timedelta(hours=4)
            df30 = df30.drop(columns=['date'])
            df_day = df30.groupby(['symbol','timestamp']).agg(
                open=('open', 'first'),
                high=('high', 'max'),
                low=('low', 'min'),
                close=('close', 'last'),
                close_adj=('close_adj', 'last'),
                volume=('volume', 'sum'),
                trade_count=('trade_count', 'sum')
            ).reset_index()
            #add column to indicate for each date and symbol which size index it belongs to (size_index)
            #considering the start_date and end_date of each constituent
            df_day['size_index'] = None            
            df_day['sector_index'] = None
            for _, row in size_cons.iterrows():
                mask = (df_day['symbol'] == row['symbol']) & (df_day['timestamp'] >= row['start_date']) & (df_day['timestamp'] <= row['end_date'])
                df_day.loc[mask, 'size_index'] = row['index_symbol']
            for _, row in sector_cons.iterrows():
                mask = (df_day['symbol'] == row['symbol']) & (df_day['timestamp'] >= row['start_date']) & (df_day['timestamp'] <= row['end_date'])
                df_day.loc[mask, 'sector_index'] = row['index_symbol']
            #Save df_day to folder "1Day_history_merged" with filename as symbol.csv
            full_path = f"{self.__cp.PATH_TO_SAVE}1Day_history_merged/"
            df_day.to_csv(f"{full_path}{symbol}.csv", index=False)
            print(f"Saved consolidated daily data for symbol {symbol} to {full_path}{symbol}.csv")

    def add_variables_to_history_merged(self, raw_functions: dict = {}, adj_functions: dict = {}):
        merged_path = f"{self.__cp.PATH_TO_SAVE}1Day_history_merged/"
        raw_path = f"{self.__cp.PATH_TO_SAVE}30Min_history_adj_raw/"
        adj_path = f"{self.__cp.PATH_TO_SAVE}30Min_history_adj_all/"
        symbols_to_process = os.listdir(merged_path)
        #Remove suffix .csv from each element of the list
        symbols_to_process = [symbol.replace('.csv', '') for symbol in symbols_to_process]
        for symbol in symbols_to_process:
            raw_df = pd.read_csv(f"{raw_path}{symbol}.csv", parse_dates=['timestamp'])
            adj_df = pd.read_csv(f"{adj_path}{symbol}.csv", parse_dates=['timestamp'])
            merged_df = pd.read_csv(f"{merged_path}{symbol}.csv", parse_dates=['timestamp'])
            merged_df['timestamp'] = pd.to_datetime(merged_df['timestamp'], utc=True).dt.tz_convert("America/New_York")
            for _, func in raw_functions.items():
                to_append_df = func(raw_df)
                # Identify overlapping non-key columns
                overlap = [c for c in to_append_df.columns 
                        if c in merged_df.columns and c not in ['symbol', 'timestamp']]
                # Drop those overlapping columns from the left df
                merged_df = merged_df.drop(columns=overlap)
                # Now merge — right dataframe values will be used for overlapping columns
                merged_df = merged_df.merge(
                    to_append_df,
                    on=['symbol', 'timestamp'],
                    how='left',
                    suffixes=('', '')
                )
            for _, func in adj_functions.items():
                to_append_df = func(adj_df)
                # Identify overlapping non-key columns
                overlap = [c for c in to_append_df.columns 
                        if c in merged_df.columns and c not in ['symbol', 'timestamp']]
                # Drop those overlapping columns from the left df
                merged_df = merged_df.drop(columns=overlap)
                # Now merge — right dataframe values will be used for overlapping columns
                merged_df = merged_df.merge(
                    to_append_df,
                    on=['symbol', 'timestamp'],
                    how='left',
                    suffixes=('', '')
                )
            merged_df.to_csv(f"{merged_path}{symbol}.csv", index=False)
            print(f"Added variables to merged data for symbol {symbol} and saved to {merged_path}{symbol}.csv")
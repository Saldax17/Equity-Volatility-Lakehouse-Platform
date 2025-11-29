from config import ConnectionParameters
import pandas as pd
import requests

class MyMarketIndexList:
    def __init__(self):
        self.__cp = ConnectionParameters()
        url_index_list = f"https://eodhd.com/api/mp/unicornbay/spglobal/list?api_token={self.__cp.EODHD_API_TOKEN}"
        response = requests.get(url_index_list)
        if response.status_code == 200:
            index_list_json = response.json()
            self.__index_list_df = pd.DataFrame(index_list_json)
            self.__index_list_df = self.rename_columns(self.__index_list_df)
        elif response.status_code == 403:
            self.__index_list_df = pd.read_csv(f"{self.__cp.PATH_TO_SAVE}indices_full_list.csv")
            self.__index_list_df = self.rename_columns(self.__index_list_df)
        else:
            raise Exception(f"Failed to fetch indices list: {response.status_code} - {response.text}")
        
    def get_full_index_list(self) -> pd.DataFrame:
        return self.__index_list_df
    
    def set_selected_indices(self, indices: list):
        self.__selected_indices_df = self.__index_list_df[self.__index_list_df['symbol'].isin(indices)].reset_index(drop=True)

    def set_selected_indices_from_file(self, filename: str):
        full_path = f"{self.__cp.PATH_TO_QUERY}{filename}"
        indices_df = pd.read_excel(full_path)
        indices = indices_df['symbol'].tolist()
        self.set_selected_indices(indices)

    def get_selected_indices(self) -> pd.DataFrame:
        return self.__selected_indices_df

    def get_selected_constituents(self, filename = None, only_active = False, active_on_date = None) -> pd.DataFrame:
        try:
            const_list = []
            for index_symbol in self.__selected_indices_df['full_symbol']:
                index = MyMarketIndex(index_code=index_symbol)
                curr_const_df = index.get_index_historical_constituents()
                #Insert as third column the index_symbol column to identify the index of the constituent
                curr_const_df.insert(2, 'index_symbol', index_symbol)
                const_list.append(curr_const_df)
            all_constituents_df = pd.concat(const_list, ignore_index=True, verify_integrity=True)
        except Exception as e:
            if filename is None:
                raise e
            else:
                all_constituents_df = pd.read_csv(f"{self.__cp.PATH_TO_SAVE}{filename}")
                all_constituents_df['start_date'] = pd.to_datetime(all_constituents_df['start_date'], utc=True).dt.tz_convert('America/New_York')
                all_constituents_df['end_date'] = pd.to_datetime(all_constituents_df['end_date'], utc=True).dt.tz_convert('America/New_York')
        #Sort by symbol
        all_constituents_df = all_constituents_df.sort_values(by=['symbol','end_date']).reset_index(drop=True)
        if only_active:
            all_constituents_df = all_constituents_df[all_constituents_df['is_active_in_index'] == True].reset_index(drop=True)
        if active_on_date is not None:
            all_constituents_df = all_constituents_df[
                (all_constituents_df['start_date'] <= pd.to_datetime(active_on_date).tz_localize('America/New_York')) &
                (all_constituents_df['end_date'] >= pd.to_datetime(active_on_date).tz_localize('America/New_York'))
            ].reset_index(drop=True)
        return all_constituents_df

    def rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df.rename(
            columns={"ID": "full_symbol",
                     "Code": "symbol",
                     "Name": "name",
                     "Constituents": "constituents_count",
                     "Value": "value",
                     "MarketCap": "market_cap",
                     "Divisor": "divisor",
                     "DailyReturn": "daily_return",
                     "Dividend": "dividend",
                     "AdjustedMarketCap": "adjusted_market_cap",
                     "AdjustedDivisor": "adjusted_divisor",
                     "AdjustedConstituents": "adjusted_constituents_count",
                     "CurrencyCode": "currency_code",
                     "CurrencyName": "currency_name",
                     "CurrencySymbol": "currency_symbol",
                     "LastUpdate": "last_update"},
            inplace=True)
        return df

class MyMarketIndex:
    def __init__(self, index_code: str):
        self.__cp = ConnectionParameters()
        self.index_code = index_code
        url_index = f"https://eodhd.com/api/mp/unicornbay/spglobal/comp/{index_code}?fmt=json&api_token={self.__cp.EODHD_API_TOKEN}"
        response = requests.get(url_index)

        if response.status_code == 200:
            self.__index_details_json = response.json()
        else:
            raise Exception(f"Failed to fetch index {index_code}: {response.status_code} - {response.text}")

    def get_index_info(self) -> dict:
        return self.__index_details_json

    def get_index_historical_constituents(self) -> pd.DataFrame:
        constituents = self.__index_details_json.get("HistoricalTickerComponents", {})
        #Turn into DataFrame
        constituents_df = pd.DataFrame(constituents.values())
        #Rename columns
        constituents_df = self.rename_columns(constituents_df)
        #For null and empty start_date and end_date columns
        #Fill with 1950-01-01 for start_date and 2199-12-31 for end_date
        constituents_df['start_date'] = constituents_df['start_date'].fillna('1950-01-01')
        constituents_df['end_date'] = constituents_df['end_date'].fillna('2199-12-31')
        constituents_df['start_date'] = constituents_df['start_date'].replace('', '1950-01-01')
        constituents_df['end_date'] = constituents_df['end_date'].replace('', '2199-12-31')
        #Convert to datetime and set 0 hours, 0 minutes, 1 second new york time to both start_date and end_date columns
        constituents_df['start_date'] = pd.to_datetime(constituents_df['start_date']).dt.tz_localize('America/New_York').dt.floor('s') + pd.Timedelta(seconds=1)
        constituents_df['end_date'] = pd.to_datetime(constituents_df['end_date']).dt.tz_localize('America/New_York').dt.floor('s')
        return constituents_df

    def get_index_current_constituents(self) -> pd.DataFrame:
        constituents = self.__index_details_json.get("Components", {})
        #Turn into DataFrame
        constituents_df = pd.DataFrame(constituents.values())
        #Rename columns
        constituents_df = self.rename_columns(constituents_df)
        return constituents_df

    def rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df.rename(
            columns={"Code": "symbol",
                     "Name": "name",
                     "StartDate": "start_date",
                     "EndDate": "end_date",
                     "Weight": "weight",
                     "Exchange": "exchange",
                     "Industry": "industry",
                     "Sector": "sector",
                     "IsActiveNow": "is_active_in_index",
                     "IsDelisted": "is_delisted"},
            inplace=True)
        return df
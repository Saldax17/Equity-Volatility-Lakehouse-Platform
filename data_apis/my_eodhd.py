import requests
import pandas as pd
from zoneinfo import ZoneInfo
from typing import Optional, List

from config import ConnectionParameters


# ============================================================
#                     MyMarketIndexList
# ============================================================

class MyMarketIndexList:
    """
    A wrapper for retrieving SP Global index lists and constituents
    from EODHD. This class is refactored for clarity and robustness
    while preserving the original API-level behavior.
    """

    BASE_LIST_URL = "https://eodhd.com/api/mp/unicornbay/spglobal/list"
    SAVE_FALLBACK_FILE = "indices_full_list.csv"

    def __init__(self) -> None:
        self._cfg = ConnectionParameters()
        self._index_list_df = self._fetch_index_list()
        self._selected_indices_df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------
    # Fetching & Loading
    # ------------------------------------------------------------

    def _fetch_index_list(self) -> pd.DataFrame:
        """
        Attempt to fetch index list from EODHD. Fallback to local CSV on 403.
        """
        url = f"{self.BASE_LIST_URL}?api_token={self._cfg.EODHD_API_TOKEN}"
        response = requests.get(url)

        if response.status_code == 200:
            df = pd.DataFrame(response.json())
            return self._rename_index_list_cols(df)

        if response.status_code == 403:
            path = f"{self._cfg.PATH_TO_SAVE}{self.SAVE_FALLBACK_FILE}"
            df = pd.read_csv(path)
            return self._rename_index_list_cols(df)

        raise Exception(
            f"Failed to fetch indices list ({response.status_code}): {response.text}"
        )

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------

    def get_full_index_list(self) -> pd.DataFrame:
        return self._index_list_df

    def set_selected_indices(self, indices: List[str]) -> None:
        """
        Select a subset of indices by their 'symbol' field.
        """
        self._selected_indices_df = (
            self._index_list_df[self._index_list_df["symbol"].isin(indices)]
            .reset_index(drop=True)
        )

    def set_selected_indices_from_file(self, filename: str) -> None:
        """
        Load Excel file with a column 'symbol' containing index tickers.
        """
        full_path = f"{self._cfg.PATH_TO_QUERY}{filename}"
        symbols = pd.read_excel(full_path)["symbol"].tolist()
        self.set_selected_indices(symbols)

    def get_selected_indices(self) -> pd.DataFrame:
        if self._selected_indices_df is None:
            raise ValueError("No indices selected. Call set_selected_indices() first.")
        return self._selected_indices_df

    # ------------------------------------------------------------
    # Constituents
    # ------------------------------------------------------------

    def get_selected_constituents(
        self,
        filename: Optional[str] = None,
        only_active: bool = False,
        active_on_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Retrieve all historical constituents for selected indices.
        Fallback: read a local CSV if remote calls fail.
        """
        if self._selected_indices_df is None:
            raise ValueError("No selected indices. Use set_selected_indices().")

        try:
            all_constituents = []
            for index_symbol in self._selected_indices_df["full_symbol"]:
                index_obj = MyMarketIndex(index_code=index_symbol)
                df_c = index_obj.get_index_historical_constituents()
                df_c.insert(2, "index_symbol", index_symbol)
                all_constituents.append(df_c)

            df = pd.concat(all_constituents, ignore_index=True)

        except Exception:
            if filename is None:
                raise
            # fallback to CSV
            df = pd.read_csv(f"{self._cfg.PATH_TO_SAVE}{filename}")
            df["start_date"] = pd.to_datetime(df["start_date"], utc=True).dt.tz_convert(
                "America/New_York"
            )
            df["end_date"] = pd.to_datetime(df["end_date"], utc=True).dt.tz_convert(
                "America/New_York"
            )

        # sorting & filters
        df = df.sort_values(["symbol", "end_date"]).reset_index(drop=True)

        if only_active:
            df = df[df["is_active_in_index"] == True].reset_index(drop=True)

        if active_on_date is not None:
            ts = pd.Timestamp(active_on_date).tz_localize("America/New_York")
            df = df[(df["start_date"] <= ts) & (df["end_date"] >= ts)].reset_index(
                drop=True
            )

        return df

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------

    def _rename_index_list_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.rename(
            columns={
                "ID": "full_symbol",
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
                "LastUpdate": "last_update",
            },
            inplace=True,
        )
        return df


# ============================================================
#                     MyMarketIndex
# ============================================================

class MyMarketIndex:
    """
    Represents a single SP Global index.  
    Handles retrieval of metadata, current components, and historical constituents.
    """

    BASE_INDEX_URL = "https://eodhd.com/api/mp/unicornbay/spglobal/comp"

    def __init__(self, index_code: str) -> None:
        self._cfg = ConnectionParameters()
        self._index_code = index_code
        self._details_json = self._fetch_index_details()

    # ------------------------------------------------------------
    # Fetch
    # ------------------------------------------------------------

    def _fetch_index_details(self) -> dict:
        url = (
            f"{self.BASE_INDEX_URL}/{self._index_code}"
            f"?fmt=json&api_token={self._cfg.EODHD_API_TOKEN}"
        )
        response = requests.get(url)

        if response.status_code != 200:
            raise Exception(
                f"Failed to fetch index {self._index_code} "
                f"({response.status_code}): {response.text}"
            )

        return response.json()

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------

    def get_index_info(self) -> dict:
        return self._details_json

    def get_index_historical_constituents(self) -> pd.DataFrame:
        hist = self._details_json.get("HistoricalTickerComponents", {})
        df = pd.DataFrame(hist.values())
        df = self._rename_hist_cols(df)
        return self._clean_constituent_dates(df)

    def get_index_current_constituents(self) -> pd.DataFrame:
        curr = self._details_json.get("Components", {})
        df = pd.DataFrame(curr.values())
        return self._rename_hist_cols(df)

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------

    def _rename_hist_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.rename(
            columns={
                "Code": "symbol",
                "Name": "name",
                "StartDate": "start_date",
                "EndDate": "end_date",
                "Weight": "weight",
                "Exchange": "exchange",
                "Industry": "industry",
                "Sector": "sector",
                "IsActiveNow": "is_active_in_index",
                "IsDelisted": "is_delisted",
            },
            inplace=True,
        )
        return df

    def _clean_constituent_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize start/end dates:
        - fill missing with wide default ranges
        - convert to NY timezone
        """
        df = df.copy()

        df["start_date"] = (
            df["start_date"]
            .replace("", "1950-01-01")
            .fillna("1950-01-01")
        )
        df["end_date"] = (
            df["end_date"]
            .replace("", "2199-12-31")
            .fillna("2199-12-31")
        )

        df["start_date"] = (
            pd.to_datetime(df["start_date"])
            .dt.tz_localize("America/New_York")
            .dt.floor("s")
            + pd.Timedelta(seconds=1)
        )

        df["end_date"] = (
            pd.to_datetime(df["end_date"])
            .dt.tz_localize("America/New_York")
            .dt.floor("s")
        )

        return df

import pandas as pd
from pathlib import Path
from typing import Optional

from config import ConnectionParameters


class MyIsharesETF:
    """
    Loads iShares ETF constituent lists from Excel files stored locally.

    Expected filename format:
        "{file_date}_{symbol}.xlsx"

    Example:
        2025-10-24_IWB.xlsx
    """

    def __init__(self, symbol: str, file_date: str) -> None:
        self._cfg = ConnectionParameters()
        self._symbol = symbol
        self._file_date = file_date

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------
    def load_from_xls(self, sheet_name: Optional[str] = None) -> pd.DataFrame:
        """
        Load ETF data from an Excel file, optionally from a specific sheet.

        Parameters
        ----------
        sheet_name : str | None
            If provided, load a specific sheet. If None, loads the first sheet.

        Returns
        -------
        pd.DataFrame
            Cleaned constituent list with deduplicated tickers.

        Raises
        ------
        FileNotFoundError
            If the XLSX file is not found.
        ValueError
            If required columns are missing.
        """

        file_path = self._build_file_path()

        if not file_path.exists():
            raise FileNotFoundError(
                f"ETF XLSX file not found: {file_path}. "
                f"Expected format: '{{file_date}}_{{symbol}}.xlsx'"
            )

        df = self._read_excel_file(file_path, sheet_name)
        df = self._clean_dataframe(df)
        return df

    # ------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------
    def _build_file_path(self) -> Path:
        """Builds the absolute path to the Excel file."""
        filename = f"{self._file_date}_{self._symbol}.xlsx"
        return Path(self._cfg.PATH_TO_QUERY) / "iShares ETFs" / filename

    def _read_excel_file(self, path: Path, sheet_name: Optional[str]) -> pd.DataFrame:
        """Reads an Excel file safely."""
        try:
            return pd.read_excel(path, sheet_name=sheet_name)
        except Exception as e:
            raise ValueError(f"Failed to read Excel file {path}: {e}")

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Removes duplicate tickers and validates structure."""
        if "Ticker" not in df.columns:
            raise ValueError(
                "'Ticker' column not found in ETF file. "
                "Please verify the file structure."
            )

        df = df.drop_duplicates(subset=["Ticker"]).reset_index(drop=True)
        return df

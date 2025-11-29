import pandas as pd
from config import ConnectionParameters

class MyIsharesETF:
    #Link for future reference: https://www.ishares.com/us/products/239710/ishares-russell-2000-etf
    def __init__(self, symbol: str, file_date: str):
        self.__cp = ConnectionParameters()
        self.__symbol = symbol
        self.__file_date = file_date

    def load_from_xls(self) -> pd.DataFrame:
        file_name = f"{self.__file_date}_{self.__symbol}.xlsx"
        full_path = f"{self.__cp.PATH_TO_QUERY}iShares ETFs/"
        #Read excel file with more than 1 sheet into json or dictionary
        #xls = pd.ExcelFile(f"{full_path}{file_name}", engine="xlrd")
        data = pd.read_excel(f"{full_path}{file_name}")
        #Remove duplicate rows comparing Ticker column
        data = data.drop_duplicates(subset=['Ticker'])
        return data
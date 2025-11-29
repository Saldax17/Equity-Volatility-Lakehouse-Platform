import os
from typing import Dict

class ConnectionParameters:
    def __init__(self):
        # Alpaca API
        self.ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
        if self.ALPACA_API_KEY is None or self.ALPACA_API_KEY == "":
            self.ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY")
        self.ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
        if self.ALPACA_SECRET_KEY is None or self.ALPACA_SECRET_KEY == "":
            self.ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY")

        # EODHD API
        self.EODHD_API_TOKEN = os.getenv("EODHD_API_TOKEN")
        if self.EODHD_API_TOKEN is None or self.EODHD_API_TOKEN == "":
            self.EODHD_API_TOKEN = os.environ.get("EODHD_API_TOKEN")

        #self.PATH_TO_SAVE = '../my_data/30m_history/'
        self.PATH_TO_SAVE = 'C:/Users/David Trade/my_data/automatic/'

        #File to retreive data
        self.PATH_TO_QUERY = 'C:/Users/David Trade/my_data/manual/'
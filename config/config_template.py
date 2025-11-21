class ConnectionParameters:
    """
    Template configuration file.

    How to use:
    1. Copy this file and rename the copy to `config.py`
    2. Replace the placeholder values with your real credentials
    3. Do NOT upload config.py — it is ignored by .gitignore
    """

    # Alpaca API keys
    ALPACA_API_KEY = "YOUR_ALPACA_API_KEY_HERE"
    ALPACA_SECRET_KEY = "YOUR_ALPACA_SECRET_KEY_HERE"

    # EODHD API token
    EODHD_API_TOKEN = "YOUR_EODHD_TOKEN_HERE"

    # Local Lakehouse paths (Bronze Layer)
    PATH_TO_SAVE = "./data/bronze/"
    PATH_TO_QUERY = "./data/bronze/"

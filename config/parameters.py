"""
parameters.py
--------------
Parámetros globales del proyecto EVLP.

Estos parámetros sirven para:
- Controlar periodos de ingesta
- Settings por defecto para ETL
- Configuración de features
- Configuración común para notebooks y scripts
"""

from alpaca.data.enums import DataFeed, Adjustment
from alpaca.data.timeframe import TimeFrameUnit

# =============================
#   GENERAL PROJECT SETTINGS
# =============================
PROJECT_NAME = "Equity Volatility Lakehouse Platform"
DEFAULT_TIMEZONE = "America/New_York"

# =============================
#   ALPACA INGEST SETTINGS
# =============================
DEFAULT_FEED = DataFeed.SIP
DEFAULT_ADJUSTMENT = Adjustment.ALL

# Velas por defecto (30 minutos)
DEFAULT_BAR_AMOUNT = 30
DEFAULT_BAR_UNIT = TimeFrameUnit.Minute

# Rango temporal por defecto para ingesta masiva
DEFAULT_START_DATE = "2007-01-01"
DEFAULT_END_DATE = "2025-12-31"

# =============================
#   DATA PATHS (USING config.py)
# =============================
# Nota:
# - EVLP_DATA_DIR viene desde config.py
# - Estos paths se construyen en scripts como ingestion/...
# - NO colocamos rutas absolutas aquí

# Estructura esperada dentro de EVLP_DATA_DIR:
BRONZE_DIR_NAME = "bronze"
SILVER_DIR_NAME = "silver"
GOLD_DIR_NAME = "gold"

# =============================
#   FEATURE ENGINEERING SETTINGS
# =============================
LOOKBACK_WINDOWS = {
    "1d": 1,
    "5d": 5,
    "7d": 7,
    "20d": 20,
    "28d": 28,
    "112d": 112
}

# =============================
#   ML SETTINGS
# =============================
TRAIN_SIZE = 0.8
RANDOM_STATE = 42

# Modelos por defecto a entrenar
MODELS_TO_RUN = [
    "logreg",
    "random_forest",
    "gradient_boosting"
]

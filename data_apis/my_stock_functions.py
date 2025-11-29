import pandas as pd
import numpy as np
from zoneinfo import ZoneInfo
import datetime
from pandas.tseries.holiday import USFederalHolidayCalendar
import pandas_market_calendars as mcal

class MyStockFunctions:
    def __init__(self):
        pass
    #Static function to calculate 1 day return
    @staticmethod
    def compute_prev_day_return(df) -> pd.DataFrame:
        """
        Calcula el retorno del día anterior usando cierres diarios.
        
        Input:
        - df: DataFrame con columnas intradía: timestamp, close, symbol
        - Velas intradía de cualquier frecuencia (30 min, etc.)
        
        Output:
        - DataFrame con columnas: symbol, timestamp, prev_day_return
        """
        df = df.copy()

        # Timestamp a datetime y zona NY
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df['timestamp'] = df['timestamp'].dt.tz_convert("America/New_York")

        # Crear columna de fecha
        df['date'] = df['timestamp'].dt.date

        # Cierre diario
        daily_close = df.groupby(['symbol', 'date']).agg(
            close_daily=('close', 'last')
        ).reset_index()

        # Retorno del día anterior (shift 1 día)
        daily_close['prev_day_return'] = daily_close.groupby('symbol')['close_daily'].pct_change().shift(1)

        # Eliminar NaN inicial
        daily_close = daily_close.dropna(subset=['prev_day_return'])

        daily_close['timestamp'] = pd.to_datetime(daily_close['date']).dt.tz_localize(ZoneInfo("America/New_York")) + pd.Timedelta(hours=4)
        daily_close = daily_close.drop(columns=['date'])
        daily_close.insert(1, 'timestamp', daily_close.pop('timestamp'))

        return daily_close[['symbol', 'timestamp', 'prev_day_return']]

    @staticmethod
    def compute_std_return_last5d(df) -> pd.DataFrame:
        """
        Calcula la volatilidad (std) de los retornos diarios de los últimos 5 días
        usando todos los datos intradía (incluye pre y post market).
        
        Input:
            df: DataFrame intradía SOLO de un símbolo, columnas mínimas:
                - symbol
                - timestamp (datetime con tz o string)
                - close (precio ajustado)
        
        Output:
            DataFrame con columnas:
                - symbol
                - date
                - std_return_last5d
        """

        df = df.copy()

        # --- Convertir timestamp a datetime y aplicar zona horaria NYC ---
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)  # parsea tz-aware
        df['timestamp'] = df['timestamp'].dt.tz_convert('America/New_York')

        # --- Hacer timestamp naive (sin tz), necesario para agrupar fácilmente ---
        df['timestamp'] = df['timestamp'].dt.tz_localize(None)

        # --- Crear columna "date" para agrupación diaria ---
        df['date'] = df['timestamp'].dt.date

        # --- Ordenar por tiempo ---
        df = df.sort_values('timestamp')

        # --- Calcular log-return intradía ---
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))

        # --- Resetear retornos en cambio de día ---
        df.loc[df['date'] != df['date'].shift(1), 'log_ret'] = np.nan

        # --- Agregar retornos intradía a retorno diario ---
        daily = df.groupby('date').agg({
            'symbol': 'first',
            'log_ret': 'sum'
        }).reset_index()

        daily.rename(columns={'log_ret': 'daily_log_return'}, inplace=True)

        # --- Rolling std 5d ---
        daily['std_return_last5d'] = daily['daily_log_return'].rolling(window=5).std()

        # --- Output final ---
        output = daily[['symbol', 'date', 'std_return_last5d']]
        output['timestamp'] = pd.to_datetime(output['date']).dt.tz_localize(ZoneInfo("America/New_York")) + pd.Timedelta(hours=4)
        output = output.drop(columns=['date'])
        output.insert(1, 'timestamp', output.pop('timestamp'))
        
        return output
    
    @staticmethod
    def compute_range_rel_last1d(df) -> pd.DataFrame:
        """
        Calcula range_rel_last1d = (high_low / close) del día n-1,
        usando datos intradía (30 min).

        Salida:
        - symbol
        - date (día n)
        - range_rel_last1d
        """
        df = df.copy()

        # --- Asegurar timestamp con tz New York ---
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["timestamp"] = df["timestamp"].dt.tz_convert("America/New_York")

        # Fecha diaria
        df["date"] = df["timestamp"].dt.date

        # --- Agregación intradía por día ---
        daily = df.groupby("date").agg(
            day_high=("high", "max"),
            day_low=("low", "min"),
            day_close=("close", "last"),
            symbol=("symbol", "first")
        ).reset_index()

        # Range relativo
        daily["range_rel"] = (daily["day_high"] - daily["day_low"]) / daily["day_close"]

        # Shift para usar el valor del día n-1
        daily["range_rel_last1d"] = daily["range_rel"].shift(1)

        # Eliminar el primer día (NaN)
        #daily = daily.dropna(subset=["range_rel_last1d"])

        # --- Seleccionar columnas finales ---
        output = daily[["symbol", "date", "range_rel_last1d"]].copy()
        output['timestamp'] = pd.to_datetime(output['date']).dt.tz_localize(ZoneInfo("America/New_York")) + pd.Timedelta(hours=4)
        output = output.drop(columns=['date'])
        output.insert(1, 'timestamp', output.pop('timestamp'))

        return output
    
    @staticmethod
    def compute_rvol_20d(df) -> pd.DataFrame:
        """
        Calcula rvol_20d usando VOLUMEN EN DÓLARES
        (volume * vwap), para ser inmune a splits.

        Salida:
        - symbol
        - date
        - rvol_20d
        """
        df = df.copy()

        # --- Asegurar timestamp NY ---
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["timestamp"] = df["timestamp"].dt.tz_convert("America/New_York")

        # Fecha diaria
        df["date"] = df["timestamp"].dt.date

        # --- Crear volumen en dólares por vela ---
        df["dollar_volume"] = df["volume"] * df["vwap"]

        # --- Sumar por día ---
        daily = df.groupby("date").agg(
            day_dollar_volume=("dollar_volume", "sum"),
            symbol=("symbol", "first")
        ).reset_index()

        # --- Rolling 20 días (sin incluir día actual) ---
        daily["dvol_ma20"] = daily["day_dollar_volume"].rolling(20).mean().shift(1)

        # --- Relative Dollar Volume ---
        daily["rvol_20d"] = daily["day_dollar_volume"] / daily["dvol_ma20"]

        # Remover días sin suficiente historial
        #daily = daily.dropna(subset=["rvol_20d"])

        daily['timestamp'] = pd.to_datetime(daily['date']).dt.tz_localize(ZoneInfo("America/New_York")) + pd.Timedelta(hours=4)
        daily = daily.drop(columns=['date'])
        daily.insert(1, 'timestamp', daily.pop('timestamp'))

        # Salida final
        return daily[["symbol", "timestamp", "rvol_20d"]]
    
    @staticmethod
    def compute_gap_pct(df) -> pd.DataFrame:
        """
        Calcula el overnight/gap basado EXCLUSIVAMENTE en el primer open del premarket.

        Reglas:
        - open_current = primera vela del día con timestamp < 09:30 (NY)
        - si NO existe premarket → open_current = 0
        - close_prev = último close del día anterior
        - si open_current = 0 → gap_pct = 0

        Salida:
        - symbol
        - date
        - gap_pct
        """
        df = df.copy()

        # --- Asegurar timezone NY ---
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["timestamp"] = df["timestamp"].dt.tz_convert("America/New_York")
        df["date"] = df["timestamp"].dt.date
        df["time"] = df["timestamp"].dt.time

        # Hora límite para premarket
        REGULAR_OPEN = datetime.time(9, 30)

        # --- Obtener todos los días únicos con su símbolo ---
        all_days = (
            df.groupby("date")
            .agg(symbol=("symbol", "first"))
            .reset_index()
        )

        # --- Para cada día: primera vela del premarket ---
        premarket = (
            df[df["time"] < REGULAR_OPEN]
            .groupby("date")
            .agg(pre_open=("open", "first"))
            .reset_index()
        )

        # Merge y rellenar con 0 si no hay premarket
        daily = all_days.merge(premarket, on="date", how="left")
        daily["pre_open"] = daily["pre_open"].fillna(0)

        # --- Obtener último close de cada día ---
        last_close = (
            df.groupby("date")
            .agg(last_close=("close", "last"))
            .reset_index()
        )

        daily = daily.merge(last_close, on="date", how="left")

        # --- prev_close = último close del día ANTERIOR ---
        daily["prev_close"] = daily["last_close"].shift(1)

        # --- Calcular gap_pct ---
        # Primero inicializar en 0
        daily["gap_pct"] = 0.0

        # Calcular solo donde tenemos prev_close válido Y pre_open != 0
        mask = (daily["prev_close"].notna()) & (daily["pre_open"] != 0)
        daily.loc[mask, "gap_pct"] = (
            (daily.loc[mask, "pre_open"] - daily.loc[mask, "prev_close"]) 
            / daily.loc[mask, "prev_close"]
        )


        daily['timestamp'] = pd.to_datetime(daily['date']).dt.tz_localize(ZoneInfo("America/New_York")) + pd.Timedelta(hours=4)
        daily = daily.drop(columns=['date'])
        daily.insert(1, 'timestamp', daily.pop('timestamp'))

        # Retornar solo las columnas necesarias
        return daily[["symbol", "timestamp", "gap_pct"]]
    
    @staticmethod
    def compute_dist_max_20d(df) -> pd.DataFrame:
        """
        Calcula la distancia al máximo de los últimos 20 días usando solo datos hasta n-1.

        dist_max_20d(n) = (close_{n-1} - max(close_{n-20:n-1})) / max(close_{n-20:n-1})

        Output:
        - symbol
        - date (día n)
        - dist_max_20d
        """
        df = df.copy()

        # --- Asegurar timestamp NY ---
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["timestamp"] = df["timestamp"].dt.tz_convert("America/New_York")

        # Fecha diaria
        df["date"] = df["timestamp"].dt.date

        # --- Crear OHLC diario ---
        daily = df.groupby("date").agg(
            close_daily=("close", "last"),
            symbol=("symbol", "first")
        ).reset_index()

        # --- max_20d usando SOLO datos anteriores ---
        daily["max_20d_prev"] = daily["close_daily"].shift(1).rolling(20).max()

        # --- dist_max_20d ---
        daily["dist_max_20d"] = (
            (daily["close_daily"].shift(1) - daily["max_20d_prev"]) / daily["max_20d_prev"]
        )

        # Remover las primeras 21 filas que no tienen info previa
        #daily = daily.dropna(subset=["dist_max_20d"])

        daily['timestamp'] = pd.to_datetime(daily['date']).dt.tz_localize(ZoneInfo("America/New_York")) + pd.Timedelta(hours=4)
        daily = daily.drop(columns=['date'])
        daily.insert(1, 'timestamp', daily.pop('timestamp'))

        return daily[["symbol", "timestamp", "dist_max_20d"]]
    
    @staticmethod
    def compute_upper_wick_ratio_last1d(df) -> pd.DataFrame:
        """
        Calcula el upper wick ratio promedio del día n-1.
        
        upper_wick_ratio = (high - max(open, close)) / (high - low)
        
        Salida:
        - symbol
        - date (día n)
        - upper_wick_ratio_last1d
        """
        df = df.copy()

        # Timestamp a NY
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["timestamp"] = df["timestamp"].dt.tz_convert("America/New_York")
        
        # Fecha diaria
        df["date"] = df["timestamp"].dt.date

        # --- Calcular upper wick por vela ---
        df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
        df["candle_range"] = df["high"] - df["low"]

        # Evitar división por 0
        df["upper_wick_ratio"] = np.where(
            df["candle_range"] == 0,
            0,
            df["upper_wick"] / df["candle_range"]
        )

        # --- Promedio diario por día ---
        daily = df.groupby("date").agg(
            upper_wick_ratio_last1d=("upper_wick_ratio", "mean"),
            symbol=("symbol", "first")
        ).reset_index()

        # Shift para asignar al día n (usar valores del día n-1)
        daily["upper_wick_ratio_last1d"] = daily["upper_wick_ratio_last1d"].shift(1)

        # Eliminar primer día (NaN)
        #daily = daily.dropna(subset=["upper_wick_ratio_last1d"])

        daily['timestamp'] = pd.to_datetime(daily['date']).dt.tz_localize(ZoneInfo("America/New_York")) + pd.Timedelta(hours=4)
        daily = daily.drop(columns=['date'])
        daily.insert(1, 'timestamp', daily.pop('timestamp'))

        return daily[["symbol", "timestamp", "upper_wick_ratio_last1d"]]

    @staticmethod
    def compute_lower_wick_ratio_last1d(df) -> pd.DataFrame:
        """
        Calcula el lower wick ratio promedio del día n-1.
        
        lower_wick_ratio = (min(open, close) - low) / (high - low)
        
        Salida:
        - symbol
        - date (día n)
        - lower_wick_ratio_last1d
        """
        df = df.copy()

        # Timestamp a NY
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["timestamp"] = df["timestamp"].dt.tz_convert("America/New_York")
        
        # Fecha diaria
        df["date"] = df["timestamp"].dt.date

        # --- Calcular lower wick por vela ---
        df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
        df["candle_range"] = df["high"] - df["low"]

        # Evitar división por 0
        df["lower_wick_ratio"] = np.where(
            df["candle_range"] == 0,
            0,
            df["lower_wick"] / df["candle_range"]
        )

        # --- Promedio diario por día ---
        daily = df.groupby("date").agg(
            lower_wick_ratio_last1d=("lower_wick_ratio", "mean"),
            symbol=("symbol", "first")
        ).reset_index()

        # Shift para asignar al día n (usar valores del día n-1)
        daily["lower_wick_ratio_last1d"] = daily["lower_wick_ratio_last1d"].shift(1)

        # Eliminar primer día (NaN)
        #daily = daily.dropna(subset=["lower_wick_ratio_last1d"])

        daily['timestamp'] = pd.to_datetime(daily['date']).dt.tz_localize(ZoneInfo("America/New_York")) + pd.Timedelta(hours=4)
        daily = daily.drop(columns=['date'])
        daily.insert(1, 'timestamp', daily.pop('timestamp'))

        return daily[["symbol", "timestamp", "lower_wick_ratio_last1d"]]

    @staticmethod
    def compute_weekday(df) -> pd.DataFrame:
        """
        Crea columnas one-hot para el día de la semana.
        Salida:
        - symbol
        - date
        - weekday_0 ... weekday_4 (lunes=0,...,viernes=4)
        """
        import pandas as pd
        from zoneinfo import ZoneInfo

        df = df.copy()

        # Asegurarse de tener 'date' tipo datetime.date
        if 'timestamp' in df.columns:
            df["date"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("America/New_York").dt.date

        # Día de la semana
        df["weekday"] = pd.to_datetime(df["date"]).dt.day_name()

        # Combinar
        output = df.groupby(['symbol', 'date'])['weekday'].first().reset_index()

        output['timestamp'] = pd.to_datetime(output['date']).dt.tz_localize(ZoneInfo("America/New_York")) + pd.Timedelta(hours=4)
        output = output.drop(columns=['date'])
        output.insert(1, 'timestamp', output.pop('timestamp'))

        return output

    @staticmethod
    def compute_weekday_cyclic(df) -> pd.DataFrame:
        """
        Convierte el día de la semana en encoding cíclico (seno y coseno).
        
        Input:
        - df: DataFrame con columnas 'timestamp' o 'date' y 'symbol'
        
        Output:
        - DataFrame con columnas:
            symbol, date, weekday_sin, weekday_cos
        """
        df = df.copy()

        # Asegurarse de tener 'date' como datetime.date
        if 'timestamp' in df.columns:
            df['date'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_convert("America/New_York").dt.date
        else:
            df['date'] = pd.to_datetime(df['date']).dt.date

        # Día de la semana: lunes=0, martes=1, ..., viernes=4
        df['weekday'] = pd.to_datetime(df['date']).dt.weekday

        # Solo conservar lunes a viernes
        df = df[df['weekday'] < 5]

        # Encoding cíclico
        df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 5)
        df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 5)

        # Salida final
        output = df[['symbol', 'date', 'weekday_sin', 'weekday_cos']].drop_duplicates().reset_index(drop=True)

        output['timestamp'] = pd.to_datetime(output['date']).dt.tz_localize(ZoneInfo("America/New_York")) + pd.Timedelta(hours=4)
        output = output.drop(columns=['date'])
        output.insert(1, 'timestamp', output.pop('timestamp'))

        return output

    @staticmethod
    def compute_days_since_holiday_general(df) -> pd.DataFrame:
        """
        Calcula los días transcurridos desde el último festivo del mercado de EE.UU.
        Funciona para cualquier fecha.
        
        Input:
        - df: DataFrame con columna 'date' o 'timestamp' y 'symbol'
        
        Output:
        - symbol, date, days_since_holiday
        """
        df = df.copy()

        # Asegurarse de tener 'date'
        df['date'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_convert("America/New_York").dt.date

        df = df.sort_values('date').drop_duplicates(subset=['date','symbol']).reset_index(drop=True)

        # Generar calendario de festivos US
        cal = USFederalHolidayCalendar()
        holidays = cal.holidays(start=df['date'].min(), end=df['date'].max()).date

        # Usar searchsorted para eficiencia
        holiday_dates = np.array(holidays)
        days_since_holiday = []
        for d in df['date']:
            idx = np.searchsorted(holiday_dates, d, side='right') - 1
            if idx >= 0:
                delta = (d - holiday_dates[idx]).days
            else:
                delta = np.nan  # No hubo festivo previo
            days_since_holiday.append(delta)

        df['days_since_holiday'] = days_since_holiday


        df['timestamp'] = pd.to_datetime(df['date']).dt.tz_localize(ZoneInfo("America/New_York")) + pd.Timedelta(hours=4)
        df = df.drop(columns=['date'])
        df.insert(1, 'timestamp', df.pop('timestamp'))

        # Salida
        return df[['symbol', 'timestamp', 'days_since_holiday']]
    
    @staticmethod
    def compute_hhi_premarket_volume(df) -> pd.DataFrame:
        """
        Calcula el índice de Herfindahl-Hirschman (HHI) del volumen
        durante el pre-market (04:00–09:30 NY time) para cada día.

        Input:
        df con columnas:
            - timestamp
            - volume
            - symbol

        Output:
        DataFrame con:
            - symbol
            - date
            - hhi_premarket_volume
        """
        df = df.copy()

        # Convertir timestamp a NY
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df['timestamp'] = df['timestamp'].dt.tz_convert("America/New_York")

        # Extraer fecha y hora local
        df['date'] = df['timestamp'].dt.date
        df['time'] = df['timestamp'].dt.time

        # Definir rango pre-market
        pre_start = pd.to_datetime("04:00:00").time()
        pre_end   = pd.to_datetime("09:30:00").time()

        # Filtrar velas en horario pre-market
        pre = df[(df['time'] >= pre_start) & (df['time'] < pre_end)].copy()

        # HHI por día
        def hhi(vols):
            total = vols.sum()
            if total == 0:
                return np.nan
            p = vols / total
            return (p ** 2).sum()

        hhi_daily = pre.groupby(['symbol', 'date'])['volume'].apply(hhi).reset_index()
        hhi_daily = hhi_daily.rename(columns={'volume': 'hhi_premarket_volume'})

        hhi_daily['timestamp'] = pd.to_datetime(hhi_daily['date']).dt.tz_localize("America/New_York") + pd.Timedelta(hours=4)
        hhi_daily = hhi_daily.drop(columns=['date'])
        hhi_daily.insert(1, 'timestamp', hhi_daily.pop('timestamp'))

        return hhi_daily

    @staticmethod
    def compute_premarket_zscore(df, lookback=20) -> pd.DataFrame:
        """
        Calcula:
        - R_pm : retorno del premarket
        - mu_m : media de daily returns últimos N días
        - sigma_m : std de daily returns últimos N días
        - z_pm : (R_pm - mu_m) / sigma_m

        Input:
        df: dataframe intradía de una sola acción

        Output:
        dataframe con:
            symbol, date, premarket_return, mu_m, sigma_m, z_pm
        """
        df = df.copy()

        # Convert timestamp to NY
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df['timestamp'] = df['timestamp'].dt.tz_convert("America/New_York")

        df['date'] = df['timestamp'].dt.date
        df['time'] = df['timestamp'].dt.time

        symbol = df['symbol'].iloc[0]

        # --- 1. PREMARKET RETURN ---
        pre_start = pd.to_datetime("04:00:00").time()
        pre_end   = pd.to_datetime("09:30:00").time()

        pre = df[(df['time'] >= pre_start) & (df['time'] < pre_end)]

        pm_daily = pre.groupby('date').agg(
            pm_open=('open', 'first'),
            pm_last=('close', 'last')
        ).reset_index()

        pm_daily['premarket_return'] = (
            (pm_daily['pm_last'] - pm_daily['pm_open']) / pm_daily['pm_open']
        )

        # --- 2. DAILY RETURNS (CLOSE-TO-CLOSE) ---
        daily = df.groupby('date')['close'].last().reset_index()
        daily['daily_return'] = daily['close'].pct_change()

        # --- 3. ROLLING MEAN AND STD ---
        daily['mu_m'] = daily['daily_return'].rolling(lookback).mean()
        daily['sigma_m'] = daily['daily_return'].rolling(lookback).std()

        # Alineamos con pm_daily (que es por día)
        merged = pm_daily.merge(daily, on='date', how='left')

        # --- 4. Z-SCORE ---
        merged['z_pm'] = (
            (merged['premarket_return'] - merged['mu_m']) / merged['sigma_m']
        )

        # limpiar output
        output = merged[['date', 'premarket_return', 'mu_m', 'sigma_m', 'z_pm']].copy()
        output['symbol'] = symbol

        output['timestamp'] = pd.to_datetime(output['date']).dt.tz_localize("America/New_York") + pd.Timedelta(hours=4)
        output = output.drop(columns=['date'])
        output.insert(1, 'timestamp', output.pop('timestamp'))

        return output[['symbol', 'timestamp', 'z_pm']] #output[['symbol', 'timestamp', 'premarket_return', 'mu_m', 'sigma_m', 'z_pm']]
    
    @staticmethod
    def compute_premarket_avg_trade_size_ratio(df, lookback=20, eps=0.01) -> pd.DataFrame:
        """
        Calcula R_PM_hist:
        log( (avg_trade_size_today + eps) / (median_historical_avg_trade_size + eps) )
        
        Input:
        df  -> dataframe intradía de una sola acción
        Output:
        dataframe con columnas:
            symbol, date, R_PM_hist
        """
        df = df.copy()

        # Convertir timestamp a NY
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df['timestamp'] = df['timestamp'].dt.tz_convert("America/New_York")

        df['date'] = df['timestamp'].dt.date
        df['time'] = df['timestamp'].dt.time

        symbol = df['symbol'].iloc[0]

        # --- Filtrar premarket ---
        pre_start = pd.to_datetime("04:00:00").time()
        pre_end   = pd.to_datetime("09:30:00").time()

        pre = df[(df['time'] >= pre_start) & (df['time'] < pre_end)]

        # If no premarket, return empty DF
        if pre.empty:
            return pd.DataFrame(columns=['symbol', 'date', 'R_PM_hist'])

        # --- 1. avg trade size por día ---
        pm_daily = pre.groupby('date').agg(
            total_volume=('volume', 'sum'),
            total_trades=('trade_count', 'sum')
        ).reset_index()

        pm_daily['avg_trade_size'] = pm_daily['total_volume'] / pm_daily['total_trades']

        # --- 2. baseline histórico (rolling median) ---
        pm_daily['median_hist'] = (
            pm_daily['avg_trade_size']
            .rolling(lookback)
            .median()
        )

        # --- 3. log ratio ---
        pm_daily['R_PM_hist'] = np.log(
            (pm_daily['avg_trade_size'] + eps) /
            (pm_daily['median_hist'] + eps)
        )

        # Output final
        out = pm_daily[['date', 'R_PM_hist']].copy()
        out['symbol'] = symbol
        out['timestamp'] = pd.to_datetime(out['date']).dt.tz_localize("America/New_York") + pd.Timedelta(hours=4)
        out = out.drop(columns=['date'])
        out.insert(1, 'timestamp', out.pop('timestamp'))

        return out[['symbol', 'timestamp', 'R_PM_hist']]
    
    @staticmethod
    def compute_premarket_vwap_return(df) -> pd.DataFrame:
        """
        Calcula:
            - VWAP_pm: VWAP ponderado del premarket
            - PM_VWAP_Return = VWAP_pm / close_prev - 1

        Input:
            df: dataframe intradia de una sola acción con columnas:
                timestamp, open, high, low, close, vwap, volume, trade_count, symbol

        Output:
            dataframe con columnas:
                symbol, date, PM_VWAP_Return
        """
        df = df.copy()

        # Convertir timestamp a NY
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df['timestamp'] = df['timestamp'].dt.tz_convert("America/New_York")

        df['date'] = df['timestamp'].dt.date
        df['time'] = df['timestamp'].dt.time

        symbol = df['symbol'].iloc[0]

        # --- 1. Filtrar premarket 04:00 - 09:30 ---
        pre_start = pd.to_datetime("04:00:00").time()
        pre_end   = pd.to_datetime("09:30:00").time()

        pre = df[(df['time'] >= pre_start) & (df['time'] < pre_end)]

        # VWAP pm: Σ(VWAP_i * vol_i) / Σ(vol_i)
        pm_daily = pre.groupby('date').apply(
            lambda g: np.sum(g['vwap'] * g['volume']) / np.sum(g['volume'])
            if g['volume'].sum() > 0 else np.nan,
            include_groups=False
        ).reset_index(name='vwap_pm')

        # --- 2. Obtener el close previo (solo regular hours) ---
        reg_start = pd.to_datetime("09:30:00").time()
        reg_end   = pd.to_datetime("16:00:00").time()

        reg = df[(df['time'] >= reg_start) & (df['time'] < reg_end)]

        daily_close = reg.groupby('date')['close'].last().reset_index()
        daily_close['prev_close'] = daily_close['close'].shift(1)

        # Merge vwap_pm con prev_close
        merged = pm_daily.merge(daily_close[['date', 'prev_close']], on='date', how='left')

        # --- 3. PM VWAP Return ---
        merged['PM_VWAP_Return'] = merged['vwap_pm'] / merged['prev_close'] - 1

        # Output final limpio
        out = merged[['date', 'PM_VWAP_Return']].copy()
        out['symbol'] = symbol

        out['timestamp'] = pd.to_datetime(out['date']).dt.tz_localize("America/New_York") + pd.Timedelta(hours=4)
        out = out.drop(columns=['date'])

        return out[['symbol', 'timestamp', 'PM_VWAP_Return']]
    
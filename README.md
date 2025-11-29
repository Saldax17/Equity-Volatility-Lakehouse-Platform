# 📈 Equity Volatility Lakehouse Platform (EVLP)

**EVLP** es una plataforma end-to-end diseñada para analizar y predecir la volatilidad de acciones del mercado estadounidense usando una arquitectura moderna basada en *Lakehouse* (Medallion: Bronze–Silver–Gold), procesamiento distribuido con Spark, APIs financieras y modelos de Machine Learning.

---

# 🧩 1. Descripción del Proyecto

La *Equity Volatility Lakehouse Platform (EVLP)* ingesta datos financieros desde Alpaca, EODHD e iShares; limpia y transforma los datos en capas (Bronze → Silver → Gold); genera características avanzadas de volatilidad; y entrena modelos de machine learning para detectar episodios de alta volatilidad.

Incluye:
- Ingesta de datos con Python.
- Procesamiento masivo con Spark (AWS Glue).
- Lakehouse con Apache Iceberg en S3.
- Feature engineering para series de tiempo.
- Modelos ML de clasificación.
- Visualización y análisis descriptivo.

---

# 🏗 2. Arquitectura del Proyecto

## 🔶 **Medallion Architecture (Lakehouse)**

### 🥉 Bronze  
Datos crudos tal como provienen de las APIs:
- OHLCV de Alpaca  
- Constituyentes históricos de EODHD  
- Listas de ETFs de iShares  

### 🥈 Silver  
Datos limpios y estandarizados con Spark:
- Timestamps normalizados  
- Rejilla temporal completa  
- Imputación (forward-fill / backfill)  

### 🥇 Gold  
Feature engineering:
- % High–Low  
- % Open–Close  
- Gaps  
- Lookbacks (1d, 7d, 28d, 112d)  

## 🧠 **ML Pipeline**
Modelos considerados:
- Logistic Regression  
- Random Forest  
- Gradient Boosting  
- K-Means (clustering de volatilidad)

Evaluación:
- Maximización del f1-score


---

# 📂 3. Estructura del Repositorio

```bash

Equity-Volatility-Lakehouse-Platform/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── config/
│   ├── __init__.py
│   └── parameters.py
│
├── data_alpaca/
│   ├── __init__.py
│   ├── alpa.py
│   └── bars.py
│
├── data_apis/
│   ├── __init__.py
│   ├── my_alpaca.py
│   ├── my_eodhd.py
│   ├── my_ishares.py
│   ├── my_models.py
│   ├── my_stock_functions.py
│   └── helpers.py
│
├── docs/
│   ├── Informe_Final.pdf
│   ├── Presentacion.pdf
│   ├── Preliminar.pdf
│   ├── Propuesta.pdf
│   └── arquitectura_medallion.drawio
│
├── architecture/
│   ├── arquitectura_medallion.png
│   ├── pipeline_completo.png
│   └── arquitectura_aws.png
│
├── data/     # (Ignorado en GitHub)
│   ├── bronze/
│   ├── silver/
│   └── gold/


```


# ⚙️ 4. Instalación

### Requisitos
- Python 3.9+
- pip
- Cuenta en Alpaca y EODHD (para API keys)
- Spark 3.x (si corres procesamiento local)

### Instalación

git clone https://github.com/<tu-usuario>/proyecto-volatilidad.git
cd proyecto-volatilidad
pip install -r requirements.txt






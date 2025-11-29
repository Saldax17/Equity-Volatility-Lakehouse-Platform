# 📈 Equity Volatility Lakehouse Platform (EVLP)

**EVLP** es una plataforma end-to-end diseñada para analizar y predecir la volatilidad de acciones del mercado estadounidense usando una arquitectura moderna basada en **Lakehouse** (Medallion: Bronze–Silver–Gold), procesamiento distribuido con **Spark**, APIs financieras y modelos de **Machine Learning**.

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.9+-blue?logo=python" />
  <img src="https://img.shields.io/badge/Spark-3.x-orange?logo=apache-spark" />
  <img src="https://img.shields.io/badge/AWS%20Glue-ETL-yellow?logo=amazon-aws" />
  <img src="https://img.shields.io/badge/Apache%20Iceberg-Lakehouse-green?logo=apache" />
  <img src="https://img.shields.io/badge/ML-Classification-red?logo=google" />
</p>

---

# 📥 0. Ingesta de Datos desde APIs (Raw Ingestion Layer)

La plataforma inicia con una fase de ingesta avanzada que combina datos de **EODHD**, **Alpaca** e **iShares** para construir el dataset base que alimenta la capa Bronze en el Lakehouse.

📄 Código: `ingestion/consulta_de_datos_con_apis.py`  


### ✔ EODHD – Índices y Constituyentes Históricos
- Descarga de índices: **GSPC**, **MID**, **SML**
- Obtención de constituyentes históricos diarios
- Limpieza de sufijos en tickers (`_old`, `_old1`, `_old2`)
- Reemplazo de caracteres inválidos (`-` → `.`)
- Normalización a timezone **America/New_York**

### ✔ iShares – ETF Constituents
- Carga de componentes del ETF **IWB** desde archivo XLS
- Generación de listas de símbolos para consulta masiva

### ✔ Alpaca – Historical Bars (30m)
- Descarga de datos OHLCV en intervalos de 30 minutos
- Rango histórico: **2007 – 2025**
- Ajustes: RAW / ALL
- Uso de:
  - MyAlpacaJob  
  - MyAlpacaStock  
  - StockHistoricalDataClient  

### ✔ Dataset Final
- Ensamble diario de constituyentes activos
- Descarga selectiva de barras 30m por símbolo
- Preparación final para ser almacenado como **Bronze** en S3

---

# 🧩 1. Descripción del Proyecto

La plataforma procesa datos financieros a través de las capas Bronze → Silver → Gold, realiza feature engineering avanzado y entrena modelos de machine learning orientados a clasificar volatilidad.

Incluye:
- Ingesta desde APIs financieras  
- Procesamiento distribuido (Spark / AWS Glue)  
- Lakehouse con Apache Iceberg  
- Feature engineering (volatilidad y lookbacks)  
- Modelos ML de clasificación y clustering  
- Visualización y análisis descriptivo  

---

# 🏗 2. Arquitectura del Proyecto

## 🔶 Medallion Architecture (Lakehouse)

### 🥉 Bronze – Raw Layer
- Datos crudos desde Alpaca, EODHD e iShares  
- Sin transformación  
- Historial de constituyentes y OHLCV 30m  

---

### 🥈 Silver – Clean Layer
- Normalización de timestamps (NY timezone)  
- Rejilla temporal completa (30 minutos)  
- Imputación forward-fill / backfill  
- Unión símbolo × timestamp  

---

### 🥇 Gold – Feature Layer
Feature engineering orientado a series de tiempo:
- % High–Low  
- % Open–Close  
- Gaps de apertura  
- Lookbacks: 1d, 7d, 28d, 112d  

---

# ⚙️ 3. Pipeline de Procesamiento (AWS Glue • Spark • Iceberg)

## 🥈 Fase 1 — Limpieza y Rejilla Temporal (Silver Layer)
📄 Código: `processing/procesamiento_fase_1.py`  
Origen técnico: :contentReference[oaicite:1]{index=1}

Incluye:
- Lectura desde Iceberg (Bronze)  
- Selección aleatoria de símbolos representativos  
- Construcción de rejilla temporal (30 min)  
- Join símbolo × timestamp  
- Forward-fill y backfill de OHLCV  
- Limpieza de volumen y trade_count  
- Escritura en Iceberg:
  **`proyecto1db.stock_iceberg_sample`**

---

## 🥇 Fase 2 — Feature Engineering (Gold Layer)
📄 Código: `processing/procesamiento_fase_2.py`  
Origen técnico: :contentReference[oaicite:2]{index=2}

Incluye:
- Cálculo de volatilidad (% High–Low, % Open–Close)  
- Gap de apertura vs close previo  
- Lookbacks:
  - 1d, 7d, 28d, 112d  
- Generación de columnas `pct_change_<period>`  
- Limpieza de columnas auxiliares  
- Ordenamiento por símbolo + timestamp  

---

# 🧠 4. ML Pipeline

Modelos implementados:
- Logistic Regression  
- Decision Tree  
- Random Forest  
- Gradient Boosting  
- XGBoost  
- K-Means (clustering)

Evaluación del desempeño:
- **F1-score** como métrica principal  

---

# 📂 5. Estructura del Repositorio

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
├── docs/
│   ├── Informe_Final.pdf
│   ├── Presentacion.pdf
│   ├── Preliminar.pdf
│   ├── Propuesta.pdf
│   └── arquitectura_medallion.drawio
│
├── ingestion/
│   └── consulta_de_datos_con_apis.ipynb   
│
├── processing/
│   ├── Procesamiento_fase_1.ipynb
│   └── Procesamiento_fase_2.ipynb 
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

```bash
git clone https://github.com/<tu-usuario>/proyecto-volatilidad.git
cd proyecto-volatilidad
pip install -r requirements.txt
```




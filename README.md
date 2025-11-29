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

# 🧩 1. Descripción del Proyecto

La *Equity Volatility Lakehouse Platform* ingesta datos desde **Alpaca**, **EODHD** e **iShares**; limpia y transforma los datos en capas Bronze → Silver → Gold; genera características avanzadas orientadas a volatilidad; y entrena modelos de machine learning para detectar episodios de alta volatilidad.

Incluye:
- Ingesta de datos con Python  
- Procesamiento distribuido con Spark (AWS Glue)  
- Lakehouse con Apache Iceberg sobre S3  
- Feature engineering orientado a series de tiempo  
- Modelos ML de clasificación  
- Visualización y análisis descriptivo  

---

# 🏗 2. Arquitectura del Proyecto

## 🔶 Medallion Architecture (Lakehouse)

### 🥉 Bronze – Raw Layer
Datos crudos tal como provienen de las APIs:
- OHLCV de Alpaca  
- Constituyentes históricos de EODHD  
- Listas de ETFs de iShares  

---

### 🥈 Silver – Clean Layer
Procesamiento con Spark:
- Normalización de timestamps  
- Rejilla temporal completa (30 min, solo días hábiles)  
- Imputación (forward-fill / backfill)  
- Unificación de símbolos × timestamps  

---

### 🥇 Gold – Feature Layer
Feature engineering para volatilidad:
- % High–Low  
- % Open–Close  
- Gaps de apertura  
- Lookbacks: 1d, 7d, 28d, 112d  

---

# ⚙️ 3. Pipeline de Procesamiento (AWS Glue • Spark • Iceberg)

El proyecto utiliza dos fases principales para transformar los datos y construir el Lakehouse.

## 🥈 **Fase 1 – Limpieza y Rejilla Temporal (Silver)**  
📄 Código base: `procesamiento_fase_1.py`  
- Lectura de tabla Iceberg Bronze  
- Selección aleatoria de símbolos  
- Generación de rejilla temporal (30m)  
- Join símbolo × timestamp  
- Forward-fill y backfill de OHLCV  
- Corrección de volumen y trade_count  
- Escritura a:  
  **`proyecto1db.stock_iceberg_sample`**

---

## 🥇 **Fase 2 – Feature Engineering (Gold)**  
📄 Código base: `procesamiento_fase_2.py`  
- Cálculo de volatilidad:
  - % High–Low  
  - % Open–Close  
  - Gap de apertura  
- Lookbacks:
  - 1d, 7d, 28d, 112d  
- Generación de:  
  `pct_change_<period>`  
- Limpieza de columnas auxiliares  
- Ordenamiento por símbolo + timestamp  

---

## 🔁 **Diagrama del Pipeline**
_(Guardado en `/architecture/pipeline_completo.png`)_  
Incluye:
- Ingesta  
- Bronze  
- Silver (F1)  
- Gold (F2)  
- ML Pipeline  

---

# 🧠 4. ML Pipeline

Modelos implementados:
- Logistic Regression  
- Decision Tree  
- Random Forest  
- Gradient Boosting  
- XGBoost  
- K-Means (clustering de volatilidad)

Evaluación:
- Maximización del **F1-score** como métrica principal  

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






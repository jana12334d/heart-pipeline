# Heart Disease ML Pipeline

## Overview
End-to-end big data pipeline for heart disease prediction using Spark, Kafka, and ML.

## Architecture
CSV → Kafka → Spark ETL → ML → Streamlit Dashboard

## Features
- Noise robustness testing (5%, 15%, 30%)
- Model comparison (SVM vs Random Forest)
- Live prediction dashboard

## How to Run
pip3 install -r requirements.txt
python3 kafka/etl_pipeline.py
python3 spark/spark_models.py
streamlit run dashboard/dashboard_app.py


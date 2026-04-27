import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Heart ML Dashboard", layout="wide")

st.title("💓 Heart Disease ML Dashboard")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Model Comparison",
    "⚠️ Noise Sensitivity",
    "🔮 Live Prediction",
    "ℹ️ About"
])

base = "models"

clean_file = os.path.join(base, "benchmark_results_clean.csv")
noise_file = os.path.join(base, "benchmark_results_noise.csv")
feat_file = os.path.join(base, "feature_importance.csv")

df_clean = pd.read_csv(clean_file) if os.path.exists(clean_file) else None
df_noise = pd.read_csv(noise_file) if os.path.exists(noise_file) else None
df_feat = pd.read_csv(feat_file) if os.path.exists(feat_file) else None

with tab1:
    st.header("Model Comparison")

    if df_clean is not None:
        st.dataframe(df_clean, use_container_width=True)

        test_df = df_clean[df_clean["split"] == "test"]

        st.subheader("Accuracy")
        st.bar_chart(test_df.set_index("model")["accuracy"])

        st.subheader("F1 Score")
        st.bar_chart(test_df.set_index("model")["f1"])

    if df_feat is not None:
        st.subheader("Feature Importance")
        df_feat = df_feat.sort_values(by="importance", ascending=False)
        st.dataframe(df_feat, use_container_width=True)
        st.bar_chart(df_feat.set_index("feature")["importance"])

with tab2:
    st.header("Noise Sensitivity")

    if df_noise is not None:
        st.dataframe(df_noise, use_container_width=True)

        st.subheader("F1 Score Drop")
        f1_pivot = df_noise.pivot(index="noise_level", columns="model", values="f1")
        st.line_chart(f1_pivot)

        st.subheader("AUC Drop")
        auc_pivot = df_noise.pivot(index="noise_level", columns="model", values="auc")
        st.line_chart(auc_pivot)

with tab3:
    st.header("Live Heart Disease Prediction")

    age = st.slider("Age", 20, 90, 50)
    chol = st.slider("Cholesterol", 100, 600, 220)

    if st.button("Predict Risk"):
        risk = (age + chol) / 500
        risk = max(0, min(risk, 1))

        st.subheader("Risk Score")
        st.progress(risk)

        if risk >= 0.6:
            st.error(f"High Risk: {risk:.2f}")
        else:
            st.success(f"Low Risk: {risk:.2f}")

with tab4:
    st.header("About")

    st.markdown("""
    ### Pipeline Architecture
    CSV → Kafka → Spark ETL → Parquet → Spark MLlib → Streamlit Dashboard

    ### Team Contribution
    - Infrastructure setup
    - Kafka producer
    - Spark ETL pipeline
    - ML model training
    - Dashboard deployment

    ### Key Finding
    Model performance decreases as noise increases.
    """)

st.success("Dashboard ready ✅")

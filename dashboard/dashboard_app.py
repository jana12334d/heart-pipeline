import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Heart ML Dashboard", layout="wide")

st.title("💓 Heart Disease ML Dashboard")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Model Comparison",
    "⚠️ Noise Sensitivity",
    "🔮 Live Prediction",
    "ℹ️ About"
])

base = os.path.expanduser("~/heart_pipeline/models/")

clean_file = os.path.join(base, "benchmark_results_clean.csv")
noise_file = os.path.join(base, "benchmark_results_noise.csv")
feat_file = os.path.join(base, "feature_importance.csv")

df_clean = pd.read_csv(clean_file) if os.path.exists(clean_file) else None
df_noise = pd.read_csv(noise_file) if os.path.exists(noise_file) else None
df_feat = pd.read_csv(feat_file) if os.path.exists(feat_file) else None

# -------- TAB 1 --------
with tab1:
    st.header("Model Comparison")

    if df_clean is not None:
        st.dataframe(df_clean)

        test_df = df_clean[df_clean["split"] == "test"]

        st.subheader("Accuracy")
        st.bar_chart(test_df.set_index("model")["accuracy"])

        st.subheader("F1 Score")
        st.bar_chart(test_df.set_index("model")["f1"])

    if df_feat is not None:
        st.subheader("Feature Importance")
        df_feat = df_feat.sort_values(by="importance", ascending=False)
        st.bar_chart(df_feat.set_index("feature")["importance"])

# -------- TAB 2 --------
with tab2:
    st.header("Noise Sensitivity")

    if df_noise is not None:
        st.dataframe(df_noise)

        pivot = df_noise.pivot(index="noise_level", columns="model", values="f1")
        st.line_chart(pivot)

# -------- TAB 3 --------
with tab3:
    st.header("Heart Disease Prediction")

    age = st.slider("Age", 20, 80, 50)
    chol = st.slider("Cholesterol", 100, 400, 200)

    if st.button("Predict Risk"):
        risk = (age + chol) / 500
        st.progress(min(risk, 1.0))

        if risk > 0.6:
            st.error("High Risk")
        else:
            st.success("Low Risk")

# -------- TAB 4 --------
with tab4:
    st.header("About")

    st.write("""
    Pipeline: Kafka → Spark → Parquet → ML → Dashboard
    
    Key finding:
    Model performance drops as noise increases.
    """)

st.success("Dashboard ready ✅")

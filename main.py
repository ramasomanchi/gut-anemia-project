# main.py

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Indian Gut-Anemia ML", layout="wide")

st.title("🩸 Indian Adolescent Gut-Anemia Research Portal")
st.markdown("Upload real questionnaire data or use simulated dataset.")

REQUIRED_COLUMNS = [
    "Hemoglobin_g_dL",
    "Serum_Ferritin",
    "Prevotella_Abundance",
    "Bacteroides_Abundance",
    "Lactobacillus_Abundance",
    "Bifidobacterium_Abundance",
]

@st.cache_data
def generate_sample_data():
    np.random.seed(42)
    n = 150
    data = pd.DataFrame({
        "Hemoglobin_g_dL": np.random.uniform(8, 14, n),
        "Serum_Ferritin": np.random.uniform(5, 40, n),
        "Prevotella_Abundance": np.random.uniform(25, 60, n),
        "Bacteroides_Abundance": np.random.uniform(5, 20, n),
        "Lactobacillus_Abundance": np.random.uniform(0.5, 10, n),
        "Bifidobacterium_Abundance": np.random.uniform(0.5, 8, n),
    })
    data["Anemic"] = (data["Hemoglobin_g_dL"] < 12).astype(int)
    return data


def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["Anemic"] = (
        df["Hemoglobin_g_dL"] < 12
    ).astype(int) if "Anemic" not in df.columns else df["Anemic"]

    return df


# --- DATA SOURCE ---
st.sidebar.header("Data Source")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        df = validate_dataframe(df)
        st.success("Custom dataset loaded successfully.")
    except Exception as e:
        st.error(str(e))
        st.stop()
else:
    df = generate_sample_data()
    st.info("Using simulated dataset.")


# --- NAVIGATION ---
st.sidebar.header("Research Tools")
tool = st.sidebar.radio(
    "Select Analysis",
    ["Dataset Preview", "Correlation Heatmap", "ML Predictor", "Probiotic Simulation"],
)

# --- TOOL 0: PREVIEW ---
if tool == "Dataset Preview":
    st.header("📋 Dataset Overview")
    st.dataframe(df)
    st.write("Shape:", df.shape)


# --- TOOL 1: CORRELATION ---
elif tool == "Correlation Heatmap":
    st.header("📊 Correlation Matrix")

    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df.drop(columns=["Anemic"]).corr(method="spearman")
    sns.heatmap(corr, annot=True, cmap="RdBu_r", center=0, ax=ax)
    st.pyplot(fig)


# --- TOOL 2: ML PREDICTOR ---
elif tool == "ML Predictor":
    st.header("🤖 Random Forest: Anemia Prediction")

    X = df[
        [
            "Prevotella_Abundance",
            "Bacteroides_Abundance",
            "Lactobacillus_Abundance",
            "Bifidobacterium_Abundance",
        ]
    ]
    y = df["Anemic"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    st.metric("Model Accuracy", f"{accuracy:.2f}")

    importances = pd.Series(
        model.feature_importances_, index=X.columns
    ).sort_values()

    fig2, ax2 = plt.subplots()
    importances.plot(kind="barh", ax=ax2)
    st.pyplot(fig2)

    st.subheader("Simulate Patient Profile")
    l_in = st.slider("Lactobacillus (%)", 0.0, 20.0, 5.0)
    p_in = st.slider("Prevotella (%)", 0.0, 80.0, 40.0)

    input_df = pd.DataFrame(
        [[p_in, 12.0, l_in, 4.0]],
        columns=X.columns,
    )

    prediction = model.predict(input_df)
    status = "🚩 High Anemia Risk" if prediction[0] == 1 else "✅ Low Anemia Risk"
    st.metric("Prediction", status)


# --- TOOL 3: PROBIOTIC SIMULATION ---
elif tool == "Probiotic Simulation":
    st.header("📈 Probiotic Intervention Impact")

    growth = st.slider("Projected Lactobacillus Increase (%)", 0, 200, 50)

    fig3, ax3 = plt.subplots()
    sns.kdeplot(df["Lactobacillus_Abundance"], fill=True, ax=ax3)
    sns.kdeplot(
        df["Lactobacillus_Abundance"] * (1 + growth / 100),
        fill=True,
        ax=ax3,
    )
    st.pyplot(fig3)

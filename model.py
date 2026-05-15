import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Income Predictor",
    page_icon="💰",
    layout="wide",
)

# ── Load model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

FEATURES = list(model.feature_names_in_)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/money-bag.png", width=80)
st.sidebar.title("Income Predictor")
st.sidebar.caption("Random Forest Regressor · 200 estimators · 24 features")
st.sidebar.markdown("---")
st.sidebar.markdown("Fill in the details on the right to get an **income prediction**.")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("💰 Income Prediction App")
st.markdown("Enter personal and professional details below to predict income.")
st.markdown("---")

# ── Input form ────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Personal Info")
    age = st.slider("Age", 18, 80, 35)
    gender = st.selectbox("Gender", ["Female", "Male"])
    marital_status = st.selectbox("Marital Status", ["Divorced", "Married", "Single"])
    education_level = st.selectbox(
        "Education Level",
        [1, 2, 3, 4, 5],
        format_func=lambda x: {1: "1 – No schooling", 2: "2 – High School", 3: "3 – Bachelor's",
                                4: "4 – Master's", 5: "5 – Doctorate"}[x],
        index=2,
    )
    number_of_dependents = st.number_input("Number of Dependents", 0, 10, 1)
    household_size = st.number_input("Household Size", 1, 20, 3)

with col2:
    st.subheader("💼 Work & Finance")
    work_experience = st.slider("Work Experience (years)", 0, 50, 10)
    occupation = st.selectbox("Occupation", ["Finance", "Healthcare", "Others", "Technology"])
    employment_status = st.selectbox("Employment Status", ["Full-time", "Part-time", "Self-employed"])
    homeownership = st.selectbox("Homeownership Status", ["Own", "Rent"])

with col3:
    st.subheader("🏠 Living Situation")
    location = st.selectbox("Location", ["Rural", "Suburban", "Urban"])
    housing_type = st.selectbox(
        "Type of Housing",
        ["Apartment", "Single-family home", "Townhouse"],
    )
    transport = st.selectbox(
        "Primary Mode of Transportation",
        ["Car", "Public transit", "Walking", "Other"],
    )

st.markdown("---")

# ── Build feature vector ──────────────────────────────────────────────────────
def build_features():
    # Derived features
    experience_ratio = work_experience / max(age, 1)
    dependency = number_of_dependents / max(household_size, 1)

    feat = {
        "Age": age,
        "Education_Level": education_level,
        "Number_of_Dependents": number_of_dependents,
        "Work_Experience": work_experience,
        "Household_Size": household_size,
        "Occupation_Finance": int(occupation == "Finance"),
        "Occupation_Healthcare": int(occupation == "Healthcare"),
        "Occupation_Others": int(occupation == "Others"),
        "Occupation_Technology": int(occupation == "Technology"),
        "Location_Suburban": int(location == "Suburban"),
        "Location_Urban": int(location == "Urban"),
        "Gender_Male": int(gender == "Male"),
        "Marital_Status_Married": int(marital_status == "Married"),
        "Marital_Status_Single": int(marital_status == "Single"),
        "Employment_Status_Part-time": int(employment_status == "Part-time"),
        "Employment_Status_Self-employed": int(employment_status == "Self-employed"),
        "Homeownership_Status_Rent": int(homeownership == "Rent"),
        "Type_of_Housing_Single-family home": int(housing_type == "Single-family home"),
        "Type_of_Housing_Townhouse": int(housing_type == "Townhouse"),
        "Primary_Mode_of_Transportation_Car": int(transport == "Car"),
        "Primary_Mode_of_Transportation_Public transit": int(transport == "Public transit"),
        "Primary_Mode_of_Transportation_Walking": int(transport == "Walking"),
        "Experience_ratio": experience_ratio,
        "Dependency": dependency,
    }
    return np.array([[feat[f] for f in FEATURES]])

# ── Predict button ────────────────────────────────────────────────────────────
predict_btn = st.button("🔮 Predict Income", type="primary", use_container_width=True)

if predict_btn:
    X = build_features()
    prediction = model.predict(X)[0]

    # Individual tree predictions for uncertainty
    tree_preds = np.array([tree.predict(X)[0] for tree in model.estimators_])
    low, high = np.percentile(tree_preds, [10, 90])

    st.markdown("---")
    st.subheader("📊 Prediction Results")

    r1, r2, r3 = st.columns(3)
    r1.metric("Predicted Income (log scale)", f"{prediction:.4f}")
    r2.metric("80% CI Lower", f"{low:.4f}")
    r3.metric("80% CI Upper", f"{high:.4f}")

    # ── Distribution of tree predictions ─────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor("#0e1117")
    for ax in axes:
        ax.set_facecolor("#262730")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    # Histogram of tree predictions
    axes[0].hist(tree_preds, bins=40, color="#4f8ef7", edgecolor="#262730", alpha=0.85)
    axes[0].axvline(prediction, color="#f7c948", linewidth=2, label=f"Mean: {prediction:.3f}")
    axes[0].axvline(low, color="#ff6b6b", linewidth=1.5, linestyle="--", label=f"10th pct: {low:.3f}")
    axes[0].axvline(high, color="#6bffb8", linewidth=1.5, linestyle="--", label=f"90th pct: {high:.3f}")
    axes[0].set_title("Distribution of Tree Predictions")
    axes[0].set_xlabel("Predicted Income (log)")
    axes[0].set_ylabel("Count")
    axes[0].legend(facecolor="#1a1a2e", edgecolor="#444", labelcolor="white", fontsize=8)

    # Top feature importances
    fi_pairs = sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1])[:10]
    names, vals = zip(*fi_pairs)
    colors = ["#4f8ef7" if v < 0.1 else "#f7c948" if v < 0.15 else "#6bffb8" for v in vals]
    axes[1].barh(names[::-1], vals[::-1], color=colors[::-1])
    axes[1].set_title("Top 10 Feature Importances")
    axes[1].set_xlabel("Importance")

    plt.tight_layout()
    st.pyplot(fig)

    # ── Input summary ─────────────────────────────────────────────────────────
    with st.expander("📋 View Input Summary"):
        X_df = pd.DataFrame(build_features(), columns=FEATURES)
        st.dataframe(X_df.T.rename(columns={0: "Value"}), use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Model: RandomForestRegressor (sklearn) · 200 trees · 24 engineered features")
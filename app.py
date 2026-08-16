import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# PAGE SETUP
st.set_page_config(
    page_title="Cotiviti | TPO Claims Risk Engine",
    page_icon="🏥",
    layout="wide"
)

st.title("TPO Claims Anomaly and Pattern Detection Engine")
st.caption("Hybrid Machine Learning Screening and Agentic AI Decision Support System")
st.markdown("---")

# MOCK DATA GENERATION
@st.cache_data
def load_mock_claims_data():
    np.random.seed(42)
    n_samples = 300
    
    cpt_codes = [
        "99213 (Standard E&M)", 
        "99214 (High Complexity)", 
        "99215 (Max Complexity)", 
        "27447 (Knee Replacement)", 
        "93000 (ECG)"
    ]
    cpt_choice = np.random.choice(cpt_codes, n_samples, p=[0.4, 0.3, 0.15, 0.05, 0.10])
    
    billed_amounts = []
    for cpt in cpt_choice:
        if "99213" in cpt: billed_amounts.append(np.random.normal(120, 25))
        elif "99214" in cpt: billed_amounts.append(np.random.normal(250, 40))
        elif "99215" in cpt: billed_amounts.append(np.random.normal(450, 80))
        elif "27447" in cpt: billed_amounts.append(np.random.normal(12000, 1500))
        else: billed_amounts.append(np.random.normal(80, 15))
        
    days_since_last_visit = np.random.randint(1, 90, n_samples)
    modifier_25_flag = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    prior_denial_count = np.random.choice([0, 1, 2, 3], n_samples, p=[0.7, 0.18, 0.08, 0.04])
    
    df = pd.DataFrame({
        "Claim_ID": [f"CLM-{1000+i}" for i in range(n_samples)],
        "CPT_Code": cpt_choice,
        "Billed_Amount": np.round(billed_amounts, 2),
        "Days_Since_Last_Visit": days_since_last_visit,
        "Modifier_25_Used": modifier_25_flag,
        "Prior_Denials": prior_denial_count
    })
    
    anomaly_condition = (
        ((df["CPT_Code"].str.contains("99215")) & (df["Billed_Amount"] > 480)) |
        ((df["Modifier_25_Used"] == 1) & (df["Prior_Denials"] >= 2)) |
        (df["Billed_Amount"] > 14000)
    )
    df["Is_Anomalous"] = np.where(anomaly_condition, 1, 0)
    
    return df

df_claims = load_mock_claims_data()

# MODEL TRAINING
@st.cache_resource
def train_model(data):
    X = data[["Billed_Amount", "Days_Since_Last_Visit", "Modifier_25_Used", "Prior_Denials"]]
    y = data["Is_Anomalous"]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X_scaled, y)
    
    return clf, scaler

model, scaler = train_model(df_claims)

# SIDEBAR INTERFACE
st.sidebar.header("Ingestion and Control Panel")
selected_claim_id = st.sidebar.selectbox("Select Claim ID to Audit:", df_claims["Claim_ID"])

st.sidebar.markdown("---")
st.sidebar.subheader("Adjust Decision Parameters")
risk_threshold = st.sidebar.slider("Anomaly Risk Flag Threshold (%)", 30, 90, 60)

# Fetch Selected Record
claim_row = df_claims[df_claims["Claim_ID"] == selected_claim_id].iloc[0]

# Prepare Data for Inference
sample_features = np.array([[
    claim_row["Billed_Amount"],
    claim_row["Days_Since_Last_Visit"],
    claim_row["Modifier_25_Used"],
    claim_row["Prior_Denials"]
]])
sample_scaled = scaler.transform(sample_features)

# Calculate Risk Score
risk_score = model.predict_proba(sample_scaled)[0][1] * 100

# MAIN DASHBOARD LAYOUT
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Selected Claim Details")
    st.metric(label="Claim Identification", value=claim_row["Claim_ID"])
    
    details_df = pd.DataFrame({
        "Attribute": ["CPT Code", "Billed Amount ($)", "Days Since Last Visit", "Modifier-25 Present", "Historical Denials"],
        "Value": [
            str(claim_row["CPT_Code"]),
            f"${claim_row['Billed_Amount']:.2f}",
            str(claim_row["Days_Since_Last_Visit"]),
            "Yes" if claim_row["Modifier_25_Used"] == 1 else "No",
            str(claim_row["Prior_Denials"])
        ]
    })
    st.table(details_df)

with col2:
    st.subheader("Stage 1: Predictive Risk Score")
    
    st.metric(
        label="Evaluated Risk Score",
        value=f"{risk_score:.1f}%",
        delta="FLAGGED FOR HUMAN REVIEW" if risk_score >= risk_threshold else "CLEARED FOR PAYMENT",
        delta_color="inverse" if risk_score >= risk_threshold else "normal"
    )
    
    if risk_score >= risk_threshold:
        st.error(f"High Risk Anomaly Detected. Score exceeds the {risk_threshold}% threshold.")
    else:
        st.success(f"Low Risk Claim. Score is below the {risk_threshold}% threshold.")

st.markdown("---")

# STAGE 2: AGENTIC AI DECISION SUPPORT
st.subheader("Stage 2: Agentic AI Explanation and Decision Support")

feature_names = ["Billed Amount", "Recency (Days)", "Modifier 25", "Prior Denials"]
importances = model.feature_importances_

col_exp1, col_exp2 = st.columns([1, 1])

with col_exp1:
    if risk_score >= risk_threshold:
        top_feature_idx = np.argmax(importances)
        top_feature = feature_names[top_feature_idx]
        
        st.markdown("#### Audit Rationale Summary (Generated for Auditor Review)")
        
        explanation_text = f"""
        - Primary Driver: The system flagged this claim primarily due to elevated values in {top_feature} relative to established baseline distributions for {claim_row['CPT_Code']}.
        - Clinical/Billing Pattern: Billed amount of ${claim_row['Billed_Amount']:.2f} with {claim_row['Prior_Denials']} prior denials indicates potential upcoding or unbundling risk.
        - Recommended Action: Request itemized medical record notes to verify service complexity before releasing payment.
        """
        st.info(explanation_text)
        st.warning("Human-in-the-Loop Oversight: AI identifies pattern anomalies; final approval/denial remains with the medical auditor.")
    else:
        st.write("No agentic explanation required. Claim meets standard pre-payment integrity metrics.")

with col_exp2:
    st.markdown("#### Global Model Feature Importance Driver")
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance Weight": importances
    }).sort_values(by="Importance Weight", ascending=True)
    
    st.bar_chart(importance_df, x="Feature", y="Importance Weight", horizontal=True)

# BATCH DATA VISUALIZATION AND PREVIEW
st.markdown("---")
st.subheader("Batch Claims Analytics and Baseline Distribution")

col_graph1, col_graph2 = st.columns([1, 1])

with col_graph1:
    st.markdown("#### Anomaly Frequency by CPT Category")
    cpt_summary = df_claims.groupby("CPT_Code")["Is_Anomalous"].agg(
        Total_Claims="count",
        Anomalies_Flagged="sum"
    ).reset_index()
    
    st.bar_chart(cpt_summary, x="CPT_Code", y=["Total_Claims", "Anomalies_Flagged"])

with col_graph2:
    st.markdown("#### Sample Ingested Claims Records")
    st.dataframe(df_claims.head(8), use_container_width='stretch')
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from google import genai
import os

# ==========================================
# 1. Page Config & UI Setup
# ==========================================
st.set_page_config(page_title="AgriLLM Evaluation Dashboard", layout="wide", initial_sidebar_state="collapsed")
st.title("🌾 AgriLLM Performance & Action Plan")
st.markdown("Evaluating agricultural model performance, safety, and domain-specific accuracy.")

# ==========================================
# 1.5 File Upload
# ==========================================
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload evaluation CSV", type=["csv"])

# ==========================================
# 2. Data Loading & KPI Calculation
# ==========================================
@st.cache_data
def load_data(file):
    # 1. If the user explicitly uploads a file via the sidebar, prioritize that
    if file is not None:
        return pd.read_csv(file)
    
    # 2. If no file is uploaded, pull the default data directly from GitHub
    github_url = "https://raw.githubusercontent.com/BongweKE/agrillm-eval-dashboard/main/agrillm_gemini_evaluation.csv"
    
    try:
        # Pandas can read CSVs directly from standard http/https links
        return pd.read_csv(github_url)
    except Exception as e:
        # 3. Final local fallback (useful for when you are testing locally without internet)
        if os.path.exists('agrillm_gemini_evaluation.csv'):
            return pd.read_csv('agrillm_gemini_evaluation.csv')
        else:
            st.error(f"Could not load data from GitHub. Error: {e}")
            return pd.DataFrame()

df = load_data(uploaded_file)

if df.empty:
    st.warning("No data available. Please upload a CSV file or ensure `agrillm_gemini_evaluation.csv` is present.")
    st.stop()

# Support potential naming variations based on how GEval saved it
def get_col(df, possible_names):
    for name in possible_names:
        if name in df.columns:
            return name
        # Also check without spaces
        if name.replace(" ", "") in df.columns:
            return name.replace(" ", "")
    return None

# Mapping expected columns
col_bias = get_col(df, ['Bias_score'])
col_tox = get_col(df, ['Toxicity_score'])
col_rel = get_col(df, ['AnswerRelevancy_score'])
col_loc = get_col(df, ['ContextualLocalization[GEval]_score', 'ContextualLocalization_score'])
col_fac = get_col(df, ['AgroforestryFactualAccuracy[GEval]_score', 'AgroforestryFactualAccuracy_score'])
col_sov = get_col(df, ['DataSovereigntyandEthics[GEval]_score', 'DataSovereigntyandEthics_score'])

metrics = {
    'Bias': df[col_bias].mean() if col_bias else 0.0,
    'Toxicity': df[col_tox].mean() if col_tox else 0.0,
    'Relevancy': df[col_rel].mean() if col_rel else 0.0,
    'Localization': df[col_loc].mean(skipna=True) if col_loc else 0.0,
    'Factual Accuracy': df[col_fac].mean(skipna=True) if col_fac else 0.0,
    'Data Sovereignty': df[col_sov].mean(skipna=True) if col_sov else 0.0
}

# Fill NaN with 0 for display
metrics = {k: (v if pd.notna(v) else 0.0) for k, v in metrics.items()}

# ==========================================
# 3. Top-Level KPI Metric Cards (Always Visible)
# ==========================================
cols = st.columns(6)
cols[0].metric("Bias", f"{metrics['Bias']:.2f}", delta="Pass" if metrics['Bias'] == 0 else "Fail", delta_color="normal" if metrics['Bias'] == 0 else "inverse")
cols[1].metric("Toxicity", f"{metrics['Toxicity']:.2f}", delta="Pass" if metrics['Toxicity'] == 0 else "Fail", delta_color="normal" if metrics['Toxicity'] == 0 else "inverse")
cols[2].metric("Relevancy", f"{metrics['Relevancy']*100:.0f}%", delta="Excellent" if metrics['Relevancy'] > 0.8 else "Needs Work", delta_color="normal")
cols[3].metric("Localization", f"{metrics['Localization']*100:.0f}%", delta="Excellent" if metrics['Localization'] > 0.8 else "Needs Work", delta_color="normal")
cols[4].metric("Factual Accuracy", f"{metrics['Factual Accuracy']*100:.0f}%", delta="Target: 90%", delta_color="inverse" if metrics['Factual Accuracy'] < 0.9 else "normal")
cols[5].metric("Data Sovereignty", f"{metrics['Data Sovereignty']*100:.0f}%", delta="Target: 90%", delta_color="inverse" if metrics['Data Sovereignty'] < 0.9 else "normal")

st.divider()

# ==========================================
# 4. Tabbed Interface Architecture
# ==========================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Executive Overview", 
    "🛡️ Safety & Relevancy", 
    "🌱 Domain Specifics", 
    "🛠️ Rec: Fix Factual Accuracy", 
    "⚖️ Rec: Fix Data Sovereignty"
])

# --- TAB 1: EXECUTIVE OVERVIEW ---
with tab1:
    st.subheader("Overall Model Capability Footprint")
    categories = list(metrics.keys())
    values = [
        1.0 - metrics['Bias'], 
        1.0 - metrics['Toxicity'], 
        metrics['Relevancy'], 
        metrics['Localization'], 
        metrics['Factual Accuracy'], 
        metrics['Data Sovereignty']
    ]
    
    fig = go.Figure(data=go.Scatterpolar(
        r=values + [values[0]], # Close the loop
        theta=categories + [categories[0]],
        fill='toself',
        line_color='#2E7D32'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        height=500,
        margin=dict(t=20, b=20, l=20, r=20)
    )
    st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: SAFETY & RELEVANCY ---
with tab2:
    st.subheader("Baseline Safety Checks")
    # Clean dataframe view hiding raw outputs, focusing on reasons
    cols_to_show = ['category', 'input']
    for c in ['Bias_reason', 'Toxicity_reason', 'AnswerRelevancy_reason']:
        if get_col(df, [c]):
            cols_to_show.append(get_col(df, [c]))
            
    safe_df = df[cols_to_show] if len(cols_to_show) > 2 else df
    st.dataframe(safe_df, use_container_width=True, hide_index=True)

# --- TAB 3: DOMAIN SPECIFICS (ERRORS) ---
with tab3:
    st.subheader("Agricultural & Ethical Hallucinations")
    
    col1, col2 = st.columns(2)
    with col1:
        if col_fac:
            fac_reason = get_col(df, ['AgroforestryFactualAccuracy[GEval]_reason', 'AgroforestryFactualAccuracy_reason'])
            df_acc = df.dropna(subset=[col_fac])
            if not df_acc.empty:
                fig_acc = px.scatter(
                    df_acc,
                    x='category', y=col_fac,
                    color=col_fac,
                    color_continuous_scale='RdYlGn',
                    hover_data=[fac_reason] if fac_reason else [],
                    title="Factual Accuracy by Category"
                )
                fig_acc.update_yaxes(range=[-0.1, 1.1])
                st.plotly_chart(fig_acc, use_container_width=True)
            else:
                st.info("No Factual Accuracy data available.")
        
    with col2:
        if col_sov:
            sov_reason = get_col(df, ['DataSovereigntyandEthics[GEval]_reason', 'DataSovereigntyandEthics_reason'])
            df_sov = df.dropna(subset=[col_sov])
            if not df_sov.empty:
                fig_sov = px.scatter(
                    df_sov,
                    x='category', y=col_sov,
                    color=col_sov,
                    color_continuous_scale='RdYlGn',
                    hover_data=[sov_reason] if sov_reason else [],
                    title="Data Sovereignty by Category"
                )
                fig_sov.update_yaxes(range=[-0.1, 1.1])
                st.plotly_chart(fig_sov, use_container_width=True)
            else:
                st.info("No Data Sovereignty data available.")

# ==========================================
# GEMINI RECOMMENDATION ENGINE
# ==========================================
@st.cache_data(show_spinner=False)
def generate_recommendation(failures_text, topic):
    prompt = f"Based on the following evaluation failures regarding '{topic}', provide a specific strategic recommendation and architecture change to improve the model. The failures are:\n{failures_text}\n\nFormat as markdown."
    try:
        client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
        response = client.models.generate_content(
            model='gemini-2.5-pro',
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Error generating recommendation: {e}\n\nPlease check your API key or network connection."

# --- TAB 4: RECOMMENDATION - FACTUAL ACCURACY ---
with tab4:
    st.subheader(f"🛠️ Action Plan: Improving Factual Accuracy (Current: {metrics['Factual Accuracy']*100:.0f}%)")
    
    if col_fac:
        fac_reason = get_col(df, ['AgroforestryFactualAccuracy[GEval]_reason', 'AgroforestryFactualAccuracy_reason'])
        failed_fac = df[df[col_fac] < 1.0]
        
        if not failed_fac.empty and fac_reason:
            failures_list = failed_fac[['input', fac_reason]].to_dict('records')
            failures_text = "\n".join([f"Input: {f['input']}\nReason: {f[fac_reason]}" for f in failures_list])
            
            with st.spinner("Generating AI-driven action plan using Gemini..."):
                rec = generate_recommendation(failures_text, "Agroforestry Factual Accuracy")
            st.markdown(rec)
            
            with st.expander("View the Specific Factual Failures in the Dataset"):
                st.dataframe(failed_fac[['input', fac_reason]], hide_index=True)
        else:
            st.success("No factual accuracy failures detected! Great job.")
    else:
        st.info("Factual Accuracy metric not found in the dataset.")

# --- TAB 5: RECOMMENDATION - DATA SOVEREIGNTY ---
with tab5:
    st.subheader(f"⚖️ Action Plan: Enforcing Data Sovereignty (Current: {metrics['Data Sovereignty']*100:.0f}%)")
    
    if col_sov:
        sov_reason = get_col(df, ['DataSovereigntyandEthics[GEval]_reason', 'DataSovereigntyandEthics_reason'])
        failed_sov = df[df[col_sov] < 1.0]
        
        if not failed_sov.empty and sov_reason:
            failures_list = failed_sov[['input', sov_reason]].to_dict('records')
            failures_text = "\n".join([f"Input: {f['input']}\nReason: {f[sov_reason]}" for f in failures_list])
            
            with st.spinner("Generating AI-driven action plan using Gemini..."):
                rec = generate_recommendation(failures_text, "Data Sovereignty and Ethics")
            st.markdown(rec)
            
            with st.expander("View the Specific Sovereignty Failures in the Dataset"):
                st.dataframe(failed_sov[['input', sov_reason]], hide_index=True)
        else:
            st.success("No data sovereignty failures detected! Great job.")
    else:
        st.info("Data Sovereignty metric not found in the dataset.")

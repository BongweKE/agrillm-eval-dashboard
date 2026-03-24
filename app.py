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
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Executive Overview", 
    "🛡️ Safety & Relevancy", 
    "🌱 Domain Specifics", 
    "💡 Automated Recommendations"
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
    st.plotly_chart(fig, width='stretch')

# --- TAB 2: SAFETY & RELEVANCY ---
with tab2:
    st.subheader("Baseline Safety Checks")
    # Clean dataframe view hiding raw outputs, focusing on reasons
    cols_to_show = ['category', 'input']
    for c in ['Bias_reason', 'Toxicity_reason', 'AnswerRelevancy_reason']:
        if get_col(df, [c]):
            cols_to_show.append(get_col(df, [c]))
            
    safe_df = df[cols_to_show] if len(cols_to_show) > 2 else df
    st.dataframe(safe_df, width='stretch', hide_index=True)

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
                st.plotly_chart(fig_acc, width='stretch')
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
                st.plotly_chart(fig_sov, width='stretch')
            else:
                st.info("No Data Sovereignty data available.")

# ==========================================
# GEMINI RECOMMENDATION ENGINE
# ==========================================
@st.cache_data(show_spinner=False)
def generate_recommendation(failures_text, topic):
    model_url = "https://huggingface.co/AI71ai/Llama-agrillm-3.3-70B"
    prompt = f"""You are an AI architect tasked with improving the performance of a specialized agricultural model.
The model being evaluated is: {model_url} (A LLaMA 3.3 70B agricultural fine-tune).

Based on the model's architecture and the following evaluation failures regarding '{topic}', provide specific, highly-tailored strategic recommendations to improve it.

The failures are:
{failures_text}

Format your response as markdown, and ensure your recommendations are practically applicable to this specific open-source deployment."""
    try:
        client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
        response = client.models.generate_content(
            model='gemini-2.5-pro',
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Error generating recommendation: {e}\n\nPlease check your API key or network connection."

# --- TAB 4: AUTOMATED RECOMMENDATIONS ---
with tab4:
    st.subheader("💡 Automated AI Action Plans")
    st.markdown("Dynamically generated strategic recommendations for any evaluation metrics scoring below 100%.")
    
    # 1. Define the metrics we want to monitor dynamically
    monitored_metrics = [
        {
            "name": "Agroforestry Factual Accuracy", 
            "score_col": col_fac, 
            "reason_col": get_col(df, ['AgroforestryFactualAccuracy[GEval]_reason', 'AgroforestryFactualAccuracy_reason'])
        },
        {
            "name": "Data Sovereignty and Ethics", 
            "score_col": col_sov, 
            "reason_col": get_col(df, ['DataSovereigntyandEthics[GEval]_reason', 'DataSovereigntyandEthics_reason'])
        },
        {
            "name": "Contextual Localization", 
            "score_col": col_loc, 
            "reason_col": get_col(df, ['ContextualLocalization[GEval]_reason', 'ContextualLocalization_reason'])
        },
        {
            "name": "Answer Relevancy", 
            "score_col": col_rel, 
            "reason_col": get_col(df, ['AnswerRelevancy_reason'])
        }
    ]

    issues_found = False

    # 2. Loop through each metric to check for failures
    for metric in monitored_metrics:
        score_c = metric["score_col"]
        reason_c = metric["reason_col"]
        m_name = metric["name"]

        # Only proceed if the metric actually exists in the uploaded CSV
        if score_c and reason_c:
            # Filter for any row where the model didn't get a perfect score
            failed_df = df[df[score_c] < 1.0]
            
            if not failed_df.empty:
                issues_found = True
                
                # Create a visually distinct section for this failing metric
                st.markdown(f"### ⚠️ Area for Improvement: {m_name}")
                st.info(f"Detected {len(failed_df)} sub-par responses requiring attention. Gemini 2.5 Pro is analyzing the failures and generating recommendations...")
                
                # Compile the failures into a text string for Gemini
                failures_list = failed_df[['input', reason_c]].to_dict('records')
                failures_text = "\n".join([f"- Prompt: {f['input']}\n  Judge Critique: {f[reason_c]}\n" for f in failures_list])
                
                # Fetch the recommendation
                with st.spinner(f"Analyzing {m_name} failures with Gemini..."):
                    rec = generate_recommendation(failures_text, m_name)
                
                # Display the recommendation
                st.markdown(rec)
                
                # Provide the raw data dropdown
                with st.expander(f"🔍 View the {len(failed_df)} specific failures for {m_name}"):
                    st.dataframe(failed_df[['input', reason_c]], hide_index=True)
                
                st.divider() # Adds a neat horizontal line between different metric recommendations

    # 3. Handle the perfect scenario
    if not issues_found:
        st.success("🎉 All monitored metrics passed perfectly! No structural recommendations needed at this time.")
        st.balloons()

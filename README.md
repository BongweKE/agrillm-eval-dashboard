# 🌾 AgriLLM Evaluation & Action Plan Dashboard

This repository contains a full end-to-end pipeline for evaluating custom Agricultural Large Language Models (LLMs) and a Streamlit-powered dashboard that visually analyzes the results and uses Gemini to generate actionable architecture/model recommendations.

The evaluation process uses **DeepEval** with **Gemini 2.5 Pro** as an LLM-as-a-judge to evaluate:
- **Global Baselines**: Toxicity, Bias, Answer Relevancy.
- **Domain-Specific GEval Metrics**: Agroforestry Factual Accuracy, Data Sovereignty & Ethics, Contextual Localization.

---

## ⚙️ Prerequisites & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_GITHUB_USERNAME/agrillm-eval-dashboard.git
   cd agrillm-eval-dashboard
   ```

2. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Secure Environment Variables (Crucial)**
   The pipeline relies on cloud API keys. **Never commit your API keys to the repository.** The `.gitignore` is already set up to protect your local credential files.
   
   - **For Python Evaluation (`evaluate_model.py`):**
     Create a new file named `.env` in the root folder and add your keys:
     ```env
     GEMINI_API_KEY="your_gemini_api_key_here"
     HF_ENDPOINT_KEY="your_huggingface_endpoint_key_here"
     ```
     
   - **For Local Streamlit Dashboard (`app.py`):**
     Create a hidden folder `.streamlit` and a file inside it named `secrets.toml`:
     ```toml
     GEMINI_API_KEY = "your_gemini_api_key_here"
     ```

---

## 🏃‍♂️ Step 1: Run the Evaluation Locally

Once your environment variables are configured, test your LLM outputs and grade them against the rigorous GEval matrix.

```bash
python evaluate_model.py
```

This will systematically route your domain questions defined in `eval_dataset.json`, run inference against your target LLM, and query Gemini 2.5 Pro as the judge. 

**Outputs Generated:**
- `agrillm_raw_outputs.csv` (The raw LLM responses)
- `agrillm_gemini_evaluation.csv` (The final strict evaluations and pass rates)

---

## 📊 Step 2: View the Dashboard Locally

You can dynamically render and review the model's action plan via Streamlit.

```bash
streamlit run app.py
```

- The app will automatically boot to `http://localhost:8501`.
- It will automatically fallback to load `agrillm_gemini_evaluation.csv` local data.
- **Live Recommender**: Tabs 4 and 5 utilize your `st.secrets` to dynamically send the failing subsets of your data to Gemini and provide architectural fixes in real-time.

*(Need to review a different evaluation run? Use the file uploader exposed in the left sidebar!)*

---

## 🚀 Step 3: Deploy Publicly (Streamlit Community Cloud)

You can instantly host this dashboard publicly and for free utilizing Streamlit Community Cloud. By design, the deployed app will seamlessly pull its default CSV data using your repository's raw GitHub URL.

1. Commit all your changes and push to your public GitHub repository:
   ```bash
   git push origin main
   ```
2. Navigate to [share.streamlit.io](https://share.streamlit.io/) and click **New App**.
3. Select your GitHub repository and point the path to `app.py`.
4. **⚠️ CRITICAL - Apply API Secrets:**
   Before clicking deploy, click on **Advanced Settings**. Under the **Secrets** section, securely paste your Gemini key just like your local TOML file:
   ```toml
   GEMINI_API_KEY = "your_gemini_api_key_here"
   ```
5. **⚠️ CRITICAL - Python Environment:**
   In the same **Advanced Settings** menu, ensure that your **Python Version** dropdown is set down to **`3.12`**. Newer Python 3.14 builds contain standard library breakages that conflict with `altair` typing standards.
6. Click **Deploy!** 

Your dashboard is now live and synchronized to automatically pull your committed evaluation updates directly from GitHub!

import streamlit as st
import fitz  # PyMuPDF
from transformers import pipeline
# Updated prompt to include pros and cons analysis
def create_prompt(pdf_text):
    return f"""
    You are a medical lab report expert. Analyze the following diagnostic report text and extract:

    1. A summary table with: Test Name, Patient Result, Normal Range, and whether it's Low / Normal / High.
    2. A brief interpretation for any abnormal results.
    3. The pros and cons of the patient's results, considering the findings in the report.

    --- Begin Report ---

    {pdf_text}

    --- End Report ---

    Respond only with your analysis.
    """
# import os
# import ssl
# import urllib3

# ssl._create_default_https_context = ssl._create_unverified_context
# urllib3.disable_warnings()

# os.environ['CURL_CA_BUNDLE'] = ''

# Load the LLM (small, fast, CPU-friendly)
@st.cache_resource
def load_llm():
    return pipeline("text2text-generation", model="google/flan-t5-base", max_length=512)

llm = load_llm()

# Extract all text from PDF
def extract_pdf_text(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Prompt to guide the LLM
# def create_prompt(pdf_text):
#     return f"""
#     You are a medical lab report expert. Analyze the following diagnostic report text and extract:

#     1. A summary table with: Test Name, Patient Result, Normal Range, and whether it's Low / Normal / High.
#     2. A brief interpretation for any abnormal results.

#     --- Begin Report ---

#     {pdf_text}

#     --- End Report ---

#     Respond only with your analysis.
#     """

# Streamlit UI
st.set_page_config(page_title="LLM Diagnostic Analyzer")
st.title("🧠📄 LLM-Powered Medical Report Analyzer")

uploaded_pdf = st.file_uploader("Upload Diagnostic PDF", type=["pdf"])

if uploaded_pdf:
    with st.spinner("📄 Reading PDF..."):
        report_text = extract_pdf_text(uploaded_pdf)

    with st.spinner("🧠 Analyzing with LLM..."):
        prompt = create_prompt(report_text)
        # Run the LLM on the extracted text and display results
        result = llm(prompt)
        # Extract generated text from the LLM result
        output = result[0]["generated_text"] if result and isinstance(result, list) and "generated_text" in result[0] else "No analysis generated."

    st.subheader("✅ Analysis Report")
    st.markdown(output)

    # with st.expander("📃 Raw Extracted Text"):
    #     st.text(report_text)
else:
    st.info("Upload a diagnostic report PDF to begin.")
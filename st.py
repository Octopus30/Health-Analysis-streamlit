import streamlit as st
from dotenv import load_dotenv
import os
import boto3
import json

# Load environment variables from .env
load_dotenv()

# Access credentials from environment
aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
region = os.getenv("AWS_DEFAULT_REGION")

# Connect to AWS service (e.g., S3)
s3 = boto3.client(
    's3',
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key,
    region_name=region
)



from amazon import (log_with_timestamp,
                    extract_text_from_response,
                    wait_for_job_completion,
                    chunk_text,
                    create_csv_from_results,
                    process_with_bedrock_Analysis)


# # Streamlit UI
st.set_page_config(page_title="LLM Diagnostic Analyzer")
st.title("🧠📄 LLM-Powered Medical Report Analyzer")

uploaded_file = st.file_uploader("Upload Diagnostic PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("📄 Reading PDF..."):
        # extracting text from the uploaded PDF using textract
        s3 = boto3.client('s3')
        textract = boto3.client('textract', region_name='us-east-1')

        file_name, file_extension = os.path.splitext(os.path.basename(uploaded_file.name))
        print(f"File Name: {file_name}, File Extension: {file_extension}")

        raw_text = ""  # Initialize raw_text to avoid UnboundLocalError
        try:
            if file_extension.lower() in ['.png', '.jpg', '.jpeg']:               
                file_content = uploaded_file.read()
                textract_response = textract.detect_document_text(Document={'Bytes': file_content})
                raw_text = extract_text_from_response(textract_response)
            elif file_extension.lower() == '.pdf':
                # Reset file pointer and upload to S3
                uploaded_file.seek(0)
                s3_bucket = "processeddocsb2"  # <-- Replace with your S3 bucket name
                s3_key = file_name + file_extension
                s3.upload_fileobj(uploaded_file, s3_bucket, s3_key)

                textract_response = textract.start_document_text_detection(
                    DocumentLocation={
                        'S3Object': {
                            'Bucket': s3_bucket,
                            'Name': s3_key
                        }
                    }
                )
                job_id = textract_response['JobId']
                raw_text = wait_for_job_completion(textract, job_id)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            raw_text = ""

    if raw_text:
        text_content = ''.join(raw_text)  # Chunk text for processing
        with st.spinner("🧠 Analyzing with LLM..."):
            bedrock_results = process_with_bedrock_Analysis(text_content)
        st.subheader("✅ Analysis Report")
        # Only display the parsed/decoded result, not the full response object

    # ...existing code...
    if isinstance(bedrock_results, dict):
        # Try to get 'text' from the first result
        text_content = (
            bedrock_results.get("results", [{}])[0].get("text")
            or bedrock_results.get("results", [{}])[0].get("content")[0].get("text")
            or "No content found."
        )
        st.markdown(text_content)
    elif isinstance(bedrock_results, list) and len(bedrock_results) > 0:
        # If it's a list, try to get 'text' or 'content' from the first item
        text_content = (
            bedrock_results[0].get("text")
            or bedrock_results[0].get("content")[0].get("text")
            or "No content found."
        )
        st.markdown(text_content)
    else:
        st.markdown(str(bedrock_results))
else:
    st.info("Upload a diagnostic report PDF to begin.")
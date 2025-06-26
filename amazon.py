import boto3
import botocore.config
import os
import time
import json
import csv
from datetime import datetime
from io import StringIO

def log_with_timestamp(message, is_start=False, is_end=False):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prefix = "🟢 STARTING" if is_start else "🔴 COMPLETED" if is_end else "ℹ"
    print(f"[{timestamp}] {prefix} {message}")

def extract_text_from_response(response):
    log_with_timestamp("Extracting text from Textract response", is_start=True)
    text_lines = [item['Text'] for item in response['Blocks'] if item['BlockType'] == 'LINE']
    log_with_timestamp(f"Extracted {len(text_lines)} lines of text", is_end=True)
    return text_lines

def wait_for_job_completion(textract, job_id):
    log_with_timestamp(f"Waiting for Textract job {job_id} to complete", is_start=True)
    while True:
        response = textract.get_document_text_detection(JobId=job_id)
        status = response['JobStatus']
        log_with_timestamp(f"Current job status: {status}")
        if status in ['SUCCEEDED', 'FAILED']:
            break
        time.sleep(5)
    
    if status == 'SUCCEEDED':
        log_with_timestamp("Collecting results from successful Textract job", is_start=True)
        raw_text = []
        for item in response['Blocks']:
            if item['BlockType'] == 'LINE':
                raw_text.append(item['Text'])
        
        while 'NextToken' in response:
            log_with_timestamp("Fetching next page of Textract results")
            response = textract.get_document_text_detection(JobId=job_id, NextToken=response['NextToken'])
            for item in response['Blocks']:
                if item['BlockType'] == 'LINE':
                    raw_text.append(item['Text'])
        
        log_with_timestamp(f"Collected {len(raw_text)} lines of text from Textract", is_end=True)
        return raw_text
    else:
        error_msg = f"Textract job failed: {response.get('StatusMessage', 'No error message provided')}"
        log_with_timestamp(f"ERROR: {error_msg}")
        raise Exception(error_msg)

def chunk_text(text, max_chunk_size=6000):
    """Split text into chunks while preserving context"""
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0
    
    for word in words:
        if current_size + len(word) + 1 > max_chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_size = len(word)
        else:
            current_chunk.append(word)
            current_size += len(word) + 1
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def process_with_bedrock_scraping(text_content):
    log_with_timestamp("Preparing Bedrock API call", is_start=True)
    
    config = botocore.config.Config(
        read_timeout=300,
        connect_timeout=300,
        retries={'max_attempts': 3}
    )
    
    bedrock_runtime = boto3.client(
        service_name='bedrock-runtime',
        region_name='us-east-1',
        config=config
    )
    
    chunks = chunk_text(text_content)
    all_results = []
    
    for i, chunk in enumerate(chunks):
        log_with_timestamp(f"Processing chunk {i+1} of {len(chunks)}")
        
        prompt = """Analyze this medical report and provide the results in JSON format. Extract all test results and patient information.

Required format:
{   
    
    "test_groups": [
        {
            "group_name": "Test Group Name",
            "name" : "Patient Name",
            "date":"Date of Test",
            "age":"Patient Age",
            "tests": [
                {
                    "test_name": "Test Name",
                    "result": "Result Value",
                    "reference_range": "Reference Range",
                    "unit": "Unit of Measurement"
                }
            ]
            
        }
    ]
}

Important:
1. Include all test results found in the report
2. Keep original values exactly as shown
3. Group related tests together
4. Include reference ranges and units when available
5. Maintain the exact format specified above
6. Date should be the day the sample is collected.
7. when retriving age get only age,do not add any unnecessary text.

Parse this portion of the medical report:"""

        try:
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 8000,
                "temperature": 0.1,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"{prompt}\n\n{chunk}"
                            }
                        ]
                    }
                ]
            }
            
            response = bedrock_runtime.invoke_model(
                modelId='anthropic.claude-3-sonnet-20240229-v1:0',
                contentType='application/json',
                accept='application/json',
                body=json.dumps(request_body)
            )
            
            result = json.loads(response['body'].read())
            all_results.append(result)
            
        except Exception as e:
            log_with_timestamp(f"Error processing chunk {i+1}: {str(e)}")
            continue
    
    log_with_timestamp("Bedrock processing completed", is_end=True)
    return all_results


def create_csv_from_results(all_results):
    log_with_timestamp("Creating CSV from results", is_start=True)
    output = StringIO()
    writer = csv.writer(output)

    # Write the header row for the CSV
    writer.writerow(["Test_Group","Patient_Name","age","Date_of_test","Test_Name", "Result", "Reference_Range", "Unit" ])

    for result in all_results:
        try:
            # Validate and parse the response
            response_text = result.get('content', [{}])[0].get('text', '')
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1

            if json_start >= 0 and json_end > json_start:
                # Parse JSON data from the response
                parsed_data = json.loads(response_text[json_start:json_end])

                for group in parsed_data.get('test_groups', []):
                    group_name = group.get('group_name', '')

                    patient_name = group.get('name', '')
                    test_date = group.get('date', '')
                    age = group.get('age', '')

                    

                    for test in group.get('tests', []):
                        # Write each test result as a row in the CSV
                        writer.writerow([
                            group_name,
                            patient_name,
                            age,
                            test_date,
                            test.get('test_name', ''),
                            test.get('result', ''),
                            test.get('reference_range', ''),
                            test.get('unit', '')
                            
                        ])

            log_with_timestamp(f"Successfully processed result of {patient_name}.", is_start=True)
        except Exception as e:
            # Log errors with detailed information
            log_with_timestamp(f"Error processing result: {str(e)}")
            continue

    log_with_timestamp("CSV creation completed", is_end=True)
    return output.getvalue(),patient_name,test_date


# def lambda_handler(event, context):
#     log_with_timestamp("Lambda function started", is_start=True)
#     s3 = boto3.client('s3')
#     textract = boto3.client('textract', region_name='us-east-1')

#     input_bucket = 'inputdocb2'
#     output_bucket = 'outputdocb2'
#     processed_bucket = 'processeddocsb2'
    
#     try:
#         # Get the input file details
#         file_key = event['Records'][0]['s3']['object']['key']
#         file_name, file_extension = os.path.splitext(os.path.basename(file_key))
#         modified_file_name = file_name.replace(' ', '_')

#         log_with_timestamp(f"extracted file name {file_name},{file_key}", is_start=True)
        
#         # Process with Textract
#         log_with_timestamp("Starting Textract processing", is_start=True)
#         if file_extension.lower() in ['.png', '.jpg', '.jpeg']:
#             response = s3.get_object(Bucket=input_bucket, Key=file_key)
#             file_content = response['Body'].read()
#             textract_response = textract.detect_document_text(Document={'Bytes': file_content})
#             raw_text = extract_text_from_response(textract_response)
#         elif file_extension.lower() == '.pdf':
#             textract_response = textract.start_document_text_detection(
#                 DocumentLocation={'S3Object': {'Bucket': input_bucket, 'Name': file_key}}
#             )
#             job_id = textract_response['JobId']
#             raw_text = wait_for_job_completion(textract, job_id)
#         else:
#             raise ValueError(f"Unsupported file type: {file_extension}")

#         # Save extracted text
#         text_content = '\n'.join(raw_text)
#         output_text_key = f"{modified_file_name}_textract.txt"
        
#         s3.put_object(
#             Bucket=output_bucket,
#             Key=output_text_key,
#             Body=text_content.encode('utf-8')
#         )

#         # Process with Bedrock
#         bedrock_results = process_with_bedrock(text_content)
        
#         # Create and save CSV
#         csv_content,person_name,test_date = create_csv_from_results(bedrock_results)

#         test_date  = test_date.replace('/','')
        
#         csv_filename = f"{person_name}{datetime.now().strftime("%d%m%Y")}{modified_file_name}_results.csv"
        
#         s3.put_object(
#             Bucket=processed_bucket,
#             Key=csv_filename,
#             Body=csv_content.encode('utf-8'),
#             ContentType='text/csv'
#         )

#         # Save raw Bedrock response for debugging
#         debug_filename = f"{person_name}{datetime.now().strftime("%d%m%Y")}{modified_file_name}_bedrock_response.json"
#         s3.put_object(
#             Bucket=processed_bucket,
#             Key=debug_filename,
#             Body=json.dumps(bedrock_results, indent=2).encode('utf-8'),
#             ContentType='application/json'
#         )

#         return {
#             'statusCode': 200,
#             'body': json.dumps({
#                 'message': 'Processing completed successfully',
#                 'files': {
#                     'original': f"s3://{input_bucket}/{file_key}",
#                     'textract': f"s3://{output_bucket}/{output_text_key}",
#                     'results': f"s3://{processed_bucket}/{csv_filename}",
#                     'debug': f"s3://{processed_bucket}/{debug_filename}"
#                 }
#             })
#         }

#     except Exception as e:
#         error_message = f"Error processing file: {str(e)}"
#         log_with_timestamp(f"ERROR: {error_message}")
#         import traceback
#         log_with_timestamp(f"Traceback: {traceback.format_exc()}")
        
#         return {
#             'statusCode': 500,
#             'body': json.dumps({
#                 'error': error_message,
#                 'traceback': traceback.format_exc()
#     })
# }
    
def process_with_bedrock_Analysis(text_content):
    log_with_timestamp("Preparing Bedrock API call", is_start=True)
    
    config = botocore.config.Config(
        read_timeout=300,
        connect_timeout=300,
        retries={'max_attempts': 3}
    )
    
    bedrock_runtime = boto3.client(
        service_name='bedrock-runtime',
        region_name='us-east-1',
        config=config
    )
    
    chunks = chunk_text(text_content)
    all_results = []
    
    for i, chunk in enumerate(chunks):
        log_with_timestamp(f"Processing chunk {i+1} of {len(chunks)}")
        
        prompt = """You are a medical assistant specialized in analyzing diagnostic health reports. I will give you the extracted text from a diagnostic report. 

            Your task is to:
            1. Read and understand the results from tests such as blood work, imaging, and other diagnostics.
            2. Summarize the findings in simple, non-technical language.
            3. Identify and list:
            - ✅ Pros: parameters that are within normal range or showing improvement.
            - ❌ Cons: parameters that are outside the normal range or indicating a potential health concern.
            4. Give suggestions for lifestyle improvements, further tests, or follow-ups if necessary — but DO NOT give any diagnosis.

            Format your answer like this:

            📋 Summary:
            - [Brief, simple explanation of the overall health based on report]

            ✅ Pros:
            - [Positive finding 1]
            - [Positive finding 2]

            ❌ Cons:
            - [Concern 1 with a short explanation]
            - [Concern 2 with a short explanation]

            📌 Suggestions:
            - [Advice or follow-up if applicable]
        """

        try:
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 8000,
                "temperature": 0.1,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"{prompt}\n\n{chunk}"
                            }
                        ]
                    }
                ]
            }
            
            response = bedrock_runtime.invoke_model(
                modelId='anthropic.claude-3-sonnet-20240229-v1:0',
                contentType='application/json',
                accept='application/json',
                body=json.dumps(request_body)
            )
            print(response)
            result = json.loads(response['body'].read())
            all_results.append(result)
            
        except Exception as e:
            log_with_timestamp(f"Error processing chunk {i+1}: {str(e)}")
            continue
    
    log_with_timestamp("Bedrock processing completed")
    return all_results
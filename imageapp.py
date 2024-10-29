# Author: Gal Vekselman
# streamlit run imageapp.py


import os
import streamlit as st
import boto3
import json
import base64
from io import BytesIO
from PIL import Image
import PyPDF2
from docx import Document
import requests
from dotenv import load_dotenv

load_dotenv()

# Securely retrieve AWS credentials
AWS_ACCESS_KEY_ID =  os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
AWS_REGION_NAME =  os.environ.get('AWS_REGION_NAME')

# Initialize Bedrock client
bedrock = boto3.client(
    'bedrock-runtime',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION_NAME
)

model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"
accept = 'application/json'
contentType = 'application/json'


def load_image(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        return img
    except requests.exceptions.RequestException as e:
        st.error(f"Error loading image: {e}")
        return None


def extract_text_from_docx(file_path):
    doc = Document(file_path)
    full_text = [para.text for para in doc.paragraphs]
    return '\n'.join(full_text)


def get_examples(folder_path):
    concatenated_output = ""
    for file_name in os.listdir(folder_path):
        if file_name.endswith(('.png', '.jpg')):
            image_file_path = os.path.join(folder_path, file_name)
            json_file_path = os.path.join(folder_path, file_name.rsplit('.', 1)[0] + '.json')

            with open(image_file_path, 'rb') as image_file:
                image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

            if os.path.exists(json_file_path):
                try:
                    with open(json_file_path, 'r', encoding='utf-8') as json_file:
                        json_content = json.load(json_file)
                    concatenated_output += f"Example Input (Base64) for {file_name}:\n{image_base64}\n\n"
                    concatenated_output += f"Output (JSON content) for {file_name}:\n{json.dumps(json_content, indent=4)}\n\n"
                except (UnicodeDecodeError, json.JSONDecodeError) as e:
                    concatenated_output += f"Error reading JSON file {file_name}: {e}\n\n"
            else:
                concatenated_output += f"No corresponding JSON file found for {file_name}\n\n"
    return concatenated_output


def analyze_image(image_data):
    encoded_image = base64.b64encode(image_data).decode()

    instructions = """
    Please provide a JSON input that includes the following details:

    1. Document Description: Describe what this document is.
    2. Image Quality Assessment: Evaluate the quality of the image within the document.
    3. Extracted Attributes: List all attributes or key pieces of information you can identify in the document.

    Return the JSON output in Hebrew.

    אם איכות התמונה לא טובה אל תחזיר JSON ותבקש לעלות את התמונה מחדש
    """
    wkdir = os.getcwd()
    example = extract_text_from_docx(f"{wkdir}\examples_from_word.docx")

    prompt = f"""
    <instructions>
    {instructions}
    </instructions>

    <example>
    {example}
    </example>
    """

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 8000,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": encoded_image}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    })

    try:
        response = bedrock.invoke_model(modelId=model_id, body=body)
        response_body = json.loads(response.get("body").read())
        return response_body['content'][0]['text']
    except Exception as e:
        st.error(f"Error invoking model: {e}")
        return None


def extract_text_from_pdf(pdf_data):
    reader = PyPDF2.PdfReader(BytesIO(pdf_data))
    text = "".join(page.extract_text() for page in reader.pages)
    return text


def main():
    img_url = "https://www.drupal.org/files/styles/grid-4-2x/public/bar-ilan-university-logo.png?itok=BtP0hVC5"
    img = load_image(img_url)
    if img:
        st.image(img, caption="בר אילן - סורק תמונות", use_column_width=True)

    st.markdown("<h1 style='text-align: center;'>סרוק את המסמך המבוקש לחילוץ מידע</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>בחר תמונה לניתוח</p>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png", "pdf"])

    if uploaded_file is not None:
        file_type = uploaded_file.type
        file_bytes = uploaded_file.read()

        if file_type in ["image/jpeg", "image/png"]:
            image = Image.open(BytesIO(file_bytes))
            st.image(image, caption='Uploaded Image', use_column_width=True)
            st.write("Analyzing the image...")
            analysis_result = analyze_image(file_bytes)
            if analysis_result:
                st.write(analysis_result)

        elif file_type == "application/pdf":
            st.write("Extracting text from the PDF...")
            pdf_text = extract_text_from_pdf(file_bytes)
            st.text_area("Extracted Text", pdf_text, height=300)


if __name__ == "__main__":
    main()

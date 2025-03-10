import os
import json
import pandas as pd
from flask import Flask, request, jsonify, render_template
from PyPDF2 import PdfReader
from together import Together
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
import io
import datetime

app = Flask(__name__)
client = Together(api_key=os.getenv("TOGETHER_API_KEY"))

PDF_FOLDER = "pdfs"


def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    text = ""
    try:
        pdf_reader = PdfReader(pdf_path)
        for page in pdf_reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"
    except Exception as e:
        print(f"⚠️ Error reading {pdf_path}: {e}")
    return text


def load_all_pdfs(folder):
    """Loads and combines text from all PDFs in the folder."""
    combined_text = ""
    for filename in os.listdir(folder):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(folder, filename)
            combined_text += extract_text_from_pdf(pdf_path) + "\n"
    return combined_text


# Pre-load PDF texts at server startup
pdf_text = load_all_pdfs(PDF_FOLDER)


def format_response(response_text):
    """Formats response with proper line breaks and bullet points where necessary."""
    response_lines = response_text.split(". ")
    formatted_response = "\n".join([f"- {line.strip()}" for line in response_lines if line])
    return formatted_response


def answer_question(pdf_text, question):
    """Generates an AI response based on PDF content and user query."""
    if not pdf_text.strip():
        return "⚠️ No text extracted from PDFs."

    prompt = f"""
    The following is your knowledge base:\n\n{pdf_text[:4000]}\n\n
    Answer the following question in a friendly, concise, and structured manner:\n{question}
    """

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=[
            {"role": "system", "content": "You are an EZ-Invoice FAQ Bot AI assistant answering questions based on your knowledge base made by eZee Technosys (M) Sdn. Bhd.. Do NOT mention any section references. Keep responses friendly, concise, and structured."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.7,
        top_p=0.9,
    )

    response_text = response.choices[0].message.content.strip()
    return format_response(response_text)


# --- GOOGLE DRIVE INTEGRATION ---
def get_google_drive_service():
    """Authenticates and returns a Google Drive service instance."""
    google_creds_json = os.getenv("GOOGLE_CREDENTIALS")
    if not google_creds_json:
        raise ValueError("⚠️ GOOGLE_CREDENTIALS not found in environment variables!")

    google_creds_dict = json.loads(google_creds_json)
    creds = Credentials.from_service_account_info(google_creds_dict)
    return build("sheets", "v4", credentials=creds)


def append_to_google_sheet(data):
    """Appends chatbot logs to a Google Sheet instead of creating a new file every time."""
    service = get_google_drive_service()
    spreadsheet_id = os.getenv("GOOGLE_SHEET_ID")
    range_name = "ChatLogs!A:C"  

    values = [[datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), data[0][1], data[0][2]]]
    body = {"values": values}

    service.spreadsheets().values().append(
        spreadsheetId=spreadsheet_id,
        range=range_name,
        valueInputOption="RAW",
        insertDataOption="INSERT_ROWS",
        body=body
    ).execute()


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    """Handles user queries and logs them to Google Sheets."""
    data = request.json
    user_message = data.get("message", "")

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    bot_response = answer_question(pdf_text, user_message)

    # Log query & response
    chat_log = [[datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), user_message, bot_response]]
    append_to_google_sheet(chat_log)

    return jsonify({"response": bot_response})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

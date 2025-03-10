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

def answer_question(pdf_text, question):
    """Generates an AI response based on PDF content and user query."""
    if not pdf_text.strip():
        return "⚠️ No text extracted from PDFs."

    prompt = f"The following is your knowledge base:\n\n{pdf_text[:4000]}\n\nAnswer the following question in a friendly, concise, and structured manner:\n{question}"

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

    # Add line breaks for better readability
    formatted_response = response_text.replace(". ", ".\n")

    return formatted_response

# --- GOOGLE DRIVE INTEGRATION ---
def get_google_drive_service():
    """Authenticates and returns a Google Drive service instance."""
    google_creds_json = os.getenv("GOOGLE_CREDENTIALS")
    if not google_creds_json:
        raise ValueError("⚠️ GOOGLE_CREDENTIALS not found in environment variables!")

    google_creds_dict = json.loads(google_creds_json)
    creds = Credentials.from_service_account_info(google_creds_dict)
    return build("drive", "v3", credentials=creds)

def upload_to_google_drive(data):
    """Saves chatbot logs to Google Drive as an Excel file."""
    service = get_google_drive_service()
    folder_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID") 

    # Create a DataFrame
    df = pd.DataFrame(data, columns=["Timestamp", "User Query", "Bot Response"])
    
    # Convert DataFrame to Excel in-memory
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="ChatLogs")
    excel_buffer.seek(0)

    # Upload to Google Drive
    file_metadata = {
        "name": f"chatbot_logs_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.xlsx",
        "mimeType": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "parents": [folder_id]
    }

    media = MediaIoBaseUpload(excel_buffer, mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    service.files().create(body=file_metadata, media_body=media, fields="id").execute()

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    """Handles user queries and logs them to Google Drive."""
    data = request.json
    user_message = data.get("message", "")

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    bot_response = answer_question(pdf_text, user_message)

    # Log query & response
    chat_log = [[datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), user_message, bot_response]]
    upload_to_google_drive(chat_log)

    return jsonify({"response": bot_response})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

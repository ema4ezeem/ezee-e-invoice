import os
import json
import logging
import traceback
import re
import hashlib
from functools import lru_cache
from flask import Flask, request, jsonify, render_template
from PyPDF2 import PdfReader
from together import Together
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
import datetime
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("chatbot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ez-invoice-bot")

app = Flask(__name__)
client = Together(api_key=os.getenv("TOGETHER_API_KEY"))

# Initialize rate limiter
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "30 per hour"]
)

PDF_FOLDER = "pdfs"

# Track PDF content version for cache invalidation
pdf_version = "1.0"

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
        logger.error(f"Error reading {pdf_path}: {e}")
    return text

def load_all_pdfs(folder):
    """Loads text from all PDFs in the folder with metadata."""
    pdf_data = {}
    if not os.path.exists(folder):
        logger.warning(f"PDF folder {folder} does not exist.")
        return pdf_data
    
    for filename in os.listdir(folder):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(folder, filename)
            logger.info(f"Loading PDF: {filename}")
            pdf_data[filename] = extract_text_from_pdf(pdf_path)
    
    # Create a version hash based on filenames and modification times
    global pdf_version
    version_data = ""
    for filename in pdf_data.keys():
        file_path = os.path.join(folder, filename)
        mod_time = os.path.getmtime(file_path)
        version_data += f"{filename}:{mod_time};"
    
    pdf_version = hashlib.md5(version_data.encode()).hexdigest()[:8]
    logger.info(f"PDF content version: {pdf_version}")
    
    return pdf_data

def get_relevant_chunks(pdf_data, question, chunk_size=1000, overlap=200):
    """Find relevant chunks from PDF content based on keyword matching."""
    all_chunks = []
    
    # Extract potential keywords from the question
    keywords = [word.lower() for word in re.findall(r'\b\w+\b', question) 
                if len(word) > 3 and word.lower() not in ['what', 'when', 'where', 'which', 'how', 'does', 'is', 'are', 'the', 'and', 'that']]
    
    # Create chunks with metadata
    for filename, content in pdf_data.items():
        if not content:
            continue
            
        for i in range(0, len(content), chunk_size - overlap):
            chunk = content[i:i + chunk_size]
            
            # Calculate a relevance score based on keyword matches
            score = sum(1 for keyword in keywords if keyword in chunk.lower())
            
            all_chunks.append({
                "text": chunk,
                "source": filename,
                "score": score
            })
    
    # Sort by relevance score and take top chunks
    relevant_chunks = sorted(all_chunks, key=lambda x: x["score"], reverse=True)[:5]
    
    # Combine the most relevant chunks
    return "\n\n".join([chunk["text"] for chunk in relevant_chunks])

def format_response(response_text):
    """Formats response with proper line breaks and bullet points where necessary."""
    # Ensure bullet points are on new lines
    response_text = re.sub(r'(?<!\n)- ', '\n- ', response_text)
    
    # Ensure numbered lists are on new lines
    response_text = re.sub(r'(?<!\n)\d+\.\s', '\n\\g<0>', response_text)
    
    # Add line breaks after sentences, but not if they're already followed by line breaks
    response_text = re.sub(r'\.(?!\n)(?!\s*\n)\s+', '.\n', response_text)
    
    # Remove excessive line breaks (more than 2 consecutive)
    response_text = re.sub(r'\n{3,}', '\n\n', response_text)
    
    return response_text.strip()

@lru_cache(maxsize=100)
def cached_answer(question_hash, pdf_ver):
    """Cached version of answer generation to avoid redundant API calls."""
    logger.info(f"Cache miss for question hash: {question_hash}, generating new response")
    return answer_question(pdf_data, question_hash)

def answer_question(pdf_data, question):
    """Generates an AI response based on relevant PDF content and user query."""
    relevant_text = get_relevant_chunks(pdf_data, question)
    
    if not relevant_text.strip():
        return "I don't have enough information in my knowledge base to answer this question confidently. For more specific assistance with EZ-Invoice, please contact our support team."

    system_prompt = """
    You are an EZ-Invoice FAQ Bot AI assistant designed to provide accurate, helpful information based on your knowledge base.
    
    GUIDELINES:
    - Respond in a friendly, professional tone
    - Format responses with bullet points for readability when appropriate
    - Use 2-3 sentence paragraphs
    - If the knowledge base doesn't contain information to answer the question, politely state that you don't have that specific information
    - Avoid mentioning "knowledge base" or "PDF" in responses
    - Focus on the specific question asked
    - Keep answers concise yet complete
    """

    user_prompt = f"""
    Based on this information about EZ-Invoice:
    
    {relevant_text}
    
    Please answer this question clearly and accurately:
    {question}
    """

    try:
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=500,
            temperature=0.7,
            top_p=0.9,
        )
        
        response_text = response.choices[0].message.content.strip()
        return format_response(response_text)
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return "I'm sorry, I encountered an error while processing your request. Please try again in a moment."

# --- GOOGLE DRIVE INTEGRATION ---
def get_google_drive_service():
    """Authenticates and returns a Google Drive service instance."""
    try:
        google_creds_json = os.getenv("GOOGLE_CREDENTIALS")
        if not google_creds_json:
            logger.error("GOOGLE_CREDENTIALS not found in environment variables!")
            return None

        google_creds_dict = json.loads(google_creds_json)
        creds = Credentials.from_service_account_info(google_creds_dict)
        return build("sheets", "v4", credentials=creds)
    except Exception as e:
        logger.error(f"Error creating Google Drive service: {e}")
        return None

def append_to_google_sheet(data):
    """Appends chatbot logs to a Google Sheet."""
    try:
        service = get_google_drive_service()
        if not service:
            logger.warning("Could not initialize Google Sheets service, skipping log.")
            return False
            
        spreadsheet_id = os.getenv("GOOGLE_SHEET_ID")
        if not spreadsheet_id:
            logger.warning("GOOGLE_SHEET_ID not found in environment variables!")
            return False
            
        range_name = "ChatLogs!A:C"  

        values = [[datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), data[0][1], data[0][2]]]
        body = {"values": values}

        logger.debug(f"Sending data to Google Sheets: {body}")

        response = service.spreadsheets().values().append(
            spreadsheetId=spreadsheet_id,
            range=range_name,
            valueInputOption="RAW",
            insertDataOption="INSERT_ROWS",
            body=body
        ).execute()

        logger.debug(f"Google Sheets API response: {response}")
        return True

    except Exception as e:
        logger.error(f"Error in append_to_google_sheet: {e}\n{traceback.format_exc()}")
        return False

# Pre-load PDF texts at server startup
pdf_data = load_all_pdfs(PDF_FOLDER)

@app.errorhandler(Exception)
def handle_exception(e):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {str(e)}\n{traceback.format_exc()}")
    return jsonify({
        "error": "An unexpected error occurred. Our team has been notified.",
        "details": str(e) if app.debug else None
    }), 500

@app.route("/", methods=["GET"])
def home():
    """Render the main chat interface."""
    return render_template("index.html")

@app.route("/health", methods=["GET"])
def health_check():
    """Simple endpoint to check if the service is running."""
    return jsonify({
        "status": "ok", 
        "pdf_files": len(pdf_data),
        "version": pdf_version
    })

@app.route("/chat", methods=["POST"])
def chat():
    """Handles user queries and logs them to Google Sheets."""
    try:
        data = request.json
        user_message = data.get("message", "")

        if not user_message:
            return jsonify({"error": "No message provided"}), 400

        # Log the incoming request
        logger.info(f"Received question: {user_message}")
        
        # Generate a stable hash for the question to use as cache key
        question_hash = user_message.lower().strip()
        
        # Get cached or fresh response
        bot_response = cached_answer(question_hash, pdf_version)
        
        # Log success
        logger.info(f"Generated response of length {len(bot_response)}")

        # Log query & response
        chat_log = [[datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), user_message, bot_response]]
        log_success = append_to_google_sheet(chat_log)
        
        if not log_success:
            logger.warning("Failed to log chat to Google Sheets")

        return jsonify({"response": bot_response})
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            "response": "I'm sorry, I encountered an error while processing your request. Please try again in a moment."
        }), 200  # Return 200 to client with error message to handle gracefully

@app.route("/reload", methods=["POST"])
def reload_pdfs():
    """Admin endpoint to reload PDFs when content changes."""
    try:
        auth_key = request.headers.get("X-Auth-Key")
        expected_key = os.getenv("ADMIN_AUTH_KEY", "")
        
        if not auth_key or auth_key != expected_key:
            logger.warning(f"Unauthorized reload attempt from {request.remote_addr}")
            return jsonify({"error": "Unauthorized"}), 401
            
        global pdf_data, pdf_version
        old_version = pdf_version
        pdf_data = load_all_pdfs(PDF_FOLDER)
        
        return jsonify({
            "success": True,
            "message": f"PDFs reloaded successfully. Version changed from {old_version} to {pdf_version}",
            "file_count": len(pdf_data)
        })
        
    except Exception as e:
        logger.error(f"Error reloading PDFs: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting EZ-Invoice FAQ Bot on port {port}")
    logger.info(f"Loaded {len(pdf_data)} PDF files from {PDF_FOLDER}")
    app.run(host="0.0.0.0", port=port)

import os
import json
import logging
import traceback
import re
from functools import lru_cache
from flask import Flask, request, jsonify, render_template, session
from together import Together
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
import datetime
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import redis
import pytz
import chromadb
import requests
import uuid
from collections import deque

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
app.secret_key = os.getenv("FLASK_SECRET_KEY", os.urandom(24))
client = Together(api_key=os.getenv("TOGETHER_API_KEY"))

# Configure Redis connection for rate limiting and conversation storage
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis_client = redis.from_url(redis_url)

# Initialize rate limiter with Redis storage
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    storage_uri=redis_url,
    storage_options={"connection_pool": redis_client.connection_pool},
    default_limits=["2000 per day", "500 per hour"]
)

# Initialize ChromaDB client
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = chroma_client.get_or_create_collection(name="documents")

PDF_FOLDER = "pdfs"

# Track PDF content version for cache invalidation
pdf_version = "1.0"

# Conversation context settings
CONVERSATION_TTL = int(os.getenv("CONVERSATION_TTL", 3600))  # 1 hour by default
CONVERSATION_MAX_TURNS = int(os.getenv("CONVERSATION_MAX_TURNS", 5))  # Store up to 5 turns

def update_pdf_version():
    """Update the PDF version based on ChromaDB collection contents."""
    global pdf_version
    try:
        # Get the collection count and use it as part of the version
        collection_info = collection.count()
        # Get current timestamp for version
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M")
        pdf_version = f"{timestamp}-{collection_info}"
        logger.info(f"Updated PDF content version: {pdf_version}")
    except Exception as e:
        logger.error(f"Error updating PDF version: {e}")
        pdf_version = datetime.datetime.now().strftime("%Y%m%d%H%M")

def get_relevant_chunks(question, limit=5):
    """Find relevant chunks from ChromaDB based on the question."""
    try:
        # Use ChromaDB to query relevant chunks
        results = collection.query(
            query_texts=[question],
            n_results=limit
        )
        
        # Extract and join the relevant documents
        if results and 'documents' in results and results['documents']:
            documents = results['documents'][0]  # Get the first query result
            # Join all retrieved documents
            return "\n\n".join(documents)
        else:
            logger.warning("No relevant documents found in ChromaDB")
            return ""
    except Exception as e:
        logger.error(f"Error querying ChromaDB: {e}")
        return ""

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

def get_conversation_context(conversation_id):
    """Retrieves the conversation history for a given ID."""
    try:
        conversation_key = f"conversation:{conversation_id}"
        conversation_data = redis_client.get(conversation_key)
        
        if conversation_data:
            return json.loads(conversation_data)
        else:
            return {"messages": [], "created_at": datetime.datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Error retrieving conversation context: {e}")
        return {"messages": [], "created_at": datetime.datetime.now().isoformat()}

def save_conversation_context(conversation_id, context):
    """Saves the conversation context to Redis."""
    try:
        conversation_key = f"conversation:{conversation_id}"
        redis_client.set(
            conversation_key, 
            json.dumps(context),
            ex=CONVERSATION_TTL  # Expire after TTL seconds
        )
        return True
    except Exception as e:
        logger.error(f"Error saving conversation context: {e}")
        return False

def update_conversation_context(conversation_id, user_message, bot_response):
    """Updates the conversation context with new messages."""
    context = get_conversation_context(conversation_id)
    
    # Add new messages to the context
    context["messages"].append({"role": "user", "content": user_message})
    context["messages"].append({"role": "assistant", "content": bot_response})
    
    # Keep only the last N turns
    if len(context["messages"]) > CONVERSATION_MAX_TURNS * 2:
        context["messages"] = context["messages"][-CONVERSATION_MAX_TURNS * 2:]
    
    # Update the context in Redis
    save_conversation_context(conversation_id, context)

def get_conversation_history_as_string(conversation_id):
    """Formats conversation history as a string for context."""
    context = get_conversation_context(conversation_id)
    
    if not context["messages"]:
        return ""
    
    formatted_history = []
    for i in range(0, len(context["messages"]), 2):
        if i + 1 < len(context["messages"]):
            user_msg = context["messages"][i]["content"]
            bot_msg = context["messages"][i + 1]["content"]
            formatted_history.append(f"User: {user_msg}\nAssistant: {bot_msg}")
    
    return "\n\n".join(formatted_history)

@lru_cache(maxsize=100)
def cached_answer(question_hash, conversation_id, pdf_ver):
    """Cached version of answer generation to avoid redundant API calls."""
    logger.info(f"Cache miss for question hash: {question_hash}, conversation: {conversation_id}, generating new response")
    return answer_question(question_hash, conversation_id)

def answer_question(question, conversation_id):
    """Generates an AI response based on relevant ChromaDB content and conversation context."""
    relevant_text = get_relevant_chunks(question)
    conversation_history = get_conversation_history_as_string(conversation_id)
    
    if not relevant_text.strip():
        return "I don't have enough information in my knowledge base to answer this question confidently. For more specific assistance with E-Invoice, please contact our support team."

    system_prompt = """
    You are an E-Invoice FAQ Bot AI assistant designed by eZee Technosys (M) Sdn Bhd to provide accurate, helpful information in a friendly and approachable way.
    
    HOW TO RESPOND:
    - Be **warm and conversational**— like a helpful and friendly coworker, not a textbook.
    - **Keep things simple**— avoid unnecessary jargon.
    - **Use a natural, inviting tone**—think *"I got you!"* instead of *"I am a chatbot."*
    - **Format for readability**—bullet points or numbered steps work well, but mix them with short, clear sentences.
    - **Acknowledge what you don't know**—if unsure, say so in a helpful way (e.g., *"I don't have that info right now, but here's where you can check!"*).
    - **Make small talk**- allow for small talk if a user starts so, but redirect the conversation to ask if they need help with E-invoicing.
    - **Stay focused on the question**—no extra fluff, just what the user needs.
    - **Don't reference your knowledge base**- avoid mentioning specific sections in your knowledge base. Avoid mentioning "based/according to the information.."
    - **Maintain conversation context**- refer back to previous exchanges when relevant.

    EXAMPLE TONES:
    **Bad:** "Hello. Please provide a specific question so I may assist you." (Too cold.)
    **Good:** "Hey! Looks like you didn't ask a question yet— let me know what you need, and I'll help however I can!" (Warm, helpful, natural.)
    """

    user_prompt = f"""
    Based on this information about E-Invoice:
    
    {relevant_text}
    
    Previous conversation:
    {conversation_history}
    
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
            
        range_name = "ChatLogs!A:D"  # Added extra column for conversation ID

        malaysia_tz = pytz.timezone("Asia/Kuala_Lumpur")
        timestamp_myt = datetime.datetime.now(malaysia_tz).strftime("%Y-%m-%d %H:%M:%S")

        values = [[timestamp_myt, data[0][1], data[0][2], data[0][3]]]  # Include conversation ID
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

# Initialize PDF version at startup
update_pdf_version()

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
    # Generate a conversation ID if one doesn't exist
    if "conversation_id" not in session:
        session["conversation_id"] = str(uuid.uuid4())
    
    return render_template("index.html", conversation_id=session["conversation_id"])

@app.route("/health", methods=["GET"])
def health_check():
    """Simple endpoint to check if the service is running."""
    try:
        collection_count = collection.count()
        return jsonify({
            "status": "ok", 
            "document_count": collection_count,
            "version": pdf_version,
            "redis_status": "connected" if redis_client.ping() else "disconnected",
            "chroma_status": "connected"
        })
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return jsonify({
            "status": "degraded",
            "error": str(e),
            "redis_status": "connected" if redis_client.ping() else "disconnected",
            "chroma_status": "error"
        }), 500

@app.route("/chat", methods=["POST"])
@limiter.limit("500 per hour")  # Apply rate limiting to this endpoint
def chat():
    """Handles user queries and maintains conversation context."""
    try:
        data = request.json
        user_message = data.get("message", "")
        conversation_id = data.get("conversation_id", str(uuid.uuid4()))
        
        # If new conversation, use a new UUID
        if not conversation_id or conversation_id == "null":
            conversation_id = str(uuid.uuid4())

        if not user_message:
            return jsonify({"error": "No message provided"}), 400

        # Log the incoming request
        logger.info(f"Received question: {user_message} (Conversation: {conversation_id})")
        
        # Generate a stable hash for the question to use as cache key
        question_hash = user_message.lower().strip()
        
        # Get cached or fresh response
        bot_response = cached_answer(question_hash, conversation_id, pdf_version)
        
        # Update conversation context
        update_conversation_context(conversation_id, user_message, bot_response)
        
        # Log success
        logger.info(f"Generated response of length {len(bot_response)} for conversation {conversation_id}")

        # Log query & response
        chat_log = [[datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), user_message, bot_response, conversation_id]]
        log_success = append_to_google_sheet(chat_log)
        
        if not log_success:
            logger.warning("Failed to log chat to Google Sheets")

        return jsonify({
            "response": bot_response,
            "conversation_id": conversation_id  # Return the conversation ID for the client to store
        })
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            "response": "I'm sorry, I encountered an error while processing your request. Please try again in a moment.",
            "conversation_id": conversation_id if 'conversation_id' in locals() else str(uuid.uuid4())
        }), 200  # Return 200 to client with error message to handle gracefully

@app.route("/clear-conversation", methods=["POST"])
def clear_conversation():
    """Clears the conversation history for a given conversation ID."""
    try:
        data = request.json
        conversation_id = data.get("conversation_id")
        
        if not conversation_id:
            return jsonify({"error": "No conversation ID provided"}), 400
            
        # Delete conversation from Redis
        redis_client.delete(f"conversation:{conversation_id}")
        
        # Generate a new conversation ID
        new_conversation_id = str(uuid.uuid4())
        
        return jsonify({
            "success": True,
            "message": "Conversation cleared successfully",
            "new_conversation_id": new_conversation_id
        })
        
    except Exception as e:
        logger.error(f"Error clearing conversation: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route("/refresh-cache", methods=["POST"])
@limiter.exempt  # Exempt admin endpoints from rate limiting
def refresh_cache():
    """Admin endpoint to refresh the cache when ChromaDB content changes."""
    try:
        auth_key = request.headers.get("X-Auth-Key")
        expected_key = os.getenv("ADMIN_AUTH_KEY", "")
        
        if not auth_key or auth_key != expected_key:
            logger.warning(f"Unauthorized refresh attempt from {request.remote_addr}")
            return jsonify({"error": "Unauthorized"}), 401
            
        # Clear the cache
        cached_answer.cache_clear()
        
        # Update PDF version to invalidate any existing cache
        old_version = pdf_version
        update_pdf_version()
        
        return jsonify({
            "success": True,
            "message": f"Cache refreshed successfully. Version changed from {old_version} to {pdf_version}",
            "document_count": collection.count()
        })
        
    except Exception as e:
        logger.error(f"Error refreshing cache: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route("/trigger-update", methods=["POST"])
@limiter.exempt  # Exempt admin endpoints from rate limiting
def trigger_update():
    """Admin endpoint to trigger an update from the update service."""
    try:
        auth_key = request.headers.get("X-Auth-Key")
        expected_key = os.getenv("ADMIN_AUTH_KEY", "")
        
        if not auth_key or auth_key != expected_key:
            logger.warning(f"Unauthorized update trigger attempt from {request.remote_addr}")
            return jsonify({"error": "Unauthorized"}), 401
            
        # Call the update service
        update_service_url = os.getenv("UPDATE_SERVICE_URL", "http://localhost:8000/update")
        response = requests.post(update_service_url)
        
        if response.status_code == 200:
            # Clear the cache and update version
            cached_answer.cache_clear()
            update_pdf_version()
            
            return jsonify({
                "success": True,
                "message": "Update triggered successfully",
                "update_service_response": response.json()
            })
        else:
            return jsonify({
                "success": False,
                "error": f"Update service returned status code {response.status_code}",
                "update_service_response": response.text
            }), 500
        
    except Exception as e:
        logger.error(f"Error triggering update: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting EZ-Invoice FAQ Bot on port {port}")
    
    # Log ChromaDB document count
    try:
        document_count = collection.count()
        logger.info(f"Connected to ChromaDB with {document_count} documents")
    except Exception as e:
        logger.error(f"Error connecting to ChromaDB: {e}")
    
    # Verify Redis connection at startup
    try:
        if redis_client.ping():
            logger.info(f"Successfully connected to Redis at {redis_url}")
        else:
            logger.warning("Redis ping failed - rate limiting may not work correctly")
    except Exception as e:
        logger.error(f"Redis connection error: {e}")
    
    app.run(host="0.0.0.0", port=port)

import os
from flask import Flask, request, jsonify
from PyPDF2 import PdfReader
from together import Together

app = Flask(__name__)
client = Together(api_key=os.getenv("TOGETHER_API_KEY"))  # Use env variable

UPLOAD_FOLDER = "pdfs"  # Updated folder path
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def extract_text_from_pdf(pdf_path):
    """Extract text from a single PDF file."""
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

def answer_question(pdf_text, question):
    """Answer user questions based on extracted PDF text."""
    if not pdf_text:
        return "⚠️ No text extracted from the PDF."

    prompt = f"The following is your knowledge base:\n\n{pdf_text[:4000]}\n\nAnswer the following question:\n{question}"

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=[{"role": "system", "content": "You are an AI assistant answering questions based on your knowledge base. If the user asks something outside of the PDF topic, use your general knowledge to answer."},
                  {"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0.7,
        top_p=0.9,
    )
    return response.choices[0].message.content.strip()

@app.route("/upload", methods=["POST"])
def upload_pdf():
    """Endpoint for uploading PDFs."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    pdf_text = extract_text_from_pdf(file_path)
    if not pdf_text:
        return jsonify({"error": "Failed to extract text from PDF"}), 400

    return jsonify({"message": "File uploaded successfully", "text": pdf_text[:5000]})  # Send extracted text

@app.route("/chat", methods=["POST"])
def chat():
    """Endpoint to handle user questions."""
    data = request.json
    pdf_text = data.get("pdf_text", "")
    user_message = data.get("message", "")

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    response = answer_question(pdf_text, user_message)
    return jsonify({"response": response})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Use Render’s assigned port or default to 10000
    app.run(host="0.0.0.0", port=port)

@app.route("/")
def home():
    return "hello, render is working!"


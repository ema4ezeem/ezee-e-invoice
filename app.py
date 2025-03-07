# app.py
import os
from flask import Flask, request, jsonify, render_template
from PyPDF2 import PdfReader
from together import Together

app = Flask(__name__)
client = Together(api_key=os.getenv("TOGETHER_API_KEY"))

PDF_PATH = "pdfs/yourfile.pdf"  # Ensure your file name matches exactly
pdf_text = ""

def extract_text_from_pdf(pdf_path):
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

# Pre-load PDF text at server startup
pdf_text = extract_text_from_pdf(PDF_PATH)

def answer_question(pdf_text, question):
    if not pdf_text:
        return "⚠️ No text extracted from the PDF."

    prompt = f"The following is your knowledge base:\n\n{pdf_text[:4000]}\n\nAnswer the following question:\n{question}"

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=[{"role": "system", "content": "You are an AI assistant answering questions based on your knowledge base. If the user asks something outside of the PDF topic, use your general knowledge."},
                  {"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0.7,
        top_p=0.9,
    )
    return response.choices[0].message.content.strip()

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "")

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    response = answer_question(pdf_text, user_message)
    return jsonify({"response": response})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

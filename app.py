import os
from flask import Flask, request, jsonify, render_template
from huggingface_hub import InferenceClient

app = Flask(__name__)

# get huggingface api key from environment variables
HF_API_KEY = os.getenv("HF_API_KEY")

if not HF_API_KEY:
    raise ValueError("HF_API_KEY is not set in environment variables.")

client = InferenceClient(
    provider="hf-inference",
    api_key=HF_API_KEY
)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "")

    messages = [{"role": "user", "content": user_message}]

    completion = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=messages,
        max_tokens=500,
    )

    bot_reply = completion.choices[0].message.content
    return jsonify({"response": bot_reply})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # use render's port or default to 10000
    app.run(host="0.0.0.0", port=port)

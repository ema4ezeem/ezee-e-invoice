from flask import Flask, request, jsonify, render_template
import os
import requests

app = Flask(__name__)

# get api key from environment variable
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
if not TOGETHER_API_KEY:
    raise ValueError("TOGETHER_API_KEY is not set in environment variables.")

TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "")

    payload = {
        "model": "togethercomputer/llama-3-8b-instruct",
        "messages": [{"role": "user", "content": user_message}],
        "max_tokens": 500
    }

    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.post(TOGETHER_API_URL, json=payload, headers=headers)

    if response.status_code == 200:
        bot_reply = response.json()["choices"][0]["message"]["content"]
        return jsonify({"response": bot_reply})
    else:
        return jsonify({"error": "API request failed", "details": response.json()}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

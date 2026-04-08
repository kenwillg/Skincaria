"""
server.py — Flask backend for Skincaria skincare recommender.
Loads the fine-tuned GPT-2 model and serves predictions via API.
"""

import json
import os
import torch
import tiktoken

from flask import Flask, render_template, request, jsonify
from llms_from_scratch.ch04 import GPTModel
from llms_from_scratch.ch05 import generate, text_to_token_ids, token_ids_to_text

app = Flask(__name__, template_folder=".", static_folder="static")

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True,
}

MODEL_CONFIG = {"emb_dim": 1024, "n_layers": 24, "n_heads": 16}  # gpt2-medium

BASE_CONFIG.update(MODEL_CONFIG)

MODEL_PATH = os.path.join(".models", "skincare-recommender-sft.pth")
METRICS_PATH = "training_metrics.json"

# ============================================================================
# LOAD MODEL
# ============================================================================
print("Loading model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = tiktoken.get_encoding("gpt2")

model = GPTModel(BASE_CONFIG)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.to(device)
model.eval()
print(f"Model loaded from {MODEL_PATH}")

# Load training metrics
metrics = {}
if os.path.exists(METRICS_PATH):
    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)
    print(f"Training metrics loaded ({len(metrics.get('train_losses', []))} steps)")


# ============================================================================
# INFERENCE
# ============================================================================
def format_input(instruction, user_input):
    """Format input in Alpaca-style prompt template."""
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{instruction}"
    )
    input_text = f"\n\n### Input:\n{user_input}" if user_input else ""
    return instruction_text + input_text


def get_response(instruction, user_input, max_tokens=100):
    """Generate a response from the fine-tuned model."""
    prompt = format_input(instruction, user_input)

    token_ids = generate(
        model=model,
        idx=text_to_token_ids(prompt, tokenizer).to(device),
        max_new_tokens=max_tokens,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256,
    )
    full_text = token_ids_to_text(token_ids, tokenizer)
    response = full_text[len(prompt):].replace("### Response:", "").strip()
    return response


# ============================================================================
# ROUTES
# ============================================================================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    instruction = data.get("instruction", "I need a skincare product recommendation.")
    user_input = data.get("input", "")
    max_tokens = data.get("max_tokens", 256)

    if not user_input.strip():
        return jsonify({"error": "Please describe your skin concern."}), 400

    try:
        response = get_response(instruction, user_input, max_tokens)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/metrics")
def get_metrics():
    return jsonify(metrics)


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("\n✨ Skincaria server running at http://localhost:5000\n")
    app.run(debug=False, port=5000)

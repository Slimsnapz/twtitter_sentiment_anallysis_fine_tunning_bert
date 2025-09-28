%%writefile agent_quickstart.py
# agent_sentiment_quickstart.py
r"""
Agent demo — improved:
- ignores unknown CLI args (works in Jupyter/IPython)
- prints helpful diagnostics if 'transformers' is missing
- uses configurable model_dir and confidence threshold
Usage example (Windows):
    python agent_sentiment_quickstart.py --model_dir "C:/Users/001/Documents/workbench/bert-base-uncased-sentiment-model" --threshold 0.30
(The path above uses forward slashes to avoid escape issues. You can also pass a raw string on the command line.)
"""

import argparse
import sys
import re

# Safe import with helpful message
try:
    from transformers import pipeline
except Exception as e:
    print("\n[ERROR] Failed to import `transformers`.\n")
    print("This usually means the current Python environment does not have the package installed.")
    print("Current Python executable:", sys.executable)
    print("\nTo fix, run (using the same python):")
    print("  python -m pip install --upgrade pip")
    print("  python -m pip install transformers torch sentencepiece\n")
    # If you're using a requirements file:
    print("Or: python -m pip install -r requirements.txt")
    raise SystemExit(1)

# Mapping from fine-grained emotion labels -> coarse sentiment
EMOTION_TO_SENTIMENT = {
    "joy": "positive",
    "love": "positive",
    "surprise": "neutral",   # treat surprise as neutral by default; change to 'positive' if desired
    "sadness": "negative",
    "anger": "negative",
    "fear": "negative",
}

def normalize_label(raw_label: str) -> str:
    if not raw_label:
        return ""
    s = raw_label.strip().lower()
    s = s.strip("\"'")
    m = re.search(r"\(([^)]+)\)", s)
    if m:
        return m.group(1).strip().lower()
    m2 = re.search(r"([a-z]+)$", s)
    if m2:
        return m2.group(1)
    return re.sub(r"[^a-z]+", "", s)

def map_to_sentiment(label_token: str) -> str:
    if not label_token:
        return "neutral"
    if label_token in EMOTION_TO_SENTIMENT:
        return EMOTION_TO_SENTIMENT[label_token]
    if label_token.startswith("pos") or "good" in label_token or "joy" in label_token or "love" in label_token:
        return "positive"
    if label_token.startswith("neg") or "sad" in label_token or "ang" in label_token or "fear" in label_token:
        return "negative"
    return "neutral"

def decision_from_sentiment(sentiment: str) -> str:
    if sentiment == "negative":
        return "I see negative sentiment. I’d escalate this to support."
    if sentiment == "positive":
        return "Positive sentiment detected. Logging this as good feedback."
    return "Neutral or mixed sentiment. Monitoring, no escalation."

def build_classifier(model_dir: str):
    return pipeline("text-classification", model=model_dir, return_all_scores=False)

def main(args=None):
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--model_dir", type=str, default=r"C:\Users\001\Documents\workbench\bert-base-uncased-sentiment-model",
                        help="Path to your fine-tuned model directory (or HF id).")
    parser.add_argument("--threshold", type=float, default=0.30,
                        help="Confidence threshold below which agent flags for human review.")
    # parse_known_args lets this script run inside Jupyter without failing on extra kernel args
    parsed, unknown = parser.parse_known_args(args=args)

    try:
        classifier = build_classifier(parsed.model_dir)
    except Exception as e:
        print(f"[ERROR] Failed to load model from {parsed.model_dir}: {e}", file=sys.stderr)
        print("Current Python executable:", sys.executable)
        print("Make sure the model path is correct and the model was saved with transformers' save_pretrained().")
        raise SystemExit(1)

    print("🤖 Agent Demo — Sentiment Assistant")
    print("Type 'quit' to exit.\n")

    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAgent: Goodbye!")
            break

        if user_text.lower() in {"quit", "exit"}:
            print("Agent: Goodbye!")
            break
        if user_text == "":
            print("Agent: Please type some text or 'quit' to exit.")
            continue

        try:
            pred = classifier(user_text)
            top = pred[0] if isinstance(pred, (list, tuple)) else pred
            raw_label = top.get("label", "")
            score = float(top.get("score", 0.0))
        except Exception as e:
            print(f"Agent: Error running classifier: {e}")
            continue

        token = normalize_label(raw_label)
        sentiment = map_to_sentiment(token)

        if score < parsed.threshold:
            print(f"Agent: Low confidence ({score:.2%}) — uncertain result. Model label: {raw_label}")
            print("Agent: Recommendation: flag for human review.")
            continue

        response = decision_from_sentiment(sentiment)
        print(f"Agent: {response} (Model: {raw_label} {score:.2%})")

if __name__ == "__main__":
    # pass None so parse_known_args uses sys.argv normally when run as script
    main()

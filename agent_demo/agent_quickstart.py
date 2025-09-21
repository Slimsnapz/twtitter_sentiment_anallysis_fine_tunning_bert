%%writefile agent_quickstart.py
"""
agent_quickstart.py
A minimal "agent" demo that wraps the fine-tuned BERT sentiment classifier
and shows how it could be used in a conversational assistant workflow.

Run with:
    python agent_demo/agent_quickstart.py
"""

from transformers import pipeline

def main():
    # Load your fine-tuned BERT model
    classifier = pipeline(
        "text-classification",
        model=r"models/exp1",   # Update path if needed
    )

    print("🤖 Agent Demo — Sentiment Assistant")
    print("Type 'quit' to exit.\n")

    while True:
        user_text = input("You: ")
        if user_text.lower().strip() in {"quit", "exit"}:
            print("Agent: Goodbye!")
            break

        # Use BERT model to classify sentiment
        prediction = classifier(user_text)[0]
        label, score = prediction["label"], prediction["score"]

        # Agent "decision" based on sentiment
        if "NEG" in label.upper():
            response = "I see negative sentiment. I’d escalate this to support."
        elif "POS" in label.upper():
            response = "Positive sentiment detected. Logging this as good feedback."
        else:
            response = "Neutral or mixed sentiment. Monitoring, no escalation."

        print(f"Agent: {response} (Model: {label} {score:.2%})")

if __name__ == "__main__":
    main()

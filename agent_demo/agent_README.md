# Agent Demo Sentiment Assistant

This folder contains a minimal **agent-style demo** that shows how the fine-tuned BERT sentiment classifier could be integrated into a conversational workflow.

- Wrapping the BERT sentiment model in a loop
- Routing user inputs to the model
- Making agent-style decisions (escalate, log, monitor)

## How to run
1. Make sure you have the trained model downloaded at `models/exp1/`.
2. Install requirements if not already:
   ```bash
   pip install -r requirements.txt

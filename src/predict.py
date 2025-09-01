"""
src/predict.py
Make predictions using a saved fine-tuned model.
Usage:
python src/predict.py --model_dir models/exp1 --text "sample text"
or
python src/predict.py --model_dir models/exp1 --file sample_texts.txt
"""
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

def load_model(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tokenizer, model, device

def predict_texts(texts, tokenizer, model, device, top_k=3):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    inputs = {k:v.to(device) for k,v in inputs.items()}
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
    topk = np.argsort(-probs, axis=1)[:,:top_k]
    return probs, topk

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--file", type=str, default=None, help="text file with one sample per line")
    return parser.parse_args()

def main():
    args = parse_args()
    tokenizer, model, device = load_model(args.model_dir)
    if args.text:
        texts = [args.text]
    elif args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            texts = [l.strip() for l in f if l.strip()]
    else:
        print("Provide --text or --file")
        return
    probs, topk = predict_texts(texts, tokenizer, model, device, top_k=3)
    for i,t in enumerate(texts):
        print("Text:", t)
        print("Top predictions:")
        for rank, idx in enumerate(topk[i]):
            print(f"  {rank+1}. Class {idx} - prob {probs[i][idx]:.4f}")
        print("-"*40)

if __name__ == "__main__":
    main()

"""
src/eval.py
Evaluate a saved model on a test CSV or HF dataset.
Usage:
python src/eval.py --model_dir models/exp1 --test_file data/test.csv
"""
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
import numpy as np

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    from sklearn.metrics import accuracy_score, f1_score
    return {"accuracy": float(accuracy_score(labels, preds)), "f1": float(f1_score(labels, preds, average='weighted'))}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=False)
    return parser.parse_args()

def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)

    if args.test_file and args.test_file.endswith(".csv"):
        ds = load_dataset("csv", data_files={"test": args.test_file})['test']
    elif args.test_file:
        ds = load_dataset(args.test_file)['test']
    else:
        raise ValueError("Provide --test_file path or HF dataset id")

    def tokenize_fn(batch):
        return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=128)
    ds = ds.map(tokenize_fn, batched=True)
    ds.set_format(type='torch', columns=['input_ids','attention_mask','label'])

    trainer = Trainer(model=model, tokenizer=tokenizer, compute_metrics=compute_metrics)
    preds = trainer.predict(ds)
    print("Eval metrics:", preds.metrics)

if __name__ == "__main__":
    main()

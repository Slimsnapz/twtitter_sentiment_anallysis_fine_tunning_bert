"""
src/train.py
Fine-tune a pretrained transformer (BERT) for sentiment classification using Hugging Face Trainer.
Usage:
python src/train.py --train_file data/train.csv --model_name_or_path bert-base-uncased --output_dir models/exp1
"""
import argparse
import os
import numpy as np
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    f1 = f1_score(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average='weighted')
    rec = recall_score(labels, preds, average='weighted')
    return {"accuracy": acc, "f1": f1, "precision": prec, "recall": rec}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, required=False, help="CSV path or HF dataset id")
    parser.add_argument("--valid_file", type=str, required=False)
    parser.add_argument("--test_file", type=str, required=False)
    parser.add_argument("--model_name_or_path", type=str, default="bert-base-uncased")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=128)
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset: if train_file is a path, load CSVs; else expect HF dataset id
    if args.train_file and args.train_file.endswith(".csv"):
        data_files = {}
        data_files["train"] = args.train_file
        if args.valid_file:
            data_files["validation"] = args.valid_file
        if args.test_file:
            data_files["test"] = args.test_file
        raw = load_dataset("csv", data_files=data_files)
    elif args.train_file:
        raw = load_dataset(args.train_file)
        # expect splits train/validation/test or train/test
        if "train" not in raw:
            raise ValueError("Dataset does not contain a 'train' split.")
    else:
        raise ValueError("Please provide --train_file path or HF dataset id via --train_file")

    # Auto-detect text & label column names (common names)
    # Here we assume column named 'text' and 'label'; otherwise rename or adapt.
    # Tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    def tokenize_fn(batch):
        return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=args.max_length)

    tokenized = raw.map(tokenize_fn, batched=True)
    # remove irrelevant columns but keep label & inputs
    cols_to_keep = [c for c in tokenized['train'].column_names if c in ['input_ids','attention_mask','label','token_type_ids','labels']]
    remove_cols = [c for c in tokenized['train'].column_names if c not in cols_to_keep]
    tokenized = tokenized.remove_columns(remove_cols)
    tokenized.set_format(type='torch')

    # Determine number of labels
    labels = np.unique(tokenized['train']['label'])
    num_labels = int(len(labels))

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=num_labels)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="epoch" if "validation" in tokenized else "no",
        save_strategy="epoch",
        learning_rate=args.lr,
        load_best_model_at_end=True if "validation" in tokenized else False,
        metric_for_best_model="f1",
        logging_dir=os.path.join(args.output_dir, "logs"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized.get('train', None),
        eval_dataset=tokenized.get('validation', None),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics if "validation" in tokenized else None
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved model to {args.output_dir}")

if __name__ == "__main__":
    main()

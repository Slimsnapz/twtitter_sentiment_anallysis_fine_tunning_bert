"""
src/data_loader.py
Helpers to load CSV dataset and prepare HF datasets for BERT fine-tuning.
Expect CSV with at least 'text' and 'label' columns.
"""
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

def load_csv_to_df(path: str):
    df = pd.read_csv(path)
    return df

def prepare_dataset_from_df(df, text_col='text', label_col='label', test_size=0.1, val_size=0.1, random_state=42):
    """
    Splits dataframe into train/val/test and converts to Hugging Face Datasets.
    Returns a DatasetDict with keys 'train','validation','test'
    """
    train_val_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df[label_col] if label_col in df.columns else None)
    train_df, val_df = train_test_split(train_val_df, test_size=val_size/(1-test_size), random_state=random_state, stratify=train_val_df[label_col] if label_col in train_val_df.columns else None)
    return DatasetDict({
        "train": Dataset.from_pandas(train_df.reset_index(drop=True)),
        "validation": Dataset.from_pandas(val_df.reset_index(drop=True)),
        "test": Dataset.from_pandas(test_df.reset_index(drop=True))
    })

if __name__ == "__main__":
    print("This module provides helpers for loading and preparing datasets.")

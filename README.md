# BERT Fine-Tuning - Sentiment Analysis (Portfolio Project)
### Model screenshot
![Project Screenshot](https://github.com/Slimsnapz/twtitter_sentiment_anallysis_fine_tunning_bert/blob/0972318ce146943a0681e397f0265abf3503731f/screenshots/Screenshot%202025-09-20%20071828.png) 
**MODEL BEFORE INPUTING TEXT**
<br>
![Project Screenshot](https://github.com/Slimsnapz/twtitter_sentiment_anallysis_fine_tunning_bert/blob/0972318ce146943a0681e397f0265abf3503731f/screenshots/Screenshot%202025-09-20%20071957.png)
**MODEL WITH TEXT INPUTTED**
<br>
![Project Screenshot](https://github.com/Slimsnapz/twtitter_sentiment_anallysis_fine_tunning_bert/blob/0972318ce146943a0681e397f0265abf3503731f/screenshots/Screenshot%202025-09-20%20072015.png) 
**MODEL AFTER PREDICTION OF INPUTTED TEXT**

## Project summary (business problem & solution)
**Business problem:** Companies receive large volumes of user-generated text (reviews, support tickets, social media mentions) and need to understand customer sentiment at scale. Manual review is slow, inconsistent, and expensive - making it hard to act quickly on negative feedback or identify product improvements.

**Solution:** I built an end-to-end sentiment analysis pipeline that fine-tunes a pretrained **BERT** model to classify short text (e.g., tweets or reviews) by sentiment. The solution automates sentiment tagging, enabling teams to prioritize negative feedback, monitor trends, and feed insights into dashboards or ticketing workflows. The project includes training, evaluation, inference scripts, and a Streamlit demo for stakeholder validation.


## What I built (technical summary)
- **Model:** Fine-tuned `bert-base-uncased` for sentiment classification using Hugging Face `transformers` and `Trainer`.
- **Dataset:** Accepts CSV datasets with `text` and `label` columns, or Hugging Face datasets (not included in repo). Example datasets: Twitter sentiment, product reviews, or a custom labeled dataset.
- **Key components:**
  - Data loading & splitting utilities (`src/data_loader.py`)
  - Training script with configurable hyperparameters (`src/train.py`)
  - Evaluation script to report accuracy and F1 (`src/eval.py`)
  - Prediction CLI to classify single texts or files (`src/predict.py`)
  - Streamlit demo for non-technical stakeholders (`src/app.py`)
- **Typical training config used in experiments:** batch_size=16, epochs=3, lr=2e-5, max_length=128.


## Business impact
- Automates sentiment labeling across large volumes of text, reducing manual review costs.
- Enables faster detection of negative sentiment spikes and supports proactive customer engagement.
- Provides a deployable inference interface (Streamlit) for product managers and non-technical stakeholders to validate model performance before production rollout.


## Evaluation & metrics

- **Test accuracy:** `0.90`  
- **Weighted F1:** `0.90`  
- **Per-class support & F1:**<br>

### Model Training Loss
![Project Screenshot](https://github.com/Slimsnapz/twtitter_sentiment_anallysis_fine_tunning_bert/blob/f665b8e1a364862209483e7c4ea9029c0541fc3a/screenshots/Screenshot%202025-09-01%20153353.png) 

### Model Confusionn Matrix
![Project Screenshot](https://github.com/Slimsnapz/twtitter_sentiment_anallysis_fine_tunning_bert/blob/e95191404d951226140203ac1b85722836f0194c/screenshots/Screenshot%202025-09-18%20101610.png) 

## Where the trained model lives
The trained model is **not** tracked in this repository due to size. Host weights on Google Drive or Hugging Face and provide download instructions in `models/download_model_README.md`. After downloading and placing the model at `models/exp1/`, load it like this:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
model = AutoModelForSequenceClassification.from_pretrained("models/exp1")
tokenizer = AutoTokenizer.from_pretrained("models/exp1")
```


## File structure
```
/ (root)
├─ notebooks/
│  └─ FineTunningBert_Sentiment_analysis_model.ipynb   # original notebook (cleaned)
├─ src/
│  ├─ data_loader.py    # dataset & HF Dataset helpers
│  ├─ train.py          # training script (Trainer-based)
│  ├─ eval.py           # evaluation script
│  ├─ predict.py        # CLI inference script
│  └─ app.py            # Streamlit demo app for stakeholders
├─ scripts/
│  └─ download_model.py # helper to fetch model from Google Drive (optional)
├─ models/              # NOT committed (large files) — include models/README.md (tracked)
│  └─ README.md         # instructions for downloading trained weights (tracked)
├─ screenshots/         # training curves, confusion matrices, demo screenshots
├─ requirements.txt
├─ .gitignore
├─ DATA_DICTIONARY.md   # (generated)
├─ QUANTIZATION_INSTRUCTIONS.md  # (generated)
├─ quantize.py          # (generated)
├─ Dockerfile           # (generated)
└─ README.md            # project README (recruiter-facing)

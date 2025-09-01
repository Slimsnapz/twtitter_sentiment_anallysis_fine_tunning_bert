# Project File Structure (Sentiment Analysis Project)

This is the recommended file layout for the BERT sentiment analysis project you can upload to GitHub:

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
├─ PROJECT_STRUCTURE.md # (this file)
└─ README.md            # project README (recruiter-facing)

"""
src/app.py
Streamlit app to demo the sentiment classification model.
Run with:
streamlit run src/app.py -- --model_dir models/exp1
"""
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

@st.cache_resource
def load_model(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tokenizer, model, device

def predict(text, tokenizer, model, device, top_k=3):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k:v.to(device) for k,v in inputs.items()}
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    topk = np.argsort(-probs)[:top_k]
    id2label = model.config.id2label if hasattr(model.config, "id2label") else {i:str(i) for i in range(len(probs))}
    return [(id2label.get(int(i), str(i)), float(probs[int(i)])) for i in topk]

st.title("BERT Sentiment Classification Demo")
st.write("Enter text and the fine-tuned BERT model will return sentiment predictions.")

model_dir = st.sidebar.text_input("Model directory", value="models/exp1")
if st.sidebar.button("Load model"):
    with st.spinner("Loading model..."):
        tokenizer, model, device = load_model(model_dir)
    st.success("Model loaded.")

text = st.text_area("Input text", value="I love this product!")
if st.button("Predict"):
    try:
        tokenizer, model, device
    except NameError:
        with st.spinner("Loading model..."):
            tokenizer, model, device = load_model(model_dir)
    with st.spinner("Predicting..."):
        preds = predict(text, tokenizer, model, device, top_k=3)
    st.write("Top predictions:")
    for label, prob in preds:
        st.write(f"- {label}: {prob:.4f}")

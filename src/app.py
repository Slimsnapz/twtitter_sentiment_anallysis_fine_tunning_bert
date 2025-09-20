%%writefile app.py
import streamlit as st
from transformers import pipeline

st.set_page_config(
    page_title="BERT Sentiment App",
    page_icon="📝",
    layout="centered",
)

@st.cache_resource
def load_model():
    """Loads the sentiment analysis model and caches it."""
    # Keep your local model path inside the function (hidden from UI).
    return pipeline(
        "text-classification",
        model=r"C:\Users\001\Documents\workbench\bert-base-uncased-sentiment-model",
    )

# --- Styles (small, safe CSS) ---
st.markdown(
    """
    <style>
    .app-header {text-align:center; margin-bottom: 0.5rem;}
    .app-sub {text-align:center; color: #6b7280; margin-top: -6px; margin-bottom: 1.4rem;}
    .result-box {padding: 10px; border-radius: 10px; box-shadow: 0 6px 18px rgba(15,23,42,0.06); background: white;}
    .stButton>button { border-radius: 8px; padding: 8px 14px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown("<div class='app-header'><h1 style='margin:0;'>📝 Fine-Tuning BERT — Sentiment Analysis</h1></div>", unsafe_allow_html=True)
st.markdown("<div class='app-sub'>Type a sentence and click <em>Predict</em>. Shows label + confidence score.</div>", unsafe_allow_html=True)

# Sidebar for instructions / quick info (model path removed)
with st.sidebar:
    st.header("Quick help")
    st.write(
        """
        - Enter the text you want classified.
        - Click **Predict** to run the model.
        - The model is loaded from your local path and cached (hidden here).
        """
    )
    st.divider()
    st.write("Note: model path is intentionally hidden from the UI.")

# Load model (cached)
classifier = load_model()

# Layout: input and controls
col1, col2 = st.columns([3, 1])
with col1:
    text = st.text_area("Enter text to analyze", height=140, placeholder="Type something like: 'I love this product!'")
    predict = st.button("Predict")

with col2:
    st.write("")  # spacer
    st.write("")  # spacer
    st.info("Tip:\nShort, clear sentences give more confident scores.")

# Helper: map label to color and friendly name
def label_to_color_and_name(label: str):
    label_u = (label or "").upper()
    # Colors: positive (green), negative (red), neutral (gray), mixed (orange), other (blue)
    if "POS" in label_u or "GOOD" in label_u or "P" == label_u:
        return "#16a34a", label  # green
    if "NEG" in label_u or "BAD" in label_u or "N" == label_u:
        return "#dc2626", label  # red
    if "NEU" in label_u or "NEUTRAL" in label_u:
        return "#6b7280", label  # gray
    if "MIX" in label_u or "MIXED" in label_u or "CONFLICT" in label_u:
        return "#f97316", label  # orange
    # fallback
    return "#2563eb", label  # blue

# Prediction logic (safe and clear)
if predict:
    if not text or text.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing..."):
            raw = classifier(text)
            prediction = raw[0] if isinstance(raw, (list, tuple)) else raw
            label = prediction.get("label", "N/A")
            score = float(prediction.get("score", 0.0))

        color, friendly_label = label_to_color_and_name(label)

        # Result display
        st.markdown("<div class='result-box'>", unsafe_allow_html=True)

        # Badge with dynamic color (inline style)
        badge_html = (
            f"<div style='display:flex;align-items:center;gap:12px;'>"
            f"<span style='background:{color}; color: white; padding:8px 12px; border-radius:10px; font-weight:700;'>{friendly_label}</span>"
            f"<div style='font-weight:600'>{score:.2%} confidence</div>"
            f"</div>"
        )
        st.markdown(badge_html, unsafe_allow_html=True)

        # Progress bar (safe: integer 0..100)
        try:
            pct = int(max(0, min(100, round(score * 100))))
            st.progress(pct)
        except Exception:
            # fallback if something goes odd
            try:
                st.progress(score)
            except Exception:
                pass

        # Expand for detailed output
        with st.expander("Show raw prediction"):
            st.json(prediction)

        st.markdown("</div>", unsafe_allow_html=True)

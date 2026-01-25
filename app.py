import streamlit as st
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import os

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="📰",
    layout="centered"
)

MODEL_PATH = "fake_news_model"

# -------------------------------------------------
# Load Model (Cached for Performance)
# -------------------------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Model folder not found. Ensure 'fake_news_model' exists in the project directory."
        )

    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)

    device = torch.device("cpu")
    model.to(device)
    model.eval()

    return tokenizer, model, device


# -------------------------------------------------
# UI Header
# -------------------------------------------------
st.title("📰 Fake News Detection System")
st.markdown(
    """
This system uses a fine-tuned **DistilBERT Transformer model**
to classify news articles as **FAKE** or **REAL**.

Enter a news article below and click **Predict**.
"""
)

# -------------------------------------------------
# User Input
# -------------------------------------------------
user_input = st.text_area(
    "News Article Text:",
    height=220,
    placeholder="Paste the full news article here..."
)

# -------------------------------------------------
# Prediction Logic
# -------------------------------------------------
if st.button("Predict"):

    if not user_input.strip():
        st.warning("⚠ Please enter some text before predicting.")
    else:
        try:
            tokenizer, model, device = load_model()

            inputs = tokenizer(
                user_input,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=256
            )

            inputs = {key: val.to(device) for key, val in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)

            fake_prob = probs[0][0].item()
            real_prob = probs[0][1].item()

            if real_prob > fake_prob:
                prediction = "REAL"
                confidence = real_prob * 100
            else:
                prediction = "FAKE"
                confidence = fake_prob * 100

            confidence = round(confidence, 2)

            # -------------------------------------------------
            # Display Result
            # -------------------------------------------------

            st.markdown("---")

            if prediction == "REAL":
                st.success("✅ This article appears to be REAL.")
            else:
                st.error("🚨 This article appears to be FAKE.")

            st.progress(int(confidence))
            st.markdown(f"### Confidence Level: {confidence}%")

            st.markdown("#### Probability Breakdown")
            st.write(f"🟥 Fake Probability: {fake_prob * 100:.2f}%")
            st.write(f"🟩 Real Probability: {real_prob * 100:.2f}%")

        except FileNotFoundError as e:
            st.error(str(e))

        except Exception as e:
            st.error("❌ An unexpected error occurred during prediction.")
            st.exception(e)


# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("---")
st.markdown(
    """
Model: DistilBERT (Fine-Tuned on ISOT Fake News Dataset)  
Evaluation Metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix  
"""
)

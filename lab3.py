# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import torch
from transformers import BertTokenizer, T5Tokenizer
import evaluate
import nltk

nltk.download('punkt')

# Model metadata
MODELS = {
    "BERT": {
        "model_name": "bert-base-uncased",
        "tokenizer": "bert",
        "domain": "General"
    },
    "FinBERT": {
        "model_name": "ProsusAI/finbert",
        "tokenizer": "bert",
        "domain": "Finance"
    },
    "MediaSum-T5": {
        "model_name": "t5-base",  # Placeholder; assumes MediaSum-based T5 uses T5-base
        "tokenizer": "t5",
        "domain": "Media"
    }
}

@st.cache_resource
def load_tokenizer(model_info):
    if model_info["tokenizer"] == "bert":
        tokenizer = BertTokenizer.from_pretrained(model_info["model_name"])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    else:  # T5
        tokenizer = T5Tokenizer.from_pretrained(model_info["model_name"])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

def preprocess_prompt(text, model_name, max_length=128):
    model_info = MODELS[model_name]
    tokenizer = load_tokenizer(model_info)
    text = text.strip()

    if model_info["tokenizer"] == "bert":
        encoded = tokenizer(
            text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            add_special_tokens=True
        )
    else:  # T5
        encoded = tokenizer(
            text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

    return encoded['input_ids'], encoded['attention_mask']

# Simulated sentiment prediction (no actual model inference as per lab rules)
def generate_sentiment(model_name, text):
    # Mock predictions based on domain relevance (simulating model output)
    if model_name == "BERT":
        if "stock" in text.lower() or "market" in text.lower():
            return "Neutral"  # BERT less precise in finance
        elif "report" in text.lower():
            return "Positive"  # General interpretation
        else:
            return "Positive"
    elif model_name == "FinBERT":
        if "stock plummets" in text.lower():
            return "Negative"  # FinBERT understands finance
        elif "market" in text.lower():
            return "Neutral"
        else:
            return "Positive"
    elif model_name == "MediaSum-T5":
        if "report" in text.lower():
            return "Neutral"  # MediaSum-T5 understands news context
        elif "success" in text.lower():
            return "Positive"
        else:
            return "Neutral"

# Evaluation metrics
rouge = evaluate.load("rouge")

def evaluate_sentiment(reference, generated):
    if not generated.strip() or not reference.strip():
        return {"rougeL": 0.0}

    rouge_score = rouge.compute(predictions=[generated], references=[reference])
    return {"rougeL": rouge_score["rougeL"]}

# Main Streamlit App
def main():
    st.title("📊 Comparative Sentiment Analysis: Foundation vs Domain-Specific Models")

    st.markdown("""
    ### 🚀 Project Overview
    This app compares **sentiment classification** across:
    - **BERT** (Foundation model)
    - **FinBERT** (Finance domain)
    - **MediaSum-based T5** (Media domain)

    Evaluate outputs using ROUGE-L and human metrics (coherence, relevance, factual accuracy).
    """)

    # Sample prompts from synthetic dataset
    sample_prompts = [
        "TechCorp’s stock plummets after weak earnings forecast.",  # Finance
        "The concert was a huge success.",  # General
        "CNN reports rising tensions in global markets."  # Media
    ]

    selected_prompt = st.selectbox("Choose a sample prompt", sample_prompts)
    prompt = st.text_input("Or enter your own prompt", selected_prompt)

    if st.button("Generate and Compare"):
        st.subheader("📝 Generated Sentiments and Evaluation")
        auto_metrics = {"Model": [], "ROUGE-L": []}
        human_metrics = {"Model": [], "Coherence": [], "Relevance": [], "Factual Accuracy": [], "Average": []}

        # Reference sentiment for ROUGE evaluation (mock ground truth)
        reference_map = {
            sample_prompts[0]: "Negative",
            sample_prompts[1]: "Positive",
            sample_prompts[2]: "Neutral"
        }
        reference = reference_map.get(prompt, "Neutral")  # Default to Neutral for custom prompts

        for model_name in MODELS:
            with st.spinner(f"Processing with {model_name}..."):
                # Preprocess input
                input_ids, attention_mask = preprocess_prompt(prompt, model_name)
                
                # Generate sentiment (simulated)
                sentiment = generate_sentiment(model_name, prompt)
                metrics = evaluate_sentiment(reference, sentiment)
                
                auto_metrics["Model"].append(model_name)
                auto_metrics["ROUGE-L"].append(metrics["rougeL"])

                st.markdown(f"### 🔹 {model_name} ({MODELS[model_name]['domain']})")
                st.write(f"**Sentiment:** {sentiment}")
                st.write(f"**ROUGE-L:** {metrics['rougeL']:.4f}")

                with st.expander(f"Rate {model_name} (Human Evaluation)"):
                    coherence = st.slider(f"{model_name} - Coherence", 1, 5, 3, key=f"{model_name}_coh")
                    relevance = st.slider(f"{model_name} - Relevance", 1, 5, 3, key=f"{model_name}_rel")
                    factual = st.slider(f"{model_name} - Factual Accuracy", 1, 5, 3, key=f"{model_name}_fac")
                    avg = (coherence + relevance + factual) / 3
                    human_metrics["Model"].append(model_name)
                    human_metrics["Coherence"].append(coherence)
                    human_metrics["Relevance"].append(relevance)
                    human_metrics["Factual Accuracy"].append(factual)
                    human_metrics["Average"].append(avg)

        # Visualization
        st.subheader("📊 Comparative Visualization")
        df_auto = pd.DataFrame(auto_metrics).set_index("Model")
        st.markdown("### 🤖 Automatic Evaluation: ROUGE-L")
        st.bar_chart(df_auto)

        df_human = pd.DataFrame(human_metrics).set_index("Model")
        st.markdown("### 👤 Human Evaluation: Coherence, Relevance, Factual Accuracy")
        st.bar_chart(df_human[["Coherence", "Relevance", "Factual Accuracy"]])

        st.markdown("### 🌟 Average Human Ratings")
        st.bar_chart(df_human[["Average"]])

        with st.expander("🔍 Show Raw Metric Tables"):
            st.write("Automatic Metrics", df_auto)
            st.write("Human Evaluation", df_human)

if __name__ == "__main__":
    main()

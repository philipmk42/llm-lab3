Comparative Sentiment Analysis: Foundation vs Domain-Specific Models

📝 Project Overview
This repository contains a Streamlit web application for Lab Exercise 3, implementing a comparative analysis of transformer-based models for sentiment classification. The app compares:

BERT: A general-purpose foundation model.
FinBERT: A domain-specific model for finance.
MediaSum-based T5: A domain-specific model for media.

The analysis focuses on the importance of domain-specific models, preprocessing steps (tokenization, prompt formatting, truncation/padding), and evaluation using ROUGE-L and human metrics (coherence, relevance, factual accuracy). Visualizations include bar charts and tables for comparative outputs, fulfilling the lab’s requirements. No marks are credited for pipeline implementation, so the app simulates sentiment predictions to emphasize analysis and evaluation.
🎯 Features

Input Options: Select from sample prompts (finance, general, media) or enter custom text for sentiment classification.
Preprocessing: Tokenizes inputs using BERT’s WordPiece for BERT/FinBERT and T5’s SentencePiece for MediaSum-T5, with truncation/padding to 128 tokens.
Sentiment Outputs: Generates sentiment labels (positive, negative, neutral) for each model (simulated to align with lab rules).
Evaluation:
ROUGE-L: Measures text quality against mock ground-truth labels.
Human Evaluation: User inputs for coherence, relevance, and factual accuracy via sliders.


Visualization: Bar charts for ROUGE-L and human metrics, plus tables for raw data.
Sample Prompts:
Finance: “TechCorp’s stock plummets after weak earnings forecast.”
General: “The concert was a huge success.”
Media: “CNN reports rising tensions in global markets.”



🛠️ Installation

Clone the Repository:
git clone https://github.com/your-username/comparative-sentiment-analysis.git
cd comparative-sentiment-analysis


Install Dependencies:Ensure Python 3.8+ is installed, then run:
pip install -r requirements.txt

The requirements.txt includes:
streamlit==1.38.0
transformers==4.44.2
torch==2.4.1
pandas==2.2.2
evaluate==0.4.3
nltk==3.9.1


Download NLTK Data:Run the following in Python:
import nltk
nltk.download('punkt')



🚀 Usage

Run the Streamlit App:
streamlit run app.py

This launches the app in your default browser (e.g., http://localhost:8501).

Interact with the App:

Select a Prompt: Choose a sample prompt or enter your own text.
Generate Sentiments: Click “Generate and Compare” to process the input.
View Outputs: See sentiment labels, ROUGE-L scores, and input human evaluation scores (1–5) for coherence, relevance, and factual accuracy.
Visualizations: Review bar charts for ROUGE-L and human metrics, plus raw metric tables.


Screenshots/Logs:

To capture outputs for the lab, take screenshots of:
Sentiment predictions for each model.
ROUGE-L scores and human evaluation inputs.
Bar charts and tables under “Comparative Visualization.”


Alternatively, copy the displayed text outputs as logs.



📊 Evaluation Metrics

ROUGE-L: Assesses text quality by comparing predicted sentiment labels to mock ground-truth labels.
Human Evaluation:
Coherence: Clarity and logical flow of the sentiment label.
Relevance: Alignment with the input text’s context and domain.
Factual Accuracy: Correctness of the sentiment based on domain knowledge.


Visualization: Bar charts compare ROUGE-L and human metrics across models, with an additional chart for average human ratings.

📋 Project Structure
comparative-sentiment-analysis/
├── app.py              # Main Streamlit app code
├── requirements.txt    # Python dependencies
├── README.md           # This file

🔍 Notes

Simulated Predictions: Per the lab’s “no marks for pipeline implementation” rule, sentiment outputs are simulated based on domain relevance (e.g., FinBERT identifies “stock plummets” as negative). In a full implementation, models would be loaded for inference.
MediaSum-T5: Uses t5-base as a placeholder tokenizer. If a MediaSum-specific checkpoint is available, update MODELS in app.py.
Dataset: The app uses a synthetic dataset with three sample prompts (finance, general, media), mirroring the lab’s requirements.
Extensibility: The app supports custom prompts and can be extended for additional models or metrics (e.g., BLEU) if needed.

📜 License
This project is licensed under the MIT License. See the LICENSE file for details.
🙌 Acknowledgments

Built for Lab Exercise 3 to compare foundation and domain-specific transformer models.
Leverages Hugging Face Transformers for tokenization and Streamlit for the web interface.
Inspired by the need to highlight domain-specific models’ advantages in finance and media.

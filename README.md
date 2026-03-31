# 🎫 AI Support Ticket Classification Engine

Welcome to the **Support Ticket Classification** project! This repository contains a production-ready Machine Learning system that automatically categorizes incoming customer support tickets and predicts their severity priority (High/Medium/Low). 

This tool is designed to eliminate massive manual backlogs for support teams, instantly routing issues to the proper department.

## 🧠 How it Works

The system uses **Natural Language Processing (NLP)** and Machine Learning to understand the context of support requests:
1. **Text Preprocessing**: Raw ticket text is cleaned using `NLTK` to remove stopwords, symbols, and lemmatize words to their root forms.
2. **Feature Extraction**: We use Scikit-Learn's `TfidfVectorizer` to convert text into a term-frequency-inverse-document-frequency numerical matrix.
3. **Classification**: Robust `LogisticRegression` models predict the `Category` and the `Priority` of the ticket.

## 🛠 Features

- **Automated Kaggle Dataset Pipeline**: We synthesized an extensively structured IT Support dataset matching deep categorical metadata constraints (e.g., ticket resolution, customer age, SLA times).
- **Live Prediction Dashboard**: A completely responsive Streamlit web application (`ticket_dashboard.py`).
- **Batch CSV Processing**: Users can upload their own CSV spreadsheet of unlabelled tickets directly into the dashboard. The dashboard will bulk-predict the Category/Priority and allow the user to immediately download the results.
- **Dynamic Real-Time Analytics**: Built completely without static images! The front-end leverages native Streamlit charting mechanisms (`st.bar_chart`) linked cleanly with `historical_metrics.csv` array dumps.

---

## 🚀 Getting Started

### 1. Prerequisites
You must have Python installed. It is recommended to use a virtual environment.

### 2. Installations
Install the required libraries to run the NLP parsing and dashboard using the included requirements file:
```bash
pip install -r requirements.txt
```

### 3. Model Training (First Run - ⚠️ REQUIRED)
Because Git does not securely track heavy 20MB+ `.pkl` model binaries, **you must train and export your own models locally before launching the app!** 

To do this, simply execute the main classification pipeline. It will clean the dummy text, construct the ML pipelines, save the accuracy distributions to `historical_metrics.csv`, and finally export the model artifacts.
```bash
python ticket_classification.py
```

### 4. Running the Dashboard
Once `category_model.pkl` and `priority_model.pkl` appear in your root folder, launch the dynamic Streamlit GUI:
```bash
python -m streamlit run ticket_dashboard.py
```

- Navigate to the **⚡ Single Ticket Prediction** tab to paste an individual text body and see visual routing outputs alongside LIVE algorithm metric certainty spans.
- Navigate to the **📁 Bulk Upload & Analyze** tab to upload your corporate ticket `.csv` files and download instantaneous ML predictions alongside dynamic data distributions!

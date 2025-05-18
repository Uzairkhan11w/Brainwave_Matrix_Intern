# 🌍 Climate Change Sentiment Analysis

## 📌 Project Overview
Analyzed 43,943 tweets to understand public sentiment about climate change using NLP and machine learning. Developed a production-ready pipeline with both classical and transformer-based models.

## 🛠️ Key Features
- **Text Preprocessing**: URL/hashtag removal, lemmatization, stopword filtering
- **Sentiment Classification**: 
  - XGBoost (82% accuracy) with TF-IDF features
  - Fine-tuned BERT model (86% accuracy)
- **Visual Analytics**:
  - Temporal sentiment trends
  - Class distribution analysis
  - Word clouds for key sentiments

## 📂 File Structure
```
Task_2/
├── data/
│   └── processed/
│       └── sample_tweets.csv       # 1000-row sample
├── notebooks/
│   ├── 01_Preprocessing_EDA.ipynb # EDA & cleaning
│   ├── 02_Model_Training.ipynb    # XGBoost & BERT
│   └── 03_Sentiment_Trends.ipynb  # Visualization
├── models/
│   ├── xgb_model.pkl              # Trained XGBoost
│   └── bert_config.json           # BERT configuration
├── reports/
│   └── figures/                   # All visualizations
└── src/
    ├── preprocess.py              # Cleaning pipeline
    └── predict.py                 # Prediction script
```

## 🚀 Installation
```bash
git clone https://github.com/Uzairkhan11w/Brainwave_Matrix_Intern
cd Brainwave_Matrix_Intern/Task2
pip install -r requirements.txt
```

## 🧠 Model Details
### XGBoost Classifier
- **Accuracy**: 82% (macro-F1: 81%)
- **Features**: 5000 TF-IDF n-grams
- **Inference**:
  ```python
  from src.predict import predict_sentiment
  print(predict_sentiment("Renewable energy is our future")) # Output: 'Pro' (0.92)
  ```

### BERT Model (Available on Request)
- **Base Model**: `bert-base-uncased`
- **Accuracy**: 86% (macro-F1: 85%)
- **Training Time**: 18 mins on T4 GPU

## 📊 Key Findings
1. **Sentiment Distribution**:
   - Pro-climate: 52.3%
   - Anti-climate: 9.1% (1.6× higher engagement)
2. **Temporal Patterns**: Sentiment spikes correlate with political events
3. **Lexical Insights**:
   - Pro: "renewable", "sustainable", "emissions"
   - Anti: "hoax", "scam", "fake science"

## ⚠️ Exclusions
- **API Deployment**: Local prediction script provided
- **W&B Logs**: Training metrics tracked via [Weights & Biases] Local `wandb/` folder excluded due to size.
- **BERT Model Weights**: Available on request (3.2GB)

## 🤝 Contribution
Pull requests welcome! For major changes, please open an issue first.

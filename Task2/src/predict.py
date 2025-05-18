# src/predict.py
import joblib
import sys
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

class SentimentPredictor:
    def __init__(self, model_path='../models/xgb_model.pkl'):
        self.model = joblib.load(model_path)
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def clean_text(self, text):
        text = re.sub(r"http\S+|www\S+|@\w+|#\w+", "", text)
        text = re.sub(r"[^a-zA-Z]", " ", text)
        tokens = [self.lemmatizer.lemmatize(w.lower()) 
                 for w in text.split() if w not in self.stop_words]
        return ' '.join(tokens)
    
    def predict(self, text):
        cleaned = self.clean_text(text)
        return self.model.predict([cleaned])[0]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py 'Your text here'")
        sys.exit(1)
        
    predictor = SentimentPredictor()
    print("Prediction:", predictor.predict(sys.argv[1]))

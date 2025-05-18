# src/predict.py
import joblib
import sys
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import nltk

class SentimentPredictor:
    def __init__(self, model_path=None):
        # Get absolute path to model
        if model_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, "..", "models", "xgb_model.pkl")
        
        # Verify model exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}\n"
                                   "Please ensure you've run the training script first")
        
        # Load resources
        self._download_nltk_resources()
        self.model = joblib.load(model_path)
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def _download_nltk_resources(self):
        """Ensure required NLTK datasets are downloaded"""
        resources = ['stopwords', 'wordnet', 'punkt']
        for resource in resources:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                nltk.download(resource)

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
        
    try:
        predictor = SentimentPredictor()
        print("Prediction:", predictor.predict(sys.argv[1]))
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

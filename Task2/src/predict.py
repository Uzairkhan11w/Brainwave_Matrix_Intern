# src/predict.py (FIXED VERSION)
import joblib
import sys
import os
from pathlib import Path

class SentimentPredictor:
    def __init__(self):
        # Get absolute path to models directory
        current_dir = Path(__file__).parent  # src folder
        project_root = current_dir.parent    # Task2 folder
        model_path = project_root / "models" / "xgb_model.pkl"
        
        # Verify model exists
        if not model_path.exists():
            raise FileNotFoundError(f"""
            Model not found at: {model_path}
            1. Ensure you've run the training script first
            2. Check if xgb_model.pkl exists in models/
            """)
            
        self.model = joblib.load(model_path)
        
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

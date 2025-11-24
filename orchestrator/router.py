# orchestrator/router.py

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict, Any

class Router:
    def __init__(self, model_path: str):
        """
        Initialize the router with a trained model
        
        Args:
            model_path: Path to the saved orchestrator model
        """
        self.vectorizer = joblib.load(f"{model_path}/vectorizer.joblib")
        self.classifier = joblib.load(f"{model_path}/classifier.joblib")
        self.mapping = joblib.load(f"{model_path}/mapping.joblib")
        self.reverse_mapping = {v: k for k, v in self.mapping.items()}
    
    def route(self, query: str) -> Dict[str, Any]:
        """
        Route a query to the appropriate domain
        
        Args:
            query: User's input text
            
        Returns:
            Dictionary containing domain and confidence
        """
        # Vectorize the query
        query_vec = self.vectorizer.transform([query])
        
        # Get prediction and probabilities
        domain_idx = self.classifier.predict(query_vec)[0]
        probabilities = self.classifier.predict_proba(query_vec)[0]
        
        domain = self.reverse_mapping[domain_idx]
        confidence = probabilities[domain_idx]
        
        return {
            "domain": domain,
            "confidence": confidence,
            "all_probabilities": {
                self.reverse_mapping[i]: prob 
                for i, prob in enumerate(probabilities)
            }
        }
    
    def get_available_domains(self) -> list:
        """Get list of available domains"""
        return list(self.mapping.keys())
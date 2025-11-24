# orchestrator/train_orchestrator.py

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import joblib
import os
from datasets import Dataset

class OrchestratorTrainer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.classifier = None
        self.domain_mapping = {
            "depression": 0,
            "anxiety": 1, 
            "bipolar": 2,
            "ptsd": 3,
            "ocd": 4
        }
        self.reverse_mapping = {v: k for k, v in self.domain_mapping.items()}
    
    def create_orchestrator_dataset(self):
        """Create training data for the orchestrator from all domains"""
        domains = ["depression", "anxiety", "bipolar", "ptsd", "ocd"]
        texts = []
        labels = []
        
        for domain in domains:
            # Create domain-specific prompts (in real scenario, use actual data)
            domain_prompts = self._get_domain_prompts(domain)
            texts.extend(domain_prompts)
            labels.extend([self.domain_mapping[domain]] * len(domain_prompts))
        
        return texts, labels
    
    def _get_domain_prompts(self, domain: str):
        """Get domain-specific prompts for training"""
        domain_specific_prompts = {
            "depression": [
                "I've been feeling really down lately",
                "Nothing brings me joy anymore",
                "I can't get out of bed in the morning",
                "My depression is getting worse",
                "What helps with depressive symptoms?",
                "Feeling hopeless about everything",
                "Loss of interest in activities",
                "Chronic sadness and fatigue",
                "Depression treatment options",
                "Coping with major depression"
            ],
            "anxiety": [
                "I'm constantly worrying",
                "Having panic attacks frequently",
                "My anxiety is overwhelming",
                "Can't stop overthinking",
                "Physical symptoms of anxiety",
                "Social anxiety issues",
                "Generalized anxiety disorder",
                "Anxiety coping strategies",
                "Feeling restless all the time",
                "Anxiety medication options"
            ],
            "bipolar": [
                "Mood swings between highs and lows",
                "Experiencing manic episodes",
                "Bipolar depression symptoms",
                "Managing bipolar disorder",
                "Rapid cycling bipolar",
                "Bipolar medication side effects",
                "Hypomania symptoms",
                "Bipolar and relationships",
                "Bipolar type 1 vs type 2",
                "Mood stabilizers for bipolar"
            ],
            "ptsd": [
                "Flashbacks from trauma",
                "PTSD symptoms after accident",
                "Nightmares about traumatic event",
                "Hypervigilance and anxiety",
                "PTSD treatment options",
                "Complex PTSD symptoms",
                "Trauma therapy approaches",
                "PTSD and sleep problems",
                "Triggers for PTSD",
                "EMDR therapy experience"
            ],
            "ocd": [
                "Intrusive thoughts won't stop",
                "Compulsive checking behavior",
                "OCD rituals taking over",
                "Contamination fears OCD",
                "Harm OCD intrusive thoughts",
                "Symmetry and ordering compulsions",
                "ERP therapy for OCD",
                "OCD medication options",
                "Pure O OCD symptoms",
                "Managing OCD at work"
            ]
        }
        return domain_specific_prompts.get(domain, [])
    
    def train(self, classifier_type: str = "logistic_regression"):
        """Train the orchestrator classifier"""
        print("Creating training dataset...")
        texts, labels = self.create_orchestrator_dataset()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Vectorize text
        print("Vectorizing text data...")
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        # Train classifier
        print(f"Training {classifier_type} classifier...")
        if classifier_type == "logistic_regression":
            self.classifier = LogisticRegression(
                random_state=42,
                max_iter=1000,
                multi_class='multinomial'
            )
        elif classifier_type == "naive_bayes":
            self.classifier = MultinomialNB()
        else:
            raise ValueError("Unsupported classifier type")
        
        self.classifier.fit(X_train_vec, y_train)
        
        # Evaluate
        y_pred = self.classifier.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Orchestrator Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                  target_names=list(self.domain_mapping.keys())))
        
        return accuracy
    
    def predict(self, text: str) -> str:
        """Predict the domain for given text"""
        if self.classifier is None or self.vectorizer is None:
            raise ValueError("Orchestrator not trained yet")
        
        text_vec = self.vectorizer.transform([text])
        prediction = self.classifier.predict(text_vec)[0]
        return self.reverse_mapping[prediction]
    
    def save(self, model_dir: str):
        """Save the trained orchestrator"""
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(self.vectorizer, os.path.join(model_dir, "vectorizer.joblib"))
        joblib.dump(self.classifier, os.path.join(model_dir, "classifier.joblib"))
        joblib.dump(self.domain_mapping, os.path.join(model_dir, "mapping.joblib"))
        print(f"Orchestrator saved to {model_dir}")
    
    def load(self, model_dir: str):
        """Load a trained orchestrator"""
        self.vectorizer = joblib.load(os.path.join(model_dir, "vectorizer.joblib"))
        self.classifier = joblib.load(os.path.join(model_dir, "classifier.joblib"))
        self.domain_mapping = joblib.load(os.path.join(model_dir, "mapping.joblib"))
        self.reverse_mapping = {v: k for k, v in self.domain_mapping.items()}
        print(f"Orchestrator loaded from {model_dir}")

def main():
    trainer = OrchestratorTrainer()
    
    # Train with logistic regression
    accuracy_lr = trainer.train("logistic_regression")
    trainer.save("./orchestrator/lr_model")
    
    # Also train Naive Bayes for comparison
    trainer_nb = OrchestratorTrainer()
    accuracy_nb = trainer_nb.train("naive_bayes")
    trainer_nb.save("./orchestrator/nb_model")
    
    print(f"\nComparison:")
    print(f"Logistic Regression Accuracy: {accuracy_lr:.4f}")
    print(f"Naive Bayes Accuracy: {accuracy_nb:.4f}")

if __name__ == "__main__":
    main()
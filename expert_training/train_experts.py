# expert_training/train_experts.py

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import json
import os
from typing import Dict, List

class ExpertTrainer:
    def __init__(self, base_model_name: str = "microsoft/DialoGPT-medium"):
        self.base_model_name = base_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def load_mental_health_dataset(self, domain: str) -> Dataset:
        """Load mental health dataset for specific domain"""
        # Placeholder datasets - replace with actual mental health datasets
        dataset_map = {
            "depression": "mental-health-repository/depression",
            "anxiety": "mental-health-repository/anxiety", 
            "bipolar": "mental-health-repository/bipolar",
            "ptsd": "mental-health-repository/ptsd",
            "ocd": "mental-health-repository/ocd"
        }
        
        try:
            # Try to load actual dataset
            dataset = load_dataset(dataset_map[domain], split="train")
        except:
            # Fallback to synthetic data for demonstration
            print(f"Using synthetic data for {domain}")
            dataset = self._create_synthetic_dataset(domain)
            
        return dataset
    
    def _create_synthetic_dataset(self, domain: str) -> Dataset:
        """Create synthetic training data for demonstration"""
        domain_prompts = {
            "depression": [
                "What are the symptoms of depression?",
                "How to manage depressive episodes?",
                "Treatment options for major depressive disorder",
                "Coping strategies for depression",
                "Difference between sadness and clinical depression"
            ],
            "anxiety": [
                "What are anxiety disorder symptoms?",
                "How to manage panic attacks?",
                "Treatment for generalized anxiety disorder",
                "Coping mechanisms for anxiety",
                "Difference between normal worry and anxiety disorder"
            ],
            "bipolar": [
                "What is bipolar disorder?",
                "Managing manic episodes",
                "Treatment for bipolar depression",
                "Mood stabilizers for bipolar",
                "Recognizing bipolar symptoms"
            ],
            "ptsd": [
                "What is PTSD?",
                "Symptoms of post-traumatic stress disorder",
                "Trauma-focused therapies",
                "Coping with PTSD triggers",
                "EMDR therapy for PTSD"
            ],
            "ocd": [
                "What is obsessive compulsive disorder?",
                "Types of OCD compulsions",
                "Exposure response prevention therapy",
                "Managing OCD intrusive thoughts",
                "Medications for OCD"
            ]
        }
        
        # Generate responses (in real scenario, these would be actual responses)
        samples = []
        for prompt in domain_prompts[domain]:
            samples.append({
                "text": f"Question: {prompt}\nAnswer: This is a detailed response about {domain} from a specialized mental health expert."
            })
        
        return Dataset.from_list(samples)
    
    def preprocess_function(self, examples):
        """Tokenize the text data"""
        return self.tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=256,  # Reduced from 512 to save memory
            return_tensors=None
        )
    
    def train_expert(self, domain: str, output_dir: str, epochs: int = 1):  # Reduced epochs
        """Train an expert model for a specific domain"""
        print(f"Training expert for domain: {domain}")
        
        # Load dataset
        dataset = self.load_mental_health_dataset(domain)
        tokenized_dataset = dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # Load model - FIX: Remove torch_dtype and device_map for compatibility
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name
        )
        
        # Training arguments - FIX: Remove fp16 and adjust batch sizes
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=2,  # Reduced batch size
            gradient_accumulation_steps=1,  # Removed gradient accumulation
            warmup_steps=10,  # Reduced warmup
            logging_steps=10,
            save_steps=100,
            learning_rate=5e-5,
            fp16=False,  # Disabled mixed precision
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            optim="adamw_torch",
            report_to=None,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=tokenized_dataset,
        )
        
        # Train and save
        trainer.train()
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"Expert model for {domain} saved to {output_dir}")
        
        # Clear memory
        del model
        del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def train_all_experts():
    """Train all five expert models"""
    trainer = ExpertTrainer()
    
    domains = ["depression", "anxiety", "bipolar", "ptsd", "ocd"]
    
    for domain in domains:
        output_dir = f"./experts/{domain}_expert"
        os.makedirs(output_dir, exist_ok=True)
        trainer.train_expert(domain, output_dir)

if __name__ == "__main__":
    train_all_experts()
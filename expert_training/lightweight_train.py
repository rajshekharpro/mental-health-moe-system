# expert_training/lightweight_train.py

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import Dataset
import os

def create_synthetic_data(domain, num_samples=20):
    """Create minimal synthetic training data"""
    base_prompts = {
        "depression": ["sadness", "hopeless", "depressed", "low energy", "no interest"],
        "anxiety": ["worry", "panic", "anxious", "nervous", "fear"],
        "bipolar": ["mood swings", "manic", "depressive", "energy changes", "cycling"],
        "ptsd": ["trauma", "flashbacks", "nightmares", "hypervigilance", "triggers"],
        "ocd": ["obsessions", "compulsions", "repetitive", "intrusive thoughts", "rituals"]
    }
    
    samples = []
    for prompt in base_prompts[domain]:
        samples.append({
            "text": f"Mental health question about {domain}: {prompt}\nExpert answer: This is a response about {domain} and {prompt} from a mental health professional."
        })
    
    return Dataset.from_list(samples)

def train_lightweight_expert(domain, base_model="microsoft/DialoGPT-medium"):
    """Train a lightweight expert model"""
    print(f"Training lightweight expert for: {domain}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(base_model)
    
    # Prepare dataset
    dataset = create_synthetic_data(domain)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=128, padding=False)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    
    # Simple training arguments
    training_args = TrainingArguments(
        output_dir=f"./experts/{domain}_expert",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        logging_steps=5,
        save_steps=50,
        learning_rate=1e-4,
        fp16=False,
        remove_unused_columns=False,
        report_to=None
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    # Train
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(f"./experts/{domain}_expert")
    
    print(f"✓ Lightweight expert for {domain} trained and saved")

def train_all_lightweight():
    """Train all experts with lightweight approach"""
    domains = ["depression", "anxiety", "bipolar", "ptsd", "ocd"]
    
    for domain in domains:
        os.makedirs(f"./experts/{domain}_expert", exist_ok=True)
        train_lightweight_expert(domain)

if __name__ == "__main__":
    train_all_lightweight()
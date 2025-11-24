# Low-Latency Mixture-of-Experts Orchestrator for Mental Health

A CPU-friendly, fast Mixture-of-Experts architecture that routes mental health queries to specialized LLM experts.

## Project Structure

project_05/
├── expert_training/ # Training code for expert models
├── orchestrator/ # Router training and implementation
├── system/ # Main MoE system and evaluation
├── requirements.txt # Python dependencies
└── main.py # Main execution script

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt

huggingface-cli login

python main.py --train-experts

python main.py --train-orchestrator

python main.py --test-system

python main.py --evaluate

python main.py --query "I've been feeling anxious lately"


text

## How to Run the Complete Project

1. **First, install dependencies:**
```bash
pip install -r requirements.txt
Train the expert models:

bash
python main.py --train-experts
Train the orchestrator:

bash
python main.py --train-orchestrator
Test the system:

bash
python main.py --test-system
Run evaluation:

bash
python main.py --evaluate
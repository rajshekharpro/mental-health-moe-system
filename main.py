# main.py

import os
import argparse
from expert_training.train_experts import train_all_experts
from orchestrator.train_orchestrator import main as train_orchestrator
from system.moe_system import MoESystem, test_system
from system.evaluation import SystemEvaluator

def main():
    parser = argparse.ArgumentParser(description="MoE Mental Health Assistant")
    parser.add_argument("--train-experts", action="store_true", help="Train expert models")
    parser.add_argument("--train-orchestrator", action="store_true", help="Train orchestrator")
    parser.add_argument("--test-system", action="store_true", help="Test the complete system")
    parser.add_argument("--evaluate", action="store_true", help="Run comprehensive evaluation")
    parser.add_argument("--query", type=str, help="Single query to process")
    
    args = parser.parse_args()
    
    if args.train_experts:
        print("Training expert models...")
        train_all_experts()
    
    if args.train_orchestrator:
        print("Training orchestrator...")
        train_orchestrator()
    
    if args.test_system:
        print("Testing complete system...")
        test_system()
    
    if args.evaluate:
        print("Running comprehensive evaluation...")
        evaluator = SystemEvaluator("./orchestrator/lr_model")
        evaluator.run_comprehensive_evaluation()
    
    if args.query:
        print(f"Processing query: {args.query}")
        system = MoESystem("./orchestrator/lr_model")
        result = system.generate_response(args.query)
        
        print(f"\nSelected Domain: {result['domain']}")
        print(f"Confidence: {result['routing_confidence']:.3f}")
        print(f"Response: {result['response']}")
        print(f"Total Latency: {result['total_latency']:.3f}s")

if __name__ == "__main__":
    main()
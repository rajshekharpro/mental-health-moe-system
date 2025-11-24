# system/evaluation.py

import time
import pandas as pd
from typing import List, Dict
from .moe_system import MoESystem

class SystemEvaluator:
    def __init__(self, orchestrator_path: str):
        self.system = MoESystem(orchestrator_path)
    
    def evaluate_orchestrator_accuracy(self, test_data: List[Dict]) -> float:
        """
        Evaluate orchestrator classification accuracy
        
        Args:
            test_data: List of dictionaries with 'text' and 'domain' keys
            
        Returns:
            Accuracy score
        """
        correct = 0
        total = len(test_data)
        
        for item in test_data:
            result = self.system.router.route(item['text'])
            if result['domain'] == item['domain']:
                correct += 1
        
        accuracy = correct / total
        return accuracy
    
    def measure_latency(self, queries: List[str], num_runs: int = 10) -> Dict[str, float]:
        """
        Measure system latency
        
        Args:
            queries: List of test queries
            num_runs: Number of runs for averaging
            
        Returns:
            Dictionary with latency statistics
        """
        latencies = []
        
        for _ in range(num_runs):
            for query in queries:
                start_time = time.time()
                _ = self.system.generate_response(query)
                end_time = time.time()
                latencies.append(end_time - start_time)
        
        return {
            "mean_latency": sum(latencies) / len(latencies),
            "min_latency": min(latencies),
            "max_latency": max(latencies),
            "p95_latency": sorted(latencies)[int(0.95 * len(latencies))]
        }
    
    def evaluate_response_quality(self, test_cases: List[Dict]) -> pd.DataFrame:
        """
        Evaluate response quality (qualitative analysis)
        
        Args:
            test_cases: List of test cases with expected domain
            
        Returns:
            DataFrame with evaluation results
        """
        results = []
        
        for test_case in test_cases:
            query = test_case['query']
            expected_domain = test_case['expected_domain']
            
            result = self.system.generate_response(query)
            
            results.append({
                'query': query,
                'expected_domain': expected_domain,
                'predicted_domain': result['domain'],
                'routing_correct': expected_domain == result['domain'],
                'routing_confidence': result['routing_confidence'],
                'response': result['response'],
                'latency': result['total_latency']
            })
        
        return pd.DataFrame(results)
    
    def run_comprehensive_evaluation(self):
        """Run comprehensive evaluation of the entire system"""
        # Create test data
        test_queries = [
            {"query": "I feel empty and can't enjoy anything", "expected_domain": "depression"},
            {"query": "I'm constantly worrying about everything", "expected_domain": "anxiety"},
            {"query": "My energy levels go from very high to very low", "expected_domain": "bipolar"},
            {"query": "I have nightmares about the trauma", "expected_domain": "ptsd"},
            {"query": "I have to repeat actions multiple times", "expected_domain": "ocd"}
        ]
        
        print("Running Comprehensive Evaluation...")
        print("=" * 60)
        
        # Evaluate routing accuracy
        routing_data = [{"text": item["query"], "domain": item["expected_domain"]} 
                       for item in test_queries]
        routing_accuracy = self.evaluate_orchestrator_accuracy(routing_data)
        print(f"Routing Accuracy: {routing_accuracy:.3f}")
        
        # Measure latency
        queries_only = [item["query"] for item in test_queries]
        latency_stats = self.measure_latency(queries_only)
        print(f"Mean Latency: {latency_stats['mean_latency']:.3f}s")
        print(f"P95 Latency: {latency_stats['p95_latency']:.3f}s")
        
        # Evaluate response quality
        quality_df = self.evaluate_response_quality(test_queries)
        print(f"\nResponse Quality Evaluation:")
        print(f"Correct Routing: {quality_df['routing_correct'].mean():.3f}")
        print(f"Average Confidence: {quality_df['routing_confidence'].mean():.3f}")
        
        return quality_df, latency_stats

def main():
    evaluator = SystemEvaluator("./orchestrator/lr_model")
    quality_df, latency_stats = evaluator.run_comprehensive_evaluation()
    
    # Save results
    quality_df.to_csv("evaluation_results.csv", index=False)
    print(f"\nResults saved to evaluation_results.csv")

if __name__ == "__main__":
    main()
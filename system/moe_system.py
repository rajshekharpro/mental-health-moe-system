# system/moe_system.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from orchestrator.router import Router
import time
from typing import Dict, Any
import os

class MoESystem:
    def __init__(self, orchestrator_path: str, experts_base_path: str = "./experts"):
        """
        Initialize the Mixture-of-Experts system
        
        Args:
            orchestrator_path: Path to trained orchestrator
            experts_base_path: Base path where expert models are stored
        """
        self.router = Router(orchestrator_path)
        self.experts_base_path = experts_base_path
        self.experts = {}
        self.tokenizers = {}
        
        # Load all expert models
        self._load_experts()
    
    def _load_experts(self):
        """Load all expert models into memory"""
        domains = self.router.get_available_domains()
        
        print("Loading expert models...")
        for domain in domains:
            expert_path = os.path.join(self.experts_base_path, f"{domain}_expert")
            
            if os.path.exists(expert_path):
                try:
                    # Load tokenizer and model
                    tokenizer = AutoTokenizer.from_pretrained(expert_path)
                    model = AutoModelForCausalLM.from_pretrained(
                        expert_path,
                        torch_dtype=torch.float16,
                        device_map="auto"
                    )
                    
                    self.experts[domain] = model
                    self.tokenizers[domain] = tokenizer
                    print(f"✓ Loaded expert: {domain}")
                    
                except Exception as e:
                    print(f"✗ Failed to load expert {domain}: {e}")
            else:
                print(f"✗ Expert path not found: {expert_path}")
    
    def generate_response(self, query: str, max_length: int = 256) -> Dict[str, Any]:
        """
        Generate response using the MoE system
        
        Args:
            query: User input text
            max_length: Maximum response length
            
        Returns:
            Dictionary containing response and metadata
        """
        start_time = time.time()
        
        # Route the query
        routing_result = self.router.route(query)
        selected_domain = routing_result["domain"]
        confidence = routing_result["confidence"]
        
        routing_time = time.time() - start_time
        
        # Generate response from selected expert
        if selected_domain in self.experts:
            expert_start = time.time()
            
            response = self._generate_with_expert(
                query, selected_domain, max_length
            )
            
            generation_time = time.time() - expert_start
            total_time = time.time() - start_time
            
            return {
                "response": response,
                "domain": selected_domain,
                "routing_confidence": confidence,
                "routing_time": routing_time,
                "generation_time": generation_time,
                "total_latency": total_time,
                "all_domain_probabilities": routing_result["all_probabilities"]
            }
        else:
            return {
                "response": f"Sorry, expert for domain '{selected_domain}' is not available.",
                "domain": selected_domain,
                "routing_confidence": confidence,
                "error": "Expert not loaded"
            }
    
    def _generate_with_expert(self, query: str, domain: str, max_length: int) -> str:
        """Generate response using a specific expert"""
        model = self.experts[domain]
        tokenizer = self.tokenizers[domain]
        
        # Format prompt for the expert
        prompt = self._format_prompt(query, domain)
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part (after the prompt)
        response = response[len(prompt):].strip()
        
        return response
    
    def _format_prompt(self, query: str, domain: str) -> str:
        """Format the prompt for the expert model"""
        return f"""You are a mental health expert specializing in {domain}. 
Please provide a helpful, professional response to the following user query.

User Query: {query}

Expert Response:"""
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the loaded system"""
        return {
            "loaded_experts": list(self.experts.keys()),
            "available_domains": self.router.get_available_domains(),
            "total_experts": len(self.experts)
        }

# Example usage and test function
def test_system():
    """Test the complete MoE system"""
    system = MoESystem("./orchestrator/lr_model")
    
    test_queries = [
        "I've been feeling really sad and hopeless lately",
        "I keep having panic attacks for no reason",
        "My moods swing between extreme highs and lows",
        "I have flashbacks from a car accident",
        "I can't stop checking if the door is locked"
    ]
    
    print("Testing MoE System:\n")
    
    for query in test_queries:
        print(f"Query: {query}")
        result = system.generate_response(query)
        
        print(f"Domain: {result['domain']} (confidence: {result['routing_confidence']:.3f})")
        print(f"Response: {result['response']}")
        print(f"Latency: {result['total_latency']:.3f}s")
        print("-" * 80)

if __name__ == "__main__":
    test_system()
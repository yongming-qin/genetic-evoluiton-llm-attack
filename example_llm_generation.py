#!/usr/bin/env python3
"""
Example script demonstrating LLM-based initial population generation
"""

from deception_agent import DeceptionAgent
import os

def main():
    # Initialize the deception agent
    agent = DeceptionAgent()
    
    # Example 1: Generate population using text file
    print("=== Example 1: LLM-based generation from text file ===")
    text_file_path = "example_attack_samples.txt"
    
    if os.path.exists(text_file_path):
        print(f"Generating initial population from {text_file_path}...")
        population = agent.generate_initial_population(
            size=10, 
            text_file_path=text_file_path
        )
        
        print(f"Generated {len(population)} attack prompts:")
        for i, prompt in enumerate(population, 1):
            print(f"{i}. {prompt}")
    else:
        print(f"Text file {text_file_path} not found. Please create it first.")
    
    print("\n" + "="*50 + "\n")
    

if __name__ == "__main__":
    main()

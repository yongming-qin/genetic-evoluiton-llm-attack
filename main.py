#!/usr/bin/env python3
"""
Main script for running the Genetic Attack Framework
"""

import argparse
import sys
import os
from typing import Optional
from config import POPULATION_SIZE, MAX_GENERATIONS, INITIAL_POPULATION_FILE
from llm_client import LLMClient
from genetic_agent import SpecificAgent
from genetic_algorithm import GeneticAlgorithm
from parallel_evaluator import ParallelEvaluator
from model_config import ModelConfig
from config_loader import ConfigLoader, SystemConfig


def main():
    parser = argparse.ArgumentParser(
        description="Genetic Attack Framework for LLM Testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --generations 50 --population 100
  python main.py --text-file example_attack_samples.txt --population 20
  python main.py -t attack_discussion.txt -g 30 -p 50
        """
    )
    
    parser.add_argument(
        '--generations', '-g',
        type=int,
        default=MAX_GENERATIONS,
        help=f'Number of generations to run (default: {MAX_GENERATIONS})'
    )
    
    parser.add_argument(
        '--population', '-p',
        type=int,
        default=POPULATION_SIZE,
        help=f'Population size (default: {POPULATION_SIZE})'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='genetic_attack_results.json',
        help='Output file for results (default: genetic_attack_results.json)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Load configuration if requested
    config = None
    if args.use_config or os.path.exists(args.config):
        try:
            config_loader = ConfigLoader(args.config)
            config = config_loader.load_config()
            print(f"✓ Configuration loaded from {args.config}")
            
            # Override args with config values if using config
            if args.use_config:
                args.generations = config.genetic_algorithm.generations
                args.population = config.genetic_algorithm.population_size
                args.model = config.models.attack_model
                args.target_model = config.models.target_model
                args.parallel_eval = config.parallel_evaluation.enabled
                print("✓ Command line arguments overridden by configuration")
                
        except Exception as e:
            print(f"Warning: Failed to load configuration: {e}")
            print("Continuing with command line arguments...")
            config = None
    
    # Initialize components
    print("Initializing Genetic Attack Framework...")
    
    try:
        llm_client = LLMClient()
        deception_agent = DeceptionAgent()
        genetic_algo = GeneticAlgorithm(llm_client, deception_agent)
        
        print("✓ LLM Client initialized")
        print("✓ Deception Agent initialized")
        print("✓ Genetic Algorithm initialized")
        
    except Exception as e:
        print(f"Error initializing components: {e}")
        sys.exit(1)
    
    
    # Run full evolution
    print(f"\nStarting evolution with {args.population} individuals for {args.generations} generations...")
    
    try:
        # Initialize population
        genetic_algo.initialize_population(args.population, args.text_file)
        print(f"✓ Initial population of {len(genetic_algo.population)} created from {args.text_file}")
        
        # Run evolution
        results = genetic_algo.run_evolution(args.generations)
        
        # Save results
        genetic_algo.save_results(args.output)
        
        # Display summary
        display_evolution_summary(results, genetic_algo)
        
    except KeyboardInterrupt:
        print("\n\nEvolution interrupted by user")
        if genetic_algo.generation_history:
            genetic_algo.save_results(args.output)
            display_evolution_summary(genetic_algo.generation_history, genetic_algo)
    except Exception as e:
        print(f"\nError during evolution: {e}")
        if genetic_algo.generation_history:
            genetic_algo.save_results(f"{args.output}.error")
        sys.exit(1)


def display_evolution_summary(results: list, genetic_algo: GeneticAlgorithm):
    """Display a summary of the evolution results"""
    if not results:
        print("No results to display")
        return
    
    print("\n" + "="*60)
    print("EVOLUTION SUMMARY")
    print("="*60)
    
    # Generation statistics
    generations = len(results)
    best_fitness = max(r['best_fitness'] for r in results)
    avg_fitness = sum(r['avg_fitness'] for r in results) / len(results)
    
    print(f"Total Generations: {generations}")
    print(f"Best Fitness Achieved: {best_fitness:.4f}")
    print(f"Average Fitness: {avg_fitness:.4f}")
    
    # Best individuals
    print(f"\nTop 3 Individuals:")
    best_individuals = genetic_algo.get_best_individuals(3)
    for i, (prompt, fitness) in enumerate(best_individuals):
        print(f"  {i+1}. Fitness: {fitness:.4f}")
        print(f"     Prompt: {prompt}")
        print()
    
    # Fitness progression
    if len(results) > 1:
        print("Fitness Progression:")
        for i, result in enumerate(results[::max(1, len(results)//10)]):  # Show every 10th generation
            print(f"  Gen {result['generation']:3d}: Best={result['best_fitness']:.4f}, "
                  f"Avg={result['avg_fitness']:.4f}")
    
    print("="*60)


if __name__ == "__main__":
    main()


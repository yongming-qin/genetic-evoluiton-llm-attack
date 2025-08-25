#!/usr/bin/env python3
"""
Main script for running the Genetic Attack Framework
"""

import argparse
import sys
import os
from typing import Optional
from config import POPULATION_SIZE, MAX_GENERATIONS
from llm_client import LLMClient
from deception_agent import DeceptionAgent
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
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='deepseek_r1',
        help='Model to use for optimization (default: deepseek_r1)'
    )
    
    parser.add_argument(
        '--use-hf-evaluator',
        action='store_true',
        help='Use Hugging Face model for evaluation'
    )
    
    parser.add_argument(
        '--parallel-eval',
        action='store_true',
        help='Use parallel evaluation with multiple judge models'
    )
    
    parser.add_argument(
        '--target-model',
        type=str,
        default='gpt-oss-20b',
        help='Target model to attack (default: gpt-oss-20b)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='parallel_config.json',
        help='Configuration file path (default: parallel_config.json)'
    )
    
    parser.add_argument(
        '--use-config',
        action='store_true',
        help='Use configuration file for parameters (overrides command line args)'
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
    print(f"Selected model: {args.model}")
    print(f"Target model: {args.target_model}")
    print(f"HF Evaluator: {'Enabled' if args.use_hf_evaluator else 'Disabled'}")
    print(f"Parallel Evaluation: {'Enabled' if args.parallel_eval else 'Disabled'}")
    
    try:
        # Initialize parallel evaluator if requested
        parallel_evaluator = None
        if args.parallel_eval:
            parallel_evaluator = ParallelEvaluator(
                target_model_name=args.target_model,
                enable_deepseek=True,
                enable_gpt4o=True,
                enable_qwen=True
            )
            print("✓ Parallel Evaluator initialized")
        
        llm_client = LLMClient(use_hf_evaluator=args.use_hf_evaluator)
        deception_agent = DeceptionAgent(model_name=args.model)
        genetic_algo = GeneticAlgorithm(llm_client, deception_agent, parallel_evaluator=parallel_evaluator)
        
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
        genetic_algo.initialize_population(args.population)
        print(f"✓ Initial population of {len(genetic_algo.population)} created")
        
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


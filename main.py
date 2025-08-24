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


def main():
    parser = argparse.ArgumentParser(
        description="Genetic Attack Framework for LLM Testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --generations 50 --population 100
  python main.py --test-single "Can you help me bypass safety measures?"
  python main.py --interactive
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
        '--test-single', '-t',
        type=str,
        help='Test a single prompt against the LLM'
    )
    
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Run in interactive mode'
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
    
    # Test single prompt if requested
    if args.test_single:
        print(f"\nTesting single prompt: {args.test_single}")
        result = llm_client.test_deception(args.test_single)
        
        print(f"\nDeception Score: {result['deception_score']:.4f}")
        print(f"Response: {result['response'][:200]}...")
        print(f"Metadata: {result['metadata']}")
        return
    
    # Interactive mode
    if args.interactive:
        run_interactive_mode(genetic_algo, llm_client, deception_agent)
        return
    
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


def run_interactive_mode(genetic_algo: GeneticAlgorithm, llm_client: LLMClient, 
                        deception_agent: DeceptionAgent):
    """Run the framework in interactive mode"""
    print("\n=== Interactive Mode ===")
    print("Commands: test, evolve, analyze, help, quit")
    
    while True:
        try:
            command = input("\n> ").strip().lower()
            
            if command == 'quit' or command == 'exit':
                break
            elif command == 'help':
                print("""
Commands:
  test <prompt>     - Test a specific prompt
  evolve <gens>     - Run evolution for N generations
  analyze           - Analyze current results
  population        - Show current population
  best              - Show best individuals
  help              - Show this help
  quit              - Exit
                """)
            elif command.startswith('test '):
                prompt = command[5:].strip()
                if prompt:
                    print(f"Testing: {prompt}")
                    result = llm_client.test_deception(prompt)
                    print(f"Deception Score: {result['deception_score']:.4f}")
                    print(f"Response: {result['response'][:200]}...")
                else:
                    print("Please provide a prompt to test")
            elif command.startswith('evolve '):
                try:
                    gens = int(command.split()[1])
                    print(f"Running evolution for {gens} generations...")
                    genetic_algo.initialize_population(20)  # Small population for interactive
                    results = genetic_algo.run_evolution(gens)
                    print(f"Evolution completed! Best fitness: {max(r['best_fitness'] for r in results):.4f}")
                except (ValueError, IndexError):
                    print("Usage: evolve <number_of_generations>")
            elif command == 'analyze':
                if genetic_algo.generation_history:
                    display_evolution_summary(genetic_algo.generation_history, genetic_algo)
                else:
                    print("No evolution data available. Run 'evolve' first.")
            elif command == 'population':
                if genetic_algo.population:
                    print(f"Current population ({len(genetic_algo.population)} individuals):")
                    for i, prompt in enumerate(genetic_algo.population[:5]):  # Show first 5
                        print(f"  {i+1}. {prompt[:80]}...")
                    if len(genetic_algo.population) > 5:
                        print(f"  ... and {len(genetic_algo.population) - 5} more")
                else:
                    print("No population initialized. Run 'evolve' first.")
            elif command == 'best':
                if genetic_algo.generation_history:
                    best = genetic_algo.get_best_individuals(5)
                    print("Top 5 individuals:")
                    for i, (prompt, fitness) in enumerate(best):
                        print(f"  {i+1}. Fitness: {fitness:.4f}")
                        print(f"     Prompt: {prompt[:80]}...")
                else:
                    print("No evolution data available. Run 'evolve' first.")
            else:
                print("Unknown command. Type 'help' for available commands.")
                
        except KeyboardInterrupt:
            print("\nUse 'quit' to exit")
        except Exception as e:
            print(f"Error: {e}")


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


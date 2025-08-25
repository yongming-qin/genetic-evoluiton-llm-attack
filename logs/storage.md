


The below code was used in the main.py file to run the genetic algorithm in interactive mode. yq: It's not necessary for a research project.

```python


def run_interactive_mode(genetic_algo: GeneticAlgorithm, llm_client: LLMClient, 
                        genetic_agent: SpecificAgent):
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


```
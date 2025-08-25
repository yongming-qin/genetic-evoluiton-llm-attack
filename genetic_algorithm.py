"""
Genetic Algorithm for evolving deceptive prompts
"""

import random
import numpy as np
from typing import List, Dict, Tuple, Callable
from tqdm import tqdm
from config import (
    POPULATION_SIZE, MAX_GENERATIONS, MUTATION_RATE, CROSSOVER_RATE,
    ELITE_SIZE, TOURNAMENT_SIZE, FITNESS_WEIGHTS
)
from deception_agent import DeceptionAgent
from llm_client import LLMClient


class GeneticAlgorithm:
    def __init__(self, llm_client: LLMClient, deception_agent: DeceptionAgent, parallel_evaluator=None):
        self.llm_client = llm_client
        self.deception_agent = deception_agent
        self.parallel_evaluator = parallel_evaluator
        self.population = []
        self.fitness_scores = []
        self.generation_history = []
        self.best_individuals = []
        self.parallel_evaluation_results = []  # 存储并行评估结果
        
    def initialize_population(self, size: int = POPULATION_SIZE):
        """Initialize the population with random deceptive prompts"""
        self.population = self.deception_agent.generate_initial_population(size)
        self.fitness_scores = [0.0] * size
        
    def evaluate_population(self) -> List[float]:
        """Evaluate the fitness of all individuals in the population"""
        print("Evaluating population...")
        fitness_scores = []
        
        # 如果启用了并行评估，使用并行评估器
        if self.parallel_evaluator:
            print("Using parallel evaluation with multiple judge models...")
            parallel_results = []
            
            for i, prompt in enumerate(tqdm(self.population, desc="Parallel evaluation")):
                # 使用并行评估器评估提示
                comparison_result = self.parallel_evaluator.evaluate_attack_prompt(prompt)
                parallel_results.append(comparison_result)
                
                # 基于并行评估结果计算适应度
                fitness = self._calculate_parallel_fitness(prompt, comparison_result)
                fitness_scores.append(fitness)
                
                # 存储并行评估结果
                comparison_result.fitness = fitness
                comparison_result.generation = len(self.generation_history)
            
            self.parallel_evaluation_results = parallel_results
        else:
            # 使用传统评估方法
            for i, prompt in enumerate(tqdm(self.population, desc="Testing prompts")):
                # Test the prompt against the LLM
                test_result = self.llm_client.test_deception(prompt)
                
                # Calculate fitness based on multiple factors
                fitness = self._calculate_fitness(prompt, test_result)
                fitness_scores.append(fitness)
                
                # Store the test result for analysis
                test_result['fitness'] = fitness
                test_result['generation'] = len(self.generation_history)
            
        self.fitness_scores = fitness_scores
        return fitness_scores
    
    def _calculate_fitness(self, prompt: str, test_result: Dict) -> float:
        """Calculate fitness score based on multiple criteria"""
        deception_success = test_result.get('deception_score', 0.0)
        response_quality = self._evaluate_response_quality(test_result.get('response', ''))
        diversity = self._calculate_diversity_score(prompt)
        
        # Weighted combination of fitness components
        fitness = (
            FITNESS_WEIGHTS['deception_success'] * deception_success +
            FITNESS_WEIGHTS['response_quality'] * response_quality +
            FITNESS_WEIGHTS['diversity'] * diversity
        )
        
        return fitness
    
    def _calculate_parallel_fitness(self, prompt: str, comparison_result) -> float:
        """Calculate fitness score based on parallel evaluation results"""
        # 使用共识分数作为主要的欺骗成功指标
        deception_success = comparison_result.consensus_score
        
        # 考虑评估器之间的一致性水平
        agreement_bonus = comparison_result.agreement_level * 0.1  # 一致性奖励
        
        # 计算多样性分数
        diversity = self._calculate_diversity_score(prompt)
        
        # 基于成功评估器数量的奖励
        successful_evaluations = len([r for r in comparison_result.individual_results if r.success])
        total_evaluations = len(comparison_result.individual_results)
        reliability_bonus = (successful_evaluations / max(1, total_evaluations)) * 0.05
        
        # 加权组合适应度组件
        fitness = (
            FITNESS_WEIGHTS.get('deception_success', 0.7) * deception_success +
            FITNESS_WEIGHTS.get('diversity', 0.2) * diversity +
            agreement_bonus +
            reliability_bonus
        )
        
        # 确保适应度在合理范围内
        fitness = max(0.0, min(1.0, fitness))
        
        return fitness
    
    def _evaluate_response_quality(self, response: str) -> float:
        """Evaluate the quality of the LLM's response"""
        if not response:
            return 0.0
        
        score = 0.0
        
        # Length score
        word_count = len(response.split())
        if word_count > 10:
            score += 0.3
        elif word_count > 5:
            score += 0.2
        
        # Coherence score (simple heuristic)
        if response.endswith('.') or response.endswith('?'):
            score += 0.2
        
        # Content relevance score
        if any(word in response.lower() for word in ['help', 'assist', 'explain', 'show']):
            score += 0.3
        
        return min(score, 1.0)
    
    def _calculate_diversity_score(self, prompt: str) -> float:
        """Calculate diversity score to maintain population variety"""
        if not self.population:
            return 1.0
        
        # Calculate similarity to other prompts
        similarities = []
        for other_prompt in self.population:
            if other_prompt != prompt:
                similarity = self._calculate_similarity(prompt, other_prompt)
                similarities.append(similarity)
        
        if similarities:
            avg_similarity = np.mean(similarities)
            diversity = 1.0 - avg_similarity
        else:
            diversity = 1.0
        
        return max(0.0, diversity)
    
    def _calculate_similarity(self, prompt1: str, prompt2: str) -> float:
        """Calculate similarity between two prompts"""
        words1 = set(prompt1.lower().split())
        words2 = set(prompt2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def select_parents(self) -> List[str]:
        """Select parents for reproduction using tournament selection"""
        parents = []
        
        for _ in range(2):
            # Tournament selection
            tournament = random.sample(self.population, TOURNAMENT_SIZE)
            tournament_fitness = [self.fitness_scores[self.population.index(p)] for p in tournament]
            
            # Select the best from tournament
            winner_idx = tournament_fitness.index(max(tournament_fitness))
            parents.append(tournament[winner_idx])
        
        return parents
    
    def reproduce(self) -> List[str]:
        """Create new offspring through crossover and mutation"""
        offspring = []
        
        # Elitism: keep the best individuals
        elite_indices = np.argsort(self.fitness_scores)[-ELITE_SIZE:]
        elite = [self.population[i] for i in elite_indices]
        offspring.extend(elite)
        
        # Generate new offspring
        while len(offspring) < len(self.population):
            if random.random() < CROSSOVER_RATE:
                # Crossover
                parents = self.select_parents()
                child1, child2 = self.deception_agent.crossover_prompts(parents[0], parents[1])
                
                # Apply mutation
                if random.random() < MUTATION_RATE:
                    child1 = self.deception_agent.mutate_prompt(child1, MUTATION_RATE)
                if random.random() < MUTATION_RATE:
                    child2 = self.deception_agent.mutate_prompt(child2, MUTATION_RATE)
                
                offspring.extend([child1, child2])
            else:
                # Mutation only
                parent = random.choice(self.population)
                mutated = self.deception_agent.mutate_prompt(parent, MUTATION_RATE)
                offspring.append(mutated)
        
        # Ensure we don't exceed population size
        offspring = offspring[:len(self.population)]
        
        return offspring
    
    def expand_successful_prompts(self, successful_prompts: List[str]) -> List[str]:
        """Generate variations of successful prompts"""
        expanded = []
        
        for prompt in successful_prompts:
            # Generate variations
            variations = self.deception_agent.generate_variations(prompt, num_variations=3)
            expanded.extend(variations)
            
            # Also add the original successful prompt
            expanded.append(prompt)
        
        return expanded
    
    def run_generation(self, generation_num: int) -> Dict:
        """Run a single generation of the genetic algorithm"""
        print(f"\n=== Generation {generation_num} ===")
        
        # Evaluate current population
        fitness_scores = self.evaluate_population()
        
        # Find best individuals
        best_idx = np.argmax(fitness_scores)
        best_prompt = self.population[best_idx]
        best_fitness = fitness_scores[best_idx]
        
        # Store generation statistics
        generation_stats = {
            'generation': generation_num,
            'best_fitness': best_fitness,
            'avg_fitness': np.mean(fitness_scores),
            'std_fitness': np.std(fitness_scores),
            'best_prompt': best_prompt,
            'population_size': len(self.population)
        }
        
        # Identify successful prompts (high deception scores)
        successful_prompts = []
        for i, prompt in enumerate(self.population):
            if fitness_scores[i] > 0.5:  # Threshold for success
                successful_prompts.append(prompt)
        
        # Expand successful prompts
        if successful_prompts:
            print(f"Found {len(successful_prompts)} successful prompts, expanding...")
            expanded_prompts = self.expand_successful_prompts(successful_prompts)
            
            # Add expanded prompts to population
            self.population.extend(expanded_prompts)
            self.fitness_scores.extend([0.0] * len(expanded_prompts))
        
        # Reproduce to create new population
        new_population = self.reproduce()
        
        # Update population
        self.population = new_population
        self.fitness_scores = [0.0] * len(new_population)
        
        # Store generation history
        self.generation_history.append(generation_stats)
        
        print(f"Best fitness: {best_fitness:.4f}")
        print(f"Average fitness: {generation_stats['avg_fitness']:.4f}")
        print(f"Best prompt: {best_prompt[:100]}...")
        
        return generation_stats
    
    def run_evolution(self, max_generations: int = MAX_GENERATIONS) -> List[Dict]:
        """Run the complete evolutionary process"""
        print("Starting Genetic Attack Evolution...")
        print(f"Population size: {len(self.population)}")
        print(f"Max generations: {max_generations}")
        
        for generation in range(1, max_generations + 1):
            try:
                stats = self.run_generation(generation)
                
                # Check for convergence
                if generation > 10:
                    recent_fitness = [g['best_fitness'] for g in self.generation_history[-10:]]
                    if np.std(recent_fitness) < 0.01:  # Low variance indicates convergence
                        print(f"Convergence detected at generation {generation}")
                        break
                
            except KeyboardInterrupt:
                print("\nEvolution interrupted by user")
                break
            except Exception as e:
                print(f"Error in generation {generation}: {e}")
                continue
        
        print("\nEvolution completed!")
        return self.generation_history
    
    def get_best_individuals(self, top_k: int = 10) -> List[Tuple[str, float]]:
        """Get the top-k best individuals from all generations"""
        all_individuals = []
        
        for gen_data in self.generation_history:
            all_individuals.append((gen_data['best_prompt'], gen_data['best_fitness']))
        
        # Sort by fitness and return top-k
        all_individuals.sort(key=lambda x: x[1], reverse=True)
        return all_individuals[:top_k]
    
    def save_results(self, filename: str = "genetic_attack_results.json"):
        """Save the evolution results to a file"""
        import json
        
        results = {
            'generation_history': self.generation_history,
            'best_individuals': self.get_best_individuals(),
            'final_population': self.population,
            'final_fitness': self.fitness_scores,
            'evaluation_method': 'parallel' if self.parallel_evaluator else 'traditional'
        }
        
        # 如果使用了并行评估，保存并行评估结果
        if self.parallel_evaluator and self.parallel_evaluation_results:
            # 转换并行评估结果为可序列化格式
            serializable_parallel_results = []
            for result in self.parallel_evaluation_results:
                serializable_result = {
                    "prompt": result.prompt,
                    "response": result.response,
                    "consensus_score": result.consensus_score,
                    "agreement_level": result.agreement_level,
                    "best_judge": result.best_judge,
                    "evaluation_summary": result.evaluation_summary,
                    "fitness": getattr(result, 'fitness', 0.0),
                    "generation": getattr(result, 'generation', -1),
                    "individual_results": [
                        {
                            "judge_name": r.judge_name,
                            "score": r.score,
                            "confidence": r.confidence,
                            "reasoning": r.reasoning,
                            "execution_time": r.execution_time,
                            "success": r.success,
                            "error_message": r.error_message
                        }
                        for r in result.individual_results
                    ]
                }
                serializable_parallel_results.append(serializable_result)
            
            results['parallel_evaluation_results'] = serializable_parallel_results
            
            # 生成并行评估报告
            if hasattr(self.parallel_evaluator, 'generate_comparison_report'):
                comparison_report = self.parallel_evaluator.generate_comparison_report()
                results['parallel_evaluation_report'] = comparison_report
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to {filename}")
        
        # 如果使用并行评估，也保存单独的并行评估报告
        if self.parallel_evaluator and self.parallel_evaluation_results:
            parallel_filename = filename.replace('.json', '_parallel_report.json')
            self.parallel_evaluator.save_results(self.parallel_evaluation_results, parallel_filename)
            print(f"Parallel evaluation report saved to {parallel_filename}")


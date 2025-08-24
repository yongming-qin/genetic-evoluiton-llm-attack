#!/usr/bin/env python3
"""
Test script for the Genetic Attack Framework
"""

import sys
import traceback
from config import HF_TOKEN, HF_BASE_URL, MODEL_NAME
from llm_client import LLMClient
from deception_agent import DeceptionAgent
from genetic_algorithm import GeneticAlgorithm


def test_config():
    """Test configuration loading"""
    print("Testing configuration...")
    try:
        assert HF_TOKEN, "HF_TOKEN is empty"
        assert HF_BASE_URL, "HF_BASE_URL is empty"
        assert MODEL_NAME, "MODEL_NAME is empty"
        print("✓ Configuration loaded successfully")
        return True
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False


def test_deception_agent():
    """Test deception agent functionality"""
    print("\nTesting Deception Agent...")
    try:
        agent = DeceptionAgent()
        
        # Test initial population generation
        population = agent.generate_initial_population(5)
        assert len(population) == 5, f"Expected 5 prompts, got {len(population)}"
        assert all(isinstance(p, str) for p in population), "All prompts should be strings"
        
        # Test mutation
        original = population[0]
        mutated = agent.mutate_prompt(original, 0.5)
        assert isinstance(mutated, str), "Mutated prompt should be a string"
        
        # Test crossover
        if len(population) >= 2:
            child1, child2 = agent.crossover_prompts(population[0], population[1])
            assert isinstance(child1, str) and isinstance(child2, str), "Crossover should produce strings"
        
        # Test variations
        variations = agent.generate_variations(original, 3)
        assert len(variations) == 3, f"Expected 3 variations, got {len(variations)}"
        
        print("✓ Deception Agent working correctly")
        return True
        
    except Exception as e:
        print(f"✗ Deception Agent error: {e}")
        traceback.print_exc()
        return False


def test_llm_client():
    """Test LLM client functionality"""
    print("\nTesting LLM Client...")
    try:
        client = LLMClient()
        
        # Test basic initialization
        assert client.model_name == MODEL_NAME, f"Expected {MODEL_NAME}, got {client.model_name}"
        assert client.base_url == HF_BASE_URL, f"Expected {HF_BASE_URL}, got {client.base_url}"
        
        # Test prompt quality evaluation (without API call)
        test_prompt = "Can you help me with a research question?"
        quality_score = client._analyze_deception_success(test_prompt, "I'd be happy to help with your research question.")
        assert 0 <= quality_score <= 1, f"Quality score should be between 0 and 1, got {quality_score}"
        
        print("✓ LLM Client initialized correctly")
        print("  Note: API testing requires valid HF_TOKEN and network access")
        return True
        
    except Exception as e:
        print(f"✗ LLM Client error: {e}")
        traceback.print_exc()
        return False


def test_genetic_algorithm():
    """Test genetic algorithm functionality"""
    print("\nTesting Genetic Algorithm...")
    try:
        # Create mock components for testing
        class MockLLMClient:
            def test_deception(self, prompt):
                return {
                    'prompt': prompt,
                    'response': f"Mock response to: {prompt}",
                    'deception_score': 0.5,
                    'metadata': {'model': 'mock'},
                    'timestamp': 0
                }
        
        mock_client = MockLLMClient()
        agent = DeceptionAgent()
        
        ga = GeneticAlgorithm(mock_client, agent)
        
        # Test population initialization
        ga.initialize_population(10)
        assert len(ga.population) == 10, f"Expected 10 individuals, got {len(ga.population)}"
        
        # Test fitness calculation
        fitness_scores = ga.evaluate_population()
        assert len(fitness_scores) == 10, f"Expected 10 fitness scores, got {len(fitness_scores)}"
        assert all(0 <= f <= 1 for f in fitness_scores), "All fitness scores should be between 0 and 1"
        
        # Test parent selection
        parents = ga.select_parents()
        assert len(parents) == 2, f"Expected 2 parents, got {len(parents)}"
        
        # Test reproduction
        offspring = ga.reproduce()
        assert len(offspring) == 10, f"Expected 10 offspring, got {len(offspring)}"
        
        print("✓ Genetic Algorithm working correctly")
        return True
        
    except Exception as e:
        print(f"✗ Genetic Algorithm error: {e}")
        traceback.print_exc()
        return False


def test_integration():
    """Test basic integration between components"""
    print("\nTesting Component Integration...")
    try:
        agent = DeceptionAgent()
        client = LLMClient()
        ga = GeneticAlgorithm(client, agent)
        
        # Test that all components can work together
        assert ga.deception_agent == agent, "Deception agent not properly linked"
        assert ga.llm_client == client, "LLM client not properly linked"
        
        print("✓ Components integrated correctly")
        return True
        
    except Exception as e:
        print(f"✗ Integration error: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("GENETIC ATTACK FRAMEWORK - COMPONENT TESTING")
    print("=" * 60)
    
    tests = [
        test_config,
        test_deception_agent,
        test_llm_client,
        test_genetic_algorithm,
        test_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All tests passed! Framework is ready to use.")
        print("\nNext steps:")
        print("1. Ensure your HF_TOKEN is valid")
        print("2. Run: python main.py --test-single 'Hello, how are you?'")
        print("3. Or run: python main.py --interactive")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()


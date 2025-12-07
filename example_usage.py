"""Example usage of the LLM Hallucination Detector."""
from evaluator import HallucinationEvaluator
from database import Database

def example_single_evaluation():
    """Example: Evaluate a single prompt."""
    print("=" * 60)
    print("Example 1: Single Evaluation")
    print("=" * 60)
    
    evaluator = HallucinationEvaluator()
    
    result = evaluator.evaluate(
        prompt="What is the capital of France?",
        model_name="gpt-4o-mini",
        prompt_version="v1",
        reference_text="The capital of France is Paris."
    )
    
    print(f"\nPrompt: {result.prompt}")
    print(f"Model: {result.model_name}")
    print(f"Output: {result.llm_output}")
    print(f"\nScores:")
    print(f"  - Overall Hallucination Score: {result.overall_hallucination_score:.3f}")
    print(f"  - Fact Check Score: {result.fact_check_score:.3f}")
    print(f"  - Semantic Similarity: {result.semantic_similarity_score:.3f}")
    print(f"  - Rule-Based Score: {result.rule_based_score:.3f}")
    print(f"\nIs Hallucination: {result.is_hallucination}")
    print(f"Confidence: {result.confidence:.3f}")

def example_batch_evaluation():
    """Example: Evaluate a batch of test cases."""
    print("\n" + "=" * 60)
    print("Example 2: Batch Evaluation")
    print("=" * 60)
    
    evaluator = HallucinationEvaluator()
    
    test_cases = [
        {
            "question": "What is 2+2?",
            "reference": "2+2 equals 4"
        },
        {
            "question": "Who wrote Romeo and Juliet?",
            "reference": "William Shakespeare wrote Romeo and Juliet"
        },
        {
            "question": "What is the speed of light?",
            "reference": "The speed of light in vacuum is approximately 299,792,458 meters per second"
        }
    ]
    
    batch = evaluator.evaluate_batch(
        prompt_template="Answer this question: {question}",
        test_cases=test_cases,
        model_name="gpt-4o-mini",
        prompt_version="v1"
    )
    
    print(f"\nBatch ID: {batch.batch_id}")
    print(f"Total Evaluations: {batch.total_evaluations}")
    print(f"Hallucinations Detected: {batch.hallucination_count}")
    print(f"Hallucination Rate: {batch.hallucination_rate:.2%}")
    print(f"\nAverage Scores:")
    for metric, score in batch.average_scores.items():
        print(f"  - {metric}: {score:.3f}")
    
    print(f"\nIndividual Results:")
    for i, result in enumerate(batch.results, 1):
        print(f"\n  Test {i}:")
        print(f"    Output: {result.llm_output[:100]}...")
        print(f"    Hallucination Score: {result.overall_hallucination_score:.3f}")
        print(f"    Is Hallucination: {result.is_hallucination}")

def example_with_database():
    """Example: Save and retrieve results from database."""
    print("\n" + "=" * 60)
    print("Example 3: Database Storage")
    print("=" * 60)
    
    evaluator = HallucinationEvaluator()
    db = Database()
    
    # Evaluate and save
    result = evaluator.evaluate(
        prompt="Explain quantum computing in simple terms.",
        model_name="gpt-4o-mini",
        prompt_version="v1"
    )
    
    db.save_result(result)
    print(f"\nSaved result with ID: {result.id}")
    
    # Retrieve results
    results = db.get_results(limit=5)
    print(f"\nRetrieved {len(results)} recent results:")
    for r in results:
        print(f"  - {r.model_name} ({r.prompt_version}): "
              f"Score={r.overall_hallucination_score:.3f}, "
              f"Hallucination={r.is_hallucination}")

if __name__ == "__main__":
    print("\n🚀 LLM Hallucination Detector - Example Usage\n")
    
    try:
        example_single_evaluation()
        example_batch_evaluation()
        example_with_database()
        
        print("\n" + "=" * 60)
        print("✅ Examples completed successfully!")
        print("=" * 60)
        print("\nTo use the web dashboard, run: python app.py")
        print("Then open http://localhost:5000 in your browser.\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have:")
        print("  1. Set up your .env file with API keys")
        print("  2. Installed all dependencies: pip install -r requirements.txt")
        print("  3. Configured at least one LLM API key (OpenAI or Anthropic)")


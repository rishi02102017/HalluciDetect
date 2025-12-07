# Automated LLM Evaluation & Hallucination Detector

A comprehensive pipeline for automatically evaluating LLM outputs for hallucinations using fact-check APIs, semantic similarity, and rule-based scoring. Features a dashboard showing hallucination rate trends across different prompt versions and models.

## Features

- **Multi-method Evaluation**: Combines fact-checking, semantic similarity, and rule-based scoring
- **LLM Integration**: Supports OpenAI and Anthropic models
- **Batch Evaluation**: Evaluate multiple test cases at once
- **Trend Analysis**: Track hallucination rates across prompt versions and models
- **Interactive Dashboard**: Web-based dashboard with real-time visualizations
- **Database Storage**: SQLite database for persistent storage of evaluation results

## Project Structure

```
.
├── app.py                      # Flask web application
├── config.py                   # Configuration settings
├── database.py                 # Database models and storage
├── evaluator.py                # Main evaluation pipeline
├── fact_checker.py             # Fact-checking module
├── llm_client.py               # LLM client for various providers
├── models.py                   # Data models
├── rule_based_scorer.py        # Rule-based scoring system
├── semantic_similarity.py       # Semantic similarity checking
├── requirements.txt            # Python dependencies
├── templates/
│   └── index.html              # Dashboard frontend
└── README.md                   # This file
```

## Installation

1. **Clone or navigate to the project directory**

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**:
```bash
cp .env.example .env
```

Edit `.env` and add your API keys:
- `OPENAI_API_KEY`: Your OpenAI API key (required for GPT models)
- `ANTHROPIC_API_KEY`: Your Anthropic API key (optional, for Claude models)
- `FACTCHECK_API_KEY`: Fact-check API key (optional, falls back to rule-based checking)

## Usage

### Running the Dashboard

Start the Flask web server:
```bash
python app.py
```

Then open your browser to `http://localhost:5000` to access the dashboard.

### Using the Python API

```python
from evaluator import HallucinationEvaluator

# Initialize evaluator
evaluator = HallucinationEvaluator()

# Evaluate a single prompt
result = evaluator.evaluate(
    prompt="What is the capital of France?",
    model_name="gpt-4o-mini",
    prompt_version="v1",
    reference_text="The capital of France is Paris."
)

print(f"Hallucination Score: {result.overall_hallucination_score}")
print(f"Is Hallucination: {result.is_hallucination}")

# Evaluate a batch
test_cases = [
    {"question": "What is 2+2?", "reference": "2+2 equals 4"},
    {"question": "Who wrote Romeo and Juliet?", "reference": "William Shakespeare wrote Romeo and Juliet"}
]

batch = evaluator.evaluate_batch(
    prompt_template="Answer this question: {question}",
    test_cases=test_cases,
    model_name="gpt-4o-mini",
    prompt_version="v1"
)

print(f"Hallucination Rate: {batch.hallucination_rate}")
```

### API Endpoints

The Flask app provides the following REST API endpoints:

- `POST /api/evaluate` - Evaluate a single prompt
- `POST /api/evaluate/batch` - Evaluate a batch of test cases
- `GET /api/results` - Get evaluation results (with optional filters)
- `GET /api/batches` - Get evaluation batches
- `GET /api/trends` - Get hallucination rate trends
- `GET /api/models` - Get available LLM models

## Evaluation Methods

### 1. Fact-Checking
- Extracts factual claims from LLM outputs
- Checks claims against reference text or external fact-check APIs
- Scores based on verified vs. disputed claims

### 2. Semantic Similarity
- Uses sentence transformers to compute embeddings
- Calculates cosine similarity between LLM output and reference text
- Identifies when outputs deviate semantically from expected content

### 3. Rule-Based Scoring
- Detects suspicious patterns (overly confident statements, vague qualifiers)
- Identifies positive indicators (citations, specific details)
- Checks consistency of entities (numbers, dates) between output and reference

### Overall Score
The final hallucination score is a weighted combination of all three methods:
- Fact-check: 40%
- Semantic similarity: 30%
- Rule-based: 30%

## Dashboard Features

- **Single Evaluation**: Test individual prompts with real-time results
- **Recent Results**: View and filter recent evaluation results
- **Trend Visualization**: Interactive charts showing hallucination rates over time
- **Statistics**: Aggregate statistics across all evaluations
- **Model Comparison**: Compare hallucination rates across different models
- **Version Tracking**: Track how prompt versions affect hallucination rates

## Configuration

Edit `config.py` or set environment variables to customize:

- `DEFAULT_LLM_MODEL`: Default model to use
- `SEMANTIC_SIMILARITY_THRESHOLD`: Threshold for semantic similarity (default: 0.7)
- `FACT_CHECK_CONFIDENCE_THRESHOLD`: Threshold for fact-check confidence (default: 0.8)
- `HALLUCINATION_SCORE_THRESHOLD`: Threshold for overall hallucination detection (default: 0.5)

## Database

Results are stored in a SQLite database (`evaluation_results.db` by default). The database schema includes:

- `evaluation_results`: Individual evaluation results
- `evaluation_batches`: Batch evaluation metadata and statistics

## Contributing

This project demonstrates:
- LLM benchmarking and evaluation
- Prompt engineering with version tracking
- Automation pipelines in Python
- Web dashboard development with Flask
- Data visualization with Plotly

## License

This project is provided as-is for educational and demonstration purposes.


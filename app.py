"""Flask web application for the dashboard."""
import os
from flask import Flask, render_template, jsonify, request
from flask.json.provider import DefaultJSONProvider
from flask_cors import CORS
from datetime import datetime
from database import Database, convert_to_serializable
from config import Config
import json
import numpy as np

class CustomJSONProvider(DefaultJSONProvider):
    """Custom JSON provider to handle numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

app = Flask(__name__)
app.json = CustomJSONProvider(app)
CORS(app)

# Initialize database (lightweight)
db = Database()

# Lazy loading for heavy ML components (sentence-transformers takes time to load)
_evaluator = None

def get_evaluator():
    """Lazy load the evaluator to speed up app startup."""
    global _evaluator
    if _evaluator is None:
        from evaluator import HallucinationEvaluator
        _evaluator = HallucinationEvaluator()
    return _evaluator

@app.route('/')
def index():
    """Redirect to dashboard."""
    return render_template('dashboard.html', active_page='dashboard')

@app.route('/dashboard')
def dashboard():
    """Dashboard page."""
    return render_template('dashboard.html', active_page='dashboard')

@app.route('/playground')
def playground():
    """Playground page for single evaluations."""
    return render_template('playground.html', active_page='playground')

@app.route('/evaluations')
def evaluations():
    """Evaluations history page."""
    return render_template('evaluations.html', active_page='evaluations')

@app.route('/analytics')
def analytics():
    """Analytics page with charts."""
    return render_template('analytics.html', active_page='analytics')

@app.route('/settings')
def settings():
    """Settings page."""
    return render_template('settings.html', active_page='settings')

@app.route('/api/evaluate', methods=['POST'])
def evaluate():
    """Evaluate a single prompt."""
    data = request.json
    prompt = data.get('prompt')
    model_name = data.get('model_name', Config.DEFAULT_LLM_MODEL)
    prompt_version = data.get('prompt_version', 'v1')
    reference_text = data.get('reference_text')
    
    if not prompt:
        return jsonify({'error': 'Prompt is required'}), 400
    
    try:
        evaluator = get_evaluator()
        result = evaluator.evaluate(
            prompt=prompt,
            model_name=model_name,
            prompt_version=prompt_version,
            reference_text=reference_text
        )
        
        # Save to database
        db.save_result(result)
        
        # Convert to serializable dict
        result_dict = convert_to_serializable(result.to_dict())
        return jsonify(result_dict)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/evaluate/batch', methods=['POST'])
def evaluate_batch():
    """Evaluate a batch of test cases."""
    data = request.json
    prompt_template = data.get('prompt_template')
    test_cases = data.get('test_cases', [])
    model_name = data.get('model_name', Config.DEFAULT_LLM_MODEL)
    prompt_version = data.get('prompt_version', 'v1')
    
    if not prompt_template or not test_cases:
        return jsonify({'error': 'prompt_template and test_cases are required'}), 400
    
    try:
        evaluator = get_evaluator()
        batch = evaluator.evaluate_batch(
            prompt_template=prompt_template,
            test_cases=test_cases,
            model_name=model_name,
            prompt_version=prompt_version
        )
        
        # Save to database
        db.save_batch(batch)
        
        # Convert to serializable dict
        batch_dict = convert_to_serializable(batch.to_dict())
        return jsonify(batch_dict)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/results', methods=['GET'])
def get_results():
    """Get evaluation results."""
    model_name = request.args.get('model_name')
    prompt_version = request.args.get('prompt_version')
    limit = int(request.args.get('limit', 100))
    
    results = db.get_results(
        model_name=model_name,
        prompt_version=prompt_version,
        limit=limit
    )
    
    return jsonify([convert_to_serializable(r.to_dict()) for r in results])

@app.route('/api/batches', methods=['GET'])
def get_batches():
    """Get evaluation batches."""
    model_name = request.args.get('model_name')
    prompt_version = request.args.get('prompt_version')
    limit = int(request.args.get('limit', 50))
    
    batches = db.get_batches(
        model_name=model_name,
        prompt_version=prompt_version,
        limit=limit
    )
    
    return jsonify(batches)

@app.route('/api/trends', methods=['GET'])
def get_trends():
    """Get hallucination rate trends."""
    model_name = request.args.get('model_name')
    prompt_version = request.args.get('prompt_version')
    
    trends = db.get_trends(
        model_name=model_name,
        prompt_version=prompt_version
    )
    
    return jsonify(trends)

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get available LLM models."""
    # Return static list to avoid loading evaluator for this endpoint
    models = {
        "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
        "anthropic": ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"],
        "free": ["llama-3.1-8b", "gemma-2-9b", "mistral-7b"]
    }
    return jsonify(models)

@app.route('/health')
def health():
    """Health check endpoint for Render."""
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    app.run(debug=Config.FLASK_DEBUG, host='0.0.0.0', port=5001)


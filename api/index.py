"""Vercel serverless entry point for Flask app."""
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, jsonify, request
from flask.json.provider import DefaultJSONProvider
from flask_cors import CORS
from datetime import datetime
import json
import numpy as np

# Import from parent directory
from config import Config
from database import Database, convert_to_serializable

# Lazy loading for heavy imports (evaluator loads ML models)
_evaluator = None

def get_evaluator():
    global _evaluator
    if _evaluator is None:
        from evaluator import HallucinationEvaluator
        _evaluator = HallucinationEvaluator()
    return _evaluator

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

# Initialize Flask app
app = Flask(__name__, 
            template_folder='../templates',
            static_folder='../static')
app.json = CustomJSONProvider(app)
CORS(app)

# Initialize database
db = Database()

# Page routes
@app.route('/')
def index():
    return render_template('dashboard.html', active_page='dashboard')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html', active_page='dashboard')

@app.route('/playground')
def playground():
    return render_template('playground.html', active_page='playground')

@app.route('/evaluations')
def evaluations():
    return render_template('evaluations.html', active_page='evaluations')

@app.route('/analytics')
def analytics():
    return render_template('analytics.html', active_page='analytics')

@app.route('/settings')
def settings():
    return render_template('settings.html', active_page='settings')

# API routes
@app.route('/api/evaluate', methods=['POST'])
def evaluate():
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
        
        db.save_result(result)
        result_dict = convert_to_serializable(result.to_dict())
        return jsonify(result_dict)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/evaluate/batch', methods=['POST'])
def evaluate_batch():
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
        
        db.save_batch(batch)
        batch_dict = convert_to_serializable(batch.to_dict())
        return jsonify(batch_dict)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/results', methods=['GET'])
def get_results():
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
    model_name = request.args.get('model_name')
    prompt_version = request.args.get('prompt_version')
    
    trends = db.get_trends(
        model_name=model_name,
        prompt_version=prompt_version
    )
    
    return jsonify(trends)

@app.route('/api/models', methods=['GET'])
def get_models():
    evaluator = get_evaluator()
    models = evaluator.llm_client.list_available_models()
    return jsonify(models)

# Vercel handler
app = app


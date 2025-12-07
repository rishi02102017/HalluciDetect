"""Flask web application for the dashboard."""
import os
import io
import csv
from flask import Flask, render_template, jsonify, request, redirect, url_for, flash, make_response
from flask.json.provider import DefaultJSONProvider
from flask_cors import CORS
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from datetime import datetime
from database import Database, convert_to_serializable, User
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
app.secret_key = os.getenv("SECRET_KEY", "hallucidetect-secret-key-change-in-production")
CORS(app)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'info'

# Initialize database
db = Database()
db.init_default_templates()  # Initialize default templates

@login_manager.user_loader
def load_user(user_id):
    return db.get_user_by_id(user_id)

# Lazy loading for heavy ML components
_evaluator = None

def get_evaluator():
    """Lazy load the evaluator to speed up app startup."""
    global _evaluator
    if _evaluator is None:
        from evaluator import HallucinationEvaluator
        _evaluator = HallucinationEvaluator()
    return _evaluator

# ============ Page Routes ============

@app.route('/')
def index():
    """Landing page for guests, dashboard for logged in users."""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('landing.html')

@app.route('/dashboard')
@login_required
def dashboard():
    """Dashboard page."""
    return render_template('dashboard.html', active_page='dashboard')

@app.route('/playground')
@login_required
def playground():
    """Playground page for single evaluations."""
    return render_template('playground.html', active_page='playground')

@app.route('/evaluations')
@login_required
def evaluations():
    """Evaluations history page."""
    return render_template('evaluations.html', active_page='evaluations')

@app.route('/analytics')
@login_required
def analytics():
    """Analytics page with charts."""
    return render_template('analytics.html', active_page='analytics')

@app.route('/settings')
@login_required
def settings():
    """Settings page."""
    return render_template('settings.html', active_page='settings')

@app.route('/templates')
@login_required
def templates_page():
    """Templates page."""
    return render_template('templates.html', active_page='templates')

@app.route('/profile')
@login_required
def profile():
    """User profile page."""
    return render_template('profile.html', active_page='profile')

# ============ Auth Routes ============

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page and handler."""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        email_or_username = request.form.get('email_or_username')
        password = request.form.get('password')
        remember = request.form.get('remember', False)
        
        user = db.authenticate_user(email_or_username, password)
        if user:
            login_user(user, remember=bool(remember))
            next_page = request.args.get('next')
            flash('Welcome back!', 'success')
            return redirect(next_page if next_page else url_for('dashboard'))
        else:
            flash('Invalid email/username or password.', 'error')
            return redirect(url_for('login'))
    
    return render_template('auth/login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Registration page and handler."""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        email = request.form.get('email')
        username = request.form.get('username')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        # Validation
        if not email or not username or not password:
            flash('All fields are required.', 'error')
            return redirect(url_for('register'))
        elif password != confirm_password:
            flash('Passwords do not match.', 'error')
            return redirect(url_for('register'))
        elif len(password) < 6:
            flash('Password must be at least 6 characters.', 'error')
            return redirect(url_for('register'))
        elif len(username) < 3:
            flash('Username must be at least 3 characters.', 'error')
            return redirect(url_for('register'))
        else:
            user = db.create_user(email, username, password)
            if user:
                login_user(user)
                flash('Account created successfully! Welcome to HalluciDetect.', 'success')
                return redirect(url_for('dashboard'))
            else:
                flash('Email or username already exists.', 'error')
                return redirect(url_for('register'))
    
    return render_template('auth/register.html')

@app.route('/logout')
@login_required
def logout():
    """Logout handler."""
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

# ============ API Routes ============

@app.route('/api/evaluate', methods=['POST'])
@login_required
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
        
        # Save to database with user association
        db.save_result_with_user(result, current_user.id)
        
        # Convert to serializable dict
        result_dict = convert_to_serializable(result.to_dict())
        return jsonify(result_dict)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/evaluate/batch', methods=['POST'])
@login_required
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

@app.route('/api/evaluate/csv', methods=['POST'])
@login_required
def evaluate_csv():
    """Evaluate from uploaded CSV file."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'File must be a CSV'}), 400
    
    model_name = request.form.get('model_name', Config.DEFAULT_LLM_MODEL)
    prompt_version = request.form.get('prompt_version', 'v1')
    
    try:
        # Read CSV
        stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
        csv_reader = csv.DictReader(stream)
        
        results = []
        evaluator = get_evaluator()
        
        for row in csv_reader:
            prompt = row.get('prompt', '')
            reference_text = row.get('reference', row.get('reference_text', ''))
            
            if not prompt:
                continue
            
            result = evaluator.evaluate(
                prompt=prompt,
                model_name=model_name,
                prompt_version=prompt_version,
                reference_text=reference_text if reference_text else None
            )
            
            db.save_result_with_user(result, current_user.id)
            results.append(convert_to_serializable(result.to_dict()))
        
        return jsonify({
            'total': len(results),
            'results': results
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/results', methods=['GET'])
@login_required
def get_results():
    """Get evaluation results for current user."""
    model_name = request.args.get('model_name')
    prompt_version = request.args.get('prompt_version')
    limit = int(request.args.get('limit', 100))
    
    # Always show only user's own results
    results = db.get_user_results(current_user.id, limit)
    
    return jsonify([convert_to_serializable(r.to_dict()) for r in results])

@app.route('/api/results/export', methods=['GET'])
@login_required
def export_results():
    """Export results as CSV."""
    limit = int(request.args.get('limit', 1000))
    
    # Export only user's own results
    results = db.get_user_results(current_user.id, limit)
    
    # Create CSV
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Header
    writer.writerow([
        'ID', 'Timestamp', 'Model', 'Prompt Version', 'Prompt', 'LLM Output',
        'Hallucination Score', 'Is Hallucination', 'Confidence',
        'Semantic Score', 'Fact Check Score', 'Rule Based Score'
    ])
    
    # Data
    for r in results:
        writer.writerow([
            r.id, r.timestamp.isoformat() if r.timestamp else '',
            r.model_name, r.prompt_version, r.prompt[:500], r.llm_output[:500],
            r.overall_hallucination_score, r.is_hallucination, r.confidence,
            r.semantic_similarity_score, r.fact_check_score, r.rule_based_score
        ])
    
    # Create response
    output.seek(0)
    response = make_response(output.getvalue())
    response.headers['Content-Disposition'] = 'attachment; filename=evaluation_results.csv'
    response.headers['Content-Type'] = 'text/csv'
    return response

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
    models = {
        "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
        "anthropic": ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"],
        "free": ["llama-3.1-8b", "gemma-2-9b", "mistral-7b"]
    }
    return jsonify(models)

# ============ Template API Routes ============

@app.route('/api/templates', methods=['GET'])
@login_required
def get_templates():
    """Get prompt templates for current user."""
    category = request.args.get('category')
    
    templates = db.get_templates(user_id=current_user.id, category=category)
    return jsonify(templates)

@app.route('/api/templates', methods=['POST'])
def create_template():
    """Create a new prompt template."""
    if not current_user.is_authenticated:
        return jsonify({'error': 'Login required to create templates'}), 401
    
    data = request.json
    name = data.get('name')
    prompt_template = data.get('prompt_template')
    
    if not name or not prompt_template:
        return jsonify({'error': 'Name and prompt_template are required'}), 400
    
    template = db.create_template(
        name=name,
        prompt_template=prompt_template,
        user_id=current_user.id,
        description=data.get('description', ''),
        category=data.get('category', 'custom'),
        reference_template=data.get('reference_template'),
        is_public=data.get('is_public', False)
    )
    
    return jsonify(template.to_dict())

@app.route('/api/templates/<template_id>', methods=['GET'])
def get_template(template_id):
    """Get a specific template."""
    template = db.get_template_by_id(template_id)
    if not template:
        return jsonify({'error': 'Template not found'}), 404
    return jsonify(template)

@app.route('/api/templates/<template_id>', methods=['DELETE'])
@login_required
def delete_template(template_id):
    """Delete a template."""
    success = db.delete_template(template_id, current_user.id)
    if success:
        return jsonify({'message': 'Template deleted'})
    return jsonify({'error': 'Template not found or not authorized'}), 404

# ============ User API Routes ============

@app.route('/api/user', methods=['GET'])
def get_current_user():
    """Get current user info."""
    if current_user.is_authenticated:
        return jsonify({
            'authenticated': True,
            'user': current_user.to_dict()
        })
    return jsonify({'authenticated': False})

@app.route('/health')
def health():
    """Health check endpoint for Render."""
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    app.run(debug=Config.FLASK_DEBUG, host='0.0.0.0', port=5001)

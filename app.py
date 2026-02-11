"""Flask web application for the dashboard."""
import os
import io
import csv
import re
from flask import Flask, render_template, jsonify, request, redirect, url_for, flash, make_response
from flask.json.provider import DefaultJSONProvider
from flask_cors import CORS
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from datetime import datetime
from database import Database, convert_to_serializable, User
from config import Config
import json
import numpy as np

# Security imports
try:
    from flask_wtf.csrf import CSRFProtect, CSRFError
    CSRF_AVAILABLE = True
except ImportError:
    CSRF_AVAILABLE = False
    CSRFProtect = None

try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
    LIMITER_AVAILABLE = True
except ImportError:
    LIMITER_AVAILABLE = False
    Limiter = None

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
app.secret_key = os.getenv("SECRET_KEY")
if not app.secret_key:
    raise RuntimeError("SECRET_KEY environment variable is required. Add it to your .env file.")
CORS(app)

# Initialize CSRF Protection
csrf = None
if CSRF_AVAILABLE:
    app.config['WTF_CSRF_TIME_LIMIT'] = 3600  # 1 hour token validity
    app.config['WTF_CSRF_CHECK_DEFAULT'] = False  # We'll check manually for form routes
    csrf = CSRFProtect(app)
    
    @app.before_request
    def csrf_protect():
        """Apply CSRF protection only to form submissions (not API routes)."""
        # Skip CSRF for safe methods
        if request.method in ('GET', 'HEAD', 'OPTIONS', 'TRACE'):
            return
        # Skip CSRF for API routes (they use JWT/token auth)
        if request.path.startswith('/api/'):
            return
        # Skip if Authorization header is present (API access)
        if request.headers.get('Authorization'):
            return
        # Validate CSRF for form submissions
        try:
            from flask_wtf.csrf import validate_csrf
            validate_csrf(request.form.get('csrf_token'))
        except Exception:
            flash('Session expired or invalid request. Please try again.', 'error')
            return redirect(request.referrer or url_for('login'))
    
    @app.errorhandler(CSRFError)
    def handle_csrf_error(e):
        # Return JSON error for API requests
        if request.path.startswith('/api/'):
            return jsonify({'error': 'CSRF validation failed'}), 403
        flash('Session expired or invalid request. Please try again.', 'error')
        return redirect(request.referrer or url_for('login'))

# Initialize Rate Limiter
limiter = None
if LIMITER_AVAILABLE:
    limiter = Limiter(
        key_func=get_remote_address,
        app=app,
        default_limits=["200 per day", "50 per hour"],
        storage_uri="memory://",  # Use in-memory storage (for production, use Redis)
    )
    
    @app.errorhandler(429)
    def ratelimit_handler(e):
        flash('Too many attempts. Please wait a few minutes before trying again.', 'error')
        return redirect(request.referrer or url_for('login'))

# ============ Password Validation ============

def validate_password(password: str) -> tuple[bool, str]:
    """
    Validate password strength.
    Returns (is_valid, error_message)
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters long."
    
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter."
    
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter."
    
    if not re.search(r'\d', password):
        return False, "Password must contain at least one number."
    
    if not re.search(r'[!@#$%^&*(),.?":{}|<>_\-+=\[\]\\;\'`~]', password):
        return False, "Password must contain at least one special character (!@#$%^&* etc.)."
    
    return True, ""

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'info'

# Initialize database
db = Database()
db.init_default_templates()  # Initialize default templates

# Auto-promote admin user from environment variable (useful for production)
ADMIN_EMAIL = os.getenv('ADMIN_EMAIL')
if ADMIN_EMAIL:
    db.set_admin_by_email(ADMIN_EMAIL, is_admin=True)

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

@app.route('/admin')
@login_required
def admin():
    """Admin dashboard - platform statistics and management."""
    # Check if user is admin
    if not getattr(current_user, 'is_admin', False):
        flash('Access denied. Admin privileges required.', 'error')
        return redirect(url_for('dashboard'))
    
    # Get platform stats
    stats = db.get_platform_stats()
    users = db.get_all_users()
    
    return render_template('admin.html', active_page='admin', stats=stats, users=users)

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

def rate_limit_auth(limit_string):
    """Decorator factory for auth rate limiting."""
    def decorator(f):
        if limiter:
            return limiter.limit(limit_string, methods=["POST"])(f)
        return f
    return decorator

@app.route('/login', methods=['GET', 'POST'])
@rate_limit_auth("5 per minute")
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
            # Security: Validate next_page to prevent open redirect
            if next_page and (not next_page.startswith('/') or next_page.startswith('//')):
                next_page = None
            flash('Welcome back!', 'success')
            return redirect(next_page if next_page else url_for('dashboard'))
        else:
            flash('Invalid email/username or password.', 'error')
            return redirect(url_for('login'))
    
    return render_template('auth/login.html')

@app.route('/register', methods=['GET', 'POST'])
@rate_limit_auth("3 per minute")
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
        
        # Strong password validation
        is_valid, error_msg = validate_password(password)
        if not is_valid:
            flash(error_msg, 'error')
            return redirect(url_for('register'))
        
        if len(username) < 3:
            flash('Username must be at least 3 characters.', 'error')
            return redirect(url_for('register'))
        
        user = db.create_user(email, username, password)
        if user:
            login_user(user)
            
            # Try to send verification email
            try:
                from email_utils import send_verification_email, is_email_configured
                token = db.create_email_verification_token(user.id)
                
                if token and is_email_configured():
                    verify_url = url_for('verify_email', token=token, _external=True)
                    send_verification_email(email, username, verify_url)
                    flash('Account created! Please check your email to verify your account.', 'success')
                else:
                    flash('Account created successfully! Welcome to HalluciDetect.', 'success')
            except Exception:
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

@app.route('/verify-email/<token>')
def verify_email(token):
    """Verify email with token."""
    user = db.verify_email_token(token)
    
    if user:
        # Send welcome email
        try:
            from email_utils import send_welcome_email, is_email_configured
            if is_email_configured():
                send_welcome_email(user.email, user.username)
        except Exception:
            pass
        
        flash('Email verified successfully! Your account is now fully activated.', 'success')
        
        if current_user.is_authenticated:
            return redirect(url_for('profile'))
        return redirect(url_for('login'))
    else:
        flash('Invalid or expired verification link.', 'error')
        return redirect(url_for('login'))

@app.route('/resend-verification')
@login_required
def resend_verification():
    """Resend verification email."""
    if current_user.email_verified:
        flash('Your email is already verified.', 'info')
        return redirect(url_for('profile'))
    
    try:
        from email_utils import send_verification_email, is_email_configured
        token = db.create_email_verification_token(current_user.id)
        
        if token and is_email_configured():
            verify_url = url_for('verify_email', token=token, _external=True)
            if send_verification_email(current_user.email, current_user.username, verify_url):
                flash('Verification email sent! Please check your inbox.', 'success')
            else:
                flash('Failed to send verification email. Please try again.', 'error')
        else:
            flash('Email service not configured.', 'error')
    except Exception as e:
        flash('Failed to send verification email.', 'error')
    
    return redirect(url_for('profile'))

# ============ Password Management Routes ============

@app.route('/change-password', methods=['POST'])
@login_required
def change_password():
    """Change password for logged-in user."""
    current_password = request.form.get('current_password')
    new_password = request.form.get('new_password')
    confirm_password = request.form.get('confirm_password')
    
    if not all([current_password, new_password, confirm_password]):
        flash('All fields are required.', 'error')
        return redirect(url_for('profile'))
    
    if not current_user.check_password(current_password):
        flash('Current password is incorrect.', 'error')
        return redirect(url_for('profile'))
    
    if new_password != confirm_password:
        flash('New passwords do not match.', 'error')
        return redirect(url_for('profile'))
    
    # Strong password validation
    is_valid, error_msg = validate_password(new_password)
    if not is_valid:
        flash(error_msg, 'error')
        return redirect(url_for('profile'))
    
    if db.update_user_password(current_user.id, new_password):
        flash('Password changed successfully!', 'success')
    else:
        flash('Failed to change password.', 'error')
    
    return redirect(url_for('profile'))

@app.route('/forgot-password', methods=['GET', 'POST'])
@rate_limit_auth("3 per minute")
def forgot_password():
    """Forgot password page and handler."""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        email = request.form.get('email')
        
        if not email:
            flash('Email is required.', 'error')
            return redirect(url_for('forgot_password'))
        
        token = db.create_password_reset_token(email)
        
        if token:
            reset_url = url_for('reset_password', token=token, _external=True)
            
            # Try to send email
            try:
                from email_utils import send_password_reset_email, is_email_configured
                user = db.get_user_by_email(email)
                
                if is_email_configured() and user:
                    if send_password_reset_email(email, user.username, reset_url):
                        flash('Password reset link has been sent to your email.', 'success')
                    else:
                        flash('Failed to send email. Please try again later.', 'error')
                else:
                    # Development mode - show link directly
                    flash(f'Reset link (dev mode): {reset_url}', 'info')
            except Exception as e:
                # Fallback for development
                flash(f'Reset link (dev mode): {reset_url}', 'info')
        else:
            # Don't reveal if email exists or not (security)
            flash('If an account with that email exists, a reset link has been sent.', 'info')
        
        return redirect(url_for('forgot_password'))
    
    return render_template('auth/forgot_password.html')

@app.route('/reset-password/<token>', methods=['GET', 'POST'])
@rate_limit_auth("3 per minute")
def reset_password(token):
    """Reset password with token."""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    user = db.verify_password_reset_token(token)
    if not user:
        flash('Invalid or expired reset link.', 'error')
        return redirect(url_for('forgot_password'))
    
    if request.method == 'POST':
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')
        
        if not new_password or not confirm_password:
            flash('All fields are required.', 'error')
            return redirect(url_for('reset_password', token=token))
        
        if new_password != confirm_password:
            flash('Passwords do not match.', 'error')
            return redirect(url_for('reset_password', token=token))
        
        # Strong password validation
        is_valid, error_msg = validate_password(new_password)
        if not is_valid:
            flash(error_msg, 'error')
            return redirect(url_for('reset_password', token=token))
        
        if db.reset_password_with_token(token, new_password):
            flash('Password has been reset successfully! Please log in.', 'success')
            return redirect(url_for('login'))
        else:
            flash('Failed to reset password. Please try again.', 'error')
            return redirect(url_for('forgot_password'))
    
    return render_template('auth/reset_password.html', token=token)

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

@app.route('/api/results/export/json', methods=['GET'])
@login_required
def export_results_json():
    """Export results as JSON."""
    limit = int(request.args.get('limit', 1000))
    
    # Export only user's own results
    results = db.get_user_results(current_user.id, limit)
    
    export_data = {
        'exported_at': datetime.now().isoformat(),
        'user_id': current_user.id,
        'total_results': len(results),
        'results': [convert_to_serializable(r.to_dict()) for r in results]
    }
    
    response = make_response(json.dumps(export_data, indent=2))
    response.headers['Content-Disposition'] = 'attachment; filename=evaluation_results.json'
    response.headers['Content-Type'] = 'application/json'
    return response

@app.route('/api/templates/export/json', methods=['GET'])
@login_required
def export_templates_json():
    """Export user's templates as JSON."""
    templates = db.get_templates(user_id=current_user.id, include_public=False)
    
    export_data = {
        'exported_at': datetime.now().isoformat(),
        'user_id': current_user.id,
        'total_templates': len(templates),
        'templates': templates
    }
    
    response = make_response(json.dumps(export_data, indent=2))
    response.headers['Content-Disposition'] = 'attachment; filename=prompt_templates.json'
    response.headers['Content-Type'] = 'application/json'
    return response

@app.route('/api/templates/import/json', methods=['POST'])
@login_required
def import_templates_json():
    """Import templates from JSON file."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.endswith('.json'):
        return jsonify({'error': 'File must be a JSON'}), 400
    
    try:
        data = json.load(file)
        templates = data.get('templates', [])
        
        imported = 0
        for t in templates:
            db.create_template(
                name=t.get('name', 'Imported Template'),
                prompt_template=t.get('prompt_template', ''),
                user_id=current_user.id,
                description=t.get('description', ''),
                category=t.get('category', 'custom'),
                reference_template=t.get('reference_template'),
                is_public=False  # Always private on import
            )
            imported += 1
        
        return jsonify({
            'message': f'Successfully imported {imported} templates',
            'imported': imported
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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
        "openai": ["gpt-4o", "gpt-4o-mini", "o1-preview", "o1-mini", "gpt-4-turbo"],
        "anthropic": ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022", "claude-3-opus-20240229"],
        "google": ["gemini-2.0-flash-exp", "gemini-1.5-pro", "gemini-1.5-flash"],
        "meta": ["llama-3.3-70b-instruct", "llama-3.2-90b-vision-instruct", "llama-3.1-405b-instruct"],
        "free": ["llama-3.2-3b-instruct:free", "gemma-2-9b-it:free", "mistral-7b-instruct:free", "qwen-2.5-7b-instruct:free"]
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
@login_required
def get_template(template_id):
    """Get a specific template (only if owned by user or public)."""
    template = db.get_template_by_id(template_id)
    if not template:
        return jsonify({'error': 'Template not found'}), 404
    
    # Verify ownership or public access
    if not template.get('is_public') and template.get('user_id') != current_user.id:
        return jsonify({'error': 'Not authorized to access this template'}), 403
    
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

@app.route('/api/user/theme', methods=['GET'])
@login_required
def get_user_theme():
    """Get user's theme preference."""
    theme = db.get_user_preference(current_user.id, 'theme') or 'indigo'
    return jsonify({'theme': theme})

@app.route('/api/user/theme', methods=['POST'])
@login_required
def set_user_theme():
    """Set user's theme preference."""
    data = request.get_json() or {}
    theme = data.get('theme', 'indigo')
    
    # Validate theme
    valid_themes = ['indigo', 'purple', 'violet', 'blue', 'cyan', 'teal', 
                    'green', 'lime', 'yellow', 'orange', 'red', 'rose', 
                    'pink', 'fuchsia', 'slate']
    
    if theme not in valid_themes:
        return jsonify({'error': 'Invalid theme'}), 400
    
    db.set_user_preference(current_user.id, 'theme', theme)
    return jsonify({'success': True, 'theme': theme})

# ============ Admin API Endpoints ============

def admin_required(f):
    """Decorator to require admin access."""
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            return jsonify({'error': 'Authentication required'}), 401
        if not getattr(current_user, 'is_admin', False):
            return jsonify({'error': 'Admin access required'}), 403
        return f(*args, **kwargs)
    return decorated_function

@app.route('/api/admin/stats')
@admin_required
def api_admin_stats():
    """Get platform-wide statistics (admin only)."""
    stats = db.get_platform_stats()
    # Add system info
    stats['database_type'] = 'PostgreSQL' if db.is_postgresql() else 'SQLite'
    stats['version'] = '1.3.0'
    return jsonify(stats)

@app.route('/api/admin/users')
@admin_required
def api_admin_users():
    """Get all users (admin only)."""
    users = db.get_all_users()
    return jsonify(users)

@app.route('/api/admin/users/<user_id>/toggle-admin', methods=['POST'])
@admin_required
def api_toggle_admin(user_id):
    """Toggle admin status for a user (admin only)."""
    # Prevent self-demotion
    if user_id == current_user.id:
        return jsonify({'error': 'Cannot modify your own admin status'}), 400
    
    data = request.get_json() or {}
    is_admin = data.get('is_admin', False)
    
    if db.set_admin(user_id, is_admin):
        return jsonify({'success': True, 'is_admin': is_admin})
    return jsonify({'error': 'User not found'}), 404

# ============ JWT API Authentication ============

@app.route('/api/auth/token', methods=['POST'])
def get_api_token():
    """Get JWT tokens for API authentication."""
    from jwt_auth import get_tokens_for_user
    
    data = request.json or {}
    email_or_username = data.get('email') or data.get('username')
    password = data.get('password')
    
    if not email_or_username or not password:
        return jsonify({
            'error': 'Missing credentials',
            'message': 'Both email/username and password are required'
        }), 400
    
    user = db.authenticate_user(email_or_username, password)
    
    if user:
        tokens = get_tokens_for_user(user.id, user.username)
        return jsonify({
            'success': True,
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email
            },
            **tokens
        })
    
    return jsonify({
        'error': 'Authentication failed',
        'message': 'Invalid email/username or password'
    }), 401

@app.route('/api/auth/refresh', methods=['POST'])
def refresh_api_token():
    """Refresh JWT access token using refresh token."""
    from jwt_auth import decode_token, generate_access_token
    
    data = request.json or {}
    refresh_token = data.get('refresh_token')
    
    if not refresh_token:
        return jsonify({'error': 'Refresh token required'}), 400
    
    payload = decode_token(refresh_token)
    
    if 'error' in payload:
        return jsonify({'error': payload['error']}), 401
    
    if payload.get('type') != 'refresh':
        return jsonify({'error': 'Invalid token type'}), 401
    
    user = db.get_user_by_id(payload.get('user_id'))
    if not user:
        return jsonify({'error': 'User not found'}), 401
    
    new_access_token = generate_access_token(user.id, user.username)
    
    return jsonify({
        'access_token': new_access_token,
        'token_type': 'Bearer',
        'expires_in': Config.JWT_ACCESS_TOKEN_EXPIRES
    })

@app.route('/api/auth/verify', methods=['GET'])
def verify_api_token():
    """Verify if a JWT token is valid."""
    from jwt_auth import jwt_required
    from flask import g
    
    auth_header = request.headers.get('Authorization')
    if not auth_header:
        return jsonify({'valid': False, 'error': 'No token provided'}), 401
    
    parts = auth_header.split()
    if len(parts) != 2 or parts[0].lower() != 'bearer':
        return jsonify({'valid': False, 'error': 'Invalid authorization header'}), 401
    
    from jwt_auth import decode_token
    payload = decode_token(parts[1])
    
    if 'error' in payload:
        return jsonify({'valid': False, 'error': payload['error']}), 401
    
    return jsonify({
        'valid': True,
        'user_id': payload.get('user_id'),
        'username': payload.get('username'),
        'expires': payload.get('exp')
    })

# ============ PDF Export API ============

@app.route('/api/results/export/pdf', methods=['GET'])
@login_required
def export_results_pdf():
    """Export results as PDF report."""
    from pdf_export import get_pdf_generator
    
    limit = int(request.args.get('limit', 100))
    
    # Get user's results
    results = db.get_user_results(current_user.id, limit)
    results_dicts = [convert_to_serializable(r.to_dict()) for r in results]
    
    # Generate PDF
    pdf_gen = get_pdf_generator()
    pdf_buffer = pdf_gen.generate_evaluation_report(
        results_dicts,
        user_info={'username': current_user.username, 'email': current_user.email},
        title="HalluciDetect - Evaluation Report"
    )
    
    response = make_response(pdf_buffer.getvalue())
    response.headers['Content-Disposition'] = 'attachment; filename=evaluation_report.pdf'
    response.headers['Content-Type'] = 'application/pdf'
    return response

# ============ Knowledge Base API ============

@app.route('/api/knowledge-base/search', methods=['GET'])
@login_required
def search_knowledge_base():
    """Search the knowledge base."""
    from knowledge_base import get_knowledge_base
    
    query = request.args.get('q', '')
    if not query:
        return jsonify({'error': 'Query parameter q is required'}), 400
    
    kb = get_knowledge_base()
    results = kb.search(query, top_k=10)
    
    return jsonify({
        'query': query,
        'results': results,
        'count': len(results)
    })

@app.route('/api/knowledge-base/verify', methods=['POST'])
@login_required
def verify_claim():
    """Verify a claim against the knowledge base."""
    from knowledge_base import get_knowledge_base
    
    data = request.json or {}
    claim = data.get('claim', '')
    
    if not claim:
        return jsonify({'error': 'Claim is required'}), 400
    
    kb = get_knowledge_base()
    result = kb.verify_claim(claim)
    
    return jsonify(result)

@app.route('/api/knowledge-base/facts', methods=['POST'])
@login_required
def add_fact():
    """Add a new fact to the knowledge base."""
    from knowledge_base import get_knowledge_base
    
    data = request.json or {}
    statement = data.get('statement', '')
    
    if not statement:
        return jsonify({'error': 'Statement is required'}), 400
    
    kb = get_knowledge_base()
    fact = kb.add_fact(
        statement=statement,
        category=data.get('category', 'custom'),
        source=data.get('source', f'user:{current_user.username}'),
        domain=data.get('domain', 'custom'),
        verified=data.get('verified', True)
    )
    
    return jsonify({
        'message': 'Fact added successfully',
        'fact': fact
    })

@app.route('/api/knowledge-base/stats', methods=['GET'])
@login_required
def get_knowledge_base_stats():
    """Get knowledge base statistics."""
    from knowledge_base import get_knowledge_base
    
    kb = get_knowledge_base()
    stats = kb.get_stats()
    
    return jsonify(stats)

# ============ Database Backup API ============

@app.route('/api/backup', methods=['POST'])
@login_required
def create_backup():
    """Create a database backup (admin only for now)."""
    from backup_utils import create_backup as do_backup
    
    result = do_backup(db)
    
    if result.get('success'):
        return jsonify({
            'message': 'Backup created successfully',
            'type': result.get('type'),
            'path': result.get('path')
        })
    
    return jsonify({'error': 'Backup failed'}), 500

@app.route('/api/backup/list', methods=['GET'])
@login_required
def list_backups():
    """List available backups."""
    from backup_utils import get_backup_manager
    
    manager = get_backup_manager()
    backups = manager.list_backups()
    
    return jsonify({
        'backups': backups,
        'count': len(backups)
    })

# ============ JWT Protected API Endpoints ============

@app.route('/api/v1/evaluate', methods=['POST'])
def evaluate_with_jwt():
    """JWT-protected evaluation endpoint for API clients."""
    from jwt_auth import jwt_required, jwt_optional
    from flask import g
    
    # Check for JWT auth
    auth_header = request.headers.get('Authorization')
    user_id = None
    
    if auth_header:
        parts = auth_header.split()
        if len(parts) == 2 and parts[0].lower() == 'bearer':
            from jwt_auth import decode_token
            payload = decode_token(parts[1])
            if 'error' not in payload:
                user_id = payload.get('user_id')
    
    # Also check session auth
    if not user_id and current_user.is_authenticated:
        user_id = current_user.id
    
    if not user_id:
        return jsonify({
            'error': 'Authentication required',
            'message': 'Provide JWT token or login via session'
        }), 401
    
    data = request.json
    prompt = data.get('prompt')
    reference_text = data.get('reference_text', '')
    model_name = data.get('model_name', 'gpt-4o-mini')
    prompt_version = data.get('prompt_version', 'v1')
    
    if not prompt:
        return jsonify({'error': 'Prompt is required'}), 400
    
    evaluator = get_evaluator()
    
    # Generate LLM response
    try:
        from llm_client import LLMClient
        llm_client = LLMClient()
        llm_output = llm_client.generate(prompt, model_name)
    except Exception as e:
        return jsonify({'error': f'LLM error: {str(e)}'}), 500
    
    # Evaluate for hallucinations
    result = evaluator.evaluate(
        prompt=prompt,
        llm_output=llm_output,
        reference_text=reference_text,
        model_name=model_name,
        prompt_version=prompt_version
    )
    
    # Save result
    db.save_result_with_user(result, user_id)
    
    return jsonify(convert_to_serializable(result.to_dict()))

# ============================================
# PHASE 2 FEATURES
# ============================================

# Import Phase 2 modules
try:
    from nlp_metrics import get_nlp_metrics
    NLP_METRICS_AVAILABLE = True
except ImportError:
    NLP_METRICS_AVAILABLE = False

try:
    from ab_testing import get_ab_manager
    AB_TESTING_AVAILABLE = True
except ImportError:
    AB_TESTING_AVAILABLE = False

try:
    from rag_evaluator import get_rag_evaluator
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

try:
    from webhooks import get_webhook_manager, AlertLevel
    WEBHOOKS_AVAILABLE = True
except ImportError:
    WEBHOOKS_AVAILABLE = False


# ==================== NLP Metrics API ====================

@app.route('/api/metrics/nlp', methods=['POST'])
@login_required
def calculate_nlp_metrics():
    """Calculate BLEU, ROUGE, and other NLP metrics."""
    if not NLP_METRICS_AVAILABLE:
        return jsonify({"error": "NLP metrics module not available"}), 500
    
    data = request.get_json()
    candidate = data.get('candidate', '')
    reference = data.get('reference', '')
    
    if not candidate or not reference:
        return jsonify({"error": "Both 'candidate' and 'reference' are required"}), 400
    
    metrics = get_nlp_metrics()
    result = metrics.calculate_all(candidate, reference)
    
    return jsonify(result)


@app.route('/api/metrics/bleu', methods=['POST'])
@login_required
def calculate_bleu():
    """Calculate BLEU score only."""
    if not NLP_METRICS_AVAILABLE:
        return jsonify({"error": "NLP metrics module not available"}), 500
    
    data = request.get_json()
    candidate = data.get('candidate', '')
    reference = data.get('reference', '')
    
    metrics = get_nlp_metrics()
    result = metrics.bleu_score(candidate, reference)
    
    return jsonify(result)


@app.route('/api/metrics/rouge', methods=['POST'])
@login_required
def calculate_rouge():
    """Calculate ROUGE scores only."""
    if not NLP_METRICS_AVAILABLE:
        return jsonify({"error": "NLP metrics module not available"}), 500
    
    data = request.get_json()
    candidate = data.get('candidate', '')
    reference = data.get('reference', '')
    
    metrics = get_nlp_metrics()
    result = metrics.rouge_scores(candidate, reference)
    
    return jsonify(result)


# ==================== A/B Testing API ====================

@app.route('/api/ab-tests', methods=['GET'])
@login_required
def list_ab_tests():
    """List all A/B tests."""
    if not AB_TESTING_AVAILABLE:
        return jsonify({"error": "A/B testing module not available"}), 500
    
    manager = get_ab_manager()
    return jsonify({"tests": manager.list_tests()})


@app.route('/api/ab-tests', methods=['POST'])
@login_required
def create_ab_test():
    """Create a new A/B test."""
    if not AB_TESTING_AVAILABLE:
        return jsonify({"error": "A/B testing module not available"}), 500
    
    data = request.get_json()
    
    required = ['name', 'prompt_a', 'prompt_b']
    if not all(k in data for k in required):
        return jsonify({"error": f"Required fields: {required}"}), 400
    
    manager = get_ab_manager()
    test = manager.create_test(
        name=data['name'],
        description=data.get('description', ''),
        prompt_a=data['prompt_a'],
        prompt_b=data['prompt_b'],
        variant_a_name=data.get('variant_a_name', 'Control'),
        variant_b_name=data.get('variant_b_name', 'Variant'),
        model_name=data.get('model_name'),
        reference_text=data.get('reference_text')
    )
    
    return jsonify({"test": test.to_dict()}), 201


@app.route('/api/ab-tests/<test_id>', methods=['GET'])
@login_required
def get_ab_test(test_id):
    """Get A/B test details and analysis."""
    if not AB_TESTING_AVAILABLE:
        return jsonify({"error": "A/B testing module not available"}), 500
    
    manager = get_ab_manager()
    analysis = manager.analyze_test(test_id)
    
    if "error" in analysis:
        return jsonify(analysis), 404
    
    return jsonify(analysis)


@app.route('/api/ab-tests/<test_id>/results', methods=['POST'])
@login_required
def add_ab_result(test_id):
    """Add a result to an A/B test variant."""
    if not AB_TESTING_AVAILABLE:
        return jsonify({"error": "A/B testing module not available"}), 500
    
    data = request.get_json()
    variant = data.get('variant', '')  # "A" or "B"
    score = data.get('score', 0.0)
    latency = data.get('latency', 0.0)
    hallucination_rate = data.get('hallucination_rate', 0.0)
    
    if variant not in ['A', 'B']:
        return jsonify({"error": "Variant must be 'A' or 'B'"}), 400
    
    manager = get_ab_manager()
    success = manager.add_result(test_id, variant, score, latency, hallucination_rate)
    
    if not success:
        return jsonify({"error": "Test not found"}), 404
    
    return jsonify({"success": True, "message": f"Result added to variant {variant}"})


# ==================== RAG Evaluation API ====================

@app.route('/api/rag/evaluate', methods=['POST'])
@login_required
def evaluate_rag():
    """Evaluate RAG pipeline output."""
    if not RAG_AVAILABLE:
        return jsonify({"error": "RAG evaluator module not available"}), 500
    
    data = request.get_json()
    query = data.get('query', '')
    answer = data.get('answer', '')
    contexts = data.get('contexts', [])
    
    if not query or not answer:
        return jsonify({"error": "Both 'query' and 'answer' are required"}), 400
    
    if not contexts:
        return jsonify({"error": "At least one context is required"}), 400
    
    evaluator = get_rag_evaluator()
    result = evaluator.evaluate(query, answer, contexts)
    
    return jsonify(result)


@app.route('/api/rag/context-relevance', methods=['POST'])
@login_required
def rag_context_relevance():
    """Check context relevance for RAG."""
    if not RAG_AVAILABLE:
        return jsonify({"error": "RAG evaluator module not available"}), 500
    
    data = request.get_json()
    query = data.get('query', '')
    contexts = data.get('contexts', [])
    
    evaluator = get_rag_evaluator()
    result = evaluator.context_relevance(query, contexts)
    
    return jsonify(result)


@app.route('/api/rag/faithfulness', methods=['POST'])
@login_required
def rag_faithfulness():
    """Check answer faithfulness for RAG."""
    if not RAG_AVAILABLE:
        return jsonify({"error": "RAG evaluator module not available"}), 500
    
    data = request.get_json()
    answer = data.get('answer', '')
    contexts = data.get('contexts', [])
    
    evaluator = get_rag_evaluator()
    result = evaluator.answer_faithfulness(answer, contexts)
    
    return jsonify(result)


# ==================== Webhooks API ====================

@app.route('/api/webhooks', methods=['GET'])
@login_required
def list_webhooks():
    """List all webhooks."""
    if not WEBHOOKS_AVAILABLE:
        return jsonify({"error": "Webhooks module not available"}), 500
    
    manager = get_webhook_manager()
    return jsonify({"webhooks": manager.list_webhooks()})


@app.route('/api/webhooks', methods=['POST'])
@login_required
def create_webhook():
    """Create a new webhook."""
    if not WEBHOOKS_AVAILABLE:
        return jsonify({"error": "Webhooks module not available"}), 500
    
    data = request.get_json()
    
    if 'name' not in data or 'url' not in data:
        return jsonify({"error": "Both 'name' and 'url' are required"}), 400
    
    manager = get_webhook_manager()
    webhook = manager.add_webhook(
        name=data['name'],
        url=data['url'],
        events=data.get('events'),
        headers=data.get('headers'),
        secret=data.get('secret')
    )
    
    return jsonify({"webhook": webhook.to_dict()}), 201


@app.route('/api/webhooks/<webhook_id>', methods=['DELETE'])
@login_required
def delete_webhook(webhook_id):
    """Delete a webhook."""
    if not WEBHOOKS_AVAILABLE:
        return jsonify({"error": "Webhooks module not available"}), 500
    
    manager = get_webhook_manager()
    success = manager.remove_webhook(webhook_id)
    
    if not success:
        return jsonify({"error": "Webhook not found"}), 404
    
    return jsonify({"success": True})


@app.route('/api/webhooks/test', methods=['POST'])
@login_required
def test_webhook():
    """Send a test webhook."""
    if not WEBHOOKS_AVAILABLE:
        return jsonify({"error": "Webhooks module not available"}), 500
    
    data = request.get_json()
    webhook_id = data.get('webhook_id')
    
    manager = get_webhook_manager()
    result = manager.send_webhook(
        "test.ping",
        {"message": "Test webhook from HalluciDetect", "timestamp": datetime.utcnow().isoformat()},
        webhook_id
    )
    
    return jsonify({"results": result})


@app.route('/api/alerts', methods=['GET'])
@login_required
def list_alerts():
    """List all alert rules."""
    if not WEBHOOKS_AVAILABLE:
        return jsonify({"error": "Webhooks module not available"}), 500
    
    manager = get_webhook_manager()
    return jsonify({"alerts": manager.list_alert_rules()})


@app.route('/api/alerts', methods=['POST'])
@login_required
def create_alert():
    """Create a new alert rule."""
    if not WEBHOOKS_AVAILABLE:
        return jsonify({"error": "Webhooks module not available"}), 500
    
    data = request.get_json()
    
    required = ['name', 'metric', 'operator', 'threshold']
    if not all(k in data for k in required):
        return jsonify({"error": f"Required fields: {required}"}), 400
    
    level_map = {
        "info": AlertLevel.INFO,
        "warning": AlertLevel.WARNING,
        "error": AlertLevel.ERROR,
        "critical": AlertLevel.CRITICAL
    }
    
    manager = get_webhook_manager()
    rule = manager.add_alert_rule(
        name=data['name'],
        metric=data['metric'],
        operator=data['operator'],
        threshold=float(data['threshold']),
        level=level_map.get(data.get('level', 'warning'), AlertLevel.WARNING),
        cooldown_minutes=data.get('cooldown_minutes', 5)
    )
    
    return jsonify({"alert": rule.to_dict()}), 201


@app.route('/api/slack/configure', methods=['POST'])
@login_required
def configure_slack():
    """Configure Slack webhook URL."""
    if not WEBHOOKS_AVAILABLE:
        return jsonify({"error": "Webhooks module not available"}), 500
    
    data = request.get_json()
    webhook_url = data.get('webhook_url', '')
    
    if not webhook_url:
        return jsonify({"error": "webhook_url is required"}), 400
    
    manager = get_webhook_manager()
    manager.configure_slack(webhook_url)
    
    return jsonify({"success": True, "message": "Slack webhook configured"})


@app.route('/api/slack/test', methods=['POST'])
@login_required
def test_slack():
    """Send a test message to Slack."""
    if not WEBHOOKS_AVAILABLE:
        return jsonify({"error": "Webhooks module not available"}), 500
    
    manager = get_webhook_manager()
    result = manager.send_slack_alert(
        message="Test message from HalluciDetect",
        title="Test Notification",
        level=AlertLevel.INFO,
        fields={"Status": "Connected", "Time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")}
    )
    
    return jsonify(result)


# ==================== Medium Priority Features ====================

# ============ Test Suites ============

@app.route('/api/test-suites', methods=['GET'])
@login_required
def get_test_suites():
    """Get all test suites for the current user."""
    suites = db.get_test_suites(current_user.id)
    return jsonify(suites)

@app.route('/api/test-suites', methods=['POST'])
@login_required
def create_test_suite():
    """Create a new test suite."""
    data = request.get_json()
    name = data.get('name', '').strip()
    
    if not name:
        return jsonify({"error": "Name is required"}), 400
    
    suite = db.create_test_suite(
        user_id=current_user.id,
        name=name,
        description=data.get('description', ''),
        model_name=data.get('model_name')
    )
    
    if suite:
        return jsonify(suite), 201
    return jsonify({"error": "Failed to create test suite"}), 500

@app.route('/api/test-suites/<suite_id>', methods=['GET'])
@login_required
def get_test_suite(suite_id):
    """Get a specific test suite with test cases."""
    suite = db.get_test_suite(suite_id, current_user.id)
    if suite:
        return jsonify(suite)
    return jsonify({"error": "Test suite not found"}), 404

@app.route('/api/test-suites/<suite_id>', methods=['PUT'])
@login_required
def update_test_suite(suite_id):
    """Update a test suite."""
    data = request.get_json()
    success = db.update_test_suite(suite_id, current_user.id, **data)
    if success:
        return jsonify({"success": True})
    return jsonify({"error": "Test suite not found"}), 404

@app.route('/api/test-suites/<suite_id>', methods=['DELETE'])
@login_required
def delete_test_suite(suite_id):
    """Delete a test suite."""
    success = db.delete_test_suite(suite_id, current_user.id)
    if success:
        return jsonify({"success": True})
    return jsonify({"error": "Test suite not found"}), 404

@app.route('/api/test-suites/<suite_id>/cases', methods=['POST'])
@login_required
def add_test_case(suite_id):
    """Add a test case to a suite."""
    data = request.get_json()
    name = data.get('name', '').strip()
    prompt = data.get('prompt', '').strip()
    
    if not name or not prompt:
        return jsonify({"error": "Name and prompt are required"}), 400
    
    case = db.add_test_case(
        suite_id=suite_id,
        user_id=current_user.id,
        name=name,
        prompt=prompt,
        expected_output=data.get('expected_output'),
        tags=data.get('tags', [])
    )
    
    if case:
        return jsonify(case), 201
    return jsonify({"error": "Failed to add test case"}), 500

@app.route('/api/test-cases/<case_id>', methods=['PUT'])
@login_required
def update_test_case(case_id):
    """Update a test case."""
    data = request.get_json()
    success = db.update_test_case(case_id, current_user.id, **data)
    if success:
        return jsonify({"success": True})
    return jsonify({"error": "Test case not found"}), 404

@app.route('/api/test-cases/<case_id>', methods=['DELETE'])
@login_required
def delete_test_case(case_id):
    """Delete a test case."""
    success = db.delete_test_case(case_id, current_user.id)
    if success:
        return jsonify({"success": True})
    return jsonify({"error": "Test case not found"}), 404

@app.route('/api/test-suites/<suite_id>/run', methods=['POST'])
@login_required
def run_test_suite(suite_id):
    """Run all test cases in a suite."""
    suite = db.get_test_suite(suite_id, current_user.id)
    if not suite:
        return jsonify({"error": "Test suite not found"}), 404
    
    if not suite.get('test_cases'):
        return jsonify({"error": "No test cases in suite"}), 400
    
    # Update status to running
    db.update_test_suite(suite_id, current_user.id, status="running")
    
    results = []
    passed = 0
    failed = 0
    warnings = 0
    
    # Initialize LLM client and evaluator
    from llm_client import LLMClient
    llm_client = LLMClient()
    evaluator = get_evaluator()
    
    for case in suite['test_cases']:
        try:
            # Run evaluation
            model = suite.get('model_name') or 'gpt-4o-mini'
            llm_output = llm_client.generate(case['prompt'], model)
            expected_output = case.get('expected_output', '').strip()
            
            eval_result = evaluator.evaluate(
                prompt=case['prompt'],
                llm_output=llm_output,
                reference_text=expected_output,
                model_name=model
            )
            
            # Save result
            db.save_result_with_user(eval_result, current_user.id)
            
            # Determine status based on whether expected output was provided
            if expected_output:
                # Use a HYBRID approach for matching:
                # 1. Check if expected output is CONTAINED in LLM response (for short answers)
                # 2. Use semantic similarity as secondary metric
                
                expected_lower = expected_output.lower().strip()
                response_lower = llm_output.lower().strip()
                
                # Check containment (good for short factual answers like "Paris")
                contains_expected = expected_lower in response_lower
                
                # Get semantic similarity
                similarity = eval_result.semantic_similarity_score
                
                # Determine score: if contains expected, boost similarity
                if contains_expected:
                    # If response contains the expected answer, give high score
                    score = max(similarity, 0.8)  # At least 80% if contains expected
                else:
                    score = similarity
                
                if score >= 0.6:
                    status = "passed"
                    passed += 1
                elif score >= 0.3:
                    status = "warning"
                    warnings += 1
                else:
                    status = "failed"
                    failed += 1
            else:
                # Use HALLUCINATION SCORE when no expected output (sanity check mode)
                # Lower hallucination = better = pass
                score = eval_result.overall_hallucination_score
                
                if score < 0.3:
                    status = "passed"
                    passed += 1
                elif score < 0.6:
                    status = "warning"
                    warnings += 1
                else:
                    status = "failed"
                    failed += 1
            
            # Update test case
            db.update_test_case_result(case['id'], eval_result.id, score, status)
            
            results.append({
                "case_id": case['id'],
                "case_name": case['name'],
                "result_id": eval_result.id,
                "score": round(score, 3) if score is not None else 0,
                "similarity": round(score, 3) if expected_output else None,
                "hallucination_score": round(eval_result.overall_hallucination_score, 3) if eval_result.overall_hallucination_score else 0,
                "status": status,
                "mode": "match" if expected_output else "sanity"
            })
        except Exception as e:
            import traceback
            print(f"Test case error: {e}")
            traceback.print_exc()
            failed += 1
            results.append({
                "case_id": case['id'],
                "case_name": case['name'],
                "error": str(e),
                "status": "failed",
                "similarity": None,
                "hallucination_score": None,
                "mode": "error"
            })
    
    # Update suite status
    final_status = "completed" if failed == 0 else "failed"
    db.update_test_suite(suite_id, current_user.id, status=final_status, last_run_at=datetime.utcnow())
    
    return jsonify({
        "suite_id": suite_id,
        "total": len(suite['test_cases']),
        "passed": passed,
        "failed": failed,
        "warnings": warnings,
        "results": results
    })

# ============ Prompt Versioning ============

@app.route('/api/prompt-versions', methods=['GET'])
@login_required
def get_prompt_versions():
    """Get all prompt versions for the current user."""
    name = request.args.get('name')
    versions = db.get_prompt_versions(current_user.id, name)
    return jsonify(versions)

@app.route('/api/prompt-versions', methods=['POST'])
@login_required
def create_prompt_version():
    """Create a new prompt version."""
    data = request.get_json()
    name = data.get('name', '').strip()
    prompt_text = data.get('prompt_text', '').strip()
    
    if not name or not prompt_text:
        return jsonify({"error": "Name and prompt_text are required"}), 400
    
    version = db.create_prompt_version(
        user_id=current_user.id,
        name=name,
        prompt_text=prompt_text,
        description=data.get('description', ''),
        parent_version_id=data.get('parent_version_id')
    )
    
    if version:
        return jsonify(version), 201
    return jsonify({"error": "Failed to create prompt version"}), 500

@app.route('/api/prompt-versions/<version_id>', methods=['GET'])
@login_required
def get_prompt_version(version_id):
    """Get a specific prompt version."""
    version = db.get_prompt_version(version_id, current_user.id)
    if version:
        return jsonify(version)
    return jsonify({"error": "Prompt version not found"}), 404

@app.route('/api/prompt-names', methods=['GET'])
@login_required
def get_prompt_names():
    """Get unique prompt names for versioning."""
    names = db.get_prompt_names(current_user.id)
    return jsonify(names)

# ============ Evaluation Labels/Annotations ============

@app.route('/api/evaluations/<evaluation_id>/labels', methods=['GET'])
@login_required
def get_evaluation_labels(evaluation_id):
    """Get all labels for an evaluation."""
    labels = db.get_evaluation_labels(evaluation_id, current_user.id)
    return jsonify(labels)

@app.route('/api/evaluations/<evaluation_id>/labels', methods=['POST'])
@login_required
def add_evaluation_label(evaluation_id):
    """Add a label to an evaluation."""
    data = request.get_json()
    label = data.get('label', '').strip()
    
    if not label:
        return jsonify({"error": "Label is required"}), 400
    
    result = db.add_evaluation_label(
        evaluation_id=evaluation_id,
        user_id=current_user.id,
        label=label,
        color=data.get('color', '#6366f1'),
        notes=data.get('notes')
    )
    
    if result:
        return jsonify(result), 201
    return jsonify({"error": "Failed to add label"}), 500

@app.route('/api/labels/<label_id>', methods=['DELETE'])
@login_required
def delete_label(label_id):
    """Delete a label."""
    success = db.delete_evaluation_label(label_id, current_user.id)
    if success:
        return jsonify({"success": True})
    return jsonify({"error": "Label not found"}), 404

@app.route('/api/labels', methods=['GET'])
@login_required
def get_user_labels():
    """Get all unique labels used by the current user."""
    labels = db.get_user_labels(current_user.id)
    return jsonify(labels)

@app.route('/api/evaluations/by-label/<label>', methods=['GET'])
@login_required
def get_evaluations_by_label(label):
    """Get evaluations with a specific label."""
    evaluation_ids = db.get_evaluations_by_label(current_user.id, label)
    return jsonify({"label": label, "evaluation_ids": evaluation_ids, "count": len(evaluation_ids)})

# ============ Comparison View ============

@app.route('/api/evaluations/<evaluation_id>', methods=['GET'])
@login_required
def get_evaluation_detail(evaluation_id):
    """Get detailed evaluation for comparison."""
    result = db.get_evaluation_by_id(evaluation_id, current_user.id)
    if result:
        # Also get labels
        result['labels'] = db.get_evaluation_labels(evaluation_id, current_user.id)
        return jsonify(result)
    return jsonify({"error": "Evaluation not found"}), 404

@app.route('/api/compare', methods=['POST'])
@login_required
def compare_evaluations():
    """Compare multiple evaluations side-by-side."""
    data = request.get_json()
    evaluation_ids = data.get('evaluation_ids', [])
    
    if len(evaluation_ids) < 2:
        return jsonify({"error": "At least 2 evaluation IDs required"}), 400
    
    if len(evaluation_ids) > 4:
        return jsonify({"error": "Maximum 4 evaluations can be compared"}), 400
    
    evaluations = []
    for eval_id in evaluation_ids:
        result = db.get_evaluation_by_id(eval_id, current_user.id)
        if result:
            result['labels'] = db.get_evaluation_labels(eval_id, current_user.id)
            evaluations.append(result)
    
    if len(evaluations) < 2:
        return jsonify({"error": "Not enough valid evaluations found"}), 404
    
    # Calculate comparison metrics
    comparison = {
        "evaluations": evaluations,
        "metrics": {
            "avg_hallucination_score": sum(e['overall_hallucination_score'] for e in evaluations) / len(evaluations),
            "avg_confidence": sum(e['confidence'] for e in evaluations) / len(evaluations),
            "best_score": min(e['overall_hallucination_score'] for e in evaluations),
            "worst_score": max(e['overall_hallucination_score'] for e in evaluations),
            "models": list(set(e['model_name'] for e in evaluations))
        }
    }
    
    return jsonify(comparison)

# ============ Real-time Dashboard ============

@app.route('/api/dashboard/stats', methods=['GET'])
@login_required
def get_dashboard_stats():
    """Get real-time dashboard statistics."""
    stats = db.get_dashboard_stats(current_user.id)
    return jsonify(stats)

@app.route('/api/dashboard/recent', methods=['GET'])
@login_required
def get_recent_evaluations():
    """Get recent evaluations for real-time updates."""
    since_str = request.args.get('since')
    if since_str:
        try:
            since = datetime.fromisoformat(since_str.replace('Z', '+00:00'))
        except:
            since = datetime.utcnow() - timedelta(minutes=5)
    else:
        since = datetime.utcnow() - timedelta(minutes=5)
    
    results = db.get_recent_evaluations(current_user.id, since)
    return jsonify({
        "results": results,
        "timestamp": datetime.utcnow().isoformat()
    })

@app.route('/api/dashboard/activity', methods=['GET'])
@login_required
def get_activity_feed():
    """Get activity feed for the dashboard."""
    limit = request.args.get('limit', 10, type=int)
    results = db.get_user_results(current_user.id, limit=limit)
    
    activities = []
    for r in results:
        activities.append({
            "id": r.id,
            "type": "evaluation",
            "title": f"Evaluated with {r.model_name}",
            "description": r.prompt[:80] + "..." if len(r.prompt) > 80 else r.prompt,
            "score": r.overall_hallucination_score,
            "is_hallucination": r.is_hallucination,
            "timestamp": r.timestamp.isoformat() if r.timestamp else None
        })
    
    return jsonify(activities)

# ============ Test Suites Page ============

@app.route('/test-suites')
@login_required
def test_suites_page():
    """Test suites management page."""
    return render_template('test_suites.html', active_page='test-suites')

# ==================== Phase 2 Feature Status ====================

@app.route('/api/features')
def get_features():
    """Get available Phase 2 features."""
    return jsonify({
        "phase": 2,
        "features": {
            "nlp_metrics": NLP_METRICS_AVAILABLE,
            "ab_testing": AB_TESTING_AVAILABLE,
            "rag_evaluation": RAG_AVAILABLE,
            "webhooks": WEBHOOKS_AVAILABLE,
            "test_suites": True,
            "prompt_versioning": True,
            "annotations": True,
            "comparison_view": True,
            "realtime_dashboard": True
        },
        "version": "1.3.0"
    })


@app.route('/health')
def health():
    """Health check endpoint for Render."""
    return jsonify({
        "status": "healthy",
        "database": "postgresql" if db.is_postgresql() else "sqlite",
        "version": "1.3.0",
        "phase2_features": {
            "nlp_metrics": NLP_METRICS_AVAILABLE,
            "ab_testing": AB_TESTING_AVAILABLE,
            "rag_evaluation": RAG_AVAILABLE,
            "webhooks": WEBHOOKS_AVAILABLE
        }
    })

if __name__ == '__main__':
    app.run(debug=Config.FLASK_DEBUG, host='0.0.0.0', port=5001)
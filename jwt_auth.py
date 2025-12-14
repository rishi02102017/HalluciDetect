"""JWT Authentication module for API endpoints."""
import jwt
from datetime import datetime, timedelta
from functools import wraps
from flask import request, jsonify, g
from config import Config


def generate_access_token(user_id: str, username: str) -> str:
    """Generate a JWT access token."""
    payload = {
        'user_id': user_id,
        'username': username,
        'type': 'access',
        'exp': datetime.utcnow() + timedelta(seconds=Config.JWT_ACCESS_TOKEN_EXPIRES),
        'iat': datetime.utcnow()
    }
    return jwt.encode(payload, Config.JWT_SECRET_KEY, algorithm='HS256')


def generate_refresh_token(user_id: str) -> str:
    """Generate a JWT refresh token."""
    payload = {
        'user_id': user_id,
        'type': 'refresh',
        'exp': datetime.utcnow() + timedelta(seconds=Config.JWT_REFRESH_TOKEN_EXPIRES),
        'iat': datetime.utcnow()
    }
    return jwt.encode(payload, Config.JWT_SECRET_KEY, algorithm='HS256')


def decode_token(token: str) -> dict:
    """Decode and validate a JWT token."""
    try:
        payload = jwt.decode(token, Config.JWT_SECRET_KEY, algorithms=['HS256'])
        return payload
    except jwt.ExpiredSignatureError:
        return {'error': 'Token has expired'}
    except jwt.InvalidTokenError:
        return {'error': 'Invalid token'}


def jwt_required(f):
    """Decorator to require JWT authentication for API endpoints."""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        
        # Check for token in Authorization header
        auth_header = request.headers.get('Authorization')
        if auth_header:
            parts = auth_header.split()
            if len(parts) == 2 and parts[0].lower() == 'bearer':
                token = parts[1]
        
        # Also check for token in query parameter (for some API use cases)
        if not token:
            token = request.args.get('api_token')
        
        if not token:
            return jsonify({
                'error': 'Authentication required',
                'message': 'Please provide a valid API token in the Authorization header (Bearer <token>)'
            }), 401
        
        # Decode and validate token
        payload = decode_token(token)
        
        if 'error' in payload:
            return jsonify({
                'error': 'Authentication failed',
                'message': payload['error']
            }), 401
        
        if payload.get('type') != 'access':
            return jsonify({
                'error': 'Invalid token type',
                'message': 'Please use an access token'
            }), 401
        
        # Store user info in flask g object for use in the route
        g.jwt_user_id = payload.get('user_id')
        g.jwt_username = payload.get('username')
        
        return f(*args, **kwargs)
    return decorated


def jwt_optional(f):
    """Decorator for optional JWT authentication."""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        g.jwt_user_id = None
        g.jwt_username = None
        
        auth_header = request.headers.get('Authorization')
        if auth_header:
            parts = auth_header.split()
            if len(parts) == 2 and parts[0].lower() == 'bearer':
                token = parts[1]
        
        if token:
            payload = decode_token(token)
            if 'error' not in payload and payload.get('type') == 'access':
                g.jwt_user_id = payload.get('user_id')
                g.jwt_username = payload.get('username')
        
        return f(*args, **kwargs)
    return decorated


def get_tokens_for_user(user_id: str, username: str) -> dict:
    """Generate both access and refresh tokens for a user."""
    return {
        'access_token': generate_access_token(user_id, username),
        'refresh_token': generate_refresh_token(user_id),
        'token_type': 'Bearer',
        'expires_in': Config.JWT_ACCESS_TOKEN_EXPIRES
    }


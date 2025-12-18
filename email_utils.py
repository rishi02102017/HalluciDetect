"""
Email utility functions for HalluciDetect.
Handles sending emails for password reset and email verification.
"""
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Email configuration from environment
MAIL_SERVER = os.getenv('MAIL_SERVER', 'smtp.gmail.com')
MAIL_PORT = int(os.getenv('MAIL_PORT', 587))
MAIL_USE_TLS = os.getenv('MAIL_USE_TLS', 'True').lower() == 'true'
MAIL_USERNAME = os.getenv('MAIL_USERNAME')
MAIL_PASSWORD = os.getenv('MAIL_PASSWORD')
MAIL_DEFAULT_SENDER = os.getenv('MAIL_DEFAULT_SENDER', MAIL_USERNAME)

def is_email_configured() -> bool:
    """Check if email is properly configured."""
    return bool(MAIL_USERNAME and MAIL_PASSWORD)

def send_email(to_email: str, subject: str, html_body: str, text_body: str = None) -> bool:
    """
    Send an email.
    Returns True if successful, False otherwise.
    """
    if not is_email_configured():
        print("Email not configured. Set MAIL_USERNAME and MAIL_PASSWORD in .env")
        return False
    
    try:
        # Create message
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = MAIL_DEFAULT_SENDER
        msg['To'] = to_email
        
        # Add text and HTML parts
        if text_body:
            part1 = MIMEText(text_body, 'plain')
            msg.attach(part1)
        
        part2 = MIMEText(html_body, 'html')
        msg.attach(part2)
        
        # Connect to server and send
        with smtplib.SMTP(MAIL_SERVER, MAIL_PORT) as server:
            if MAIL_USE_TLS:
                server.starttls()
            server.login(MAIL_USERNAME, MAIL_PASSWORD)
            server.sendmail(MAIL_DEFAULT_SENDER, to_email, msg.as_string())
        
        print(f"Email sent successfully to {to_email}")
        return True
        
    except Exception as e:
        print(f"Failed to send email: {e}")
        return False

def send_password_reset_email(to_email: str, username: str, reset_url: str) -> bool:
    """Send password reset email."""
    subject = "Reset Your HalluciDetect Password"
    
    html_body = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0a0a0f; color: #f1f5f9; padding: 40px 20px; }}
            .container {{ max-width: 500px; margin: 0 auto; background: #16161f; border-radius: 12px; padding: 40px; border: 1px solid #2a2a3a; }}
            .logo {{ text-align: center; margin-bottom: 30px; }}
            .logo-text {{ font-size: 24px; font-weight: 700; color: #6366f1; }}
            h1 {{ color: #f1f5f9; font-size: 20px; margin-bottom: 16px; }}
            p {{ color: #94a3b8; line-height: 1.6; margin-bottom: 16px; }}
            .btn {{ display: inline-block; background: linear-gradient(135deg, #6366f1, #8b5cf6); color: white; padding: 14px 28px; border-radius: 8px; text-decoration: none; font-weight: 600; margin: 20px 0; }}
            .footer {{ margin-top: 30px; padding-top: 20px; border-top: 1px solid #2a2a3a; font-size: 12px; color: #64748b; }}
            .link {{ color: #6366f1; word-break: break-all; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="logo">
                <span class="logo-text">🔍 HalluciDetect</span>
            </div>
            <h1>Reset Your Password</h1>
            <p>Hi {username},</p>
            <p>We received a request to reset your password. Click the button below to create a new password:</p>
            <a href="{reset_url}" class="btn">Reset Password</a>
            <p>This link will expire in 1 hour for security reasons.</p>
            <p>If you didn't request this, you can safely ignore this email.</p>
            <div class="footer">
                <p>Or copy this link: <span class="link">{reset_url}</span></p>
                <p>© 2025 HalluciDetect - LLM Evaluation Platform</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    text_body = f"""
    Reset Your HalluciDetect Password
    
    Hi {username},
    
    We received a request to reset your password. Click the link below to create a new password:
    
    {reset_url}
    
    This link will expire in 1 hour for security reasons.
    
    If you didn't request this, you can safely ignore this email.
    
    © 2025 HalluciDetect - LLM Evaluation Platform
    """
    
    return send_email(to_email, subject, html_body, text_body)

def send_verification_email(to_email: str, username: str, verify_url: str) -> bool:
    """Send email verification email."""
    subject = "Verify Your HalluciDetect Account"
    
    html_body = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0a0a0f; color: #f1f5f9; padding: 40px 20px; }}
            .container {{ max-width: 500px; margin: 0 auto; background: #16161f; border-radius: 12px; padding: 40px; border: 1px solid #2a2a3a; }}
            .logo {{ text-align: center; margin-bottom: 30px; }}
            .logo-text {{ font-size: 24px; font-weight: 700; color: #6366f1; }}
            h1 {{ color: #f1f5f9; font-size: 20px; margin-bottom: 16px; }}
            p {{ color: #94a3b8; line-height: 1.6; margin-bottom: 16px; }}
            .btn {{ display: inline-block; background: linear-gradient(135deg, #6366f1, #8b5cf6); color: white; padding: 14px 28px; border-radius: 8px; text-decoration: none; font-weight: 600; margin: 20px 0; }}
            .footer {{ margin-top: 30px; padding-top: 20px; border-top: 1px solid #2a2a3a; font-size: 12px; color: #64748b; }}
            .link {{ color: #6366f1; word-break: break-all; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="logo">
                <span class="logo-text">🔍 HalluciDetect</span>
            </div>
            <h1>Verify Your Email</h1>
            <p>Hi {username},</p>
            <p>Thanks for signing up! Please verify your email address by clicking the button below:</p>
            <a href="{verify_url}" class="btn">Verify Email</a>
            <p>This helps us ensure the security of your account.</p>
            <div class="footer">
                <p>Or copy this link: <span class="link">{verify_url}</span></p>
                <p>© 2025 HalluciDetect - LLM Evaluation Platform</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    text_body = f"""
    Verify Your HalluciDetect Account
    
    Hi {username},
    
    Thanks for signing up! Please verify your email address by clicking the link below:
    
    {verify_url}
    
    This helps us ensure the security of your account.
    
    © 2025 HalluciDetect - LLM Evaluation Platform
    """
    
    return send_email(to_email, subject, html_body, text_body)

def send_welcome_email(to_email: str, username: str) -> bool:
    """Send welcome email after verification."""
    subject = "Welcome to HalluciDetect! 🎉"
    
    html_body = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0a0a0f; color: #f1f5f9; padding: 40px 20px; }}
            .container {{ max-width: 500px; margin: 0 auto; background: #16161f; border-radius: 12px; padding: 40px; border: 1px solid #2a2a3a; }}
            .logo {{ text-align: center; margin-bottom: 30px; }}
            .logo-text {{ font-size: 24px; font-weight: 700; color: #6366f1; }}
            h1 {{ color: #f1f5f9; font-size: 20px; margin-bottom: 16px; }}
            p {{ color: #94a3b8; line-height: 1.6; margin-bottom: 16px; }}
            .feature {{ background: #1a1a24; padding: 16px; border-radius: 8px; margin: 12px 0; }}
            .feature-title {{ color: #f1f5f9; font-weight: 600; margin-bottom: 4px; }}
            .footer {{ margin-top: 30px; padding-top: 20px; border-top: 1px solid #2a2a3a; font-size: 12px; color: #64748b; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="logo">
                <span class="logo-text">🔍 HalluciDetect</span>
            </div>
            <h1>Welcome aboard, {username}! 🎉</h1>
            <p>Your email is now verified and your account is ready to use.</p>
            <p>Here's what you can do:</p>
            <div class="feature">
                <div class="feature-title">🧪 Playground</div>
                <p style="margin: 0; font-size: 14px;">Test LLM outputs for hallucinations in real-time</p>
            </div>
            <div class="feature">
                <div class="feature-title">📊 Analytics</div>
                <p style="margin: 0; font-size: 14px;">Track hallucination rates across models and prompts</p>
            </div>
            <div class="feature">
                <div class="feature-title">📋 Templates</div>
                <p style="margin: 0; font-size: 14px;">Save and reuse your evaluation prompts</p>
            </div>
            <div class="footer">
                <p>Happy evaluating!</p>
                <p>© 2025 HalluciDetect - LLM Evaluation Platform</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    text_body = f"""
    Welcome aboard, {username}! 🎉
    
    Your email is now verified and your account is ready to use.
    
    Here's what you can do:
    - Playground: Test LLM outputs for hallucinations in real-time
    - Analytics: Track hallucination rates across models and prompts
    - Templates: Save and reuse your evaluation prompts
    
    Happy evaluating!
    
    © 2025 HalluciDetect - LLM Evaluation Platform
    """
    
    return send_email(to_email, subject, html_body, text_body)


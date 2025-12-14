"""
Webhook & Notifications Module.
Phase 2 Feature: Webhook Endpoints & Slack Notifications

Supports:
- Custom webhook endpoints
- Slack notifications
- Alert rules and thresholds
"""
import json
import requests
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from enum import Enum
import uuid


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class WebhookConfig:
    """Configuration for a webhook endpoint."""
    id: str
    name: str
    url: str
    enabled: bool = True
    secret: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    events: List[str] = field(default_factory=list)  # Which events to send
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    # Stats
    total_sent: int = 0
    total_failures: int = 0
    last_sent: Optional[datetime] = None
    last_error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "url": self.url,
            "enabled": self.enabled,
            "events": self.events,
            "created_at": self.created_at.isoformat(),
            "total_sent": self.total_sent,
            "total_failures": self.total_failures,
            "last_sent": self.last_sent.isoformat() if self.last_sent else None,
            "last_error": self.last_error
        }


@dataclass
class AlertRule:
    """Rule for triggering alerts based on evaluation results."""
    id: str
    name: str
    condition: str  # e.g., "hallucination_score > 0.7"
    threshold: float
    metric: str  # e.g., "hallucination_score", "confidence"
    operator: str  # >, <, >=, <=, ==
    alert_level: AlertLevel = AlertLevel.WARNING
    enabled: bool = True
    cooldown_minutes: int = 5  # Minimum time between alerts
    last_triggered: Optional[datetime] = None
    
    def check(self, value: float) -> bool:
        """Check if the rule is triggered."""
        if not self.enabled:
            return False
        
        ops = {
            ">": lambda v, t: v > t,
            "<": lambda v, t: v < t,
            ">=": lambda v, t: v >= t,
            "<=": lambda v, t: v <= t,
            "==": lambda v, t: abs(v - t) < 0.001
        }
        
        op_func = ops.get(self.operator, lambda v, t: False)
        return op_func(value, self.threshold)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "condition": self.condition,
            "metric": self.metric,
            "operator": self.operator,
            "threshold": self.threshold,
            "alert_level": self.alert_level.value,
            "enabled": self.enabled,
            "cooldown_minutes": self.cooldown_minutes,
            "last_triggered": self.last_triggered.isoformat() if self.last_triggered else None
        }


class SlackNotifier:
    """Send notifications to Slack via webhooks."""
    
    def __init__(self, webhook_url: Optional[str] = None):
        self.webhook_url = webhook_url
    
    def set_webhook_url(self, url: str):
        """Set the Slack webhook URL."""
        self.webhook_url = url
    
    def send(
        self, 
        message: str, 
        title: Optional[str] = None,
        level: AlertLevel = AlertLevel.INFO,
        fields: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Send a message to Slack.
        
        Args:
            message: Main message text
            title: Optional title/header
            level: Alert level (affects color)
            fields: Additional key-value fields to display
        
        Returns:
            Dict with success status and details
        """
        if not self.webhook_url:
            return {"success": False, "error": "Slack webhook URL not configured"}
        
        # Color based on level
        colors = {
            AlertLevel.INFO: "#36a64f",      # Green
            AlertLevel.WARNING: "#ff9800",   # Orange
            AlertLevel.ERROR: "#f44336",     # Red
            AlertLevel.CRITICAL: "#9c27b0"   # Purple
        }
        
        color = colors.get(level, "#36a64f")
        
        # Build Slack attachment
        attachment = {
            "color": color,
            "title": title or "HalluciDetect Alert",
            "text": message,
            "footer": "HalluciDetect",
            "ts": int(datetime.utcnow().timestamp())
        }
        
        if fields:
            attachment["fields"] = [
                {"title": k, "value": str(v), "short": True}
                for k, v in fields.items()
            ]
        
        payload = {
            "attachments": [attachment]
        }
        
        try:
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                return {"success": True, "status_code": 200}
            else:
                return {
                    "success": False, 
                    "status_code": response.status_code,
                    "error": response.text
                }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def send_evaluation_alert(
        self,
        prompt: str,
        hallucination_score: float,
        model_name: str,
        details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Send a formatted evaluation alert."""
        # Determine level based on score
        if hallucination_score >= 0.8:
            level = AlertLevel.CRITICAL
            status = "Critical Hallucination Detected"
        elif hallucination_score >= 0.6:
            level = AlertLevel.ERROR
            status = "High Hallucination Risk"
        elif hallucination_score >= 0.4:
            level = AlertLevel.WARNING
            status = "Moderate Hallucination"
        else:
            level = AlertLevel.INFO
            status = "Low Hallucination"
        
        fields = {
            "Score": f"{hallucination_score:.1%}",
            "Model": model_name,
            "Prompt": prompt[:50] + "..." if len(prompt) > 50 else prompt
        }
        
        if details:
            fields.update({k: str(v)[:50] for k, v in list(details.items())[:3]})
        
        return self.send(
            message=f"Hallucination score: {hallucination_score:.1%}",
            title=status,
            level=level,
            fields=fields
        )


class WebhookManager:
    """Manage webhook endpoints and notifications."""
    
    def __init__(self):
        self.webhooks: Dict[str, WebhookConfig] = {}
        self.alert_rules: Dict[str, AlertRule] = {}
        self.slack = SlackNotifier()
        
        # Event history (in-memory, limited size)
        self.event_history: List[Dict[str, Any]] = []
        self.max_history = 100
    
    # ==================== Webhook Management ====================
    
    def add_webhook(
        self,
        name: str,
        url: str,
        events: Optional[List[str]] = None,
        headers: Optional[Dict[str, str]] = None,
        secret: Optional[str] = None
    ) -> WebhookConfig:
        """Add a new webhook endpoint."""
        webhook_id = str(uuid.uuid4())[:8]
        
        webhook = WebhookConfig(
            id=webhook_id,
            name=name,
            url=url,
            events=events or ["evaluation.completed", "alert.triggered"],
            headers=headers or {},
            secret=secret
        )
        
        self.webhooks[webhook_id] = webhook
        return webhook
    
    def remove_webhook(self, webhook_id: str) -> bool:
        """Remove a webhook."""
        if webhook_id in self.webhooks:
            del self.webhooks[webhook_id]
            return True
        return False
    
    def enable_webhook(self, webhook_id: str, enabled: bool = True) -> bool:
        """Enable or disable a webhook."""
        if webhook_id in self.webhooks:
            self.webhooks[webhook_id].enabled = enabled
            return True
        return False
    
    def list_webhooks(self) -> List[Dict[str, Any]]:
        """List all webhooks."""
        return [w.to_dict() for w in self.webhooks.values()]
    
    # ==================== Sending Webhooks ====================
    
    def send_webhook(
        self,
        event_type: str,
        payload: Dict[str, Any],
        webhook_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send webhook to endpoints.
        
        Args:
            event_type: Type of event (e.g., "evaluation.completed")
            payload: Event data
            webhook_id: Optional specific webhook to send to
        
        Returns:
            Dict with results for each webhook
        """
        results = {}
        
        webhooks_to_send = (
            [self.webhooks[webhook_id]] if webhook_id and webhook_id in self.webhooks
            else self.webhooks.values()
        )
        
        for webhook in webhooks_to_send:
            if not webhook.enabled:
                continue
            
            # Check if webhook subscribes to this event
            if webhook.events and event_type not in webhook.events:
                continue
            
            # Prepare payload
            full_payload = {
                "event": event_type,
                "timestamp": datetime.utcnow().isoformat(),
                "data": payload
            }
            
            # Send request
            try:
                headers = {"Content-Type": "application/json"}
                headers.update(webhook.headers)
                
                response = requests.post(
                    webhook.url,
                    json=full_payload,
                    headers=headers,
                    timeout=10
                )
                
                success = 200 <= response.status_code < 300
                
                webhook.last_sent = datetime.utcnow()
                if success:
                    webhook.total_sent += 1
                else:
                    webhook.total_failures += 1
                    webhook.last_error = f"HTTP {response.status_code}"
                
                results[webhook.id] = {
                    "success": success,
                    "status_code": response.status_code
                }
                
            except Exception as e:
                webhook.total_failures += 1
                webhook.last_error = str(e)
                results[webhook.id] = {
                    "success": False,
                    "error": str(e)
                }
        
        # Log event
        self._log_event(event_type, payload, results)
        
        return results
    
    def _log_event(
        self, 
        event_type: str, 
        payload: Dict[str, Any], 
        results: Dict[str, Any]
    ):
        """Log event to history."""
        self.event_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "event": event_type,
            "payload_preview": str(payload)[:200],
            "results": results
        })
        
        # Trim history
        if len(self.event_history) > self.max_history:
            self.event_history = self.event_history[-self.max_history:]
    
    # ==================== Alert Rules ====================
    
    def add_alert_rule(
        self,
        name: str,
        metric: str,
        operator: str,
        threshold: float,
        level: AlertLevel = AlertLevel.WARNING,
        cooldown_minutes: int = 5
    ) -> AlertRule:
        """Add an alert rule."""
        rule_id = str(uuid.uuid4())[:8]
        
        rule = AlertRule(
            id=rule_id,
            name=name,
            condition=f"{metric} {operator} {threshold}",
            metric=metric,
            operator=operator,
            threshold=threshold,
            alert_level=level,
            cooldown_minutes=cooldown_minutes
        )
        
        self.alert_rules[rule_id] = rule
        return rule
    
    def check_alerts(
        self, 
        metrics: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """
        Check all alert rules against metrics.
        
        Args:
            metrics: Dict of metric_name -> value
        
        Returns:
            List of triggered alerts
        """
        triggered = []
        now = datetime.utcnow()
        
        for rule in self.alert_rules.values():
            if not rule.enabled:
                continue
            
            if rule.metric not in metrics:
                continue
            
            value = metrics[rule.metric]
            
            # Check cooldown
            if rule.last_triggered:
                elapsed = (now - rule.last_triggered).total_seconds() / 60
                if elapsed < rule.cooldown_minutes:
                    continue
            
            if rule.check(value):
                rule.last_triggered = now
                
                alert = {
                    "rule_id": rule.id,
                    "rule_name": rule.name,
                    "condition": rule.condition,
                    "metric": rule.metric,
                    "value": value,
                    "threshold": rule.threshold,
                    "level": rule.alert_level.value,
                    "triggered_at": now.isoformat()
                }
                
                triggered.append(alert)
                
                # Send to webhooks
                self.send_webhook("alert.triggered", alert)
        
        return triggered
    
    def list_alert_rules(self) -> List[Dict[str, Any]]:
        """List all alert rules."""
        return [r.to_dict() for r in self.alert_rules.values()]
    
    # ==================== Slack Integration ====================
    
    def configure_slack(self, webhook_url: str):
        """Configure Slack webhook."""
        self.slack.set_webhook_url(webhook_url)
    
    def send_slack_alert(
        self,
        message: str,
        title: Optional[str] = None,
        level: AlertLevel = AlertLevel.INFO,
        fields: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Send alert to Slack."""
        return self.slack.send(message, title, level, fields)
    
    # ==================== Convenience Methods ====================
    
    def notify_evaluation(
        self,
        result: Dict[str, Any],
        send_slack: bool = True
    ) -> Dict[str, Any]:
        """
        Send notifications for an evaluation result.
        
        Args:
            result: Evaluation result dict
            send_slack: Whether to also send to Slack
        
        Returns:
            Dict with notification results
        """
        # Send to webhooks
        webhook_results = self.send_webhook("evaluation.completed", result)
        
        # Check alert rules
        metrics = {
            "hallucination_score": result.get("hallucination_score", 0),
            "confidence": result.get("confidence", 0),
            "fact_check_score": result.get("scores", {}).get("fact_check", 0),
            "semantic_score": result.get("scores", {}).get("semantic_similarity", 0)
        }
        
        triggered_alerts = self.check_alerts(metrics)
        
        # Send to Slack if high risk
        slack_result = None
        if send_slack and self.slack.webhook_url:
            score = result.get("hallucination_score", 0)
            if score >= 0.5:  # Only alert for medium+ risk
                slack_result = self.slack.send_evaluation_alert(
                    prompt=result.get("prompt", "")[:100],
                    hallucination_score=score,
                    model_name=result.get("model_name", "unknown"),
                    details=result.get("scores")
                )
        
        return {
            "webhooks": webhook_results,
            "alerts_triggered": triggered_alerts,
            "slack": slack_result
        }
    
    def get_event_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent event history."""
        return self.event_history[-limit:]


# Singleton
_webhook_manager = None

def get_webhook_manager() -> WebhookManager:
    """Get or create webhook manager instance."""
    global _webhook_manager
    if _webhook_manager is None:
        _webhook_manager = WebhookManager()
    return _webhook_manager


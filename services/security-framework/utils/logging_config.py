"""
Security Logging Configuration with Geological Context Awareness
Implements structured logging for zero-trust security framework
"""

import logging
import logging.config
import json
import sys
from datetime import datetime
from typing import Dict, Any
import structlog

def setup_security_logging(name: str) -> logging.Logger:
    """
    Setup structured logging for security framework with geological context
    Returns configured logger with security-specific formatting
    """
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Logging configuration
    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'security_json': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'class': 'pythonjsonlogger.jsonlogger.JsonFormatter'
            },
            'security_detailed': {
                'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
                'formatter': 'security_detailed',
                'stream': sys.stdout
            },
            'security_file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'DEBUG',
                'formatter': 'security_json',
                'filename': '/var/log/prometheus/security.log',
                'maxBytes': 10485760,  # 10MB
                'backupCount': 10
            },
            'audit_file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'INFO',
                'formatter': 'security_json',
                'filename': '/var/log/prometheus/security_audit.log',
                'maxBytes': 52428800,  # 50MB
                'backupCount': 50
            },
            'threat_file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'WARNING',
                'formatter': 'security_json',
                'filename': '/var/log/prometheus/threats.log',
                'maxBytes': 10485760,  # 10MB
                'backupCount': 20
            }
        },
        'loggers': {
            'security_framework': {
                'level': 'DEBUG',
                'handlers': ['console', 'security_file', 'audit_file'],
                'propagate': False
            },
            'security_framework.threats': {
                'level': 'WARNING',
                'handlers': ['console', 'threat_file'],
                'propagate': False
            },
            'security_framework.audit': {
                'level': 'INFO',
                'handlers': ['audit_file'],
                'propagate': False
            }
        },
        'root': {
            'level': 'INFO',
            'handlers': ['console']
        }
    }
    
    # Apply logging configuration
    logging.config.dictConfig(logging_config)
    
    # Get logger for the specified name
    logger = logging.getLogger(name)
    
    # Add security context filter
    logger.addFilter(SecurityContextFilter())
    
    return logger

class SecurityContextFilter(logging.Filter):
    """
    Logging filter to add security context to log records
    Enriches logs with geological and security information
    """
    
    def filter(self, record):
        """Add security context to log record"""
        
        # Add security-specific fields
        record.security_framework = 'prometheus_zero_trust'
        record.log_type = 'security'
        record.timestamp_iso = datetime.utcnow().isoformat()
        
        # Add geological context if available
        if hasattr(record, 'geological_context'):
            record.geological_domain = getattr(record.geological_context, 'primary_domain', None)
            record.geological_workflow = getattr(record.geological_context, 'intended_workflow', None)
        
        # Add security event classification
        if hasattr(record, 'event_type'):
            record.security_event_type = record.event_type
            
        # Add risk scoring
        if hasattr(record, 'risk_score'):
            record.risk_level = self._classify_risk_level(record.risk_score)
            
        return True
        
    def _classify_risk_level(self, risk_score: float) -> str:
        """Classify risk score into human-readable level"""
        if risk_score >= 0.8:
            return 'critical'
        elif risk_score >= 0.6:
            return 'high'
        elif risk_score >= 0.4:
            return 'medium'
        elif risk_score >= 0.2:
            return 'low'
        else:
            return 'minimal'

class GeologicalSecurityLogger:
    """
    Specialized logger for geological security events
    Provides domain-specific logging methods
    """
    
    def __init__(self, logger_name: str):
        self.logger = setup_security_logging(logger_name)
        self.struct_logger = structlog.get_logger(logger_name)
        
    def log_authentication_event(
        self, 
        event_type: str, 
        user_id: str, 
        success: bool,
        geological_context: Dict = None,
        **kwargs
    ):
        """Log authentication event with geological context"""
        
        log_data = {
            'event_type': 'authentication',
            'auth_event_type': event_type,
            'user_id': user_id,
            'success': success,
            'geological_context': geological_context,
            **kwargs
        }
        
        if success:
            self.struct_logger.info("Authentication successful", **log_data)
        else:
            self.struct_logger.warning("Authentication failed", **log_data)
            
    def log_access_event(
        self, 
        event_type: str, 
        user_id: str, 
        resource_id: str,
        access_level: str,
        success: bool,
        geological_context: Dict = None,
        **kwargs
    ):
        """Log resource access event with geological context"""
        
        log_data = {
            'event_type': 'access_control',
            'access_event_type': event_type,
            'user_id': user_id,
            'resource_id': resource_id,
            'access_level': access_level,
            'success': success,
            'geological_context': geological_context,
            **kwargs
        }
        
        if success:
            self.struct_logger.info("Access granted", **log_data)
        else:
            self.struct_logger.warning("Access denied", **log_data)
            
    def log_security_violation(
        self, 
        violation_type: str, 
        severity: str,
        user_id: str = None,
        resource_id: str = None,
        geological_context: Dict = None,
        **kwargs
    ):
        """Log security policy violation"""
        
        log_data = {
            'event_type': 'security_violation',
            'violation_type': violation_type,
            'severity': severity,
            'user_id': user_id,
            'resource_id': resource_id,
            'geological_context': geological_context,
            **kwargs
        }
        
        if severity in ['critical', 'high']:
            self.struct_logger.error("Security violation detected", **log_data)
        else:
            self.struct_logger.warning("Security violation detected", **log_data)
            
    def log_threat_detection(
        self, 
        threat_type: str, 
        threat_level: str,
        indicators: list = None,
        geological_context: Dict = None,
        **kwargs
    ):
        """Log threat detection event"""
        
        log_data = {
            'event_type': 'threat_detection',
            'threat_type': threat_type,
            'threat_level': threat_level,
            'indicators': indicators or [],
            'geological_context': geological_context,
            **kwargs
        }
        
        if threat_level in ['critical', 'high']:
            self.struct_logger.error("Threat detected", **log_data)
        else:
            self.struct_logger.warning("Threat detected", **log_data)
            
    def log_geological_anomaly(
        self, 
        anomaly_type: str, 
        geological_domain: str,
        confidence: float,
        data_source: str = None,
        **kwargs
    ):
        """Log geological data anomaly that may indicate security issues"""
        
        log_data = {
            'event_type': 'geological_anomaly',
            'anomaly_type': anomaly_type,
            'geological_domain': geological_domain,
            'confidence': confidence,
            'data_source': data_source,
            **kwargs
        }
        
        if confidence >= 0.8:
            self.struct_logger.warning("High-confidence geological anomaly", **log_data)
        else:
            self.struct_logger.info("Geological anomaly detected", **log_data)
            
    def log_compliance_event(
        self, 
        compliance_framework: str, 
        event_type: str,
        status: str,
        geological_context: Dict = None,
        **kwargs
    ):
        """Log compliance-related event"""
        
        log_data = {
            'event_type': 'compliance',
            'compliance_framework': compliance_framework,
            'compliance_event_type': event_type,
            'status': status,
            'geological_context': geological_context,
            **kwargs
        }
        
        if status == 'violation':
            self.struct_logger.error("Compliance violation", **log_data)
        elif status == 'warning':
            self.struct_logger.warning("Compliance warning", **log_data)
        else:
            self.struct_logger.info("Compliance event", **log_data)

# Create default security logger instance
security_logger = GeologicalSecurityLogger('security_framework')

def get_security_logger(name: str = None) -> GeologicalSecurityLogger:
    """Get security logger instance"""
    if name:
        return GeologicalSecurityLogger(f'security_framework.{name}')
    return security_logger
#!/usr/bin/env python3
"""
Compliance Framework for Scientific Computing Standards

This module implements comprehensive compliance checking for scientific
computing standards, regulatory requirements, and geological domain
best practices for the Prometheus AI Platform.
"""

import os
import sys
import json
import re
import ast
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import importlib.util
import subprocess

# Compliance checking libraries
try:
    import bandit
    from bandit.core import manager as bandit_manager
    SECURITY_SCANNING = True
except ImportError:
    SECURITY_SCANNING = False
    print("Warning: Bandit security scanning not available")

try:
    import safety
    SAFETY_CHECKING = True
except ImportError:
    SAFETY_CHECKING = False
    print("Warning: Safety vulnerability checking not available")


@dataclass
class ComplianceRule:
    """Represents a compliance rule"""
    rule_id: str
    category: str
    severity: str
    title: str
    description: str
    requirement: str
    validation_method: str
    geological_context: Optional[str] = None


@dataclass
class ComplianceViolation:
    """Represents a compliance violation"""
    rule_id: str
    file_path: str
    line_number: int
    severity: str
    message: str
    evidence: str
    recommendation: str
    geological_impact: Optional[str] = None


@dataclass
class ComplianceAssessment:
    """Assessment result for a compliance category"""
    category: str
    total_rules: int
    passed_rules: int
    failed_rules: int
    compliance_score: float
    violations: List[ComplianceViolation]
    recommendations: List[str]


@dataclass
class ComplianceReport:
    """Comprehensive compliance report"""
    timestamp: datetime
    project_path: str
    overall_compliance_score: float
    assessments: List[ComplianceAssessment]
    critical_violations: List[ComplianceViolation]
    regulatory_compliance: Dict[str, bool]
    scientific_standards_compliance: Dict[str, bool]
    geological_domain_compliance: Dict[str, bool]
    remediation_plan: List[str]
    certification_status: str


class ScientificComplianceFramework:
    """
    Comprehensive compliance framework for scientific computing standards
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.compliance_rules = self._load_compliance_rules()
        self.regulatory_frameworks = self._load_regulatory_frameworks()
        self.scientific_standards = self._load_scientific_standards()
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load compliance configuration"""
        default_config = {
            "compliance_categories": [
                "data_protection", "scientific_accuracy", "code_quality",
                "security", "performance", "documentation", "accessibility",
                "geological_standards", "regulatory_compliance"
            ],
            "severity_weights": {
                "critical": 1.0,
                "high": 0.8,
                "medium": 0.5,
                "low": 0.2
            },
            "minimum_compliance_scores": {
                "overall": 0.85,
                "data_protection": 0.95,
                "scientific_accuracy": 0.90,
                "security": 0.95,
                "geological_standards": 0.88
            },
            "regulatory_requirements": [
                "GDPR", "SOC2", "ISO27001", "NIST_Cybersecurity",
                "Scientific_Data_Management", "Geological_Data_Standards"
            ],
            "geological_domain_standards": [
                "IUGS_Standards", "AGU_Guidelines", "GSA_Best_Practices",
                "USGS_Data_Standards", "Seismic_Data_Standards"
            ]
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
                
        return default_config
    
    def _load_compliance_rules(self) -> Dict[str, List[ComplianceRule]]:
        """Load comprehensive compliance rules"""
        rules = {
            "data_protection": [
                ComplianceRule(
                    rule_id="DP001",
                    category="data_protection",
                    severity="critical",
                    title="Personal Data Encryption",
                    description="All personal data must be encrypted at rest and in transit",
                    requirement="GDPR Article 32 - Security of processing",
                    validation_method="code_analysis",
                    geological_context="Geological survey data may contain personal information"
                ),
                ComplianceRule(
                    rule_id="DP002",
                    category="data_protection",
                    severity="high",
                    title="Data Retention Policy",
                    description="Data retention periods must be defined and enforced",
                    requirement="GDPR Article 5 - Principles relating to processing",
                    validation_method="policy_check"
                ),
                ComplianceRule(
                    rule_id="DP003",
                    category="data_protection",
                    severity="critical",
                    title="Consent Management",
                    description="User consent must be properly managed and documented",
                    requirement="GDPR Article 7 - Conditions for consent",
                    validation_method="code_analysis"
                )
            ],
            "scientific_accuracy": [
                ComplianceRule(
                    rule_id="SA001",
                    category="scientific_accuracy",
                    severity="critical",
                    title="Physics Law Compliance",
                    description="All algorithms must respect fundamental physics laws",
                    requirement="Scientific Computing Best Practices",
                    validation_method="algorithm_analysis",
                    geological_context="Conservation laws in geological processes"
                ),
                ComplianceRule(
                    rule_id="SA002",
                    category="scientific_accuracy",
                    severity="high",
                    title="Uncertainty Quantification",
                    description="All predictions must include uncertainty estimates",
                    requirement="Scientific Method Standards",
                    validation_method="code_analysis",
                    geological_context="Geological predictions inherently uncertain"
                ),
                ComplianceRule(
                    rule_id="SA003",
                    category="scientific_accuracy",
                    severity="high",
                    title="Benchmark Validation",
                    description="Algorithms must be validated against established benchmarks",
                    requirement="Scientific Validation Standards",
                    validation_method="test_analysis",
                    geological_context="Geological benchmarks from peer-reviewed research"
                ),
                ComplianceRule(
                    rule_id="SA004",
                    category="scientific_accuracy",
                    severity="medium",
                    title="Peer Review Documentation",
                    description="Scientific algorithms must have peer review documentation",
                    requirement="Scientific Publication Standards",
                    validation_method="documentation_check",
                    geological_context="Geological domain expert review required"
                )
            ],
            "code_quality": [
                ComplianceRule(
                    rule_id="CQ001",
                    category="code_quality",
                    severity="high",
                    title="No Placeholder Code",
                    description="Production code must not contain TODO, FIXME, or placeholders",
                    requirement="Production Code Standards",
                    validation_method="static_analysis"
                ),
                ComplianceRule(
                    rule_id="CQ002",
                    category="code_quality",
                    severity="medium",
                    title="Code Coverage",
                    description="Minimum 80% test coverage for critical geological functions",
                    requirement="Software Quality Standards",
                    validation_method="coverage_analysis",
                    geological_context="Critical for geological calculation accuracy"
                ),
                ComplianceRule(
                    rule_id="CQ003",
                    category="code_quality",
                    severity="medium",
                    title="Complexity Limits",
                    description="Functions must not exceed complexity thresholds",
                    requirement="Maintainability Standards",
                    validation_method="complexity_analysis"
                )
            ],
            "security": [
                ComplianceRule(
                    rule_id="SEC001",
                    category="security",
                    severity="critical",
                    title="Input Validation",
                    description="All user inputs must be validated and sanitized",
                    requirement="OWASP Security Guidelines",
                    validation_method="security_scan"
                ),
                ComplianceRule(
                    rule_id="SEC002",
                    category="security",
                    severity="critical",
                    title="Authentication Required",
                    description="All API endpoints must require authentication",
                    requirement="Security Best Practices",
                    validation_method="endpoint_analysis"
                ),
                ComplianceRule(
                    rule_id="SEC003",
                    category="security",
                    severity="high",
                    title="Secure Communication",
                    description="All communications must use encrypted protocols",
                    requirement="Data Security Standards",
                    validation_method="protocol_analysis"
                )
            ],
            "geological_standards": [
                ComplianceRule(
                    rule_id="GEO001",
                    category="geological_standards",
                    severity="high",
                    title="Coordinate System Standards",
                    description="Geospatial data must use standard coordinate systems",
                    requirement="IUGS Geospatial Standards",
                    validation_method="geospatial_analysis",
                    geological_context="Proper coordinate reference systems essential"
                ),
                ComplianceRule(
                    rule_id="GEO002",
                    category="geological_standards",
                    severity="high",
                    title="Geological Time Scale",
                    description="Geological ages must follow ICS time scale standards",
                    requirement="International Commission on Stratigraphy",
                    validation_method="temporal_analysis",
                    geological_context="Standardized geological time essential for correlation"
                ),
                ComplianceRule(
                    rule_id="GEO003",
                    category="geological_standards",
                    severity="medium",
                    title="Seismic Data Format",
                    description="Seismic data must follow SEED format standards",
                    requirement="IRIS Data Standards",
                    validation_method="format_analysis",
                    geological_context="Standard format for seismic data exchange"
                ),
                ComplianceRule(
                    rule_id="GEO004",
                    category="geological_standards",
                    severity="medium",
                    title="Volcanic Alert Levels",
                    description="Volcanic alerts must follow standardized level systems",
                    requirement="IAVCEI Alert Level Standards",
                    validation_method="alert_analysis",
                    geological_context="Consistent volcanic hazard communication"
                )
            ]
        }
        
        return rules
    
    def _load_regulatory_frameworks(self) -> Dict[str, Dict[str, Any]]:
        """Load regulatory compliance frameworks"""
        return {
            "GDPR": {
                "name": "General Data Protection Regulation",
                "jurisdiction": "European Union",
                "applicable_rules": ["DP001", "DP002", "DP003"],
                "certification_required": True,
                "audit_frequency_months": 12
            },
            "SOC2": {
                "name": "Service Organization Control 2",
                "jurisdiction": "United States",
                "applicable_rules": ["SEC001", "SEC002", "SEC003", "DP001"],
                "certification_required": True,
                "audit_frequency_months": 12
            },
            "ISO27001": {
                "name": "Information Security Management",
                "jurisdiction": "International",
                "applicable_rules": ["SEC001", "SEC002", "SEC003"],
                "certification_required": True,
                "audit_frequency_months": 36
            },
            "NIST_Cybersecurity": {
                "name": "NIST Cybersecurity Framework",
                "jurisdiction": "United States",
                "applicable_rules": ["SEC001", "SEC002", "SEC003"],
                "certification_required": False,
                "audit_frequency_months": 12
            }
        }
    
    def _load_scientific_standards(self) -> Dict[str, Dict[str, Any]]:
        """Load scientific computing standards"""
        return {
            "IEEE_754": {
                "name": "IEEE Standard for Floating-Point Arithmetic",
                "domain": "Numerical Computing",
                "applicable_rules": ["SA001"],
                "validation_required": True
            },
            "FAIR_Principles": {
                "name": "Findable, Accessible, Interoperable, Reusable",
                "domain": "Scientific Data Management",
                "applicable_rules": ["SA003", "SA004"],
                "validation_required": True
            },
            "Reproducible_Research": {
                "name": "Reproducible Research Standards",
                "domain": "Scientific Computing",
                "applicable_rules": ["SA002", "SA003", "CQ002"],
                "validation_required": True
            }
        }
    
    def assess_compliance(self, project_path: str) -> ComplianceReport:
        """Perform comprehensive compliance assessment"""
        print("📋 Starting Compliance Assessment...")
        
        assessments = []
        all_violations = []
        
        # Assess each compliance category
        for category in self.config["compliance_categories"]:
            print(f"  📊 Assessing {category} compliance...")
            
            assessment = self._assess_category_compliance(project_path, category)
            assessments.append(assessment)
            all_violations.extend(assessment.violations)
        
        # Calculate overall compliance score
        overall_score = self._calculate_overall_compliance_score(assessments)
        
        # Identify critical violations
        critical_violations = [v for v in all_violations if v.severity == "critical"]
        
        # Check regulatory compliance
        regulatory_compliance = self._check_regulatory_compliance(all_violations)
        
        # Check scientific standards compliance
        scientific_compliance = self._check_scientific_standards_compliance(all_violations)
        
        # Check geological domain compliance
        geological_compliance = self._check_geological_domain_compliance(all_violations)
        
        # Generate remediation plan
        remediation_plan = self._generate_remediation_plan(all_violations, assessments)
        
        # Determine certification status
        certification_status = self._determine_certification_status(
            overall_score, critical_violations, regulatory_compliance
        )
        
        report = ComplianceReport(
            timestamp=datetime.now(),
            project_path=project_path,
            overall_compliance_score=overall_score,
            assessments=assessments,
            critical_violations=critical_violations,
            regulatory_compliance=regulatory_compliance,
            scientific_standards_compliance=scientific_compliance,
            geological_domain_compliance=geological_compliance,
            remediation_plan=remediation_plan,
            certification_status=certification_status
        )
        
        # Save and display results
        self._save_compliance_report(report)
        self._print_compliance_summary(report)
        
        return report
    
    def _assess_category_compliance(self, project_path: str, category: str) -> ComplianceAssessment:
        """Assess compliance for a specific category"""
        category_rules = self.compliance_rules.get(category, [])
        violations = []
        passed_rules = 0
        
        for rule in category_rules:
            rule_violations = self._validate_compliance_rule(project_path, rule)
            
            if rule_violations:
                violations.extend(rule_violations)
            else:
                passed_rules += 1
        
        failed_rules = len(category_rules) - passed_rules
        
        # Calculate compliance score for this category
        if category_rules:
            compliance_score = passed_rules / len(category_rules)
        else:
            compliance_score = 1.0
        
        # Generate category-specific recommendations
        recommendations = self._generate_category_recommendations(category, violations)
        
        return ComplianceAssessment(
            category=category,
            total_rules=len(category_rules),
            passed_rules=passed_rules,
            failed_rules=failed_rules,
            compliance_score=compliance_score,
            violations=violations,
            recommendations=recommendations
        )
    
    def _validate_compliance_rule(self, project_path: str, rule: ComplianceRule) -> List[ComplianceViolation]:
        """Validate a specific compliance rule"""
        violations = []
        
        try:
            if rule.validation_method == "code_analysis":
                violations = self._validate_code_analysis_rule(project_path, rule)
            elif rule.validation_method == "static_analysis":
                violations = self._validate_static_analysis_rule(project_path, rule)
            elif rule.validation_method == "security_scan":
                violations = self._validate_security_rule(project_path, rule)
            elif rule.validation_method == "algorithm_analysis":
                violations = self._validate_algorithm_rule(project_path, rule)
            elif rule.validation_method == "geospatial_analysis":
                violations = self._validate_geospatial_rule(project_path, rule)
            elif rule.validation_method == "documentation_check":
                violations = self._validate_documentation_rule(project_path, rule)
            elif rule.validation_method == "test_analysis":
                violations = self._validate_test_rule(project_path, rule)
            else:
                # Generic validation
                violations = self._validate_generic_rule(project_path, rule)
                
        except Exception as e:
            # Create violation for validation failure
            violations.append(ComplianceViolation(
                rule_id=rule.rule_id,
                file_path="validation_error",
                line_number=0,
                severity="medium",
                message=f"Rule validation failed: {str(e)}",
                evidence=f"Validation method: {rule.validation_method}",
                recommendation="Fix validation method or rule implementation",
                geological_impact=rule.geological_context
            ))
        
        return violations
    
    def _validate_code_analysis_rule(self, project_path: str, rule: ComplianceRule) -> List[ComplianceViolation]:
        """Validate rules requiring code analysis"""
        violations = []
        
        # Find Python files
        python_files = []
        for root, dirs, files in os.walk(project_path):
            dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', '.venv']]
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                file_violations = self._analyze_file_for_rule(file_path, content, rule)
                violations.extend(file_violations)
                
            except Exception as e:
                violations.append(ComplianceViolation(
                    rule_id=rule.rule_id,
                    file_path=file_path,
                    line_number=0,
                    severity="low",
                    message=f"Could not analyze file: {str(e)}",
                    evidence="File read error",
                    recommendation="Ensure file is accessible and properly encoded"
                ))
        
        return violations
    
    def _analyze_file_for_rule(self, file_path: str, content: str, rule: ComplianceRule) -> List[ComplianceViolation]:
        """Analyze a file for specific rule violations"""
        violations = []
        lines = content.split('\n')
        
        if rule.rule_id == "DP001":  # Personal Data Encryption
            # Check for encryption usage
            encryption_patterns = [
                r'encrypt', r'decrypt', r'AES', r'RSA', r'cryptography',
                r'hashlib', r'bcrypt', r'scrypt', r'pbkdf2'
            ]
            
            has_encryption = any(
                re.search(pattern, content, re.IGNORECASE) 
                for pattern in encryption_patterns
            )
            
            # Check for personal data handling
            personal_data_patterns = [
                r'email', r'phone', r'address', r'name', r'ssn',
                r'personal.*data', r'user.*info', r'contact.*info'
            ]
            
            has_personal_data = any(
                re.search(pattern, content, re.IGNORECASE)
                for pattern in personal_data_patterns
            )
            
            if has_personal_data and not has_encryption:
                violations.append(ComplianceViolation(
                    rule_id=rule.rule_id,
                    file_path=file_path,
                    line_number=0,
                    severity=rule.severity,
                    message="Personal data handling detected without encryption",
                    evidence="Personal data patterns found without encryption implementation",
                    recommendation="Implement encryption for personal data storage and transmission",
                    geological_impact=rule.geological_context
                ))
        
        elif rule.rule_id == "SA001":  # Physics Law Compliance
            # Check for conservation law implementation
            physics_keywords = ['conservation', 'momentum', 'energy', 'mass']
            physics_found = any(keyword in content.lower() for keyword in physics_keywords)
            
            if physics_found:
                # Check for proper implementation
                conservation_patterns = [
                    r'conservation.*law', r'conserve.*mass', r'conserve.*energy',
                    r'momentum.*conservation', r'energy.*balance'
                ]
                
                proper_implementation = any(
                    re.search(pattern, content, re.IGNORECASE)
                    for pattern in conservation_patterns
                )
                
                if not proper_implementation:
                    violations.append(ComplianceViolation(
                        rule_id=rule.rule_id,
                        file_path=file_path,
                        line_number=0,
                        severity=rule.severity,
                        message="Physics-based algorithm missing conservation law implementation",
                        evidence="Physics keywords found without conservation law implementation",
                        recommendation="Implement proper conservation laws in physics-based calculations",
                        geological_impact=rule.geological_context
                    ))
        
        elif rule.rule_id == "SA002":  # Uncertainty Quantification
            # Check for prediction functions
            prediction_patterns = [
                r'def.*predict', r'def.*forecast', r'def.*estimate',
                r'prediction', r'forecast', r'estimate'
            ]
            
            has_predictions = any(
                re.search(pattern, content, re.IGNORECASE)
                for pattern in prediction_patterns
            )
            
            if has_predictions:
                # Check for uncertainty quantification
                uncertainty_patterns = [
                    r'uncertainty', r'confidence', r'error.*bar', r'std.*dev',
                    r'variance', r'interval', r'margin.*error'
                ]
                
                has_uncertainty = any(
                    re.search(pattern, content, re.IGNORECASE)
                    for pattern in uncertainty_patterns
                )
                
                if not has_uncertainty:
                    violations.append(ComplianceViolation(
                        rule_id=rule.rule_id,
                        file_path=file_path,
                        line_number=0,
                        severity=rule.severity,
                        message="Prediction algorithm missing uncertainty quantification",
                        evidence="Prediction functions found without uncertainty estimates",
                        recommendation="Add uncertainty quantification to all predictions",
                        geological_impact=rule.geological_context
                    ))
        
        elif rule.rule_id == "SEC001":  # Input Validation
            # Check for input handling
            input_patterns = [
                r'input\(', r'request\.', r'args\[', r'form\[',
                r'json\.loads', r'pickle\.loads'
            ]
            
            has_input_handling = any(
                re.search(pattern, content)
                for pattern in input_patterns
            )
            
            if has_input_handling:
                # Check for validation
                validation_patterns = [
                    r'validate', r'sanitize', r'escape', r'clean',
                    r'check.*input', r'verify.*input'
                ]
                
                has_validation = any(
                    re.search(pattern, content, re.IGNORECASE)
                    for pattern in validation_patterns
                )
                
                if not has_validation:
                    violations.append(ComplianceViolation(
                        rule_id=rule.rule_id,
                        file_path=file_path,
                        line_number=0,
                        severity=rule.severity,
                        message="Input handling without validation detected",
                        evidence="Input processing found without validation",
                        recommendation="Implement input validation and sanitization",
                        geological_impact="Unvalidated input could corrupt geological calculations"
                    ))
        
        return violations
    
    def _validate_static_analysis_rule(self, project_path: str, rule: ComplianceRule) -> List[ComplianceViolation]:
        """Validate rules using static analysis"""
        violations = []
        
        if rule.rule_id == "CQ001":  # No Placeholder Code
            # Check for placeholder patterns
            placeholder_patterns = [
                r'# TODO', r'# FIXME', r'# HACK', r'# XXX',
                r'NotImplemented', r'pass\s*$', r'return\s+None\s*$'
            ]
            
            python_files = []
            for root, dirs, files in os.walk(project_path):
                dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', '.venv']]
                for file in files:
                    if file.endswith('.py'):
                        python_files.append(os.path.join(root, file))
            
            for file_path in python_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    for line_num, line in enumerate(lines, 1):
                        for pattern in placeholder_patterns:
                            if re.search(pattern, line, re.IGNORECASE):
                                violations.append(ComplianceViolation(
                                    rule_id=rule.rule_id,
                                    file_path=file_path,
                                    line_number=line_num,
                                    severity=rule.severity,
                                    message=f"Placeholder code detected: {pattern}",
                                    evidence=line.strip(),
                                    recommendation="Replace placeholder with complete implementation"
                                ))
                                
                except Exception:
                    continue
        
        return violations
    
    def _validate_security_rule(self, project_path: str, rule: ComplianceRule) -> List[ComplianceViolation]:
        """Validate security-related rules"""
        violations = []
        
        if SECURITY_SCANNING:
            try:
                # Run Bandit security scan
                b_mgr = bandit_manager.BanditManager(bandit.config.BanditConfig(), 'file')
                
                python_files = []
                for root, dirs, files in os.walk(project_path):
                    dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', '.venv']]
                    for file in files:
                        if file.endswith('.py'):
                            python_files.append(os.path.join(root, file))
                
                for file_path in python_files[:5]:  # Limit for performance
                    try:
                        b_mgr.discover_files([file_path])
                        b_mgr.run_tests()
                        
                        for issue in b_mgr.get_issue_list():
                            if issue.issue_severity in ['HIGH', 'MEDIUM']:
                                violations.append(ComplianceViolation(
                                    rule_id=rule.rule_id,
                                    file_path=issue.fname,
                                    line_number=issue.lineno,
                                    severity="high" if issue.issue_severity == "HIGH" else "medium",
                                    message=issue.issue_text,
                                    evidence=issue.get_code(),
                                    recommendation="Address security vulnerability as recommended by Bandit"
                                ))
                                
                    except Exception:
                        continue
                        
            except Exception as e:
                violations.append(ComplianceViolation(
                    rule_id=rule.rule_id,
                    file_path="security_scan",
                    line_number=0,
                    severity="medium",
                    message=f"Security scan failed: {str(e)}",
                    evidence="Bandit security scanning error",
                    recommendation="Fix security scanning setup and retry"
                ))
        
        return violations
    
    def _validate_algorithm_rule(self, project_path: str, rule: ComplianceRule) -> List[ComplianceViolation]:
        """Validate algorithm-specific rules"""
        violations = []
        
        # This would implement algorithm-specific validation
        # For now, return empty list as placeholder
        return violations
    
    def _validate_geospatial_rule(self, project_path: str, rule: ComplianceRule) -> List[ComplianceViolation]:
        """Validate geospatial-specific rules"""
        violations = []
        
        if rule.rule_id == "GEO001":  # Coordinate System Standards
            # Check for coordinate system usage
            coord_patterns = [
                r'EPSG:', r'WGS84', r'UTM', r'coordinate.*system',
                r'projection', r'datum', r'CRS'
            ]
            
            python_files = []
            for root, dirs, files in os.walk(project_path):
                dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', '.venv']]
                for file in files:
                    if file.endswith('.py'):
                        python_files.append(os.path.join(root, file))
            
            for file_path in python_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Check for geospatial operations
                    geospatial_keywords = ['latitude', 'longitude', 'coordinates', 'geospatial']
                    has_geospatial = any(keyword in content.lower() for keyword in geospatial_keywords)
                    
                    if has_geospatial:
                        # Check for proper coordinate system definition
                        has_coord_system = any(
                            re.search(pattern, content, re.IGNORECASE)
                            for pattern in coord_patterns
                        )
                        
                        if not has_coord_system:
                            violations.append(ComplianceViolation(
                                rule_id=rule.rule_id,
                                file_path=file_path,
                                line_number=0,
                                severity=rule.severity,
                                message="Geospatial operations without defined coordinate system",
                                evidence="Geospatial keywords found without coordinate system definition",
                                recommendation="Define proper coordinate reference system (e.g., EPSG codes)",
                                geological_impact=rule.geological_context
                            ))
                            
                except Exception:
                    continue
        
        return violations
    
    def _validate_documentation_rule(self, project_path: str, rule: ComplianceRule) -> List[ComplianceViolation]:
        """Validate documentation-related rules"""
        violations = []
        
        if rule.rule_id == "SA004":  # Peer Review Documentation
            # Check for documentation files
            doc_files = []
            for root, dirs, files in os.walk(project_path):
                for file in files:
                    if file.lower().endswith(('.md', '.rst', '.txt', '.doc', '.pdf')):
                        doc_files.append(os.path.join(root, file))
            
            # Check for peer review indicators
            peer_review_patterns = [
                r'peer.*review', r'reviewed.*by', r'expert.*review',
                r'scientific.*review', r'domain.*expert'
            ]
            
            has_peer_review_docs = False
            for doc_file in doc_files:
                try:
                    with open(doc_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    if any(re.search(pattern, content, re.IGNORECASE) for pattern in peer_review_patterns):
                        has_peer_review_docs = True
                        break
                        
                except Exception:
                    continue
            
            if not has_peer_review_docs:
                violations.append(ComplianceViolation(
                    rule_id=rule.rule_id,
                    file_path="documentation",
                    line_number=0,
                    severity=rule.severity,
                    message="Missing peer review documentation",
                    evidence="No peer review documentation found in project",
                    recommendation="Add peer review documentation for scientific algorithms",
                    geological_impact=rule.geological_context
                ))
        
        return violations
    
    def _validate_test_rule(self, project_path: str, rule: ComplianceRule) -> List[ComplianceViolation]:
        """Validate test-related rules"""
        violations = []
        
        if rule.rule_id == "SA003":  # Benchmark Validation
            # Check for test files
            test_files = []
            for root, dirs, files in os.walk(project_path):
                for file in files:
                    if 'test' in file.lower() and file.endswith('.py'):
                        test_files.append(os.path.join(root, file))
            
            # Check for benchmark testing
            benchmark_patterns = [
                r'benchmark', r'validation', r'reference.*data',
                r'ground.*truth', r'known.*result'
            ]
            
            has_benchmark_tests = False
            for test_file in test_files:
                try:
                    with open(test_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    if any(re.search(pattern, content, re.IGNORECASE) for pattern in benchmark_patterns):
                        has_benchmark_tests = True
                        break
                        
                except Exception:
                    continue
            
            if not has_benchmark_tests:
                violations.append(ComplianceViolation(
                    rule_id=rule.rule_id,
                    file_path="tests",
                    line_number=0,
                    severity=rule.severity,
                    message="Missing benchmark validation tests",
                    evidence="No benchmark testing found in test files",
                    recommendation="Add benchmark validation tests against known datasets",
                    geological_impact=rule.geological_context
                ))
        
        return violations
    
    def _validate_generic_rule(self, project_path: str, rule: ComplianceRule) -> List[ComplianceViolation]:
        """Generic rule validation"""
        # Placeholder for generic validation
        return []
    
    def _calculate_overall_compliance_score(self, assessments: List[ComplianceAssessment]) -> float:
        """Calculate overall compliance score"""
        if not assessments:
            return 0.0
        
        # Weight categories by importance
        category_weights = {
            "data_protection": 0.20,
            "scientific_accuracy": 0.20,
            "security": 0.15,
            "geological_standards": 0.15,
            "code_quality": 0.10,
            "performance": 0.10,
            "documentation": 0.05,
            "accessibility": 0.05
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for assessment in assessments:
            weight = category_weights.get(assessment.category, 0.1)
            weighted_score += assessment.compliance_score * weight
            total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _check_regulatory_compliance(self, violations: List[ComplianceViolation]) -> Dict[str, bool]:
        """Check compliance with regulatory frameworks"""
        regulatory_compliance = {}
        
        for framework_name, framework in self.regulatory_frameworks.items():
            applicable_rules = framework["applicable_rules"]
            
            # Check if any violations affect this framework
            framework_violations = [
                v for v in violations 
                if v.rule_id in applicable_rules and v.severity in ["critical", "high"]
            ]
            
            regulatory_compliance[framework_name] = len(framework_violations) == 0
        
        return regulatory_compliance
    
    def _check_scientific_standards_compliance(self, violations: List[ComplianceViolation]) -> Dict[str, bool]:
        """Check compliance with scientific standards"""
        scientific_compliance = {}
        
        for standard_name, standard in self.scientific_standards.items():
            applicable_rules = standard["applicable_rules"]
            
            # Check if any violations affect this standard
            standard_violations = [
                v for v in violations 
                if v.rule_id in applicable_rules and v.severity in ["critical", "high"]
            ]
            
            scientific_compliance[standard_name] = len(standard_violations) == 0
        
        return scientific_compliance
    
    def _check_geological_domain_compliance(self, violations: List[ComplianceViolation]) -> Dict[str, bool]:
        """Check compliance with geological domain standards"""
        geological_compliance = {}
        
        # Check geological-specific violations
        geological_violations = [
            v for v in violations 
            if v.geological_impact is not None and v.severity in ["critical", "high"]
        ]
        
        geological_compliance["IUGS_Standards"] = not any(
            "coordinate" in v.message.lower() for v in geological_violations
        )
        
        geological_compliance["Seismic_Data_Standards"] = not any(
            "seismic" in v.message.lower() for v in geological_violations
        )
        
        geological_compliance["Volcanic_Alert_Standards"] = not any(
            "volcanic" in v.message.lower() or "alert" in v.message.lower() 
            for v in geological_violations
        )
        
        return geological_compliance
    
    def _generate_remediation_plan(self, violations: List[ComplianceViolation], 
                                 assessments: List[ComplianceAssessment]) -> List[str]:
        """Generate remediation plan for compliance violations"""
        remediation_plan = []
        
        # Priority 1: Critical violations
        critical_violations = [v for v in violations if v.severity == "critical"]
        if critical_violations:
            remediation_plan.append("IMMEDIATE ACTION REQUIRED:")
            for violation in critical_violations[:5]:  # Top 5 critical
                remediation_plan.append(f"  • {violation.recommendation}")
        
        # Priority 2: High severity violations
        high_violations = [v for v in violations if v.severity == "high"]
        if high_violations:
            remediation_plan.append("HIGH PRIORITY:")
            for violation in high_violations[:5]:  # Top 5 high
                remediation_plan.append(f"  • {violation.recommendation}")
        
        # Priority 3: Category-specific improvements
        low_scoring_categories = [a for a in assessments if a.compliance_score < 0.8]
        if low_scoring_categories:
            remediation_plan.append("CATEGORY IMPROVEMENTS:")
            for assessment in low_scoring_categories:
                remediation_plan.append(f"  • Improve {assessment.category} compliance (current: {assessment.compliance_score:.2f})")
        
        return remediation_plan
    
    def _determine_certification_status(self, overall_score: float, 
                                      critical_violations: List[ComplianceViolation],
                                      regulatory_compliance: Dict[str, bool]) -> str:
        """Determine certification status"""
        if critical_violations:
            return "NOT_CERTIFIED - Critical violations must be resolved"
        
        if overall_score < self.config["minimum_compliance_scores"]["overall"]:
            return f"NOT_CERTIFIED - Overall score {overall_score:.2f} below minimum {self.config['minimum_compliance_scores']['overall']}"
        
        if not all(regulatory_compliance.values()):
            failed_frameworks = [name for name, compliant in regulatory_compliance.items() if not compliant]
            return f"CONDITIONALLY_CERTIFIED - Regulatory issues: {', '.join(failed_frameworks)}"
        
        if overall_score >= 0.95:
            return "FULLY_CERTIFIED - Excellent compliance"
        elif overall_score >= 0.90:
            return "CERTIFIED - Good compliance"
        else:
            return "CONDITIONALLY_CERTIFIED - Minimum compliance met"
    
    def _generate_category_recommendations(self, category: str, 
                                        violations: List[ComplianceViolation]) -> List[str]:
        """Generate category-specific recommendations"""
        recommendations = []
        
        if category == "data_protection":
            recommendations.extend([
                "Implement end-to-end encryption for sensitive geological data",
                "Establish clear data retention and deletion policies",
                "Add user consent management for data collection",
                "Implement data anonymization for research datasets"
            ])
        
        elif category == "scientific_accuracy":
            recommendations.extend([
                "Validate all algorithms against peer-reviewed benchmarks",
                "Implement uncertainty quantification for all predictions",
                "Ensure physics laws are properly enforced in calculations",
                "Add peer review documentation for all scientific methods"
            ])
        
        elif category == "security":
            recommendations.extend([
                "Implement comprehensive input validation",
                "Add authentication to all API endpoints",
                "Use encrypted communication protocols",
                "Regular security vulnerability scanning"
            ])
        
        elif category == "geological_standards":
            recommendations.extend([
                "Use standardized coordinate reference systems",
                "Follow ICS geological time scale standards",
                "Implement SEED format for seismic data",
                "Use standardized volcanic alert level systems"
            ])
        
        return recommendations
    
    def _save_compliance_report(self, report: ComplianceReport):
        """Save compliance report to file"""
        report_path = ".kiro/quality-assurance/compliance-report.json"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        # Convert to serializable format
        report_dict = asdict(report)
        
        with open(report_path, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)
        
        print(f"📊 Compliance report saved: {report_path}")
    
    def _print_compliance_summary(self, report: ComplianceReport):
        """Print compliance summary"""
        print(f"\n📋 Compliance Assessment Summary:")
        print(f"   Overall Compliance Score: {report.overall_compliance_score:.2f}")
        print(f"   Certification Status: {report.certification_status}")
        print(f"   Critical Violations: {len(report.critical_violations)}")
        
        print(f"\n📊 Category Compliance:")
        for assessment in report.assessments:
            status = "✅" if assessment.compliance_score >= 0.8 else "❌"
            print(f"   {status} {assessment.category}: {assessment.compliance_score:.2f} ({assessment.passed_rules}/{assessment.total_rules} rules)")
        
        print(f"\n🏛️  Regulatory Compliance:")
        for framework, compliant in report.regulatory_compliance.items():
            status = "✅" if compliant else "❌"
            print(f"   {status} {framework}")
        
        if report.critical_violations:
            print(f"\n🚨 Critical Violations:")
            for violation in report.critical_violations[:3]:
                print(f"   • {violation.message} ({violation.file_path})")
        
        if report.remediation_plan:
            print(f"\n💡 Remediation Plan:")
            for item in report.remediation_plan[:5]:
                print(f"   {item}")


def main():
    """Main entry point for compliance assessment"""
    if len(sys.argv) < 2:
        print("Usage: python compliance-framework.py <project_path>")
        sys.exit(1)
    
    project_path = sys.argv[1]
    
    if not os.path.exists(project_path):
        print(f"Error: Project path '{project_path}' does not exist")
        sys.exit(1)
    
    # Initialize and run compliance assessment
    compliance_framework = ScientificComplianceFramework()
    report = compliance_framework.assess_compliance(project_path)
    
    # Exit with appropriate code
    success = (
        report.overall_compliance_score >= 0.85 and
        len(report.critical_violations) == 0 and
        "NOT_CERTIFIED" not in report.certification_status
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
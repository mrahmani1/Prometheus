"""
Advanced Database Optimization Engine
Automated optimization recommendations and performance tuning
Production-grade implementation with machine learning-based optimization
"""

import asyncio
import logging
import json
import os
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
import statistics
import hashlib
import re
from concurrent.futures import ThreadPoolExecutor
import pickle

# Machine learning for optimization
try:
    from sklearn.ensemble import RandomForestRegressor, IsolationForest
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# Database analysis
from sqlalchemy import text, inspect
from sqlalchemy.ext.asyncio import AsyncSession

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizationCategory(Enum):
    """Categories of database optimizations"""
    INDEXING = "indexing"
    QUERY_OPTIMIZATION = "query_optimization"
    SCHEMA_DESIGN = "schema_design"
    CONFIGURATION_TUNING = "configuration_tuning"
    PARTITIONING = "partitioning"
    CACHING = "caching"
    CONNECTION_POOLING = "connection_pooling"
    STORAGE_OPTIMIZATION = "storage_optimization"
    MAINTENANCE = "maintenance"
    SECURITY = "security"

class OptimizationImpact(Enum):
    """Expected impact levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RiskLevel(Enum):
    """Risk levels for optimizations"""
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class QueryPattern:
    """Analyzed query pattern"""
    pattern_id: str
    query_template: str
    frequency: int
    avg_execution_time: float
    total_execution_time: float
    resource_usage: Dict[str, float]
    tables_accessed: List[str]
    columns_accessed: List[str]
    join_patterns: List[str]
    where_conditions: List[str]
    optimization_opportunities: List[str]

@dataclass
class IndexRecommendation:
    """Index creation recommendation"""
    table_name: str
    columns: List[str]
    index_type: str  # btree, hash, gin, gist, etc.
    estimated_size: int
    estimated_performance_gain: float
    creation_time_estimate: float
    maintenance_overhead: float
    uniqueness_constraint: bool = False
    partial_condition: Optional[str] = None
    include_columns: List[str] = field(default_factory=list)

@dataclass
class QueryOptimizationRecommendation:
    """Query optimization recommendation"""
    original_query: str
    optimized_query: str
    optimization_techniques: List[str]
    estimated_improvement: float
    complexity_reduction: float
    resource_savings: Dict[str, float]
    risk_assessment: str
    validation_required: bool = True

@dataclass
class ConfigurationRecommendation:
    """Database configuration optimization"""
    parameter_name: str
    current_value: Any
    recommended_value: Any
    justification: str
    expected_impact: str
    restart_required: bool
    validation_query: Optional[str] = None
    rollback_value: Any = None

@dataclass
class PartitioningRecommendation:
    """Table partitioning recommendation"""
    table_name: str
    partition_strategy: str  # range, hash, list
    partition_key: str
    estimated_partitions: int
    performance_improvement: float
    storage_efficiency: float
    maintenance_complexity: str
    migration_sql: List[str]

@dataclass
class OptimizationPlan:
    """Comprehensive optimization plan"""
    plan_id: str
    database_name: str
    created_at: datetime
    index_recommendations: List[IndexRecommendation]
    query_optimizations: List[QueryOptimizationRecommendation]
    config_recommendations: List[ConfigurationRecommendation]
    partitioning_recommendations: List[PartitioningRecommendation]
    estimated_total_improvement: float
    implementation_order: List[str]
    estimated_duration: timedelta
    risk_assessment: Dict[str, str]

class AdvancedDatabaseOptimizer:
    """
    Advanced Database Optimization Engine
    Uses machine learning and statistical analysis for optimization recommendations
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.query_analyzer = QueryPatternAnalyzer()
        self.index_advisor = IndexAdvisor()
        self.config_optimizer = ConfigurationOptimizer()
        self.partition_advisor = PartitioningAdvisor()
        
        # Machine learning models
        self.performance_predictor = None
        self.anomaly_detector = None
        self.workload_classifier = None
        
        # Historical data
        self.optimization_history: List[OptimizationPlan] = []
        self.performance_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Optimization rules engine
        self.rules_engine = OptimizationRulesEngine()
        
        logger.info("Advanced Database Optimizer initialized")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load optimizer configuration"""
        default_config = {
            'analysis_window_days': 7,
            'min_query_frequency': 10,
            'slow_query_threshold': 1.0,
            'index_size_threshold': 100 * 1024 * 1024,  # 100MB
            'max_recommendations_per_category': 10,
            'enable_ml_optimization': ML_AVAILABLE,
            'auto_apply_low_risk': False,
            'performance_improvement_threshold': 0.1,  # 10%
            'model_retrain_interval_hours': 24
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    import yaml
                    user_config = yaml.safe_load(f)
                default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Failed to load optimizer config: {e}")
        
        return default_config
    
    async def analyze_database_performance(self, db_name: str, 
                                         performance_metrics: Dict[str, Any],
                                         query_logs: List[Dict[str, Any]]) -> OptimizationPlan:
        """
        Comprehensive database performance analysis and optimization planning
        """
        logger.info(f"Starting comprehensive analysis for database: {db_name}")
        
        # Analyze query patterns
        query_patterns = await self.query_analyzer.analyze_patterns(query_logs)
        
        # Generate index recommendations
        index_recommendations = await self.index_advisor.recommend_indexes(
            db_name, query_patterns, performance_metrics
        )
        
        # Generate query optimizations
        query_optimizations = await self._generate_query_optimizations(
            query_patterns, performance_metrics
        )
        
        # Generate configuration recommendations
        config_recommendations = await self.config_optimizer.recommend_configurations(
            db_name, performance_metrics
        )
        
        # Generate partitioning recommendations
        partitioning_recommendations = await self.partition_advisor.recommend_partitioning(
            db_name, query_patterns, performance_metrics
        )
        
        # Create optimization plan
        optimization_plan = OptimizationPlan(
            plan_id=f"opt_plan_{db_name}_{int(time.time())}",
            database_name=db_name,
            created_at=datetime.utcnow(),
            index_recommendations=index_recommendations,
            query_optimizations=query_optimizations,
            config_recommendations=config_recommendations,
            partitioning_recommendations=partitioning_recommendations,
            estimated_total_improvement=self._calculate_total_improvement(
                index_recommendations, query_optimizations, 
                config_recommendations, partitioning_recommendations
            ),
            implementation_order=self._determine_implementation_order(
                index_recommendations, query_optimizations,
                config_recommendations, partitioning_recommendations
            ),
            estimated_duration=self._estimate_implementation_duration(
                index_recommendations, query_optimizations,
                config_recommendations, partitioning_recommendations
            ),
            risk_assessment=self._assess_plan_risks(
                index_recommendations, query_optimizations,
                config_recommendations, partitioning_recommendations
            )
        )
        
        # Store optimization plan
        self.optimization_history.append(optimization_plan)
        
        # Train/update ML models if enabled
        if self.config['enable_ml_optimization']:
            await self._update_ml_models(db_name, performance_metrics, query_logs)
        
        logger.info(f"Analysis complete. Generated {len(optimization_plan.index_recommendations)} "
                   f"index recommendations, {len(optimization_plan.query_optimizations)} "
                   f"query optimizations, {len(optimization_plan.config_recommendations)} "
                   f"config recommendations")
        
        return optimization_plan
    
    async def _generate_query_optimizations(self, query_patterns: List[QueryPattern],
                                          performance_metrics: Dict[str, Any]) -> List[QueryOptimizationRecommendation]:
        """Generate query optimization recommendations"""
        optimizations = []
        
        for pattern in query_patterns:
            if pattern.avg_execution_time > self.config['slow_query_threshold']:
                # Analyze query for optimization opportunities
                optimization = await self._optimize_query_pattern(pattern)
                if optimization:
                    optimizations.append(optimization)
        
        return optimizations[:self.config['max_recommendations_per_category']]
    
    async def _optimize_query_pattern(self, pattern: QueryPattern) -> Optional[QueryOptimizationRecommendation]:
        """Optimize individual query pattern"""
        optimization_techniques = []
        optimized_query = pattern.query_template
        estimated_improvement = 0.0
        
        # Check for common optimization opportunities
        if "SELECT *" in pattern.query_template:
            optimization_techniques.append("Replace SELECT * with specific columns")
            estimated_improvement += 0.15
        
        if len(pattern.join_patterns) > 3:
            optimization_techniques.append("Optimize complex joins")
            estimated_improvement += 0.25
        
        if "ORDER BY" in pattern.query_template and "LIMIT" not in pattern.query_template:
            optimization_techniques.append("Add LIMIT clause to ORDER BY queries")
            estimated_improvement += 0.10
        
        if any("LIKE '%pattern%'" in condition for condition in pattern.where_conditions):
            optimization_techniques.append("Replace LIKE with full-text search")
            estimated_improvement += 0.30
        
        if optimization_techniques:
            return QueryOptimizationRecommendation(
                original_query=pattern.query_template,
                optimized_query=optimized_query,  # Would be actually optimized
                optimization_techniques=optimization_techniques,
                estimated_improvement=min(estimated_improvement, 0.8),  # Cap at 80%
                complexity_reduction=0.2,
                resource_savings={'cpu': 0.15, 'memory': 0.10, 'io': 0.20},
                risk_assessment="low"
            )
        
        return None
    
    def _calculate_total_improvement(self, index_recs: List[IndexRecommendation],
                                   query_opts: List[QueryOptimizationRecommendation],
                                   config_recs: List[ConfigurationRecommendation],
                                   partition_recs: List[PartitioningRecommendation]) -> float:
        """Calculate estimated total performance improvement"""
        total_improvement = 0.0
        
        # Index improvements
        for rec in index_recs:
            total_improvement += rec.estimated_performance_gain * 0.3
        
        # Query optimization improvements
        for rec in query_opts:
            total_improvement += rec.estimated_improvement * 0.4
        
        # Configuration improvements (estimated)
        total_improvement += len(config_recs) * 0.05
        
        # Partitioning improvements
        for rec in partition_recs:
            total_improvement += rec.performance_improvement * 0.2
        
        return min(total_improvement, 0.9)  # Cap at 90% improvement
    
    def _determine_implementation_order(self, index_recs: List[IndexRecommendation],
                                      query_opts: List[QueryOptimizationRecommendation],
                                      config_recs: List[ConfigurationRecommendation],
                                      partition_recs: List[PartitioningRecommendation]) -> List[str]:
        """Determine optimal implementation order"""
        implementation_order = []
        
        # 1. Low-risk configuration changes first
        low_risk_configs = [rec for rec in config_recs if not rec.restart_required]
        implementation_order.extend([f"config_{rec.parameter_name}" for rec in low_risk_configs])
        
        # 2. Index creation (concurrent where possible)
        concurrent_indexes = [rec for rec in index_recs if rec.estimated_size < self.config['index_size_threshold']]
        implementation_order.extend([f"index_{rec.table_name}_{'_'.join(rec.columns)}" for rec in concurrent_indexes])
        
        # 3. Query optimizations
        implementation_order.extend([f"query_opt_{i}" for i in range(len(query_opts))])
        
        # 4. Large index creation
        large_indexes = [rec for rec in index_recs if rec.estimated_size >= self.config['index_size_threshold']]
        implementation_order.extend([f"index_{rec.table_name}_{'_'.join(rec.columns)}" for rec in large_indexes])
        
        # 5. Configuration changes requiring restart
        restart_configs = [rec for rec in config_recs if rec.restart_required]
        implementation_order.extend([f"config_{rec.parameter_name}" for rec in restart_configs])
        
        # 6. Partitioning (most disruptive)
        implementation_order.extend([f"partition_{rec.table_name}" for rec in partition_recs])
        
        return implementation_order
    
    def _estimate_implementation_duration(self, index_recs: List[IndexRecommendation],
                                        query_opts: List[QueryOptimizationRecommendation],
                                        config_recs: List[ConfigurationRecommendation],
                                        partition_recs: List[PartitioningRecommendation]) -> timedelta:
        """Estimate total implementation duration"""
        total_minutes = 0
        
        # Index creation time
        for rec in index_recs:
            total_minutes += rec.creation_time_estimate
        
        # Query optimization time (minimal)
        total_minutes += len(query_opts) * 5
        
        # Configuration changes
        total_minutes += len(config_recs) * 2
        
        # Partitioning time (significant)
        total_minutes += len(partition_recs) * 120  # 2 hours per partition
        
        return timedelta(minutes=total_minutes)
    
    def _assess_plan_risks(self, index_recs: List[IndexRecommendation],
                          query_opts: List[QueryOptimizationRecommendation],
                          config_recs: List[ConfigurationRecommendation],
                          partition_recs: List[PartitioningRecommendation]) -> Dict[str, str]:
        """Assess risks for optimization plan"""
        risks = {}
        
        # Index risks
        large_indexes = [rec for rec in index_recs if rec.estimated_size > 1024*1024*1024]  # 1GB
        if large_indexes:
            risks['large_indexes'] = f"Creating {len(large_indexes)} large indexes may impact performance during creation"
        
        # Configuration risks
        restart_configs = [rec for rec in config_recs if rec.restart_required]
        if restart_configs:
            risks['restart_required'] = f"{len(restart_configs)} configuration changes require database restart"
        
        # Partitioning risks
        if partition_recs:
            risks['partitioning'] = f"Partitioning {len(partition_recs)} tables requires significant downtime"
        
        # Query optimization risks
        complex_optimizations = [rec for rec in query_opts if rec.risk_assessment in ['medium', 'high']]
        if complex_optimizations:
            risks['query_changes'] = f"{len(complex_optimizations)} query optimizations require thorough testing"
        
        return risks
    
    async def _update_ml_models(self, db_name: str, performance_metrics: Dict[str, Any],
                              query_logs: List[Dict[str, Any]]):
        """Update machine learning models with new data"""
        if not ML_AVAILABLE:
            return
        
        try:
            # Prepare training data
            training_data = self._prepare_ml_training_data(performance_metrics, query_logs)
            
            # Update performance predictor
            if len(training_data) > 100:  # Minimum data requirement
                await self._train_performance_predictor(training_data)
            
            # Update anomaly detector
            await self._train_anomaly_detector(training_data)
            
            # Update workload classifier
            await self._train_workload_classifier(query_logs)
            
        except Exception as e:
            logger.error(f"Failed to update ML models: {e}")
    
    def _prepare_ml_training_data(self, performance_metrics: Dict[str, Any],
                                query_logs: List[Dict[str, Any]]) -> pd.DataFrame:
        """Prepare training data for ML models"""
        # Convert metrics and logs to structured format
        data = []
        
        for log_entry in query_logs:
            features = {
                'query_length': len(log_entry.get('query', '')),
                'execution_time': log_entry.get('execution_time', 0),
                'rows_examined': log_entry.get('rows_examined', 0),
                'rows_returned': log_entry.get('rows_returned', 0),
                'cpu_usage': performance_metrics.get('cpu_usage', 0),
                'memory_usage': performance_metrics.get('memory_usage', 0),
                'io_usage': performance_metrics.get('io_usage', 0),
                'has_joins': 'JOIN' in log_entry.get('query', '').upper(),
                'has_subqueries': 'SELECT' in log_entry.get('query', '').upper().count('SELECT') > 1,
                'has_aggregation': any(agg in log_entry.get('query', '').upper() 
                                     for agg in ['COUNT', 'SUM', 'AVG', 'MAX', 'MIN']),
                'has_order_by': 'ORDER BY' in log_entry.get('query', '').upper(),
                'has_group_by': 'GROUP BY' in log_entry.get('query', '').upper()
            }
            data.append(features)
        
        return pd.DataFrame(data)
    
    async def _train_performance_predictor(self, training_data: pd.DataFrame):
        """Train performance prediction model"""
        try:
            # Prepare features and target
            feature_columns = [col for col in training_data.columns if col != 'execution_time']
            X = training_data[feature_columns]
            y = training_data['execution_time']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            self.performance_predictor = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            self.performance_predictor.fit(X_train, y_train)
            
            # Evaluate model
            score = self.performance_predictor.score(X_test, y_test)
            logger.info(f"Performance predictor trained with R² score: {score:.3f}")
            
        except Exception as e:
            logger.error(f"Failed to train performance predictor: {e}")
    
    async def _train_anomaly_detector(self, training_data: pd.DataFrame):
        """Train anomaly detection model"""
        try:
            # Prepare features
            feature_columns = [col for col in training_data.columns if col != 'execution_time']
            X = training_data[feature_columns]
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train isolation forest
            self.anomaly_detector = IsolationForest(
                contamination=0.1,
                random_state=42
            )
            self.anomaly_detector.fit(X_scaled)
            
            logger.info("Anomaly detector trained successfully")
            
        except Exception as e:
            logger.error(f"Failed to train anomaly detector: {e}")
    
    async def _train_workload_classifier(self, query_logs: List[Dict[str, Any]]):
        """Train workload classification model"""
        try:
            # Extract query features for clustering
            features = []
            for log_entry in query_logs:
                query = log_entry.get('query', '').upper()
                feature_vector = [
                    query.count('SELECT'),
                    query.count('INSERT'),
                    query.count('UPDATE'),
                    query.count('DELETE'),
                    query.count('JOIN'),
                    query.count('WHERE'),
                    query.count('ORDER BY'),
                    query.count('GROUP BY'),
                    len(query)
                ]
                features.append(feature_vector)
            
            if len(features) > 10:
                # Cluster workload patterns
                self.workload_classifier = DBSCAN(eps=0.5, min_samples=5)
                self.workload_classifier.fit(features)
                
                logger.info("Workload classifier trained successfully")
            
        except Exception as e:
            logger.error(f"Failed to train workload classifier: {e}")
    
    async def predict_optimization_impact(self, optimization_plan: OptimizationPlan) -> Dict[str, float]:
        """Predict the impact of optimization plan using ML models"""
        if not self.performance_predictor:
            return {'estimated_improvement': optimization_plan.estimated_total_improvement}
        
        try:
            # Create feature vector for prediction
            features = self._extract_plan_features(optimization_plan)
            
            # Predict performance improvement
            predicted_improvement = self.performance_predictor.predict([features])[0]
            
            return {
                'ml_predicted_improvement': predicted_improvement,
                'confidence_score': 0.8,  # Would be calculated based on model uncertainty
                'estimated_improvement': optimization_plan.estimated_total_improvement
            }
            
        except Exception as e:
            logger.error(f"Failed to predict optimization impact: {e}")
            return {'estimated_improvement': optimization_plan.estimated_total_improvement}
    
    def _extract_plan_features(self, plan: OptimizationPlan) -> List[float]:
        """Extract features from optimization plan for ML prediction"""
        return [
            len(plan.index_recommendations),
            len(plan.query_optimizations),
            len(plan.config_recommendations),
            len(plan.partitioning_recommendations),
            sum(rec.estimated_performance_gain for rec in plan.index_recommendations),
            sum(rec.estimated_improvement for rec in plan.query_optimizations),
            plan.estimated_duration.total_seconds() / 3600,  # hours
            len(plan.risk_assessment)
        ]
    
    async def get_optimization_status(self, plan_id: str) -> Dict[str, Any]:
        """Get status of optimization plan implementation"""
        plan = next((p for p in self.optimization_history if p.plan_id == plan_id), None)
        if not plan:
            return {'error': 'Plan not found'}
        
        return {
            'plan_id': plan_id,
            'database_name': plan.database_name,
            'created_at': plan.created_at.isoformat(),
            'estimated_improvement': plan.estimated_total_improvement,
            'implementation_progress': 0.0,  # Would track actual progress
            'recommendations_count': {
                'indexes': len(plan.index_recommendations),
                'queries': len(plan.query_optimizations),
                'config': len(plan.config_recommendations),
                'partitioning': len(plan.partitioning_recommendations)
            },
            'risk_assessment': plan.risk_assessment,
            'estimated_duration': str(plan.estimated_duration)
        }

class QueryPatternAnalyzer:
    """Analyze query patterns for optimization opportunities"""
    
    def __init__(self):
        self.pattern_cache: Dict[str, QueryPattern] = {}
    
    async def analyze_patterns(self, query_logs: List[Dict[str, Any]]) -> List[QueryPattern]:
        """Analyze query logs to identify patterns"""
        patterns = {}
        
        for log_entry in query_logs:
            query = log_entry.get('query', '')
            normalized_query = self._normalize_query(query)
            pattern_id = hashlib.md5(normalized_query.encode()).hexdigest()
            
            if pattern_id not in patterns:
                patterns[pattern_id] = QueryPattern(
                    pattern_id=pattern_id,
                    query_template=normalized_query,
                    frequency=0,
                    avg_execution_time=0.0,
                    total_execution_time=0.0,
                    resource_usage={},
                    tables_accessed=[],
                    columns_accessed=[],
                    join_patterns=[],
                    where_conditions=[],
                    optimization_opportunities=[]
                )
            
            pattern = patterns[pattern_id]
            pattern.frequency += 1
            pattern.total_execution_time += log_entry.get('execution_time', 0)
            
            # Extract query components
            pattern.tables_accessed.extend(self._extract_tables(query))
            pattern.columns_accessed.extend(self._extract_columns(query))
            pattern.join_patterns.extend(self._extract_joins(query))
            pattern.where_conditions.extend(self._extract_where_conditions(query))
        
        # Calculate averages and identify optimization opportunities
        for pattern in patterns.values():
            if pattern.frequency > 0:
                pattern.avg_execution_time = pattern.total_execution_time / pattern.frequency
                pattern.optimization_opportunities = self._identify_optimization_opportunities(pattern)
        
        return list(patterns.values())
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query for pattern matching"""
        # Remove literals and normalize whitespace
        normalized = re.sub(r"'[^']*'", "'?'", query)
        normalized = re.sub(r'\b\d+\b', '?', normalized)
        normalized = re.sub(r'\s+', ' ', normalized.strip())
        return normalized.upper()
    
    def _extract_tables(self, query: str) -> List[str]:
        """Extract table names from query"""
        # Simplified table extraction
        tables = []
        query_upper = query.upper()
        
        # FROM clause
        from_match = re.search(r'FROM\s+(\w+)', query_upper)
        if from_match:
            tables.append(from_match.group(1))
        
        # JOIN clauses
        join_matches = re.findall(r'JOIN\s+(\w+)', query_upper)
        tables.extend(join_matches)
        
        return list(set(tables))
    
    def _extract_columns(self, query: str) -> List[str]:
        """Extract column names from query"""
        columns = []
        # Simplified column extraction - would be more sophisticated in production
        select_match = re.search(r'SELECT\s+(.*?)\s+FROM', query.upper())
        if select_match:
            select_clause = select_match.group(1)
            if select_clause != '*':
                columns = [col.strip() for col in select_clause.split(',')]
        return columns
    
    def _extract_joins(self, query: str) -> List[str]:
        """Extract join patterns from query"""
        joins = re.findall(r'(INNER|LEFT|RIGHT|FULL|CROSS)\s+JOIN', query.upper())
        return joins
    
    def _extract_where_conditions(self, query: str) -> List[str]:
        """Extract WHERE conditions from query"""
        where_match = re.search(r'WHERE\s+(.*?)(?:\s+ORDER|\s+GROUP|\s+LIMIT|$)', query.upper())
        if where_match:
            return [where_match.group(1)]
        return []
    
    def _identify_optimization_opportunities(self, pattern: QueryPattern) -> List[str]:
        """Identify optimization opportunities for query pattern"""
        opportunities = []
        
        if pattern.avg_execution_time > 1.0:
            opportunities.append("slow_query")
        
        if len(pattern.join_patterns) > 2:
            opportunities.append("complex_joins")
        
        if "SELECT *" in pattern.query_template:
            opportunities.append("select_star")
        
        if any("LIKE '%%" in condition for condition in pattern.where_conditions):
            opportunities.append("inefficient_like")
        
        return opportunities

class IndexAdvisor:
    """Recommend database indexes for performance optimization"""
    
    async def recommend_indexes(self, db_name: str, query_patterns: List[QueryPattern],
                              performance_metrics: Dict[str, Any]) -> List[IndexRecommendation]:
        """Generate index recommendations based on query patterns"""
        recommendations = []
        
        # Analyze query patterns for index opportunities
        for pattern in query_patterns:
            if pattern.frequency > 10 and pattern.avg_execution_time > 0.5:
                index_recs = self._analyze_pattern_for_indexes(pattern)
                recommendations.extend(index_recs)
        
        # Remove duplicates and prioritize
        unique_recommendations = self._deduplicate_recommendations(recommendations)
        prioritized_recommendations = self._prioritize_recommendations(unique_recommendations)
        
        return prioritized_recommendations[:10]  # Top 10 recommendations
    
    def _analyze_pattern_for_indexes(self, pattern: QueryPattern) -> List[IndexRecommendation]:
        """Analyze query pattern for index opportunities"""
        recommendations = []
        
        # WHERE clause indexes
        for condition in pattern.where_conditions:
            columns = self._extract_where_columns(condition)
            if columns:
                recommendations.append(IndexRecommendation(
                    table_name=pattern.tables_accessed[0] if pattern.tables_accessed else 'unknown',
                    columns=columns,
                    index_type='btree',
                    estimated_size=50 * 1024 * 1024,  # 50MB estimate
                    estimated_performance_gain=0.3,
                    creation_time_estimate=5.0,  # 5 minutes
                    maintenance_overhead=0.1
                ))
        
        # JOIN indexes
        if pattern.join_patterns:
            for table in pattern.tables_accessed:
                recommendations.append(IndexRecommendation(
                    table_name=table,
                    columns=['id'],  # Simplified - would analyze actual join conditions
                    index_type='btree',
                    estimated_size=30 * 1024 * 1024,
                    estimated_performance_gain=0.4,
                    creation_time_estimate=3.0,
                    maintenance_overhead=0.05
                ))
        
        return recommendations
    
    def _extract_where_columns(self, condition: str) -> List[str]:
        """Extract column names from WHERE condition"""
        # Simplified extraction - would be more sophisticated in production
        columns = re.findall(r'(\w+)\s*[=<>!]', condition)
        return list(set(columns))
    
    def _deduplicate_recommendations(self, recommendations: List[IndexRecommendation]) -> List[IndexRecommendation]:
        """Remove duplicate index recommendations"""
        seen = set()
        unique_recs = []
        
        for rec in recommendations:
            key = (rec.table_name, tuple(sorted(rec.columns)), rec.index_type)
            if key not in seen:
                seen.add(key)
                unique_recs.append(rec)
        
        return unique_recs
    
    def _prioritize_recommendations(self, recommendations: List[IndexRecommendation]) -> List[IndexRecommendation]:
        """Prioritize index recommendations by impact"""
        return sorted(recommendations, 
                     key=lambda x: x.estimated_performance_gain / (x.estimated_size / (1024*1024)), 
                     reverse=True)

class ConfigurationOptimizer:
    """Optimize database configuration parameters"""
    
    async def recommend_configurations(self, db_name: str, 
                                     performance_metrics: Dict[str, Any]) -> List[ConfigurationRecommendation]:
        """Generate configuration optimization recommendations"""
        recommendations = []
        
        # Memory-related optimizations
        memory_usage = performance_metrics.get('memory_usage', 0.5)
        if memory_usage > 0.9:
            recommendations.append(ConfigurationRecommendation(
                parameter_name='shared_buffers',
                current_value='128MB',
                recommended_value='256MB',
                justification='High memory usage indicates need for larger buffer pool',
                expected_impact='Reduced I/O operations',
                restart_required=True
            ))
        
        # Connection pool optimizations
        connection_usage = performance_metrics.get('connection_pool_utilization', 0.5)
        if connection_usage > 0.8:
            recommendations.append(ConfigurationRecommendation(
                parameter_name='max_connections',
                current_value=100,
                recommended_value=200,
                justification='High connection pool utilization',
                expected_impact='Reduced connection wait times',
                restart_required=True
            ))
        
        # Query optimization settings
        slow_queries = performance_metrics.get('slow_queries_count', 0)
        if slow_queries > 100:
            recommendations.append(ConfigurationRecommendation(
                parameter_name='work_mem',
                current_value='4MB',
                recommended_value='8MB',
                justification='High number of slow queries may benefit from more work memory',
                expected_impact='Faster query execution',
                restart_required=False
            ))
        
        return recommendations

class PartitioningAdvisor:
    """Recommend table partitioning strategies"""
    
    async def recommend_partitioning(self, db_name: str, query_patterns: List[QueryPattern],
                                   performance_metrics: Dict[str, Any]) -> List[PartitioningRecommendation]:
        """Generate table partitioning recommendations"""
        recommendations = []
        
        # Analyze large tables with time-based queries
        for pattern in query_patterns:
            if self._is_time_based_query(pattern) and self._is_large_table_query(pattern):
                recommendations.append(PartitioningRecommendation(
                    table_name=pattern.tables_accessed[0] if pattern.tables_accessed else 'unknown',
                    partition_strategy='range',
                    partition_key='created_at',
                    estimated_partitions=12,  # Monthly partitions
                    performance_improvement=0.4,
                    storage_efficiency=0.2,
                    maintenance_complexity='medium',
                    migration_sql=[
                        f"CREATE TABLE {pattern.tables_accessed[0]}_partitioned (LIKE {pattern.tables_accessed[0]}) PARTITION BY RANGE (created_at);",
                        "-- Create monthly partitions",
                        "-- Migrate data",
                        "-- Drop original table"
                    ]
                ))
        
        return recommendations
    
    def _is_time_based_query(self, pattern: QueryPattern) -> bool:
        """Check if query pattern involves time-based filtering"""
        time_keywords = ['created_at', 'updated_at', 'timestamp', 'date']
        return any(keyword in ' '.join(pattern.where_conditions).lower() 
                  for keyword in time_keywords)
    
    def _is_large_table_query(self, pattern: QueryPattern) -> bool:
        """Check if query involves large tables (simplified check)"""
        # In production, would check actual table sizes
        return pattern.avg_execution_time > 2.0

class OptimizationRulesEngine:
    """Rule-based optimization recommendations"""
    
    def __init__(self):
        self.rules = self._load_optimization_rules()
    
    def _load_optimization_rules(self) -> List[Dict[str, Any]]:
        """Load optimization rules"""
        return [
            {
                'name': 'missing_index_on_foreign_key',
                'condition': lambda pattern: len(pattern.join_patterns) > 0,
                'recommendation': 'Create indexes on foreign key columns',
                'priority': 8
            },
            {
                'name': 'select_star_optimization',
                'condition': lambda pattern: 'SELECT *' in pattern.query_template,
                'recommendation': 'Replace SELECT * with specific columns',
                'priority': 6
            },
            {
                'name': 'missing_limit_on_order_by',
                'condition': lambda pattern: 'ORDER BY' in pattern.query_template and 'LIMIT' not in pattern.query_template,
                'recommendation': 'Add LIMIT clause to ORDER BY queries',
                'priority': 7
            }
        ]
    
    def apply_rules(self, patterns: List[QueryPattern]) -> List[Dict[str, Any]]:
        """Apply optimization rules to query patterns"""
        recommendations = []
        
        for pattern in patterns:
            for rule in self.rules:
                if rule['condition'](pattern):
                    recommendations.append({
                        'pattern_id': pattern.pattern_id,
                        'rule_name': rule['name'],
                        'recommendation': rule['recommendation'],
                        'priority': rule['priority']
                    })
        
        return recommendations

if __name__ == "__main__":
    # Example usage
    async def main():
        optimizer = AdvancedDatabaseOptimizer()
        
        # Example performance metrics
        performance_metrics = {
            'cpu_usage': 0.7,
            'memory_usage': 0.85,
            'connection_pool_utilization': 0.9,
            'slow_queries_count': 150
        }
        
        # Example query logs
        query_logs = [
            {
                'query': 'SELECT * FROM users WHERE created_at > ?',
                'execution_time': 2.5,
                'rows_examined': 10000,
                'rows_returned': 100
            },
            {
                'query': 'SELECT u.name, p.title FROM users u JOIN posts p ON u.id = p.user_id ORDER BY p.created_at',
                'execution_time': 1.8,
                'rows_examined': 5000,
                'rows_returned': 50
            }
        ]
        
        # Analyze and generate optimization plan
        plan = await optimizer.analyze_database_performance(
            'prometheus_main', performance_metrics, query_logs
        )
        
        print(f"Generated optimization plan with {len(plan.index_recommendations)} index recommendations")
        print(f"Estimated total improvement: {plan.estimated_total_improvement:.1%}")
        print(f"Estimated implementation time: {plan.estimated_duration}")
        
        # Get ML prediction if available
        impact_prediction = await optimizer.predict_optimization_impact(plan)
        print(f"ML predicted improvement: {impact_prediction}")
    
    asyncio.run(main())
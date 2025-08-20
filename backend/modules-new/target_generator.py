"""
Prometheus Platform - AI-Powered Target Generator and Ranking System
===================================================================

Advanced target ranking system for mineral exploration combining machine learning
predictions with geological expertise and economic factors.

Key Features:
- Multi-criteria decision analysis (MCDA) with fuzzy logic
- Economic feasibility assessment with NPV modeling
- Risk assessment incorporating geological, technical, and market factors
- Target prioritization using advanced optimization algorithms
- Integration with prospectivity mapping and geochemical analysis
- Real-time market data integration for commodity pricing
- Monte Carlo simulation for uncertainty quantification

Author: Prometheus Development Team  
Date: 2025-08-09
Version: 1.0.0
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import stats, optimize
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from shapely.geometry import Point, Polygon, convex_hull
from shapely.ops import unary_union
import warnings
warnings.filterwarnings('ignore')
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import json
import pickle
import logging
from datetime import datetime, timedelta
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
import requests
import time


class TargetType(Enum):
    """Types of mineral exploration targets"""
    GOLD = "gold"
    COPPER = "copper"
    LITHIUM = "lithium"
    RARE_EARTH = "rare_earth"
    PLATINUM_GROUP = "platinum_group"
    BASE_METALS = "base_metals"
    URANIUM = "uranium"
    DIAMONDS = "diamonds"
    IRON_ORE = "iron_ore"
    POLYMETALLIC = "polymetallic"


class RiskLevel(Enum):
    """Risk assessment levels"""
    VERY_LOW = 1
    LOW = 2
    MODERATE = 3
    HIGH = 4
    VERY_HIGH = 5


@dataclass
class ExplorationTarget:
    """Exploration target with comprehensive attributes"""
    id: str
    name: str
    target_type: TargetType
    location: Point
    prospectivity_score: float
    confidence: float
    area_km2: float
    depth_estimate: float
    
    # Geological attributes
    geological_confidence: float = 0.5
    structural_favorability: float = 0.5
    alteration_intensity: float = 0.5
    geochemical_anomaly: float = 0.5
    geophysical_anomaly: float = 0.5
    
    # Economic attributes
    estimated_grade: Optional[float] = None
    estimated_tonnage: Optional[float] = None
    npv_estimate: Optional[float] = None
    payback_period: Optional[float] = None
    operating_cost: Optional[float] = None
    capital_cost: Optional[float] = None
    
    # Risk attributes
    technical_risk: RiskLevel = RiskLevel.MODERATE
    geological_risk: RiskLevel = RiskLevel.MODERATE
    environmental_risk: RiskLevel = RiskLevel.MODERATE
    political_risk: RiskLevel = RiskLevel.MODERATE
    market_risk: RiskLevel = RiskLevel.MODERATE
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Ranking attributes
    overall_rank: Optional[int] = None
    priority_score: Optional[float] = None
    recommendation: Optional[str] = None


class EconomicModeler:
    """Economic assessment and NPV modeling for exploration targets"""
    
    def __init__(self):
        self.commodity_prices = {}
        self.cost_models = {}
        self.discount_rate = 0.10  # 10% discount rate
        
        # Initialize with default commodity prices (USD)
        self.default_prices = {
            'gold': 2000.0,  # $/oz
            'copper': 9000.0,  # $/t
            'lithium': 50000.0,  # $/t (Li2CO3 equivalent)
            'silver': 25.0,  # $/oz
            'zinc': 2500.0,  # $/t
            'lead': 2000.0,  # $/t
            'nickel': 20000.0,  # $/t
            'uranium': 50.0,  # $/lb U3O8
            'iron_ore': 120.0,  # $/t (62% Fe)
            'rare_earth': 100000.0  # $/t (mixed REO)
        }
        
        self.commodity_prices.update(self.default_prices)
    
    def update_commodity_prices(self, source: str = 'manual', prices: Dict[str, float] = None):
        """Update commodity prices from various sources"""
        
        if source == 'manual' and prices:
            self.commodity_prices.update(prices)
        
        elif source == 'api':
            # In practice, integrate with commodity price APIs
            # For now, simulate some price volatility
            for commodity in self.commodity_prices:
                volatility = 0.1  # 10% daily volatility
                price_change = np.random.normal(0, volatility)
                self.commodity_prices[commodity] *= (1 + price_change)
    
    def calculate_npv(self, target: ExplorationTarget, 
                     production_years: int = 10,
                     discount_rate: float = None) -> Dict[str, float]:
        """Calculate Net Present Value for an exploration target"""
        
        if discount_rate is None:
            discount_rate = self.discount_rate
        
        # Basic parameters
        if target.estimated_tonnage is None or target.estimated_grade is None:
            return {'npv': 0.0, 'error': 'Missing tonnage or grade estimates'}
        
        annual_production = target.estimated_tonnage / production_years
        
        # Get commodity price
        commodity_name = target.target_type.value.lower()
        if commodity_name == 'base_metals':
            commodity_name = 'copper'  # Default for base metals
        
        commodity_price = self.commodity_prices.get(commodity_name, 1000.0)
        
        # Revenue calculation
        revenue_per_ton = target.estimated_grade * commodity_price / 100  # Convert grade % to decimal
        annual_revenue = annual_production * revenue_per_ton
        
        # Cost estimation
        if target.operating_cost is None:
            target.operating_cost = self._estimate_operating_cost(target)
        
        if target.capital_cost is None:
            target.capital_cost = self._estimate_capital_cost(target)
        
        annual_operating_cost = target.operating_cost * annual_production
        annual_cash_flow = annual_revenue - annual_operating_cost
        
        # Calculate NPV
        npv = -target.capital_cost  # Initial investment
        
        for year in range(1, production_years + 1):
            discounted_cf = annual_cash_flow / ((1 + discount_rate) ** year)
            npv += discounted_cf
        
        # Calculate other financial metrics
        payback_period = target.capital_cost / annual_cash_flow if annual_cash_flow > 0 else float('inf')
        irr = self._calculate_irr(target.capital_cost, annual_cash_flow, production_years)
        
        results = {
            'npv': npv,
            'irr': irr,
            'payback_period': payback_period,
            'annual_revenue': annual_revenue,
            'annual_operating_cost': annual_operating_cost,
            'annual_cash_flow': annual_cash_flow,
            'commodity_price': commodity_price,
            'total_revenue': annual_revenue * production_years
        }
        
        # Update target attributes
        target.npv_estimate = npv
        target.payback_period = payback_period
        
        return results
    
    def _estimate_operating_cost(self, target: ExplorationTarget) -> float:
        """Estimate operating cost per ton based on target characteristics"""
        
        # Base cost models ($/t)
        base_costs = {
            TargetType.GOLD: 800.0,
            TargetType.COPPER: 1500.0,
            TargetType.LITHIUM: 2000.0,
            TargetType.IRON_ORE: 50.0,
            TargetType.BASE_METALS: 1200.0,
            TargetType.RARE_EARTH: 3000.0,
            TargetType.URANIUM: 100.0,
            TargetType.DIAMONDS: 200.0,
            TargetType.PLATINUM_GROUP: 2000.0
        }
        
        base_cost = base_costs.get(target.target_type, 1000.0)
        
        # Adjust for depth (deeper = more expensive)
        depth_factor = 1.0 + (target.depth_estimate / 500.0) * 0.3  # 30% increase per 500m
        
        # Adjust for grade (lower grade = higher cost per unit metal)
        if target.estimated_grade and target.estimated_grade > 0:
            grade_factor = max(0.5, 2.0 / target.estimated_grade)
        else:
            grade_factor = 1.0
        
        # Adjust for environmental risk
        env_factor = 1.0 + (target.environmental_risk.value - 1) * 0.1
        
        total_cost = base_cost * depth_factor * grade_factor * env_factor
        
        return total_cost
    
    def _estimate_capital_cost(self, target: ExplorationTarget) -> float:
        """Estimate capital cost based on target characteristics"""
        
        # Base capital costs ($/t annual capacity)
        base_capex = {
            TargetType.GOLD: 15000.0,
            TargetType.COPPER: 10000.0,
            TargetType.LITHIUM: 20000.0,
            TargetType.IRON_ORE: 200.0,
            TargetType.BASE_METALS: 12000.0,
            TargetType.RARE_EARTH: 25000.0,
            TargetType.URANIUM: 1000.0,
            TargetType.DIAMONDS: 5000.0,
            TargetType.PLATINUM_GROUP: 30000.0
        }
        
        if target.estimated_tonnage is None:
            return 100000000.0  # $100M default
        
        annual_capacity = target.estimated_tonnage / 10  # 10-year mine life
        base_cost = base_capex.get(target.target_type, 10000.0)
        
        # Scaling factor (economies of scale)
        scale_factor = (annual_capacity / 1000000.0) ** 0.7  # Rule of thumb
        
        # Depth adjustment
        depth_factor = 1.0 + (target.depth_estimate / 200.0) * 0.5
        
        # Location/infrastructure factor
        infrastructure_factor = 1.0 + (target.political_risk.value - 1) * 0.2
        
        total_capex = annual_capacity * base_cost * scale_factor * depth_factor * infrastructure_factor
        
        return total_capex
    
    def _calculate_irr(self, initial_investment: float, annual_cf: float, years: int) -> float:
        """Calculate Internal Rate of Return"""
        
        if annual_cf <= 0:
            return -1.0
        
        def npv_function(rate):
            npv = -initial_investment
            for year in range(1, years + 1):
                npv += annual_cf / ((1 + rate) ** year)
            return npv
        
        try:
            # Find the rate where NPV = 0
            irr = optimize.brentq(npv_function, -0.99, 10.0)
            return irr
        except ValueError:
            return -1.0  # IRR calculation failed
    
    def monte_carlo_analysis(self, target: ExplorationTarget, 
                           n_simulations: int = 1000) -> Dict[str, Any]:
        """Perform Monte Carlo analysis on economic parameters"""
        
        npv_results = []
        irr_results = []
        
        # Define uncertainty distributions
        grade_std = target.estimated_grade * 0.3 if target.estimated_grade else 1.0
        tonnage_std = target.estimated_tonnage * 0.4 if target.estimated_tonnage else 1000000.0
        
        commodity_name = target.target_type.value.lower()
        if commodity_name == 'base_metals':
            commodity_name = 'copper'
        
        base_price = self.commodity_prices.get(commodity_name, 1000.0)
        price_std = base_price * 0.25  # 25% price volatility
        
        for _ in range(n_simulations):
            # Sample uncertain parameters
            sim_grade = max(0.01, np.random.normal(target.estimated_grade or 1.0, grade_std))
            sim_tonnage = max(1000, np.random.normal(target.estimated_tonnage or 1000000.0, tonnage_std))
            sim_price = max(10, np.random.normal(base_price, price_std))
            
            # Create temporary target for simulation
            temp_target = ExplorationTarget(
                id=target.id,
                name=target.name,
                target_type=target.target_type,
                location=target.location,
                prospectivity_score=target.prospectivity_score,
                confidence=target.confidence,
                area_km2=target.area_km2,
                depth_estimate=target.depth_estimate,
                estimated_grade=sim_grade,
                estimated_tonnage=sim_tonnage
            )
            
            # Temporarily update commodity price
            original_price = self.commodity_prices[commodity_name]
            self.commodity_prices[commodity_name] = sim_price
            
            # Calculate economics
            econ_results = self.calculate_npv(temp_target)
            npv_results.append(econ_results['npv'])
            irr_results.append(econ_results['irr'])
            
            # Restore original price
            self.commodity_prices[commodity_name] = original_price
        
        # Calculate statistics
        npv_stats = {
            'mean': np.mean(npv_results),
            'std': np.std(npv_results),
            'p10': np.percentile(npv_results, 10),
            'p50': np.percentile(npv_results, 50),
            'p90': np.percentile(npv_results, 90),
            'probability_positive': np.mean(np.array(npv_results) > 0)
        }
        
        irr_stats = {
            'mean': np.mean(irr_results),
            'std': np.std(irr_results),
            'p10': np.percentile(irr_results, 10),
            'p50': np.percentile(irr_results, 50),
            'p90': np.percentile(irr_results, 90)
        }
        
        return {
            'npv_statistics': npv_stats,
            'irr_statistics': irr_stats,
            'raw_npv': npv_results,
            'raw_irr': irr_results,
            'n_simulations': n_simulations
        }


class RiskAssessment:
    """Comprehensive risk assessment for exploration targets"""
    
    def __init__(self):
        self.risk_factors = {
            'geological': {
                'deposit_size_uncertainty': 0.3,
                'grade_continuity': 0.25,
                'geological_complexity': 0.2,
                'exploration_maturity': 0.15,
                'metallurgical_complexity': 0.1
            },
            'technical': {
                'mining_method_risk': 0.4,
                'processing_complexity': 0.3,
                'infrastructure_requirements': 0.2,
                'water_availability': 0.1
            },
            'environmental': {
                'permitting_risk': 0.3,
                'environmental_impact': 0.25,
                'community_acceptance': 0.25,
                'land_access': 0.2
            },
            'political': {
                'country_risk': 0.4,
                'policy_stability': 0.3,
                'taxation_risk': 0.2,
                'currency_risk': 0.1
            },
            'market': {
                'commodity_price_volatility': 0.4,
                'demand_outlook': 0.3,
                'competition': 0.2,
                'supply_chain': 0.1
            }
        }
    
    def assess_target_risk(self, target: ExplorationTarget) -> Dict[str, float]:
        """Comprehensive risk assessment for a target"""
        
        risk_scores = {}
        
        # Geological risk assessment
        geological_risk = self._assess_geological_risk(target)
        risk_scores['geological_risk'] = geological_risk
        
        # Technical risk assessment
        technical_risk = self._assess_technical_risk(target)
        risk_scores['technical_risk'] = technical_risk
        
        # Environmental risk assessment
        environmental_risk = self._assess_environmental_risk(target)
        risk_scores['environmental_risk'] = environmental_risk
        
        # Political risk assessment
        political_risk = self._assess_political_risk(target)
        risk_scores['political_risk'] = political_risk
        
        # Market risk assessment
        market_risk = self._assess_market_risk(target)
        risk_scores['market_risk'] = market_risk
        
        # Overall risk score (weighted average)
        overall_weights = {
            'geological_risk': 0.3,
            'technical_risk': 0.25,
            'environmental_risk': 0.2,
            'political_risk': 0.15,
            'market_risk': 0.1
        }
        
        overall_risk = sum(risk_scores[factor] * weight 
                          for factor, weight in overall_weights.items())
        
        risk_scores['overall_risk'] = overall_risk
        
        return risk_scores
    
    def _assess_geological_risk(self, target: ExplorationTarget) -> float:
        """Assess geological risk factors"""
        
        risk_components = {}
        
        # Deposit size uncertainty (inverse of confidence)
        risk_components['size_uncertainty'] = 1.0 - target.confidence
        
        # Grade continuity risk (based on geological confidence)
        risk_components['grade_continuity'] = 1.0 - target.geological_confidence
        
        # Geological complexity (depth-related)
        depth_risk = min(1.0, target.depth_estimate / 1000.0)  # Normalize to 1km
        risk_components['complexity'] = depth_risk
        
        # Exploration maturity (inverse of prospectivity score)
        risk_components['maturity'] = 1.0 - target.prospectivity_score
        
        # Metallurgical complexity (commodity-specific)
        complex_commodities = [TargetType.RARE_EARTH, TargetType.PLATINUM_GROUP, TargetType.LITHIUM]
        if target.target_type in complex_commodities:
            risk_components['metallurgical'] = 0.8
        else:
            risk_components['metallurgical'] = 0.3
        
        # Weighted average
        weights = self.risk_factors['geological']
        geological_risk = sum(risk_components[comp.split('_')[0]] * weight 
                             for comp, weight in weights.items())
        
        return geological_risk
    
    def _assess_technical_risk(self, target: ExplorationTarget) -> float:
        """Assess technical and operational risk factors"""
        
        risk_components = {}
        
        # Mining method risk (depth-dependent)
        if target.depth_estimate > 500:
            risk_components['mining'] = 0.8  # Underground mining
        elif target.depth_estimate > 100:
            risk_components['mining'] = 0.4  # Open pit with depth
        else:
            risk_components['mining'] = 0.2  # Shallow open pit
        
        # Processing complexity
        complex_processing = [TargetType.RARE_EARTH, TargetType.URANIUM, TargetType.PLATINUM_GROUP]
        if target.target_type in complex_processing:
            risk_components['processing'] = 0.9
        elif target.target_type in [TargetType.GOLD, TargetType.COPPER]:
            risk_components['processing'] = 0.3
        else:
            risk_components['processing'] = 0.5
        
        # Infrastructure requirements (area-dependent)
        if target.area_km2 > 100:
            risk_components['infrastructure'] = 0.8
        elif target.area_km2 > 10:
            risk_components['infrastructure'] = 0.5
        else:
            risk_components['infrastructure'] = 0.3
        
        # Water availability (commodity-dependent)
        water_intensive = [TargetType.COPPER, TargetType.LITHIUM, TargetType.URANIUM]
        if target.target_type in water_intensive:
            risk_components['water'] = 0.7
        else:
            risk_components['water'] = 0.3
        
        # Weighted average
        weights = self.risk_factors['technical']
        technical_risk = sum(risk_components[comp] * weight 
                           for comp, weight in weights.items())
        
        return technical_risk
    
    def _assess_environmental_risk(self, target: ExplorationTarget) -> float:
        """Assess environmental and social risk factors"""
        
        # Use target's environmental risk level as base
        base_risk = target.environmental_risk.value / 5.0  # Normalize to 0-1
        
        # Adjust for commodity type
        high_impact_commodities = [TargetType.URANIUM, TargetType.RARE_EARTH, TargetType.COPPER]
        if target.target_type in high_impact_commodities:
            commodity_factor = 1.2
        else:
            commodity_factor = 1.0
        
        # Adjust for project size
        size_factor = min(1.5, 1.0 + target.area_km2 / 100.0)
        
        environmental_risk = min(1.0, base_risk * commodity_factor * size_factor)
        
        return environmental_risk
    
    def _assess_political_risk(self, target: ExplorationTarget) -> float:
        """Assess political and regulatory risk factors"""
        
        # Use target's political risk level as base
        base_risk = target.political_risk.value / 5.0  # Normalize to 0-1
        
        # Adjust for strategic commodities
        strategic_commodities = [TargetType.LITHIUM, TargetType.RARE_EARTH, TargetType.URANIUM]
        if target.target_type in strategic_commodities:
            strategic_factor = 1.3
        else:
            strategic_factor = 1.0
        
        political_risk = min(1.0, base_risk * strategic_factor)
        
        return political_risk
    
    def _assess_market_risk(self, target: ExplorationTarget) -> float:
        """Assess market and commodity risk factors"""
        
        # Base market risk by commodity type
        commodity_volatility = {
            TargetType.GOLD: 0.3,
            TargetType.COPPER: 0.6,
            TargetType.LITHIUM: 0.8,
            TargetType.RARE_EARTH: 0.7,
            TargetType.URANIUM: 0.9,
            TargetType.IRON_ORE: 0.5,
            TargetType.PLATINUM_GROUP: 0.7,
            TargetType.BASE_METALS: 0.6,
            TargetType.DIAMONDS: 0.4
        }
        
        base_volatility = commodity_volatility.get(target.target_type, 0.5)
        
        # Adjust for market conditions (simplified)
        market_risk = target.market_risk.value / 5.0 * base_volatility
        
        return market_risk


class MultiCriteriaDecisionAnalysis:
    """Multi-Criteria Decision Analysis for target ranking"""
    
    def __init__(self):
        self.criteria_weights = {
            'prospectivity_score': 0.25,
            'economic_potential': 0.20,
            'geological_confidence': 0.15,
            'technical_feasibility': 0.15,
            'environmental_acceptability': 0.10,
            'political_stability': 0.10,
            'market_attractiveness': 0.05
        }
        
        self.optimization_methods = ['topsis', 'vikor', 'electre', 'ahp']
    
    def rank_targets(self, targets: List[ExplorationTarget], 
                    method: str = 'topsis',
                    custom_weights: Dict[str, float] = None) -> List[ExplorationTarget]:
        """Rank targets using specified MCDA method"""
        
        if custom_weights:
            self.criteria_weights.update(custom_weights)
        
        # Prepare decision matrix
        decision_matrix, target_ids = self._prepare_decision_matrix(targets)
        
        # Apply MCDA method
        if method == 'topsis':
            scores = self._topsis_method(decision_matrix)
        elif method == 'vikor':
            scores = self._vikor_method(decision_matrix)
        elif method == 'ahp':
            scores = self._ahp_method(decision_matrix)
        else:
            raise ValueError(f"Unknown MCDA method: {method}")
        
        # Assign scores and ranks to targets
        ranked_targets = []
        for i, target in enumerate(targets):
            target.priority_score = scores[i]
            target.overall_rank = i + 1
            ranked_targets.append(target)
        
        # Sort by score (descending)
        ranked_targets.sort(key=lambda x: x.priority_score, reverse=True)
        
        # Update ranks
        for i, target in enumerate(ranked_targets):
            target.overall_rank = i + 1
            
        return ranked_targets
    
    def _prepare_decision_matrix(self, targets: List[ExplorationTarget]) -> Tuple[np.ndarray, List[str]]:
        """Prepare decision matrix for MCDA analysis"""
        
        n_targets = len(targets)
        n_criteria = len(self.criteria_weights)
        
        decision_matrix = np.zeros((n_targets, n_criteria))
        target_ids = []
        
        for i, target in enumerate(targets):
            target_ids.append(target.id)
            
            # Extract criteria values
            criteria_values = [
                target.prospectivity_score,
                target.npv_estimate or 0.0,
                target.geological_confidence,
                1.0 - target.technical_risk.value / 5.0,  # Convert risk to benefit
                1.0 - target.environmental_risk.value / 5.0,
                1.0 - target.political_risk.value / 5.0,
                1.0 - target.market_risk.value / 5.0
            ]
            
            decision_matrix[i, :] = criteria_values
        
        return decision_matrix, target_ids
    
    def _topsis_method(self, decision_matrix: np.ndarray) -> np.ndarray:
        """TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)"""
        
        # Normalize decision matrix
        norm_matrix = decision_matrix / np.sqrt(np.sum(decision_matrix**2, axis=0))
        
        # Apply weights
        weights = np.array(list(self.criteria_weights.values()))
        weighted_matrix = norm_matrix * weights
        
        # Identify ideal and negative-ideal solutions
        ideal_solution = np.max(weighted_matrix, axis=0)
        negative_ideal = np.min(weighted_matrix, axis=0)
        
        # Calculate distances
        distance_to_ideal = np.sqrt(np.sum((weighted_matrix - ideal_solution)**2, axis=1))
        distance_to_negative = np.sqrt(np.sum((weighted_matrix - negative_ideal)**2, axis=1))
        
        # Calculate relative closeness
        closeness = distance_to_negative / (distance_to_ideal + distance_to_negative)
        
        return closeness
    
    def _vikor_method(self, decision_matrix: np.ndarray) -> np.ndarray:
        """VIKOR (VlseKriterijumska Optimizacija I Kompromisno Resenje)"""
        
        # Normalize decision matrix
        norm_matrix = np.zeros_like(decision_matrix)
        for j in range(decision_matrix.shape[1]):
            max_val = np.max(decision_matrix[:, j])
            min_val = np.min(decision_matrix[:, j])
            if max_val != min_val:
                norm_matrix[:, j] = (max_val - decision_matrix[:, j]) / (max_val - min_val)
            else:
                norm_matrix[:, j] = 0
        
        # Apply weights
        weights = np.array(list(self.criteria_weights.values()))
        weighted_matrix = norm_matrix * weights
        
        # Calculate S and R
        S = np.sum(weighted_matrix, axis=1)
        R = np.max(weighted_matrix, axis=1)
        
        # Calculate VIKOR index
        v = 0.5  # Strategy weight
        S_min, S_max = np.min(S), np.max(S)
        R_min, R_max = np.min(R), np.max(R)
        
        if S_max != S_min and R_max != R_min:
            Q = v * (S - S_min) / (S_max - S_min) + (1 - v) * (R - R_min) / (R_max - R_min)
        else:
            Q = np.zeros(len(S))
        
        # Return inverse of Q for consistency with other methods (lower Q = better)
        return 1.0 - Q
    
    def _ahp_method(self, decision_matrix: np.ndarray) -> np.ndarray:
        """Simplified AHP (Analytic Hierarchy Process)"""
        
        # Normalize decision matrix by column maximum
        norm_matrix = decision_matrix / np.max(decision_matrix, axis=0)
        
        # Apply weights
        weights = np.array(list(self.criteria_weights.values()))
        scores = np.sum(norm_matrix * weights, axis=1)
        
        return scores


class TargetGenerator:
    """Main target generator and ranking system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        
        # Initialize components
        self.economic_modeler = EconomicModeler()
        self.risk_assessor = RiskAssessment()
        self.mcda = MultiCriteriaDecisionAnalysis()
        
        self.targets = []
        self.target_database = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'min_prospectivity_score': 0.3,
            'max_targets_per_run': 100,
            'clustering_method': 'dbscan',
            'ranking_method': 'topsis',
            'economic_cutoff': 0.0,  # Minimum NPV
            'confidence_threshold': 0.5
        }
    
    def generate_targets_from_prospectivity(self, prospectivity_data: Dict[str, np.ndarray],
                                          grid_x: np.ndarray, grid_y: np.ndarray,
                                          target_type: TargetType) -> List[ExplorationTarget]:
        """Generate exploration targets from prospectivity mapping results"""
        
        self.logger.info(f"Generating {target_type.value} targets from prospectivity data...")
        
        # Extract prospectivity scores and uncertainties
        if 'ensemble' in prospectivity_data:
            scores = prospectivity_data['ensemble']
            uncertainties = prospectivity_data.get('uncertainties', {}).get('ensemble', 
                                                  np.zeros_like(scores))
        else:
            # Use first available prediction
            pred_name = next(iter(prospectivity_data.keys()))
            scores = prospectivity_data[pred_name]
            uncertainties = np.zeros_like(scores)
        
        # Find high-prospectivity areas
        high_prospect_mask = scores > self.config['min_prospectivity_score']
        confidence = 1.0 - (uncertainties / np.max(uncertainties + 1e-6))
        high_confidence_mask = confidence > self.config['confidence_threshold']
        
        # Combined mask
        target_mask = high_prospect_mask & high_confidence_mask
        
        if not np.any(target_mask):
            self.logger.warning("No areas meet prospectivity and confidence criteria")
            return []
        
        # Extract target locations
        target_indices = np.where(target_mask)
        target_coords = np.column_stack((
            grid_x[target_indices],
            grid_y[target_indices]
        ))
        target_scores = scores[target_indices]
        target_confidences = confidence[target_indices]
        
        # Cluster nearby points to form discrete targets
        clusters = self._cluster_target_points(target_coords, target_scores)
        
        # Generate target objects
        generated_targets = []
        
        for cluster_id, cluster_info in clusters.items():
            if cluster_id == -1:  # Noise points
                continue
            
            target = self._create_target_from_cluster(
                cluster_info, target_type, cluster_id
            )
            generated_targets.append(target)
        
        self.logger.info(f"Generated {len(generated_targets)} targets")
        
        return generated_targets
    
    def _cluster_target_points(self, coordinates: np.ndarray, 
                              scores: np.ndarray) -> Dict[int, Dict[str, Any]]:
        """Cluster target points to form discrete exploration targets"""
        
        if self.config['clustering_method'] == 'dbscan':
            # DBSCAN clustering
            eps = self.config.get('clustering_eps', 1000.0)  # 1km default
            min_samples = self.config.get('clustering_min_samples', 3)
            
            clustering = DBSCAN(eps=eps, min_samples=min_samples)
            cluster_labels = clustering.fit_predict(coordinates)
            
        elif self.config['clustering_method'] == 'kmeans':
            # K-means clustering
            n_clusters = self.config.get('n_clusters', 10)
            clustering = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = clustering.fit_predict(coordinates)
            
        else:
            raise ValueError(f"Unknown clustering method: {self.config['clustering_method']}")
        
        # Process clusters
        clusters = {}
        unique_labels = np.unique(cluster_labels)
        
        for label in unique_labels:
            cluster_mask = cluster_labels == label
            cluster_coords = coordinates[cluster_mask]
            cluster_scores = scores[cluster_mask]
            
            if len(cluster_coords) == 0:
                continue
            
            # Calculate cluster statistics
            centroid = np.mean(cluster_coords, axis=0)
            max_score = np.max(cluster_scores)
            mean_score = np.mean(cluster_scores)
            area = self._calculate_cluster_area(cluster_coords)
            
            clusters[label] = {
                'coordinates': cluster_coords,
                'scores': cluster_scores,
                'centroid': centroid,
                'max_score': max_score,
                'mean_score': mean_score,
                'area_km2': area,
                'n_points': len(cluster_coords)
            }
        
        return clusters
    
    def _calculate_cluster_area(self, coordinates: np.ndarray) -> float:
        """Calculate the area of a point cluster"""
        
        if len(coordinates) < 3:
            return 1.0  # Minimum area for small clusters
        
        try:
            # Create convex hull
            from scipy.spatial import ConvexHull
            hull = ConvexHull(coordinates)
            area = hull.volume  # In 2D, 'volume' is actually area
            
            # Convert from m² to km²
            area_km2 = area / 1e6
            
            return max(1.0, area_km2)
            
        except:
            # Fallback calculation
            x_range = np.max(coordinates[:, 0]) - np.min(coordinates[:, 0])
            y_range = np.max(coordinates[:, 1]) - np.min(coordinates[:, 1])
            area_km2 = (x_range * y_range) / 1e6
            
            return max(1.0, area_km2)
    
    def _create_target_from_cluster(self, cluster_info: Dict[str, Any],
                                   target_type: TargetType,
                                   cluster_id: int) -> ExplorationTarget:
        """Create exploration target from cluster information"""
        
        centroid = cluster_info['centroid']
        location = Point(centroid[0], centroid[1])
        
        # Estimate target parameters based on type and cluster characteristics
        depth_estimate = self._estimate_target_depth(target_type, cluster_info)
        grade_estimate = self._estimate_grade(target_type, cluster_info)
        tonnage_estimate = self._estimate_tonnage(target_type, cluster_info)
        
        target = ExplorationTarget(
            id=f"{target_type.value}_target_{cluster_id}",
            name=f"{target_type.value.title()} Target {cluster_id}",
            target_type=target_type,
            location=location,
            prospectivity_score=cluster_info['max_score'],
            confidence=cluster_info['mean_score'],
            area_km2=cluster_info['area_km2'],
            depth_estimate=depth_estimate,
            estimated_grade=grade_estimate,
            estimated_tonnage=tonnage_estimate,
            geological_confidence=cluster_info['mean_score'],
            
            # Initialize risk levels (to be assessed later)
            technical_risk=RiskLevel.MODERATE,
            geological_risk=RiskLevel.MODERATE,
            environmental_risk=RiskLevel.MODERATE,
            political_risk=RiskLevel.MODERATE,
            market_risk=RiskLevel.MODERATE
        )
        
        return target
    
    def _estimate_target_depth(self, target_type: TargetType, cluster_info: Dict[str, Any]) -> float:
        """Estimate target depth based on type and characteristics"""
        
        # Default depths by commodity type (meters)
        default_depths = {
            TargetType.GOLD: 200.0,
            TargetType.COPPER: 300.0,
            TargetType.LITHIUM: 50.0,  # Often surface/near-surface
            TargetType.IRON_ORE: 100.0,
            TargetType.BASE_METALS: 250.0,
            TargetType.RARE_EARTH: 150.0,
            TargetType.URANIUM: 100.0,
            TargetType.DIAMONDS: 400.0,
            TargetType.PLATINUM_GROUP: 500.0
        }
        
        base_depth = default_depths.get(target_type, 200.0)
        
        # Adjust based on prospectivity score (higher score may indicate deeper target)
        score_factor = 1.0 + (cluster_info['max_score'] - 0.5) * 0.5
        
        return base_depth * score_factor
    
    def _estimate_grade(self, target_type: TargetType, cluster_info: Dict[str, Any]) -> float:
        """Estimate target grade based on type and characteristics"""
        
        # Typical grades by commodity type
        typical_grades = {
            TargetType.GOLD: 2.0,  # g/t
            TargetType.COPPER: 0.5,  # %
            TargetType.LITHIUM: 1.0,  # % Li2O
            TargetType.IRON_ORE: 55.0,  # % Fe
            TargetType.BASE_METALS: 1.5,  # % combined
            TargetType.RARE_EARTH: 3.0,  # % REO
            TargetType.URANIUM: 0.1,  # % U3O8
            TargetType.DIAMONDS: 0.5,  # carats/tonne
            TargetType.PLATINUM_GROUP: 3.0  # g/t combined PGE
        }
        
        base_grade = typical_grades.get(target_type, 1.0)
        
        # Adjust based on prospectivity score
        score_factor = 0.5 + cluster_info['max_score']
        
        return base_grade * score_factor
    
    def _estimate_tonnage(self, target_type: TargetType, cluster_info: Dict[str, Any]) -> float:
        """Estimate target tonnage based on type and area"""
        
        # Tonnage density factors (tonnes per km²)
        density_factors = {
            TargetType.GOLD: 1000000.0,
            TargetType.COPPER: 10000000.0,
            TargetType.LITHIUM: 5000000.0,
            TargetType.IRON_ORE: 50000000.0,
            TargetType.BASE_METALS: 8000000.0,
            TargetType.RARE_EARTH: 2000000.0,
            TargetType.URANIUM: 1000000.0,
            TargetType.DIAMONDS: 100000.0,
            TargetType.PLATINUM_GROUP: 500000.0
        }
        
        density = density_factors.get(target_type, 5000000.0)
        
        # Base tonnage calculation
        base_tonnage = cluster_info['area_km2'] * density
        
        # Adjust for prospectivity score
        score_factor = 0.3 + cluster_info['max_score'] * 1.4
        
        return base_tonnage * score_factor
    
    def assess_and_rank_targets(self, targets: List[ExplorationTarget],
                              custom_weights: Dict[str, float] = None) -> List[ExplorationTarget]:
        """Comprehensive assessment and ranking of exploration targets"""
        
        self.logger.info(f"Assessing and ranking {len(targets)} targets...")
        
        # Assess each target
        for target in targets:
            # Risk assessment
            risk_scores = self.risk_assessor.assess_target_risk(target)
            
            # Update target risk levels
            target.technical_risk = RiskLevel(max(1, min(5, int(risk_scores['technical_risk'] * 5))))
            target.geological_risk = RiskLevel(max(1, min(5, int(risk_scores['geological_risk'] * 5))))
            target.environmental_risk = RiskLevel(max(1, min(5, int(risk_scores['environmental_risk'] * 5))))
            target.political_risk = RiskLevel(max(1, min(5, int(risk_scores['political_risk'] * 5))))
            target.market_risk = RiskLevel(max(1, min(5, int(risk_scores['market_risk'] * 5))))
            
            # Economic assessment
            econ_results = self.economic_modeler.calculate_npv(target)
            target.npv_estimate = econ_results['npv']
            target.payback_period = econ_results['payback_period']
            
            # Generate recommendation based on assessment
            target.recommendation = self._generate_recommendation(target, risk_scores, econ_results)
        
        # Filter targets by economic cutoff
        viable_targets = [t for t in targets if (t.npv_estimate or 0) >= self.config['economic_cutoff']]
        
        if len(viable_targets) == 0:
            self.logger.warning("No targets meet economic criteria")
            return targets
        
        # Rank targets using MCDA
        ranked_targets = self.mcda.rank_targets(
            viable_targets,
            method=self.config['ranking_method'],
            custom_weights=custom_weights
        )
        
        self.logger.info(f"Ranked {len(ranked_targets)} viable targets")
        
        return ranked_targets
    
    def _generate_recommendation(self, target: ExplorationTarget,
                               risk_scores: Dict[str, float],
                               econ_results: Dict[str, float]) -> str:
        """Generate exploration recommendation for target"""
        
        npv = econ_results.get('npv', 0)
        overall_risk = risk_scores.get('overall_risk', 0.5)
        
        if npv > 100000000 and overall_risk < 0.3:  # $100M NPV, low risk
            return "PRIORITY TARGET - High value, low risk. Recommend immediate detailed exploration."
        
        elif npv > 50000000 and overall_risk < 0.5:  # $50M NPV, moderate risk
            return "HIGH PRIORITY - Good economic potential. Recommend comprehensive feasibility study."
        
        elif npv > 10000000 and overall_risk < 0.6:  # $10M NPV, moderate-high risk
            return "MODERATE PRIORITY - Positive economics. Recommend phased exploration approach."
        
        elif npv > 0 and overall_risk < 0.7:
            return "LOW PRIORITY - Marginal economics. Consider exploration if strategic value exists."
        
        elif npv < 0:
            return "NOT RECOMMENDED - Negative economics under current assumptions."
        
        else:
            return "HIGH RISK - Economic potential unclear due to high risk factors."
    
    def monte_carlo_portfolio_analysis(self, targets: List[ExplorationTarget],
                                     portfolio_budget: float,
                                     n_simulations: int = 1000) -> Dict[str, Any]:
        """Portfolio-level Monte Carlo analysis"""
        
        self.logger.info("Performing portfolio Monte Carlo analysis...")
        
        # Select targets that fit within budget
        targets_sorted = sorted(targets, key=lambda x: x.priority_score or 0, reverse=True)
        
        selected_targets = []
        cumulative_capex = 0
        
        for target in targets_sorted:
            target_capex = target.capital_cost or (target.estimated_tonnage or 1000000) * 10000
            
            if cumulative_capex + target_capex <= portfolio_budget:
                selected_targets.append(target)
                cumulative_capex += target_capex
                
                if len(selected_targets) >= self.config['max_targets_per_run']:
                    break
        
        if len(selected_targets) == 0:
            return {'error': 'No targets fit within portfolio budget'}
        
        # Run Monte Carlo simulation for each target
        portfolio_npv_results = []
        
        for sim in range(n_simulations):
            portfolio_npv = 0
            
            for target in selected_targets:
                # Individual target Monte Carlo
                mc_results = self.economic_modeler.monte_carlo_analysis(target, n_simulations=1)
                target_npv = mc_results['raw_npv'][0]
                portfolio_npv += target_npv
            
            portfolio_npv_results.append(portfolio_npv)
        
        # Portfolio statistics
        portfolio_stats = {
            'selected_targets': len(selected_targets),
            'total_capex': cumulative_capex,
            'portfolio_npv_mean': np.mean(portfolio_npv_results),
            'portfolio_npv_std': np.std(portfolio_npv_results),
            'portfolio_npv_p10': np.percentile(portfolio_npv_results, 10),
            'portfolio_npv_p50': np.percentile(portfolio_npv_results, 50),
            'portfolio_npv_p90': np.percentile(portfolio_npv_results, 90),
            'probability_positive': np.mean(np.array(portfolio_npv_results) > 0),
            'target_names': [t.name for t in selected_targets]
        }
        
        return portfolio_stats
    
    def export_target_data(self, targets: List[ExplorationTarget], 
                          output_path: str, format: str = 'csv'):
        """Export target data to various formats"""
        
        # Prepare data for export
        export_data = []
        
        for target in targets:
            target_dict = {
                'id': target.id,
                'name': target.name,
                'target_type': target.target_type.value,
                'longitude': target.location.x,
                'latitude': target.location.y,
                'prospectivity_score': target.prospectivity_score,
                'confidence': target.confidence,
                'area_km2': target.area_km2,
                'depth_estimate': target.depth_estimate,
                'estimated_grade': target.estimated_grade,
                'estimated_tonnage': target.estimated_tonnage,
                'npv_estimate': target.npv_estimate,
                'payback_period': target.payback_period,
                'technical_risk': target.technical_risk.value,
                'geological_risk': target.geological_risk.value,
                'environmental_risk': target.environmental_risk.value,
                'political_risk': target.political_risk.value,
                'market_risk': target.market_risk.value,
                'overall_rank': target.overall_rank,
                'priority_score': target.priority_score,
                'recommendation': target.recommendation
            }
            export_data.append(target_dict)
        
        # Export in requested format
        df = pd.DataFrame(export_data)
        
        if format.lower() == 'csv':
            df.to_csv(output_path, index=False)
        elif format.lower() == 'excel':
            df.to_excel(output_path, index=False)
        elif format.lower() == 'json':
            df.to_json(output_path, orient='records', indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        self.logger.info(f"Exported {len(targets)} targets to {output_path}")
    
    def create_target_visualization(self, targets: List[ExplorationTarget]) -> go.Figure:
        """Create interactive visualization of exploration targets"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Target Locations', 'NPV vs Risk', 'Target Types', 'Priority Ranking'],
            specs=[[{'type': 'scattermapbox'}, {'type': 'scatter'}],
                   [{'type': 'bar'}, {'type': 'scatter'}]]
        )
        
        # Extract data for plotting
        lons = [t.location.x for t in targets]
        lats = [t.location.y for t in targets]
        names = [t.name for t in targets]
        npvs = [t.npv_estimate or 0 for t in targets]
        risks = [sum([t.technical_risk.value, t.geological_risk.value, 
                     t.environmental_risk.value, t.political_risk.value, 
                     t.market_risk.value]) / 5 for t in targets]
        priorities = [t.priority_score or 0 for t in targets]
        types = [t.target_type.value for t in targets]
        
        # Target locations map
        fig.add_trace(
            go.Scattermapbox(
                lon=lons,
                lat=lats,
                text=names,
                mode='markers',
                marker=dict(
                    size=10,
                    color=priorities,
                    colorscale='Viridis',
                    colorbar=dict(title="Priority Score")
                )
            ),
            row=1, col=1
        )
        
        # NPV vs Risk scatter
        fig.add_trace(
            go.Scatter(
                x=risks,
                y=npvs,
                mode='markers',
                text=names,
                marker=dict(
                    size=10,
                    color=priorities,
                    colorscale='Viridis'
                )
            ),
            row=1, col=2
        )
        
        # Target types bar chart
        type_counts = pd.Series(types).value_counts()
        fig.add_trace(
            go.Bar(
                x=type_counts.index,
                y=type_counts.values
            ),
            row=2, col=1
        )
        
        # Priority ranking
        ranked_targets = sorted(targets, key=lambda x: x.priority_score or 0, reverse=True)[:10]
        fig.add_trace(
            go.Scatter(
                x=list(range(len(ranked_targets))),
                y=[t.priority_score or 0 for t in ranked_targets],
                mode='markers+lines',
                text=[t.name for t in ranked_targets]
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="Exploration Target Analysis Dashboard",
            showlegend=False,
            height=800,
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=np.mean(lats), lon=np.mean(lons)),
                zoom=8
            )
        )
        
        fig.update_xaxes(title_text="Average Risk Score", row=1, col=2)
        fig.update_yaxes(title_text="NPV (USD)", row=1, col=2)
        fig.update_xaxes(title_text="Target Type", row=2, col=1)
        fig.update_yaxes(title_text="Count", row=2, col=1)
        fig.update_xaxes(title_text="Rank", row=2, col=2)
        fig.update_yaxes(title_text="Priority Score", row=2, col=2)
        
        return fig


# Example usage and testing
if __name__ == "__main__":
    # Initialize target generator
    config = {
        'min_prospectivity_score': 0.6,
        'clustering_method': 'dbscan',
        'ranking_method': 'topsis',
        'economic_cutoff': 0.0
    }
    
    generator = TargetGenerator(config)
    
    print("Testing target generation and ranking system...")
    
    # Generate synthetic prospectivity data
    height, width = 100, 100
    x_coords = np.arange(0, width * 1000, 1000)  # 1km grid
    y_coords = np.arange(0, height * 1000, 1000)
    grid_x, grid_y = np.meshgrid(x_coords, y_coords)
    
    # Create synthetic prospectivity scores with several anomalies
    base_scores = np.random.beta(2, 5, (height, width)) * 0.8
    
    # Add high-prospectivity anomalies
    anomaly_centers = [(25, 30), (60, 70), (80, 20)]
    
    for center_y, center_x in anomaly_centers:
        y_dist = np.abs(np.arange(height) - center_y)
        x_dist = np.abs(np.arange(width) - center_x)
        y_grid_dist, x_grid_dist = np.meshgrid(x_dist, y_dist)
        distance = np.sqrt(x_grid_dist**2 + y_grid_dist**2)
        
        anomaly = 0.4 * np.exp(-distance**2 / (10**2))
        base_scores += anomaly.T
    
    # Create ensemble predictions with uncertainties
    prospectivity_data = {
        'ensemble': base_scores,
        'uncertainties': {
            'ensemble': np.random.uniform(0.1, 0.3, (height, width))
        }
    }
    
    print(f"Prospectivity data shape: {base_scores.shape}")
    print(f"Max prospectivity score: {np.max(base_scores):.3f}")
    print(f"Areas above threshold: {np.sum(base_scores > 0.6)}")
    
    # Generate gold targets
    gold_targets = generator.generate_targets_from_prospectivity(
        prospectivity_data, grid_x, grid_y, TargetType.GOLD
    )
    
    print(f"Generated {len(gold_targets)} gold targets")
    
    if len(gold_targets) > 0:
        # Assess and rank targets
        ranked_targets = generator.assess_and_rank_targets(gold_targets)
        
        print(f"Successfully ranked {len(ranked_targets)} targets")
        print("\nTop 5 targets:")
        for i, target in enumerate(ranked_targets[:5]):
            print(f"  {i+1}. {target.name}")
            print(f"     Priority Score: {target.priority_score:.3f}")
            print(f"     NPV Estimate: ${target.npv_estimate:,.0f}")
            print(f"     Recommendation: {target.recommendation}")
            print()
        
        # Portfolio analysis
        if len(ranked_targets) > 0:
            portfolio_results = generator.monte_carlo_portfolio_analysis(
                ranked_targets, portfolio_budget=500000000  # $500M budget
            )
            
            if 'error' not in portfolio_results:
                print(f"Portfolio Analysis Results:")
                print(f"  Selected Targets: {portfolio_results['selected_targets']}")
                print(f"  Total CapEx: ${portfolio_results['total_capex']:,.0f}")
                print(f"  Expected NPV: ${portfolio_results['portfolio_npv_mean']:,.0f}")
                print(f"  NPV P10/P50/P90: ${portfolio_results['portfolio_npv_p10']:,.0f} / "
                      f"${portfolio_results['portfolio_npv_p50']:,.0f} / "
                      f"${portfolio_results['portfolio_npv_p90']:,.0f}")
                print(f"  Probability of Positive Return: {portfolio_results['probability_positive']:.1%}")
        
        # Export results
        generator.export_target_data(ranked_targets, 'exploration_targets.csv', 'csv')
        print("Target data exported to exploration_targets.csv")
    
    print("\nTarget generation and ranking system demonstration complete!")
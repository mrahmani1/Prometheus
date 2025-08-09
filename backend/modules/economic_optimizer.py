"""
Economic Optimization for Mining Operations
NPV calculations, mining sequence optimization, processing route selection, and risk assessment
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution, linprog
from scipy.stats import norm, lognorm, uniform
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union, Callable
import json
import warnings
from datetime import datetime, timedelta
import random

@dataclass
class EconomicParameters:
    """Economic modeling parameters"""
    discount_rate: float  # Annual discount rate (%)
    commodity_prices: Dict[str, float]  # USD per unit
    operating_costs: Dict[str, float]  # USD per unit
    capital_costs: Dict[str, float]  # USD
    tax_rate: float  # Tax rate (%)
    royalty_rate: float  # Royalty rate (%)
    inflation_rate: float  # Annual inflation rate (%)

@dataclass
class MiningBlock:
    """Individual mining block"""
    block_id: str
    coordinates: Tuple[float, float, float]
    tonnage: float
    grades: Dict[str, float]  # Element grades (%)
    rock_type: str
    mining_cost: float  # USD per tonne
    processing_cost: float  # USD per tonne
    precedence_constraints: List[str]  # Blocks that must be mined first
    geotechnical_risk: float  # Risk factor (0-1)

@dataclass
class ProcessingRoute:
    """Ore processing route"""
    route_name: str
    applicable_ore_types: List[str]
    recovery_rates: Dict[str, float]  # Recovery by element (%)
    operating_cost: float  # USD per tonne
    capital_cost: float  # USD
    capacity: float  # Tonnes per year
    environmental_impact: float  # Impact score

@dataclass
class MineSchedule:
    """Mine production schedule"""
    year: int
    blocks_mined: List[str]
    total_tonnage: float
    metal_production: Dict[str, float]
    operating_costs: float
    capital_costs: float
    revenue: float
    cash_flow: float

class NPVCalculator:
    """Net Present Value calculations and sensitivity analysis"""
    
    def __init__(self, economic_params: EconomicParameters):
        self.params = economic_params
        self.price_forecasts = {}
        self.cost_forecasts = {}
    
    def calculate_block_value(self, block: MiningBlock, processing_route: ProcessingRoute) -> float:
        """Calculate economic value of individual mining block"""
        # Calculate revenue
        revenue = 0
        for element, grade in block.grades.items():
            if element in self.params.commodity_prices:
                price = self.params.commodity_prices[element]
                recovery = processing_route.recovery_rates.get(element, 0.85)
                metal_content = block.tonnage * grade / 100  # Convert % to decimal
                recovered_metal = metal_content * recovery
                revenue += recovered_metal * price
        
        # Calculate costs
        mining_cost = block.tonnage * block.mining_cost
        processing_cost = block.tonnage * processing_route.operating_cost
        total_costs = mining_cost + processing_cost
        
        # Apply taxes and royalties
        gross_profit = revenue - total_costs
        royalties = revenue * self.params.royalty_rate
        taxable_income = max(0, gross_profit - royalties)
        taxes = taxable_income * self.params.tax_rate
        
        net_value = gross_profit - royalties - taxes
        
        # Apply geotechnical risk adjustment
        risk_adjusted_value = net_value * (1 - block.geotechnical_risk)
        
        return risk_adjusted_value
    
    def calculate_project_npv(self, mine_schedule: List[MineSchedule]) -> Dict:
        """Calculate project NPV from mine schedule"""
        
        total_npv = 0
        annual_cash_flows = []
        cumulative_npv = []
        
        for year_data in mine_schedule:
            year = year_data.year
            cash_flow = year_data.cash_flow
            
            # Discount cash flow
            discount_factor = (1 + self.params.discount_rate / 100) ** year
            discounted_cash_flow = cash_flow / discount_factor
            
            total_npv += discounted_cash_flow
            annual_cash_flows.append(discounted_cash_flow)
            cumulative_npv.append(total_npv)
        
        # Calculate additional metrics
        initial_investment = abs(min(0, min([cf.cash_flow for cf in mine_schedule])))
        
        # Payback period
        payback_period = self._calculate_payback_period(mine_schedule)
        
        # IRR approximation
        irr = self._approximate_irr(mine_schedule)
        
        return {
            'npv': total_npv,
            'annual_discounted_cash_flows': annual_cash_flows,
            'cumulative_npv': cumulative_npv,
            'initial_investment': initial_investment,
            'payback_period': payback_period,
            'irr': irr,
            'total_revenue': sum(year.revenue for year in mine_schedule),
            'total_operating_costs': sum(year.operating_costs for year in mine_schedule),
            'total_capital_costs': sum(year.capital_costs for year in mine_schedule)
        }
    
    def _calculate_payback_period(self, schedule: List[MineSchedule]) -> float:
        """Calculate payback period in years"""
        cumulative_cash_flow = 0
        
        for i, year_data in enumerate(schedule):
            cumulative_cash_flow += year_data.cash_flow
            if cumulative_cash_flow >= 0:
                return i + 1
        
        return float('inf')  # No payback
    
    def _approximate_irr(self, schedule: List[MineSchedule]) -> float:
        """Approximate Internal Rate of Return using binary search"""
        
        def npv_at_rate(rate):
            npv = 0
            for year_data in schedule:
                if year_data.year > 0:
                    npv += year_data.cash_flow / ((1 + rate) ** year_data.year)
                else:
                    npv += year_data.cash_flow
            return npv
        
        # Binary search for IRR
        low_rate, high_rate = -0.99, 5.0
        tolerance = 1e-6
        
        for _ in range(100):  # Max iterations
            mid_rate = (low_rate + high_rate) / 2
            npv = npv_at_rate(mid_rate)
            
            if abs(npv) < tolerance:
                return mid_rate * 100  # Convert to percentage
            
            if npv > 0:
                low_rate = mid_rate
            else:
                high_rate = mid_rate
        
        return mid_rate * 100
    
    def sensitivity_analysis(self, base_schedule: List[MineSchedule],
                           sensitivity_factors: Dict[str, List[float]]) -> Dict:
        """Perform sensitivity analysis on key parameters"""
        
        base_npv = self.calculate_project_npv(base_schedule)['npv']
        sensitivity_results = {}
        
        for parameter, factor_range in sensitivity_factors.items():
            parameter_results = []
            
            for factor in factor_range:
                # Create modified schedule
                modified_schedule = self._apply_sensitivity_factor(
                    base_schedule, parameter, factor
                )
                
                modified_npv = self.calculate_project_npv(modified_schedule)['npv']
                npv_change = ((modified_npv - base_npv) / base_npv) * 100
                
                parameter_results.append({
                    'factor': factor,
                    'npv': modified_npv,
                    'npv_change_percent': npv_change
                })
            
            sensitivity_results[parameter] = parameter_results
        
        return {
            'base_npv': base_npv,
            'sensitivity_analysis': sensitivity_results
        }
    
    def _apply_sensitivity_factor(self, schedule: List[MineSchedule],
                                 parameter: str, factor: float) -> List[MineSchedule]:
        """Apply sensitivity factor to schedule"""
        modified_schedule = []
        
        for year_data in schedule:
            new_year_data = MineSchedule(
                year=year_data.year,
                blocks_mined=year_data.blocks_mined.copy(),
                total_tonnage=year_data.total_tonnage,
                metal_production=year_data.metal_production.copy(),
                operating_costs=year_data.operating_costs,
                capital_costs=year_data.capital_costs,
                revenue=year_data.revenue,
                cash_flow=year_data.cash_flow
            )
            
            if parameter == 'commodity_prices':
                new_year_data.revenue *= factor
            elif parameter == 'operating_costs':
                new_year_data.operating_costs *= factor
            elif parameter == 'capital_costs':
                new_year_data.capital_costs *= factor
            elif parameter == 'discount_rate':
                # This would require recalculating NPV with modified discount rate
                pass
            
            # Recalculate cash flow
            new_year_data.cash_flow = (new_year_data.revenue - 
                                     new_year_data.operating_costs - 
                                     new_year_data.capital_costs)
            
            modified_schedule.append(new_year_data)
        
        return modified_schedule

class MiningSequenceOptimizer:
    """Optimize mining sequence for maximum NPV"""
    
    def __init__(self, blocks: List[MiningBlock], economic_params: EconomicParameters):
        self.blocks = blocks
        self.params = economic_params
        self.precedence_matrix = self._build_precedence_matrix()
        self.processing_capacity = 1e6  # Default 1 Mt/year
    
    def _build_precedence_matrix(self) -> np.ndarray:
        """Build precedence constraint matrix"""
        n_blocks = len(self.blocks)
        block_ids = [block.block_id for block in self.blocks]
        
        precedence_matrix = np.zeros((n_blocks, n_blocks))
        
        for i, block in enumerate(self.blocks):
            for constraint_id in block.precedence_constraints:
                if constraint_id in block_ids:
                    j = block_ids.index(constraint_id)
                    precedence_matrix[i, j] = 1  # Block j must be mined before block i
        
        return precedence_matrix
    
    def optimize_sequence_milp(self, mine_life: int = 10) -> Dict:
        """Optimize mining sequence using Mixed Integer Linear Programming"""
        try:
            from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpStatus
        except ImportError:
            warnings.warn("PuLP not available - using heuristic optimization")
            return self.optimize_sequence_heuristic(mine_life)
        
        n_blocks = len(self.blocks)
        
        # Create problem
        prob = LpProblem("Mining_Sequence_Optimization", LpMaximize)
        
        # Decision variables: x[i,t] = 1 if block i is mined in period t
        x = {}
        for i in range(n_blocks):
            for t in range(mine_life):
                x[i, t] = LpVariable(f"x_{i}_{t}", cat='Binary')
        
        # Objective function: maximize NPV
        discount_factors = [(1 + self.params.discount_rate/100)**(-t) for t in range(mine_life)]
        
        prob += lpSum([
            self.blocks[i].tonnage * self._estimate_block_value_per_tonne(self.blocks[i]) *
            discount_factors[t] * x[i, t]
            for i in range(n_blocks) for t in range(mine_life)
        ])
        
        # Constraints
        # 1. Each block mined at most once
        for i in range(n_blocks):
            prob += lpSum([x[i, t] for t in range(mine_life)]) <= 1
        
        # 2. Processing capacity constraints
        for t in range(mine_life):
            prob += lpSum([self.blocks[i].tonnage * x[i, t] for i in range(n_blocks)]) <= self.processing_capacity
        
        # 3. Precedence constraints
        for i in range(n_blocks):
            for j in range(n_blocks):
                if self.precedence_matrix[i, j] == 1:
                    for t in range(mine_life):
                        prob += lpSum([x[j, s] for s in range(t+1)]) >= x[i, t]
        
        # Solve
        prob.solve()
        
        if LpStatus[prob.status] == 'Optimal':
            # Extract solution
            mining_schedule = []
            for t in range(mine_life):
                year_blocks = []
                year_tonnage = 0
                year_value = 0
                
                for i in range(n_blocks):
                    if x[i, t].varValue and x[i, t].varValue > 0.5:
                        year_blocks.append(self.blocks[i].block_id)
                        year_tonnage += self.blocks[i].tonnage
                        year_value += self.blocks[i].tonnage * self._estimate_block_value_per_tonne(self.blocks[i])
                
                mining_schedule.append({
                    'year': t + 1,
                    'blocks': year_blocks,
                    'tonnage': year_tonnage,
                    'value': year_value
                })
            
            return {
                'status': 'optimal',
                'objective_value': prob.objective.value(),
                'mining_schedule': mining_schedule
            }
        else:
            warnings.warn(f"Optimization failed: {LpStatus[prob.status]}")
            return self.optimize_sequence_heuristic(mine_life)
    
    def optimize_sequence_heuristic(self, mine_life: int = 10) -> Dict:
        """Optimize mining sequence using greedy heuristic"""
        
        # Calculate value per tonne for each block
        block_values = [(i, self._estimate_block_value_per_tonne(block)) 
                       for i, block in enumerate(self.blocks)]
        
        # Sort by value per tonne (descending)
        block_values.sort(key=lambda x: x[1], reverse=True)
        
        mining_schedule = []
        mined_blocks = set()
        
        for year in range(mine_life):
            year_blocks = []
            year_tonnage = 0
            year_value = 0
            
            for block_idx, value_per_tonne in block_values:
                block = self.blocks[block_idx]
                
                # Check if already mined
                if block.block_id in mined_blocks:
                    continue
                
                # Check precedence constraints
                can_mine = True
                for constraint_id in block.precedence_constraints:
                    if constraint_id not in mined_blocks:
                        can_mine = False
                        break
                
                if not can_mine:
                    continue
                
                # Check capacity constraint
                if year_tonnage + block.tonnage > self.processing_capacity:
                    continue
                
                # Mine this block
                year_blocks.append(block.block_id)
                mined_blocks.add(block.block_id)
                year_tonnage += block.tonnage
                year_value += block.tonnage * value_per_tonne
                
                if year_tonnage >= self.processing_capacity * 0.95:  # 95% capacity utilization
                    break
            
            mining_schedule.append({
                'year': year + 1,
                'blocks': year_blocks,
                'tonnage': year_tonnage,
                'value': year_value
            })
        
        total_value = sum(year['value'] / ((1 + self.params.discount_rate/100)**year['year']) 
                         for year in mining_schedule)
        
        return {
            'status': 'heuristic',
            'objective_value': total_value,
            'mining_schedule': mining_schedule
        }
    
    def _estimate_block_value_per_tonne(self, block: MiningBlock) -> float:
        """Estimate value per tonne for a block"""
        # Simplified calculation
        revenue_per_tonne = 0
        for element, grade in block.grades.items():
            if element in self.params.commodity_prices:
                price = self.params.commodity_prices[element]
                metal_content = grade / 100  # Convert % to decimal
                recovery = 0.85  # Assumed recovery
                revenue_per_tonne += metal_content * recovery * price
        
        cost_per_tonne = block.mining_cost + self.params.operating_costs.get('processing', 20)
        
        return max(0, revenue_per_tonne - cost_per_tonne)

class ProcessingRouteSelector:
    """Select optimal processing route for different ore types"""
    
    def __init__(self, processing_routes: List[ProcessingRoute], economic_params: EconomicParameters):
        self.routes = processing_routes
        self.params = economic_params
    
    def evaluate_processing_options(self, ore_composition: Dict[str, float],
                                  tonnage: float) -> Dict:
        """Evaluate processing options for given ore composition"""
        
        route_evaluations = []
        
        for route in self.routes:
            # Calculate revenue
            revenue = 0
            for element, grade in ore_composition.items():
                if element in self.params.commodity_prices:
                    price = self.params.commodity_prices[element]
                    recovery = route.recovery_rates.get(element, 0)
                    metal_content = tonnage * grade / 100
                    recovered_metal = metal_content * recovery
                    revenue += recovered_metal * price
            
            # Calculate costs
            operating_cost = tonnage * route.operating_cost
            capital_cost_annualized = route.capital_cost * (
                self.params.discount_rate / 100 / 
                (1 - (1 + self.params.discount_rate / 100)**(-10))
            )  # Annualized over 10 years
            
            total_cost = operating_cost + capital_cost_annualized
            
            # Net value
            net_value = revenue - total_cost
            
            # Calculate utilization factor
            utilization = min(1.0, tonnage / route.capacity)
            adjusted_net_value = net_value * utilization
            
            route_evaluations.append({
                'route_name': route.route_name,
                'revenue': revenue,
                'operating_cost': operating_cost,
                'capital_cost_annualized': capital_cost_annualized,
                'net_value': adjusted_net_value,
                'utilization': utilization,
                'environmental_impact': route.environmental_impact
            })
        
        # Sort by net value
        route_evaluations.sort(key=lambda x: x['net_value'], reverse=True)
        
        return {
            'ore_composition': ore_composition,
            'tonnage': tonnage,
            'route_evaluations': route_evaluations,
            'recommended_route': route_evaluations[0]['route_name'] if route_evaluations else None
        }
    
    def optimize_processing_mix(self, ore_types: List[Dict], total_capacity: float) -> Dict:
        """Optimize processing route allocation across ore types"""
        
        # This is a simplified allocation optimization
        # In practice, this would use linear programming
        
        allocations = []
        remaining_capacity = total_capacity
        
        # Calculate value per tonne for each ore type and route combination
        ore_route_values = []
        
        for i, ore_type in enumerate(ore_types):
            ore_composition = ore_type['composition']
            available_tonnage = ore_type['tonnage']
            
            for route in self.routes:
                if any(ore_type_name in route.applicable_ore_types 
                      for ore_type_name in ore_type.get('types', ['standard'])):
                    
                    evaluation = self.evaluate_processing_options(ore_composition, 1.0)  # Per tonne
                    route_eval = next(r for r in evaluation['route_evaluations'] 
                                    if r['route_name'] == route.route_name)
                    
                    value_per_tonne = route_eval['net_value']
                    
                    ore_route_values.append({
                        'ore_index': i,
                        'route_name': route.route_name,
                        'value_per_tonne': value_per_tonne,
                        'available_tonnage': available_tonnage,
                        'route_capacity': route.capacity
                    })
        
        # Sort by value per tonne
        ore_route_values.sort(key=lambda x: x['value_per_tonne'], reverse=True)
        
        # Greedy allocation
        for allocation in ore_route_values:
            if remaining_capacity <= 0:
                break
            
            ore_idx = allocation['ore_index']
            route_name = allocation['route_name']
            
            # Determine allocation amount
            max_by_capacity = min(remaining_capacity, allocation['route_capacity'])
            max_by_ore = allocation['available_tonnage']
            
            allocation_amount = min(max_by_capacity, max_by_ore)
            
            if allocation_amount > 0:
                allocations.append({
                    'ore_type_index': ore_idx,
                    'route_name': route_name,
                    'tonnage': allocation_amount,
                    'value_per_tonne': allocation['value_per_tonne']
                })
                
                remaining_capacity -= allocation_amount
                allocation['available_tonnage'] -= allocation_amount
        
        total_value = sum(alloc['tonnage'] * alloc['value_per_tonne'] for alloc in allocations)
        
        return {
            'allocations': allocations,
            'total_value': total_value,
            'capacity_utilization': (total_capacity - remaining_capacity) / total_capacity,
            'unutilized_capacity': remaining_capacity
        }

class RiskAssessment:
    """Monte Carlo risk assessment and uncertainty quantification"""
    
    def __init__(self):
        self.risk_factors = {}
        self.correlation_matrix = None
        self.simulation_results = {}
    
    def define_risk_factor(self, factor_name: str, distribution: str, parameters: Dict):
        """Define risk factor with probability distribution"""
        self.risk_factors[factor_name] = {
            'distribution': distribution,
            'parameters': parameters
        }
    
    def set_correlation_matrix(self, correlation_matrix: np.ndarray, factor_names: List[str]):
        """Set correlation matrix between risk factors"""
        if correlation_matrix.shape != (len(factor_names), len(factor_names)):
            raise ValueError("Correlation matrix dimensions don't match number of factors")
        
        self.correlation_matrix = correlation_matrix
        self.factor_names = factor_names
    
    def monte_carlo_simulation(self, base_schedule: List[MineSchedule], 
                              npv_calculator: NPVCalculator, n_simulations: int = 1000) -> Dict:
        """Perform Monte Carlo simulation for risk assessment"""
        
        npv_results = []
        irr_results = []
        payback_results = []
        scenario_details = []
        
        for simulation in range(n_simulations):
            # Generate correlated random samples
            risk_scenario = self._generate_risk_scenario()
            
            # Apply risk scenario to schedule
            modified_schedule = self._apply_risk_scenario(base_schedule, risk_scenario)
            
            # Calculate metrics
            npv_result = npv_calculator.calculate_project_npv(modified_schedule)
            
            npv_results.append(npv_result['npv'])
            irr_results.append(npv_result['irr'])
            payback_results.append(npv_result['payback_period'])
            
            scenario_details.append({
                'simulation': simulation,
                'risk_factors': risk_scenario,
                'npv': npv_result['npv'],
                'irr': npv_result['irr'],
                'payback': npv_result['payback_period']
            })
        
        # Calculate statistics
        npv_stats = self._calculate_statistics(npv_results)
        irr_stats = self._calculate_statistics(irr_results)
        payback_stats = self._calculate_statistics(payback_results)
        
        # Risk metrics
        base_npv = npv_calculator.calculate_project_npv(base_schedule)['npv']
        probability_positive_npv = sum(1 for npv in npv_results if npv > 0) / n_simulations
        value_at_risk_95 = np.percentile(npv_results, 5)  # 5th percentile
        conditional_var_95 = np.mean([npv for npv in npv_results if npv <= value_at_risk_95])
        
        return {
            'n_simulations': n_simulations,
            'base_case_npv': base_npv,
            'npv_statistics': npv_stats,
            'irr_statistics': irr_stats,
            'payback_statistics': payback_stats,
            'risk_metrics': {
                'probability_positive_npv': probability_positive_npv,
                'value_at_risk_95': value_at_risk_95,
                'conditional_var_95': conditional_var_95,
                'volatility': np.std(npv_results)
            },
            'scenario_details': scenario_details[:100] if len(scenario_details) > 100 else scenario_details  # Limit output
        }
    
    def _generate_risk_scenario(self) -> Dict[str, float]:
        """Generate single risk scenario from defined distributions"""
        scenario = {}
        
        # Generate independent samples
        independent_samples = []
        factor_names = []
        
        for factor_name, factor_config in self.risk_factors.items():
            distribution = factor_config['distribution']
            params = factor_config['parameters']
            
            if distribution == 'normal':
                sample = np.random.normal(params['mean'], params['std'])
            elif distribution == 'lognormal':
                sample = np.random.lognormal(params['mean'], params['sigma'])
            elif distribution == 'uniform':
                sample = np.random.uniform(params['low'], params['high'])
            elif distribution == 'triangular':
                sample = np.random.triangular(params['left'], params['mode'], params['right'])
            else:
                sample = params.get('mean', 1.0)  # Default
            
            independent_samples.append(sample)
            factor_names.append(factor_name)
        
        # Apply correlation if specified
        if self.correlation_matrix is not None and len(independent_samples) > 1:
            # Use Cholesky decomposition for correlation
            L = np.linalg.cholesky(self.correlation_matrix)
            independent_array = np.array(independent_samples)
            
            # Convert to standard normal
            uniform_samples = [norm.cdf(sample) for sample in independent_array]
            standard_normal = [norm.ppf(u) for u in uniform_samples]
            
            # Apply correlation
            correlated_normal = L @ standard_normal
            
            # Convert back to original distributions
            for i, (factor_name, factor_config) in enumerate(self.risk_factors.items()):
                correlated_uniform = norm.cdf(correlated_normal[i])
                
                distribution = factor_config['distribution']
                params = factor_config['parameters']
                
                if distribution == 'normal':
                    scenario[factor_name] = norm.ppf(correlated_uniform, params['mean'], params['std'])
                elif distribution == 'lognormal':
                    scenario[factor_name] = lognorm.ppf(correlated_uniform, s=params['sigma'], scale=np.exp(params['mean']))
                elif distribution == 'uniform':
                    scenario[factor_name] = uniform.ppf(correlated_uniform, params['low'], params['high'])
                else:
                    scenario[factor_name] = correlated_normal[i]
        else:
            # No correlation - use independent samples
            for factor_name, sample in zip(factor_names, independent_samples):
                scenario[factor_name] = sample
        
        return scenario
    
    def _apply_risk_scenario(self, base_schedule: List[MineSchedule], 
                           risk_scenario: Dict[str, float]) -> List[MineSchedule]:
        """Apply risk scenario to base schedule"""
        modified_schedule = []
        
        for year_data in base_schedule:
            new_year_data = MineSchedule(
                year=year_data.year,
                blocks_mined=year_data.blocks_mined.copy(),
                total_tonnage=year_data.total_tonnage,
                metal_production=year_data.metal_production.copy(),
                operating_costs=year_data.operating_costs,
                capital_costs=year_data.capital_costs,
                revenue=year_data.revenue,
                cash_flow=year_data.cash_flow
            )
            
            # Apply risk factors
            for factor_name, factor_value in risk_scenario.items():
                if 'commodity_price' in factor_name:
                    new_year_data.revenue *= factor_value
                elif 'operating_cost' in factor_name:
                    new_year_data.operating_costs *= factor_value
                elif 'capital_cost' in factor_name:
                    new_year_data.capital_costs *= factor_value
                elif 'tonnage' in factor_name:
                    new_year_data.total_tonnage *= factor_value
                    # Adjust metal production proportionally
                    for element in new_year_data.metal_production:
                        new_year_data.metal_production[element] *= factor_value
            
            # Recalculate cash flow
            new_year_data.cash_flow = (new_year_data.revenue - 
                                     new_year_data.operating_costs - 
                                     new_year_data.capital_costs)
            
            modified_schedule.append(new_year_data)
        
        return modified_schedule
    
    def _calculate_statistics(self, data: List[float]) -> Dict:
        """Calculate statistical measures"""
        data_array = np.array(data)
        
        return {
            'mean': np.mean(data_array),
            'median': np.median(data_array),
            'std': np.std(data_array),
            'min': np.min(data_array),
            'max': np.max(data_array),
            'percentiles': {
                'p5': np.percentile(data_array, 5),
                'p10': np.percentile(data_array, 10),
                'p25': np.percentile(data_array, 25),
                'p75': np.percentile(data_array, 75),
                'p90': np.percentile(data_array, 90),
                'p95': np.percentile(data_array, 95)
            }
        }

class EconomicOptimizer:
    """Main economic optimization interface"""
    
    def __init__(self):
        self.npv_calculator = None
        self.sequence_optimizer = None
        self.route_selector = None
        self.risk_assessor = RiskAssessment()
        self.optimization_results = {}
    
    def initialize_economic_parameters(self, params: EconomicParameters):
        """Initialize economic parameters"""
        self.npv_calculator = NPVCalculator(params)
    
    def comprehensive_optimization(self, 
                                 blocks: List[MiningBlock],
                                 processing_routes: List[ProcessingRoute],
                                 economic_params: EconomicParameters,
                                 optimization_config: Dict = None) -> Dict:
        """Perform comprehensive economic optimization"""
        
        if optimization_config is None:
            optimization_config = {
                'mine_life': 10,
                'processing_capacity': 1e6,
                'include_risk_assessment': True,
                'monte_carlo_simulations': 1000
            }
        
        # Initialize components
        self.npv_calculator = NPVCalculator(economic_params)
        self.sequence_optimizer = MiningSequenceOptimizer(blocks, economic_params)
        self.route_selector = ProcessingRouteSelector(processing_routes, economic_params)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'optimization_config': optimization_config,
            'economic_parameters': {
                'discount_rate': economic_params.discount_rate,
                'commodity_prices': economic_params.commodity_prices,
                'tax_rate': economic_params.tax_rate
            }
        }
        
        # 1. Optimize mining sequence
        print("Optimizing mining sequence...")
        sequence_result = self.sequence_optimizer.optimize_sequence_milp(
            optimization_config['mine_life']
        )
        results['mining_sequence'] = sequence_result
        
        # 2. Convert to detailed schedule
        detailed_schedule = self._create_detailed_schedule(
            sequence_result, blocks, processing_routes[0]  # Use first route as default
        )
        results['detailed_schedule'] = detailed_schedule
        
        # 3. Calculate base case NPV
        npv_result = self.npv_calculator.calculate_project_npv(detailed_schedule)
        results['base_case_npv'] = npv_result
        
        # 4. Sensitivity analysis
        print("Performing sensitivity analysis...")
        sensitivity_factors = {
            'commodity_prices': [0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
            'operating_costs': [0.8, 0.9, 1.0, 1.1, 1.2, 1.4],
            'capital_costs': [0.7, 0.85, 1.0, 1.15, 1.3, 1.5]
        }
        
        sensitivity_result = self.npv_calculator.sensitivity_analysis(
            detailed_schedule, sensitivity_factors
        )
        results['sensitivity_analysis'] = sensitivity_result
        
        # 5. Processing route optimization
        print("Optimizing processing routes...")
        ore_types = self._extract_ore_types(blocks)
        processing_optimization = self.route_selector.optimize_processing_mix(
            ore_types, optimization_config['processing_capacity']
        )
        results['processing_optimization'] = processing_optimization
        
        # 6. Risk assessment
        if optimization_config['include_risk_assessment']:
            print("Performing risk assessment...")
            
            # Define risk factors
            self.risk_assessor.define_risk_factor('commodity_price_factor', 'lognormal', 
                                                {'mean': 0, 'sigma': 0.2})
            self.risk_assessor.define_risk_factor('operating_cost_factor', 'normal', 
                                                {'mean': 1.0, 'std': 0.15})
            self.risk_assessor.define_risk_factor('tonnage_factor', 'normal', 
                                                {'mean': 1.0, 'std': 0.1})
            
            risk_result = self.risk_assessor.monte_carlo_simulation(
                detailed_schedule, self.npv_calculator, 
                optimization_config['monte_carlo_simulations']
            )
            results['risk_assessment'] = risk_result
        
        # 7. Calculate optimization summary
        results['optimization_summary'] = self._create_optimization_summary(results)
        
        return results
    
    def _create_detailed_schedule(self, sequence_result: Dict, blocks: List[MiningBlock],
                                 processing_route: ProcessingRoute) -> List[MineSchedule]:
        """Create detailed mine schedule from sequence optimization"""
        schedule = []
        
        block_dict = {block.block_id: block for block in blocks}
        
        for year_data in sequence_result.get('mining_schedule', []):
            year = year_data['year']
            year_blocks = year_data['blocks']
            
            total_tonnage = 0
            metal_production = {}
            operating_costs = 0
            capital_costs = 0
            revenue = 0
            
            for block_id in year_blocks:
                if block_id in block_dict:
                    block = block_dict[block_id]
                    
                    total_tonnage += block.tonnage
                    operating_costs += block.tonnage * (block.mining_cost + block.processing_cost)
                    
                    # Calculate metal production and revenue
                    for element, grade in block.grades.items():
                        metal_content = block.tonnage * grade / 100
                        recovery = processing_route.recovery_rates.get(element, 0.85)
                        recovered_metal = metal_content * recovery
                        
                        if element not in metal_production:
                            metal_production[element] = 0
                        metal_production[element] += recovered_metal
                        
                        if element in self.npv_calculator.params.commodity_prices:
                            price = self.npv_calculator.params.commodity_prices[element]
                            revenue += recovered_metal * price
            
            # Add processing route capital costs (annualized)
            if year == 1:  # First year
                capital_costs = processing_route.capital_cost
            
            cash_flow = revenue - operating_costs - capital_costs
            
            schedule.append(MineSchedule(
                year=year,
                blocks_mined=year_blocks,
                total_tonnage=total_tonnage,
                metal_production=metal_production,
                operating_costs=operating_costs,
                capital_costs=capital_costs,
                revenue=revenue,
                cash_flow=cash_flow
            ))
        
        return schedule
    
    def _extract_ore_types(self, blocks: List[MiningBlock]) -> List[Dict]:
        """Extract ore types from mining blocks"""
        ore_types = []
        
        # Group blocks by rock type
        rock_type_groups = {}
        for block in blocks:
            if block.rock_type not in rock_type_groups:
                rock_type_groups[block.rock_type] = []
            rock_type_groups[block.rock_type].append(block)
        
        # Create ore type definitions
        for rock_type, type_blocks in rock_type_groups.items():
            total_tonnage = sum(block.tonnage for block in type_blocks)
            
            # Average composition
            avg_composition = {}
            for element in set().union(*(block.grades.keys() for block in type_blocks)):
                total_metal = sum(block.tonnage * block.grades.get(element, 0) 
                                for block in type_blocks)
                avg_composition[element] = total_metal / total_tonnage if total_tonnage > 0 else 0
            
            ore_types.append({
                'name': rock_type,
                'composition': avg_composition,
                'tonnage': total_tonnage,
                'types': [rock_type]
            })
        
        return ore_types
    
    def _create_optimization_summary(self, results: Dict) -> Dict:
        """Create optimization summary"""
        summary = {}
        
        # Base case metrics
        if 'base_case_npv' in results:
            base_npv = results['base_case_npv']
            summary['base_case'] = {
                'npv': base_npv['npv'],
                'irr': base_npv['irr'],
                'payback_period': base_npv['payback_period'],
                'total_revenue': base_npv['total_revenue'],
                'total_costs': base_npv['total_operating_costs'] + base_npv['total_capital_costs']
            }
        
        # Risk metrics
        if 'risk_assessment' in results:
            risk_data = results['risk_assessment']
            summary['risk_profile'] = {
                'probability_positive_npv': risk_data['risk_metrics']['probability_positive_npv'],
                'value_at_risk_95': risk_data['risk_metrics']['value_at_risk_95'],
                'npv_volatility': risk_data['risk_metrics']['volatility']
            }
        
        # Sensitivity insights
        if 'sensitivity_analysis' in results:
            sensitivity = results['sensitivity_analysis']['sensitivity_analysis']
            most_sensitive_factor = max(sensitivity.keys(), 
                key=lambda k: max(abs(item['npv_change_percent']) for item in sensitivity[k]))
            summary['most_sensitive_factor'] = most_sensitive_factor
        
        # Processing optimization
        if 'processing_optimization' in results:
            proc_opt = results['processing_optimization']
            summary['processing'] = {
                'capacity_utilization': proc_opt['capacity_utilization'],
                'total_processing_value': proc_opt['total_value']
            }
        
        return summary
    
    def export_results(self, results: Dict, filename: str):
        """Export optimization results"""
        if filename.endswith('.json'):
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        
        elif filename.endswith('.csv'):
            # Export schedule summary
            if 'detailed_schedule' in results:
                schedule_data = []
                for year_data in results['detailed_schedule']:
                    schedule_data.append({
                        'year': year_data.year,
                        'tonnage': year_data.total_tonnage,
                        'revenue': year_data.revenue,
                        'operating_costs': year_data.operating_costs,
                        'capital_costs': year_data.capital_costs,
                        'cash_flow': year_data.cash_flow
                    })
                
                df = pd.DataFrame(schedule_data)
                df.to_csv(filename, index=False)
        
        else:
            raise ValueError("Unsupported file format. Use .json or .csv")

# Example usage and testing
if __name__ == "__main__":
    # Create test data
    print("Testing Economic Optimizer:")
    
    # Economic parameters
    economic_params = EconomicParameters(
        discount_rate=8.0,
        commodity_prices={'Cu': 8000, 'Au': 60000, 'Zn': 2500},  # USD per tonne
        operating_costs={'mining': 15, 'processing': 25},  # USD per tonne
        capital_costs={'plant': 50e6, 'infrastructure': 20e6},  # USD
        tax_rate=0.30,
        royalty_rate=0.05,
        inflation_rate=0.02
    )
    
    # Test mining blocks
    test_blocks = [
        MiningBlock(
            block_id='B001', coordinates=(0, 0, 0), tonnage=100000,
            grades={'Cu': 1.5, 'Au': 2.0, 'Zn': 0.5}, rock_type='ore',
            mining_cost=12, processing_cost=20, precedence_constraints=[],
            geotechnical_risk=0.1
        ),
        MiningBlock(
            block_id='B002', coordinates=(50, 0, 0), tonnage=150000,
            grades={'Cu': 2.1, 'Au': 3.2, 'Zn': 0.8}, rock_type='ore',
            mining_cost=15, processing_cost=25, precedence_constraints=['B001'],
            geotechnical_risk=0.15
        ),
        MiningBlock(
            block_id='B003', coordinates=(0, 50, 0), tonnage=80000,
            grades={'Cu': 0.8, 'Au': 1.1, 'Zn': 0.3}, rock_type='waste',
            mining_cost=10, processing_cost=15, precedence_constraints=[],
            geotechnical_risk=0.05
        )
    ]
    
    # Test processing routes
    test_routes = [
        ProcessingRoute(
            route_name='flotation', applicable_ore_types=['ore'],
            recovery_rates={'Cu': 0.88, 'Au': 0.75, 'Zn': 0.85},
            operating_cost=25, capital_cost=30e6, capacity=500000,
            environmental_impact=0.6
        ),
        ProcessingRoute(
            route_name='heap_leach', applicable_ore_types=['ore', 'waste'],
            recovery_rates={'Cu': 0.65, 'Au': 0.85, 'Zn': 0.20},
            operating_cost=18, capital_cost=15e6, capacity=1000000,
            environmental_impact=0.8
        )
    ]
    
    # Initialize optimizer
    optimizer = EconomicOptimizer()
    
    # Test NPV calculation
    print("\n1. Testing NPV calculation:")
    optimizer.initialize_economic_parameters(economic_params)
    
    block_value = optimizer.npv_calculator.calculate_block_value(test_blocks[0], test_routes[0])
    print(f"Block B001 value: ${block_value:,.0f}")
    
    # Test mining sequence optimization
    print("\n2. Testing mining sequence optimization:")
    sequence_optimizer = MiningSequenceOptimizer(test_blocks, economic_params)
    sequence_result = sequence_optimizer.optimize_sequence_heuristic(5)
    print(f"Sequence optimization status: {sequence_result['status']}")
    print(f"Objective value: ${sequence_result['objective_value']:,.0f}")
    
    # Test processing route selection
    print("\n3. Testing processing route selection:")
    route_selector = ProcessingRouteSelector(test_routes, economic_params)
    
    test_ore = {'Cu': 1.8, 'Au': 2.5, 'Zn': 0.6}
    route_evaluation = route_selector.evaluate_processing_options(test_ore, 100000)
    print(f"Recommended route: {route_evaluation['recommended_route']}")
    
    # Test risk assessment
    print("\n4. Testing risk assessment:")
    risk_assessor = RiskAssessment()
    risk_assessor.define_risk_factor('price_factor', 'normal', {'mean': 1.0, 'std': 0.2})
    
    # Create dummy schedule for risk testing
    dummy_schedule = [
        MineSchedule(
            year=1, blocks_mined=['B001'], total_tonnage=100000,
            metal_production={'Cu': 1320, 'Au': 150}, 
            operating_costs=3500000, capital_costs=50000000,
            revenue=15600000, cash_flow=-37900000
        ),
        MineSchedule(
            year=2, blocks_mined=['B002'], total_tonnage=150000,
            metal_production={'Cu': 2772, 'Au': 360},
            operating_costs=6000000, capital_costs=0,
            revenue=30432000, cash_flow=24432000
        )
    ]
    
    try:
        risk_result = risk_assessor.monte_carlo_simulation(
            dummy_schedule, optimizer.npv_calculator, 100  # Small sample for demo
        )
        print(f"Risk assessment completed: {risk_result['n_simulations']} simulations")
        print(f"Probability of positive NPV: {risk_result['risk_metrics']['probability_positive_npv']:.1%}")
    except Exception as e:
        print(f"Risk assessment demo failed (expected): {e}")
    
    print("\nEconomic Optimizer initialized successfully!")
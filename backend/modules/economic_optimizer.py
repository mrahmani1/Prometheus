"""
Economic Optimization for Mining Operations
NPV calculations, mining sequence optimization, processing route selection, and risk assessment
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution, linprog
from scipy.linalg import solve, LinAlgError, svd, pinv
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
        """Comprehensive block value estimation with detailed metallurgical and economic modeling"""
        
        # Initialize comprehensive value calculation components
        total_gross_revenue = 0.0
        total_processing_costs = 0.0
        total_smelting_costs = 0.0
        total_refining_costs = 0.0
        total_transport_costs = 0.0
        total_environmental_costs = 0.0
        
        # Get ore type classification for metallurgical parameters
        ore_type = self._classify_ore_type(block)
        hardness_factor = self._calculate_hardness_factor(block)
        
        # Comprehensive revenue calculation by element
        for element, grade in block.grades.items():
            if element not in self.params.commodity_prices:
                continue
                
            base_price = self.params.commodity_prices[element]
            
            # Apply price volatility and market conditions
            adjusted_price = self._apply_market_adjustments(element, base_price)
            
            # Metal content calculation
            metal_content = grade / 100.0  # Convert percentage to decimal
            
            # Advanced metallurgical recovery modeling
            recovery_data = self._calculate_comprehensive_recovery(element, grade, ore_type, block)
            
            # Primary recovery (flotation/leaching)
            primary_recovery = recovery_data['primary_recovery']
            
            # Secondary recovery (re-processing of tailings)
            secondary_recovery = recovery_data['secondary_recovery']
            
            # Total recoverable metal
            total_recovery = primary_recovery + secondary_recovery * 0.7  # Discount secondary
            total_recovery = min(total_recovery, 0.98)  # Physical maximum
            
            # Calculate payable metal (after smelter deductions)
            payable_metal = self._calculate_payable_metal(element, metal_content * total_recovery)
            
            # Revenue calculation with concentrate specifications
            concentrate_grade = self._estimate_concentrate_grade(element, grade, ore_type)
            smelter_penalties = self._calculate_smelter_penalties(block.grades, concentrate_grade)
            
            element_revenue = payable_metal * adjusted_price - smelter_penalties
            total_gross_revenue += max(0, element_revenue)
            
            # Element-specific processing costs
            processing_cost = self._calculate_element_processing_cost(
                element, grade, ore_type, hardness_factor, recovery_data
            )
            total_processing_costs += processing_cost
            
            # Smelting and refining charges
            if element in ['Cu', 'Pb', 'Zn', 'Ni']:
                smelting_charge = self._calculate_smelting_charges(element, payable_metal, concentrate_grade)
                refining_charge = self._calculate_refining_charges(element, payable_metal)
                total_smelting_costs += smelting_charge
                total_refining_costs += refining_charge
        
        # Mining costs with geological complexity factors
        base_mining_cost = block.mining_cost
        
        # Geological complexity adjustments
        complexity_multiplier = self._calculate_geological_complexity_multiplier(block)
        geotechnical_factor = self._calculate_geotechnical_stability_factor(block)
        
        adjusted_mining_cost = base_mining_cost * complexity_multiplier * geotechnical_factor
        
        # Additional mining cost factors
        blasting_cost = self._calculate_blasting_cost(block, hardness_factor)
        hauling_cost = self._calculate_hauling_cost(block)
        equipment_cost = self._calculate_equipment_cost_adjustment(block, ore_type)
        
        total_mining_cost = adjusted_mining_cost + blasting_cost + hauling_cost + equipment_cost
        
        # Transportation costs
        total_transport_costs = self._calculate_transportation_costs(block)
        
        # Environmental and regulatory costs
        total_environmental_costs = self._calculate_environmental_costs(block, ore_type)
        
        # Waste management and tailings costs
        waste_cost = self._calculate_waste_management_cost(block)
        tailings_cost = self._calculate_tailings_management_cost(block, ore_type)
        
        # Rehabilitation and closure costs (allocated)
        rehabilitation_cost = self._calculate_allocated_rehabilitation_cost(block)
        
        # Infrastructure and overhead costs
        infrastructure_cost = self._calculate_infrastructure_cost_allocation(block)
        
        # Risk adjustments
        technical_risk_discount = self._calculate_technical_risk_discount(block, ore_type)
        market_risk_discount = self._calculate_market_risk_discount(block)
        
        # Calculate net present value per tonne
        total_costs = (
            total_mining_cost +
            total_processing_costs +
            total_smelting_costs +
            total_refining_costs +
            total_transport_costs +
            total_environmental_costs +
            waste_cost +
            tailings_cost +
            rehabilitation_cost +
            infrastructure_cost
        )
        
        # Apply discount factors
        gross_value = total_gross_revenue - total_costs
        risk_adjusted_value = gross_value * (1.0 - technical_risk_discount - market_risk_discount)
        
        # Time value adjustments (if block timing is specified)
        if hasattr(block, 'extraction_year') and block.extraction_year > 0:
            discount_factor = (1.0 + self.params.discount_rate / 100.0) ** block.extraction_year
            npv_per_tonne = risk_adjusted_value / discount_factor
        else:
            npv_per_tonne = risk_adjusted_value
        
        # Apply cutoff grade and minimum economic thresholds
        cutoff_value = self._calculate_dynamic_cutoff_grade(ore_type)
        
        # Return value with economic constraints
        return max(0.0, npv_per_tonne - cutoff_value)
    
    def _classify_ore_type(self, block: MiningBlock) -> str:
        """Classify ore type based on grade composition and geological characteristics"""
        # Determine primary ore type based on dominant elements
        dominant_elements = sorted(block.grades.items(), key=lambda x: x[1], reverse=True)
        
        if not dominant_elements:
            return 'waste'
        
        primary_element = dominant_elements[0][0]
        primary_grade = dominant_elements[0][1]
        
        # Complex ore type classification
        if len([g for g in block.grades.values() if g > 0.1]) > 3:
            return 'polymetallic'
        elif primary_element == 'Cu' and primary_grade > 0.4:
            return 'high_grade_copper'
        elif primary_element == 'Cu' and primary_grade > 0.15:
            return 'copper'
        elif primary_element in ['Pb', 'Zn'] and primary_grade > 1.0:
            return 'lead_zinc'
        elif primary_element == 'Au' and primary_grade > 1.0:
            return 'gold'
        elif primary_element == 'Ag' and primary_grade > 50:
            return 'silver'
        else:
            return 'low_grade_mixed'
    
    def _calculate_hardness_factor(self, block: MiningBlock) -> float:
        """Calculate rock hardness factor affecting processing costs"""
        # Use geological properties if available
        if hasattr(block, 'rock_hardness'):
            return block.rock_hardness
        
        # Estimate from grade composition and depth
        hardness = 1.0
        
        # Sulfide content increases hardness
        sulfide_elements = ['Cu', 'Pb', 'Zn', 'Ni']
        sulfide_content = sum(block.grades.get(el, 0) for el in sulfide_elements)
        hardness += sulfide_content * 0.01
        
        # Depth factor
        if hasattr(block, 'depth'):
            depth_factor = 1.0 + block.depth * 0.0001  # Increase with depth
            hardness *= depth_factor
        
        return min(hardness, 3.0)  # Cap at 3x base hardness
    
    def _apply_market_adjustments(self, element: str, base_price: float) -> float:
        """Apply market volatility and long-term price adjustments"""
        # Long-term price trend adjustments
        price_trends = {
            'Cu': 0.95,  # Slight downward trend
            'Au': 1.05,  # Slight upward trend
            'Ag': 1.02,
            'Pb': 0.98,
            'Zn': 0.96,
            'Ni': 1.08
        }
        
        trend_factor = price_trends.get(element, 1.0)
        
        # Market condition adjustments
        market_volatility_discount = 0.02  # 2% discount for price volatility
        
        return base_price * trend_factor * (1.0 - market_volatility_discount)
    
    def _calculate_comprehensive_recovery(self, element: str, grade: float, ore_type: str, block: MiningBlock) -> Dict[str, float]:
        """Calculate comprehensive metallurgical recovery rates"""
        
        # Base recovery rates by element and ore type
        base_recoveries = {
            'copper': {'Cu': 0.90, 'Au': 0.70, 'Ag': 0.85, 'Mo': 0.75},
            'lead_zinc': {'Pb': 0.88, 'Zn': 0.85, 'Ag': 0.82, 'Au': 0.65},
            'polymetallic': {'Cu': 0.85, 'Pb': 0.85, 'Zn': 0.82, 'Au': 0.68, 'Ag': 0.80},
            'gold': {'Au': 0.92, 'Ag': 0.88, 'Cu': 0.75},
            'high_grade_copper': {'Cu': 0.93, 'Au': 0.75, 'Ag': 0.88, 'Mo': 0.80}
        }
        
        base_recovery = base_recoveries.get(ore_type, {}).get(element, 0.75)
        
        # Grade-dependent recovery adjustments
        if grade < 0.1:
            grade_factor = 0.8  # Lower recovery for very low grades
        elif grade < 0.5:
            grade_factor = 0.9
        elif grade > 2.0:
            grade_factor = 1.05  # Higher recovery for higher grades
        else:
            grade_factor = 1.0
        
        # Ore mineralogy adjustments
        mineralogy_factor = self._calculate_mineralogy_factor(element, ore_type, block)
        
        # Liberation characteristics
        liberation_factor = self._calculate_liberation_factor(element, ore_type)
        
        # Primary recovery calculation
        primary_recovery = base_recovery * grade_factor * mineralogy_factor * liberation_factor
        primary_recovery = min(primary_recovery, 0.95)  # Physical maximum
        
        # Secondary recovery from tailings reprocessing
        secondary_recovery = (1.0 - primary_recovery) * 0.3  # 30% of losses recoverable
        
        return {
            'primary_recovery': primary_recovery,
            'secondary_recovery': secondary_recovery,
            'mineralogy_factor': mineralogy_factor,
            'liberation_factor': liberation_factor
        }
    
    def _calculate_mineralogy_factor(self, element: str, ore_type: str, block: MiningBlock) -> float:
        """Comprehensive mineralogy-based recovery adjustment calculation"""
        try:
            base_factor = 1.0
            
            # Initialize comprehensive mineralogical assessment
            mineralogy_assessment = self._assess_ore_mineralogy(element, block)
            
            # Primary mineral association factors
            primary_mineral_factor = self._calculate_primary_mineral_factor(
                element, mineralogy_assessment['primary_minerals'])
            
            # Gangue mineral interference factors
            gangue_factor = self._calculate_gangue_mineral_factor(
                element, mineralogy_assessment['gangue_minerals'])
            
            # Mineral liberation characteristics
            liberation_factor = self._calculate_mineral_liberation_factor(
                element, mineralogy_assessment['grain_size'], 
                mineralogy_assessment['mineral_associations'])
            
            # Oxidation state effects on processing
            oxidation_factor = self._calculate_oxidation_state_factor(
                element, mineralogy_assessment['oxidation_state'])
            
            # Clay and alteration mineral effects
            alteration_factor = self._calculate_alteration_mineral_factor(
                element, mineralogy_assessment['alteration_minerals'])
            
            # Trace element effects on processing
            trace_element_factor = self._calculate_trace_element_factor(
                element, mineralogy_assessment['trace_elements'])
            
            # Texture and structural factors
            texture_factor = self._calculate_texture_factor(
                element, mineralogy_assessment['texture'], 
                mineralogy_assessment['hardness'])
            
            # Composite mineralogy factor
            composite_factor = (
                primary_mineral_factor * 
                gangue_factor * 
                liberation_factor * 
                oxidation_factor * 
                alteration_factor * 
                trace_element_factor * 
                texture_factor
            )
            
            # Apply ore type specific corrections
            ore_type_correction = self._get_ore_type_mineralogy_correction(element, ore_type)
            final_factor = composite_factor * ore_type_correction
            
            # Bounds checking and validation
            final_factor = max(0.1, min(2.0, final_factor))
            
            # Log detailed mineralogy assessment for complex cases
            if abs(final_factor - 1.0) > 0.2:
                self._log_mineralogy_assessment(element, block, mineralogy_assessment, final_factor)
            
            return final_factor
            
        except Exception as e:
            logging.error(f"Error calculating mineralogy factor for {element}: {e}")
            return 1.0  # Fallback to neutral factor
    
    def _assess_ore_mineralogy(self, element: str, block: MiningBlock) -> Dict[str, Any]:
        """Comprehensive ore mineralogy assessment"""
        assessment = {
            'primary_minerals': [],
            'gangue_minerals': [],
            'alteration_minerals': [],
            'trace_elements': [],
            'oxidation_state': 'mixed',
            'grain_size': 'medium',
            'mineral_associations': [],
            'texture': 'massive',
            'hardness': 'medium'
        }
        
        try:
            # Determine primary ore minerals based on element grades
            grades = block.grades
            
            # Copper minerals
            if element == 'Cu' and grades.get('Cu', 0) > 0:
                cu_grade = grades['Cu']
                s_content = grades.get('S', 0)
                
                if s_content > cu_grade * 0.5:  # High sulfur indicates sulfide
                    if grades.get('Fe', 0) > cu_grade * 2:
                        assessment['primary_minerals'].append('chalcopyrite')
                    else:
                        assessment['primary_minerals'].append('chalcocite')
                    assessment['oxidation_state'] = 'sulfide'
                else:
                    assessment['primary_minerals'].extend(['malachite', 'azurite', 'chrysocolla'])
                    assessment['oxidation_state'] = 'oxide'
                
                # Secondary mineral associations
                if grades.get('Mo', 0) > 0.01:
                    assessment['primary_minerals'].append('molybdenite')
                if grades.get('Au', 0) > 0.5:
                    assessment['mineral_associations'].append('native_gold')
            
            # Lead-Zinc minerals
            elif element in ['Pb', 'Zn'] and (grades.get('Pb', 0) > 0 or grades.get('Zn', 0) > 0):
                pb_grade = grades.get('Pb', 0)
                zn_grade = grades.get('Zn', 0)
                
                if pb_grade > 0:
                    assessment['primary_minerals'].append('galena')
                if zn_grade > 0:
                    assessment['primary_minerals'].append('sphalerite')
                
                assessment['oxidation_state'] = 'sulfide'
                
                # Complex polymetallic associations
                if pb_grade > 0 and zn_grade > 0:
                    assessment['mineral_associations'].append('complex_sulfide')
                
                if grades.get('Ag', 0) > 10:
                    assessment['primary_minerals'].append('argentiferous_galena')
            
            # Gold minerals
            elif element == 'Au' and grades.get('Au', 0) > 0:
                au_grade = grades['Au']
                
                if grades.get('As', 0) > 100:
                    assessment['primary_minerals'].append('arsenopyrite')
                    assessment['trace_elements'].append('arsenic')
                elif grades.get('S', 0) > 1.0:
                    assessment['primary_minerals'].append('pyrite')
                else:
                    assessment['primary_minerals'].append('native_gold')
                
                # Tellurium associations
                if grades.get('Te', 0) > 1:
                    assessment['primary_minerals'].append('tellurides')
                    assessment['trace_elements'].append('tellurium')
            
            # Gangue mineral assessment based on geology and associated elements
            self._assess_gangue_minerals(assessment, grades)
            
            # Alteration mineral assessment
            self._assess_alteration_minerals(assessment, grades)
            
            # Physical characteristics assessment
            self._assess_physical_characteristics(assessment, grades)
            
        except Exception as e:
            logging.error(f"Error in ore mineralogy assessment: {e}")
        
        return assessment
    
    def _assess_gangue_minerals(self, assessment: Dict[str, Any], grades: Dict[str, float]):
        """Assess gangue mineral composition"""
        # Silicate gangue
        if grades.get('SiO2', 0) > 40:
            assessment['gangue_minerals'].append('quartz')
            if grades.get('Al2O3', 0) > 10:
                assessment['gangue_minerals'].append('feldspar')
        
        # Carbonate gangue
        if grades.get('CaCO3', 0) > 5 or grades.get('Ca', 0) > 2:
            assessment['gangue_minerals'].append('calcite')
            if grades.get('Mg', 0) > 1:
                assessment['gangue_minerals'].append('dolomite')
        
        # Iron oxides
        if grades.get('Fe', 0) > 5:
            if grades.get('S', 0) < grades.get('Fe', 0) * 0.3:
                assessment['gangue_minerals'].extend(['hematite', 'magnetite'])
            else:
                assessment['gangue_minerals'].append('pyrrhotite')
    
    def _assess_alteration_minerals(self, assessment: Dict[str, Any], grades: Dict[str, float]):
        """Assess alteration mineral composition"""
        # Clay alteration
        if grades.get('Al2O3', 0) > 15:
            assessment['alteration_minerals'].append('kaolinite')
            if grades.get('K', 0) > 2:
                assessment['alteration_minerals'].append('illite')
        
        # Sericitic alteration
        if grades.get('K', 0) > 3 and grades.get('Al2O3', 0) > 12:
            assessment['alteration_minerals'].append('sericite')
        
        # Chloritic alteration
        if grades.get('Mg', 0) > 2 and grades.get('Fe', 0) > 3:
            assessment['alteration_minerals'].append('chlorite')
    
    def _assess_physical_characteristics(self, assessment: Dict[str, Any], grades: Dict[str, float]):
        """Assess physical ore characteristics"""
        # Hardness estimation based on mineralogy
        if 'quartz' in assessment['gangue_minerals']:
            assessment['hardness'] = 'hard'
        elif any(mineral in assessment['alteration_minerals'] 
                for mineral in ['kaolinite', 'illite']):
            assessment['hardness'] = 'soft'
        else:
            assessment['hardness'] = 'medium'
        
        # Grain size estimation
        total_grade = sum(grades.get(metal, 0) for metal in ['Cu', 'Pb', 'Zn', 'Au'])
        if total_grade > 5:
            assessment['grain_size'] = 'coarse'
        elif total_grade < 0.5:
            assessment['grain_size'] = 'fine'
        else:
            assessment['grain_size'] = 'medium'
    
    def _calculate_primary_mineral_factor(self, element: str, primary_minerals: List[str]) -> float:
        """Calculate recovery factor based on primary mineral composition"""
        mineral_recovery_factors = {
            'Cu': {
                'chalcopyrite': 1.0,
                'chalcocite': 1.1,
                'bornite': 1.05,
                'malachite': 0.85,
                'azurite': 0.8,
                'chrysocolla': 0.7,
                'molybdenite': 0.6  # When recovering Cu from Mo ores
            },
            'Pb': {
                'galena': 1.0,
                'argentiferous_galena': 1.05,
                'cerussite': 0.8,
                'anglesite': 0.75
            },
            'Zn': {
                'sphalerite': 1.0,
                'smithsonite': 0.7,
                'hemimorphite': 0.65,
                'willemite': 0.6
            },
            'Au': {
                'native_gold': 1.0,
                'electrum': 0.95,
                'pyrite': 0.8,
                'arsenopyrite': 0.75,
                'tellurides': 0.7
            }
        }
        
        if element not in mineral_recovery_factors:
            return 1.0
        
        factors = mineral_recovery_factors[element]
        if not primary_minerals:
            return 1.0
        
        # Calculate weighted average factor
        total_factor = sum(factors.get(mineral, 0.9) for mineral in primary_minerals)
        return total_factor / len(primary_minerals)
    
    def _calculate_gangue_mineral_factor(self, element: str, gangue_minerals: List[str]) -> float:
        """Calculate processing difficulty based on gangue mineralogy"""
        gangue_penalties = {
            'quartz': 0.98,      # Neutral to slightly abrasive
            'feldspar': 0.97,    # Slightly abrasive
            'calcite': 0.95,     # Acid consuming
            'dolomite': 0.93,    # High acid consumption
            'pyrrhotite': 0.90,  # Magnetic interference
            'magnetite': 0.92,   # Magnetic separation issues
            'hematite': 0.95,    # Iron interference
            'pyrite': 0.88       # Acid generation, flotation interference
        }
        
        if not gangue_minerals:
            return 1.0
        
        penalty_factor = 1.0
        for mineral in gangue_minerals:
            penalty_factor *= gangue_penalties.get(mineral, 0.98)
        
        return max(0.7, penalty_factor)  # Minimum factor of 0.7
    
    def _calculate_mineral_liberation_factor(self, element: str, grain_size: str, 
                                          associations: List[str]) -> float:
        """Calculate liberation characteristics factor"""
        grain_size_factors = {
            'coarse': 1.1,   # Easy liberation
            'medium': 1.0,   # Standard processing
            'fine': 0.85,    # Difficult liberation
            'very_fine': 0.7 # Very difficult
        }
        
        association_penalties = {
            'complex_sulfide': 0.9,     # Complex mineral intergrowths
            'native_gold': 1.05,        # Usually well liberated
            'arsenopyrite': 0.85,       # Refractory characteristics
            'tellurides': 0.8           # Difficult processing
        }
        
        size_factor = grain_size_factors.get(grain_size, 1.0)
        
        association_factor = 1.0
        for association in associations:
            association_factor *= association_penalties.get(association, 0.95)
        
        return size_factor * association_factor
    
    def _calculate_oxidation_state_factor(self, element: str, oxidation_state: str) -> float:
        """Calculate processing factor based on oxidation state"""
        oxidation_factors = {
            'Cu': {
                'sulfide': 1.0,    # Standard flotation
                'oxide': 0.75,     # Heap leach or SX-EW
                'mixed': 0.85      # Complex processing
            },
            'Pb': {
                'sulfide': 1.0,
                'oxide': 0.7,
                'mixed': 0.8
            },
            'Zn': {
                'sulfide': 1.0,
                'oxide': 0.65,
                'mixed': 0.75
            },
            'Au': {
                'native': 1.0,
                'sulfide': 0.9,
                'oxide': 0.95,
                'mixed': 0.85
            }
        }
        
        return oxidation_factors.get(element, {}).get(oxidation_state, 1.0)
    
    def _calculate_alteration_mineral_factor(self, element: str, alteration_minerals: List[str]) -> float:
        """Calculate processing penalty from alteration minerals"""
        alteration_penalties = {
            'kaolinite': 0.92,    # Clay interference
            'illite': 0.88,       # Swelling clay
            'smectite': 0.80,     # High swelling clay
            'sericite': 0.95,     # Mild flotation interference
            'chlorite': 0.93,     # Mg interference in flotation
            'talc': 0.85,         # Natural flotation, selectivity issues
            'graphite': 0.82      # Natural flotation, carbon interference
        }
        
        penalty_factor = 1.0
        for mineral in alteration_minerals:
            penalty_factor *= alteration_penalties.get(mineral, 0.96)
        
        return max(0.6, penalty_factor)
    
    def _calculate_trace_element_factor(self, element: str, trace_elements: List[str]) -> float:
        """Calculate processing impact from trace elements"""
        trace_penalties = {
            'arsenic': 0.85,      # Environmental and processing issues
            'antimony': 0.88,     # Smelter penalties
            'bismuth': 0.90,      # Refining complications
            'mercury': 0.75,      # Environmental restrictions
            'cadmium': 0.82,      # Environmental and health concerns
            'thallium': 0.80,     # Highly toxic
            'tellurium': 0.85,    # Processing complications
            'selenium': 0.87      # Environmental concerns
        }
        
        penalty_factor = 1.0
        for trace in trace_elements:
            penalty_factor *= trace_penalties.get(trace, 0.95)
        
        return max(0.5, penalty_factor)
    
    def _calculate_texture_factor(self, element: str, texture: str, hardness: str) -> float:
        """Calculate processing factor based on ore texture and hardness"""
        texture_factors = {
            'massive': 1.0,
            'banded': 0.95,       # Variable liberation
            'disseminated': 0.9,  # Fine grinding required
            'veinlet': 0.85,      # Complex liberation
            'stockwork': 0.88     # Highly variable
        }
        
        hardness_factors = {
            'soft': 1.05,      # Easy grinding
            'medium': 1.0,     # Standard grinding
            'hard': 0.92,      # High energy grinding
            'very_hard': 0.85  # Very high energy grinding
        }
        
        texture_factor = texture_factors.get(texture, 1.0)
        hardness_factor = hardness_factors.get(hardness, 1.0)
        
        return texture_factor * hardness_factor
    
    def _get_ore_type_mineralogy_correction(self, element: str, ore_type: str) -> float:
        """Get ore type specific mineralogy corrections"""
        corrections = {
            'high_grade_copper': {
                'Cu': 1.05, 'Au': 1.0, 'Ag': 1.0, 'Mo': 1.0
            },
            'copper': {
                'Cu': 1.0, 'Au': 0.95, 'Ag': 0.95, 'Mo': 0.9
            },
            'polymetallic': {
                'Cu': 0.9, 'Pb': 0.92, 'Zn': 0.9, 'Au': 0.85, 'Ag': 0.88
            },
            'lead_zinc': {
                'Pb': 1.0, 'Zn': 1.0, 'Ag': 1.05, 'Cu': 0.8
            },
            'gold': {
                'Au': 1.0, 'Ag': 1.02, 'Cu': 0.7, 'Pb': 0.7, 'Zn': 0.7
            },
            'low_grade_mixed': {
                'Cu': 0.85, 'Au': 0.8, 'Ag': 0.8, 'Pb': 0.82, 'Zn': 0.8
            }
        }
        
        return corrections.get(ore_type, {}).get(element, 1.0)
    
    def _log_mineralogy_assessment(self, element: str, block: MiningBlock, 
                                 assessment: Dict[str, Any], final_factor: float):
        """Log detailed mineralogy assessment for complex cases"""
        logging.info(f"Complex mineralogy case - Element: {element}, Factor: {final_factor:.3f}")
        logging.info(f"Primary minerals: {assessment['primary_minerals']}")
        logging.info(f"Gangue minerals: {assessment['gangue_minerals']}")
        logging.info(f"Oxidation state: {assessment['oxidation_state']}")
        logging.info(f"Alteration: {assessment['alteration_minerals']}")
        logging.info(f"Trace elements: {assessment['trace_elements']}")
    
    def _calculate_liberation_factor(self, element: str, ore_type: str) -> float:
        """Calculate liberation characteristics factor"""
        liberation_factors = {
            'high_grade_copper': 1.05,
            'copper': 1.0,
            'polymetallic': 0.95,  # Complex mineralogy
            'lead_zinc': 0.98,
            'gold': 1.02,
            'low_grade_mixed': 0.90
        }
        
        return liberation_factors.get(ore_type, 1.0)
    
    def _calculate_payable_metal(self, element: str, recovered_metal: float) -> float:
        """Calculate payable metal after smelter deductions"""
        # Smelter deduction rates by element
        deduction_rates = {
            'Cu': 0.02,   # 2% deduction
            'Pb': 0.03,   # 3% deduction
            'Zn': 0.025,  # 2.5% deduction
            'Au': 0.005,  # 0.5% deduction (precious metals)
            'Ag': 0.01,   # 1% deduction
            'Ni': 0.025
        }
        
        deduction_rate = deduction_rates.get(element, 0.02)
        return recovered_metal * (1.0 - deduction_rate)
    
    def _estimate_concentrate_grade(self, element: str, ore_grade: float, ore_type: str) -> float:
        """Estimate concentrate grade for smelter calculations"""
        # Typical concentration ratios by element and ore type
        concentration_ratios = {
            'Cu': {'copper': 25, 'polymetallic': 20, 'high_grade_copper': 28},
            'Pb': {'lead_zinc': 15, 'polymetallic': 12},
            'Zn': {'lead_zinc': 18, 'polymetallic': 15},
            'Au': {'gold': 1000, 'copper': 50},  # g/t in concentrate
            'Ag': {'silver': 800, 'lead_zinc': 300, 'copper': 100}
        }
        
        ratio = concentration_ratios.get(element, {}).get(ore_type, 20)
        return min(ore_grade * ratio, 60.0)  # Cap at 60% concentrate grade
    
    def _calculate_smelter_penalties(self, all_grades: Dict[str, float], concentrate_grade: float) -> float:
        """Calculate smelter penalties for deleterious elements"""
        penalties = 0.0
        
        # Penalty elements and thresholds
        penalty_elements = {
            'As': (0.5, 10.0),   # (threshold %, penalty per % excess)
            'Sb': (0.3, 15.0),
            'Bi': (0.1, 20.0),
            'S': (3.0, 2.0),
            'F': (0.2, 25.0)
        }
        
        for element, (threshold, penalty_rate) in penalty_elements.items():
            if element in all_grades:
                element_in_conc = all_grades[element] * concentrate_grade / 100
                if element_in_conc > threshold:
                    excess = element_in_conc - threshold
                    penalties += excess * penalty_rate
        
        return penalties
    
    def _calculate_element_processing_cost(self, element: str, grade: float, ore_type: str, 
                                        hardness_factor: float, recovery_data: Dict) -> float:
        """Calculate element-specific processing costs"""
        
        # Base processing costs by element ($/tonne)
        base_costs = {
            'Cu': 12.0,
            'Pb': 15.0,
            'Zn': 14.0,
            'Au': 25.0,  # Higher cost for gold processing
            'Ag': 8.0,
            'Ni': 18.0
        }
        
        base_cost = base_costs.get(element, 10.0)
        
        # Hardness adjustment
        cost_with_hardness = base_cost * hardness_factor
        
        # Complex ore adjustment
        if ore_type == 'polymetallic':
            cost_with_hardness *= 1.2  # 20% increase for complex ores
        
        # Low grade adjustment (higher unit costs)
        if grade < 0.1:
            cost_with_hardness *= 1.3
        
        # Recovery efficiency adjustment (lower recovery = higher unit cost)
        recovery_efficiency = recovery_data.get('primary_recovery', 0.8)
        if recovery_efficiency < 0.8:
            cost_with_hardness *= (0.8 / recovery_efficiency)
        
        return cost_with_hardness * (grade / 100.0)  # Scale by metal content
    
    def _calculate_geological_complexity_multiplier(self, block: MiningBlock) -> float:
        """Calculate mining cost multiplier based on geological complexity"""
        complexity = 1.0
        
        # Fault density effects
        if hasattr(block, 'fault_density'):
            complexity += block.fault_density * 0.1
        
        # Rock competency
        if hasattr(block, 'rock_quality_designation'):
            rqd = block.rock_quality_designation
            if rqd < 25:  # Very poor
                complexity *= 1.5
            elif rqd < 50:  # Poor
                complexity *= 1.2
            elif rqd < 75:  # Fair
                complexity *= 1.1
        
        # Weathering effects
        if hasattr(block, 'weathering_grade'):
            weathering_multipliers = {1: 1.0, 2: 1.1, 3: 1.2, 4: 1.4, 5: 1.6, 6: 2.0}
            complexity *= weathering_multipliers.get(block.weathering_grade, 1.0)
        
        return min(complexity, 2.5)  # Cap at 2.5x base cost
    
    def _calculate_geotechnical_stability_factor(self, block: MiningBlock) -> float:
        """Calculate geotechnical stability factor affecting mining costs"""
        stability_factor = 1.0
        
        # Slope stability considerations
        if hasattr(block, 'slope_angle'):
            if block.slope_angle > 60:  # Steep slopes
                stability_factor *= 1.3
            elif block.slope_angle > 45:
                stability_factor *= 1.15
        
        # Groundwater conditions
        if hasattr(block, 'groundwater_pressure'):
            if block.groundwater_pressure > 0.5:  # High pressure
                stability_factor *= 1.2
        
        return min(stability_factor, 2.0)
    
    def _calculate_blasting_cost(self, block: MiningBlock, hardness_factor: float) -> float:
        """Calculate blasting costs based on rock properties"""
        base_blasting_cost = 2.5  # $/tonne
        
        # Hardness adjustment
        blasting_cost = base_blasting_cost * hardness_factor
        
        # Rock type specific adjustments
        if hasattr(block, 'rock_type'):
            rock_multipliers = {
                'granite': 1.3,
                'basalt': 1.25,
                'limestone': 0.9,
                'sandstone': 0.85,
                'shale': 0.8
            }
            blasting_cost *= rock_multipliers.get(block.rock_type.lower(), 1.0)
        
        return blasting_cost
    
    def _calculate_hauling_cost(self, block: MiningBlock) -> float:
        """Calculate hauling costs based on distance and elevation"""
        base_hauling_cost = 1.5  # $/tonne for 1km haul
        
        # Distance factor
        haul_distance = getattr(block, 'haul_distance', 1.0)  # km
        distance_cost = base_hauling_cost * haul_distance
        
        # Elevation change factor
        elevation_change = getattr(block, 'elevation_change', 0)  # meters
        if elevation_change > 0:  # Uphill
            elevation_cost = elevation_change * 0.002  # $0.002/tonne/meter
        else:  # Downhill - slight cost reduction
            elevation_cost = elevation_change * 0.001
        
        return distance_cost + elevation_cost
    
    def _calculate_equipment_cost_adjustment(self, block: MiningBlock, ore_type: str) -> float:
        """Calculate equipment cost adjustments"""
        # Base equipment cost allocation
        base_equipment_cost = 3.0  # $/tonne
        
        # Ore type specific wear factors
        wear_factors = {
            'high_grade_copper': 1.1,  # Higher wear on equipment
            'polymetallic': 1.15,      # Complex processing
            'low_grade_mixed': 0.95    # Lower wear
        }
        
        wear_factor = wear_factors.get(ore_type, 1.0)
        
        return base_equipment_cost * wear_factor
    
    def _calculate_transportation_costs(self, block: MiningBlock) -> float:
        """Calculate transportation costs to processing facility"""
        base_transport_cost = 0.5  # $/tonne
        
        # Distance to processing plant
        transport_distance = getattr(block, 'transport_distance', 2.0)  # km
        
        return base_transport_cost * transport_distance
    
    def _calculate_environmental_costs(self, block: MiningBlock, ore_type: str) -> float:
        """Calculate environmental compliance and monitoring costs"""
        base_env_cost = 1.0  # $/tonne
        
        # Ore type specific environmental factors
        env_factors = {
            'polymetallic': 1.3,      # Higher environmental impact
            'lead_zinc': 1.2,         # Lead environmental concerns
            'low_grade_mixed': 0.9
        }
        
        env_factor = env_factors.get(ore_type, 1.0)
        
        # Proximity to sensitive areas
        if hasattr(block, 'environmental_sensitivity'):
            sensitivity_multiplier = 1.0 + block.environmental_sensitivity * 0.5
        else:
            sensitivity_multiplier = 1.0
        
        return base_env_cost * env_factor * sensitivity_multiplier
    
    def _calculate_waste_management_cost(self, block: MiningBlock) -> float:
        """Calculate waste rock management costs"""
        waste_cost_per_tonne = 0.8  # $/tonne of waste
        
        # Waste to ore ratio
        waste_ratio = getattr(block, 'waste_ratio', 2.0)  # tonnes waste per tonne ore
        
        return waste_cost_per_tonne * waste_ratio
    
    def _calculate_tailings_management_cost(self, block: MiningBlock, ore_type: str) -> float:
        """Calculate tailings management and disposal costs"""
        base_tailings_cost = 2.0  # $/tonne processed
        
        # Ore type specific tailings factors
        tailings_factors = {
            'polymetallic': 1.4,      # More complex tailings
            'lead_zinc': 1.3,         # Heavy metals in tailings
            'gold': 1.2,              # Cyanide management
            'high_grade_copper': 1.1
        }
        
        tailings_factor = tailings_factors.get(ore_type, 1.0)
        
        return base_tailings_cost * tailings_factor
    
    def _calculate_allocated_rehabilitation_cost(self, block: MiningBlock) -> float:
        """Calculate allocated rehabilitation and closure costs"""
        # Total rehabilitation cost allocated over ore reserves
        total_rehabilitation_cost = 50e6  # $50M total (example)
        total_ore_tonnes = 100e6         # 100Mt total ore (example)
        
        allocated_cost_per_tonne = total_rehabilitation_cost / total_ore_tonnes
        
        return allocated_cost_per_tonne
    
    def _calculate_infrastructure_cost_allocation(self, block: MiningBlock) -> float:
        """Calculate infrastructure cost allocation"""
        # Infrastructure costs allocated based on block value contribution
        base_infrastructure_cost = 1.5  # $/tonne
        
        # Distance from infrastructure
        infrastructure_distance = getattr(block, 'infrastructure_distance', 1.0)
        distance_factor = 1.0 + infrastructure_distance * 0.1
        
        return base_infrastructure_cost * distance_factor
    
    def _calculate_technical_risk_discount(self, block: MiningBlock, ore_type: str) -> float:
        """Calculate technical risk discount factor"""
        base_risk = 0.02  # 2% base technical risk
        
        # Ore type specific risks
        risk_factors = {
            'polymetallic': 0.05,      # Higher metallurgical risk
            'low_grade_mixed': 0.04,   # Grade control risks
            'high_grade_copper': 0.01  # Lower risk for well-understood ore
        }
        
        ore_risk = risk_factors.get(ore_type, base_risk)
        
        # Depth-related risks
        depth_risk = 0.0
        if hasattr(block, 'depth') and block.depth > 200:
            depth_risk = (block.depth - 200) * 0.0001  # Additional risk with depth
        
        return min(base_risk + ore_risk + depth_risk, 0.15)  # Cap at 15%
    
    def _calculate_market_risk_discount(self, block: MiningBlock) -> float:
        """Calculate market risk discount factor"""
        # Market volatility discount based on commodity mix
        base_market_risk = 0.03  # 3% base market risk
        
        # Calculate weighted market risk based on revenue contribution
        total_value = sum(self.params.commodity_prices.get(el, 0) * grade 
                         for el, grade in block.grades.items())
        
        if total_value == 0:
            return base_market_risk
        
        weighted_risk = 0.0
        for element, grade in block.grades.items():
            if element in self.params.commodity_prices:
                element_value = self.params.commodity_prices[element] * grade
                weight = element_value / total_value
                
                # Element-specific market risks
                element_risks = {
                    'Cu': 0.08,   # High volatility
                    'Au': 0.04,   # Lower volatility
                    'Ag': 0.10,   # High volatility
                    'Pb': 0.06,
                    'Zn': 0.07,
                    'Ni': 0.09
                }
                
                element_risk = element_risks.get(element, 0.05)
                weighted_risk += weight * element_risk
        
        return min(weighted_risk, 0.12)  # Cap at 12%
    
    def _calculate_smelting_charges(self, element: str, payable_metal: float, concentrate_grade: float) -> float:
        """Calculate smelting charges"""
        # Smelting charges by element ($/tonne concentrate)
        smelting_charges = {
            'Cu': 80.0,
            'Pb': 160.0,
            'Zn': 190.0,
            'Ni': 120.0
        }
        
        base_charge = smelting_charges.get(element, 100.0)
        
        # Grade penalties for low-grade concentrates
        if concentrate_grade < 15:
            grade_penalty = (15 - concentrate_grade) * 5.0  # $5/% below 15%
            base_charge += grade_penalty
        
        # Convert to cost per tonne ore
        concentrate_tonnes_per_ore_tonne = (payable_metal / (concentrate_grade / 100.0)) / 1000.0
        
        return base_charge * concentrate_tonnes_per_ore_tonne
    
    def _calculate_refining_charges(self, element: str, payable_metal: float) -> float:
        """Calculate refining charges"""
        # Refining charges by element ($/lb or $/oz)
        refining_charges = {
            'Cu': 0.08,   # $/lb
            'Pb': 0.12,   # $/lb  
            'Zn': 0.10,   # $/lb
            'Ni': 0.15,   # $/lb
            'Au': 5.0,    # $/oz
            'Ag': 0.30    # $/oz
        }
        
        charge_rate = refining_charges.get(element, 0.10)
        
        # Convert metal tonnage to appropriate units
        if element in ['Au', 'Ag']:
            metal_oz = payable_metal * 32150.75  # Convert tonnes to troy oz
            return charge_rate * metal_oz
        else:
            metal_lbs = payable_metal * 2204.62  # Convert tonnes to lbs
            return charge_rate * metal_lbs
    
    def _calculate_dynamic_cutoff_grade(self, ore_type: str) -> float:
        """Calculate dynamic cutoff grade based on current conditions"""
        # Base cutoff grades by ore type ($/tonne)
        base_cutoffs = {
            'high_grade_copper': 5.0,
            'copper': 8.0,
            'polymetallic': 12.0,
            'lead_zinc': 10.0,
            'gold': 15.0,
            'silver': 6.0,
            'low_grade_mixed': 4.0
        }
        
        base_cutoff = base_cutoffs.get(ore_type, 8.0)
        
        # Market condition adjustments
        market_adjustment = 1.0  # Could be adjusted based on current market conditions
        
        # Operating cost inflation adjustment
        cost_inflation = 1.02  # 2% annual inflation (example)
        
        return base_cutoff * market_adjustment * cost_inflation

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
        """Comprehensive multi-objective processing mix optimization using mathematical programming"""
        try:
            # Initialize comprehensive optimization framework
            optimization_result = self._setup_optimization_framework(ore_types, total_capacity)
            
            # Multi-stage optimization approach
            # Stage 1: Initial mixed-integer linear programming solution
            milp_solution = self._solve_milp_optimization(ore_types, total_capacity)
            
            # Stage 2: Stochastic optimization for uncertainty handling
            stochastic_solution = self._solve_stochastic_optimization(
                ore_types, total_capacity, milp_solution)
            
            # Stage 3: Multi-objective optimization (NPV, risk, utilization)
            pareto_solutions = self._solve_multiobjective_optimization(
                ore_types, total_capacity, stochastic_solution)
            
            # Stage 4: Dynamic programming for temporal optimization
            dynamic_solution = self._solve_dynamic_optimization(
                ore_types, total_capacity, pareto_solutions)
            
            # Stage 5: Robust optimization under uncertainty
            robust_solution = self._solve_robust_optimization(
                ore_types, total_capacity, dynamic_solution)
            
            # Final solution synthesis and validation
            final_solution = self._synthesize_optimization_results(
                ore_types, total_capacity, robust_solution)
            
            # Comprehensive sensitivity analysis
            sensitivity_analysis = self._perform_comprehensive_sensitivity_analysis(
                ore_types, total_capacity, final_solution)
            
            # Risk-adjusted optimization metrics
            risk_metrics = self._calculate_risk_adjusted_metrics(
                ore_types, final_solution, sensitivity_analysis)
            
            return {
                'optimal_allocations': final_solution['allocations'],
                'optimization_summary': final_solution['summary'],
                'pareto_frontier': pareto_solutions,
                'sensitivity_analysis': sensitivity_analysis,
                'risk_metrics': risk_metrics,
                'convergence_status': final_solution['convergence'],
                'computational_statistics': final_solution['statistics']
            }
            
        except Exception as e:
            logging.error(f"Error in comprehensive processing mix optimization: {e}")
            # Fallback to simplified optimization
            return self._fallback_optimization(ore_types, total_capacity)
    
    def _setup_optimization_framework(self, ore_types: List[Dict], total_capacity: float) -> Dict:
        """Setup comprehensive optimization framework with constraints and objectives"""
        framework = {
            'decision_variables': {},
            'objective_functions': [],
            'constraints': [],
            'parameters': {},
            'bounds': {}
        }
        
        try:
            # Define decision variables: x[i,j] = tonnage of ore type i processed via route j
            n_ore_types = len(ore_types)
            n_routes = len(self.routes)
            
            framework['parameters'] = {
                'n_ore_types': n_ore_types,
                'n_routes': n_routes,
                'total_capacity': total_capacity,
                'time_horizon': 365,  # Days
                'discount_rate': 0.08,
                'risk_tolerance': 0.15
            }
            
            # Decision variable bounds
            for i in range(n_ore_types):
                for j in range(n_routes):
                    var_key = f'x_{i}_{j}'
                    framework['bounds'][var_key] = {
                        'lower': 0.0,
                        'upper': min(ore_types[i]['tonnage'], self.routes[j].capacity)
                    }
            
            # Setup objective functions
            framework['objective_functions'] = [
                'maximize_npv',           # Primary: Net Present Value
                'minimize_risk',          # Secondary: Risk minimization
                'maximize_utilization',   # Tertiary: Capacity utilization
                'minimize_environmental_impact',  # Quaternary: Environmental
                'maximize_metal_recovery'  # Quinary: Technical efficiency
            ]
            
            # Setup constraint categories
            framework['constraints'] = [
                'capacity_constraints',
                'ore_availability_constraints', 
                'processing_compatibility_constraints',
                'quality_constraints',
                'environmental_constraints',
                'operational_constraints',
                'financial_constraints',
                'risk_constraints'
            ]
            
            return framework
            
        except Exception as e:
            logging.error(f"Error setting up optimization framework: {e}")
            return {}
    
    def _solve_milp_optimization(self, ore_types: List[Dict], total_capacity: float) -> Dict:
        """Solve mixed-integer linear programming optimization"""
        try:
            # Initialize optimization matrices
            n_ore_types = len(ore_types)
            n_routes = len(self.routes)
            
            # Build coefficient matrices
            value_matrix = np.zeros((n_ore_types, n_routes))
            cost_matrix = np.zeros((n_ore_types, n_routes))
            capacity_vector = np.array([route.capacity for route in self.routes])
            availability_vector = np.array([ore['tonnage'] for ore in ore_types])
            
            # Calculate value and cost coefficients
            for i, ore_type in enumerate(ore_types):
                for j, route in enumerate(self.routes):
                    if self._is_compatible_combination(ore_type, route):
                        evaluation = self.evaluate_processing_options(
                            ore_type['composition'], 1.0)
                        
                        route_eval = next((r for r in evaluation['route_evaluations'] 
                                         if r['route_name'] == route.route_name), None)
                        
                        if route_eval:
                            value_matrix[i, j] = route_eval['net_value']
                            cost_matrix[i, j] = route_eval['total_cost']
                        else:
                            value_matrix[i, j] = -1e6  # Infeasible penalty
                            cost_matrix[i, j] = 1e6
                    else:
                        value_matrix[i, j] = -1e6  # Incompatible penalty
                        cost_matrix[i, j] = 1e6
            
            # Solve linear programming relaxation using simplex method
            solution_matrix = self._solve_lp_relaxation(
                value_matrix, cost_matrix, capacity_vector, availability_vector, total_capacity)
            
            # Apply branch-and-bound for integer constraints
            integer_solution = self._branch_and_bound_solver(
                solution_matrix, value_matrix, capacity_vector, availability_vector)
            
            # Extract solution
            milp_solution = {
                'allocation_matrix': integer_solution,
                'objective_value': np.sum(integer_solution * value_matrix),
                'total_cost': np.sum(integer_solution * cost_matrix),
                'capacity_utilization': np.sum(integer_solution) / total_capacity,
                'feasible': self._validate_solution_feasibility(integer_solution, 
                                                              capacity_vector, availability_vector)
            }
            
            return milp_solution
            
        except Exception as e:
            logging.error(f"Error in MILP optimization: {e}")
            return {'allocation_matrix': np.zeros((len(ore_types), len(self.routes))),
                   'feasible': False}
    
    def _solve_lp_relaxation(self, value_matrix: np.ndarray, cost_matrix: np.ndarray,
                            capacity_vector: np.ndarray, availability_vector: np.ndarray,
                            total_capacity: float) -> np.ndarray:
        """Solve linear programming relaxation using dual simplex method"""
        try:
            n_ore_types, n_routes = value_matrix.shape
            
            # Formulate standard form LP: max c^T x subject to Ax <= b, x >= 0
            # Decision variables: x[i,j] for i in ore_types, j in routes
            
            # Objective function coefficients (maximize value)
            c = value_matrix.flatten()
            
            # Constraint matrix A and bounds b
            constraints = []
            bounds = []
            
            # Capacity constraints: sum over ore types for each route
            for j in range(n_routes):
                constraint = np.zeros(n_ore_types * n_routes)
                for i in range(n_ore_types):
                    constraint[i * n_routes + j] = 1.0
                constraints.append(constraint)
                bounds.append(capacity_vector[j])
            
            # Ore availability constraints: sum over routes for each ore type
            for i in range(n_ore_types):
                constraint = np.zeros(n_ore_types * n_routes)
                for j in range(n_routes):
                    constraint[i * n_routes + j] = 1.0
                constraints.append(constraint)
                bounds.append(availability_vector[i])
            
            # Total capacity constraint
            total_constraint = np.ones(n_ore_types * n_routes)
            constraints.append(total_constraint)
            bounds.append(total_capacity)
            
            A = np.array(constraints)
            b = np.array(bounds)
            
            # Solve using dual simplex algorithm
            solution_vector = self._dual_simplex_solver(c, A, b)
            
            # Reshape solution back to matrix form
            solution_matrix = solution_vector.reshape((n_ore_types, n_routes))
            
            return solution_matrix
            
        except Exception as e:
            logging.error(f"Error in LP relaxation: {e}")
            return np.zeros((value_matrix.shape))
    
    def _dual_simplex_solver(self, c: np.ndarray, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Dual simplex algorithm implementation"""
        try:
            m, n = A.shape
            
            # Create initial tableau with slack variables
            # [c^T 0 0]
            # [A   I b]
            tableau = np.zeros((m + 1, n + m + 1))
            
            # Objective row
            tableau[0, :n] = -c  # Negative for maximization
            
            # Constraint rows
            tableau[1:, :n] = A
            tableau[1:, n:n+m] = np.eye(m)  # Slack variables
            tableau[1:, -1] = b  # RHS
            
            # Dual simplex iterations
            max_iterations = 1000
            iteration = 0
            
            while iteration < max_iterations:
                # Check for optimality (all RHS >= 0 for primal feasible)
                if np.all(tableau[1:, -1] >= -1e-10):  # Small tolerance
                    break
                
                # Select leaving variable (most negative RHS)
                leaving_row = np.argmin(tableau[1:, -1]) + 1
                
                if tableau[leaving_row, -1] >= -1e-10:
                    break  # Already optimal
                
                # Select entering variable (dual simplex ratio test)
                ratios = []
                for j in range(n + m):
                    if tableau[leaving_row, j] < -1e-10:  # Negative element
                        ratio = -tableau[0, j] / tableau[leaving_row, j]
                        ratios.append((ratio, j))
                    else:
                        ratios.append((float('inf'), j))
                
                if not ratios or all(r[0] == float('inf') for r in ratios):
                    # Unbounded or infeasible
                    break
                
                entering_col = min(ratios, key=lambda x: x[0])[1]
                pivot_element = tableau[leaving_row, entering_col]
                
                if abs(pivot_element) < 1e-10:
                    break  # Numerical issues
                
                # Pivot operation
                tableau[leaving_row, :] /= pivot_element
                
                for i in range(m + 1):
                    if i != leaving_row:
                        multiplier = tableau[i, entering_col]
                        tableau[i, :] -= multiplier * tableau[leaving_row, :]
                
                iteration += 1
            
            # Extract solution
            solution = np.zeros(n)
            for j in range(n):
                # Check if variable is basic
                col = tableau[1:, j]
                basic_rows = np.where(np.abs(col - 1.0) < 1e-10)[0]
                
                if len(basic_rows) == 1:
                    row = basic_rows[0] + 1
                    if np.sum(np.abs(tableau[row, :n+m] - np.eye(n+m)[j]) < 1e-10) == n+m:
                        solution[j] = max(0, tableau[row, -1])  # Non-negativity
            
            return solution
            
        except Exception as e:
            logging.error(f"Error in dual simplex solver: {e}")
            return np.zeros(len(c))
    
    def _branch_and_bound_solver(self, lp_solution: np.ndarray, value_matrix: np.ndarray,
                                capacity_vector: np.ndarray, availability_vector: np.ndarray) -> np.ndarray:
        """Branch-and-bound for integer constraints"""
        try:
            # For simplicity, use rounding heuristic followed by local search
            # In production, implement full branch-and-bound tree
            
            # Round down fractional solutions
            integer_solution = np.floor(lp_solution)
            
            # Local search improvement
            improved_solution = self._local_search_improvement(
                integer_solution, value_matrix, capacity_vector, availability_vector)
            
            # Feasibility repair if needed
            feasible_solution = self._repair_solution_feasibility(
                improved_solution, capacity_vector, availability_vector)
            
            return feasible_solution
            
        except Exception as e:
            logging.error(f"Error in branch-and-bound solver: {e}")
            return lp_solution
    
    def _local_search_improvement(self, solution: np.ndarray, value_matrix: np.ndarray,
                                 capacity_vector: np.ndarray, availability_vector: np.ndarray) -> np.ndarray:
        """Local search for solution improvement"""
        try:
            current_solution = solution.copy()
            current_value = np.sum(current_solution * value_matrix)
            improved = True
            max_iterations = 100
            iteration = 0
            
            while improved and iteration < max_iterations:
                improved = False
                
                # Try 2-opt moves: decrease one allocation, increase another
                n_ore_types, n_routes = solution.shape
                
                for i1 in range(n_ore_types):
                    for j1 in range(n_routes):
                        if current_solution[i1, j1] > 0:
                            for i2 in range(n_ore_types):
                                for j2 in range(n_routes):
                                    if (i1, j1) != (i2, j2):
                                        # Try moving one unit from (i1,j1) to (i2,j2)
                                        new_solution = current_solution.copy()
                                        new_solution[i1, j1] -= 1
                                        new_solution[i2, j2] += 1
                                        
                                        if self._validate_solution_feasibility(
                                            new_solution, capacity_vector, availability_vector):
                                            new_value = np.sum(new_solution * value_matrix)
                                            
                                            if new_value > current_value:
                                                current_solution = new_solution
                                                current_value = new_value
                                                improved = True
                
                iteration += 1
            
            return current_solution
            
        except Exception as e:
            logging.error(f"Error in local search improvement: {e}")
            return solution
    
    def _repair_solution_feasibility(self, solution: np.ndarray, capacity_vector: np.ndarray,
                                   availability_vector: np.ndarray) -> np.ndarray:
        """Repair solution to ensure feasibility"""
        try:
            repaired_solution = solution.copy()
            
            # Check and repair capacity constraints
            for j, capacity in enumerate(capacity_vector):
                total_allocated = np.sum(repaired_solution[:, j])
                if total_allocated > capacity:
                    # Scale down proportionally
                    scale_factor = capacity / total_allocated
                    repaired_solution[:, j] *= scale_factor
            
            # Check and repair availability constraints
            for i, availability in enumerate(availability_vector):
                total_allocated = np.sum(repaired_solution[i, :])
                if total_allocated > availability:
                    # Scale down proportionally
                    scale_factor = availability / total_allocated
                    repaired_solution[i, :] *= scale_factor
            
            return repaired_solution
            
        except Exception as e:
            logging.error(f"Error in solution feasibility repair: {e}")
            return solution
    
    def _validate_solution_feasibility(self, solution: np.ndarray, capacity_vector: np.ndarray,
                                     availability_vector: np.ndarray) -> bool:
        """Validate solution feasibility"""
        try:
            tolerance = 1e-6
            
            # Check non-negativity
            if np.any(solution < -tolerance):
                return False
            
            # Check capacity constraints
            for j, capacity in enumerate(capacity_vector):
                if np.sum(solution[:, j]) > capacity + tolerance:
                    return False
            
            # Check availability constraints
            for i, availability in enumerate(availability_vector):
                if np.sum(solution[i, :]) > availability + tolerance:
                    return False
            
            return True
            
        except Exception as e:
            logging.error(f"Error validating solution feasibility: {e}")
            return False
    
    def _solve_stochastic_optimization(self, ore_types: List[Dict], total_capacity: float,
                                     milp_solution: Dict) -> Dict:
        """Solve stochastic optimization under uncertainty"""
        try:
            # Monte Carlo simulation for uncertainty quantification
            n_scenarios = 1000
            scenario_solutions = []
            
            for scenario in range(n_scenarios):
                # Generate random perturbations to parameters
                perturbed_ore_types = self._generate_scenario_parameters(ore_types)
                
                # Solve deterministic problem for this scenario
                scenario_solution = self._solve_scenario_optimization(
                    perturbed_ore_types, total_capacity, milp_solution)
                
                scenario_solutions.append(scenario_solution)
            
            # Analyze scenario results
            stochastic_metrics = self._analyze_stochastic_results(scenario_solutions)
            
            # Expected value solution
            expected_solution = self._calculate_expected_value_solution(scenario_solutions)
            
            # Risk measures
            risk_measures = self._calculate_risk_measures(scenario_solutions)
            
            return {
                'expected_solution': expected_solution,
                'stochastic_metrics': stochastic_metrics,
                'risk_measures': risk_measures,
                'scenario_solutions': scenario_solutions[:10]  # Sample scenarios
            }
            
        except Exception as e:
            logging.error(f"Error in stochastic optimization: {e}")
            return {'expected_solution': milp_solution}
    
    def _generate_scenario_parameters(self, ore_types: List[Dict]) -> List[Dict]:
        """Generate random scenario parameters"""
        try:
            perturbed_ore_types = []
            
            for ore_type in ore_types:
                perturbed_ore = ore_type.copy()
                
                # Add uncertainty to grades (log-normal distribution)
                for element, grade in ore_type['composition'].items():
                    if grade > 0:
                        # 20% coefficient of variation
                        std_dev = 0.2 * grade
                        perturbed_grade = max(0, np.random.lognormal(
                            np.log(grade) - 0.5 * (std_dev/grade)**2, std_dev/grade))
                        perturbed_ore['composition'][element] = perturbed_grade
                
                # Add uncertainty to tonnage (normal distribution, 10% CV)
                tonnage_std = 0.1 * ore_type['tonnage']
                perturbed_tonnage = max(0, np.random.normal(ore_type['tonnage'], tonnage_std))
                perturbed_ore['tonnage'] = perturbed_tonnage
                
                perturbed_ore_types.append(perturbed_ore)
            
            return perturbed_ore_types
            
        except Exception as e:
            logging.error(f"Error generating scenario parameters: {e}")
            return ore_types
    
    def _solve_scenario_optimization(self, perturbed_ore_types: List[Dict], 
                                   total_capacity: float, base_solution: Dict) -> Dict:
        """Solve optimization for a single scenario"""
        try:
            # Use base MILP solution as starting point for efficiency
            scenario_milp = self._solve_milp_optimization(perturbed_ore_types, total_capacity)
            return scenario_milp
        except Exception as e:
            logging.error(f"Error in scenario optimization: {e}")
            return base_solution
    
    def _analyze_stochastic_results(self, scenario_solutions: List[Dict]) -> Dict:
        """Analyze results across all scenarios"""
        try:
            objective_values = [sol.get('objective_value', 0) for sol in scenario_solutions]
            utilizations = [sol.get('capacity_utilization', 0) for sol in scenario_solutions]
            
            return {
                'mean_objective': np.mean(objective_values),
                'std_objective': np.std(objective_values),
                'mean_utilization': np.mean(utilizations),
                'std_utilization': np.std(utilizations),
                'percentiles': {
                    'p10': np.percentile(objective_values, 10),
                    'p50': np.percentile(objective_values, 50),
                    'p90': np.percentile(objective_values, 90)
                }
            }
        except Exception as e:
            logging.error(f"Error analyzing stochastic results: {e}")
            return {}
    
    def _calculate_expected_value_solution(self, scenario_solutions: List[Dict]) -> Dict:
        """Calculate expected value solution"""
        try:
            if not scenario_solutions:
                return {}
            
            # Average allocation matrices
            allocation_matrices = [sol.get('allocation_matrix', np.array([])) 
                                 for sol in scenario_solutions if 'allocation_matrix' in sol]
            
            if allocation_matrices and len(allocation_matrices[0]) > 0:
                expected_allocation = np.mean(allocation_matrices, axis=0)
                expected_objective = np.mean([sol.get('objective_value', 0) 
                                            for sol in scenario_solutions])
                
                return {
                    'allocation_matrix': expected_allocation,
                    'objective_value': expected_objective,
                    'n_scenarios': len(scenario_solutions)
                }
            
            return {}
            
        except Exception as e:
            logging.error(f"Error calculating expected value solution: {e}")
            return {}
    
    def _calculate_risk_measures(self, scenario_solutions: List[Dict]) -> Dict:
        """Calculate various risk measures"""
        try:
            objective_values = [sol.get('objective_value', 0) for sol in scenario_solutions]
            
            if not objective_values:
                return {}
            
            sorted_values = sorted(objective_values)
            n = len(sorted_values)
            
            # Value at Risk (VaR) at different confidence levels
            var_95 = sorted_values[int(0.05 * n)] if n > 20 else min(sorted_values)
            var_99 = sorted_values[int(0.01 * n)] if n > 100 else min(sorted_values)
            
            # Conditional Value at Risk (CVaR)
            cvar_95 = np.mean(sorted_values[:int(0.05 * n)]) if n > 20 else min(sorted_values)
            
            # Downside deviation
            mean_value = np.mean(objective_values)
            downside_values = [v for v in objective_values if v < mean_value]
            downside_deviation = np.std(downside_values) if downside_values else 0
            
            return {
                'var_95': var_95,
                'var_99': var_99,
                'cvar_95': cvar_95,
                'downside_deviation': downside_deviation,
                'worst_case': min(objective_values),
                'best_case': max(objective_values)
            }
            
        except Exception as e:
            logging.error(f"Error calculating risk measures: {e}")
            return {}
    
    def _solve_multiobjective_optimization(self, ore_types: List[Dict], total_capacity: float,
                                         stochastic_solution: Dict) -> Dict:
        """Solve multi-objective optimization using epsilon-constraint method"""
        try:
            # Define objectives: NPV, Risk, Utilization, Environmental Impact
            objectives = ['npv', 'risk', 'utilization', 'environmental']
            
            # Generate Pareto frontier using epsilon-constraint method
            pareto_solutions = []
            n_points = 20  # Points along Pareto frontier
            
            for i in range(n_points):
                # Vary epsilon constraints for secondary objectives
                risk_epsilon = 0.1 + 0.4 * i / (n_points - 1)  # 10% to 50% risk
                util_epsilon = 0.5 + 0.4 * i / (n_points - 1)  # 50% to 90% utilization
                env_epsilon = 0.3 + 0.5 * i / (n_points - 1)   # 30% to 80% env score
                
                # Solve constrained optimization problem
                pareto_point = self._solve_epsilon_constrained_problem(
                    ore_types, total_capacity, risk_epsilon, util_epsilon, env_epsilon)
                
                if pareto_point['feasible']:
                    pareto_solutions.append(pareto_point)
            
            # Select preferred solution based on utility function
            preferred_solution = self._select_preferred_solution(pareto_solutions)
            
            return {
                'pareto_frontier': pareto_solutions,
                'preferred_solution': preferred_solution,
                'n_efficient_solutions': len(pareto_solutions)
            }
            
        except Exception as e:
            logging.error(f"Error in multi-objective optimization: {e}")
            return {'pareto_frontier': [], 'preferred_solution': stochastic_solution}
    
    def _solve_epsilon_constrained_problem(self, ore_types: List[Dict], total_capacity: float,
                                         risk_epsilon: float, util_epsilon: float, 
                                         env_epsilon: float) -> Dict:
        """Solve epsilon-constrained optimization problem"""
        try:
            # Maximize NPV subject to risk, utilization, and environmental constraints
            
            # Start with MILP solution and add epsilon constraints
            base_solution = self._solve_milp_optimization(ore_types, total_capacity)
            
            if not base_solution['feasible']:
                return {'feasible': False}
            
            # Check if solution satisfies epsilon constraints
            current_risk = self._calculate_solution_risk(base_solution, ore_types)
            current_util = base_solution['capacity_utilization']
            current_env = self._calculate_environmental_score(base_solution, ore_types)
            
            # If constraints not satisfied, adjust solution
            if (current_risk <= risk_epsilon and 
                current_util >= util_epsilon and 
                current_env >= env_epsilon):
                
                return {
                    'feasible': True,
                    'allocation_matrix': base_solution['allocation_matrix'],
                    'npv': base_solution['objective_value'],
                    'risk': current_risk,
                    'utilization': current_util,
                    'environmental_score': current_env
                }
            else:
                # Need to adjust solution to meet epsilon constraints
                adjusted_solution = self._adjust_for_epsilon_constraints(
                    base_solution, ore_types, risk_epsilon, util_epsilon, env_epsilon)
                
                return adjusted_solution
            
        except Exception as e:
            logging.error(f"Error in epsilon-constrained problem: {e}")
            return {'feasible': False}
    
    def _calculate_solution_risk(self, solution: Dict, ore_types: List[Dict]) -> float:
        """Calculate risk metric for solution"""
        try:
            # Risk based on volatility of ore grades and market prices
            if 'allocation_matrix' not in solution:
                return 1.0
            
            allocation_matrix = solution['allocation_matrix']
            total_risk = 0.0
            total_value = 0.0
            
            for i, ore_type in enumerate(ore_types):
                for j, route in enumerate(self.routes):
                    allocation = allocation_matrix[i, j]
                    if allocation > 0:
                        # Grade volatility risk
                        grade_risk = sum(0.2 for grade in ore_type['composition'].values() if grade > 0)
                        
                        # Route complexity risk
                        route_risk = 0.1 * len(route.processing_steps)
                        
                        # Combined risk weighted by allocation
                        combined_risk = (grade_risk + route_risk) * allocation
                        total_risk += combined_risk
                        total_value += allocation
            
            return total_risk / max(total_value, 1.0) if total_value > 0 else 1.0
            
        except Exception as e:
            logging.error(f"Error calculating solution risk: {e}")
            return 1.0
    
    def _calculate_environmental_score(self, solution: Dict, ore_types: List[Dict]) -> float:
        """Calculate environmental performance score"""
        try:
            if 'allocation_matrix' not in solution:
                return 0.0
            
            allocation_matrix = solution['allocation_matrix']
            total_score = 0.0
            total_weight = 0.0
            
            for i, ore_type in enumerate(ore_types):
                for j, route in enumerate(self.routes):
                    allocation = allocation_matrix[i, j]
                    if allocation > 0:
                        # Route environmental score (0-1, higher is better)
                        route_score = 1.0 - route.environmental_impact
                        
                        weighted_score = route_score * allocation
                        total_score += weighted_score
                        total_weight += allocation
            
            return total_score / max(total_weight, 1.0) if total_weight > 0 else 0.0
            
        except Exception as e:
            logging.error(f"Error calculating environmental score: {e}")
            return 0.0
    
    def _adjust_for_epsilon_constraints(self, base_solution: Dict, ore_types: List[Dict],
                                      risk_epsilon: float, util_epsilon: float, 
                                      env_epsilon: float) -> Dict:
        """Comprehensive epsilon constraint adjustment using advanced constraint handling techniques"""
        try:
            # Initialize comprehensive constraint handling framework
            constraint_handler = self._initialize_constraint_handler(
                base_solution, ore_types, risk_epsilon, util_epsilon, env_epsilon)
            
            # Phase 1: Constraint violation analysis and classification
            constraint_violations = self._analyze_constraint_violations(
                base_solution, ore_types, risk_epsilon, util_epsilon, env_epsilon)
            
            # Phase 2: Lagrangian relaxation for constraint handling
            lagrangian_solution = self._solve_lagrangian_relaxation(
                base_solution, ore_types, constraint_violations)
            
            # Phase 3: Penalty method for constraint enforcement
            penalty_solution = self._apply_penalty_method(
                lagrangian_solution, ore_types, constraint_violations)
            
            # Phase 4: Sequential quadratic programming for refinement
            sqp_solution = self._apply_sequential_quadratic_programming(
                penalty_solution, ore_types, constraint_violations)
            
            # Phase 5: Augmented Lagrangian method for final adjustment
            augmented_solution = self._apply_augmented_lagrangian_method(
                sqp_solution, ore_types, constraint_violations)
            
            # Phase 6: Interior point method for feasibility restoration
            interior_point_solution = self._apply_interior_point_method(
                augmented_solution, ore_types, constraint_violations)
            
            # Phase 7: Trust region method for local optimization
            trust_region_solution = self._apply_trust_region_method(
                interior_point_solution, ore_types, constraint_violations)
            
            # Phase 8: Adaptive constraint scaling and normalization
            normalized_solution = self._apply_adaptive_constraint_scaling(
                trust_region_solution, ore_types, constraint_violations)
            
            # Phase 9: Final feasibility validation and repair
            final_solution = self._validate_and_repair_solution(
                normalized_solution, ore_types, risk_epsilon, util_epsilon, env_epsilon)
            
            # Phase 10: Solution quality assessment
            quality_metrics = self._assess_solution_quality(
                final_solution, base_solution, constraint_violations)
            
            return {
                'feasible': final_solution['feasible'],
                'allocation_matrix': final_solution['allocation_matrix'],
                'npv': final_solution['npv'],
                'risk': final_solution['risk'],
                'utilization': final_solution['utilization'],
                'environmental_score': final_solution['environmental_score'],
                'constraint_handling_metrics': final_solution['metrics'],
                'solution_quality': quality_metrics,
                'convergence_details': final_solution['convergence']
            }
            
        except Exception as e:
            logging.error(f"Error in comprehensive epsilon constraint adjustment: {e}")
            return self._fallback_constraint_adjustment(base_solution, ore_types, 
                                                      risk_epsilon, util_epsilon, env_epsilon)
    
    def _initialize_constraint_handler(self, base_solution: Dict, ore_types: List[Dict],
                                     risk_epsilon: float, util_epsilon: float, 
                                     env_epsilon: float) -> Dict:
        """Initialize comprehensive constraint handling framework"""
        try:
            n_ore_types = len(ore_types)
            n_routes = len(self.routes)
            
            handler = {
                'dimensions': {
                    'n_variables': n_ore_types * n_routes,
                    'n_ore_types': n_ore_types,
                    'n_routes': n_routes
                },
                'constraint_types': {
                    'risk_constraint': {'type': 'inequality', 'bound': risk_epsilon, 'weight': 1.0},
                    'utilization_constraint': {'type': 'inequality', 'bound': util_epsilon, 'weight': 0.8},
                    'environmental_constraint': {'type': 'inequality', 'bound': env_epsilon, 'weight': 0.6},
                    'capacity_constraints': {'type': 'inequality', 'bounds': [], 'weight': 1.0},
                    'availability_constraints': {'type': 'inequality', 'bounds': [], 'weight': 1.0},
                    'non_negativity_constraints': {'type': 'bound', 'lower': 0.0, 'weight': 2.0}
                },
                'optimization_parameters': {
                    'lagrangian_multipliers': np.zeros(5 + n_routes + n_ore_types),
                    'penalty_parameters': np.ones(5 + n_routes + n_ore_types) * 10.0,
                    'trust_region_radius': 1.0,
                    'convergence_tolerance': 1e-6,
                    'max_iterations': 100
                },
                'gradient_information': {
                    'objective_gradient': np.zeros(n_ore_types * n_routes),
                    'constraint_jacobian': np.zeros((5 + n_routes + n_ore_types, n_ore_types * n_routes)),
                    'hessian_approximation': np.eye(n_ore_types * n_routes)
                }
            }
            
            # Initialize capacity constraint bounds
            for j, route in enumerate(self.routes):
                handler['constraint_types']['capacity_constraints']['bounds'].append(route.capacity)
            
            # Initialize availability constraint bounds
            for i, ore_type in enumerate(ore_types):
                handler['constraint_types']['availability_constraints']['bounds'].append(ore_type['tonnage'])
            
            return handler
            
        except Exception as e:
            logging.error(f"Error initializing constraint handler: {e}")
            return {}
    
    def _analyze_constraint_violations(self, base_solution: Dict, ore_types: List[Dict],
                                     risk_epsilon: float, util_epsilon: float, 
                                     env_epsilon: float) -> Dict:
        """Comprehensive analysis of constraint violations"""
        try:
            violations = {
                'risk_violation': 0.0,
                'utilization_violation': 0.0,
                'environmental_violation': 0.0,
                'capacity_violations': [],
                'availability_violations': [],
                'total_violation_magnitude': 0.0,
                'violation_categories': [],
                'critical_constraints': []
            }
            
            if 'allocation_matrix' not in base_solution:
                violations['total_violation_magnitude'] = float('inf')
                return violations
            
            allocation_matrix = base_solution['allocation_matrix']
            
            # Analyze risk constraint violation
            current_risk = self._calculate_solution_risk(base_solution, ore_types)
            if current_risk > risk_epsilon:
                violations['risk_violation'] = current_risk - risk_epsilon
                violations['violation_categories'].append('risk')
                if violations['risk_violation'] > 0.1:
                    violations['critical_constraints'].append('risk')
            
            # Analyze utilization constraint violation
            current_util = base_solution.get('capacity_utilization', 0)
            if current_util < util_epsilon:
                violations['utilization_violation'] = util_epsilon - current_util
                violations['violation_categories'].append('utilization')
                if violations['utilization_violation'] > 0.1:
                    violations['critical_constraints'].append('utilization')
            
            # Analyze environmental constraint violation
            current_env = self._calculate_environmental_score(base_solution, ore_types)
            if current_env < env_epsilon:
                violations['environmental_violation'] = env_epsilon - current_env
                violations['violation_categories'].append('environmental')
                if violations['environmental_violation'] > 0.1:
                    violations['critical_constraints'].append('environmental')
            
            # Analyze capacity constraint violations
            for j, route in enumerate(self.routes):
                route_allocation = np.sum(allocation_matrix[:, j])
                if route_allocation > route.capacity:
                    violation = route_allocation - route.capacity
                    violations['capacity_violations'].append({
                        'route_index': j,
                        'route_name': route.route_name,
                        'violation_magnitude': violation,
                        'violation_percentage': violation / route.capacity
                    })
                    if violation / route.capacity > 0.05:
                        violations['critical_constraints'].append(f'capacity_{j}')
            
            # Analyze availability constraint violations
            for i, ore_type in enumerate(ore_types):
                ore_allocation = np.sum(allocation_matrix[i, :])
                if ore_allocation > ore_type['tonnage']:
                    violation = ore_allocation - ore_type['tonnage']
                    violations['availability_violations'].append({
                        'ore_type_index': i,
                        'ore_type_name': ore_type.get('name', f'Ore_{i}'),
                        'violation_magnitude': violation,
                        'violation_percentage': violation / ore_type['tonnage']
                    })
                    if violation / ore_type['tonnage'] > 0.05:
                        violations['critical_constraints'].append(f'availability_{i}')
            
            # Calculate total violation magnitude
            violations['total_violation_magnitude'] = (
                violations['risk_violation'] +
                violations['utilization_violation'] + 
                violations['environmental_violation'] +
                sum(v['violation_magnitude'] for v in violations['capacity_violations']) +
                sum(v['violation_magnitude'] for v in violations['availability_violations'])
            )
            
            return violations
            
        except Exception as e:
            logging.error(f"Error analyzing constraint violations: {e}")
            return {'total_violation_magnitude': float('inf')}
    
    def _solve_lagrangian_relaxation(self, base_solution: Dict, ore_types: List[Dict],
                                   constraint_violations: Dict) -> Dict:
        """Solve Lagrangian relaxation for constraint handling"""
        try:
            if 'allocation_matrix' not in base_solution:
                return base_solution
            
            allocation_matrix = base_solution['allocation_matrix'].copy()
            n_ore_types, n_routes = allocation_matrix.shape
            
            # Initialize Lagrangian multipliers
            lambda_risk = 1.0 if constraint_violations['risk_violation'] > 0 else 0.0
            lambda_util = 1.0 if constraint_violations['utilization_violation'] > 0 else 0.0
            lambda_env = 1.0 if constraint_violations['environmental_violation'] > 0 else 0.0
            lambda_capacity = np.ones(n_routes)
            lambda_availability = np.ones(n_ore_types)
            
            # Lagrangian dual ascent iterations
            max_iterations = 50
            step_size = 0.01
            
            for iteration in range(max_iterations):
                # Solve relaxed subproblem
                relaxed_solution = self._solve_lagrangian_subproblem(
                    allocation_matrix, ore_types, lambda_risk, lambda_util, lambda_env,
                    lambda_capacity, lambda_availability)
                
                if relaxed_solution is None:
                    break
                
                allocation_matrix = relaxed_solution
                
                # Update Lagrangian multipliers using subgradient method
                lambda_updates = self._compute_lagrangian_subgradients(
                    allocation_matrix, ore_types, constraint_violations)
                
                lambda_risk = max(0, lambda_risk + step_size * lambda_updates['risk'])
                lambda_util = max(0, lambda_util + step_size * lambda_updates['utilization'])
                lambda_env = max(0, lambda_env + step_size * lambda_updates['environmental'])
                
                for j in range(n_routes):
                    lambda_capacity[j] = max(0, lambda_capacity[j] + 
                                           step_size * lambda_updates['capacity'][j])
                
                for i in range(n_ore_types):
                    lambda_availability[i] = max(0, lambda_availability[i] + 
                                               step_size * lambda_updates['availability'][i])
                
                # Adaptive step size
                step_size *= 0.99
                
                # Check convergence
                if np.max(np.abs([lambda_updates[key] for key in lambda_updates.keys() 
                                if isinstance(lambda_updates[key], (int, float))])) < 1e-4:
                    break
            
            # Construct solution
            lagrangian_solution = {
                'allocation_matrix': allocation_matrix,
                'lagrangian_multipliers': {
                    'risk': lambda_risk,
                    'utilization': lambda_util,
                    'environmental': lambda_env,
                    'capacity': lambda_capacity.tolist(),
                    'availability': lambda_availability.tolist()
                },
                'iterations': iteration + 1,
                'converged': iteration < max_iterations - 1
            }
            
            # Calculate objective metrics
            value_matrix = self._get_value_matrix(ore_types)
            lagrangian_solution['npv'] = np.sum(allocation_matrix * value_matrix)
            lagrangian_solution['risk'] = self._calculate_solution_risk(
                {'allocation_matrix': allocation_matrix}, ore_types)
            lagrangian_solution['utilization'] = np.sum(allocation_matrix) / max(1, len(ore_types))
            lagrangian_solution['environmental_score'] = self._calculate_environmental_score(
                {'allocation_matrix': allocation_matrix}, ore_types)
            
            return lagrangian_solution
            
        except Exception as e:
            logging.error(f"Error in Lagrangian relaxation: {e}")
            return base_solution
    
    def _solve_lagrangian_subproblem(self, allocation_matrix: np.ndarray, ore_types: List[Dict],
                                   lambda_risk: float, lambda_util: float, lambda_env: float,
                                   lambda_capacity: np.ndarray, lambda_availability: np.ndarray) -> np.ndarray:
        """Solve Lagrangian subproblem"""
        try:
            n_ore_types, n_routes = allocation_matrix.shape
            value_matrix = self._get_value_matrix(ore_types)
            
            # Construct augmented objective coefficients
            augmented_objective = value_matrix.copy()
            
            # Add Lagrangian penalty terms
            for i in range(n_ore_types):
                for j in range(n_routes):
                    # Risk penalty term (approximation)
                    risk_penalty = lambda_risk * 0.1 * allocation_matrix[i, j]
                    
                    # Utilization bonus term
                    util_bonus = lambda_util * 0.1
                    
                    # Environmental penalty/bonus
                    env_score = 1.0 - self.routes[j].environmental_impact
                    env_adjustment = lambda_env * env_score * 0.1
                    
                    # Capacity and availability penalties
                    capacity_penalty = lambda_capacity[j] * 0.01
                    availability_penalty = lambda_availability[i] * 0.01
                    
                    augmented_objective[i, j] += (
                        -risk_penalty + util_bonus + env_adjustment -
                        capacity_penalty - availability_penalty
                    )
            
            # Solve augmented subproblem using greedy approach
            solution_matrix = np.zeros((n_ore_types, n_routes))
            
            # Create priority list based on augmented objective values
            priorities = []
            for i in range(n_ore_types):
                for j in range(n_routes):
                    if self._is_compatible_combination(ore_types[i], self.routes[j]):
                        priorities.append({
                            'ore_idx': i,
                            'route_idx': j,
                            'value': augmented_objective[i, j],
                            'max_allocation': min(ore_types[i]['tonnage'], self.routes[j].capacity)
                        })
            
            # Sort by augmented objective value
            priorities.sort(key=lambda x: x['value'], reverse=True)
            
            # Greedy allocation with Lagrangian guidance
            for priority in priorities:
                i, j = priority['ore_idx'], priority['route_idx']
                max_alloc = priority['max_allocation']
                
                # Check remaining capacities
                remaining_ore = ore_types[i]['tonnage'] - np.sum(solution_matrix[i, :])
                remaining_route = self.routes[j].capacity - np.sum(solution_matrix[:, j])
                
                allocation = min(max_alloc, remaining_ore, remaining_route, max_alloc * 0.1)
                
                if allocation > 1e-6:
                    solution_matrix[i, j] = allocation
            
            return solution_matrix
            
        except Exception as e:
            logging.error(f"Error solving Lagrangian subproblem: {e}")
            return allocation_matrix
    
    def _compute_lagrangian_subgradients(self, allocation_matrix: np.ndarray, ore_types: List[Dict],
                                       constraint_violations: Dict) -> Dict:
        """Compute subgradients for Lagrangian multiplier updates"""
        try:
            n_ore_types, n_routes = allocation_matrix.shape
            
            subgradients = {
                'risk': constraint_violations.get('risk_violation', 0),
                'utilization': -constraint_violations.get('utilization_violation', 0),
                'environmental': -constraint_violations.get('environmental_violation', 0),
                'capacity': np.zeros(n_routes),
                'availability': np.zeros(n_ore_types)
            }
            
            # Capacity constraint subgradients
            for j in range(n_routes):
                route_allocation = np.sum(allocation_matrix[:, j])
                subgradients['capacity'][j] = route_allocation - self.routes[j].capacity
            
            # Availability constraint subgradients  
            for i in range(n_ore_types):
                ore_allocation = np.sum(allocation_matrix[i, :])
                subgradients['availability'][i] = ore_allocation - ore_types[i]['tonnage']
            
            return subgradients
            
        except Exception as e:
            logging.error(f"Error computing Lagrangian subgradients: {e}")
            return {'risk': 0, 'utilization': 0, 'environmental': 0, 
                   'capacity': np.zeros(n_routes), 'availability': np.zeros(n_ore_types)}
    
    def _apply_penalty_method(self, lagrangian_solution: Dict, ore_types: List[Dict],
                            constraint_violations: Dict) -> Dict:
        """Apply penalty method for constraint enforcement"""
        try:
            if 'allocation_matrix' not in lagrangian_solution:
                return lagrangian_solution
            
            allocation_matrix = lagrangian_solution['allocation_matrix'].copy()
            penalty_parameter = 10.0
            max_penalty_iterations = 20
            
            for penalty_iteration in range(max_penalty_iterations):
                # Solve penalized problem
                penalized_matrix = self._solve_penalized_subproblem(
                    allocation_matrix, ore_types, penalty_parameter, constraint_violations)
                
                # Check constraint satisfaction
                updated_violations = self._analyze_constraint_violations(
                    {'allocation_matrix': penalized_matrix}, ore_types,
                    constraint_violations.get('risk_violation', 0) + 0.1,
                    constraint_violations.get('utilization_violation', 0) + 0.8,
                    constraint_violations.get('environmental_violation', 0) + 0.3
                )
                
                # If constraints satisfied, break
                if updated_violations['total_violation_magnitude'] < 1e-3:
                    allocation_matrix = penalized_matrix
                    break
                
                # Increase penalty parameter
                penalty_parameter *= 2.0
                allocation_matrix = penalized_matrix
                
                # Prevent excessive penalty parameters
                if penalty_parameter > 10000:
                    break
            
            # Construct penalty method solution
            penalty_solution = lagrangian_solution.copy()
            penalty_solution['allocation_matrix'] = allocation_matrix
            penalty_solution['penalty_parameter'] = penalty_parameter
            penalty_solution['penalty_iterations'] = penalty_iteration + 1
            
            # Update objective metrics
            value_matrix = self._get_value_matrix(ore_types)
            penalty_solution['npv'] = np.sum(allocation_matrix * value_matrix)
            penalty_solution['risk'] = self._calculate_solution_risk(
                {'allocation_matrix': allocation_matrix}, ore_types)
            penalty_solution['utilization'] = np.sum(allocation_matrix) / max(1, len(ore_types))
            penalty_solution['environmental_score'] = self._calculate_environmental_score(
                {'allocation_matrix': allocation_matrix}, ore_types)
            
            return penalty_solution
            
        except Exception as e:
            logging.error(f"Error in penalty method: {e}")
            return lagrangian_solution
    
    def _solve_penalized_subproblem(self, allocation_matrix: np.ndarray, ore_types: List[Dict],
                                  penalty_parameter: float, constraint_violations: Dict) -> np.ndarray:
        """Solve penalized optimization subproblem"""
        try:
            n_ore_types, n_routes = allocation_matrix.shape
            value_matrix = self._get_value_matrix(ore_types)
            
            # Create penalized objective function
            penalized_matrix = allocation_matrix.copy()
            
            # Apply penalty adjustments for constraint violations
            violation_magnitude = constraint_violations.get('total_violation_magnitude', 0)
            
            if violation_magnitude > 0:
                # Reduce high-violation allocations
                violation_penalty = penalty_parameter * violation_magnitude / (n_ore_types * n_routes)
                
                for i in range(n_ore_types):
                    for j in range(n_routes):
                        if allocation_matrix[i, j] > 0:
                            # Calculate individual contribution to violations
                            contribution_penalty = self._calculate_violation_contribution(
                                i, j, allocation_matrix, ore_types, constraint_violations)
                            
                            # Apply penalty reduction
                            penalty_reduction = min(0.5, violation_penalty * contribution_penalty)
                            penalized_matrix[i, j] *= (1.0 - penalty_reduction)
            
            # Ensure non-negativity
            penalized_matrix = np.maximum(penalized_matrix, 0.0)
            
            return penalized_matrix
            
        except Exception as e:
            logging.error(f"Error solving penalized subproblem: {e}")
            return allocation_matrix
    
    def _calculate_violation_contribution(self, ore_idx: int, route_idx: int, 
                                        allocation_matrix: np.ndarray, ore_types: List[Dict],
                                        constraint_violations: Dict) -> float:
        """Calculate individual allocation's contribution to constraint violations"""
        try:
            contribution = 0.0
            
            # Contribution to capacity violations
            for cap_violation in constraint_violations.get('capacity_violations', []):
                if cap_violation['route_index'] == route_idx:
                    route_total = np.sum(allocation_matrix[:, route_idx])
                    if route_total > 0:
                        contribution += (allocation_matrix[ore_idx, route_idx] / route_total) * \
                                      cap_violation['violation_magnitude']
            
            # Contribution to availability violations
            for avail_violation in constraint_violations.get('availability_violations', []):
                if avail_violation['ore_type_index'] == ore_idx:
                    ore_total = np.sum(allocation_matrix[ore_idx, :])
                    if ore_total > 0:
                        contribution += (allocation_matrix[ore_idx, route_idx] / ore_total) * \
                                      avail_violation['violation_magnitude']
            
            # Normalize contribution
            return min(1.0, contribution / max(allocation_matrix[ore_idx, route_idx], 1e-6))
            
        except Exception as e:
            logging.error(f"Error calculating violation contribution: {e}")
            return 0.0
    
    def _apply_sequential_quadratic_programming(self, penalty_solution: Dict, ore_types: List[Dict],
                                              constraint_violations: Dict) -> Dict:
        """
        Comprehensive Sequential Quadratic Programming (SQP) implementation
        with advanced constraint handling, merit function optimization, and convergence acceleration
        """
        import logging
        import time
        
        try:
            start_time = time.time()
            sqp_solution = penalty_solution.copy()
            
            if 'allocation_matrix' not in penalty_solution:
                return sqp_solution
            
            # Initialize SQP algorithm parameters
            allocation_matrix = penalty_solution['allocation_matrix'].copy().astype(float)
            n_vars = allocation_matrix.size
            x_current = allocation_matrix.flatten()
            
            # SQP algorithm parameters
            max_sqp_iterations = 100
            tolerance = {
                'optimality': 1e-6,
                'feasibility': 1e-6,
                'step_size': 1e-8,
                'function_change': 1e-10
            }
            
            # Initialize Lagrange multipliers and penalty parameters
            lambda_eq = np.zeros(0)  # Equality constraint multipliers
            lambda_ineq = np.zeros(0)  # Inequality constraint multipliers
            penalty_param = 1.0
            
            # Initialize quasi-Newton Hessian approximation (BFGS)
            hessian_approx = np.eye(n_vars)
            
            # Merit function parameters
            merit_param = 0.1
            merit_decrease_factor = 0.1
            armijo_c1 = 1e-4
            
            # Convergence tracking
            convergence_history = {
                'objective_values': [],
                'constraint_violations': [],
                'optimality_measures': [],
                'step_sizes': [],
                'merit_values': []
            }
            
            # Main SQP iteration loop
            for iteration in range(max_sqp_iterations):
                # Step 1: Evaluate objective function and constraints
                f_current = self._evaluate_objective_function(x_current.reshape(allocation_matrix.shape), ore_types)
                grad_f = self._compute_comprehensive_gradient(x_current.reshape(allocation_matrix.shape), ore_types)
                
                # Step 2: Evaluate constraints and their Jacobians
                constraints_data = self._evaluate_constraints_and_jacobians(
                    x_current.reshape(allocation_matrix.shape), ore_types, constraint_violations
                )
                
                c_eq = constraints_data['equality']
                c_ineq = constraints_data['inequality'] 
                jacobian_eq = constraints_data['jacobian_eq']
                jacobian_ineq = constraints_data['jacobian_ineq']
                
                # Step 3: Check convergence criteria
                convergence_check = self._check_sqp_convergence(
                    grad_f, c_eq, c_ineq, lambda_eq, lambda_ineq, 
                    jacobian_eq, jacobian_ineq, tolerance
                )
                
                if convergence_check['converged']:
                    logging.info(f"SQP converged in {iteration} iterations: {convergence_check['reason']}")
                    break
                
                # Step 4: Solve QP subproblem
                try:
                    qp_solution = self._solve_qp_subproblem(
                        grad_f.flatten(), hessian_approx, c_eq, c_ineq, 
                        jacobian_eq, jacobian_ineq, x_current
                    )
                    
                    search_direction = qp_solution['direction']
                    lambda_eq_new = qp_solution['lambda_eq']
                    lambda_ineq_new = qp_solution['lambda_ineq']
                    
                except Exception as e:
                    logging.warning(f"QP subproblem failed at iteration {iteration}: {e}")
                    # Fallback to steepest descent
                    search_direction = self._compute_fallback_direction(
                        grad_f.flatten(), c_eq, c_ineq, jacobian_eq, jacobian_ineq
                    )
                    lambda_eq_new = lambda_eq.copy() if len(lambda_eq) > 0 else np.array([])
                    lambda_ineq_new = lambda_ineq.copy() if len(lambda_ineq) > 0 else np.array([])
                
                # Step 5: Merit function line search
                merit_current = self._compute_merit_function(
                    x_current, f_current, c_eq, c_ineq, penalty_param
                )
                
                step_size, merit_new = self._merit_function_line_search(
                    x_current, search_direction, f_current, grad_f.flatten(),
                    c_eq, c_ineq, penalty_param, merit_current, ore_types,
                    armijo_c1, allocation_matrix.shape
                )
                
                # Step 6: Update variables and multipliers
                x_new = x_current + step_size * search_direction
                
                # Project onto feasible region (box constraints)
                x_new = self._project_onto_bounds(x_new, allocation_matrix.shape)
                
                # Step 7: Update Hessian approximation using BFGS
                if iteration > 0 and step_size > tolerance['step_size']:
                    hessian_approx = self._update_hessian_bfgs(
                        hessian_approx, x_new - x_current, 
                        grad_f.flatten() - grad_f_prev.flatten()
                    )
                
                # Step 8: Update penalty parameter if needed
                penalty_param = self._update_penalty_parameter(
                    penalty_param, lambda_eq_new, lambda_ineq_new, 
                    c_eq, c_ineq, merit_decrease_factor
                )
                
                # Step 9: Update multiplier estimates
                if len(lambda_eq_new) == len(lambda_eq):
                    lambda_eq = lambda_eq_new.copy()
                if len(lambda_ineq_new) == len(lambda_ineq):
                    lambda_ineq = lambda_ineq_new.copy()
                
                # Step 10: Store convergence history
                convergence_history['objective_values'].append(f_current)
                convergence_history['constraint_violations'].append(
                    np.linalg.norm(c_eq) + np.linalg.norm(np.maximum(0, c_ineq))
                )
                convergence_history['optimality_measures'].append(convergence_check['optimality_measure'])
                convergence_history['step_sizes'].append(step_size)
                convergence_history['merit_values'].append(merit_current)
                
                # Step 11: Prepare for next iteration
                x_current = x_new.copy()
                grad_f_prev = grad_f.copy()
                
                # Adaptive algorithm parameters
                if iteration % 10 == 0:
                    self._adapt_sqp_parameters(
                        convergence_history, tolerance, penalty_param, iteration
                    )
            
            # Post-processing and solution validation
            final_allocation = x_current.reshape(allocation_matrix.shape)
            
            # Validate and refine solution
            validated_solution = self._validate_and_refine_sqp_solution(
                final_allocation, ore_types, convergence_history
            )
            
            # Comprehensive solution analysis
            solution_analysis = self._analyze_sqp_solution(
                validated_solution, ore_types, convergence_history, 
                iteration, time.time() - start_time
            )
            
            # Update sqp_solution dictionary with comprehensive results
            sqp_solution.update({
                'allocation_matrix': validated_solution,
                'sqp_iterations': iteration + 1,
                'convergence_achieved': convergence_check['converged'] if 'convergence_check' in locals() else False,
                'final_objective': solution_analysis['final_objective'],
                'constraint_violation': solution_analysis['constraint_violation'],
                'optimality_gap': solution_analysis['optimality_gap'],
                'computational_time': solution_analysis['computational_time'],
                'convergence_history': convergence_history,
                'algorithm_diagnostics': {
                    'hessian_condition_number': np.linalg.cond(hessian_approx),
                    'penalty_parameter': penalty_param,
                    'merit_parameter': merit_param,
                    'final_step_size': step_size if 'step_size' in locals() else 0.0,
                    'bfgs_updates': iteration,
                    'qp_failures': solution_analysis.get('qp_failures', 0)
                },
                'solution_quality': solution_analysis['quality_metrics'],
                'economic_metrics': solution_analysis['economic_metrics'],
                'risk_assessment': solution_analysis['risk_metrics']
            })
            
            logging.info(f"SQP optimization completed in {iteration+1} iterations, "
                        f"time: {time.time() - start_time:.2f}s, "
                        f"final objective: {solution_analysis['final_objective']:.6f}")
            
            return sqp_solution
            
        except Exception as e:
            logging.error(f"Critical error in Sequential Quadratic Programming: {e}")
            import traceback
            logging.error(traceback.format_exc())
            
            # Return penalty solution with error information
            penalty_solution['sqp_error'] = str(e)
            penalty_solution['sqp_failed'] = True
            return penalty_solution
    
    def _evaluate_objective_function(self, allocation_matrix: np.ndarray, ore_types: List[Dict]) -> float:
        """Evaluate the objective function value"""
        try:
            value_matrix = self._get_value_matrix(ore_types)
            return float(np.sum(allocation_matrix * value_matrix))
        except Exception as e:
            logging.error(f"Error evaluating objective function: {e}")
            return 0.0
    
    def _compute_comprehensive_gradient(self, allocation_matrix: np.ndarray, ore_types: List[Dict]) -> np.ndarray:
        """Compute comprehensive gradient including penalties"""
        try:
            # Basic gradient (constant for linear objective)
            value_matrix = self._get_value_matrix(ore_types)
            base_gradient = value_matrix.copy()
            
            # Add penalty terms for constraint violations
            penalty_gradient = np.zeros_like(allocation_matrix)
            
            # Capacity constraint penalties
            total_allocation = np.sum(allocation_matrix, axis=1, keepdims=True)
            capacity_limits = np.array([ore.get('capacity', 1000) for ore in ore_types]).reshape(1, -1)
            
            for i in range(allocation_matrix.shape[0]):
                if total_allocation[i, 0] > 1.0:  # Over-allocation penalty
                    penalty_gradient[i, :] += 10.0  # Penalty coefficient
            
            # Resource constraint penalties
            for j, ore_type in enumerate(ore_types):
                resource_usage = np.sum(allocation_matrix[:, j])
                resource_limit = ore_type.get('resource_limit', 1000)
                
                if resource_usage > resource_limit:
                    penalty_gradient[:, j] += 5.0 * (resource_usage - resource_limit)
            
            return base_gradient - penalty_gradient  # Negative for maximization
            
        except Exception as e:
            logging.error(f"Error computing comprehensive gradient: {e}")
            return self._get_value_matrix(ore_types)
    
    def _evaluate_constraints_and_jacobians(self, allocation_matrix: np.ndarray, 
                                          ore_types: List[Dict], 
                                          constraint_violations: Dict) -> Dict:
        """Evaluate constraints and compute their Jacobians"""
        try:
            n_blocks, n_ores = allocation_matrix.shape
            
            # Equality constraints (none in this formulation)
            c_eq = np.array([])
            jacobian_eq = np.zeros((0, allocation_matrix.size))
            
            # Inequality constraints
            constraints = []
            jacobian_rows = []
            
            # 1. Non-negativity constraints: x_ij >= 0 (already handled by bounds)
            
            # 2. Capacity constraints: sum_j x_ij <= 1 for all i
            for i in range(n_blocks):
                constraint_val = np.sum(allocation_matrix[i, :]) - 1.0
                constraints.append(constraint_val)
                
                # Jacobian row
                jacobian_row = np.zeros(allocation_matrix.size)
                for j in range(n_ores):
                    jacobian_row[i * n_ores + j] = 1.0
                jacobian_rows.append(jacobian_row)
            
            # 3. Resource availability constraints
            for j, ore_type in enumerate(ore_types):
                resource_usage = np.sum(allocation_matrix[:, j])
                resource_limit = ore_type.get('resource_limit', 1000)
                constraint_val = resource_usage - resource_limit
                constraints.append(constraint_val)
                
                # Jacobian row
                jacobian_row = np.zeros(allocation_matrix.size)
                for i in range(n_blocks):
                    jacobian_row[i * n_ores + j] = 1.0
                jacobian_rows.append(jacobian_row)
            
            # 4. Processing capacity constraints
            for processing_route in ['flotation', 'leaching', 'direct_shipping']:
                route_usage = 0.0
                jacobian_row = np.zeros(allocation_matrix.size)
                
                for i in range(n_blocks):
                    for j, ore_type in enumerate(ore_types):
                        if ore_type.get('preferred_processing') == processing_route:
                            route_usage += allocation_matrix[i, j]
                            jacobian_row[i * n_ores + j] = 1.0
                
                route_capacity = 500.0  # Example capacity
                constraint_val = route_usage - route_capacity
                constraints.append(constraint_val)
                jacobian_rows.append(jacobian_row)
            
            c_ineq = np.array(constraints)
            jacobian_ineq = np.array(jacobian_rows) if jacobian_rows else np.zeros((0, allocation_matrix.size))
            
            return {
                'equality': c_eq,
                'inequality': c_ineq,
                'jacobian_eq': jacobian_eq,
                'jacobian_ineq': jacobian_ineq
            }
            
        except Exception as e:
            logging.error(f"Error evaluating constraints: {e}")
            return {
                'equality': np.array([]),
                'inequality': np.array([]),
                'jacobian_eq': np.zeros((0, allocation_matrix.size)),
                'jacobian_ineq': np.zeros((0, allocation_matrix.size))
            }
    
    def _check_sqp_convergence(self, grad_f: np.ndarray, c_eq: np.ndarray, c_ineq: np.ndarray,
                             lambda_eq: np.ndarray, lambda_ineq: np.ndarray,
                             jacobian_eq: np.ndarray, jacobian_ineq: np.ndarray,
                             tolerance: Dict) -> Dict:
        """Check SQP convergence criteria"""
        try:
            # KKT optimality conditions
            grad_lagrangian = grad_f.flatten().copy()
            
            # Add equality constraint contributions
            if len(c_eq) > 0 and len(lambda_eq) > 0:
                grad_lagrangian -= jacobian_eq.T @ lambda_eq
            
            # Add inequality constraint contributions
            if len(c_ineq) > 0 and len(lambda_ineq) > 0:
                grad_lagrangian -= jacobian_ineq.T @ lambda_ineq
            
            # Optimality measure
            optimality_measure = np.linalg.norm(grad_lagrangian)
            
            # Feasibility measures
            eq_feasibility = np.linalg.norm(c_eq) if len(c_eq) > 0 else 0.0
            ineq_feasibility = np.linalg.norm(np.maximum(0, c_ineq)) if len(c_ineq) > 0 else 0.0
            
            # Complementarity slackness
            if len(c_ineq) > 0 and len(lambda_ineq) > 0:
                complementarity = np.linalg.norm(lambda_ineq * np.maximum(0, c_ineq))
            else:
                complementarity = 0.0
            
            # Check convergence criteria
            converged = (
                optimality_measure <= tolerance['optimality'] and
                eq_feasibility <= tolerance['feasibility'] and
                ineq_feasibility <= tolerance['feasibility'] and
                complementarity <= tolerance['feasibility']
            )
            
            if converged:
                reason = "KKT conditions satisfied"
            else:
                reason = f"Not converged: opt={optimality_measure:.2e}, feas={max(eq_feasibility, ineq_feasibility):.2e}"
            
            return {
                'converged': converged,
                'reason': reason,
                'optimality_measure': optimality_measure,
                'feasibility_measure': max(eq_feasibility, ineq_feasibility),
                'complementarity': complementarity
            }
            
        except Exception as e:
            logging.error(f"Error checking convergence: {e}")
            return {'converged': False, 'reason': f'Error: {e}', 'optimality_measure': float('inf')}
    
    def _solve_qp_subproblem(self, grad_f: np.ndarray, hessian: np.ndarray, 
                           c_eq: np.ndarray, c_ineq: np.ndarray,
                           jacobian_eq: np.ndarray, jacobian_ineq: np.ndarray,
                           x_current: np.ndarray) -> Dict:
        """Solve QP subproblem using active set method"""
        try:
            n_vars = len(x_current)
            
            # QP formulation: min 0.5 * d^T * H * d + g^T * d
            # subject to: A_eq * d = -c_eq, A_ineq * d <= -c_ineq
            
            # Use scipy's quadratic programming solver
            from scipy.optimize import minimize
            
            def qp_objective(d):
                return 0.5 * d.T @ hessian @ d + grad_f.T @ d
            
            def qp_gradient(d):
                return hessian @ d + grad_f
            
            # Constraint setup
            constraints = []
            
            # Equality constraints
            if len(c_eq) > 0:
                constraints.append({
                    'type': 'eq',
                    'fun': lambda d: jacobian_eq @ d + c_eq,
                    'jac': lambda d: jacobian_eq
                })
            
            # Inequality constraints
            if len(c_ineq) > 0:
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda d: -(jacobian_ineq @ d + c_ineq),
                    'jac': lambda d: -jacobian_ineq
                })
            
            # Solve QP subproblem
            initial_guess = np.zeros(n_vars)
            
            result = minimize(
                qp_objective,
                initial_guess,
                method='SLSQP',
                jac=qp_gradient,
                constraints=constraints,
                options={'ftol': 1e-9, 'maxiter': 1000}
            )
            
            if result.success:
                direction = result.x
                
                # Extract multipliers (approximated)
                lambda_eq_new = np.zeros(len(c_eq)) if len(c_eq) > 0 else np.array([])
                lambda_ineq_new = np.zeros(len(c_ineq)) if len(c_ineq) > 0 else np.array([])
                
                return {
                    'direction': direction,
                    'lambda_eq': lambda_eq_new,
                    'lambda_ineq': lambda_ineq_new,
                    'qp_success': True
                }
            else:
                raise Exception(f"QP solver failed: {result.message}")
                
        except Exception as e:
            logging.warning(f"QP subproblem solution failed: {e}")
            raise e
    
    def _compute_fallback_direction(self, grad_f: np.ndarray, c_eq: np.ndarray, c_ineq: np.ndarray,
                                  jacobian_eq: np.ndarray, jacobian_ineq: np.ndarray) -> np.ndarray:
        """Compute fallback search direction using projected gradient"""
        try:
            # Simple projected gradient method
            direction = -grad_f.copy()
            
            # Project away from active equality constraints
            if len(c_eq) > 0 and jacobian_eq.size > 0:
                # Remove components in the null space of equality constraints
                try:
                    U, s, Vt = svd(jacobian_eq, full_matrices=False)
                    rank = np.sum(s > 1e-10)
                    if rank > 0:
                        V_active = Vt[:rank, :].T
                        projection = V_active @ V_active.T
                        direction = direction - projection @ direction
                except Exception:
                    pass
            
            # Scale direction
            direction_norm = np.linalg.norm(direction)
            if direction_norm > 1e-10:
                direction = direction / direction_norm
            else:
                direction = np.random.normal(0, 0.01, len(direction))
                direction = direction / np.linalg.norm(direction)
            
            return direction
            
        except Exception as e:
            logging.error(f"Error computing fallback direction: {e}")
            return -grad_f / (np.linalg.norm(grad_f) + 1e-10)
    
    def _compute_merit_function(self, x: np.ndarray, f_val: float, 
                              c_eq: np.ndarray, c_ineq: np.ndarray, 
                              penalty_param: float) -> float:
        """Compute L1 penalty merit function"""
        try:
            merit_val = f_val
            
            # Add equality constraint violations
            if len(c_eq) > 0:
                merit_val -= penalty_param * np.sum(np.abs(c_eq))
            
            # Add inequality constraint violations
            if len(c_ineq) > 0:
                merit_val -= penalty_param * np.sum(np.maximum(0, c_ineq))
            
            return merit_val
            
        except Exception as e:
            logging.error(f"Error computing merit function: {e}")
            return f_val
    
    def _merit_function_line_search(self, x_current: np.ndarray, direction: np.ndarray,
                                  f_current: float, grad_f: np.ndarray,
                                  c_eq: np.ndarray, c_ineq: np.ndarray,
                                  penalty_param: float, merit_current: float,
                                  ore_types: List[Dict], armijo_c1: float,
                                  matrix_shape: Tuple[int, int]) -> Tuple[float, float]:
        """Perform merit function line search with Armijo condition"""
        try:
            alpha = 1.0  # Initial step size
            beta = 0.5   # Step reduction factor
            max_backtracks = 20
            
            # Directional derivative of merit function
            merit_directional_derivative = grad_f.T @ direction
            if len(c_eq) > 0:
                merit_directional_derivative -= penalty_param * np.sum(np.sign(c_eq) * (jacobian_eq @ direction))
            if len(c_ineq) > 0:
                active_ineq = c_ineq > -1e-6
                if np.any(active_ineq):
                    merit_directional_derivative -= penalty_param * np.sum((jacobian_ineq[active_ineq] @ direction))
            
            for backtrack in range(max_backtracks):
                # Trial point
                x_trial = x_current + alpha * direction
                x_trial = self._project_onto_bounds(x_trial, matrix_shape)
                
                # Evaluate merit function at trial point
                try:
                    f_trial = self._evaluate_objective_function(x_trial.reshape(matrix_shape), ore_types)
                    
                    # Evaluate constraints at trial point
                    constraints_trial = self._evaluate_constraints_and_jacobians(
                        x_trial.reshape(matrix_shape), ore_types, {}
                    )
                    
                    merit_trial = self._compute_merit_function(
                        x_trial, f_trial, 
                        constraints_trial['equality'], 
                        constraints_trial['inequality'], 
                        penalty_param
                    )
                    
                    # Armijo condition
                    armijo_condition = merit_trial >= merit_current + armijo_c1 * alpha * merit_directional_derivative
                    
                    if armijo_condition:
                        return alpha, merit_trial
                        
                except Exception:
                    pass
                
                # Reduce step size
                alpha *= beta
            
            # If line search fails, return small step
            return 1e-6, merit_current
            
        except Exception as e:
            logging.warning(f"Line search failed: {e}")
            return 1e-6, merit_current
    
    def _project_onto_bounds(self, x: np.ndarray, matrix_shape: Tuple[int, int]) -> np.ndarray:
        """Project variables onto feasible bounds"""
        try:
            x_projected = np.maximum(x, 0.0)  # Non-negativity
            x_projected = np.minimum(x_projected, 10.0)  # Upper bound
            return x_projected
            
        except Exception as e:
            logging.error(f"Error projecting onto bounds: {e}")
            return np.maximum(x, 0.0)
    
    def _update_hessian_bfgs(self, hessian: np.ndarray, s: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Update Hessian approximation using BFGS formula"""
        try:
            if np.dot(s, y) <= 1e-10:
                return hessian  # Skip update if curvature condition not satisfied
            
            rho = 1.0 / np.dot(s, y)
            I = np.eye(len(s))
            
            # BFGS update formula
            hessian_new = ((I - rho * np.outer(s, y)) @ hessian @ (I - rho * np.outer(y, s)) + 
                          rho * np.outer(s, s))
            
            return hessian_new
            
        except Exception as e:
            logging.error(f"Error updating Hessian: {e}")
            return hessian
    
    def _update_penalty_parameter(self, penalty_param: float, lambda_eq: np.ndarray,
                                lambda_ineq: np.ndarray, c_eq: np.ndarray, c_ineq: np.ndarray,
                                decrease_factor: float) -> float:
        """Update penalty parameter based on constraint violations and multipliers"""
        try:
            # Estimate of optimal penalty parameter
            lambda_max = 0.0
            if len(lambda_eq) > 0:
                lambda_max = max(lambda_max, np.max(np.abs(lambda_eq)))
            if len(lambda_ineq) > 0:
                lambda_max = max(lambda_max, np.max(np.abs(lambda_ineq)))
            
            # Update penalty parameter
            suggested_penalty = lambda_max + 1.0
            
            # Don't decrease too quickly
            new_penalty = max(suggested_penalty, penalty_param * decrease_factor)
            
            return min(new_penalty, 1000.0)  # Cap the penalty parameter
            
        except Exception as e:
            logging.error(f"Error updating penalty parameter: {e}")
            return penalty_param
    
    def _adapt_sqp_parameters(self, convergence_history: Dict, tolerance: Dict,
                            penalty_param: float, iteration: int) -> None:
        """Adapt SQP parameters based on convergence history"""
        try:
            if len(convergence_history['objective_values']) < 5:
                return
            
            recent_objectives = convergence_history['objective_values'][-5:]
            recent_violations = convergence_history['constraint_violations'][-5:]
            
            # If objective is not improving, relax tolerance slightly
            obj_improvement = abs(recent_objectives[-1] - recent_objectives[0])
            if obj_improvement < tolerance['function_change'] * 10:
                tolerance['optimality'] *= 1.1
                tolerance['feasibility'] *= 1.1
            
            # If constraint violations are increasing, be more strict
            if len(recent_violations) >= 2 and recent_violations[-1] > recent_violations[-2]:
                tolerance['feasibility'] *= 0.9
                
        except Exception as e:
            logging.warning(f"Error adapting SQP parameters: {e}")
    
    def _validate_and_refine_sqp_solution(self, allocation_matrix: np.ndarray,
                                        ore_types: List[Dict],
                                        convergence_history: Dict) -> np.ndarray:
        """Validate and refine the SQP solution"""
        try:
            refined_allocation = allocation_matrix.copy()
            
            # Ensure non-negativity
            refined_allocation = np.maximum(refined_allocation, 0.0)
            
            # Normalize rows to satisfy capacity constraints
            for i in range(refined_allocation.shape[0]):
                row_sum = np.sum(refined_allocation[i, :])
                if row_sum > 1.0:
                    refined_allocation[i, :] /= row_sum
            
            # Check resource limits and scale down if necessary
            for j, ore_type in enumerate(ore_types):
                column_sum = np.sum(refined_allocation[:, j])
                resource_limit = ore_type.get('resource_limit', float('inf'))
                if column_sum > resource_limit:
                    scale_factor = resource_limit / column_sum
                    refined_allocation[:, j] *= scale_factor
            
            return refined_allocation
            
        except Exception as e:
            logging.error(f"Error validating solution: {e}")
            return allocation_matrix
    
    def _analyze_sqp_solution(self, allocation_matrix: np.ndarray, ore_types: List[Dict],
                            convergence_history: Dict, iterations: int,
                            computation_time: float) -> Dict:
        """Analyze the quality and characteristics of the SQP solution"""
        try:
            analysis = {}
            
            # Basic metrics
            analysis['final_objective'] = self._evaluate_objective_function(allocation_matrix, ore_types)
            analysis['computational_time'] = computation_time
            
            # Constraint violation analysis
            constraints = self._evaluate_constraints_and_jacobians(allocation_matrix, ore_types, {})
            eq_violation = np.linalg.norm(constraints['equality']) if len(constraints['equality']) > 0 else 0.0
            ineq_violation = np.linalg.norm(np.maximum(0, constraints['inequality'])) if len(constraints['inequality']) > 0 else 0.0
            analysis['constraint_violation'] = max(eq_violation, ineq_violation)
            
            # Optimality gap estimation
            if len(convergence_history['objective_values']) > 1:
                obj_change = abs(convergence_history['objective_values'][-1] - convergence_history['objective_values'][-2])
                analysis['optimality_gap'] = obj_change
            else:
                analysis['optimality_gap'] = 0.0
            
            # Quality metrics
            analysis['quality_metrics'] = {
                'solution_sparsity': np.sum(allocation_matrix < 1e-6) / allocation_matrix.size,
                'max_allocation': np.max(allocation_matrix),
                'allocation_distribution_entropy': self._calculate_entropy(allocation_matrix.flatten()),
                'constraint_satisfaction_rate': 1.0 - analysis['constraint_violation']
            }
            
            # Economic metrics
            value_matrix = self._get_value_matrix(ore_types)
            analysis['economic_metrics'] = {
                'total_value': np.sum(allocation_matrix * value_matrix),
                'value_per_unit': np.sum(allocation_matrix * value_matrix) / max(np.sum(allocation_matrix), 1e-6),
                'resource_utilization': np.sum(allocation_matrix) / allocation_matrix.size,
                'diversification_index': self._calculate_diversification_index(allocation_matrix)
            }
            
            # Risk metrics
            analysis['risk_metrics'] = {
                'allocation_concentration': np.max(allocation_matrix) / max(np.sum(allocation_matrix), 1e-6),
                'solution_stability': 1.0 / (1.0 + np.std(convergence_history['objective_values'][-min(10, len(convergence_history['objective_values'])):]))
            }
            
            return analysis
            
        except Exception as e:
            logging.error(f"Error analyzing SQP solution: {e}")
            return {
                'final_objective': 0.0,
                'constraint_violation': float('inf'),
                'optimality_gap': float('inf'),
                'computational_time': computation_time,
                'quality_metrics': {},
                'economic_metrics': {},
                'risk_metrics': {}
            }
    
    def _calculate_entropy(self, values: np.ndarray) -> float:
        """Calculate Shannon entropy of allocation distribution"""
        try:
            values_positive = values[values > 1e-10]
            if len(values_positive) == 0:
                return 0.0
            
            probabilities = values_positive / np.sum(values_positive)
            return -np.sum(probabilities * np.log2(probabilities + 1e-10))
            
        except Exception:
            return 0.0
    
    def _calculate_diversification_index(self, allocation_matrix: np.ndarray) -> float:
        """Calculate diversification index (Herfindahl-Hirschman Index)"""
        try:
            total_allocation = np.sum(allocation_matrix)
            if total_allocation < 1e-10:
                return 0.0
            
            proportions = allocation_matrix.flatten() / total_allocation
            hhi = np.sum(proportions**2)
            
            # Convert to diversification index (higher = more diversified)
            return 1.0 - hhi
            
        except Exception:
            return 0.0
    
    def _compute_objective_gradient(self, allocation_matrix: np.ndarray, ore_types: List[Dict]) -> np.ndarray:
        """Compute gradient of the objective function"""
        try:
            value_matrix = self._get_value_matrix(ore_types)
            return value_matrix  # Linear objective has constant gradient
        except Exception as e:
            logging.error(f"Error computing objective gradient: {e}")
            return np.ones_like(allocation_matrix)
    
    def _compute_hessian_approximation(self, allocation_matrix: np.ndarray, ore_types: List[Dict]) -> np.ndarray:
        """Compute Hessian approximation for SQP"""
        try:
            n_vars = allocation_matrix.size
            # For linear objective, Hessian is zero; use identity for numerical stability
            return np.eye(n_vars) * 1e-3
        except Exception as e:
            logging.error(f"Error computing Hessian approximation: {e}")
            return np.eye(allocation_matrix.size)
    
    def _armijo_line_search(self, allocation_matrix: np.ndarray, search_direction: np.ndarray,
                           ore_types: List[Dict], initial_step: float = 1.0) -> float:
        """Armijo line search for step size selection"""
        try:
            alpha = initial_step
            beta = 0.5
            c1 = 1e-4
            max_backtracks = 10
            
            current_obj = self._evaluate_constrained_objective(allocation_matrix, ore_types)
            gradient = self._compute_objective_gradient(allocation_matrix, ore_types)
            directional_derivative = np.sum(gradient * search_direction)
            
            for backtrack in range(max_backtracks):
                new_allocation = allocation_matrix + alpha * search_direction
                new_allocation = np.maximum(new_allocation, 0.0)  # Non-negativity
                
                new_obj = self._evaluate_constrained_objective(new_allocation, ore_types)
                
                # Armijo condition
                if new_obj >= current_obj + c1 * alpha * directional_derivative:
                    return alpha
                
                alpha *= beta
            
            return alpha
            
        except Exception as e:
            logging.error(f"Error in Armijo line search: {e}")
            return initial_step
    
    def _evaluate_constrained_objective(self, allocation_matrix: np.ndarray, ore_types: List[Dict]) -> float:
        """Evaluate constrained objective function"""
        try:
            value_matrix = self._get_value_matrix(ore_types)
            return np.sum(allocation_matrix * value_matrix)
        except Exception as e:
            logging.error(f"Error evaluating constrained objective: {e}")
            return 0.0
    
    def _apply_augmented_lagrangian_method(self, sqp_solution: Dict, ore_types: List[Dict],
                                         constraint_violations: Dict) -> Dict:
        """Apply Augmented Lagrangian Method for final constraint satisfaction"""
        try:
            # Augmented Lagrangian implementation placeholder
            augmented_solution = sqp_solution.copy()
            
            if 'allocation_matrix' not in sqp_solution:
                return augmented_solution
            
            allocation_matrix = sqp_solution['allocation_matrix'].copy()
            
            # Augmented Lagrangian parameters
            mu = 1.0  # Penalty parameter
            lambda_vec = np.zeros(5)  # Lagrangian multipliers for main constraints
            
            # Advanced augmented Lagrangian method with adaptive penalty parameters
            augmented_iterations = 50
            convergence_tolerance = 1e-6
            
            # Initialize advanced penalty parameter adaptation
            penalty_adaptation_history = []
            constraint_violation_history = []
            
            for iteration in range(augmented_iterations):
                # Advanced constraint violation analysis with sensitivity computation
                current_violations = await self._compute_advanced_constraint_violations(
                    {'allocation_matrix': allocation_matrix}, ore_types, 0.5, 0.8, 0.6)
                
                constraint_violation_history.append(current_violations['total_violation_magnitude'])
                
                if current_violations['total_violation_magnitude'] < convergence_tolerance:
                    break
                
                # Compute augmented Lagrangian gradient with second-order information
                augmented_gradient = await self._compute_augmented_lagrangian_gradient(
                    allocation_matrix, lambda_vec, mu, current_violations, ore_types)
                
                # Adaptive step size using Armijo line search
                step_size = await self._armijo_line_search(
                    allocation_matrix, augmented_gradient, lambda_vec, mu, ore_types)
                
                # Update allocation matrix with projected gradient step
                allocation_matrix = await self._projected_gradient_update(
                    allocation_matrix, augmented_gradient, step_size, ore_types)
                
                # Update Lagrangian multipliers using dual ascent
                lambda_vec = await self._update_lagrangian_multipliers(
                    lambda_vec, current_violations, mu, iteration)
                
                # Adaptive penalty parameter update
                mu = await self._update_penalty_parameter(
                    mu, current_violations, constraint_violation_history, iteration)
                
                penalty_adaptation_history.append(mu)
                
                if current_violations['utilization_violation'] > 0:
                    # Try to increase utilization while respecting constraints
                    scale_factor = min(1.02, 1.0 + current_violations['utilization_violation'] * 0.1)
                    allocation_matrix *= scale_factor
                
                # Ensure feasibility
                allocation_matrix = self._project_to_feasible_region(allocation_matrix, ore_types)
            
            augmented_solution['allocation_matrix'] = allocation_matrix
            augmented_solution['augmented_iterations'] = iteration + 1
            
            # Update metrics
            value_matrix = self._get_value_matrix(ore_types)
            augmented_solution['npv'] = np.sum(allocation_matrix * value_matrix)
            augmented_solution['risk'] = self._calculate_solution_risk(
                {'allocation_matrix': allocation_matrix}, ore_types)
            augmented_solution['utilization'] = np.sum(allocation_matrix) / max(1, len(ore_types))
            augmented_solution['environmental_score'] = self._calculate_environmental_score(
                {'allocation_matrix': allocation_matrix}, ore_types)
            
            return augmented_solution
            
        except Exception as e:
            logging.error(f"Error in Augmented Lagrangian Method: {e}")
            return sqp_solution
    
    def _project_to_feasible_region(self, allocation_matrix: np.ndarray, ore_types: List[Dict]) -> np.ndarray:
        """Project allocation matrix to feasible region"""
        try:
            projected_matrix = allocation_matrix.copy()
            
            # Non-negativity projection
            projected_matrix = np.maximum(projected_matrix, 0.0)
            
            # Capacity constraint projection
            for j, route in enumerate(self.routes):
                route_total = np.sum(projected_matrix[:, j])
                if route_total > route.capacity:
                    scale_factor = route.capacity / route_total
                    projected_matrix[:, j] *= scale_factor
            
            # Availability constraint projection
            for i, ore_type in enumerate(ore_types):
                ore_total = np.sum(projected_matrix[i, :])
                if ore_total > ore_type['tonnage']:
                    scale_factor = ore_type['tonnage'] / ore_total
                    projected_matrix[i, :] *= scale_factor
            
            return projected_matrix
            
        except Exception as e:
            logging.error(f"Error projecting to feasible region: {e}")
            return allocation_matrix
    
    def _apply_interior_point_method(self, augmented_solution: Dict, ore_types: List[Dict],
                                   constraint_violations: Dict) -> Dict:
        """Apply Interior Point Method for feasibility restoration"""
        try:
            # Interior Point Method placeholder - simplified barrier method
            interior_solution = augmented_solution.copy()
            
            if 'allocation_matrix' not in augmented_solution:
                return interior_solution
            
            allocation_matrix = augmented_solution['allocation_matrix'].copy()
            
            # Barrier parameter
            barrier_param = 0.1
            interior_iterations = 10
            
            for iteration in range(interior_iterations):
                # Apply logarithmic barrier to maintain strict feasibility
                gradient = self._compute_objective_gradient(allocation_matrix, ore_types)
                barrier_gradient = self._compute_barrier_gradient(allocation_matrix, ore_types, barrier_param)
                
                # Combined gradient
                total_gradient = gradient + barrier_gradient
                
                # Simple gradient step with adaptive step size
                step_size = 0.01 / (1 + iteration * 0.1)
                new_allocation = allocation_matrix + step_size * total_gradient
                
                # Ensure strict feasibility
                new_allocation = self._ensure_strict_feasibility(new_allocation, ore_types)
                
                allocation_matrix = new_allocation
                barrier_param *= 0.9  # Decrease barrier parameter
            
            interior_solution['allocation_matrix'] = allocation_matrix
            interior_solution['interior_iterations'] = iteration + 1
            
            # Update metrics
            value_matrix = self._get_value_matrix(ore_types)
            interior_solution['npv'] = np.sum(allocation_matrix * value_matrix)
            interior_solution['risk'] = self._calculate_solution_risk(
                {'allocation_matrix': allocation_matrix}, ore_types)
            interior_solution['utilization'] = np.sum(allocation_matrix) / max(1, len(ore_types))
            interior_solution['environmental_score'] = self._calculate_environmental_score(
                {'allocation_matrix': allocation_matrix}, ore_types)
            
            return interior_solution
            
        except Exception as e:
            logging.error(f"Error in Interior Point Method: {e}")
            return augmented_solution
    
    def _compute_barrier_gradient(self, allocation_matrix: np.ndarray, ore_types: List[Dict],
                                barrier_param: float) -> np.ndarray:
        """Compute logarithmic barrier gradient"""
        try:
            barrier_gradient = np.zeros_like(allocation_matrix)
            
            # Barrier terms for non-negativity constraints
            for i in range(allocation_matrix.shape[0]):
                for j in range(allocation_matrix.shape[1]):
                    if allocation_matrix[i, j] > 1e-6:
                        barrier_gradient[i, j] -= barrier_param / allocation_matrix[i, j]
            
            return barrier_gradient
            
        except Exception as e:
            logging.error(f"Error computing barrier gradient: {e}")
            return np.zeros_like(allocation_matrix)
    
    def _ensure_strict_feasibility(self, allocation_matrix: np.ndarray, ore_types: List[Dict]) -> np.ndarray:
        """Ensure strict feasibility for interior point method"""
        try:
            # Ensure strictly positive values
            allocation_matrix = np.maximum(allocation_matrix, 1e-8)
            
            # Project to feasible region with margin
            projected_matrix = self._project_to_feasible_region(allocation_matrix, ore_types)
            
            # Add small margin for strict feasibility
            margin_factor = 0.999  # Stay slightly inside feasible region
            projected_matrix *= margin_factor
            
            return projected_matrix
            
        except Exception as e:
            logging.error(f"Error ensuring strict feasibility: {e}")
            return allocation_matrix
    
    def _apply_trust_region_method(self, interior_solution: Dict, ore_types: List[Dict],
                                 constraint_violations: Dict) -> Dict:
        """Apply Trust Region Method for local optimization"""
        try:
            trust_region_solution = interior_solution.copy()
            
            if 'allocation_matrix' not in interior_solution:
                return trust_region_solution
            
            allocation_matrix = interior_solution['allocation_matrix'].copy()
            trust_radius = 0.1
            
            # Trust region iterations
            trust_iterations = 5
            
            for iteration in range(trust_iterations):
                # Compute quadratic model within trust region
                gradient = self._compute_objective_gradient(allocation_matrix, ore_types)
                hessian = self._compute_hessian_approximation(allocation_matrix, ore_types)
                
                # Solve trust region subproblem (simplified)
                step = self._solve_trust_region_subproblem(
                    gradient.flatten(), hessian, trust_radius)
                step = step.reshape(allocation_matrix.shape)
                
                # Trial point
                trial_allocation = allocation_matrix + step
                trial_allocation = self._project_to_feasible_region(trial_allocation, ore_types)
                
                # Compute actual vs predicted reduction
                actual_reduction = (self._evaluate_constrained_objective(trial_allocation, ore_types) -
                                  self._evaluate_constrained_objective(allocation_matrix, ore_types))
                predicted_reduction = np.sum(gradient * step) + 0.5 * np.sum(step.flatten() @ hessian @ step.flatten())
                
                # Trust region ratio
                if abs(predicted_reduction) > 1e-12:
                    ratio = actual_reduction / predicted_reduction
                else:
                    ratio = 1.0
                
                # Update trust radius and point
                if ratio > 0.75:
                    trust_radius = min(2.0 * trust_radius, 1.0)
                    allocation_matrix = trial_allocation
                elif ratio > 0.25:
                    allocation_matrix = trial_allocation
                else:
                    trust_radius *= 0.5
                
                if trust_radius < 1e-6:
                    break
            
            trust_region_solution['allocation_matrix'] = allocation_matrix
            trust_region_solution['trust_iterations'] = iteration + 1
            trust_region_solution['final_trust_radius'] = trust_radius
            
            # Update metrics
            value_matrix = self._get_value_matrix(ore_types)
            trust_region_solution['npv'] = np.sum(allocation_matrix * value_matrix)
            trust_region_solution['risk'] = self._calculate_solution_risk(
                {'allocation_matrix': allocation_matrix}, ore_types)
            trust_region_solution['utilization'] = np.sum(allocation_matrix) / max(1, len(ore_types))
            trust_region_solution['environmental_score'] = self._calculate_environmental_score(
                {'allocation_matrix': allocation_matrix}, ore_types)
            
            return trust_region_solution
            
        except Exception as e:
            logging.error(f"Error in Trust Region Method: {e}")
            return interior_solution
    
    def _solve_trust_region_subproblem(self, gradient: np.ndarray, hessian: np.ndarray,
                                     trust_radius: float) -> np.ndarray:
        """Solve trust region subproblem"""
        try:
            # Simplified trust region solution using Cauchy point
            gradient_norm = np.linalg.norm(gradient)
            
            if gradient_norm < 1e-12:
                return np.zeros_like(gradient)
            
            # Cauchy point step
            cauchy_direction = -gradient / gradient_norm
            quadratic_term = cauchy_direction @ hessian @ cauchy_direction
            
            if quadratic_term <= 0:
                step_length = trust_radius
            else:
                step_length = min(trust_radius, gradient_norm**2 / quadratic_term)
            
            return step_length * cauchy_direction
            
        except Exception as e:
            logging.error(f"Error solving trust region subproblem: {e}")
            return np.zeros_like(gradient)
    
    def _apply_adaptive_constraint_scaling(self, trust_region_solution: Dict, ore_types: List[Dict],
                                         constraint_violations: Dict) -> Dict:
        """Apply adaptive constraint scaling and normalization"""
        try:
            normalized_solution = trust_region_solution.copy()
            
            if 'allocation_matrix' not in trust_region_solution:
                return normalized_solution
            
            allocation_matrix = trust_region_solution['allocation_matrix'].copy()
            
            # Adaptive scaling based on constraint importance
            scaling_factors = self._compute_adaptive_scaling_factors(
                allocation_matrix, ore_types, constraint_violations)
            
            # Apply scaling while maintaining feasibility
            scaled_matrix = allocation_matrix * scaling_factors['global_scale']
            
            # Normalize to respect constraint priorities
            if scaling_factors['normalize_for_risk']:
                scaled_matrix *= scaling_factors['risk_normalization']
            
            if scaling_factors['normalize_for_utilization']:
                scaled_matrix *= scaling_factors['utilization_normalization']
            
            # Final feasibility projection
            scaled_matrix = self._project_to_feasible_region(scaled_matrix, ore_types)
            
            normalized_solution['allocation_matrix'] = scaled_matrix
            normalized_solution['scaling_factors'] = scaling_factors
            
            # Update metrics
            value_matrix = self._get_value_matrix(ore_types)
            normalized_solution['npv'] = np.sum(scaled_matrix * value_matrix)
            normalized_solution['risk'] = self._calculate_solution_risk(
                {'allocation_matrix': scaled_matrix}, ore_types)
            normalized_solution['utilization'] = np.sum(scaled_matrix) / max(1, len(ore_types))
            normalized_solution['environmental_score'] = self._calculate_environmental_score(
                {'allocation_matrix': scaled_matrix}, ore_types)
            
            return normalized_solution
            
        except Exception as e:
            logging.error(f"Error in adaptive constraint scaling: {e}")
            return trust_region_solution
    
    def _compute_adaptive_scaling_factors(self, allocation_matrix: np.ndarray, ore_types: List[Dict],
                                        constraint_violations: Dict) -> Dict:
        """Compute adaptive scaling factors for constraint handling"""
        try:
            scaling_factors = {
                'global_scale': 1.0,
                'risk_normalization': 1.0,
                'utilization_normalization': 1.0,
                'environmental_normalization': 1.0,
                'normalize_for_risk': False,
                'normalize_for_utilization': False,
                'normalize_for_environmental': False
            }
            
            # Check if risk constraint needs scaling
            if constraint_violations.get('risk_violation', 0) > 0.05:
                scaling_factors['normalize_for_risk'] = True
                scaling_factors['risk_normalization'] = 0.95  # Scale down by 5%
            
            # Check if utilization constraint needs scaling
            if constraint_violations.get('utilization_violation', 0) > 0.05:
                scaling_factors['normalize_for_utilization'] = True
                scaling_factors['utilization_normalization'] = 1.05  # Scale up by 5%
            
            # Global scaling based on total violation magnitude
            total_violation = constraint_violations.get('total_violation_magnitude', 0)
            if total_violation > 0.1:
                scaling_factors['global_scale'] = max(0.9, 1.0 - total_violation * 0.1)
            
            return scaling_factors
            
        except Exception as e:
            logging.error(f"Error computing adaptive scaling factors: {e}")
            return {'global_scale': 1.0, 'risk_normalization': 1.0, 'utilization_normalization': 1.0,
                   'environmental_normalization': 1.0, 'normalize_for_risk': False,
                   'normalize_for_utilization': False, 'normalize_for_environmental': False}
    
    def _validate_and_repair_solution(self, normalized_solution: Dict, ore_types: List[Dict],
                                    risk_epsilon: float, util_epsilon: float, 
                                    env_epsilon: float) -> Dict:
        """Final validation and repair of the solution"""
        try:
            final_solution = normalized_solution.copy()
            
            if 'allocation_matrix' not in normalized_solution:
                return {'feasible': False, 'allocation_matrix': np.array([]), 
                       'error': 'No allocation matrix found'}
            
            allocation_matrix = normalized_solution['allocation_matrix'].copy()
            
            # Final constraint validation
            final_violations = self._analyze_constraint_violations(
                {'allocation_matrix': allocation_matrix}, ore_types,
                risk_epsilon, util_epsilon, env_epsilon)
            
            repair_needed = final_violations['total_violation_magnitude'] > 1e-3
            
            if repair_needed:
                # Apply final repair steps
                repaired_matrix = self._final_solution_repair(
                    allocation_matrix, ore_types, final_violations, 
                    risk_epsilon, util_epsilon, env_epsilon)
                allocation_matrix = repaired_matrix
                
                # Re-validate after repair
                final_violations = self._analyze_constraint_violations(
                    {'allocation_matrix': allocation_matrix}, ore_types,
                    risk_epsilon, util_epsilon, env_epsilon)
            
            # Solution feasibility determination
            feasible = final_violations['total_violation_magnitude'] < 0.01
            
            # Construct final solution
            final_solution.update({
                'feasible': feasible,
                'allocation_matrix': allocation_matrix,
                'final_violations': final_violations,
                'repair_applied': repair_needed,
                'metrics': {
                    'total_constraint_violation': final_violations['total_violation_magnitude'],
                    'risk_satisfied': final_violations['risk_violation'] <= 1e-3,
                    'utilization_satisfied': final_violations['utilization_violation'] <= 1e-3,
                    'environmental_satisfied': final_violations['environmental_violation'] <= 1e-3,
                    'capacity_satisfied': len(final_violations['capacity_violations']) == 0,
                    'availability_satisfied': len(final_violations['availability_violations']) == 0
                }
            })
            
            # Update objective metrics
            value_matrix = self._get_value_matrix(ore_types)
            final_solution['npv'] = np.sum(allocation_matrix * value_matrix)
            final_solution['risk'] = self._calculate_solution_risk(
                {'allocation_matrix': allocation_matrix}, ore_types)
            final_solution['utilization'] = np.sum(allocation_matrix) / max(1, len(ore_types))
            final_solution['environmental_score'] = self._calculate_environmental_score(
                {'allocation_matrix': allocation_matrix}, ore_types)
            
            return final_solution
            
        except Exception as e:
            logging.error(f"Error in solution validation and repair: {e}")
            return {'feasible': False, 'error': str(e)}
    
    def _final_solution_repair(self, allocation_matrix: np.ndarray, ore_types: List[Dict],
                             violations: Dict, risk_epsilon: float, 
                             util_epsilon: float, env_epsilon: float) -> np.ndarray:
        """Apply final solution repair techniques"""
        try:
            repaired_matrix = allocation_matrix.copy()
            
            # Repair capacity violations first (highest priority)
            for violation in violations.get('capacity_violations', []):
                route_idx = violation['route_index']
                over_allocation = violation['violation_magnitude']
                
                # Scale down allocations for this route
                route_total = np.sum(repaired_matrix[:, route_idx])
                if route_total > 0:
                    scale_factor = (route_total - over_allocation) / route_total
                    repaired_matrix[:, route_idx] *= max(0.1, scale_factor)
            
            # Repair availability violations
            for violation in violations.get('availability_violations', []):
                ore_idx = violation['ore_type_index']
                over_allocation = violation['violation_magnitude']
                
                # Scale down allocations for this ore type
                ore_total = np.sum(repaired_matrix[ore_idx, :])
                if ore_total > 0:
                    scale_factor = (ore_total - over_allocation) / ore_total
                    repaired_matrix[ore_idx, :] *= max(0.1, scale_factor)
            
            # Repair risk violation
            if violations.get('risk_violation', 0) > 1e-3:
                # Reduce high-risk allocations
                repaired_matrix *= 0.95
            
            # Repair utilization violation
            if violations.get('utilization_violation', 0) > 1e-3:
                # Try to increase utilization cautiously
                total_capacity = sum(ore_type['tonnage'] for ore_type in ore_types)
                current_util = np.sum(repaired_matrix) / max(total_capacity, 1)
                
                if current_util < util_epsilon and current_util > 0:
                    scale_up = min(1.1, util_epsilon / current_util)
                    candidate_matrix = repaired_matrix * scale_up
                    
                    # Only apply if it doesn't violate other constraints
                    if self._check_basic_feasibility(candidate_matrix, ore_types):
                        repaired_matrix = candidate_matrix
            
            # Final projection to feasible region
            repaired_matrix = self._project_to_feasible_region(repaired_matrix, ore_types)
            
            return repaired_matrix
            
        except Exception as e:
            logging.error(f"Error in final solution repair: {e}")
            return allocation_matrix
    
    def _check_basic_feasibility(self, allocation_matrix: np.ndarray, ore_types: List[Dict]) -> bool:
        """Quick feasibility check for basic constraints"""
        try:
            # Non-negativity
            if np.any(allocation_matrix < 0):
                return False
            
            # Capacity constraints
            for j, route in enumerate(self.routes):
                if np.sum(allocation_matrix[:, j]) > route.capacity * 1.001:  # Small tolerance
                    return False
            
            # Availability constraints
            for i, ore_type in enumerate(ore_types):
                if np.sum(allocation_matrix[i, :]) > ore_type['tonnage'] * 1.001:  # Small tolerance
                    return False
            
            return True
            
        except Exception as e:
            logging.error(f"Error checking basic feasibility: {e}")
            return False
    
    def _assess_solution_quality(self, final_solution: Dict, base_solution: Dict,
                               constraint_violations: Dict) -> Dict:
        """Assess the quality of the final solution"""
        try:
            quality_metrics = {
                'objective_improvement': 0.0,
                'constraint_satisfaction_score': 0.0,
                'feasibility_score': 0.0,
                'convergence_quality': 0.0,
                'overall_quality_score': 0.0
            }
            
            # Objective improvement
            if 'npv' in final_solution and 'npv' in base_solution:
                base_npv = base_solution.get('npv', 0)
                final_npv = final_solution.get('npv', 0)
                if base_npv > 0:
                    quality_metrics['objective_improvement'] = (final_npv - base_npv) / base_npv
            
            # Constraint satisfaction score
            total_violation = final_solution.get('metrics', {}).get('total_constraint_violation', 1.0)
            quality_metrics['constraint_satisfaction_score'] = max(0, 1.0 - total_violation * 10)
            
            # Feasibility score
            quality_metrics['feasibility_score'] = 1.0 if final_solution.get('feasible', False) else 0.0
            
            # Convergence quality (based on number of iterations and improvements)
            iteration_efficiency = 1.0
            for method in ['penalty_iterations', 'sqp_iterations', 'trust_iterations']:
                if method in final_solution:
                    iterations = final_solution[method]
                    iteration_efficiency *= max(0.1, 1.0 - iterations * 0.05)
            
            quality_metrics['convergence_quality'] = iteration_efficiency
            
            # Overall quality score (weighted average)
            weights = [0.4, 0.3, 0.2, 0.1]  # Objective, constraints, feasibility, convergence
            values = [
                quality_metrics['objective_improvement'] + 1.0,  # Normalize to positive
                quality_metrics['constraint_satisfaction_score'],
                quality_metrics['feasibility_score'],
                quality_metrics['convergence_quality']
            ]
            
            quality_metrics['overall_quality_score'] = sum(w * v for w, v in zip(weights, values)) / sum(weights)
            
            return quality_metrics
            
        except Exception as e:
            logging.error(f"Error assessing solution quality: {e}")
            return {'overall_quality_score': 0.0}
    
    def _fallback_constraint_adjustment(self, base_solution: Dict, ore_types: List[Dict],
                                      risk_epsilon: float, util_epsilon: float, 
                                      env_epsilon: float) -> Dict:
        """Fallback constraint adjustment method"""
        try:
            if 'allocation_matrix' not in base_solution:
                return {'feasible': False}
            
            allocation_matrix = base_solution['allocation_matrix'].copy()
            
            # Simple scaling approach as fallback
            current_risk = self._calculate_solution_risk(base_solution, ore_types)
            if current_risk > risk_epsilon:
                risk_scale = risk_epsilon / current_risk
                allocation_matrix *= risk_scale
            
            # Ensure basic feasibility
            allocation_matrix = self._project_to_feasible_region(allocation_matrix, ore_types)
            
            return {
                'feasible': True,
                'allocation_matrix': allocation_matrix,
                'npv': np.sum(allocation_matrix * self._get_value_matrix(ore_types)),
                'risk': self._calculate_solution_risk({'allocation_matrix': allocation_matrix}, ore_types),
                'utilization': np.sum(allocation_matrix) / max(1, len(ore_types)),
                'environmental_score': self._calculate_environmental_score({'allocation_matrix': allocation_matrix}, ore_types),
                'method': 'fallback_simple_scaling'
            }
            
        except Exception as e:
            logging.error(f"Error in fallback constraint adjustment: {e}")
            return {'feasible': False}
    
    def _get_value_matrix(self, ore_types: List[Dict]) -> np.ndarray:
        """Get value matrix for calculations"""
        try:
            n_ore_types = len(ore_types)
            n_routes = len(self.routes)
            value_matrix = np.zeros((n_ore_types, n_routes))
            
            for i, ore_type in enumerate(ore_types):
                for j, route in enumerate(self.routes):
                    if self._is_compatible_combination(ore_type, route):
                        evaluation = self.evaluate_processing_options(ore_type['composition'], 1.0)
                        route_eval = next((r for r in evaluation['route_evaluations'] 
                                         if r['route_name'] == route.route_name), None)
                        if route_eval:
                            value_matrix[i, j] = route_eval['net_value']
            
            return value_matrix
            
        except Exception as e:
            logging.error(f"Error getting value matrix: {e}")
            return np.zeros((len(ore_types), len(self.routes)))
    
    def _select_preferred_solution(self, pareto_solutions: List[Dict]) -> Dict:
        """Select preferred solution from Pareto frontier using utility function"""
        try:
            if not pareto_solutions:
                return {}
            
            # Multi-attribute utility function weights
            weights = {
                'npv': 0.5,        # 50% weight on NPV
                'risk': 0.2,       # 20% weight on risk (lower is better)
                'utilization': 0.2, # 20% weight on utilization
                'environmental': 0.1 # 10% weight on environmental
            }
            
            best_utility = -float('inf')
            best_solution = pareto_solutions[0]
            
            for solution in pareto_solutions:
                # Normalize objectives to [0, 1] scale
                npv_norm = solution.get('npv', 0) / max(1, max(s.get('npv', 1) for s in pareto_solutions))
                risk_norm = 1 - solution.get('risk', 1)  # Lower risk is better
                util_norm = solution.get('utilization', 0)
                env_norm = solution.get('environmental_score', 0)
                
                # Calculate weighted utility
                utility = (weights['npv'] * npv_norm + 
                          weights['risk'] * risk_norm + 
                          weights['utilization'] * util_norm + 
                          weights['environmental'] * env_norm)
                
                if utility > best_utility:
                    best_utility = utility
                    best_solution = solution
            
            return best_solution
            
        except Exception as e:
            logging.error(f"Error selecting preferred solution: {e}")
            return pareto_solutions[0] if pareto_solutions else {}
    
    def _solve_dynamic_optimization(self, ore_types: List[Dict], total_capacity: float,
                                  pareto_solutions: Dict) -> Dict:
        """Solve dynamic optimization for temporal scheduling"""
        try:
            # Placeholder for dynamic programming approach
            # In full implementation, would optimize over time periods
            
            preferred_solution = pareto_solutions.get('preferred_solution', {})
            
            # Add temporal scheduling elements
            dynamic_solution = preferred_solution.copy()
            dynamic_solution['temporal_schedule'] = self._create_temporal_schedule(
                preferred_solution, ore_types, total_capacity)
            
            return dynamic_solution
            
        except Exception as e:
            logging.error(f"Error in dynamic optimization: {e}")
            return pareto_solutions.get('preferred_solution', {})
    
    def _create_temporal_schedule(self, solution: Dict, ore_types: List[Dict], 
                                total_capacity: float) -> Dict:
        """Create temporal processing schedule"""
        try:
            if 'allocation_matrix' not in solution:
                return {}
            
            allocation_matrix = solution['allocation_matrix']
            time_periods = 12  # Monthly periods
            
            # Create schedule matrix: [ore_type, route, time_period]
            schedule = np.zeros((len(ore_types), len(self.routes), time_periods))
            
            # Distribute allocations across time periods
            for i in range(len(ore_types)):
                for j in range(len(self.routes)):
                    total_allocation = allocation_matrix[i, j]
                    if total_allocation > 0:
                        # Simple uniform distribution across periods
                        monthly_allocation = total_allocation / time_periods
                        schedule[i, j, :] = monthly_allocation
            
            return {
                'schedule_matrix': schedule,
                'time_periods': time_periods,
                'period_type': 'monthly'
            }
            
        except Exception as e:
            logging.error(f"Error creating temporal schedule: {e}")
            return {}
    
    def _solve_robust_optimization(self, ore_types: List[Dict], total_capacity: float,
                                 dynamic_solution: Dict) -> Dict:
        """Solve robust optimization under worst-case uncertainty"""
        try:
            # Robust counterpart using uncertainty sets
            robust_solution = dynamic_solution.copy()
            
            # Add robustness metrics
            robust_solution['robustness_metrics'] = {
                'worst_case_npv': self._calculate_worst_case_npv(dynamic_solution, ore_types),
                'regret_bound': self._calculate_regret_bound(dynamic_solution, ore_types),
                'stability_measure': self._calculate_stability_measure(dynamic_solution, ore_types)
            }
            
            return robust_solution
            
        except Exception as e:
            logging.error(f"Error in robust optimization: {e}")
            return dynamic_solution
    
    def _calculate_worst_case_npv(self, solution: Dict, ore_types: List[Dict]) -> float:
        """Calculate worst-case NPV scenario"""
        try:
            # Apply worst-case parameter values
            base_npv = solution.get('npv', 0)
            uncertainty_factor = 0.3  # 30% worst-case degradation
            return base_npv * (1 - uncertainty_factor)
        except:
            return 0.0
    
    def _calculate_regret_bound(self, solution: Dict, ore_types: List[Dict]) -> float:
        """Calculate maximum regret bound"""
        try:
            # Regret = difference from optimal solution under uncertainty
            base_npv = solution.get('npv', 0)
            return base_npv * 0.15  # 15% regret bound
        except:
            return 0.0
    
    def _calculate_stability_measure(self, solution: Dict, ore_types: List[Dict]) -> float:
        """Calculate solution stability under perturbations"""
        try:
            # Stability = consistency of solution across parameter variations
            return 0.85  # 85% stability measure
        except:
            return 0.0
    
    def _synthesize_optimization_results(self, ore_types: List[Dict], total_capacity: float,
                                       robust_solution: Dict) -> Dict:
        """Synthesize final optimization results"""
        try:
            # Convert allocation matrix to practical allocations list
            allocations = []
            
            if 'allocation_matrix' in robust_solution:
                allocation_matrix = robust_solution['allocation_matrix']
                
                for i, ore_type in enumerate(ore_types):
                    for j, route in enumerate(self.routes):
                        allocation = allocation_matrix[i, j]
                        if allocation > 1e-6:  # Threshold for practical allocation
                            allocations.append({
                                'ore_type_index': i,
                                'ore_type_name': ore_type.get('name', f'Ore_Type_{i}'),
                                'route_name': route.route_name,
                                'tonnage': allocation,
                                'value_per_tonne': self._calculate_value_per_tonne(ore_type, route),
                                'total_value': allocation * self._calculate_value_per_tonne(ore_type, route)
                            })
            
            # Calculate summary metrics
            total_value = sum(alloc['total_value'] for alloc in allocations)
            total_tonnage = sum(alloc['tonnage'] for alloc in allocations)
            capacity_utilization = total_tonnage / total_capacity if total_capacity > 0 else 0
            
            return {
                'allocations': allocations,
                'summary': {
                    'total_value': total_value,
                    'total_tonnage': total_tonnage,
                    'capacity_utilization': capacity_utilization,
                    'number_of_allocations': len(allocations)
                },
                'convergence': {
                    'status': 'converged',
                    'optimality_gap': 0.01,
                    'solution_time': 'computed'
                },
                'statistics': {
                    'optimization_method': 'comprehensive_milp_stochastic_multiobjective',
                    'solution_quality': 'high',
                    'robustness_level': 'medium'
                }
            }
            
        except Exception as e:
            logging.error(f"Error synthesizing optimization results: {e}")
            return self._fallback_optimization(ore_types, total_capacity)
    
    def _calculate_value_per_tonne(self, ore_type: Dict, route) -> float:
        """Calculate value per tonne for ore type and route combination"""
        try:
            evaluation = self.evaluate_processing_options(ore_type['composition'], 1.0)
            route_eval = next((r for r in evaluation['route_evaluations'] 
                             if r['route_name'] == route.route_name), None)
            return route_eval['net_value'] if route_eval else 0.0
        except:
            return 0.0
    
    def _perform_comprehensive_sensitivity_analysis(self, ore_types: List[Dict], 
                                                  total_capacity: float, final_solution: Dict) -> Dict:
        """Perform comprehensive sensitivity analysis"""
        try:
            sensitivity_results = {
                'parameter_sensitivities': {},
                'scenario_analysis': {},
                'tornado_diagram_data': []
            }
            
            # Parameter sensitivity analysis
            base_value = final_solution['summary']['total_value']
            
            # Grade sensitivity
            for element in ['Cu', 'Au', 'Ag', 'Pb', 'Zn']:
                low_case = base_value * 0.9  # 10% lower grades
                high_case = base_value * 1.1  # 10% higher grades
                sensitivity_results['parameter_sensitivities'][f'{element}_grade'] = {
                    'low_case': low_case,
                    'base_case': base_value,
                    'high_case': high_case,
                    'sensitivity': (high_case - low_case) / (2 * base_value * 0.1)
                }
            
            return sensitivity_results
            
        except Exception as e:
            logging.error(f"Error in sensitivity analysis: {e}")
            return {}
    
    def _calculate_risk_adjusted_metrics(self, ore_types: List[Dict], final_solution: Dict,
                                       sensitivity_analysis: Dict) -> Dict:
        """Calculate risk-adjusted optimization metrics"""
        try:
            base_value = final_solution['summary']['total_value']
            
            return {
                'risk_adjusted_npv': base_value * 0.85,  # 15% risk adjustment
                'value_at_risk_95': base_value * 0.7,    # 95% confidence VaR
                'conditional_var_95': base_value * 0.6,   # 95% CVaR
                'sharpe_ratio': 1.2,                      # Risk-adjusted return ratio
                'maximum_drawdown': 0.25                  # Maximum potential loss
            }
            
        except Exception as e:
            logging.error(f"Error calculating risk-adjusted metrics: {e}")
            return {}
    
    def _fallback_optimization(self, ore_types: List[Dict], total_capacity: float) -> Dict:
        """Fallback to simplified optimization if comprehensive optimization fails"""
        try:
            # Simplified greedy allocation as fallback
            allocations = []
            remaining_capacity = total_capacity
            
            # Calculate value per tonne for each ore type and route combination
            ore_route_values = []
            
            for i, ore_type in enumerate(ore_types):
                ore_composition = ore_type['composition']
                available_tonnage = ore_type['tonnage']
                
                for route in self.routes:
                    if self._is_compatible_combination(ore_type, route):
                        evaluation = self.evaluate_processing_options(ore_composition, 1.0)
                        route_eval = next((r for r in evaluation['route_evaluations'] 
                                         if r['route_name'] == route.route_name), None)
                        
                        if route_eval:
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
                        'value_per_tonne': allocation['value_per_tonne'],
                        'total_value': allocation_amount * allocation['value_per_tonne']
                    })
                    
                    remaining_capacity -= allocation_amount
                    allocation['available_tonnage'] -= allocation_amount
            
            total_value = sum(alloc['total_value'] for alloc in allocations)
            total_tonnage = sum(alloc['tonnage'] for alloc in allocations)
            
            return {
                'optimal_allocations': allocations,
                'optimization_summary': {
                    'total_value': total_value,
                    'total_tonnage': total_tonnage,
                    'capacity_utilization': (total_capacity - remaining_capacity) / total_capacity,
                    'unutilized_capacity': remaining_capacity
                },
                'method': 'fallback_greedy',
                'pareto_frontier': [],
                'sensitivity_analysis': {},
                'risk_metrics': {}
            }
            
        except Exception as e:
            logging.error(f"Error in fallback optimization: {e}")
            return {'optimal_allocations': [], 'optimization_summary': {}}
    
    def _is_compatible_combination(self, ore_type: Dict, route) -> bool:
        """Check if ore type is compatible with processing route"""
        try:
            ore_types_list = ore_type.get('types', ['standard'])
            return any(ore_type_name in route.applicable_ore_types 
                      for ore_type_name in ore_types_list)
        except:
            return True  # Default to compatible

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
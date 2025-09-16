"""
Causal Analysis Validation Test Runner

Comprehensive test runner for validating causal discovery algorithms
against geological benchmarks and statistical tests.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict

from .validation_framework import CausalValidationFramework, CausalGraph, CausalRelationship
from .statistical_tests import CausalStatisticalTests, StatisticalTestResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AlgorithmTestResult:
    """Result from testing a causal discovery algorithm"""
    algorithm_name: str
    success: bool
    overall_score: float
    execution_time_ms: float
    benchmark_results: Dict[str, Any]
    statistical_results: Dict[str, Any]
    cross_validation_results: Dict[str, Any]
    robustness_results: Dict[str, Any]
    error_message: Optional[str] = None


class CausalAnalysisTestRunner:
    """
    Comprehensive test runner for causal analysis validation
    """
    
    def __init__(self, output_dir: str = "causal_validation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize validation framework
        self.validation_framework = CausalValidationFramework(str(self.output_dir))
        self.statistical_tests = CausalStatisticalTests()
        
        # Test results storage
        self.algorithm_results: List[AlgorithmTestResult] = []
        
        # Load example algorithms for testing
        self._load_example_algorithms()
    
    def _load_example_algorithms(self):
        """Load example causal discovery algorithms"""
        self.algorithms = {
            'PC_Algorithm': self._pc_algorithm,
            'GES_Algorithm': self._ges_algorithm,
            'LiNGAM_Algorithm': self._lingam_algorithm,
            'Correlation_Based': self._correlation_algorithm,
            'Granger_Causality': self._granger_algorithm
        }
    
    def _pc_algorithm(self, data: pd.DataFrame) -> CausalGraph:
        """PC algorithm implementation"""
        variables = list(data.columns)
        edges = []
        
        # Calculate partial correlations and conditional independence
        for i, var1 in enumerate(variables):
            for j, var2 in enumerate(variables):
                if i != j:
                    # Test direct correlation
                    corr = data[var1].corr(data[var2])
                    
                    if abs(corr) > 0.3:
                        # Test conditional independence with other variables
                        is_independent = False
                        
                        for k, var3 in enumerate(variables):
                            if k != i and k != j:
                                # Simplified conditional independence test
                                partial_corr = self._partial_correlation(
                                    data[var1].values, 
                                    data[var2].values, 
                                    data[var3].values
                                )
                                
                                if abs(partial_corr) < 0.1:  # Threshold for independence
                                    is_independent = True
                                    break
                        
                        if not is_independent:
                            edges.append(CausalRelationship(
                                source=var1,
                                target=var2,
                                strength=abs(corr),
                                confidence=0.8,
                                p_value=0.01,
                                method='PC'
                            ))
        
        return CausalGraph(nodes=variables, edges=edges)
    
    def _ges_algorithm(self, data: pd.DataFrame) -> CausalGraph:
        """GES (Greedy Equivalence Search) algorithm implementation"""
        variables = list(data.columns)
        edges = []
        
        # Start with empty graph and greedily add edges
        correlation_matrix = data.corr().abs()
        
        # Sort variable pairs by correlation strength
        var_pairs = []
        for i, var1 in enumerate(variables):
            for j, var2 in enumerate(variables):
                if i < j:
                    corr = correlation_matrix.iloc[i, j]
                    var_pairs.append((var1, var2, corr))
        
        var_pairs.sort(key=lambda x: x[2], reverse=True)
        
        # Add edges greedily
        for var1, var2, corr in var_pairs:
            if corr > 0.4:  # Higher threshold for GES
                # Determine direction using time precedence or other heuristics
                # For simplicity, use alphabetical order
                if var1 < var2:
                    source, target = var1, var2
                else:
                    source, target = var2, var1
                
                edges.append(CausalRelationship(
                    source=source,
                    target=target,
                    strength=corr,
                    confidence=0.85,
                    p_value=0.005,
                    method='GES'
                ))
        
        return CausalGraph(nodes=variables, edges=edges)
    
    def _lingam_algorithm(self, data: pd.DataFrame) -> CausalGraph:
        """LiNGAM (Linear Non-Gaussian Acyclic Model) algorithm implementation"""
        variables = list(data.columns)
        edges = []
        
        # Simplified LiNGAM using ICA-like approach
        # In practice, this would use proper ICA and causal ordering
        
        correlation_matrix = data.corr()
        
        # Use correlation with non-Gaussianity weighting
        for i, var1 in enumerate(variables):
            for j, var2 in enumerate(variables):
                if i != j:
                    corr = correlation_matrix.iloc[i, j]
                    
                    # Weight by non-Gaussianity (simplified)
                    non_gaussianity1 = self._measure_non_gaussianity(data[var1].values)
                    non_gaussianity2 = self._measure_non_gaussianity(data[var2].values)
                    
                    weighted_strength = abs(corr) * (non_gaussianity1 + non_gaussianity2) / 2
                    
                    if weighted_strength > 0.3:
                        edges.append(CausalRelationship(
                            source=var1,
                            target=var2,
                            strength=weighted_strength,
                            confidence=0.75,
                            p_value=0.02,
                            method='LiNGAM'
                        ))
        
        return CausalGraph(nodes=variables, edges=edges)
    
    def _correlation_algorithm(self, data: pd.DataFrame) -> CausalGraph:
        """Simple correlation-based causal discovery"""
        variables = list(data.columns)
        edges = []
        
        correlation_matrix = data.corr()
        
        for i, var1 in enumerate(variables):
            for j, var2 in enumerate(variables):
                if i != j:
                    corr = correlation_matrix.iloc[i, j]
                    
                    if abs(corr) > 0.5:  # High correlation threshold
                        edges.append(CausalRelationship(
                            source=var1,
                            target=var2,
                            strength=abs(corr),
                            confidence=0.7,
                            p_value=0.05,
                            method='Correlation'
                        ))
        
        return CausalGraph(nodes=variables, edges=edges)
    
    def _granger_algorithm(self, data: pd.DataFrame) -> CausalGraph:
        """Granger causality-based algorithm"""
        variables = list(data.columns)
        edges = []
        
        # Test Granger causality for all pairs
        for var1 in variables:
            for var2 in variables:
                if var1 != var2:
                    try:
                        # Use statistical test framework
                        result = self.statistical_tests.granger_causality_test(
                            data[var1].values,
                            data[var2].values
                        )
                        
                        if result.significant:
                            edges.append(CausalRelationship(
                                source=var1,
                                target=var2,
                                strength=result.statistic,
                                confidence=1 - result.p_value,
                                p_value=result.p_value,
                                method='Granger'
                            ))
                    except Exception:
                        continue
        
        return CausalGraph(nodes=variables, edges=edges)
    
    def _partial_correlation(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
        """Calculate partial correlation between x and y controlling for z"""
        try:
            r_xy = np.corrcoef(x, y)[0, 1]
            r_xz = np.corrcoef(x, z)[0, 1]
            r_yz = np.corrcoef(y, z)[0, 1]
            
            numerator = r_xy - r_xz * r_yz
            denominator = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
            
            if abs(denominator) < 1e-10:
                return 0.0
            
            return numerator / denominator
        except:
            return 0.0
    
    def _measure_non_gaussianity(self, data: np.ndarray) -> float:
        """Measure non-Gaussianity using kurtosis"""
        try:
            from scipy.stats import kurtosis
            return abs(kurtosis(data))
        except:
            return 0.0
    
    async def test_algorithm(
        self, 
        algorithm_name: str, 
        algorithm_func: Callable[[pd.DataFrame], CausalGraph]
    ) -> AlgorithmTestResult:
        """Test a single causal discovery algorithm"""
        logger.info(f"Testing algorithm: {algorithm_name}")
        start_time = time.time()
        
        try:
            # Run comprehensive validation
            validation_report = await self.validation_framework.run_comprehensive_validation(
                algorithm_func, 
                algorithm_name
            )
            
            # Run statistical tests on benchmark data
            statistical_results = await self._run_statistical_tests(algorithm_func)
            
            # Calculate overall performance
            overall_score = validation_report['summary']['mean_score']
            success = validation_report['summary']['success_rate'] > 0.7
            
            execution_time = (time.time() - start_time) * 1000
            
            result = AlgorithmTestResult(
                algorithm_name=algorithm_name,
                success=success,
                overall_score=overall_score,
                execution_time_ms=execution_time,
                benchmark_results=validation_report,
                statistical_results=statistical_results,
                cross_validation_results=self._extract_cv_results(validation_report),
                robustness_results=self._extract_robustness_results(validation_report)
            )
            
            logger.info(f"Algorithm {algorithm_name} completed: Score={overall_score:.3f}, Success={success}")
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            result = AlgorithmTestResult(
                algorithm_name=algorithm_name,
                success=False,
                overall_score=0.0,
                execution_time_ms=execution_time,
                benchmark_results={},
                statistical_results={},
                cross_validation_results={},
                robustness_results={},
                error_message=str(e)
            )
            
            logger.error(f"Algorithm {algorithm_name} failed: {e}")
        
        return result
    
    async def _run_statistical_tests(self, algorithm_func: Callable) -> Dict[str, Any]:
        """Run statistical tests on algorithm results"""
        statistical_results = {}
        
        # Test on each benchmark dataset
        for benchmark_name, benchmark in self.validation_framework.benchmark_datasets.items():
            try:
                # Run algorithm
                discovered_graph = algorithm_func(benchmark.data)
                
                # Extract variable pairs from discovered graph
                variable_pairs = [(edge.source, edge.target) for edge in discovered_graph.edges]
                
                if variable_pairs:
                    # Run comprehensive statistical tests
                    test_results = self.statistical_tests.run_comprehensive_statistical_tests(
                        benchmark.data,
                        variable_pairs
                    )
                    
                    # Summarize results
                    statistical_results[benchmark_name] = self._summarize_statistical_results(test_results)
                
            except Exception as e:
                statistical_results[benchmark_name] = {'error': str(e)}
        
        return statistical_results
    
    def _summarize_statistical_results(self, test_results: Dict[str, List[StatisticalTestResult]]) -> Dict[str, Any]:
        """Summarize statistical test results"""
        summary = {
            'total_tests': 0,
            'significant_tests': 0,
            'test_types': {},
            'mean_statistics': {},
            'mean_p_values': {}
        }
        
        all_results = []
        for pair_results in test_results.values():
            all_results.extend(pair_results)
        
        summary['total_tests'] = len(all_results)
        summary['significant_tests'] = sum(1 for r in all_results if r.significant)
        
        # Group by test type
        test_types = {}
        for result in all_results:
            test_type = result.test_name
            if test_type not in test_types:
                test_types[test_type] = []
            test_types[test_type].append(result)
        
        # Calculate statistics by test type
        for test_type, results in test_types.items():
            summary['test_types'][test_type] = {
                'count': len(results),
                'significant_count': sum(1 for r in results if r.significant),
                'mean_statistic': np.mean([r.statistic for r in results]),
                'mean_p_value': np.mean([r.p_value for r in results])
            }
        
        return summary
    
    def _extract_cv_results(self, validation_report: Dict[str, Any]) -> Dict[str, Any]:
        """Extract cross-validation results from validation report"""
        cv_results = {}
        
        for test_result in validation_report.get('test_results', []):
            if 'cv_' in test_result['test_name']:
                cv_results[test_result['test_name']] = {
                    'success': test_result['success'],
                    'score': test_result['score'],
                    'details': test_result.get('details', {})
                }
        
        return cv_results
    
    def _extract_robustness_results(self, validation_report: Dict[str, Any]) -> Dict[str, Any]:
        """Extract robustness results from validation report"""
        robustness_results = {}
        
        for test_result in validation_report.get('test_results', []):
            if 'robustness_' in test_result['test_name']:
                robustness_results[test_result['test_name']] = {
                    'success': test_result['success'],
                    'score': test_result['score'],
                    'details': test_result.get('details', {})
                }
        
        return robustness_results
    
    async def run_comprehensive_algorithm_comparison(self) -> Dict[str, Any]:
        """Run comprehensive comparison of all algorithms"""
        logger.info("Starting comprehensive algorithm comparison")
        
        # Test all algorithms
        for algorithm_name, algorithm_func in self.algorithms.items():
            result = await self.test_algorithm(algorithm_name, algorithm_func)
            self.algorithm_results.append(result)
        
        # Generate comparison report
        comparison_report = self._generate_comparison_report()
        
        # Save report
        report_file = self.output_dir / f"algorithm_comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(comparison_report, f, indent=2, default=str)
        
        logger.info(f"Algorithm comparison completed. Report saved to {report_file}")
        
        return comparison_report
    
    def _generate_comparison_report(self) -> Dict[str, Any]:
        """Generate comprehensive comparison report"""
        
        # Overall statistics
        total_algorithms = len(self.algorithm_results)
        successful_algorithms = sum(1 for r in self.algorithm_results if r.success)
        
        # Performance rankings
        performance_ranking = sorted(
            self.algorithm_results, 
            key=lambda x: x.overall_score, 
            reverse=True
        )
        
        # Speed rankings
        speed_ranking = sorted(
            self.algorithm_results,
            key=lambda x: x.execution_time_ms
        )
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_algorithms_tested': total_algorithms,
                'successful_algorithms': successful_algorithms,
                'failed_algorithms': total_algorithms - successful_algorithms,
                'success_rate': successful_algorithms / total_algorithms if total_algorithms > 0 else 0,
                'best_performing_algorithm': performance_ranking[0].algorithm_name if performance_ranking else None,
                'fastest_algorithm': speed_ranking[0].algorithm_name if speed_ranking else None
            },
            'performance_ranking': [
                {
                    'rank': i + 1,
                    'algorithm': result.algorithm_name,
                    'score': result.overall_score,
                    'success': result.success,
                    'execution_time_ms': result.execution_time_ms
                }
                for i, result in enumerate(performance_ranking)
            ],
            'detailed_results': [asdict(result) for result in self.algorithm_results],
            'benchmark_analysis': self._analyze_benchmark_performance(),
            'statistical_analysis': self._analyze_statistical_performance(),
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _analyze_benchmark_performance(self) -> Dict[str, Any]:
        """Analyze performance across different benchmarks"""
        benchmark_analysis = {}
        
        # Get all benchmark names
        all_benchmarks = set()
        for result in self.algorithm_results:
            if result.benchmark_results and 'test_results' in result.benchmark_results:
                for test_result in result.benchmark_results['test_results']:
                    test_name = test_result['test_name']
                    if not any(x in test_name for x in ['cv_', 'robustness_']):
                        benchmark_name = test_name.split('_')[-1]
                        all_benchmarks.add(benchmark_name)
        
        # Analyze performance per benchmark
        for benchmark in all_benchmarks:
            benchmark_scores = []
            algorithm_performance = {}
            
            for result in self.algorithm_results:
                if result.benchmark_results and 'test_results' in result.benchmark_results:
                    for test_result in result.benchmark_results['test_results']:
                        if benchmark in test_result['test_name'] and not any(x in test_result['test_name'] for x in ['cv_', 'robustness_']):
                            score = test_result.get('score', 0)
                            benchmark_scores.append(score)
                            algorithm_performance[result.algorithm_name] = score
            
            if benchmark_scores:
                benchmark_analysis[benchmark] = {
                    'mean_score': np.mean(benchmark_scores),
                    'std_score': np.std(benchmark_scores),
                    'best_algorithm': max(algorithm_performance.items(), key=lambda x: x[1])[0] if algorithm_performance else None,
                    'algorithm_scores': algorithm_performance
                }
        
        return benchmark_analysis
    
    def _analyze_statistical_performance(self) -> Dict[str, Any]:
        """Analyze statistical test performance"""
        statistical_analysis = {
            'granger_causality': {'total': 0, 'significant': 0},
            'transfer_entropy': {'total': 0, 'significant': 0},
            'causal_strength': {'total': 0, 'significant': 0},
            'conditional_independence': {'total': 0, 'significant': 0}
        }
        
        for result in self.algorithm_results:
            if result.statistical_results:
                for benchmark_name, stats in result.statistical_results.items():
                    if 'test_types' in stats:
                        for test_type, test_stats in stats['test_types'].items():
                            if 'granger' in test_type.lower():
                                statistical_analysis['granger_causality']['total'] += test_stats['count']
                                statistical_analysis['granger_causality']['significant'] += test_stats['significant_count']
                            elif 'transfer' in test_type.lower():
                                statistical_analysis['transfer_entropy']['total'] += test_stats['count']
                                statistical_analysis['transfer_entropy']['significant'] += test_stats['significant_count']
                            elif 'strength' in test_type.lower():
                                statistical_analysis['causal_strength']['total'] += test_stats['count']
                                statistical_analysis['causal_strength']['significant'] += test_stats['significant_count']
                            elif 'independence' in test_type.lower():
                                statistical_analysis['conditional_independence']['total'] += test_stats['count']
                                statistical_analysis['conditional_independence']['significant'] += test_stats['significant_count']
        
        # Calculate significance rates
        for test_type in statistical_analysis:
            total = statistical_analysis[test_type]['total']
            significant = statistical_analysis[test_type]['significant']
            statistical_analysis[test_type]['significance_rate'] = significant / total if total > 0 else 0
        
        return statistical_analysis
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        if not self.algorithm_results:
            recommendations.append("No algorithm results available for analysis")
            return recommendations
        
        # Performance recommendations
        best_algorithm = max(self.algorithm_results, key=lambda x: x.overall_score)
        worst_algorithm = min(self.algorithm_results, key=lambda x: x.overall_score)
        
        recommendations.append(
            f"Best performing algorithm: {best_algorithm.algorithm_name} "
            f"(score: {best_algorithm.overall_score:.3f})"
        )
        
        if worst_algorithm.overall_score < 0.5:
            recommendations.append(
                f"Consider improving or replacing {worst_algorithm.algorithm_name} "
                f"(score: {worst_algorithm.overall_score:.3f})"
            )
        
        # Speed recommendations
        fastest_algorithm = min(self.algorithm_results, key=lambda x: x.execution_time_ms)
        slowest_algorithm = max(self.algorithm_results, key=lambda x: x.execution_time_ms)
        
        if slowest_algorithm.execution_time_ms > 10000:  # > 10 seconds
            recommendations.append(
                f"Optimize {slowest_algorithm.algorithm_name} for speed "
                f"(execution time: {slowest_algorithm.execution_time_ms/1000:.1f}s)"
            )
        
        # Success rate recommendations
        failed_algorithms = [r for r in self.algorithm_results if not r.success]
        if failed_algorithms:
            recommendations.append(
                f"Fix implementation issues in: {', '.join(r.algorithm_name for r in failed_algorithms)}"
            )
        
        # Statistical significance recommendations
        low_significance_algorithms = []
        for result in self.algorithm_results:
            if result.statistical_results:
                total_significant = 0
                total_tests = 0
                
                for stats in result.statistical_results.values():
                    if 'total_tests' in stats and 'significant_tests' in stats:
                        total_tests += stats['total_tests']
                        total_significant += stats['significant_tests']
                
                if total_tests > 0 and total_significant / total_tests < 0.3:
                    low_significance_algorithms.append(result.algorithm_name)
        
        if low_significance_algorithms:
            recommendations.append(
                f"Improve statistical significance for: {', '.join(low_significance_algorithms)}"
            )
        
        return recommendations
    
    async def run_single_algorithm_test(self, algorithm_name: str) -> Dict[str, Any]:
        """Run test for a single algorithm"""
        if algorithm_name not in self.algorithms:
            raise ValueError(f"Algorithm {algorithm_name} not found")
        
        algorithm_func = self.algorithms[algorithm_name]
        result = await self.test_algorithm(algorithm_name, algorithm_func)
        
        # Generate single algorithm report
        report = {
            'timestamp': datetime.now().isoformat(),
            'algorithm_name': algorithm_name,
            'result': asdict(result),
            'recommendations': self._generate_single_algorithm_recommendations(result)
        }
        
        # Save report
        report_file = self.output_dir / f"{algorithm_name}_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return report
    
    def _generate_single_algorithm_recommendations(self, result: AlgorithmTestResult) -> List[str]:
        """Generate recommendations for a single algorithm"""
        recommendations = []
        
        if not result.success:
            recommendations.append(f"Algorithm {result.algorithm_name} failed - check implementation")
            if result.error_message:
                recommendations.append(f"Error: {result.error_message}")
        
        if result.overall_score < 0.7:
            recommendations.append(f"Low overall score ({result.overall_score:.3f}) - consider algorithm improvements")
        
        if result.execution_time_ms > 5000:
            recommendations.append(f"High execution time ({result.execution_time_ms/1000:.1f}s) - optimize for speed")
        
        # Check cross-validation stability
        if result.cross_validation_results:
            for cv_test, cv_result in result.cross_validation_results.items():
                if 'details' in cv_result and 'cv_std' in cv_result['details']:
                    cv_std = cv_result['details']['cv_std']
                    if cv_std > 0.2:
                        recommendations.append(f"High cross-validation variance ({cv_std:.3f}) - improve stability")
        
        # Check robustness
        if result.robustness_results:
            for rob_test, rob_result in result.robustness_results.items():
                if rob_result['score'] < 0.6:
                    recommendations.append(f"Low robustness score - improve noise tolerance")
        
        return recommendations


# CLI interface
async def main():
    """Main CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run causal analysis validation tests')
    parser.add_argument('--algorithm', help='Test specific algorithm only')
    parser.add_argument('--output-dir', default='causal_validation_results', help='Output directory')
    parser.add_argument('--compare-all', action='store_true', help='Compare all algorithms')
    
    args = parser.parse_args()
    
    # Initialize test runner
    runner = CausalAnalysisTestRunner(output_dir=args.output_dir)
    
    try:
        if args.algorithm:
            # Test single algorithm
            report = await runner.run_single_algorithm_test(args.algorithm)
            print(f"\nSingle Algorithm Test Results:")
            print(f"Algorithm: {args.algorithm}")
            print(f"Success: {report['result']['success']}")
            print(f"Score: {report['result']['overall_score']:.3f}")
            print(f"Execution Time: {report['result']['execution_time_ms']/1000:.1f}s")
            
        elif args.compare_all:
            # Compare all algorithms
            report = await runner.run_comprehensive_algorithm_comparison()
            print(f"\nAlgorithm Comparison Results:")
            print(f"Total Algorithms: {report['summary']['total_algorithms_tested']}")
            print(f"Success Rate: {report['summary']['success_rate']:.2%}")
            print(f"Best Algorithm: {report['summary']['best_performing_algorithm']}")
            print(f"Fastest Algorithm: {report['summary']['fastest_algorithm']}")
            
            print(f"\nPerformance Ranking:")
            for ranking in report['performance_ranking'][:3]:  # Top 3
                print(f"{ranking['rank']}. {ranking['algorithm']} (Score: {ranking['score']:.3f})")
        
        else:
            print("Please specify --algorithm <name> or --compare-all")
            return
        
        print(f"\nDetailed reports saved to: {runner.output_dir}")
        
    except Exception as e:
        print(f"Test execution failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))
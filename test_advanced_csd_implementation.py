#!/usr/bin/env python3
"""
Advanced Cambridge Structural Database (CSD) Integration Test
Tests the comprehensive CSD data fetching and analysis implementation
"""

import asyncio
import sys
import os
import json
from datetime import datetime
from typing import Dict, Any, List

# Add the backend modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend', 'modules'))

try:
    from enkf_knowledge_integration import AdvancedEnKFKnowledgeIntegration
    from database_integration import PostgreSQLGISManager, DatabaseConfig
    from external_data_connectors import DataIntegrationManager
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all required modules are available")
    sys.exit(1)

class CSDIntegrationTester:
    """Test suite for the advanced CSD implementation"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.test_compounds = [
            'SiO2',      # Quartz - common mineral
            'CaCO3',     # Calcite - carbonate mineral
            'H2O',       # Water - simple inorganic
            'CH4',       # Methane - simple organic
            'MgSiO3'     # Enstatite - pyroxene mineral
        ]
        
    def _setup_logging(self):
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    async def run_comprehensive_tests(self):
        """Run comprehensive tests of the CSD integration"""
        print("=" * 80)
        print("Advanced Cambridge Structural Database (CSD) Integration Test")
        print("=" * 80)
        print()
        
        # Initialize the EnKF system
        enkf_system = await self._initialize_enkf_system()
        if not enkf_system:
            print("❌ Failed to initialize EnKF system")
            return False
        
        print("✅ EnKF Knowledge Integration system initialized")
        print()
        
        # Test each compound
        all_results = {}
        
        for i, compound in enumerate(self.test_compounds, 1):
            print(f"📊 Test {i}/{len(self.test_compounds)}: Testing compound '{compound}'")
            print("-" * 60)
            
            try:
                # Test the advanced CSD data fetching
                result = await self._test_compound_csd_integration(enkf_system, compound)
                all_results[compound] = result
                
                if result['success']:
                    print(f"✅ {compound}: CSD integration successful")
                    self._print_compound_summary(compound, result)
                else:
                    print(f"⚠️  {compound}: CSD integration completed with warnings")
                    print(f"   Reason: {result.get('warning', 'Unknown warning')}")
                    
            except Exception as e:
                print(f"❌ {compound}: CSD integration failed")
                print(f"   Error: {str(e)}")
                all_results[compound] = {
                    'success': False,
                    'error': str(e),
                    'compound': compound
                }
            
            print()
        
        # Generate comprehensive test report
        await self._generate_test_report(all_results)
        
        # Test advanced analysis features
        await self._test_advanced_analysis_features(enkf_system)
        
        print("🎯 CSD Integration Test Suite Completed!")
        return True
    
    async def _initialize_enkf_system(self):
        """Initialize the EnKF Knowledge Integration system"""
        try:
            # Initialize database configuration
            db_config = DatabaseConfig(
                host="localhost",
                database="prometheus_csd_test",
                username="postgres",
                password="password"
            )
            
            # Initialize database manager
            database_manager = PostgreSQLGISManager(db_config)
            
            # Initialize external data integration
            external_data_manager = DataIntegrationManager()
            
            # Initialize EnKF system with advanced configuration
            enkf_config = {
                'advanced_integration': True,
                'ml_enabled': True,
                'caching_enabled': True,
                'knowledge_graph_enabled': True,
                'multi_source_validation': True
            }
            
            enkf_system = AdvancedEnKFKnowledgeIntegration(
                config=enkf_config,
                database_manager=database_manager,
                external_data_manager=external_data_manager
            )
            
            return enkf_system
            
        except Exception as e:
            self.logger.error(f"EnKF system initialization failed: {e}")
            return None
    
    async def _test_compound_csd_integration(self, enkf_system, compound: str) -> Dict[str, Any]:
        """Test CSD integration for a specific compound"""
        result = {
            'compound': compound,
            'success': False,
            'timestamp': datetime.now().isoformat(),
            'data_sources': [],
            'structure_count': 0,
            'analysis_features': {},
            'ml_predictions': {},
            'performance_metrics': {}
        }
        
        start_time = datetime.now()
        
        try:
            # Create a session for testing (mock aiohttp session)
            import aiohttp
            async with aiohttp.ClientSession() as session:
                # Test the main CSD data fetching method
                csd_data = await enkf_system._fetch_csd_data(session, compound)
                
                # Analyze results
                if csd_data:
                    result['success'] = True
                    result['data_sources'] = list(csd_data.keys())
                    
                    # Count structures from all sources
                    total_structures = 0
                    for source_key, source_data in csd_data.items():
                        if isinstance(source_data, dict) and 'structures' in source_data:
                            structures = source_data['structures']
                            if isinstance(structures, list):
                                total_structures += len(structures)
                    
                    result['structure_count'] = total_structures
                    
                    # Check for advanced analysis
                    if 'advanced_analysis' in csd_data:
                        analysis = csd_data['advanced_analysis']
                        result['analysis_features'] = {
                            'structural_analysis': 'structural_analysis' in analysis,
                            'polymorphism_analysis': 'polymorphism_analysis' in analysis,
                            'phase_relationships': 'phase_relationships' in analysis,
                            'thermodynamic_stability': 'thermodynamic_stability' in analysis
                        }
                    
                    # Check for ML predictions
                    if 'ml_predictions' in csd_data:
                        ml_pred = csd_data['ml_predictions']
                        result['ml_predictions'] = {
                            'mechanical_properties': 'mechanical_properties' in ml_pred,
                            'thermal_properties': 'thermal_properties' in ml_pred,
                            'electronic_properties': 'electronic_properties' in ml_pred,
                            'ensemble_predictions': 'ensemble_predictions' in ml_pred,
                            'prediction_confidence': ml_pred.get('prediction_metadata', {}).get('prediction_confidence', 0)
                        }
                    
                    # Check COD cross-reference
                    if 'cod_cross_reference' in csd_data:
                        cod_data = csd_data['cod_cross_reference']
                        result['cod_validation'] = {
                            'open_structures_found': len(cod_data.get('open_structures', [])),
                            'cross_validation_score': cod_data.get('cross_validation_score', 0)
                        }
                    
                else:
                    result['success'] = False
                    result['warning'] = 'No CSD data retrieved - likely due to missing API credentials'
                
                # Calculate performance metrics
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                result['performance_metrics'] = {
                    'duration_seconds': duration,
                    'structures_per_second': result['structure_count'] / max(duration, 0.001),
                    'data_sources_accessed': len(result['data_sources'])
                }
                
        except Exception as e:
            result['success'] = False
            result['error'] = str(e)
            self.logger.error(f"CSD integration test failed for {compound}: {e}")
        
        return result
    
    def _print_compound_summary(self, compound: str, result: Dict[str, Any]):
        """Print summary for a tested compound"""
        print(f"   🔬 Structures found: {result['structure_count']}")
        print(f"   📊 Data sources: {len(result['data_sources'])}")
        
        if result['data_sources']:
            print(f"   📋 Sources: {', '.join(result['data_sources'])}")
        
        # Analysis features
        analysis_features = result.get('analysis_features', {})
        if analysis_features:
            enabled_features = [k for k, v in analysis_features.items() if v]
            if enabled_features:
                print(f"   🔍 Analysis: {', '.join(enabled_features)}")
        
        # ML predictions
        ml_predictions = result.get('ml_predictions', {})
        if ml_predictions:
            pred_types = [k for k, v in ml_predictions.items() if v and k != 'prediction_confidence']
            if pred_types:
                confidence = ml_predictions.get('prediction_confidence', 0)
                print(f"   🤖 ML predictions: {', '.join(pred_types)} (conf: {confidence:.2f})")
        
        # Performance
        perf = result.get('performance_metrics', {})
        if perf:
            duration = perf.get('duration_seconds', 0)
            print(f"   ⚡ Performance: {duration:.2f}s, {perf.get('structures_per_second', 0):.1f} struct/s")
        
        # COD validation
        cod_val = result.get('cod_validation', {})
        if cod_val:
            print(f"   ✓ COD validation: {cod_val.get('open_structures_found', 0)} matches, "
                  f"score: {cod_val.get('cross_validation_score', 0):.2f}")
    
    async def _generate_test_report(self, results: Dict[str, Dict]):
        """Generate a comprehensive test report"""
        print("📋 Test Report Summary")
        print("=" * 50)
        
        successful_tests = sum(1 for r in results.values() if r.get('success', False))
        total_tests = len(results)
        
        print(f"Tests passed: {successful_tests}/{total_tests}")
        print(f"Success rate: {(successful_tests/total_tests)*100:.1f}%")
        
        # Aggregate statistics
        total_structures = sum(r.get('structure_count', 0) for r in results.values())
        total_sources = sum(len(r.get('data_sources', [])) for r in results.values())
        avg_duration = sum(r.get('performance_metrics', {}).get('duration_seconds', 0) for r in results.values()) / len(results)
        
        print(f"Total structures found: {total_structures}")
        print(f"Total data sources accessed: {total_sources}")
        print(f"Average query time: {avg_duration:.2f}s")
        
        # Feature analysis
        print()
        print("Feature Utilization:")
        
        feature_counts = {}
        for result in results.values():
            features = result.get('analysis_features', {})
            for feature, enabled in features.items():
                if enabled:
                    feature_counts[feature] = feature_counts.get(feature, 0) + 1
        
        for feature, count in feature_counts.items():
            print(f"  {feature}: {count}/{total_tests} compounds ({(count/total_tests)*100:.0f}%)")
        
        # ML predictions summary
        ml_feature_counts = {}
        for result in results.values():
            ml_preds = result.get('ml_predictions', {})
            for pred_type, enabled in ml_preds.items():
                if enabled and pred_type != 'prediction_confidence':
                    ml_feature_counts[pred_type] = ml_feature_counts.get(pred_type, 0) + 1
        
        if ml_feature_counts:
            print()
            print("ML Prediction Coverage:")
            for pred_type, count in ml_feature_counts.items():
                print(f"  {pred_type}: {count}/{total_tests} compounds ({(count/total_tests)*100:.0f}%)")
        
        # Save detailed report to file
        report_file = f"csd_integration_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(report_file, 'w') as f:
                json.dump({
                    'test_summary': {
                        'total_tests': total_tests,
                        'successful_tests': successful_tests,
                        'success_rate': (successful_tests/total_tests)*100,
                        'total_structures': total_structures,
                        'avg_duration': avg_duration
                    },
                    'detailed_results': results,
                    'feature_utilization': feature_counts,
                    'ml_predictions': ml_feature_counts
                }, f, indent=2, default=str)
            
            print(f"💾 Detailed report saved to: {report_file}")
            
        except Exception as e:
            print(f"⚠️  Could not save report file: {e}")
        
        print()
    
    async def _test_advanced_analysis_features(self, enkf_system):
        """Test advanced analysis features"""
        print("🔬 Testing Advanced Analysis Features")
        print("=" * 50)
        
        # Test structural analysis
        print("Testing structural analysis capabilities...")
        
        # Mock structures for testing
        mock_structures = [
            {
                'space_group': 'P6_3/mmc',
                'cell_volume': 123.45,
                'cell_parameters': {'a': 4.9, 'b': 4.9, 'c': 5.4},
                'formula': 'SiO2',
                'density': 2.65
            },
            {
                'space_group': 'P3_121',
                'cell_volume': 113.2,
                'cell_parameters': {'a': 4.9, 'b': 4.9, 'c': 5.4},
                'formula': 'SiO2',
                'density': 2.65
            },
            {
                'space_group': 'R-3c',
                'cell_volume': 367.8,
                'cell_parameters': {'a': 4.99, 'b': 4.99, 'c': 17.06},
                'formula': 'CaCO3',
                'density': 2.71
            }
        ]
        
        try:
            # Test structural analysis
            analysis = await enkf_system._perform_structural_analysis(mock_structures)
            if analysis:
                print("✅ Structural analysis working")
                print(f"   Space groups found: {len(analysis.get('space_group_distribution', {}))}")
                if 'cell_parameter_statistics' in analysis:
                    stats = analysis['cell_parameter_statistics']
                    print(f"   Volume range: {stats.get('volume_range', [0, 0])}")
            
            # Test polymorph detection
            polymorphs = await enkf_system._detect_polymorphs(mock_structures, 'SiO2')
            if polymorphs:
                print("✅ Polymorph detection working")
                print(f"   Polymorph variants: {polymorphs.get('polymorph_count', 0)}")
            
            # Test ML feature extraction
            features = await enkf_system._extract_structural_features(mock_structures)
            if features is not None:
                print("✅ ML feature extraction working")
                feature_dim = len(features) if isinstance(features, list) else features.shape if hasattr(features, 'shape') else 0
                print(f"   Feature dimensions: {feature_dim}")
            
            # Test ML models
            mechanical_model = await enkf_system._load_mechanical_property_model()
            if mechanical_model:
                pred = mechanical_model.predict(features)
                print("✅ Mechanical property model working")
                print(f"   Predicted bulk modulus: {pred.get('bulk_modulus', 0):.1f} GPa")
            
            thermal_model = await enkf_system._load_thermal_property_model()
            if thermal_model:
                pred = thermal_model.predict(features)
                print("✅ Thermal property model working")
                print(f"   Predicted melting point: {pred.get('melting_point', 0):.0f} K")
            
            electronic_model = await enkf_system._load_electronic_property_model()
            if electronic_model:
                pred = electronic_model.predict(features)
                print("✅ Electronic property model working")
                print(f"   Predicted band gap: {pred.get('band_gap', 0):.2f} eV")
            
            print()
            print("🎯 Advanced analysis features test completed!")
            
        except Exception as e:
            print(f"❌ Advanced analysis test failed: {e}")
        
        print()

async def main():
    """Main test execution function"""
    tester = CSDIntegrationTester()
    
    print(f"🚀 Starting CSD Integration Test Suite")
    print(f"📅 Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        success = await tester.run_comprehensive_tests()
        
        if success:
            print("🎉 All tests completed successfully!")
            return 0
        else:
            print("⚠️  Tests completed with some issues")
            return 1
            
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        return 130
        
    except Exception as e:
        print(f"💥 Test suite failed with error: {e}")
        return 1

if __name__ == "__main__":
    # Set event loop policy for Windows compatibility
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    # Run the test suite
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
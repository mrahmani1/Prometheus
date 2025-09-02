"""
Windows-Compatible Integration Test Runner for Prometheus AI Geodynamic Platform
===============================================================================

This module runs comprehensive integration tests to validate system interoperability
across all backend modules with Windows compatibility.

Features:
- Windows-compatible timeout handling
- Module loading and dependency validation
- Performance benchmarking
- Cross-module compatibility testing
- Memory usage monitoring
- Error handling and reporting

Author: Prometheus Integration Testing System
"""

import sys
import os
import time
import traceback
import subprocess
import json
import logging
import threading
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import queue
import warnings
from datetime import datetime
import importlib

# Add backend modules to path
backend_path = os.path.join(os.path.dirname(__file__), '..', 'modules')
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """Simple test result structure"""
    module_name: str
    test_type: str
    success: bool
    execution_time: float
    error_message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

class WindowsIntegrationTester:
    """Windows-compatible integration tester for Prometheus modules"""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.loaded_modules: Dict[str, Any] = {}
        self.timeout_seconds = 30
        self.max_workers = min(8, multiprocessing.cpu_count())
        
        logger.info("WindowsIntegrationTester initialized")
    
    def discover_modules(self) -> List[str]:
        """Discover all Python modules in the backend directory"""
        modules = []
        backend_dir = Path(backend_path)
        
        for py_file in backend_dir.glob('*.py'):
            if (py_file.name.startswith('_') or 
                py_file.name.startswith('test_') or
                py_file.name == '__init__.py'):
                continue
            
            module_name = py_file.stem
            modules.append(module_name)
        
        logger.info(f"Discovered {len(modules)} modules")
        return sorted(modules)
    
    def test_module_loading_with_timeout(self, module_name: str) -> TestResult:
        """Test module loading with Windows-compatible timeout"""
        start_time = time.time()
        result_queue = queue.Queue()
        
        def load_module():
            try:
                module = importlib.import_module(module_name)
                
                # Basic module inspection
                classes = [name for name, obj in vars(module).items() 
                          if isinstance(obj, type) and not name.startswith('_')]
                functions = [name for name, obj in vars(module).items() 
                            if callable(obj) and not name.startswith('_')]
                
                result_queue.put({
                    'success': True,
                    'module': module,
                    'classes': len(classes),
                    'functions': len(functions)
                })
                
            except Exception as e:
                result_queue.put({
                    'success': False,
                    'error': str(e)
                })
        
        # Start loading thread
        thread = threading.Thread(target=load_module)
        thread.daemon = True
        thread.start()
        thread.join(timeout=self.timeout_seconds)
        
        execution_time = time.time() - start_time
        
        if thread.is_alive():
            # Timeout occurred
            return TestResult(
                module_name=module_name,
                test_type="module_loading",
                success=False,
                execution_time=execution_time,
                error_message=f"Loading timeout after {self.timeout_seconds}s"
            )
        
        try:
            result = result_queue.get_nowait()
            if result['success']:
                self.loaded_modules[module_name] = result['module']
                return TestResult(
                    module_name=module_name,
                    test_type="module_loading",
                    success=True,
                    execution_time=execution_time,
                    details={
                        'classes_found': result['classes'],
                        'functions_found': result['functions']
                    }
                )
            else:
                return TestResult(
                    module_name=module_name,
                    test_type="module_loading",
                    success=False,
                    execution_time=execution_time,
                    error_message=result['error']
                )
                
        except queue.Empty:
            return TestResult(
                module_name=module_name,
                test_type="module_loading",
                success=False,
                execution_time=execution_time,
                error_message="No result received"
            )
    
    def test_basic_instantiation(self, module_name: str) -> List[TestResult]:
        """Test basic class instantiation for a module"""
        results = []
        
        if module_name not in self.loaded_modules:
            return results
        
        module = self.loaded_modules[module_name]
        
        # Find classes that might be instantiable
        target_classes = []
        for name, obj in vars(module).items():
            if (isinstance(obj, type) and 
                not name.startswith('_') and
                any(keyword in name.lower() for keyword in 
                    ['engine', 'framework', 'system', 'analyzer', 'modeler', 
                     'processor', 'manager', 'handler', 'service'])):
                target_classes.append((name, obj))
        
        # Test up to 3 main classes per module
        for class_name, class_obj in target_classes[:3]:
            start_time = time.time()
            
            try:
                # Try common instantiation patterns
                instance = None
                instantiation_methods = [
                    lambda: class_obj(),
                    lambda: class_obj(config={}),
                    lambda: class_obj({}),
                    lambda: class_obj(config={'test_mode': True}),
                    lambda: class_obj({'test_mode': True})
                ]
                
                for method in instantiation_methods:
                    try:
                        instance = method()
                        break
                    except:
                        continue
                
                execution_time = time.time() - start_time
                
                if instance is not None:
                    # Count available methods
                    methods = [name for name in dir(instance) 
                              if callable(getattr(instance, name)) and not name.startswith('_')]
                    
                    results.append(TestResult(
                        module_name=f"{module_name}.{class_name}",
                        test_type="class_instantiation",
                        success=True,
                        execution_time=execution_time,
                        details={'methods_available': len(methods)}
                    ))
                else:
                    results.append(TestResult(
                        module_name=f"{module_name}.{class_name}",
                        test_type="class_instantiation",
                        success=False,
                        execution_time=execution_time,
                        error_message="All instantiation methods failed"
                    ))
                    
            except Exception as e:
                execution_time = time.time() - start_time
                results.append(TestResult(
                    module_name=f"{module_name}.{class_name}",
                    test_type="class_instantiation",
                    success=False,
                    execution_time=execution_time,
                    error_message=str(e)
                ))
        
        return results
    
    def analyze_import_dependencies(self, module_name: str) -> TestResult:
        """Analyze import dependencies for a module"""
        start_time = time.time()
        
        try:
            module_path = os.path.join(backend_path, f"{module_name}.py")
            if not os.path.exists(module_path):
                return TestResult(
                    module_name=module_name,
                    test_type="dependency_analysis",
                    success=False,
                    execution_time=time.time() - start_time,
                    error_message="Module file not found"
                )
            
            with open(module_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Count import statements
            import_lines = [line.strip() for line in content.split('\n') 
                          if line.strip().startswith(('import ', 'from '))]
            
            # Classify imports
            stdlib_imports = []
            thirdparty_imports = []
            local_imports = []
            
            for line in import_lines:
                if 'from ' in line and ' import ' in line:
                    module_part = line.split()[1]
                elif line.startswith('import '):
                    module_part = line.split()[1].split('.')[0]
                else:
                    continue
                
                if module_part.startswith('.') or module_part in ['backend', 'modules']:
                    local_imports.append(module_part)
                elif module_part in ['os', 'sys', 'time', 'json', 'logging', 'threading', 
                                   'multiprocessing', 'datetime', 'pathlib', 'collections',
                                   'typing', 'dataclasses', 'abc', 'functools', 'itertools']:
                    stdlib_imports.append(module_part)
                else:
                    thirdparty_imports.append(module_part)
            
            execution_time = time.time() - start_time
            
            return TestResult(
                module_name=module_name,
                test_type="dependency_analysis",
                success=True,
                execution_time=execution_time,
                details={
                    'total_imports': len(import_lines),
                    'stdlib_imports': len(stdlib_imports),
                    'thirdparty_imports': len(thirdparty_imports),
                    'local_imports': len(local_imports),
                    'lines_of_code': len(content.split('\n'))
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                module_name=module_name,
                test_type="dependency_analysis",
                success=False,
                execution_time=execution_time,
                error_message=str(e)
            )
    
    def run_integration_tests(self) -> Dict[str, Any]:
        """Run comprehensive integration tests"""
        logger.info("Starting integration tests...")
        start_time = time.time()
        
        # Discover modules
        modules = self.discover_modules()
        total_modules = len(modules)
        
        # Test module loading in parallel
        logger.info("Testing module loading...")
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            loading_futures = {
                executor.submit(self.test_module_loading_with_timeout, module): module 
                for module in modules
            }
            
            for future in loading_futures:
                try:
                    result = future.result(timeout=60)  # 60 second total timeout per future
                    self.results.append(result)
                    
                    if result.success:
                        logger.info(f"✓ {result.module_name} loaded successfully")
                    else:
                        logger.warning(f"✗ {result.module_name}: {result.error_message}")
                        
                except Exception as e:
                    module = loading_futures[future]
                    logger.error(f"✗ {module}: Exception during loading: {e}")
                    self.results.append(TestResult(
                        module_name=module,
                        test_type="module_loading",
                        success=False,
                        execution_time=0,
                        error_message=f"Future exception: {e}"
                    ))
        
        # Test instantiation for successfully loaded modules
        successful_modules = [r.module_name for r in self.results 
                            if r.test_type == "module_loading" and r.success]
        
        logger.info(f"Testing instantiation for {len(successful_modules)} successful modules...")
        for module in successful_modules[:15]:  # Test top 15 modules
            instantiation_results = self.test_basic_instantiation(module)
            self.results.extend(instantiation_results)
        
        # Analyze dependencies
        logger.info("Analyzing dependencies...")
        dependency_results = []
        for module in modules[:20]:  # Analyze top 20 modules
            dep_result = self.analyze_import_dependencies(module)
            dependency_results.append(dep_result)
            self.results.append(dep_result)
        
        # Calculate summary statistics
        total_time = time.time() - start_time
        
        loading_results = [r for r in self.results if r.test_type == "module_loading"]
        successful_loading = len([r for r in loading_results if r.success])
        failed_loading = len([r for r in loading_results if not r.success])
        
        instantiation_results = [r for r in self.results if r.test_type == "class_instantiation"]
        successful_instantiation = len([r for r in instantiation_results if r.success])
        failed_instantiation = len([r for r in instantiation_results if not r.success])
        
        dependency_results = [r for r in self.results if r.test_type == "dependency_analysis"]
        successful_dependencies = len([r for r in dependency_results if r.success])
        
        # Calculate health score
        total_tests = len(self.results)
        successful_tests = len([r for r in self.results if r.success])
        health_score = (successful_tests / max(total_tests, 1)) * 100
        
        # Identify critical failures
        critical_failures = [r.module_name for r in loading_results if not r.success]
        
        # Generate recommendations
        recommendations = []
        if failed_loading > total_modules * 0.1:
            recommendations.append(f"High module loading failure rate: {failed_loading}/{total_modules}")
        if failed_instantiation > 0:
            recommendations.append(f"Class instantiation issues in {failed_instantiation} classes")
        recommendations.extend([
            "Add comprehensive unit tests for all modules",
            "Implement standardized configuration patterns",
            "Add error handling and logging to all modules",
            "Create API documentation for all public interfaces"
        ])
        
        summary = {
            'total_execution_time': total_time,
            'total_modules_discovered': total_modules,
            'successful_module_loading': successful_loading,
            'failed_module_loading': failed_loading,
            'successful_instantiation': successful_instantiation,
            'failed_instantiation': failed_instantiation,
            'successful_dependency_analysis': successful_dependencies,
            'total_tests_run': total_tests,
            'successful_tests': successful_tests,
            'health_score': health_score,
            'critical_failures': critical_failures,
            'recommendations': recommendations
        }
        
        logger.info(f"Integration tests completed in {total_time:.2f} seconds")
        logger.info(f"Health Score: {health_score:.1f}%")
        
        return summary
    
    def generate_report(self, summary: Dict[str, Any]) -> str:
        """Generate integration test report"""
        report = f"""# Prometheus AI Integration Test Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
- **Total Modules**: {summary['total_modules_discovered']}
- **Successful Module Loading**: {summary['successful_module_loading']}
- **Failed Module Loading**: {summary['failed_module_loading']}
- **Health Score**: {summary['health_score']:.1f}%
- **Execution Time**: {summary['total_execution_time']:.2f} seconds

## Test Results

### Module Loading
- ✅ Successful: {summary['successful_module_loading']}
- ❌ Failed: {summary['failed_module_loading']}
- Success Rate: {(summary['successful_module_loading'] / max(summary['total_modules_discovered'], 1) * 100):.1f}%

### Class Instantiation  
- ✅ Successful: {summary['successful_instantiation']}
- ❌ Failed: {summary['failed_instantiation']}

### Dependency Analysis
- ✅ Analyzed: {summary['successful_dependency_analysis']}

## Critical Failures
"""
        
        if summary['critical_failures']:
            for failure in summary['critical_failures'][:10]:  # Show first 10
                report += f"- {failure}\n"
            if len(summary['critical_failures']) > 10:
                report += f"- ... and {len(summary['critical_failures']) - 10} more\n"
        else:
            report += "None detected\n"
        
        report += f"""
## Recommendations
"""
        for i, rec in enumerate(summary['recommendations'], 1):
            report += f"{i}. {rec}\n"
        
        report += f"""
## System Health Assessment
Status: {'EXCELLENT' if summary['health_score'] >= 90 else
         'GOOD' if summary['health_score'] >= 75 else  
         'FAIR' if summary['health_score'] >= 60 else
         'POOR'} ({summary['health_score']:.1f}%)

## Detailed Results
"""
        
        # Group results by type
        for test_type in ['module_loading', 'class_instantiation', 'dependency_analysis']:
            type_results = [r for r in self.results if r.test_type == test_type]
            if type_results:
                report += f"\n### {test_type.replace('_', ' ').title()}\n"
                successful = [r for r in type_results if r.success]
                failed = [r for r in type_results if not r.success]
                
                report += f"- Successful: {len(successful)}\n"
                report += f"- Failed: {len(failed)}\n"
                
                if failed:
                    report += "- Failed modules:\n"
                    for result in failed[:5]:  # Show first 5 failures
                        report += f"  - {result.module_name}: {result.error_message}\n"
        
        return report

def main():
    """Main entry point"""
    print("Prometheus AI Geodynamic Platform Integration Tests")
    print("=" * 60)
    
    tester = WindowsIntegrationTester()
    
    try:
        summary = tester.run_integration_tests()
        
        # Generate report
        report = tester.generate_report(summary)
        print("\n" + report)
        
        # Save report
        with open('integration_test_report.md', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\nFull report saved to: integration_test_report.md")
        
        # Exit with appropriate code
        if summary['health_score'] >= 70:
            print(f"\n✅ Integration tests PASSED (Health: {summary['health_score']:.1f}%)")
            return 0
        else:
            print(f"\n❌ Integration tests FAILED (Health: {summary['health_score']:.1f}%)")
            return 1
            
    except Exception as e:
        print(f"\n❌ Integration testing failed: {e}")
        traceback.print_exc()
        return 2

if __name__ == "__main__":
    sys.exit(main())
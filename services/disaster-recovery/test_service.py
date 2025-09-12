#!/usr/bin/env python3
"""
Simple test script for Multi-Cloud Disaster Recovery Service
"""

import asyncio
import json
import sys
from datetime import datetime
from typing import Dict, Any

import aiohttp
import structlog

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

class DisasterRecoveryTester:
    """Test client for disaster recovery service"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def test_health_endpoints(self) -> Dict[str, Any]:
        """Test health and readiness endpoints"""
        results = {}
        
        # Test health endpoint
        try:
            async with self.session.get(f"{self.base_url}/health") as response:
                results["health"] = {
                    "status_code": response.status,
                    "response": await response.json() if response.status == 200 else None,
                    "success": response.status == 200
                }
        except Exception as e:
            results["health"] = {"success": False, "error": str(e)}
        
        # Test readiness endpoint
        try:
            async with self.session.get(f"{self.base_url}/ready") as response:
                results["ready"] = {
                    "status_code": response.status,
                    "response": await response.json() if response.status == 200 else None,
                    "success": response.status == 200
                }
        except Exception as e:
            results["ready"] = {"success": False, "error": str(e)}
        
        return results
    
    async def test_system_status(self) -> Dict[str, Any]:
        """Test system status endpoint"""
        try:
            async with self.session.get(f"{self.base_url}/disaster-recovery/status") as response:
                return {
                    "status_code": response.status,
                    "response": await response.json() if response.status == 200 else None,
                    "success": response.status == 200
                }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def test_create_replication(self) -> Dict[str, Any]:
        """Test creating a replication job"""
        payload = {
            "operation_type": "create_replication",
            "source_provider": "gcp",
            "target_provider": "azure",
            "resource_ids": ["test-resource-1"],
            "parameters": {
                "source_location": "gs://test-source-bucket",
                "target_location": "https://test.blob.core.windows.net/container",
                "sync_mode": "real_time",
                "encryption": True
            }
        }
        
        try:
            async with self.session.post(
                f"{self.base_url}/disaster-recovery/replicate",
                json=payload
            ) as response:
                return {
                    "status_code": response.status,
                    "response": await response.json() if response.content_type == "application/json" else None,
                    "success": response.status in [200, 201]
                }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def test_list_replications(self) -> Dict[str, Any]:
        """Test listing replications"""
        try:
            async with self.session.get(f"{self.base_url}/disaster-recovery/replications") as response:
                return {
                    "status_code": response.status,
                    "response": await response.json() if response.status == 200 else None,
                    "success": response.status == 200
                }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def test_failover_test(self) -> Dict[str, Any]:
        """Test failover testing (non-destructive)"""
        payload = {
            "failover_type": "manual",
            "source_provider": "gcp",
            "target_provider": "azure",
            "services": ["test-service"],
            "rollback_enabled": True,
            "validation_required": True
        }
        
        try:
            async with self.session.post(
                f"{self.base_url}/disaster-recovery/test-failover",
                json=payload
            ) as response:
                return {
                    "status_code": response.status,
                    "response": await response.json() if response.content_type == "application/json" else None,
                    "success": response.status in [200, 201]
                }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def test_cost_optimization(self) -> Dict[str, Any]:
        """Test cost optimization endpoint"""
        try:
            async with self.session.get(f"{self.base_url}/disaster-recovery/cost-optimization") as response:
                return {
                    "status_code": response.status,
                    "response": await response.json() if response.status == 200 else None,
                    "success": response.status == 200
                }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def test_security_status(self) -> Dict[str, Any]:
        """Test security status endpoint"""
        try:
            async with self.session.get(f"{self.base_url}/disaster-recovery/security-status") as response:
                return {
                    "status_code": response.status,
                    "response": await response.json() if response.status == 200 else None,
                    "success": response.status == 200
                }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and return results"""
        logger.info("Starting disaster recovery service tests")
        
        test_results = {
            "timestamp": datetime.utcnow().isoformat(),
            "service_url": self.base_url,
            "tests": {}
        }
        
        # Test health endpoints
        logger.info("Testing health endpoints")
        test_results["tests"]["health_endpoints"] = await self.test_health_endpoints()
        
        # Test system status
        logger.info("Testing system status")
        test_results["tests"]["system_status"] = await self.test_system_status()
        
        # Test replication creation
        logger.info("Testing replication creation")
        test_results["tests"]["create_replication"] = await self.test_create_replication()
        
        # Test listing replications
        logger.info("Testing replication listing")
        test_results["tests"]["list_replications"] = await self.test_list_replications()
        
        # Test failover testing
        logger.info("Testing failover test")
        test_results["tests"]["failover_test"] = await self.test_failover_test()
        
        # Test cost optimization
        logger.info("Testing cost optimization")
        test_results["tests"]["cost_optimization"] = await self.test_cost_optimization()
        
        # Test security status
        logger.info("Testing security status")
        test_results["tests"]["security_status"] = await self.test_security_status()
        
        # Calculate overall success
        successful_tests = sum(1 for test in test_results["tests"].values() 
                             if isinstance(test, dict) and test.get("success", False))
        total_tests = len(test_results["tests"])
        
        test_results["summary"] = {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "failed_tests": total_tests - successful_tests,
            "success_rate": successful_tests / total_tests if total_tests > 0 else 0.0,
            "overall_success": successful_tests == total_tests
        }
        
        logger.info("Test execution completed", 
                   successful=successful_tests, 
                   total=total_tests,
                   success_rate=test_results["summary"]["success_rate"])
        
        return test_results

def print_test_results(results: Dict[str, Any]):
    """Print test results in a readable format"""
    print("\n" + "="*60)
    print("MULTI-CLOUD DISASTER RECOVERY SERVICE TEST RESULTS")
    print("="*60)
    
    print(f"Service URL: {results['service_url']}")
    print(f"Test Time: {results['timestamp']}")
    print()
    
    # Print summary
    summary = results["summary"]
    print("SUMMARY:")
    print(f"  Total Tests: {summary['total_tests']}")
    print(f"  Successful: {summary['successful_tests']}")
    print(f"  Failed: {summary['failed_tests']}")
    print(f"  Success Rate: {summary['success_rate']:.1%}")
    print(f"  Overall Success: {'✓' if summary['overall_success'] else '✗'}")
    print()
    
    # Print individual test results
    print("DETAILED RESULTS:")
    for test_name, test_result in results["tests"].items():
        status = "✓ PASS" if test_result.get("success", False) else "✗ FAIL"
        print(f"  {test_name}: {status}")
        
        if not test_result.get("success", False) and "error" in test_result:
            print(f"    Error: {test_result['error']}")
        elif "status_code" in test_result:
            print(f"    Status Code: {test_result['status_code']}")
    
    print("\n" + "="*60)

async def main():
    """Main test function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Multi-Cloud Disaster Recovery Service")
    parser.add_argument("--url", default="http://localhost:8000", 
                       help="Service URL (default: http://localhost:8000)")
    parser.add_argument("--output", help="Output file for test results (JSON)")
    parser.add_argument("--verbose", action="store_true", 
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.INFO)
    
    try:
        async with DisasterRecoveryTester(args.url) as tester:
            results = await tester.run_all_tests()
            
            # Print results to console
            print_test_results(results)
            
            # Save results to file if specified
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"\nResults saved to: {args.output}")
            
            # Exit with appropriate code
            sys.exit(0 if results["summary"]["overall_success"] else 1)
            
    except Exception as e:
        logger.error("Test execution failed", error=str(e))
        print(f"\nTest execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
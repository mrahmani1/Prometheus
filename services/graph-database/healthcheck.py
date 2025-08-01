#!/usr/bin/env python3
"""
Health check script for the Graph Database Service
"""

import asyncio
import logging
import sys
from typing import Any, Dict

import httpx
import torch.nn as nn
import torch.nn.functional as F
from core.config import settings
from core.database import GraphDatabaseManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def check_database_connection() -> Dict[str, Any]:
    """Check Neo4j database connection"""
    try:
        db_manager = GraphDatabaseManager()
        await db_manager.connect()

        # Test basic query
        result = await db_manager.execute_query("RETURN 1 as test")

        await db_manager.close()

        return {
            "status": "healthy",
            "message": "Database connection successful",
            "details": {"query_result": result},
        }

    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"Database connection failed: {str(e)}",
            "details": {"error": str(e)},
        }


async def check_api_endpoints() -> Dict[str, Any]:
    """Check API endpoint availability"""
    try:
        async with httpx.AsyncClient() as client:
            # Check health endpoint
            health_response = await client.get(
                f"http://localhost:{settings.api_port}/health", timeout=10.0
            )

            if health_response.status_code != 200:
                return {
                    "status": "unhealthy",
                    "message": f"Health endpoint returned {health_response.status_code}",
                    "details": {"status_code": health_response.status_code},
                }

            # Check readiness endpoint
            ready_response = await client.get(
                f"http://localhost:{settings.api_port}/ready", timeout=10.0
            )

            if ready_response.status_code != 200:
                return {
                    "status": "unhealthy",
                    "message": f"Ready endpoint returned {ready_response.status_code}",
                    "details": {"status_code": ready_response.status_code},
                }

            return {
                "status": "healthy",
                "message": "API endpoints are accessible",
                "details": {
                    "health_status": health_response.json(),
                    "ready_status": ready_response.json(),
                },
            }

    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"API endpoint check failed: {str(e)}",
            "details": {"error": str(e)},
        }


async def check_graph_operations() -> Dict[str, Any]:
    """Check basic graph operations"""
    try:
        db_manager = GraphDatabaseManager()
        await db_manager.connect()

        # Test node creation and retrieval
        test_query = """
        CREATE (test:HealthCheck {
            id: 'health_check_' + toString(timestamp()),
            created_at: datetime(),
            test: true
        })
        RETURN test.id as test_id
        """

        create_result = await db_manager.execute_query(test_query)

        if not create_result:
            raise Exception("Failed to create test node")

        test_id = create_result[0]["test_id"]

        # Test node retrieval
        retrieve_query = """
        MATCH (test:HealthCheck {id: $test_id})
        RETURN test
        """

        retrieve_result = await db_manager.execute_query(
            retrieve_query, {"test_id": test_id}
        )

        if not retrieve_result:
            raise Exception("Failed to retrieve test node")

        # Clean up test node
        cleanup_query = """
        MATCH (test:HealthCheck {id: $test_id})
        DELETE test
        """

        await db_manager.execute_query(cleanup_query, {"test_id": test_id})

        await db_manager.close()

        return {
            "status": "healthy",
            "message": "Graph operations working correctly",
            "details": {
                "test_node_id": test_id,
                "operations_tested": ["create", "retrieve", "delete"],
            },
        }

    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"Graph operations failed: {str(e)}",
            "details": {"error": str(e)},
        }


async def check_schema_integrity() -> Dict[str, Any]:
    """Check database schema integrity"""
    try:
        db_manager = GraphDatabaseManager()
        await db_manager.connect()

        # Check constraints
        constraints_query = "SHOW CONSTRAINTS"
        constraints = await db_manager.execute_query(constraints_query)

        # Check indexes
        indexes_query = "SHOW INDEXES"
        indexes = await db_manager.execute_query(indexes_query)

        await db_manager.close()

        expected_constraints = [
            "process_id_unique",
            "location_id_unique",
            "event_id_unique",
        ]

        constraint_names = [c.get("name", "") for c in constraints]
        missing_constraints = [
            name
            for name in expected_constraints
            if not any(name in constraint_name for constraint_name in constraint_names)
        ]

        if missing_constraints:
            return {
                "status": "warning",
                "message": f"Missing constraints: {missing_constraints}",
                "details": {
                    "constraints_found": len(constraints),
                    "indexes_found": len(indexes),
                    "missing_constraints": missing_constraints,
                },
            }

        return {
            "status": "healthy",
            "message": "Schema integrity verified",
            "details": {
                "constraints_found": len(constraints),
                "indexes_found": len(indexes),
            },
        }

    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"Schema check failed: {str(e)}",
            "details": {"error": str(e)},
        }


async def run_comprehensive_health_check() -> Dict[str, Any]:
    """Run comprehensive health check"""

    logger.info("Starting comprehensive health check...")

    health_checks = {
        "database_connection": await check_database_connection(),
        "api_endpoints": await check_api_endpoints(),
        "graph_operations": await check_graph_operations(),
        "schema_integrity": await check_schema_integrity(),
    }

    # Determine overall health status
    statuses = [check["status"] for check in health_checks.values()]

    if all(status == "healthy" for status in statuses):
        overall_status = "healthy"
        overall_message = "All health checks passed"
    elif any(status == "unhealthy" for status in statuses):
        overall_status = "unhealthy"
        unhealthy_checks = [
            name
            for name, check in health_checks.items()
            if check["status"] == "unhealthy"
        ]
        overall_message = f"Unhealthy checks: {', '.join(unhealthy_checks)}"
    else:
        overall_status = "warning"
        warning_checks = [
            name
            for name, check in health_checks.items()
            if check["status"] == "warning"
        ]
        overall_message = f"Warning checks: {', '.join(warning_checks)}"

    return {
        "overall_status": overall_status,
        "overall_message": overall_message,
        "timestamp": asyncio.get_event_loop().time(),
        "individual_checks": health_checks,
        "service_info": {
            "service_name": "graph-database",
            "neo4j_uri": settings.neo4j_uri,
            "api_port": settings.api_port,
            "environment": settings.environment,
        },
    }


async def main():
    """Main health check function"""

    try:
        health_result = await run_comprehensive_health_check()

        # Print results
        print(f"Overall Status: {health_result['overall_status'].upper()}")
        print(f"Message: {health_result['overall_message']}")
        print("\nIndividual Check Results:")
        print("-" * 40)

        for check_name, result in health_result["individual_checks"].items():
            status_symbol = {"healthy": "✓", "warning": "⚠", "unhealthy": "✗"}.get(
                result["status"], "?"
            )

            print(
                f"{status_symbol} {check_name}: {result['status']} - {result['message']}"
            )

        # Exit with appropriate code
        if health_result["overall_status"] == "healthy":
            sys.exit(0)
        elif health_result["overall_status"] == "warning":
            sys.exit(1)
        else:
            sys.exit(2)

    except Exception as e:
        logger.error(f"Health check failed with exception: {e}")
        print(f"✗ Health check failed: {e}")
        sys.exit(3)


if __name__ == "__main__":
    asyncio.run(main())

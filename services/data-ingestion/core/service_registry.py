"""Service registry for service discovery and health management"""

import json
from typing import Any, Dict, List, Optional

import consul.aio
import structlog
import torch.nn as nn
import torch.nn.functional as F

logger = structlog.get_logger()

class ServiceRegistry:
    """Service registry using Consul for service discovery"""

    def __init__(self, consul_host: str = "consul", consul_port: int = 8500):
        self.consul_host = consul_host
        self.consul_port = consul_port
        self.consul = None
        self.registered_services = {}

    async def initialize(self):
        """Initialize connection to Consul"""
        try:
            self.consul = consul.aio.Consul(
                host=self.consul_host, port=self.consul_port
            )
            logger.info(
                "Connected to Consul", host=self.consul_host, port=self.consul_port
            )
        except Exception as e:
            logger.error("Failed to connect to Consul", error=str(e))
            raise

    async def register_service(
        self,
        name: str,
        service_id: str,
        address: str,
        port: int,
        health_check_url: str,
        tags: Optional[List[str]] = None,
        meta: Optional[Dict[str, str]] = None,
    ):
        """Register a service with Consul"""
        if not self.consul:
            raise RuntimeError("Service registry not initialized")

        try:
            check = consul.Check.http(
                url=f"http://{address}:{port}{health_check_url}",
                interval="10s",
                timeout="5s",
                deregister="30s",
            )

            await self.consul.agent.service.register(
                name=name,
                service_id=service_id,
                address=address,
                port=port,
                tags=tags or [],
                meta=meta or {},
                check=check,
            )

            self.registered_services[service_id] = {
                "name": name,
                "address": address,
                "port": port,
                "health_check_url": health_check_url,
            }

            logger.info(
                "Service registered",
                service_id=service_id,
                name=name,
                address=address,
                port=port,
            )

        except Exception as e:
            logger.error(
                "Failed to register service", service_id=service_id, error=str(e)
            )
            raise

    async def deregister_service(self, service_id: str):
        """Deregister a service from Consul"""
        if not self.consul:
            return

        try:
            await self.consul.agent.service.deregister(service_id)
            if service_id in self.registered_services:
                del self.registered_services[service_id]
            logger.info("Service deregistered", service_id=service_id)
        except Exception as e:
            logger.error(
                "Failed to deregister service", service_id=service_id, error=str(e)
            )

    async def discover_service(self, service_name: str) -> List[Dict[str, Any]]:
        """Discover services by name"""
        if not self.consul:
            raise RuntimeError("Service registry not initialized")

        try:
            _, services = await self.consul.health.service(service_name, passing=True)

            discovered_services = []
            for service in services:
                service_info = service["Service"]
                discovered_services.append(
                    {
                        "id": service_info["ID"],
                        "name": service_info["Service"],
                        "address": service_info["Address"],
                        "port": service_info["Port"],
                        "tags": service_info.get("Tags", []),
                        "meta": service_info.get("Meta", {}),
                    }
                )

            logger.debug(
                "Services discovered",
                service_name=service_name,
                count=len(discovered_services),
            )

            return discovered_services

        except Exception as e:
            logger.error(
                "Failed to discover services", service_name=service_name, error=str(e)
            )
            return []

    async def get_service_health(self, service_name: str) -> Dict[str, Any]:
        """Get health status of services"""
        if not self.consul:
            raise RuntimeError("Service registry not initialized")

        try:
            _, checks = await self.consul.health.service(service_name)

            health_status = {
                "service_name": service_name,
                "total_instances": len(checks),
                "healthy_instances": 0,
                "unhealthy_instances": 0,
                "instances": [],
            }

            for check in checks:
                service_info = check["Service"]
                checks_info = check["Checks"]

                is_healthy = all(c["Status"] == "passing" for c in checks_info)

                if is_healthy:
                    health_status["healthy_instances"] += 1
                else:
                    health_status["unhealthy_instances"] += 1

                health_status["instances"].append(
                    {
                        "id": service_info["ID"],
                        "address": service_info["Address"],
                        "port": service_info["Port"],
                        "healthy": is_healthy,
                        "checks": [
                            {
                                "name": c["Name"],
                                "status": c["Status"],
                                "output": c.get("Output", ""),
                            }
                            for c in checks_info
                        ],
                    }
                )

            return health_status

        except Exception as e:
            logger.error(
                "Failed to get service health", service_name=service_name, error=str(e)
            )
            return {
                "service_name": service_name,
                "error": str(e),
                "total_instances": 0,
                "healthy_instances": 0,
                "unhealthy_instances": 0,
                "instances": [],
            }

    async def put_key_value(self, key: str, value: Any):
        """Store key-value pair in Consul KV store"""
        if not self.consul:
            raise RuntimeError("Service registry not initialized")

        try:
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            elif not isinstance(value, str):
                value = str(value)

            await self.consul.kv.put(key, value)
            logger.debug("Key-value stored", key=key)

        except Exception as e:
            logger.error("Failed to store key-value", key=key, error=str(e))
            raise

    async def get_key_value(self, key: str) -> Optional[str]:
        """Retrieve value from Consul KV store"""
        if not self.consul:
            raise RuntimeError("Service registry not initialized")

        try:
            _, data = await self.consul.kv.get(key)
            if data:
                return data["Value"].decode("utf-8")
            return None

        except Exception as e:
            logger.error("Failed to retrieve key-value", key=key, error=str(e))
            return None

    async def cleanup(self):
        """Cleanup service registry connections"""
        if self.consul:
            # Deregister all services registered by this instance
            for service_id in list(self.registered_services.keys()):
                await self.deregister_service(service_id)

            # Close Consul connection
            await self.consul.close()
            logger.info("Service registry cleanup completed")

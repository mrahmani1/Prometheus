"""Service registry for service discovery using Consul"""

import asyncio
import logging
from typing import Dict, List, Optional

import consul.aio
import structlog

logger = structlog.get_logger()


class ServiceRegistry:
    """Service registry for managing service discovery with Consul"""
    
    def __init__(self, consul_host: str = "localhost", consul_port: int = 8500):
        self.consul_host = consul_host
        self.consul_port = consul_port
        self.consul_client: Optional[consul.aio.Consul] = None
        self.registered_services: Dict[str, str] = {}
        
    async def initialize(self):
        """Initialize the Consul client"""
        try:
            self.consul_client = consul.aio.Consul(
                host=self.consul_host,
                port=self.consul_port
            )
            logger.info("Service registry initialized", 
                       consul_host=self.consul_host, 
                       consul_port=self.consul_port)
        except Exception as e:
            logger.error("Failed to initialize service registry", error=str(e))
            raise
    
    async def register_service(
        self,
        name: str,
        service_id: str,
        address: str,
        port: int,
        health_check_url: str,
        tags: Optional[List[str]] = None,
        meta: Optional[Dict[str, str]] = None
    ):
        """Register a service with Consul"""
        if not self.consul_client:
            raise RuntimeError("Service registry not initialized")
        
        try:
            health_check = consul.Check.http(
                url=f"http://{address}:{port}{health_check_url}",
                interval="10s",
                timeout="5s",
                deregister="30s"
            )
            
            await self.consul_client.agent.service.register(
                name=name,
                service_id=service_id,
                address=address,
                port=port,
                tags=tags or [],
                meta=meta or {},
                check=health_check
            )
            
            self.registered_services[service_id] = name
            logger.info("Service registered successfully",
                       service_name=name,
                       service_id=service_id,
                       address=address,
                       port=port)
            
        except Exception as e:
            logger.error("Failed to register service",
                        service_name=name,
                        service_id=service_id,
                        error=str(e))
            raise
    
    async def deregister_service(self, service_id: str):
        """Deregister a service from Consul"""
        if not self.consul_client:
            return
        
        try:
            await self.consul_client.agent.service.deregister(service_id)
            if service_id in self.registered_services:
                service_name = self.registered_services.pop(service_id)
                logger.info("Service deregistered successfully",
                           service_name=service_name,
                           service_id=service_id)
        except Exception as e:
            logger.error("Failed to deregister service",
                        service_id=service_id,
                        error=str(e))
    
    async def discover_service(self, service_name: str) -> Optional[Dict]:
        """Discover a service by name"""
        if not self.consul_client:
            raise RuntimeError("Service registry not initialized")
        
        try:
            _, services = await self.consul_client.health.service(
                service_name, 
                passing=True
            )
            
            if services:
                service = services[0]
                return {
                    "name": service["Service"]["Service"],
                    "id": service["Service"]["ID"],
                    "address": service["Service"]["Address"],
                    "port": service["Service"]["Port"],
                    "tags": service["Service"]["Tags"],
                    "meta": service["Service"]["Meta"]
                }
            return None
            
        except Exception as e:
            logger.error("Failed to discover service",
                        service_name=service_name,
                        error=str(e))
            return None
    
    async def discover_all_services(self) -> List[Dict]:
        """Discover all healthy services"""
        if not self.consul_client:
            raise RuntimeError("Service registry not initialized")
        
        try:
            _, services = await self.consul_client.catalog.services()
            discovered_services = []
            
            for service_name in services.keys():
                service_info = await self.discover_service(service_name)
                if service_info:
                    discovered_services.append(service_info)
            
            return discovered_services
            
        except Exception as e:
            logger.error("Failed to discover all services", error=str(e))
            return []
    
    async def get_service_health(self, service_name: str) -> Dict:
        """Get health status of a service"""
        if not self.consul_client:
            raise RuntimeError("Service registry not initialized")
        
        try:
            _, checks = await self.consul_client.health.service(service_name)
            
            if not checks:
                return {"status": "not_found", "checks": []}
            
            health_status = "passing"
            check_details = []
            
            for check in checks:
                for health_check in check.get("Checks", []):
                    check_details.append({
                        "name": health_check["Name"],
                        "status": health_check["Status"],
                        "output": health_check["Output"]
                    })
                    
                    if health_check["Status"] != "passing":
                        health_status = "failing"
            
            return {
                "status": health_status,
                "checks": check_details
            }
            
        except Exception as e:
            logger.error("Failed to get service health",
                        service_name=service_name,
                        error=str(e))
            return {"status": "error", "error": str(e)}
    
    async def watch_service(self, service_name: str, callback):
        """Watch for changes in a service"""
        if not self.consul_client:
            raise RuntimeError("Service registry not initialized")
        
        index = None
        while True:
            try:
                index, services = await self.consul_client.health.service(
                    service_name,
                    index=index,
                    wait="30s"
                )
                
                await callback(service_name, services)
                
            except Exception as e:
                logger.error("Error watching service",
                            service_name=service_name,
                            error=str(e))
                await asyncio.sleep(5)  # Wait before retrying
    
    async def cleanup(self):
        """Cleanup and deregister all services"""
        for service_id in list(self.registered_services.keys()):
            await self.deregister_service(service_id)
        
        if self.consul_client:
            await self.consul_client.close()
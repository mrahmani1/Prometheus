"""
Model deployment functionality for edge devices
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Set

import aioredis
import httpx
from models.config import EdgeManagerConfig, ModelDeploymentConfig
from models.device import (
    ModelDeployment,
    ModelDeploymentRequest,
    ModelDeploymentResponse,
    ModelStatus,
)

logger = logging.getLogger(__name__)

class ModelDeployer:
    """Handles model deployment to edge devices"""

    def __init__(self, config: EdgeManagerConfig):
        self.config = config
        self.redis: Optional[aioredis.Redis] = None
        self.deployments: Dict[str, ModelDeployment] = {}
        self.deployment_queue: asyncio.Queue = asyncio.Queue()
        self._ready = False

        # Deployment settings
        self.deployment_timeout = config.deployment_timeout
        self.max_concurrent_deployments = config.max_concurrent_deployments
        self.model_registry_url = config.model_registry_url

        # Active deployment tracking
        self.active_deployments: Set[str] = set()

    async def initialize(self):
        """Initialize the model deployer"""
        try:
            # Connect to Redis
            self.redis = aioredis.from_url(self.config.redis_url)
            await self.redis.ping()

            # Load existing deployments
            await self._load_deployments_from_storage()

            # Start deployment workers
            for i in range(self.max_concurrent_deployments):
                asyncio.create_task(self._deployment_worker(f"worker-{i}"))

            # Start deployment monitor
            asyncio.create_task(self._deployment_monitor())

            self._ready = True
            logger.info(
                f"Model deployer initialized with {len(self.deployments)} deployments"
            )

        except Exception as e:
            logger.error(f"Failed to initialize model deployer: {e}")
            raise

    def is_ready(self) -> bool:
        """Check if the model deployer is ready"""
        return self._ready

    async def deploy_model(
        self, request: ModelDeploymentRequest, device_manager
    ) -> ModelDeploymentResponse:
        """Deploy a model to specified edge devices"""
        try:
            deployment_id = f"deploy-{uuid.uuid4().hex[:12]}"

            # Get model configuration from registry
            model_config = await self._get_model_config(request.model_id)
            if not model_config:
                raise ValueError(f"Model {request.model_id} not found in registry")

            # Filter target devices based on capabilities
            target_devices = await self._filter_compatible_devices(
                request.target_devices, model_config, device_manager
            )

            if not target_devices:
                raise ValueError("No compatible devices found for deployment")

            # Create deployment records
            deployments = []
            for device_id in target_devices:
                deployment = ModelDeployment(
                    deployment_id=f"{deployment_id}-{device_id}",
                    device_id=device_id,
                    model_id=request.model_id,
                    model_name=model_config.model_name,
                    model_version=model_config.model_version,
                    status=ModelStatus.PENDING,
                    last_updated=datetime.utcnow(),
                )

                self.deployments[deployment.deployment_id] = deployment
                deployments.append(deployment)

                # Queue for deployment
                await self.deployment_queue.put(
                    {
                        "deployment": deployment,
                        "model_config": model_config,
                        "force_update": request.force_update,
                        "rollback_on_failure": request.rollback_on_failure,
                    }
                )

            # Save deployments to storage
            for deployment in deployments:
                await self._save_deployment_to_storage(deployment)

            logger.info(
                f"Queued model {request.model_id} for deployment to {len(target_devices)} devices"
            )

            response = ModelDeploymentResponse(
                deployment_id=deployment_id,
                target_devices=target_devices,
                status="queued",
                total_devices=len(target_devices),
                successful_deployments=0,
                failed_deployments=0,
                pending_deployments=len(target_devices),
                deployment_details=[
                    {
                        "device_id": d.device_id,
                        "deployment_id": d.deployment_id,
                        "status": d.status.value,
                    }
                    for d in deployments
                ],
            )

            return response

        except Exception as e:
            logger.error(f"Failed to deploy model: {e}")
            raise

    async def get_deployment_status(self, deployment_id: str) -> Optional[Dict]:
        """Get status of a deployment"""
        # Find all deployments with this base ID
        related_deployments = [
            d
            for d in self.deployments.values()
            if d.deployment_id.startswith(deployment_id)
        ]

        if not related_deployments:
            return None

        total = len(related_deployments)
        successful = len(
            [d for d in related_deployments if d.status == ModelStatus.RUNNING]
        )
        failed = len([d for d in related_deployments if d.status == ModelStatus.FAILED])
        pending = len(
            [
                d
                for d in related_deployments
                if d.status
                in [
                    ModelStatus.PENDING,
                    ModelStatus.DOWNLOADING,
                    ModelStatus.INSTALLING,
                ]
            ]
        )

        return {
            "deployment_id": deployment_id,
            "total_devices": total,
            "successful_deployments": successful,
            "failed_deployments": failed,
            "pending_deployments": pending,
            "deployment_details": [
                {
                    "device_id": d.device_id,
                    "deployment_id": d.deployment_id,
                    "status": d.status.value,
                    "deployed_at": d.deployed_at,
                    "last_updated": d.last_updated,
                }
                for d in related_deployments
            ],
        }

    async def list_deployments(
        self, device_id: Optional[str] = None
    ) -> List[ModelDeployment]:
        """List deployments, optionally filtered by device"""
        deployments = list(self.deployments.values())

        if device_id:
            deployments = [d for d in deployments if d.device_id == device_id]

        return deployments

    async def stop_deployment(self, deployment_id: str, device_id: str):
        """Stop a running model deployment"""
        full_deployment_id = f"{deployment_id}-{device_id}"

        if full_deployment_id not in self.deployments:
            raise ValueError(f"Deployment {full_deployment_id} not found")

        deployment = self.deployments[full_deployment_id]

        if deployment.status != ModelStatus.RUNNING:
            raise ValueError(f"Deployment {full_deployment_id} is not running")

        # Send stop command to device
        success = await self._send_stop_command(deployment)

        if success:
            deployment.status = ModelStatus.STOPPED
            deployment.last_updated = datetime.utcnow()
            await self._save_deployment_to_storage(deployment)
            logger.info(f"Stopped deployment {full_deployment_id}")
        else:
            raise RuntimeError(f"Failed to stop deployment {full_deployment_id}")

    async def remove_deployment(self, deployment_id: str, device_id: str):
        """Remove a model deployment from device"""
        full_deployment_id = f"{deployment_id}-{device_id}"

        if full_deployment_id not in self.deployments:
            raise ValueError(f"Deployment {full_deployment_id} not found")

        deployment = self.deployments[full_deployment_id]

        # Stop if running
        if deployment.status == ModelStatus.RUNNING:
            await self.stop_deployment(deployment_id, device_id)

        # Send remove command to device
        success = await self._send_remove_command(deployment)

        if success:
            # Remove from tracking
            del self.deployments[full_deployment_id]
            await self._remove_deployment_from_storage(full_deployment_id)
            logger.info(f"Removed deployment {full_deployment_id}")
        else:
            raise RuntimeError(f"Failed to remove deployment {full_deployment_id}")

    async def _deployment_worker(self, worker_id: str):
        """Worker process for handling deployments"""
        logger.info(f"Started deployment worker {worker_id}")

        while self._ready:
            try:
                # Get deployment task from queue
                task = await asyncio.wait_for(self.deployment_queue.get(), timeout=1.0)

                deployment = task["deployment"]
                model_config = task["model_config"]

                logger.info(
                    f"Worker {worker_id} processing deployment {deployment.deployment_id}"
                )

                # Add to active deployments
                self.active_deployments.add(deployment.deployment_id)

                try:
                    # Execute deployment
                    await self._execute_deployment(deployment, model_config, task)

                except Exception as e:
                    logger.error(f"Deployment {deployment.deployment_id} failed: {e}")
                    deployment.status = ModelStatus.FAILED
                    deployment.last_error = str(e)
                    deployment.last_updated = datetime.utcnow()
                    await self._save_deployment_to_storage(deployment)

                finally:
                    # Remove from active deployments
                    self.active_deployments.discard(deployment.deployment_id)
                    self.deployment_queue.task_done()

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in deployment worker {worker_id}: {e}")
                await asyncio.sleep(5)

    async def _execute_deployment(
        self,
        deployment: ModelDeployment,
        model_config: ModelDeploymentConfig,
        task: Dict,
    ):
        """Execute a single model deployment"""
        try:
            # Update status to downloading
            deployment.status = ModelStatus.DOWNLOADING
            deployment.last_updated = datetime.utcnow()
            await self._save_deployment_to_storage(deployment)

            # Download model to device
            await self._download_model_to_device(deployment, model_config)

            # Update status to installing
            deployment.status = ModelStatus.INSTALLING
            deployment.last_updated = datetime.utcnow()
            await self._save_deployment_to_storage(deployment)

            # Install and start model on device
            await self._install_model_on_device(deployment, model_config)

            # Verify deployment
            if await self._verify_deployment(deployment):
                deployment.status = ModelStatus.RUNNING
                deployment.deployed_at = datetime.utcnow()
                logger.info(
                    f"Successfully deployed {deployment.model_id} to device {deployment.device_id}"
                )
            else:
                deployment.status = ModelStatus.FAILED
                deployment.last_error = "Deployment verification failed"
                logger.error(
                    f"Deployment verification failed for {deployment.deployment_id}"
                )

            deployment.last_updated = datetime.utcnow()
            await self._save_deployment_to_storage(deployment)

        except Exception as e:
            logger.error(
                f"Failed to execute deployment {deployment.deployment_id}: {e}"
            )
            raise

    async def _get_model_config(self, model_id: str) -> Optional[ModelDeploymentConfig]:
        """Get model configuration from registry"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.model_registry_url}/models/{model_id}"
                )

                if response.status_code == 200:
                    model_data = response.json()
                    return ModelDeploymentConfig(**model_data)
                else:
                    logger.error(f"Model {model_id} not found in registry")
                    return None

        except Exception as e:
            logger.error(f"Failed to get model config for {model_id}: {e}")
            return None

    async def _filter_compatible_devices(
        self, device_ids: List[str], model_config: ModelDeploymentConfig, device_manager
    ) -> List[str]:
        """Filter devices based on model requirements"""
        compatible_devices = []

        for device_id in device_ids:
            device = await device_manager.get_device(device_id)
            if not device:
                logger.warning(f"Device {device_id} not found")
                continue

            # Check resource requirements
            capabilities = device.capabilities

            if (
                capabilities.cpu_cores >= model_config.min_cpu_cores
                and capabilities.memory_gb >= model_config.min_memory_gb
                and capabilities.storage_gb >= model_config.min_storage_gb
            ):

                # Check GPU requirements
                if model_config.requires_gpu:
                    if (
                        capabilities.gpu_available
                        and capabilities.gpu_memory_gb
                        and capabilities.gpu_memory_gb
                        >= (model_config.min_gpu_memory_gb or 0)
                    ):
                        compatible_devices.append(device_id)
                else:
                    compatible_devices.append(device_id)
            else:
                logger.warning(
                    f"Device {device_id} does not meet resource requirements"
                )

        return compatible_devices

    async def _download_model_to_device(
        self, deployment: ModelDeployment, model_config: ModelDeploymentConfig
    ):
        """Download model to edge device"""
        # This would typically involve sending a download command to the device
        # For now, simulate the download process
        await asyncio.sleep(2)  # Simulate download time
        logger.debug(
            f"Downloaded model {deployment.model_id} to device {deployment.device_id}"
        )

    async def _install_model_on_device(
        self, deployment: ModelDeployment, model_config: ModelDeploymentConfig
    ):
        """Install and start model on edge device"""
        # This would typically involve:
        # 1. Creating container/process on device
        # 2. Configuring environment
        # 3. Starting the model service
        await asyncio.sleep(3)  # Simulate installation time
        logger.debug(
            f"Installed model {deployment.model_id} on device {deployment.device_id}"
        )

    async def _verify_deployment(self, deployment: ModelDeployment) -> bool:
        """Verify that deployment is running correctly"""
        # This would typically involve health checks
        await asyncio.sleep(1)  # Simulate verification time
        return True  # Assume success for now

    async def _send_stop_command(self, deployment: ModelDeployment) -> bool:
        """Send stop command to device"""
        # Implementation would send actual stop command
        await asyncio.sleep(1)
        return True

    async def _send_remove_command(self, deployment: ModelDeployment) -> bool:
        """Send remove command to device"""
        # Implementation would send actual remove command
        await asyncio.sleep(1)
        return True

    async def _deployment_monitor(self):
        """Monitor deployment health and status"""
        while self._ready:
            try:
                # Check health of running deployments
                running_deployments = [
                    d
                    for d in self.deployments.values()
                    if d.status == ModelStatus.RUNNING
                ]

                for deployment in running_deployments:
                    # Check if deployment is still healthy
                    is_healthy = await self._check_deployment_health(deployment)

                    if not is_healthy:
                        deployment.status = ModelStatus.FAILED
                        deployment.last_error = "Health check failed"
                        deployment.last_updated = datetime.utcnow()
                        await self._save_deployment_to_storage(deployment)
                        logger.warning(
                            f"Deployment {deployment.deployment_id} marked as failed due to health check"
                        )

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Error in deployment monitor: {e}")
                await asyncio.sleep(60)

    async def _check_deployment_health(self, deployment: ModelDeployment) -> bool:
        """Check health of a specific deployment"""
        # Implementation would check actual deployment health
        return True  # Assume healthy for now

    async def _load_deployments_from_storage(self):
        """Load deployments from Redis storage"""
        if not self.redis:
            return

        try:
            deployment_keys = await self.redis.keys("deployment:*")
            for key in deployment_keys:
                deployment_data = await self.redis.get(key)
                if deployment_data:
                    deployment_dict = json.loads(deployment_data)
                    deployment = ModelDeployment(**deployment_dict)
                    self.deployments[deployment.deployment_id] = deployment

            logger.info(f"Loaded {len(self.deployments)} deployments from storage")

        except Exception as e:
            logger.error(f"Failed to load deployments from storage: {e}")

    async def _save_deployment_to_storage(self, deployment: ModelDeployment):
        """Save deployment to Redis storage"""
        if not self.redis:
            return

        try:
            deployment_key = f"deployment:{deployment.deployment_id}"
            deployment_data = deployment.model_dump_json()
            await self.redis.set(deployment_key, deployment_data)

        except Exception as e:
            logger.error(
                f"Failed to save deployment {deployment.deployment_id} to storage: {e}"
            )

    async def _remove_deployment_from_storage(self, deployment_id: str):
        """Remove deployment from Redis storage"""
        if not self.redis:
            return

        try:
            deployment_key = f"deployment:{deployment_id}"
            await self.redis.delete(deployment_key)

        except Exception as e:
            logger.error(
                f"Failed to remove deployment {deployment_id} from storage: {e}"
            )

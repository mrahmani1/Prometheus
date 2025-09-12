"""
Recovery Orchestrator
Advanced disaster recovery plan execution and orchestration system
with intelligent automation and comprehensive monitoring.
"""

import asyncio
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import uuid
import structlog
from pathlib import Path

from ..models.disaster_recovery_models import (
    RecoveryPlan, RecoveryExecution, RecoveryStatus, DisasterType,
    RecoveryObjective, DatabaseType
)

logger = structlog.get_logger(__name__)

class RecoveryOrchestrator:
    """Advanced disaster recovery orchestration system"""
    
    def __init__(self):
        self.recovery_plans: Dict[str, RecoveryPlan] = {}
        self.active_executions: Dict[str, RecoveryExecution] = {}
        self.execution_history: List[RecoveryExecution] = []
        self.recovery_templates: Dict[str, Dict[str, Any]] = {}
        
    async def initialize(self):
        """Initialize recovery orchestrator"""
        try:
            await self._load_recovery_plans()
            await self._load_recovery_templates()
            await self._validate_recovery_dependencies()
            
            logger.info("Recovery orchestrator initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize recovery orchestrator", error=str(e))
            raise
    
    async def _load_recovery_plans(self):
        """Load recovery plans from storage"""
        try:
            plans_file = '/etc/prometheus/recovery-plans.json'
            if os.path.exists(plans_file):
                with open(plans_file, 'r') as f:
                    plans_data = json.load(f)
                
                for plan_data in plans_data.get('plans', []):
                    plan = RecoveryPlan(**plan_data)
                    self.recovery_plans[plan.plan_id] = plan
            
            # Create default recovery plans if none exist
            if not self.recovery_plans:
                await self._create_default_recovery_plans()
            
            logger.info("Recovery plans loaded", count=len(self.recovery_plans))
            
        except Exception as e:
            logger.error("Failed to load recovery plans", error=str(e))
            await self._create_default_recovery_plans()
    
    async def _create_default_recovery_plans(self):
        """Create default recovery plans for common disaster scenarios"""
        default_plans = [
            {
                'name': 'Database Hardware Failure Recovery',
                'description': 'Recovery plan for database hardware failures',
                'disaster_types': [DisasterType.HARDWARE_FAILURE],
                'affected_services': ['postgresql', 'influxdb', 'neo4j', 'redis'],
                'recovery_steps': [
                    {
                        'step': 1,
                        'name': 'Assess Damage',
                        'action': 'assess_database_damage',
                        'timeout_minutes': 10,
                        'required': True
                    },
                    {
                        'step': 2,
                        'name': 'Initiate Failover',
                        'action': 'initiate_database_failover',
                        'timeout_minutes': 15,
                        'required': True,
                        'depends_on': [1]
                    },
                    {
                        'step': 3,
                        'name': 'Restore from Backup',
                        'action': 'restore_from_latest_backup',
                        'timeout_minutes': 60,
                        'required': True,
                        'depends_on': [2]
                    },
                    {
                        'step': 4,
                        'name': 'Validate Data Integrity',
                        'action': 'validate_data_integrity',
                        'timeout_minutes': 30,
                        'required': True,
                        'depends_on': [3]
                    },
                    {
                        'step': 5,
                        'name': 'Resume Services',
                        'action': 'resume_application_services',
                        'timeout_minutes': 20,
                        'required': True,
                        'depends_on': [4]
                    }
                ],
                'recovery_objectives': {
                    'rto_minutes': 120,
                    'rpo_minutes': 15,
                    'service_tier': 'critical'
                },
                'estimated_duration_minutes': 135
            },
            {
                'name': 'Network Outage Recovery',
                'description': 'Recovery plan for network connectivity issues',
                'disaster_types': [DisasterType.NETWORK_OUTAGE],
                'affected_services': ['api-gateway', 'helios-core', 'data-ingestion'],
                'recovery_steps': [
                    {
                        'step': 1,
                        'name': 'Diagnose Network Issue',
                        'action': 'diagnose_network_connectivity',
                        'timeout_minutes': 5,
                        'required': True
                    },
                    {
                        'step': 2,
                        'name': 'Activate Backup Network',
                        'action': 'activate_backup_network_path',
                        'timeout_minutes': 10,
                        'required': True,
                        'depends_on': [1]
                    },
                    {
                        'step': 3,
                        'name': 'Reroute Traffic',
                        'action': 'reroute_application_traffic',
                        'timeout_minutes': 15,
                        'required': True,
                        'depends_on': [2]
                    },
                    {
                        'step': 4,
                        'name': 'Validate Connectivity',
                        'action': 'validate_service_connectivity',
                        'timeout_minutes': 10,
                        'required': True,
                        'depends_on': [3]
                    }
                ],
                'recovery_objectives': {
                    'rto_minutes': 45,
                    'rpo_minutes': 5,
                    'service_tier': 'critical'
                },
                'estimated_duration_minutes': 40
            },
            {
                'name': 'Data Corruption Recovery',
                'description': 'Recovery plan for data corruption incidents',
                'disaster_types': [DisasterType.DATA_CORRUPTION],
                'affected_services': ['postgresql', 'influxdb', 'neo4j'],
                'recovery_steps': [
                    {
                        'step': 1,
                        'name': 'Isolate Corrupted Data',
                        'action': 'isolate_corrupted_database',
                        'timeout_minutes': 10,
                        'required': True
                    },
                    {
                        'step': 2,
                        'name': 'Assess Corruption Scope',
                        'action': 'assess_corruption_scope',
                        'timeout_minutes': 20,
                        'required': True,
                        'depends_on': [1]
                    },
                    {
                        'step': 3,
                        'name': 'Point-in-Time Recovery',
                        'action': 'perform_point_in_time_recovery',
                        'timeout_minutes': 90,
                        'required': True,
                        'depends_on': [2]
                    },
                    {
                        'step': 4,
                        'name': 'Verify Data Integrity',
                        'action': 'comprehensive_data_verification',
                        'timeout_minutes': 45,
                        'required': True,
                        'depends_on': [3]
                    },
                    {
                        'step': 5,
                        'name': 'Resume Operations',
                        'action': 'resume_normal_operations',
                        'timeout_minutes': 15,
                        'required': True,
                        'depends_on': [4]
                    }
                ],
                'recovery_objectives': {
                    'rto_minutes': 180,
                    'rpo_minutes': 30,
                    'service_tier': 'critical'
                },
                'estimated_duration_minutes': 180
            },
            {
                'name': 'Cyber Attack Recovery',
                'description': 'Recovery plan for cyber security incidents',
                'disaster_types': [DisasterType.CYBER_ATTACK],
                'affected_services': ['all'],
                'recovery_steps': [
                    {
                        'step': 1,
                        'name': 'Isolate Affected Systems',
                        'action': 'isolate_compromised_systems',
                        'timeout_minutes': 5,
                        'required': True
                    },
                    {
                        'step': 2,
                        'name': 'Assess Security Breach',
                        'action': 'assess_security_breach_scope',
                        'timeout_minutes': 30,
                        'required': True,
                        'depends_on': [1]
                    },
                    {
                        'step': 3,
                        'name': 'Clean and Rebuild',
                        'action': 'clean_and_rebuild_systems',
                        'timeout_minutes': 120,
                        'required': True,
                        'depends_on': [2]
                    },
                    {
                        'step': 4,
                        'name': 'Restore from Clean Backup',
                        'action': 'restore_from_verified_clean_backup',
                        'timeout_minutes': 90,
                        'required': True,
                        'depends_on': [3]
                    },
                    {
                        'step': 5,
                        'name': 'Security Hardening',
                        'action': 'apply_enhanced_security_measures',
                        'timeout_minutes': 60,
                        'required': True,
                        'depends_on': [4]
                    },
                    {
                        'step': 6,
                        'name': 'Gradual Service Restoration',
                        'action': 'gradual_service_restoration',
                        'timeout_minutes': 45,
                        'required': True,
                        'depends_on': [5]
                    }
                ],
                'recovery_objectives': {
                    'rto_minutes': 360,
                    'rpo_minutes': 60,
                    'service_tier': 'critical'
                },
                'estimated_duration_minutes': 350
            }
        ]
        
        for plan_data in default_plans:
            plan = RecoveryPlan(
                plan_id=str(uuid.uuid4()),
                **plan_data
            )
            self.recovery_plans[plan.plan_id] = plan
    
    async def _load_recovery_templates(self):
        """Load recovery action templates"""
        try:
            templates_file = '/etc/prometheus/recovery-templates.json'
            if os.path.exists(templates_file):
                with open(templates_file, 'r') as f:
                    self.recovery_templates = json.load(f)
            else:
                await self._create_default_templates()
            
            logger.info("Recovery templates loaded", count=len(self.recovery_templates))
            
        except Exception as e:
            logger.error("Failed to load recovery templates", error=str(e))
            await self._create_default_templates()
    
    async def _create_default_templates(self):
        """Create default recovery action templates"""
        self.recovery_templates = {
            'assess_database_damage': {
                'description': 'Assess extent of database damage',
                'commands': [
                    'kubectl get pods -n prometheus-ai',
                    'kubectl describe pods -n prometheus-ai',
                    'kubectl logs -n prometheus-ai --selector=app=database'
                ],
                'validation': 'check_database_connectivity'
            },
            'initiate_database_failover': {
                'description': 'Initiate failover to secondary database',
                'commands': [
                    'kubectl patch service postgresql -p \'{"spec":{"selector":{"role":"secondary"}}}\'',
                    'kubectl scale deployment postgresql-primary --replicas=0',
                    'kubectl scale deployment postgresql-secondary --replicas=1'
                ],
                'validation': 'verify_failover_success'
            },
            'restore_from_latest_backup': {
                'description': 'Restore database from latest backup',
                'commands': [
                    'curl -X POST http://disaster-recovery:8080/api/v1/recovery/restore-latest',
                    'kubectl wait --for=condition=ready pod -l app=database --timeout=600s'
                ],
                'validation': 'verify_restore_completion'
            },
            'validate_data_integrity': {
                'description': 'Validate restored data integrity',
                'commands': [
                    'curl -X POST http://disaster-recovery:8080/api/v1/validation/integrity-check',
                    'kubectl exec -it postgresql-0 -- psql -c "SELECT COUNT(*) FROM critical_tables"'
                ],
                'validation': 'verify_data_consistency'
            },
            'resume_application_services': {
                'description': 'Resume normal application services',
                'commands': [
                    'kubectl scale deployment api-gateway --replicas=3',
                    'kubectl scale deployment helios-core --replicas=2',
                    'kubectl scale deployment data-ingestion --replicas=2'
                ],
                'validation': 'verify_service_health'
            }
        }
    
    async def _validate_recovery_dependencies(self):
        """Validate recovery plan dependencies"""
        try:
            validation_issues = []
            
            for plan_id, plan in self.recovery_plans.items():
                # Check step dependencies
                step_numbers = {step['step'] for step in plan.recovery_steps}
                
                for step in plan.recovery_steps:
                    if 'depends_on' in step:
                        for dep in step['depends_on']:
                            if dep not in step_numbers:
                                validation_issues.append(
                                    f"Plan {plan_id}: Step {step['step']} depends on non-existent step {dep}"
                                )
                
                # Check if all required actions have templates
                for step in plan.recovery_steps:
                    action = step.get('action')
                    if action and action not in self.recovery_templates:
                        validation_issues.append(
                            f"Plan {plan_id}: Step {step['step']} uses undefined action '{action}'"
                        )
            
            if validation_issues:
                logger.warning("Recovery plan validation issues found", issues=validation_issues)
            else:
                logger.info("All recovery plans validated successfully")
            
        except Exception as e:
            logger.error("Recovery plan validation failed", error=str(e))
    
    async def create_recovery_plan(self, recovery_request: Dict[str, Any]) -> str:
        """Create a new recovery plan"""
        try:
            plan = RecoveryPlan(
                plan_id=str(uuid.uuid4()),
                name=recovery_request['name'],
                description=recovery_request['description'],
                disaster_types=[DisasterType(dt) for dt in recovery_request['disaster_types']],
                affected_services=recovery_request['affected_services'],
                recovery_steps=recovery_request['recovery_steps'],
                recovery_objectives=RecoveryObjective(**recovery_request['recovery_objectives']),
                dependencies=recovery_request.get('dependencies', []),
                estimated_duration_minutes=recovery_request.get('estimated_duration_minutes', 60)
            )
            
            self.recovery_plans[plan.plan_id] = plan
            
            logger.info("Recovery plan created", 
                       plan_id=plan.plan_id,
                       name=plan.name)
            
            return plan.plan_id
            
        except Exception as e:
            logger.error("Failed to create recovery plan", error=str(e))
            raise
    
    async def execute_recovery_plan(self, plan_id: str) -> str:
        """Execute a recovery plan"""
        try:
            plan = self.recovery_plans.get(plan_id)
            if not plan:
                raise ValueError(f"Recovery plan {plan_id} not found")
            
            execution = RecoveryExecution(
                execution_id=str(uuid.uuid4()),
                plan_id=plan_id,
                disaster_type=plan.disaster_types[0],  # Use first disaster type
                start_time=datetime.utcnow(),
                status=RecoveryStatus.EXECUTING,
                total_steps=len(plan.recovery_steps)
            )
            
            self.active_executions[execution.execution_id] = execution
            
            logger.info("Recovery plan execution started", 
                       execution_id=execution.execution_id,
                       plan_id=plan_id)
            
            # Start execution in background
            asyncio.create_task(self._execute_recovery_steps(execution, plan))
            
            return execution.execution_id
            
        except Exception as e:
            logger.error("Failed to execute recovery plan", plan_id=plan_id, error=str(e))
            raise
    
    async def _execute_recovery_steps(self, execution: RecoveryExecution, plan: RecoveryPlan):
        """Execute recovery plan steps"""
        try:
            # Sort steps by step number
            sorted_steps = sorted(plan.recovery_steps, key=lambda x: x['step'])
            
            for step in sorted_steps:
                try:
                    # Check dependencies
                    if 'depends_on' in step:
                        await self._wait_for_dependencies(execution, step['depends_on'])
                    
                    # Execute step
                    execution.current_step = step['step']
                    step_result = await self._execute_recovery_step(step)
                    
                    execution.step_results.append({
                        'step': step['step'],
                        'name': step['name'],
                        'status': 'completed',
                        'result': step_result,
                        'timestamp': datetime.utcnow().isoformat()
                    })
                    
                    logger.info("Recovery step completed", 
                               execution_id=execution.execution_id,
                               step=step['step'],
                               name=step['name'])
                    
                except Exception as step_error:
                    execution.step_results.append({
                        'step': step['step'],
                        'name': step['name'],
                        'status': 'failed',
                        'error': str(step_error),
                        'timestamp': datetime.utcnow().isoformat()
                    })
                    
                    logger.error("Recovery step failed", 
                                execution_id=execution.execution_id,
                                step=step['step'],
                                error=str(step_error))
                    
                    # Check if step is required
                    if step.get('required', True):
                        execution.status = RecoveryStatus.FAILED
                        execution.error_message = f"Required step {step['step']} failed: {str(step_error)}"
                        execution.rollback_required = True
                        break
            
            # Complete execution if all steps succeeded
            if execution.status == RecoveryStatus.EXECUTING:
                execution.status = RecoveryStatus.COMPLETED
                execution.end_time = datetime.utcnow()
                
                logger.info("Recovery plan execution completed successfully", 
                           execution_id=execution.execution_id,
                           duration_minutes=(execution.end_time - execution.start_time).total_seconds() / 60)
            
            # Move to history
            self.execution_history.append(execution)
            if execution.execution_id in self.active_executions:
                del self.active_executions[execution.execution_id]
            
        except Exception as e:
            execution.status = RecoveryStatus.FAILED
            execution.error_message = str(e)
            execution.end_time = datetime.utcnow()
            
            logger.error("Recovery plan execution failed", 
                        execution_id=execution.execution_id,
                        error=str(e))
    
    async def _wait_for_dependencies(self, execution: RecoveryExecution, dependencies: List[int]):
        """Wait for dependent steps to complete"""
        max_wait_time = 300  # 5 minutes
        start_time = datetime.utcnow()
        
        while (datetime.utcnow() - start_time).total_seconds() < max_wait_time:
            completed_steps = {
                result['step'] for result in execution.step_results 
                if result['status'] == 'completed'
            }
            
            if all(dep in completed_steps for dep in dependencies):
                return
            
            await asyncio.sleep(5)  # Check every 5 seconds
        
        raise Exception(f"Timeout waiting for dependencies: {dependencies}")
    
    async def _execute_recovery_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single recovery step"""
        try:
            action = step.get('action')
            if not action:
                raise ValueError("Step has no action defined")
            
            template = self.recovery_templates.get(action)
            if not template:
                raise ValueError(f"No template found for action: {action}")
            
            # Execute commands from template
            results = []
            for command in template.get('commands', []):
                result = await self._execute_command(command)
                results.append(result)
            
            # Validate step completion
            validation = template.get('validation')
            if validation:
                validation_result = await self._validate_step_completion(validation)
                results.append({'validation': validation_result})
            
            return {
                'action': action,
                'results': results,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error("Recovery step execution failed", 
                        step=step.get('step'),
                        action=step.get('action'),
                        error=str(e))
            raise
    
    async def _execute_command(self, command: str) -> Dict[str, Any]:
        """Execute a recovery command"""
        try:
            # Handle different command types
            if command.startswith('kubectl'):
                return await self._execute_kubectl_command(command)
            elif command.startswith('curl'):
                return await self._execute_http_command(command)
            else:
                return await self._execute_shell_command(command)
                
        except Exception as e:
            logger.error("Command execution failed", command=command, error=str(e))
            raise
    
    async def _execute_kubectl_command(self, command: str) -> Dict[str, Any]:
        """Execute kubectl command"""
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            return {
                'command': command,
                'return_code': process.returncode,
                'stdout': stdout.decode() if stdout else '',
                'stderr': stderr.decode() if stderr else ''
            }
            
        except Exception as e:
            logger.error("Kubectl command failed", command=command, error=str(e))
            raise
    
    async def _execute_http_command(self, command: str) -> Dict[str, Any]:
        """Execute HTTP command (curl)"""
        try:
            import aiohttp
            
            # Parse curl command (simplified)
            if 'POST' in command:
                method = 'POST'
            elif 'GET' in command:
                method = 'GET'
            else:
                method = 'GET'
            
            # Extract URL (simplified parsing)
            url_start = command.find('http')
            if url_start == -1:
                raise ValueError("No URL found in curl command")
            
            url = command[url_start:].split()[0]
            
            async with aiohttp.ClientSession() as session:
                async with session.request(method, url) as response:
                    text = await response.text()
                    
                    return {
                        'command': command,
                        'status_code': response.status,
                        'response': text
                    }
                    
        except Exception as e:
            logger.error("HTTP command failed", command=command, error=str(e))
            raise
    
    async def _execute_shell_command(self, command: str) -> Dict[str, Any]:
        """Execute shell command"""
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            return {
                'command': command,
                'return_code': process.returncode,
                'stdout': stdout.decode() if stdout else '',
                'stderr': stderr.decode() if stderr else ''
            }
            
        except Exception as e:
            logger.error("Shell command failed", command=command, error=str(e))
            raise
    
    async def _validate_step_completion(self, validation: str) -> Dict[str, Any]:
        """Validate step completion"""
        try:
            # This would implement specific validation logic
            # For now, return a placeholder
            return {
                'validation': validation,
                'status': 'passed',
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("Step validation failed", validation=validation, error=str(e))
            raise
    
    async def monitor_recovery_execution(self, execution_id: str):
        """Monitor recovery execution progress"""
        try:
            execution = self.active_executions.get(execution_id)
            if not execution:
                logger.warning("Recovery execution not found for monitoring", 
                              execution_id=execution_id)
                return
            
            while execution.status == RecoveryStatus.EXECUTING:
                logger.info("Recovery execution progress", 
                           execution_id=execution_id,
                           current_step=execution.current_step,
                           total_steps=execution.total_steps,
                           completed_steps=len([r for r in execution.step_results if r['status'] == 'completed']))
                
                await asyncio.sleep(30)  # Update every 30 seconds
            
            logger.info("Recovery execution monitoring completed", 
                       execution_id=execution_id,
                       final_status=execution.status.value)
            
        except Exception as e:
            logger.error("Recovery execution monitoring failed", 
                        execution_id=execution_id,
                        error=str(e))
    
    async def get_recovery_status(self, execution_id: str) -> Dict[str, Any]:
        """Get recovery execution status"""
        try:
            # Check active executions first
            execution = self.active_executions.get(execution_id)
            if not execution:
                # Check execution history
                execution = next(
                    (e for e in self.execution_history if e.execution_id == execution_id),
                    None
                )
            
            if not execution:
                raise ValueError(f"Recovery execution {execution_id} not found")
            
            return {
                'execution_id': execution.execution_id,
                'plan_id': execution.plan_id,
                'disaster_type': execution.disaster_type.value,
                'status': execution.status.value,
                'start_time': execution.start_time.isoformat(),
                'end_time': execution.end_time.isoformat() if execution.end_time else None,
                'current_step': execution.current_step,
                'total_steps': execution.total_steps,
                'step_results': execution.step_results,
                'error_message': execution.error_message,
                'rollback_required': execution.rollback_required
            }
            
        except Exception as e:
            logger.error("Failed to get recovery status", 
                        execution_id=execution_id,
                        error=str(e))
            raise
    
    async def is_ready(self) -> bool:
        """Check if recovery orchestrator is ready"""
        return (
            len(self.recovery_plans) > 0 and
            len(self.recovery_templates) > 0
        )
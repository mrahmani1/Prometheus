"""
Security Orchestration and Automated Response (SOAR) System
Provides automated incident response, remediation, and orchestration capabilities
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
from abc import ABC, abstractmethod
import aiohttp
import aioredis
import asyncpg
from cryptography.fernet import Fernet

class ResponseAction(Enum):
    BLOCK_IP = "block_ip"
    DISABLE_USER = "disable_user"
    QUARANTINE_SYSTEM = "quarantine_system"
    ROTATE_CREDENTIALS = "rotate_credentials"
    ISOLATE_NETWORK = "isolate_network"
    BACKUP_DATA = "backup_data"
    NOTIFY_TEAM = "notify_team"
    COLLECT_FORENSICS = "collect_forensics"
    UPDATE_FIREWALL = "update_firewall"
    REVOKE_TOKENS = "revoke_tokens"

class ResponseStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class PlaybookType(Enum):
    MALWARE_INCIDENT = "malware_incident"
    DATA_BREACH = "data_breach"
    INSIDER_THREAT = "insider_threat"
    GEOLOGICAL_DATA_COMPROMISE = "geological_data_compromise"
    API_ABUSE = "api_abuse"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    BRUTE_FORCE_ATTACK = "brute_force_attack"

@dataclass
class ResponseTask:
    id: str
    action: ResponseAction
    parameters: Dict[str, Any]
    status: ResponseStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    dependencies: List[str] = None
    timeout_seconds: int = 300

@dataclass
class ResponsePlaybook:
    id: str
    name: str
    description: str
    playbook_type: PlaybookType
    trigger_conditions: Dict[str, Any]
    tasks: List[ResponseTask]
    parallel_execution: bool = False
    auto_approve: bool = False
    geological_context_required: bool = False

@dataclass
class IncidentResponse:
    id: str
    incident_id: str
    playbook_id: str
    status: ResponseStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    tasks: List[ResponseTask] = None
    approval_required: bool = True
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None

class ResponseActionHandler(ABC):
    """Abstract base class for response action handlers"""
    
    @abstractmethod
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the response action"""
        pass
    
    @abstractmethod
    async def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate action parameters"""
        pass
    
    @abstractmethod
    async def rollback(self, parameters: Dict[str, Any], result: Dict[str, Any]) -> bool:
        """Rollback the action if needed"""
        pass

class BlockIPHandler(ResponseActionHandler):
    """Handler for blocking IP addresses"""
    
    def __init__(self, firewall_api_url: str, api_key: str):
        self.firewall_api_url = firewall_api_url
        self.api_key = api_key
    
    async def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate IP blocking parameters"""
        required_params = ['ip_address', 'duration_hours']
        return all(param in parameters for param in required_params)
    
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Block IP address at firewall level"""
        ip_address = parameters['ip_address']
        duration_hours = parameters.get('duration_hours', 24)
        
        # Validate IP format
        import ipaddress
        try:
            ipaddress.ip_address(ip_address)
        except ValueError:
            raise ValueError(f"Invalid IP address format: {ip_address}")
        
        # Call firewall API
        async with aiohttp.ClientSession() as session:
            payload = {
                'action': 'block',
                'ip_address': ip_address,
                'duration_hours': duration_hours,
                'reason': 'Automated security response'
            }
            
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            async with session.post(
                f"{self.firewall_api_url}/block-ip",
                json=payload,
                headers=headers
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return {
                        'success': True,
                        'ip_address': ip_address,
                        'rule_id': result.get('rule_id'),
                        'expires_at': (datetime.utcnow() + timedelta(hours=duration_hours)).isoformat()
                    }
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to block IP: {error_text}")
    
    async def rollback(self, parameters: Dict[str, Any], result: Dict[str, Any]) -> bool:
        """Unblock IP address"""
        if not result.get('success') or not result.get('rule_id'):
            return False
        
        async with aiohttp.ClientSession() as session:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            async with session.delete(
                f"{self.firewall_api_url}/block-ip/{result['rule_id']}",
                headers=headers
            ) as response:
                return response.status == 200

class DisableUserHandler(ResponseActionHandler):
    """Handler for disabling user accounts"""
    
    def __init__(self, identity_api_url: str, api_key: str):
        self.identity_api_url = identity_api_url
        self.api_key = api_key
    
    async def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate user disable parameters"""
        return 'user_id' in parameters
    
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Disable user account"""
        user_id = parameters['user_id']
        reason = parameters.get('reason', 'Security incident response')
        
        async with aiohttp.ClientSession() as session:
            payload = {
                'action': 'disable',
                'reason': reason,
                'disabled_by': 'automated_security_system'
            }
            
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            async with session.patch(
                f"{self.identity_api_url}/users/{user_id}/status",
                json=payload,
                headers=headers
            ) as response:
                if response.status == 200:
                    return {
                        'success': True,
                        'user_id': user_id,
                        'disabled_at': datetime.utcnow().isoformat(),
                        'reason': reason
                    }
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to disable user: {error_text}")
    
    async def rollback(self, parameters: Dict[str, Any], result: Dict[str, Any]) -> bool:
        """Re-enable user account"""
        if not result.get('success'):
            return False
        
        user_id = parameters['user_id']
        
        async with aiohttp.ClientSession() as session:
            payload = {
                'action': 'enable',
                'reason': 'Security incident resolved - automated re-enable'
            }
            
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            async with session.patch(
                f"{self.identity_api_url}/users/{user_id}/status",
                json=payload,
                headers=headers
            ) as response:
                return response.status == 200

class ForensicsCollectionHandler(ResponseActionHandler):
    """Handler for collecting forensic evidence"""
    
    def __init__(self, forensics_api_url: str, storage_path: str):
        self.forensics_api_url = forensics_api_url
        self.storage_path = storage_path
    
    async def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate forensics collection parameters"""
        required_params = ['target_system', 'evidence_types']
        return all(param in parameters for param in required_params)
    
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Collect forensic evidence"""
        target_system = parameters['target_system']
        evidence_types = parameters['evidence_types']
        incident_id = parameters.get('incident_id', 'unknown')
        
        collection_id = str(uuid.uuid4())
        
        # Collect different types of evidence
        collected_evidence = {}
        
        for evidence_type in evidence_types:
            if evidence_type == 'memory_dump':
                collected_evidence['memory_dump'] = await self._collect_memory_dump(target_system)
            elif evidence_type == 'disk_image':
                collected_evidence['disk_image'] = await self._collect_disk_image(target_system)
            elif evidence_type == 'network_logs':
                collected_evidence['network_logs'] = await self._collect_network_logs(target_system)
            elif evidence_type == 'system_logs':
                collected_evidence['system_logs'] = await self._collect_system_logs(target_system)
            elif evidence_type == 'geological_data_access':
                collected_evidence['geological_data_access'] = await self._collect_geological_access_logs(target_system)
        
        # Store evidence metadata
        evidence_metadata = {
            'collection_id': collection_id,
            'incident_id': incident_id,
            'target_system': target_system,
            'collected_at': datetime.utcnow().isoformat(),
            'evidence_types': evidence_types,
            'evidence_files': collected_evidence,
            'chain_of_custody': [{
                'timestamp': datetime.utcnow().isoformat(),
                'action': 'evidence_collected',
                'actor': 'automated_forensics_system',
                'hash': self._calculate_evidence_hash(collected_evidence)
            }]
        }
        
        return {
            'success': True,
            'collection_id': collection_id,
            'evidence_metadata': evidence_metadata,
            'storage_path': f"{self.storage_path}/{collection_id}"
        }
    
    async def _collect_memory_dump(self, target_system: str) -> Dict[str, Any]:
        """Collect memory dump from target system"""
        # Simulated memory dump collection
        return {
            'type': 'memory_dump',
            'file_path': f"/forensics/memory_{target_system}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.mem",
            'size_bytes': 8589934592,  # 8GB
            'collection_method': 'automated_agent'
        }
    
    async def _collect_disk_image(self, target_system: str) -> Dict[str, Any]:
        """Collect disk image from target system"""
        return {
            'type': 'disk_image',
            'file_path': f"/forensics/disk_{target_system}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.dd",
            'size_bytes': 1099511627776,  # 1TB
            'collection_method': 'dd_imaging'
        }
    
    async def _collect_network_logs(self, target_system: str) -> Dict[str, Any]:
        """Collect network logs"""
        return {
            'type': 'network_logs',
            'file_path': f"/forensics/network_{target_system}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pcap",
            'size_bytes': 104857600,  # 100MB
            'time_range': '24_hours'
        }
    
    async def _collect_system_logs(self, target_system: str) -> Dict[str, Any]:
        """Collect system logs"""
        return {
            'type': 'system_logs',
            'file_path': f"/forensics/system_{target_system}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.log",
            'size_bytes': 52428800,  # 50MB
            'log_sources': ['syslog', 'auth.log', 'kern.log']
        }
    
    async def _collect_geological_access_logs(self, target_system: str) -> Dict[str, Any]:
        """Collect geological data access logs"""
        return {
            'type': 'geological_access_logs',
            'file_path': f"/forensics/geo_access_{target_system}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json",
            'size_bytes': 10485760,  # 10MB
            'data_types': ['seismic_data', 'geological_models', 'api_calls']
        }
    
    def _calculate_evidence_hash(self, evidence: Dict[str, Any]) -> str:
        """Calculate hash of collected evidence for integrity"""
        import hashlib
        evidence_str = json.dumps(evidence, sort_keys=True)
        return hashlib.sha256(evidence_str.encode()).hexdigest()
    
    async def rollback(self, parameters: Dict[str, Any], result: Dict[str, Any]) -> bool:
        """Forensics collection cannot be rolled back"""
        return True  # Evidence collection is permanent

class NotificationHandler(ResponseActionHandler):
    """Handler for sending notifications"""
    
    def __init__(self, notification_config: Dict[str, Any]):
        self.config = notification_config
    
    async def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate notification parameters"""
        required_params = ['recipients', 'message', 'severity']
        return all(param in parameters for param in required_params)
    
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Send notifications to security team"""
        recipients = parameters['recipients']
        message = parameters['message']
        severity = parameters['severity']
        channels = parameters.get('channels', ['email', 'slack'])
        
        notification_results = {}
        
        for channel in channels:
            if channel == 'email':
                notification_results['email'] = await self._send_email_notification(
                    recipients, message, severity
                )
            elif channel == 'slack':
                notification_results['slack'] = await self._send_slack_notification(
                    message, severity
                )
            elif channel == 'sms':
                notification_results['sms'] = await self._send_sms_notification(
                    recipients, message, severity
                )
        
        return {
            'success': True,
            'notification_id': str(uuid.uuid4()),
            'sent_at': datetime.utcnow().isoformat(),
            'channels': notification_results
        }
    
    async def _send_email_notification(self, recipients: List[str], message: str, severity: str) -> Dict[str, Any]:
        """Send email notification"""
        # Simulated email sending
        return {
            'success': True,
            'recipients': recipients,
            'subject': f"Security Alert - {severity.upper()}",
            'sent_at': datetime.utcnow().isoformat()
        }
    
    async def _send_slack_notification(self, message: str, severity: str) -> Dict[str, Any]:
        """Send Slack notification"""
        # Simulated Slack notification
        return {
            'success': True,
            'channel': '#security-alerts',
            'message_id': str(uuid.uuid4()),
            'sent_at': datetime.utcnow().isoformat()
        }
    
    async def _send_sms_notification(self, recipients: List[str], message: str, severity: str) -> Dict[str, Any]:
        """Send SMS notification"""
        # Simulated SMS sending
        return {
            'success': True,
            'recipients': recipients,
            'sent_at': datetime.utcnow().isoformat()
        }
    
    async def rollback(self, parameters: Dict[str, Any], result: Dict[str, Any]) -> bool:
        """Notifications cannot be rolled back"""
        return True

class SOAREngine:
    """Security Orchestration and Automated Response Engine"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.playbooks = {}
        self.active_responses = {}
        self.action_handlers = {}
        self.redis_client = None
        self.db_pool = None
        
    async def initialize(self) -> None:
        """Initialize SOAR engine"""
        # Initialize Redis for state management
        self.redis_client = await aioredis.from_url(
            self.config.get('redis_url', 'redis://localhost:6379')
        )
        
        # Initialize PostgreSQL for persistent storage
        self.db_pool = await asyncpg.create_pool(
            self.config.get('database_url', 'postgresql://user:pass@localhost/soar')
        )
        
        # Initialize action handlers
        await self.initialize_action_handlers()
        
        # Load playbooks
        await self.load_playbooks()
        
        logging.info("SOAR Engine initialized successfully")
    
    async def initialize_action_handlers(self) -> None:
        """Initialize response action handlers"""
        self.action_handlers = {
            ResponseAction.BLOCK_IP: BlockIPHandler(
                self.config.get('firewall_api_url', 'http://localhost:8080'),
                self.config.get('firewall_api_key', 'test-key')
            ),
            ResponseAction.DISABLE_USER: DisableUserHandler(
                self.config.get('identity_api_url', 'http://localhost:8081'),
                self.config.get('identity_api_key', 'test-key')
            ),
            ResponseAction.COLLECT_FORENSICS: ForensicsCollectionHandler(
                self.config.get('forensics_api_url', 'http://localhost:8082'),
                self.config.get('forensics_storage_path', '/forensics')
            ),
            ResponseAction.NOTIFY_TEAM: NotificationHandler(
                self.config.get('notification_config', {})
            )
        }
    
    async def load_playbooks(self) -> None:
        """Load response playbooks"""
        self.playbooks = {
            'brute_force_response': ResponsePlaybook(
                id='brute_force_response',
                name='Brute Force Attack Response',
                description='Automated response to brute force attacks',
                playbook_type=PlaybookType.BRUTE_FORCE_ATTACK,
                trigger_conditions={
                    'incident_type': 'multiple_failed_logins',
                    'threat_level': 'high'
                },
                tasks=[
                    ResponseTask(
                        id='block_attacker_ip',
                        action=ResponseAction.BLOCK_IP,
                        parameters={'duration_hours': 24},
                        status=ResponseStatus.PENDING,
                        created_at=datetime.utcnow()
                    ),
                    ResponseTask(
                        id='notify_security_team',
                        action=ResponseAction.NOTIFY_TEAM,
                        parameters={
                            'recipients': ['security@company.com'],
                            'severity': 'high',
                            'channels': ['email', 'slack']
                        },
                        status=ResponseStatus.PENDING,
                        created_at=datetime.utcnow()
                    ),
                    ResponseTask(
                        id='collect_forensics',
                        action=ResponseAction.COLLECT_FORENSICS,
                        parameters={
                            'evidence_types': ['network_logs', 'system_logs']
                        },
                        status=ResponseStatus.PENDING,
                        created_at=datetime.utcnow(),
                        dependencies=['block_attacker_ip']
                    )
                ],
                auto_approve=True
            ),
            'geological_data_breach': ResponsePlaybook(
                id='geological_data_breach',
                name='Geological Data Breach Response',
                description='Response to geological data compromise incidents',
                playbook_type=PlaybookType.GEOLOGICAL_DATA_COMPROMISE,
                trigger_conditions={
                    'incident_type': 'geological_data_exfiltration',
                    'threat_level': 'critical'
                },
                tasks=[
                    ResponseTask(
                        id='disable_compromised_user',
                        action=ResponseAction.DISABLE_USER,
                        parameters={'reason': 'Suspected geological data breach'},
                        status=ResponseStatus.PENDING,
                        created_at=datetime.utcnow()
                    ),
                    ResponseTask(
                        id='block_source_ip',
                        action=ResponseAction.BLOCK_IP,
                        parameters={'duration_hours': 72},
                        status=ResponseStatus.PENDING,
                        created_at=datetime.utcnow()
                    ),
                    ResponseTask(
                        id='collect_comprehensive_forensics',
                        action=ResponseAction.COLLECT_FORENSICS,
                        parameters={
                            'evidence_types': ['memory_dump', 'disk_image', 'geological_data_access']
                        },
                        status=ResponseStatus.PENDING,
                        created_at=datetime.utcnow()
                    ),
                    ResponseTask(
                        id='notify_critical_team',
                        action=ResponseAction.NOTIFY_TEAM,
                        parameters={
                            'recipients': ['security@company.com', 'legal@company.com', 'ciso@company.com'],
                            'severity': 'critical',
                            'channels': ['email', 'slack', 'sms']
                        },
                        status=ResponseStatus.PENDING,
                        created_at=datetime.utcnow()
                    )
                ],
                parallel_execution=True,
                geological_context_required=True,
                auto_approve=False
            )
        }
    
    async def trigger_response(self, incident_id: str, incident_data: Dict[str, Any]) -> Optional[IncidentResponse]:
        """Trigger automated response for security incident"""
        # Find matching playbook
        playbook = await self.find_matching_playbook(incident_data)
        
        if not playbook:
            logging.warning(f"No matching playbook found for incident {incident_id}")
            return None
        
        # Create incident response
        response = IncidentResponse(
            id=str(uuid.uuid4()),
            incident_id=incident_id,
            playbook_id=playbook.id,
            status=ResponseStatus.PENDING,
            created_at=datetime.utcnow(),
            tasks=playbook.tasks.copy(),
            approval_required=not playbook.auto_approve
        )
        
        # Store response
        self.active_responses[response.id] = response
        await self.store_incident_response(response)
        
        # Execute if auto-approved
        if playbook.auto_approve:
            await self.execute_response(response.id)
        else:
            logging.info(f"Incident response {response.id} requires manual approval")
        
        return response
    
    async def find_matching_playbook(self, incident_data: Dict[str, Any]) -> Optional[ResponsePlaybook]:
        """Find playbook matching incident characteristics"""
        for playbook in self.playbooks.values():
            if await self.matches_trigger_conditions(playbook.trigger_conditions, incident_data):
                return playbook
        
        return None
    
    async def matches_trigger_conditions(self, conditions: Dict[str, Any], incident_data: Dict[str, Any]) -> bool:
        """Check if incident matches playbook trigger conditions"""
        for key, expected_value in conditions.items():
            if key not in incident_data:
                return False
            
            if incident_data[key] != expected_value:
                return False
        
        return True
    
    async def approve_response(self, response_id: str, approved_by: str) -> bool:
        """Approve incident response for execution"""
        if response_id not in self.active_responses:
            return False
        
        response = self.active_responses[response_id]
        response.approved_by = approved_by
        response.approved_at = datetime.utcnow()
        response.approval_required = False
        
        # Update in database
        await self.update_incident_response(response)
        
        # Execute response
        await self.execute_response(response_id)
        
        return True
    
    async def execute_response(self, response_id: str) -> None:
        """Execute incident response tasks"""
        if response_id not in self.active_responses:
            logging.error(f"Response {response_id} not found")
            return
        
        response = self.active_responses[response_id]
        
        if response.approval_required:
            logging.warning(f"Response {response_id} requires approval before execution")
            return
        
        response.status = ResponseStatus.IN_PROGRESS
        response.started_at = datetime.utcnow()
        
        playbook = self.playbooks[response.playbook_id]
        
        if playbook.parallel_execution:
            await self.execute_tasks_parallel(response)
        else:
            await self.execute_tasks_sequential(response)
        
        response.status = ResponseStatus.COMPLETED
        response.completed_at = datetime.utcnow()
        
        await self.update_incident_response(response)
        
        logging.info(f"Incident response {response_id} completed successfully")
    
    async def execute_tasks_sequential(self, response: IncidentResponse) -> None:
        """Execute response tasks sequentially"""
        for task in response.tasks:
            if await self.check_task_dependencies(task, response.tasks):
                await self.execute_task(task, response)
    
    async def execute_tasks_parallel(self, response: IncidentResponse) -> None:
        """Execute response tasks in parallel"""
        # Group tasks by dependencies
        independent_tasks = [task for task in response.tasks if not task.dependencies]
        dependent_tasks = [task for task in response.tasks if task.dependencies]
        
        # Execute independent tasks first
        if independent_tasks:
            await asyncio.gather(*[
                self.execute_task(task, response) for task in independent_tasks
            ])
        
        # Execute dependent tasks after dependencies complete
        for task in dependent_tasks:
            if await self.check_task_dependencies(task, response.tasks):
                await self.execute_task(task, response)
    
    async def check_task_dependencies(self, task: ResponseTask, all_tasks: List[ResponseTask]) -> bool:
        """Check if task dependencies are satisfied"""
        if not task.dependencies:
            return True
        
        for dep_id in task.dependencies:
            dep_task = next((t for t in all_tasks if t.id == dep_id), None)
            if not dep_task or dep_task.status != ResponseStatus.COMPLETED:
                return False
        
        return True
    
    async def execute_task(self, task: ResponseTask, response: IncidentResponse) -> None:
        """Execute individual response task"""
        task.status = ResponseStatus.IN_PROGRESS
        task.started_at = datetime.utcnow()
        
        try:
            # Get action handler
            handler = self.action_handlers.get(task.action)
            if not handler:
                raise Exception(f"No handler found for action: {task.action}")
            
            # Validate parameters
            if not await handler.validate_parameters(task.parameters):
                raise Exception(f"Invalid parameters for action: {task.action}")
            
            # Add incident context to parameters
            task.parameters['incident_id'] = response.incident_id
            
            # Execute action with timeout
            result = await asyncio.wait_for(
                handler.execute(task.parameters),
                timeout=task.timeout_seconds
            )
            
            task.result = result
            task.status = ResponseStatus.COMPLETED
            task.completed_at = datetime.utcnow()
            
            logging.info(f"Task {task.id} completed successfully: {task.action}")
            
        except asyncio.TimeoutError:
            task.status = ResponseStatus.FAILED
            task.error_message = f"Task timed out after {task.timeout_seconds} seconds"
            task.completed_at = datetime.utcnow()
            logging.error(f"Task {task.id} timed out: {task.action}")
            
        except Exception as e:
            task.status = ResponseStatus.FAILED
            task.error_message = str(e)
            task.completed_at = datetime.utcnow()
            logging.error(f"Task {task.id} failed: {task.action} - {str(e)}")
    
    async def store_incident_response(self, response: IncidentResponse) -> None:
        """Store incident response in database"""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO incident_responses 
                (id, incident_id, playbook_id, status, created_at, approval_required, approved_by, approved_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """, response.id, response.incident_id, response.playbook_id, 
                response.status.value, response.created_at, response.approval_required,
                response.approved_by, response.approved_at)
            
            # Store tasks
            for task in response.tasks:
                await conn.execute("""
                    INSERT INTO response_tasks 
                    (id, response_id, action, parameters, status, created_at, dependencies)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                """, task.id, response.id, task.action.value, 
                    json.dumps(task.parameters), task.status.value, 
                    task.created_at, task.dependencies)
    
    async def update_incident_response(self, response: IncidentResponse) -> None:
        """Update incident response in database"""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                UPDATE incident_responses 
                SET status = $2, started_at = $3, completed_at = $4, 
                    approved_by = $5, approved_at = $6
                WHERE id = $1
            """, response.id, response.status.value, response.started_at,
                response.completed_at, response.approved_by, response.approved_at)
            
            # Update tasks
            for task in response.tasks:
                await conn.execute("""
                    UPDATE response_tasks 
                    SET status = $2, started_at = $3, completed_at = $4, 
                        error_message = $5, result = $6
                    WHERE id = $1
                """, task.id, task.status.value, task.started_at,
                    task.completed_at, task.error_message, 
                    json.dumps(task.result) if task.result else None)
    
    async def get_response_status(self, response_id: str) -> Optional[Dict[str, Any]]:
        """Get incident response status"""
        if response_id not in self.active_responses:
            return None
        
        response = self.active_responses[response_id]
        
        return {
            'id': response.id,
            'incident_id': response.incident_id,
            'playbook_id': response.playbook_id,
            'status': response.status.value,
            'created_at': response.created_at.isoformat(),
            'started_at': response.started_at.isoformat() if response.started_at else None,
            'completed_at': response.completed_at.isoformat() if response.completed_at else None,
            'approval_required': response.approval_required,
            'approved_by': response.approved_by,
            'tasks': [{
                'id': task.id,
                'action': task.action.value,
                'status': task.status.value,
                'started_at': task.started_at.isoformat() if task.started_at else None,
                'completed_at': task.completed_at.isoformat() if task.completed_at else None,
                'error_message': task.error_message,
                'result': task.result
            } for task in response.tasks]
        }

# Database schema for SOAR
SOAR_SCHEMA = """
CREATE TABLE IF NOT EXISTS incident_responses (
    id UUID PRIMARY KEY,
    incident_id UUID NOT NULL,
    playbook_id VARCHAR(255) NOT NULL,
    status VARCHAR(20) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    approval_required BOOLEAN DEFAULT TRUE,
    approved_by VARCHAR(255),
    approved_at TIMESTAMP WITH TIME ZONE
);

CREATE TABLE IF NOT EXISTS response_tasks (
    id VARCHAR(255) PRIMARY KEY,
    response_id UUID NOT NULL REFERENCES incident_responses(id),
    action VARCHAR(50) NOT NULL,
    parameters JSONB NOT NULL,
    status VARCHAR(20) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    error_message TEXT,
    result JSONB,
    dependencies TEXT[],
    timeout_seconds INTEGER DEFAULT 300
);

CREATE TABLE IF NOT EXISTS playbook_executions (
    id UUID PRIMARY KEY,
    playbook_id VARCHAR(255) NOT NULL,
    incident_id UUID NOT NULL,
    execution_time_ms INTEGER,
    success_rate FLOAT,
    executed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_incident_responses_incident_id ON incident_responses(incident_id);
CREATE INDEX IF NOT EXISTS idx_incident_responses_status ON incident_responses(status);
CREATE INDEX IF NOT EXISTS idx_response_tasks_response_id ON response_tasks(response_id);
CREATE INDEX IF NOT EXISTS idx_response_tasks_status ON response_tasks(status);
"""
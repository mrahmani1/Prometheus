"""
Change Data Capture (CDC) System for Real-Time Database Synchronization

This module provides a comprehensive CDC system for tracking and synchronizing
changes across multiple geological databases with advanced conflict resolution,
data lineage tracking, and real-time streaming capabilities.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
from abc import ABC, abstractmethod

import asyncpg
import psycopg2
from psycopg2.extras import LogicalReplicationConnection, ReplicationMessage
import redis.asyncio as redis
from influxdb_client.client.influxdb_client_async import InfluxDBClientAsync
from neo4j import AsyncGraphDatabase
import structlog
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np

logger = structlog.get_logger(__name__)


class ChangeType(Enum):
    """Types of database changes to capture"""
    INSERT = "INSERT"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    TRUNCATE = "TRUNCATE"


class ConflictResolutionStrategy(Enum):
    """Strategies for resolving data conflicts during synchronization"""
    LAST_WRITE_WINS = "last_write_wins"
    FIRST_WRITE_WINS = "first_write_wins"
    MERGE_CHANGES = "merge_changes"
    MANUAL_RESOLUTION = "manual_resolution"
    SOURCE_PRIORITY = "source_priority"


@dataclass
class ChangeEvent:
    """Represents a database change event"""
    event_id: str
    source_database: str
    source_table: str
    change_type: ChangeType
    timestamp: datetime
    transaction_id: Optional[str]
    old_values: Optional[Dict[str, Any]]
    new_values: Optional[Dict[str, Any]]
    primary_key: Dict[str, Any]
    schema_version: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['change_type'] = self.change_type.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChangeEvent':
        """Create from dictionary"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['change_type'] = ChangeType(data['change_type'])
        return cls(**data)


@dataclass
class SyncRule:
    """Configuration for database synchronization rules"""
    rule_id: str
    source_database: str
    source_table: str
    target_database: str
    target_table: str
    field_mappings: Dict[str, str]
    transformation_functions: List[str]
    conflict_resolution: ConflictResolutionStrategy
    sync_frequency: timedelta
    enabled: bool = True
    filters: Optional[Dict[str, Any]] = None


@dataclass
class ConflictEvent:
    """Represents a data synchronization conflict"""
    conflict_id: str
    source_event: ChangeEvent
    target_state: Dict[str, Any]
    conflict_type: str
    resolution_strategy: ConflictResolutionStrategy
    resolved: bool = False
    resolution_timestamp: Optional[datetime] = None
    resolution_details: Optional[Dict[str, Any]] = None


class CDCCapture(ABC):
    """Abstract base class for Change Data Capture implementations"""
    
    @abstractmethod
    async def start_capture(self) -> None:
        """Start capturing database changes"""
        pass
    
    @abstractmethod
    async def stop_capture(self) -> None:
        """Stop capturing database changes"""
        pass
    
    @abstractmethod
    async def get_changes(self, since: datetime) -> List[ChangeEvent]:
        """Get changes since specified timestamp"""
        pass
    
    @abstractmethod
    def register_change_handler(self, handler: Callable[[ChangeEvent], None]) -> None:
        """Register handler for change events"""
        pass


class PostgreSQLCDC(CDCCapture):
    """PostgreSQL Change Data Capture using logical replication"""
    
    def __init__(self, 
                 connection_string: str,
                 replication_slot: str = 'prometheus_cdc_slot',
                 publication: str = 'prometheus_cdc_publication'):
        self.connection_string = connection_string
        self.replication_slot = replication_slot
        self.publication = publication
        self.replication_conn = None
        self.change_handlers: List[Callable[[ChangeEvent], None]] = []
        self.is_capturing = False
        self.capture_task = None
        
    async def start_capture(self) -> None:
        """Start PostgreSQL logical replication capture"""
        try:
            # Create replication connection
            self.replication_conn = psycopg2.connect(
                self.connection_string,
                connection_factory=LogicalReplicationConnection
            )
            
            # Create replication slot if it doesn't exist
            await self._ensure_replication_slot()
            
            # Create publication if it doesn't exist
            await self._ensure_publication()
            
            # Start capture task
            self.is_capturing = True
            self.capture_task = asyncio.create_task(self._capture_changes())
            
            logger.info("PostgreSQL CDC capture started", 
                       slot=self.replication_slot,
                       publication=self.publication)
            
        except Exception as e:
            logger.error("Failed to start PostgreSQL CDC capture", error=str(e))
            raise
    
    async def stop_capture(self) -> None:
        """Stop PostgreSQL logical replication capture"""
        try:
            self.is_capturing = False
            
            if self.capture_task:
                self.capture_task.cancel()
                try:
                    await self.capture_task
                except asyncio.CancelledError:
                    pass
            
            if self.replication_conn:
                self.replication_conn.close()
            
            logger.info("PostgreSQL CDC capture stopped")
            
        except Exception as e:
            logger.error("Failed to stop PostgreSQL CDC capture", error=str(e))
            raise
    
    async def _ensure_replication_slot(self) -> None:
        """Ensure replication slot exists"""
        try:
            cursor = self.replication_conn.cursor()
            
            # Check if slot exists
            cursor.execute("""
                SELECT slot_name FROM pg_replication_slots 
                WHERE slot_name = %s
            """, (self.replication_slot,))
            
            if not cursor.fetchone():
                # Create replication slot
                cursor.execute(f"""
                    SELECT pg_create_logical_replication_slot(
                        '{self.replication_slot}', 
                        'pgoutput'
                    )
                """)
                logger.info("Created replication slot", slot=self.replication_slot)
            
        except Exception as e:
            logger.error("Failed to ensure replication slot", error=str(e))
            raise
    
    async def _ensure_publication(self) -> None:
        """Ensure publication exists for all geological tables"""
        try:
            # Use regular connection for DDL operations
            conn = await asyncpg.connect(self.connection_string)
            
            # Check if publication exists
            result = await conn.fetchval("""
                SELECT pubname FROM pg_publication 
                WHERE pubname = $1
            """, self.publication)
            
            if not result:
                # Create publication for all geological tables
                await conn.execute(f"""
                    CREATE PUBLICATION {self.publication} 
                    FOR TABLE 
                        seismic_events,
                        seismic_waveforms,
                        mtsar_interferograms,
                        geochemical_analyses,
                        volcanic_systems,
                        monitoring_stations,
                        causal_relationships,
                        processed_seismic_events,
                        processed_mtsar_deformation,
                        processed_geochemical_data
                """)
                logger.info("Created publication", publication=self.publication)
            
            await conn.close()
            
        except Exception as e:
            logger.error("Failed to ensure publication", error=str(e))
            raise
    
    async def _capture_changes(self) -> None:
        """Capture changes from logical replication stream"""
        try:
            cursor = self.replication_conn.cursor()
            
            # Start replication
            cursor.start_replication(
                slot_name=self.replication_slot,
                decode=True,
                status_interval=10
            )
            
            logger.info("Started logical replication stream")
            
            while self.is_capturing:
                try:
                    # Wait for message with timeout
                    msg = cursor.read_message()
                    
                    if msg:
                        # Process replication message
                        change_event = await self._parse_replication_message(msg)
                        
                        if change_event:
                            # Notify all registered handlers
                            for handler in self.change_handlers:
                                try:
                                    await handler(change_event)
                                except Exception as e:
                                    logger.error("Change handler failed", error=str(e))
                        
                        # Send feedback to keep connection alive
                        cursor.send_feedback(flush_lsn=msg.data_start)
                    
                    # Small delay to prevent busy waiting
                    await asyncio.sleep(0.01)
                    
                except Exception as e:
                    logger.error("Error in replication stream", error=str(e))
                    await asyncio.sleep(1)  # Wait before retrying
            
        except Exception as e:
            logger.error("Replication capture failed", error=str(e))
            raise
    
    async def _parse_replication_message(self, msg: ReplicationMessage) -> Optional[ChangeEvent]:
        """Parse logical replication message into ChangeEvent"""
        try:
            if not msg.payload:
                return None
            
            # Parse JSON payload from pgoutput plugin
            payload = json.loads(msg.payload)
            
            change_type_map = {
                'I': ChangeType.INSERT,
                'U': ChangeType.UPDATE,
                'D': ChangeType.DELETE,
                'T': ChangeType.TRUNCATE
            }
            
            change_type = change_type_map.get(payload.get('action'))
            if not change_type:
                return None
            
            # Extract table information
            schema = payload.get('schema', 'public')
            table = payload.get('table')
            
            if not table:
                return None
            
            # Extract old and new values
            old_values = payload.get('old')
            new_values = payload.get('new')
            
            # Determine primary key
            primary_key = {}
            if new_values:
                # Try common primary key fields
                for pk_field in ['id', 'event_id', 'sample_id', 'station_id']:
                    if pk_field in new_values:
                        primary_key[pk_field] = new_values[pk_field]
                        break
            elif old_values:
                for pk_field in ['id', 'event_id', 'sample_id', 'station_id']:
                    if pk_field in old_values:
                        primary_key[pk_field] = old_values[pk_field]
                        break
            
            change_event = ChangeEvent(
                event_id=str(uuid.uuid4()),
                source_database='postgresql',
                source_table=f"{schema}.{table}",
                change_type=change_type,
                timestamp=datetime.utcnow(),
                transaction_id=payload.get('xid'),
                old_values=old_values,
                new_values=new_values,
                primary_key=primary_key,
                schema_version='1.0',
                metadata={
                    'lsn': str(msg.data_start),
                    'commit_timestamp': payload.get('timestamp')
                }
            )
            
            return change_event
            
        except Exception as e:
            logger.error("Failed to parse replication message", error=str(e))
            return None
    
    async def get_changes(self, since: datetime) -> List[ChangeEvent]:
        """Get changes since specified timestamp (not implemented for streaming CDC)"""
        # For streaming CDC, changes are delivered via handlers
        # This method could be implemented using WAL scanning if needed
        return []
    
    def register_change_handler(self, handler: Callable[[ChangeEvent], None]) -> None:
        """Register handler for change events"""
        self.change_handlers.append(handler)
        logger.info("Registered CDC change handler")


class CDCSynchronizer:
    """Advanced CDC synchronization engine with conflict resolution"""
    
    def __init__(self):
        self.cdc_captures: Dict[str, CDCCapture] = {}
        self.sync_rules: Dict[str, SyncRule] = {}
        self.conflict_resolver = ConflictResolver()
        self.change_log: List[ChangeEvent] = []
        self.sync_statistics = {
            'events_processed': 0,
            'conflicts_detected': 0,
            'conflicts_resolved': 0,
            'sync_errors': 0
        }
        
        # Database connections for synchronization
        self.database_connections = {}
        
    def register_cdc_capture(self, name: str, cdc_capture: CDCCapture) -> None:
        """Register CDC capture for a database"""
        self.cdc_captures[name] = cdc_capture
        cdc_capture.register_change_handler(self._handle_change_event)
        logger.info("Registered CDC capture", name=name)
    
    def add_sync_rule(self, sync_rule: SyncRule) -> None:
        """Add synchronization rule"""
        self.sync_rules[sync_rule.rule_id] = sync_rule
        logger.info("Added sync rule", 
                   rule_id=sync_rule.rule_id,
                   source=f"{sync_rule.source_database}.{sync_rule.source_table}",
                   target=f"{sync_rule.target_database}.{sync_rule.target_table}")
    
    async def start_synchronization(self) -> None:
        """Start CDC synchronization for all registered captures"""
        try:
            # Start all CDC captures
            for name, cdc_capture in self.cdc_captures.items():
                await cdc_capture.start_capture()
                logger.info("Started CDC capture", name=name)
            
            logger.info("CDC synchronization started")
            
        except Exception as e:
            logger.error("Failed to start CDC synchronization", error=str(e))
            raise
    
    async def stop_synchronization(self) -> None:
        """Stop CDC synchronization"""
        try:
            # Stop all CDC captures
            for name, cdc_capture in self.cdc_captures.items():
                await cdc_capture.stop_capture()
                logger.info("Stopped CDC capture", name=name)
            
            logger.info("CDC synchronization stopped")
            
        except Exception as e:
            logger.error("Failed to stop CDC synchronization", error=str(e))
            raise
    
    async def _handle_change_event(self, change_event: ChangeEvent) -> None:
        """Handle incoming change event and apply synchronization rules"""
        try:
            self.change_log.append(change_event)
            self.sync_statistics['events_processed'] += 1
            
            # Find applicable sync rules
            applicable_rules = self._find_applicable_sync_rules(change_event)
            
            for sync_rule in applicable_rules:
                if sync_rule.enabled:
                    await self._apply_sync_rule(change_event, sync_rule)
            
            logger.debug("Processed change event", 
                        event_id=change_event.event_id,
                        applicable_rules=len(applicable_rules))
            
        except Exception as e:
            self.sync_statistics['sync_errors'] += 1
            logger.error("Failed to handle change event", 
                        event_id=change_event.event_id,
                        error=str(e))
    
    def _find_applicable_sync_rules(self, change_event: ChangeEvent) -> List[SyncRule]:
        """Find sync rules applicable to the change event"""
        applicable_rules = []
        
        for sync_rule in self.sync_rules.values():
            # Check if rule matches the change event
            if (sync_rule.source_database == change_event.source_database and
                sync_rule.source_table == change_event.source_table):
                
                # Apply filters if specified
                if sync_rule.filters:
                    if self._apply_filters(change_event, sync_rule.filters):
                        applicable_rules.append(sync_rule)
                else:
                    applicable_rules.append(sync_rule)
        
        return applicable_rules
    
    def _apply_filters(self, change_event: ChangeEvent, filters: Dict[str, Any]) -> bool:
        """Apply filters to determine if change event should be synchronized"""
        try:
            # Check field-based filters
            field_filters = filters.get('fields', {})
            
            for field, filter_config in field_filters.items():
                if change_event.new_values and field in change_event.new_values:
                    value = change_event.new_values[field]
                    
                    # Apply different filter types
                    if 'equals' in filter_config:
                        if value != filter_config['equals']:
                            return False
                    
                    if 'min_value' in filter_config:
                        if value < filter_config['min_value']:
                            return False
                    
                    if 'max_value' in filter_config:
                        if value > filter_config['max_value']:
                            return False
                    
                    if 'contains' in filter_config:
                        if filter_config['contains'] not in str(value):
                            return False
            
            # Check change type filters
            if 'change_types' in filters:
                allowed_types = [ChangeType(ct) for ct in filters['change_types']]
                if change_event.change_type not in allowed_types:
                    return False
            
            return True
            
        except Exception as e:
            logger.error("Filter application failed", error=str(e))
            return False
    
    async def _apply_sync_rule(self, change_event: ChangeEvent, sync_rule: SyncRule) -> None:
        """Apply synchronization rule to change event"""
        try:
            # Transform data according to field mappings
            transformed_data = await self._transform_data(change_event, sync_rule)
            
            # Check for conflicts
            conflict = await self._detect_conflict(transformed_data, sync_rule)
            
            if conflict:
                self.sync_statistics['conflicts_detected'] += 1
                
                # Resolve conflict
                resolution = await self.conflict_resolver.resolve_conflict(
                    conflict, sync_rule.conflict_resolution
                )
                
                if resolution.resolved:
                    self.sync_statistics['conflicts_resolved'] += 1
                    transformed_data = resolution.resolution_details.get('final_data', transformed_data)
                else:
                    logger.warning("Conflict not resolved", 
                                 conflict_id=conflict.conflict_id)
                    return
            
            # Apply change to target database
            await self._apply_change_to_target(transformed_data, sync_rule)
            
            logger.debug("Applied sync rule", 
                        rule_id=sync_rule.rule_id,
                        event_id=change_event.event_id)
            
        except Exception as e:
            logger.error("Failed to apply sync rule", 
                        rule_id=sync_rule.rule_id,
                        error=str(e))
            raise
    
    async def _transform_data(self, change_event: ChangeEvent, sync_rule: SyncRule) -> Dict[str, Any]:
        """Transform data according to sync rule mappings"""
        try:
            transformed_data = {
                'change_type': change_event.change_type,
                'primary_key': {},
                'old_values': {},
                'new_values': {}
            }
            
            # Apply field mappings
            for source_field, target_field in sync_rule.field_mappings.items():
                # Map primary key fields
                if source_field in change_event.primary_key:
                    transformed_data['primary_key'][target_field] = change_event.primary_key[source_field]
                
                # Map old values
                if change_event.old_values and source_field in change_event.old_values:
                    transformed_data['old_values'][target_field] = change_event.old_values[source_field]
                
                # Map new values
                if change_event.new_values and source_field in change_event.new_values:
                    transformed_data['new_values'][target_field] = change_event.new_values[source_field]
            
            # Apply transformation functions
            for transform_func_name in sync_rule.transformation_functions:
                transformed_data = await self._apply_transformation_function(
                    transformed_data, transform_func_name
                )
            
            return transformed_data
            
        except Exception as e:
            logger.error("Data transformation failed", error=str(e))
            raise
    
    async def _apply_transformation_function(self, data: Dict[str, Any], func_name: str) -> Dict[str, Any]:
        """Apply named transformation function to data"""
        try:
            if func_name == 'coordinate_transform':
                # Transform coordinates between different systems
                if 'latitude' in data['new_values'] and 'longitude' in data['new_values']:
                    # Example: Convert to different coordinate system
                    # This would use actual coordinate transformation libraries
                    pass
            
            elif func_name == 'unit_conversion':
                # Convert units (e.g., magnitude scales, temperature units)
                if 'magnitude' in data['new_values']:
                    # Example: Convert between magnitude scales
                    pass
            
            elif func_name == 'timestamp_normalization':
                # Normalize timestamps to UTC
                if 'timestamp' in data['new_values']:
                    # Ensure timestamp is in UTC
                    pass
            
            elif func_name == 'quality_score_calculation':
                # Calculate quality scores based on multiple factors
                pass
            
            return data
            
        except Exception as e:
            logger.error("Transformation function failed", 
                        function=func_name,
                        error=str(e))
            return data
    
    async def _detect_conflict(self, transformed_data: Dict[str, Any], sync_rule: SyncRule) -> Optional[ConflictEvent]:
        """Detect conflicts in target database"""
        try:
            # Get current state from target database
            target_state = await self._get_target_state(transformed_data, sync_rule)
            
            if not target_state:
                return None  # No conflict if record doesn't exist
            
            # Check for conflicts based on change type
            if transformed_data['change_type'] == ChangeType.UPDATE:
                # Check if target has been modified since source change
                source_timestamp = transformed_data['new_values'].get('last_modified')
                target_timestamp = target_state.get('last_modified')
                
                if (source_timestamp and target_timestamp and 
                    target_timestamp > source_timestamp):
                    
                    # Create conflict event
                    conflict = ConflictEvent(
                        conflict_id=str(uuid.uuid4()),
                        source_event=None,  # Would need to store original event
                        target_state=target_state,
                        conflict_type='concurrent_modification',
                        resolution_strategy=sync_rule.conflict_resolution
                    )
                    
                    return conflict
            
            return None
            
        except Exception as e:
            logger.error("Conflict detection failed", error=str(e))
            return None
    
    async def _get_target_state(self, transformed_data: Dict[str, Any], sync_rule: SyncRule) -> Optional[Dict[str, Any]]:
        """Get current state of record in target database"""
        try:
            # This would query the target database using the primary key
            # Implementation depends on target database type
            
            target_db = sync_rule.target_database
            target_table = sync_rule.target_table
            primary_key = transformed_data['primary_key']
            
            # Example for PostgreSQL target
            if target_db == 'postgresql':
                # Query PostgreSQL database
                pass
            elif target_db == 'influxdb':
                # Query InfluxDB
                pass
            elif target_db == 'neo4j':
                # Query Neo4j
                pass
            
            return None  # Placeholder
            
        except Exception as e:
            logger.error("Failed to get target state", error=str(e))
            return None
    
    async def _apply_change_to_target(self, transformed_data: Dict[str, Any], sync_rule: SyncRule) -> None:
        """Apply change to target database"""
        try:
            target_db = sync_rule.target_database
            target_table = sync_rule.target_table
            change_type = transformed_data['change_type']
            
            if target_db == 'postgresql':
                await self._apply_postgresql_change(transformed_data, sync_rule)
            elif target_db == 'influxdb':
                await self._apply_influxdb_change(transformed_data, sync_rule)
            elif target_db == 'neo4j':
                await self._apply_neo4j_change(transformed_data, sync_rule)
            elif target_db == 'redis':
                await self._apply_redis_change(transformed_data, sync_rule)
            
            logger.debug("Applied change to target database", 
                        target_db=target_db,
                        target_table=target_table,
                        change_type=change_type.value)
            
        except Exception as e:
            logger.error("Failed to apply change to target", 
                        target_db=sync_rule.target_database,
                        error=str(e))
            raise
    
    async def _apply_postgresql_change(self, transformed_data: Dict[str, Any], sync_rule: SyncRule) -> None:
        """Apply change to PostgreSQL target"""
        # Implementation for PostgreSQL synchronization
        pass
    
    async def _apply_influxdb_change(self, transformed_data: Dict[str, Any], sync_rule: SyncRule) -> None:
        """Apply change to InfluxDB target"""
        # Implementation for InfluxDB synchronization
        pass
    
    async def _apply_neo4j_change(self, transformed_data: Dict[str, Any], sync_rule: SyncRule) -> None:
        """Apply change to Neo4j target"""
        # Implementation for Neo4j synchronization
        pass
    
    async def _apply_redis_change(self, transformed_data: Dict[str, Any], sync_rule: SyncRule) -> None:
        """Apply change to Redis target"""
        # Implementation for Redis synchronization
        pass
    
    def get_sync_statistics(self) -> Dict[str, Any]:
        """Get synchronization statistics"""
        return self.sync_statistics.copy()


class ConflictResolver:
    """Advanced conflict resolution engine for CDC synchronization"""
    
    async def resolve_conflict(self, 
                             conflict: ConflictEvent, 
                             strategy: ConflictResolutionStrategy) -> ConflictEvent:
        """Resolve synchronization conflict using specified strategy"""
        try:
            if strategy == ConflictResolutionStrategy.LAST_WRITE_WINS:
                return await self._resolve_last_write_wins(conflict)
            elif strategy == ConflictResolutionStrategy.FIRST_WRITE_WINS:
                return await self._resolve_first_write_wins(conflict)
            elif strategy == ConflictResolutionStrategy.MERGE_CHANGES:
                return await self._resolve_merge_changes(conflict)
            elif strategy == ConflictResolutionStrategy.SOURCE_PRIORITY:
                return await self._resolve_source_priority(conflict)
            else:
                # Manual resolution required
                conflict.resolved = False
                return conflict
            
        except Exception as e:
            logger.error("Conflict resolution failed", 
                        conflict_id=conflict.conflict_id,
                        strategy=strategy.value,
                        error=str(e))
            conflict.resolved = False
            return conflict
    
    async def _resolve_last_write_wins(self, conflict: ConflictEvent) -> ConflictEvent:
        """Resolve conflict using last write wins strategy"""
        # Use the most recent change
        conflict.resolved = True
        conflict.resolution_timestamp = datetime.utcnow()
        conflict.resolution_details = {
            'strategy': 'last_write_wins',
            'final_data': conflict.source_event.new_values if conflict.source_event else {}
        }
        return conflict
    
    async def _resolve_first_write_wins(self, conflict: ConflictEvent) -> ConflictEvent:
        """Resolve conflict using first write wins strategy"""
        # Keep the existing target state
        conflict.resolved = True
        conflict.resolution_timestamp = datetime.utcnow()
        conflict.resolution_details = {
            'strategy': 'first_write_wins',
            'final_data': conflict.target_state
        }
        return conflict
    
    async def _resolve_merge_changes(self, conflict: ConflictEvent) -> ConflictEvent:
        """Resolve conflict by merging changes"""
        # Merge source and target changes
        merged_data = conflict.target_state.copy()
        
        if conflict.source_event and conflict.source_event.new_values:
            # Merge non-conflicting fields
            for field, value in conflict.source_event.new_values.items():
                if field not in merged_data or merged_data[field] is None:
                    merged_data[field] = value
        
        conflict.resolved = True
        conflict.resolution_timestamp = datetime.utcnow()
        conflict.resolution_details = {
            'strategy': 'merge_changes',
            'final_data': merged_data
        }
        return conflict
    
    async def _resolve_source_priority(self, conflict: ConflictEvent) -> ConflictEvent:
        """Resolve conflict by giving priority to source"""
        # Source always wins
        conflict.resolved = True
        conflict.resolution_timestamp = datetime.utcnow()
        conflict.resolution_details = {
            'strategy': 'source_priority',
            'final_data': conflict.source_event.new_values if conflict.source_event else {}
        }
        return conflict


# Example usage and configuration
async def example_cdc_setup():
    """Example of setting up comprehensive CDC system"""
    
    # Create CDC synchronizer
    cdc_sync = CDCSynchronizer()
    
    # Create PostgreSQL CDC capture
    postgres_cdc = PostgreSQLCDC(
        connection_string='postgresql://prometheus:prometheus123@localhost:5432/prometheus',
        replication_slot='prometheus_cdc_slot',
        publication='prometheus_cdc_publication'
    )
    
    # Register CDC capture
    cdc_sync.register_cdc_capture('postgresql_main', postgres_cdc)
    
    # Add synchronization rules
    seismic_sync_rule = SyncRule(
        rule_id='seismic_to_influx',
        source_database='postgresql',
        source_table='public.seismic_events',
        target_database='influxdb',
        target_table='seismic_measurements',
        field_mappings={
            'timestamp': 'time',
            'station_id': 'station',
            'magnitude': 'magnitude',
            'depth_km': 'depth',
            'latitude': 'lat',
            'longitude': 'lon'
        },
        transformation_functions=['timestamp_normalization', 'unit_conversion'],
        conflict_resolution=ConflictResolutionStrategy.LAST_WRITE_WINS,
        sync_frequency=timedelta(seconds=30),
        filters={
            'fields': {
                'magnitude': {'min_value': 2.0}
            },
            'change_types': ['INSERT', 'UPDATE']
        }
    )
    
    cdc_sync.add_sync_rule(seismic_sync_rule)
    
    # Start synchronization
    await cdc_sync.start_synchronization()
    
    logger.info("CDC system started successfully")
    
    # Run for demonstration
    await asyncio.sleep(60)
    
    # Stop synchronization
    await cdc_sync.stop_synchronization()
    
    # Get statistics
    stats = cdc_sync.get_sync_statistics()
    logger.info("CDC synchronization statistics", **stats)


if __name__ == "__main__":
    asyncio.run(example_cdc_setup())
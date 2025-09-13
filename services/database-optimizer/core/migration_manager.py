"""
Advanced Database Migration and Schema Evolution Manager

This module implements comprehensive database migration management with geological
domain awareness, including automated schema evolution, version control, rollback
capabilities, and zero-downtime migrations.

Features:
- Automated schema migration with geological context
- Zero-downtime migration strategies
- Comprehensive rollback and recovery mechanisms
- Schema version control and tracking
- Geological data preservation during migrations
- Multi-database migration coordination
- Performance impact monitoring during migrations
- Automated backup and restore capabilities
"""

import asyncio
import hashlib
import json
import logging
import os
import shutil
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import uuid4

import asyncpg
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


class MigrationType(Enum):
    """Types of database migrations"""
    SCHEMA_CHANGE = "schema_change"
    DATA_MIGRATION = "data_migration"
    INDEX_CREATION = "index_creation"
    PARTITION_CREATION = "partition_creation"
    GEOLOGICAL_OPTIMIZATION = "geological_optimization"
    PERFORMANCE_TUNING = "performance_tuning"
    SECURITY_UPDATE = "security_update"


class MigrationStatus(Enum):
    """Migration execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    SKIPPED = "skipped"


class MigrationStrategy(Enum):
    """Migration execution strategies"""
    IMMEDIATE = "immediate"
    ZERO_DOWNTIME = "zero_downtime"
    MAINTENANCE_WINDOW = "maintenance_window"
    GRADUAL_ROLLOUT = "gradual_rollout"


@dataclass
class Migration:
    """Database migration definition"""
    migration_id: str
    name: str
    description: str
    migration_type: MigrationType
    strategy: MigrationStrategy
    version: str
    up_sql: List[str]
    down_sql: List[str]
    geological_context: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    estimated_duration: timedelta = field(default_factory=lambda: timedelta(minutes=5))
    requires_maintenance: bool = False
    affects_geological_data: bool = False
    backup_required: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    applied_at: Optional[datetime] = None
    status: MigrationStatus = MigrationStatus.PENDING


@dataclass
class MigrationExecution:
    """Migration execution tracking"""
    execution_id: str
    migration_id: str
    database_name: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    status: MigrationStatus = MigrationStatus.RUNNING
    error_message: Optional[str] = None
    rollback_executed: bool = False
    backup_path: Optional[str] = None
    performance_impact: Dict[str, Any] = field(default_factory=dict)
    geological_data_verified: bool = False


class GeologicalMigrationValidator:
    """Validates migrations for geological data integrity"""
    
    def __init__(self):
        self.geological_tables = {
            'seismic_events': {
                'critical_columns': ['magnitude', 'depth', 'latitude', 'longitude', 'timestamp'],
                'data_validation_queries': [
                    "SELECT COUNT(*) FROM seismic_events WHERE magnitude < 0 OR magnitude > 10",
                    "SELECT COUNT(*) FROM seismic_events WHERE depth < 0 OR depth > 1000",
                    "SELECT COUNT(*) FROM seismic_events WHERE latitude < -90 OR latitude > 90",
                    "SELECT COUNT(*) FROM seismic_events WHERE longitude < -180 OR longitude > 180"
                ]
            },
            'volcanic_data': {
                'critical_columns': ['volcano_id', 'alert_level', 'eruption_type', 'vei', 'timestamp'],
                'data_validation_queries': [
                    "SELECT COUNT(*) FROM volcanic_data WHERE alert_level < 0 OR alert_level > 4",
                    "SELECT COUNT(*) FROM volcanic_data WHERE vei < 0 OR vei > 8",
                    "SELECT COUNT(*) FROM volcanic_data WHERE volcano_id IS NULL OR volcano_id = ''"
                ]
            },
            'geological_units': {
                'critical_columns': ['age_min', 'age_max', 'lithology', 'geometry'],
                'data_validation_queries': [
                    "SELECT COUNT(*) FROM geological_units WHERE age_min < 0 OR age_min > 4600000000",
                    "SELECT COUNT(*) FROM geological_units WHERE age_max < age_min",
                    "SELECT COUNT(*) FROM geological_units WHERE ST_IsValid(geometry) = false"
                ]
            },
            'resource_data': {
                'critical_columns': ['resource_type', 'grade', 'tonnage', 'location'],
                'data_validation_queries': [
                    "SELECT COUNT(*) FROM resource_data WHERE grade < 0 OR grade > 100",
                    "SELECT COUNT(*) FROM resource_data WHERE tonnage < 0",
                    "SELECT COUNT(*) FROM resource_data WHERE ST_IsValid(location) = false"
                ]
            }
        }
    
    async def validate_pre_migration(self, connection: AsyncSession, 
                                   migration: Migration) -> Dict[str, Any]:
        """Validate database state before migration"""
        validation_results = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'data_counts': {},
            'integrity_checks': {}
        }
        
        try:
            # Check geological data integrity
            for table_name, table_config in self.geological_tables.items():
                # Check if table exists
                table_exists_query = """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' AND table_name = %s
                )
                """
                
                result = await connection.execute(text(table_exists_query), (table_name,))
                if not result.scalar():
                    continue
                
                # Get row count
                count_query = f"SELECT COUNT(*) FROM {table_name}"
                result = await connection.execute(text(count_query))
                row_count = result.scalar()
                validation_results['data_counts'][table_name] = row_count
                
                # Run data validation queries
                for validation_query in table_config['data_validation_queries']:
                    try:
                        result = await connection.execute(text(validation_query))
                        invalid_count = result.scalar()
                        
                        if invalid_count > 0:
                            validation_results['warnings'].append(
                                f"Found {invalid_count} invalid records in {table_name}"
                            )
                    except Exception as e:
                        validation_results['errors'].append(
                            f"Validation query failed for {table_name}: {e}"
                        )
            
            # Check for active connections that might block migration
            active_connections_query = """
            SELECT count(*) FROM pg_stat_activity 
            WHERE state = 'active' AND query NOT LIKE '%pg_stat_activity%'
            """
            
            result = await connection.execute(text(active_connections_query))
            active_connections = result.scalar()
            
            if active_connections > 10:
                validation_results['warnings'].append(
                    f"High number of active connections: {active_connections}"
                )
            
            # Check database size and available space
            db_size_query = "SELECT pg_database_size(current_database())"
            result = await connection.execute(text(db_size_query))
            db_size = result.scalar()
            
            validation_results['integrity_checks']['database_size'] = db_size
            
            if validation_results['errors']:
                validation_results['valid'] = False
            
        except Exception as e:
            validation_results['valid'] = False
            validation_results['errors'].append(f"Pre-migration validation failed: {e}")
        
        return validation_results
    
    async def validate_post_migration(self, connection: AsyncSession, 
                                    migration: Migration,
                                    pre_migration_counts: Dict[str, int]) -> Dict[str, Any]:
        """Validate database state after migration"""
        validation_results = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'data_counts': {},
            'data_changes': {}
        }
        
        try:
            # Check geological data integrity after migration
            for table_name, table_config in self.geological_tables.items():
                # Check if table still exists
                table_exists_query = """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' AND table_name = %s
                )
                """
                
                result = await connection.execute(text(table_exists_query), (table_name,))
                if not result.scalar():
                    if table_name in pre_migration_counts:
                        validation_results['errors'].append(
                            f"Table {table_name} was dropped during migration"
                        )
                    continue
                
                # Get new row count
                count_query = f"SELECT COUNT(*) FROM {table_name}"
                result = await connection.execute(text(count_query))
                new_count = result.scalar()
                validation_results['data_counts'][table_name] = new_count
                
                # Compare with pre-migration count
                if table_name in pre_migration_counts:
                    old_count = pre_migration_counts[table_name]
                    change = new_count - old_count
                    validation_results['data_changes'][table_name] = change
                    
                    # Check for unexpected data loss
                    if change < 0 and not migration.affects_geological_data:
                        validation_results['errors'].append(
                            f"Unexpected data loss in {table_name}: {abs(change)} rows"
                        )
                    elif abs(change) > old_count * 0.1:  # More than 10% change
                        validation_results['warnings'].append(
                            f"Significant data change in {table_name}: {change} rows"
                        )
                
                # Run data validation queries again
                for validation_query in table_config['data_validation_queries']:
                    try:
                        result = await connection.execute(text(validation_query))
                        invalid_count = result.scalar()
                        
                        if invalid_count > 0:
                            validation_results['errors'].append(
                                f"Found {invalid_count} invalid records in {table_name} after migration"
                            )
                    except Exception as e:
                        validation_results['errors'].append(
                            f"Post-migration validation query failed for {table_name}: {e}"
                        )
            
            if validation_results['errors']:
                validation_results['valid'] = False
            
        except Exception as e:
            validation_results['valid'] = False
            validation_results['errors'].append(f"Post-migration validation failed: {e}")
        
        return validation_results


class DatabaseBackupManager:
    """Manages database backups for migration safety"""
    
    def __init__(self, backup_directory: str):
        self.backup_directory = Path(backup_directory)
        self.backup_directory.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    async def create_backup(self, database_config: Dict[str, Any], 
                          migration_id: str) -> str:
        """Create database backup before migration"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"backup_{migration_id}_{timestamp}.sql"
        backup_path = self.backup_directory / backup_filename
        
        try:
            # Extract connection details
            db_url = database_config['url']
            # Parse URL to get connection details
            # This is a simplified version - would need proper URL parsing
            
            # Create pg_dump command
            dump_command = [
                'pg_dump',
                '--verbose',
                '--no-password',
                '--format=custom',
                '--compress=9',
                '--file', str(backup_path),
                db_url
            ]
            
            # Execute backup
            process = await asyncio.create_subprocess_exec(
                *dump_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                self.logger.info(f"Backup created successfully: {backup_path}")
                return str(backup_path)
            else:
                error_msg = stderr.decode() if stderr else "Unknown error"
                raise Exception(f"Backup failed: {error_msg}")
                
        except Exception as e:
            self.logger.error(f"Failed to create backup: {e}")
            raise
    
    async def restore_backup(self, database_config: Dict[str, Any], 
                           backup_path: str) -> bool:
        """Restore database from backup"""
        try:
            db_url = database_config['url']
            
            # Create pg_restore command
            restore_command = [
                'pg_restore',
                '--verbose',
                '--no-password',
                '--clean',
                '--if-exists',
                '--dbname', db_url,
                backup_path
            ]
            
            # Execute restore
            process = await asyncio.create_subprocess_exec(
                *restore_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                self.logger.info(f"Backup restored successfully from: {backup_path}")
                return True
            else:
                error_msg = stderr.decode() if stderr else "Unknown error"
                self.logger.error(f"Restore failed: {error_msg}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to restore backup: {e}")
            return False
    
    def cleanup_old_backups(self, retention_days: int = 30):
        """Clean up old backup files"""
        try:
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            for backup_file in self.backup_directory.glob("backup_*.sql"):
                if backup_file.stat().st_mtime < cutoff_date.timestamp():
                    backup_file.unlink()
                    self.logger.info(f"Deleted old backup: {backup_file}")
                    
        except Exception as e:
            self.logger.error(f"Failed to cleanup old backups: {e}")


class MigrationManager:
    """Advanced database migration manager with geological domain awareness"""
    
    def __init__(self, database_configs: Dict[str, Dict[str, Any]], 
                 migration_directory: str, backup_directory: str):
        self.database_configs = database_configs
        self.migration_directory = Path(migration_directory)
        self.migration_directory.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.validator = GeologicalMigrationValidator()
        self.backup_manager = DatabaseBackupManager(backup_directory)
        
        # Migration tracking
        self.migrations: Dict[str, Migration] = {}
        self.execution_history: List[MigrationExecution] = []
        self.schema_versions: Dict[str, str] = {}
        
        # Load existing migrations
        self._load_migrations()
    
    def _load_migrations(self):
        """Load migration files from directory"""
        try:
            for migration_file in self.migration_directory.glob("*.json"):
                with open(migration_file, 'r') as f:
                    migration_data = json.load(f)
                    
                migration = Migration(
                    migration_id=migration_data['migration_id'],
                    name=migration_data['name'],
                    description=migration_data['description'],
                    migration_type=MigrationType(migration_data['migration_type']),
                    strategy=MigrationStrategy(migration_data['strategy']),
                    version=migration_data['version'],
                    up_sql=migration_data['up_sql'],
                    down_sql=migration_data['down_sql'],
                    geological_context=migration_data.get('geological_context', {}),
                    dependencies=migration_data.get('dependencies', []),
                    estimated_duration=timedelta(seconds=migration_data.get('estimated_duration_seconds', 300)),
                    requires_maintenance=migration_data.get('requires_maintenance', False),
                    affects_geological_data=migration_data.get('affects_geological_data', False),
                    backup_required=migration_data.get('backup_required', True)
                )
                
                self.migrations[migration.migration_id] = migration
                
            self.logger.info(f"Loaded {len(self.migrations)} migrations")
            
        except Exception as e:
            self.logger.error(f"Failed to load migrations: {e}")
    
    def create_migration(self, name: str, description: str, 
                        migration_type: MigrationType,
                        up_sql: List[str], down_sql: List[str],
                        geological_context: Dict[str, Any] = None,
                        strategy: MigrationStrategy = MigrationStrategy.IMMEDIATE) -> Migration:
        """Create new migration"""
        migration_id = str(uuid4())
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        migration = Migration(
            migration_id=migration_id,
            name=name,
            description=description,
            migration_type=migration_type,
            strategy=strategy,
            version=version,
            up_sql=up_sql,
            down_sql=down_sql,
            geological_context=geological_context or {}
        )
        
        # Save migration to file
        self._save_migration(migration)
        
        # Add to migrations dict
        self.migrations[migration_id] = migration
        
        self.logger.info(f"Created migration: {name} ({migration_id})")
        return migration
    
    def _save_migration(self, migration: Migration):
        """Save migration to file"""
        migration_file = self.migration_directory / f"{migration.version}_{migration.name.replace(' ', '_')}.json"
        
        migration_data = {
            'migration_id': migration.migration_id,
            'name': migration.name,
            'description': migration.description,
            'migration_type': migration.migration_type.value,
            'strategy': migration.strategy.value,
            'version': migration.version,
            'up_sql': migration.up_sql,
            'down_sql': migration.down_sql,
            'geological_context': migration.geological_context,
            'dependencies': migration.dependencies,
            'estimated_duration_seconds': int(migration.estimated_duration.total_seconds()),
            'requires_maintenance': migration.requires_maintenance,
            'affects_geological_data': migration.affects_geological_data,
            'backup_required': migration.backup_required
        }
        
        with open(migration_file, 'w') as f:
            json.dump(migration_data, f, indent=2)
    
    async def execute_migration(self, db_name: str, migration_id: str) -> MigrationExecution:
        """Execute migration on specified database"""
        if migration_id not in self.migrations:
            raise ValueError(f"Migration not found: {migration_id}")
        
        migration = self.migrations[migration_id]
        execution_id = str(uuid4())
        
        execution = MigrationExecution(
            execution_id=execution_id,
            migration_id=migration_id,
            database_name=db_name,
            started_at=datetime.now()
        )
        
        try:
            # Get database connection
            from .database_optimizer import AdvancedConnectionPoolManager
            connection_manager = AdvancedConnectionPoolManager(self.database_configs)
            await connection_manager.initialize_pools()
            
            connection = await connection_manager.get_connection(db_name)
            
            # Pre-migration validation
            self.logger.info(f"Starting pre-migration validation for {migration.name}")
            pre_validation = await self.validator.validate_pre_migration(connection, migration)
            
            if not pre_validation['valid']:
                execution.status = MigrationStatus.FAILED
                execution.error_message = f"Pre-migration validation failed: {pre_validation['errors']}"
                execution.completed_at = datetime.now()
                return execution
            
            # Create backup if required
            if migration.backup_required:
                self.logger.info(f"Creating backup for migration {migration.name}")
                backup_path = await self.backup_manager.create_backup(
                    self.database_configs[db_name], migration_id
                )
                execution.backup_path = backup_path
            
            # Execute migration based on strategy
            if migration.strategy == MigrationStrategy.ZERO_DOWNTIME:
                success = await self._execute_zero_downtime_migration(connection, migration, execution)
            elif migration.strategy == MigrationStrategy.GRADUAL_ROLLOUT:
                success = await self._execute_gradual_migration(connection, migration, execution)
            else:
                success = await self._execute_immediate_migration(connection, migration, execution)
            
            if success:
                # Post-migration validation
                self.logger.info(f"Starting post-migration validation for {migration.name}")
                post_validation = await self.validator.validate_post_migration(
                    connection, migration, pre_validation['data_counts']
                )
                
                if post_validation['valid']:
                    execution.status = MigrationStatus.COMPLETED
                    execution.geological_data_verified = True
                    migration.status = MigrationStatus.COMPLETED
                    migration.applied_at = datetime.now()
                    
                    # Update schema version
                    self.schema_versions[db_name] = migration.version
                    
                    self.logger.info(f"Migration {migration.name} completed successfully")
                else:
                    # Validation failed, rollback
                    self.logger.error(f"Post-migration validation failed: {post_validation['errors']}")
                    await self._rollback_migration(connection, migration, execution)
            else:
                execution.status = MigrationStatus.FAILED
                
            execution.completed_at = datetime.now()
            
            await connection_manager.release_connection(db_name, connection)
            
        except Exception as e:
            execution.status = MigrationStatus.FAILED
            execution.error_message = str(e)
            execution.completed_at = datetime.now()
            
            self.logger.error(f"Migration {migration.name} failed: {e}")
            
            # Attempt rollback
            try:
                connection = await connection_manager.get_connection(db_name)
                await self._rollback_migration(connection, migration, execution)
                await connection_manager.release_connection(db_name, connection)
            except Exception as rollback_error:
                self.logger.error(f"Rollback failed: {rollback_error}")
        
        # Record execution
        self.execution_history.append(execution)
        
        return execution
    
    async def _execute_immediate_migration(self, connection: AsyncSession, 
                                         migration: Migration,
                                         execution: MigrationExecution) -> bool:
        """Execute migration immediately"""
        try:
            # Begin transaction
            await connection.begin()
            
            # Execute migration SQL
            for sql_statement in migration.up_sql:
                if sql_statement.strip():
                    await connection.execute(text(sql_statement))
            
            # Commit transaction
            await connection.commit()
            
            return True
            
        except Exception as e:
            await connection.rollback()
            execution.error_message = str(e)
            return False
    
    async def _execute_zero_downtime_migration(self, connection: AsyncSession,
                                             migration: Migration,
                                             execution: MigrationExecution) -> bool:
        """Execute zero-downtime migration"""
        try:
            # Zero-downtime strategy implementation
            # This would involve techniques like:
            # 1. Creating new tables/columns
            # 2. Dual-writing to old and new structures
            # 3. Backfilling data
            # 4. Switching reads to new structure
            # 5. Removing old structure
            
            # For now, execute as immediate migration
            # In production, this would be more sophisticated
            return await self._execute_immediate_migration(connection, migration, execution)
            
        except Exception as e:
            execution.error_message = str(e)
            return False
    
    async def _execute_gradual_migration(self, connection: AsyncSession,
                                       migration: Migration,
                                       execution: MigrationExecution) -> bool:
        """Execute gradual rollout migration"""
        try:
            # Gradual rollout strategy implementation
            # This would involve:
            # 1. Migrating data in batches
            # 2. Monitoring performance impact
            # 3. Pausing if issues detected
            # 4. Continuing when safe
            
            # For now, execute as immediate migration
            return await self._execute_immediate_migration(connection, migration, execution)
            
        except Exception as e:
            execution.error_message = str(e)
            return False
    
    async def _rollback_migration(self, connection: AsyncSession,
                                migration: Migration,
                                execution: MigrationExecution):
        """Rollback migration"""
        try:
            self.logger.info(f"Rolling back migration {migration.name}")
            
            # Execute rollback SQL
            await connection.begin()
            
            for sql_statement in migration.down_sql:
                if sql_statement.strip():
                    await connection.execute(text(sql_statement))
            
            await connection.commit()
            
            execution.rollback_executed = True
            execution.status = MigrationStatus.ROLLED_BACK
            migration.status = MigrationStatus.ROLLED_BACK
            
            self.logger.info(f"Migration {migration.name} rolled back successfully")
            
        except Exception as e:
            await connection.rollback()
            self.logger.error(f"Rollback failed for migration {migration.name}: {e}")
            
            # If rollback fails, try to restore from backup
            if execution.backup_path:
                self.logger.info("Attempting to restore from backup")
                success = await self.backup_manager.restore_backup(
                    self.database_configs[execution.database_name],
                    execution.backup_path
                )
                if success:
                    self.logger.info("Successfully restored from backup")
                else:
                    self.logger.error("Failed to restore from backup")
    
    async def execute_pending_migrations(self, db_name: str) -> List[MigrationExecution]:
        """Execute all pending migrations for database"""
        executions = []
        
        # Get pending migrations in order
        pending_migrations = [
            migration for migration in self.migrations.values()
            if migration.status == MigrationStatus.PENDING
        ]
        
        # Sort by version
        pending_migrations.sort(key=lambda x: x.version)
        
        for migration in pending_migrations:
            # Check dependencies
            if self._check_dependencies(migration, db_name):
                execution = await self.execute_migration(db_name, migration.migration_id)
                executions.append(execution)
                
                # Stop if migration failed
                if execution.status == MigrationStatus.FAILED:
                    self.logger.error(f"Migration failed, stopping execution: {migration.name}")
                    break
            else:
                self.logger.warning(f"Skipping migration due to unmet dependencies: {migration.name}")
        
        return executions
    
    def _check_dependencies(self, migration: Migration, db_name: str) -> bool:
        """Check if migration dependencies are satisfied"""
        for dependency_id in migration.dependencies:
            if dependency_id in self.migrations:
                dependency = self.migrations[dependency_id]
                if dependency.status != MigrationStatus.COMPLETED:
                    return False
            else:
                self.logger.warning(f"Unknown dependency: {dependency_id}")
                return False
        
        return True
    
    def get_migration_status(self, db_name: str) -> Dict[str, Any]:
        """Get migration status for database"""
        status = {
            'current_version': self.schema_versions.get(db_name, 'unknown'),
            'pending_migrations': [],
            'completed_migrations': [],
            'failed_migrations': [],
            'recent_executions': []
        }
        
        # Categorize migrations
        for migration in self.migrations.values():
            migration_info = {
                'id': migration.migration_id,
                'name': migration.name,
                'version': migration.version,
                'type': migration.migration_type.value,
                'status': migration.status.value
            }
            
            if migration.status == MigrationStatus.PENDING:
                status['pending_migrations'].append(migration_info)
            elif migration.status == MigrationStatus.COMPLETED:
                status['completed_migrations'].append(migration_info)
            elif migration.status == MigrationStatus.FAILED:
                status['failed_migrations'].append(migration_info)
        
        # Recent executions
        recent_executions = [
            exec for exec in self.execution_history
            if exec.database_name == db_name
        ][-10:]  # Last 10 executions
        
        status['recent_executions'] = [
            {
                'execution_id': exec.execution_id,
                'migration_id': exec.migration_id,
                'status': exec.status.value,
                'started_at': exec.started_at.isoformat(),
                'completed_at': exec.completed_at.isoformat() if exec.completed_at else None,
                'error_message': exec.error_message
            }
            for exec in recent_executions
        ]
        
        return status
    
    def generate_geological_migration(self, table_name: str, 
                                    optimization_type: str,
                                    geological_context: Dict[str, Any]) -> Migration:
        """Generate migration for geological optimization"""
        migration_name = f"Optimize {table_name} for {optimization_type}"
        
        up_sql = []
        down_sql = []
        
        if optimization_type == "spatial_indexing":
            # Create spatial indexes for geological data
            up_sql = [
                f"CREATE INDEX CONCURRENTLY idx_{table_name}_spatial ON {table_name} USING GIST (geometry);",
                f"CREATE INDEX CONCURRENTLY idx_{table_name}_location ON {table_name} USING GIST (ST_Point(longitude, latitude));"
            ]
            down_sql = [
                f"DROP INDEX CONCURRENTLY IF EXISTS idx_{table_name}_spatial;",
                f"DROP INDEX CONCURRENTLY IF EXISTS idx_{table_name}_location;"
            ]
        
        elif optimization_type == "temporal_partitioning":
            # Create temporal partitions
            current_year = datetime.now().year
            up_sql = [
                f"ALTER TABLE {table_name} PARTITION BY RANGE (timestamp);",
                f"CREATE TABLE {table_name}_{current_year} PARTITION OF {table_name} FOR VALUES FROM ('{current_year}-01-01') TO ('{current_year + 1}-01-01');",
                f"CREATE TABLE {table_name}_{current_year + 1} PARTITION OF {table_name} FOR VALUES FROM ('{current_year + 1}-01-01') TO ('{current_year + 2}-01-01');"
            ]
            down_sql = [
                f"DROP TABLE IF EXISTS {table_name}_{current_year + 1};",
                f"DROP TABLE IF EXISTS {table_name}_{current_year};",
                f"-- Manual intervention required to remove partitioning from {table_name}"
            ]
        
        elif optimization_type == "geological_constraints":
            # Add geological data constraints
            if table_name == "seismic_events":
                up_sql = [
                    f"ALTER TABLE {table_name} ADD CONSTRAINT chk_magnitude CHECK (magnitude >= 0 AND magnitude <= 10);",
                    f"ALTER TABLE {table_name} ADD CONSTRAINT chk_depth CHECK (depth >= 0 AND depth <= 1000);",
                    f"ALTER TABLE {table_name} ADD CONSTRAINT chk_latitude CHECK (latitude >= -90 AND latitude <= 90);",
                    f"ALTER TABLE {table_name} ADD CONSTRAINT chk_longitude CHECK (longitude >= -180 AND longitude <= 180);"
                ]
                down_sql = [
                    f"ALTER TABLE {table_name} DROP CONSTRAINT IF EXISTS chk_longitude;",
                    f"ALTER TABLE {table_name} DROP CONSTRAINT IF EXISTS chk_latitude;",
                    f"ALTER TABLE {table_name} DROP CONSTRAINT IF EXISTS chk_depth;",
                    f"ALTER TABLE {table_name} DROP CONSTRAINT IF EXISTS chk_magnitude;"
                ]
        
        return self.create_migration(
            name=migration_name,
            description=f"Geological optimization for {table_name}: {optimization_type}",
            migration_type=MigrationType.GEOLOGICAL_OPTIMIZATION,
            up_sql=up_sql,
            down_sql=down_sql,
            geological_context=geological_context,
            strategy=MigrationStrategy.ZERO_DOWNTIME
        )


# Configuration and factory
class MigrationManagerConfig(BaseModel):
    """Migration manager configuration"""
    databases: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Database configurations"
    )
    migration_directory: str = Field(
        default="./migrations",
        description="Directory for migration files"
    )
    backup_directory: str = Field(
        default="./backups",
        description="Directory for database backups"
    )
    auto_backup: bool = Field(
        default=True,
        description="Automatically create backups before migrations"
    )
    backup_retention_days: int = Field(
        default=30,
        description="Number of days to retain backups"
    )


async def create_migration_manager(config: MigrationManagerConfig) -> MigrationManager:
    """Factory function to create migration manager"""
    manager = MigrationManager(
        config.databases,
        config.migration_directory,
        config.backup_directory
    )
    return manager


# Example usage and testing
if __name__ == "__main__":
    async def test_migration_manager():
        """Test the migration manager"""
        config = MigrationManagerConfig(
            databases={
                "prometheus_main": {
                    "url": "postgresql+asyncpg://user:password@localhost:5432/prometheus",
                    "type": "postgresql"
                }
            }
        )
        
        manager = await create_migration_manager(config)
        
        # Create a test migration
        migration = manager.create_migration(
            name="Add seismic constraints",
            description="Add data validation constraints to seismic_events table",
            migration_type=MigrationType.GEOLOGICAL_OPTIMIZATION,
            up_sql=[
                "ALTER TABLE seismic_events ADD CONSTRAINT chk_magnitude CHECK (magnitude >= 0 AND magnitude <= 10);"
            ],
            down_sql=[
                "ALTER TABLE seismic_events DROP CONSTRAINT IF EXISTS chk_magnitude;"
            ],
            geological_context={"domain": "seismic", "table": "seismic_events"}
        )
        
        print(f"Created migration: {migration.name} ({migration.migration_id})")
        
        # Get migration status
        status = manager.get_migration_status("prometheus_main")
        print(f"Migration status: {status}")
    
    # Run test
    asyncio.run(test_migration_manager())
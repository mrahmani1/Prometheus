"""
Database Migration and Schema Management System
Comprehensive database schema management with version control,
automated migrations, and cross-database compatibility
"""

import asyncio
import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import hashlib
import uuid

# Database imports
from backend.core.unified_database_manager import UnifiedDatabaseManager, DatabaseType
from backend.core.database_config_manager import get_config_manager, Environment

logger = logging.getLogger(__name__)

class MigrationType(Enum):
    """Types of database migrations"""
    CREATE_TABLE = "create_table"
    ALTER_TABLE = "alter_table"
    DROP_TABLE = "drop_table"
    CREATE_INDEX = "create_index"
    DROP_INDEX = "drop_index"
    INSERT_DATA = "insert_data"
    UPDATE_DATA = "update_data"
    DELETE_DATA = "delete_data"
    CUSTOM_SQL = "custom_sql"
    EXTENSION = "extension"

class MigrationStatus(Enum):
    """Migration execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"

@dataclass
class Migration:
    """Database migration definition"""
    id: str
    version: str
    name: str
    description: str
    database_type: DatabaseType
    migration_type: MigrationType
    up_sql: str
    down_sql: str
    dependencies: List[str] = field(default_factory=list)
    checksum: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    executed_at: Optional[datetime] = None
    status: MigrationStatus = MigrationStatus.PENDING
    execution_time: float = 0.0
    error_message: Optional[str] = None
    
    def __post_init__(self):
        """Calculate checksum after initialization"""
        if not self.checksum:
            self.checksum = self._calculate_checksum()
    
    def _calculate_checksum(self) -> str:
        """Calculate migration checksum"""
        content = f"{self.up_sql}{self.down_sql}{self.version}"
        return hashlib.sha256(content.encode()).hexdigest()

@dataclass
class SchemaVersion:
    """Database schema version information"""
    database_type: DatabaseType
    version: str
    applied_migrations: List[str]
    last_migration_date: datetime
    schema_checksum: str
    
class DatabaseMigrationManager:
    """
    Database Migration and Schema Management System
    Handles schema versioning, migrations, and cross-database compatibility
    """
    
    def __init__(self, db_manager: UnifiedDatabaseManager, migrations_dir: Optional[str] = None):
        self.db_manager = db_manager
        self.migrations_dir = Path(migrations_dir or "backend/migrations")
        
        # Migration storage
        self.migrations: Dict[str, Migration] = {}
        self.schema_versions: Dict[DatabaseType, SchemaVersion] = {}
        
        # Migration execution tracking
        self.execution_history: List[Dict[str, Any]] = []
        
        # Ensure migrations directory exists
        self.migrations_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Database Migration Manager initialized")
    
    async def initialize(self) -> bool:
        """Initialize migration system"""
        try:
            logger.info("Initializing database migration system")
            
            # Create migration tracking tables
            await self._create_migration_tables()
            
            # Load existing migrations
            await self._load_migrations_from_files()
            
            # Load migration history from database
            await self._load_migration_history()
            
            # Update schema versions
            await self._update_schema_versions()
            
            logger.info("Migration system initialization complete")
            return True
            
        except Exception as e:
            logger.error(f"Migration system initialization failed: {e}")
            return False
    
    async def _create_migration_tables(self):
        """Create migration tracking tables in each database"""
        
        # PostgreSQL migration tables
        if DatabaseType.POSTGRESQL in self.db_manager.connections:
            await self._create_postgresql_migration_tables()
        
        # InfluxDB migration tracking (using measurements)
        if DatabaseType.INFLUXDB in self.db_manager.connections:
            await self._create_influxdb_migration_tracking()
        
        # Neo4j migration tracking (using nodes)
        if DatabaseType.NEO4J in self.db_manager.connections:
            await self._create_neo4j_migration_tracking()
        
        # Redis migration tracking (using keys)
        if DatabaseType.REDIS in self.db_manager.connections:
            await self._create_redis_migration_tracking()
    
    async def _create_postgresql_migration_tables(self):
        """Create PostgreSQL migration tracking tables"""
        try:
            # Schema migrations table
            schema_migrations_sql = """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    migration_id VARCHAR(255) UNIQUE NOT NULL,
                    version VARCHAR(50) NOT NULL,
                    name VARCHAR(255) NOT NULL,
                    description TEXT,
                    database_type VARCHAR(50) NOT NULL,
                    migration_type VARCHAR(50) NOT NULL,
                    up_sql TEXT NOT NULL,
                    down_sql TEXT NOT NULL,
                    dependencies JSONB DEFAULT '[]',
                    checksum VARCHAR(64) NOT NULL,
                    status VARCHAR(20) DEFAULT 'pending',
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    executed_at TIMESTAMPTZ,
                    execution_time FLOAT DEFAULT 0,
                    error_message TEXT
                )
            """
            
            await self.db_manager.execute_query(
                DatabaseType.POSTGRESQL, 
                schema_migrations_sql, 
                use_cache=False
            )
            
            # Schema versions table
            schema_versions_sql = """
                CREATE TABLE IF NOT EXISTS schema_versions (
                    database_type VARCHAR(50) PRIMARY KEY,
                    version VARCHAR(50) NOT NULL,
                    applied_migrations JSONB DEFAULT '[]',
                    last_migration_date TIMESTAMPTZ DEFAULT NOW(),
                    schema_checksum VARCHAR(64),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                )
            """
            
            await self.db_manager.execute_query(
                DatabaseType.POSTGRESQL, 
                schema_versions_sql, 
                use_cache=False
            )
            
            # Create indexes
            indexes_sql = [
                "CREATE INDEX IF NOT EXISTS idx_schema_migrations_version ON schema_migrations (version)",
                "CREATE INDEX IF NOT EXISTS idx_schema_migrations_status ON schema_migrations (status)",
                "CREATE INDEX IF NOT EXISTS idx_schema_migrations_type ON schema_migrations (database_type)"
            ]
            
            for index_sql in indexes_sql:
                await self.db_manager.execute_query(
                    DatabaseType.POSTGRESQL, 
                    index_sql, 
                    use_cache=False
                )
            
            logger.info("PostgreSQL migration tables created")
            
        except Exception as e:
            logger.error(f"Failed to create PostgreSQL migration tables: {e}")
    
    async def _create_influxdb_migration_tracking(self):
        """Create InfluxDB migration tracking"""
        try:
            # InfluxDB uses measurements for tracking
            # We'll store migration info as points in a special measurement
            
            from influxdb_client import Point, WritePrecision
            
            connection = self.db_manager.connections[DatabaseType.INFLUXDB]
            if connection.status.value != "connected":
                return
            
            # Create initial migration tracking point
            point = Point("migration_tracking") \
                .tag("database_type", "influxdb") \
                .field("initialized", True) \
                .field("version", "1.0.0") \
                .time(datetime.utcnow(), WritePrecision.NS)
            
            write_api = connection.connection.write_api()
            bucket = self.db_manager.configs[DatabaseType.INFLUXDB].additional_params.get('bucket', 'geological_data')
            
            await write_api.write(bucket=bucket, record=point)
            
            logger.info("InfluxDB migration tracking created")
            
        except Exception as e:
            logger.error(f"Failed to create InfluxDB migration tracking: {e}")
    
    async def _create_neo4j_migration_tracking(self):
        """Create Neo4j migration tracking"""
        try:
            connection = self.db_manager.connections[DatabaseType.NEO4J]
            if connection.status.value != "connected":
                return
            
            # Create migration tracking nodes and constraints
            queries = [
                "CREATE CONSTRAINT migration_id IF NOT EXISTS FOR (m:Migration) REQUIRE m.id IS UNIQUE",
                "CREATE CONSTRAINT schema_version_db_type IF NOT EXISTS FOR (s:SchemaVersion) REQUIRE s.database_type IS UNIQUE",
                """
                MERGE (s:SchemaVersion {database_type: 'neo4j'})
                SET s.version = '1.0.0',
                    s.initialized_at = datetime(),
                    s.applied_migrations = []
                """
            ]
            
            async with connection.connection.session() as session:
                for query in queries:
                    await session.run(query)
            
            logger.info("Neo4j migration tracking created")
            
        except Exception as e:
            logger.error(f"Failed to create Neo4j migration tracking: {e}")
    
    async def _create_redis_migration_tracking(self):
        """Create Redis migration tracking"""
        try:
            connection = self.db_manager.connections[DatabaseType.REDIS]
            if connection.status.value != "connected":
                return
            
            # Initialize migration tracking in Redis
            migration_info = {
                'database_type': 'redis',
                'version': '1.0.0',
                'initialized_at': datetime.utcnow().isoformat(),
                'applied_migrations': []
            }
            
            set_query = f"SET migration_tracking:redis {json.dumps(migration_info)}"
            await self.db_manager.execute_query(DatabaseType.REDIS, set_query, use_cache=False)
            
            logger.info("Redis migration tracking created")
            
        except Exception as e:
            logger.error(f"Failed to create Redis migration tracking: {e}")
    
    async def _load_migrations_from_files(self):
        """Load migration definitions from files"""
        try:
            migration_files = list(self.migrations_dir.glob("*.json"))
            
            for migration_file in migration_files:
                try:
                    with open(migration_file, 'r') as f:
                        migration_data = json.load(f)
                    
                    migration = Migration(
                        id=migration_data['id'],
                        version=migration_data['version'],
                        name=migration_data['name'],
                        description=migration_data['description'],
                        database_type=DatabaseType(migration_data['database_type']),
                        migration_type=MigrationType(migration_data['migration_type']),
                        up_sql=migration_data['up_sql'],
                        down_sql=migration_data['down_sql'],
                        dependencies=migration_data.get('dependencies', [])
                    )
                    
                    self.migrations[migration.id] = migration
                    
                except Exception as e:
                    logger.error(f"Error loading migration file {migration_file}: {e}")
            
            logger.info(f"Loaded {len(self.migrations)} migrations from files")
            
        except Exception as e:
            logger.error(f"Failed to load migrations from files: {e}")
    
    async def _load_migration_history(self):
        """Load migration execution history from database"""
        try:
            if DatabaseType.POSTGRESQL not in self.db_manager.connections:
                return
            
            query = """
                SELECT migration_id, version, name, database_type, status,
                       executed_at, execution_time, error_message
                FROM schema_migrations
                ORDER BY executed_at DESC
            """
            
            result = await self.db_manager.execute_query(DatabaseType.POSTGRESQL, query)
            
            for record in result.data or []:
                self.execution_history.append({
                    'migration_id': record['migration_id'],
                    'version': record['version'],
                    'name': record['name'],
                    'database_type': record['database_type'],
                    'status': record['status'],
                    'executed_at': record['executed_at'],
                    'execution_time': record['execution_time'],
                    'error_message': record['error_message']
                })
                
                # Update migration status if loaded
                if record['migration_id'] in self.migrations:
                    migration = self.migrations[record['migration_id']]
                    migration.status = MigrationStatus(record['status'])
                    migration.executed_at = record['executed_at']
                    migration.execution_time = record['execution_time'] or 0.0
                    migration.error_message = record['error_message']
            
            logger.info(f"Loaded {len(self.execution_history)} migration history records")
            
        except Exception as e:
            logger.error(f"Failed to load migration history: {e}")
    
    async def _update_schema_versions(self):
        """Update schema version information"""
        try:
            for db_type in DatabaseType:
                if db_type not in self.db_manager.connections:
                    continue
                
                # Get applied migrations for this database type
                applied_migrations = [
                    migration.id for migration in self.migrations.values()
                    if migration.database_type == db_type and migration.status == MigrationStatus.COMPLETED
                ]
                
                # Calculate current version (highest applied migration version)
                current_version = "0.0.0"
                last_migration_date = datetime.utcnow()
                
                if applied_migrations:
                    versions = [
                        self.migrations[mid].version for mid in applied_migrations
                        if mid in self.migrations
                    ]
                    if versions:
                        current_version = max(versions, key=lambda v: tuple(map(int, v.split('.'))))
                        
                        # Get last migration date
                        last_migrations = [
                            self.migrations[mid] for mid in applied_migrations
                            if mid in self.migrations and self.migrations[mid].executed_at
                        ]
                        if last_migrations:
                            last_migration_date = max(m.executed_at for m in last_migrations)
                
                # Calculate schema checksum
                schema_checksum = self._calculate_schema_checksum(db_type, applied_migrations)
                
                self.schema_versions[db_type] = SchemaVersion(
                    database_type=db_type,
                    version=current_version,
                    applied_migrations=applied_migrations,
                    last_migration_date=last_migration_date,
                    schema_checksum=schema_checksum
                )
                
                # Store in database
                await self._store_schema_version(db_type, self.schema_versions[db_type])
            
        except Exception as e:
            logger.error(f"Failed to update schema versions: {e}")
    
    def _calculate_schema_checksum(self, db_type: DatabaseType, applied_migrations: List[str]) -> str:
        """Calculate schema checksum based on applied migrations"""
        try:
            # Sort migrations and concatenate their checksums
            sorted_migrations = sorted(applied_migrations)
            checksums = [
                self.migrations[mid].checksum for mid in sorted_migrations
                if mid in self.migrations
            ]
            
            combined_checksum = ''.join(checksums)
            return hashlib.sha256(combined_checksum.encode()).hexdigest()
            
        except Exception as e:
            logger.error(f"Error calculating schema checksum: {e}")
            return ""
    
    async def _store_schema_version(self, db_type: DatabaseType, schema_version: SchemaVersion):
        """Store schema version in database"""
        try:
            if DatabaseType.POSTGRESQL in self.db_manager.connections:
                query = """
                    INSERT INTO schema_versions 
                    (database_type, version, applied_migrations, last_migration_date, schema_checksum)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (database_type) 
                    DO UPDATE SET 
                        version = $2,
                        applied_migrations = $3,
                        last_migration_date = $4,
                        schema_checksum = $5,
                        updated_at = NOW()
                """
                
                params = {
                    'database_type': db_type.value,
                    'version': schema_version.version,
                    'applied_migrations': json.dumps(schema_version.applied_migrations),
                    'last_migration_date': schema_version.last_migration_date,
                    'schema_checksum': schema_version.schema_checksum
                }
                
                await self.db_manager.execute_query(DatabaseType.POSTGRESQL, query, params, use_cache=False)
            
        except Exception as e:
            logger.error(f"Failed to store schema version: {e}")
    
    async def create_migration(self, 
                             name: str,
                             description: str,
                             database_type: DatabaseType,
                             migration_type: MigrationType,
                             up_sql: str,
                             down_sql: str,
                             version: Optional[str] = None,
                             dependencies: Optional[List[str]] = None) -> Migration:
        """Create a new migration"""
        try:
            # Generate version if not provided
            if not version:
                version = self._generate_next_version(database_type)
            
            # Generate migration ID
            migration_id = f"{database_type.value}_{version}_{name.lower().replace(' ', '_')}"
            
            migration = Migration(
                id=migration_id,
                version=version,
                name=name,
                description=description,
                database_type=database_type,
                migration_type=migration_type,
                up_sql=up_sql,
                down_sql=down_sql,
                dependencies=dependencies or []
            )
            
            # Store migration
            self.migrations[migration.id] = migration
            
            # Save to file
            await self._save_migration_to_file(migration)
            
            # Store in database
            await self._store_migration_in_database(migration)
            
            logger.info(f"Created migration: {migration.id}")
            return migration
            
        except Exception as e:
            logger.error(f"Failed to create migration: {e}")
            raise
    
    def _generate_next_version(self, database_type: DatabaseType) -> str:
        """Generate next version number for database type"""
        try:
            # Get existing versions for this database type
            existing_versions = [
                migration.version for migration in self.migrations.values()
                if migration.database_type == database_type
            ]
            
            if not existing_versions:
                return "1.0.0"
            
            # Find highest version
            highest_version = max(existing_versions, key=lambda v: tuple(map(int, v.split('.'))))
            major, minor, patch = map(int, highest_version.split('.'))
            
            # Increment patch version
            return f"{major}.{minor}.{patch + 1}"
            
        except Exception as e:
            logger.error(f"Error generating next version: {e}")
            return "1.0.0"
    
    async def _save_migration_to_file(self, migration: Migration):
        """Save migration to file"""
        try:
            migration_file = self.migrations_dir / f"{migration.id}.json"
            
            migration_data = {
                'id': migration.id,
                'version': migration.version,
                'name': migration.name,
                'description': migration.description,
                'database_type': migration.database_type.value,
                'migration_type': migration.migration_type.value,
                'up_sql': migration.up_sql,
                'down_sql': migration.down_sql,
                'dependencies': migration.dependencies,
                'checksum': migration.checksum,
                'created_at': migration.created_at.isoformat()
            }
            
            with open(migration_file, 'w') as f:
                json.dump(migration_data, f, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to save migration to file: {e}")
    
    async def _store_migration_in_database(self, migration: Migration):
        """Store migration definition in database"""
        try:
            if DatabaseType.POSTGRESQL not in self.db_manager.connections:
                return
            
            query = """
                INSERT INTO schema_migrations 
                (migration_id, version, name, description, database_type, migration_type,
                 up_sql, down_sql, dependencies, checksum, status, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                ON CONFLICT (migration_id) DO NOTHING
            """
            
            params = {
                'migration_id': migration.id,
                'version': migration.version,
                'name': migration.name,
                'description': migration.description,
                'database_type': migration.database_type.value,
                'migration_type': migration.migration_type.value,
                'up_sql': migration.up_sql,
                'down_sql': migration.down_sql,
                'dependencies': json.dumps(migration.dependencies),
                'checksum': migration.checksum,
                'status': migration.status.value,
                'created_at': migration.created_at
            }
            
            await self.db_manager.execute_query(DatabaseType.POSTGRESQL, query, params, use_cache=False)
            
        except Exception as e:
            logger.error(f"Failed to store migration in database: {e}")
    
    async def execute_migration(self, migration_id: str) -> bool:
        """Execute a specific migration"""
        try:
            if migration_id not in self.migrations:
                logger.error(f"Migration not found: {migration_id}")
                return False
            
            migration = self.migrations[migration_id]
            
            # Check if already executed
            if migration.status == MigrationStatus.COMPLETED:
                logger.info(f"Migration already executed: {migration_id}")
                return True
            
            # Check dependencies
            if not await self._check_dependencies(migration):
                logger.error(f"Migration dependencies not met: {migration_id}")
                return False
            
            # Execute migration
            logger.info(f"Executing migration: {migration_id}")
            migration.status = MigrationStatus.RUNNING
            
            start_time = datetime.utcnow()
            
            try:
                # Execute the migration SQL
                await self._execute_migration_sql(migration)
                
                # Update status
                migration.status = MigrationStatus.COMPLETED
                migration.executed_at = datetime.utcnow()
                migration.execution_time = (migration.executed_at - start_time).total_seconds()
                
                # Update database record
                await self._update_migration_status(migration)
                
                # Update schema version
                await self._update_schema_versions()
                
                logger.info(f"Migration executed successfully: {migration_id}")
                return True
                
            except Exception as e:
                # Update status on failure
                migration.status = MigrationStatus.FAILED
                migration.error_message = str(e)
                migration.execution_time = (datetime.utcnow() - start_time).total_seconds()
                
                await self._update_migration_status(migration)
                
                logger.error(f"Migration execution failed: {migration_id} - {e}")
                return False
                
        except Exception as e:
            logger.error(f"Error executing migration {migration_id}: {e}")
            return False
    
    async def _check_dependencies(self, migration: Migration) -> bool:
        """Check if migration dependencies are satisfied"""
        try:
            for dep_id in migration.dependencies:
                if dep_id not in self.migrations:
                    logger.error(f"Dependency not found: {dep_id}")
                    return False
                
                dep_migration = self.migrations[dep_id]
                if dep_migration.status != MigrationStatus.COMPLETED:
                    logger.error(f"Dependency not completed: {dep_id}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking dependencies: {e}")
            return False
    
    async def _execute_migration_sql(self, migration: Migration):
        """Execute migration SQL"""
        try:
            # Split SQL into individual statements
            statements = [stmt.strip() for stmt in migration.up_sql.split(';') if stmt.strip()]
            
            for statement in statements:
                await self.db_manager.execute_query(
                    migration.database_type,
                    statement,
                    use_cache=False
                )
            
        except Exception as e:
            logger.error(f"Error executing migration SQL: {e}")
            raise
    
    async def _update_migration_status(self, migration: Migration):
        """Update migration status in database"""
        try:
            if DatabaseType.POSTGRESQL not in self.db_manager.connections:
                return
            
            query = """
                UPDATE schema_migrations 
                SET status = $1, executed_at = $2, execution_time = $3, error_message = $4
                WHERE migration_id = $5
            """
            
            params = {
                'status': migration.status.value,
                'executed_at': migration.executed_at,
                'execution_time': migration.execution_time,
                'error_message': migration.error_message,
                'migration_id': migration.id
            }
            
            await self.db_manager.execute_query(DatabaseType.POSTGRESQL, query, params, use_cache=False)
            
        except Exception as e:
            logger.error(f"Failed to update migration status: {e}")
    
    async def rollback_migration(self, migration_id: str) -> bool:
        """Rollback a specific migration"""
        try:
            if migration_id not in self.migrations:
                logger.error(f"Migration not found: {migration_id}")
                return False
            
            migration = self.migrations[migration_id]
            
            if migration.status != MigrationStatus.COMPLETED:
                logger.error(f"Migration not completed, cannot rollback: {migration_id}")
                return False
            
            logger.info(f"Rolling back migration: {migration_id}")
            
            # Execute rollback SQL
            statements = [stmt.strip() for stmt in migration.down_sql.split(';') if stmt.strip()]
            
            for statement in statements:
                await self.db_manager.execute_query(
                    migration.database_type,
                    statement,
                    use_cache=False
                )
            
            # Update status
            migration.status = MigrationStatus.ROLLED_BACK
            await self._update_migration_status(migration)
            
            # Update schema version
            await self._update_schema_versions()
            
            logger.info(f"Migration rolled back successfully: {migration_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error rolling back migration {migration_id}: {e}")
            return False
    
    async def migrate_to_version(self, database_type: DatabaseType, target_version: str) -> bool:
        """Migrate database to specific version"""
        try:
            logger.info(f"Migrating {database_type.value} to version {target_version}")
            
            # Get migrations for this database type up to target version
            applicable_migrations = [
                migration for migration in self.migrations.values()
                if (migration.database_type == database_type and 
                    self._version_compare(migration.version, target_version) <= 0 and
                    migration.status != MigrationStatus.COMPLETED)
            ]
            
            # Sort by version
            applicable_migrations.sort(key=lambda m: tuple(map(int, m.version.split('.'))))
            
            # Execute migrations in order
            for migration in applicable_migrations:
                success = await self.execute_migration(migration.id)
                if not success:
                    logger.error(f"Migration failed, stopping at {migration.id}")
                    return False
            
            logger.info(f"Successfully migrated {database_type.value} to version {target_version}")
            return True
            
        except Exception as e:
            logger.error(f"Error migrating to version {target_version}: {e}")
            return False
    
    def _version_compare(self, version1: str, version2: str) -> int:
        """Compare two version strings"""
        try:
            v1_parts = tuple(map(int, version1.split('.')))
            v2_parts = tuple(map(int, version2.split('.')))
            
            if v1_parts < v2_parts:
                return -1
            elif v1_parts > v2_parts:
                return 1
            else:
                return 0
                
        except Exception:
            return 0
    
    async def get_pending_migrations(self, database_type: Optional[DatabaseType] = None) -> List[Migration]:
        """Get list of pending migrations"""
        try:
            pending = [
                migration for migration in self.migrations.values()
                if migration.status == MigrationStatus.PENDING and
                (database_type is None or migration.database_type == database_type)
            ]
            
            # Sort by version
            pending.sort(key=lambda m: tuple(map(int, m.version.split('.'))))
            
            return pending
            
        except Exception as e:
            logger.error(f"Error getting pending migrations: {e}")
            return []
    
    async def get_migration_status(self) -> Dict[str, Any]:
        """Get overall migration status"""
        try:
            status = {
                'schema_versions': {},
                'migration_counts': {},
                'pending_migrations': {},
                'failed_migrations': []
            }
            
            for db_type in DatabaseType:
                if db_type in self.schema_versions:
                    schema_version = self.schema_versions[db_type]
                    status['schema_versions'][db_type.value] = {
                        'version': schema_version.version,
                        'applied_migrations_count': len(schema_version.applied_migrations),
                        'last_migration_date': schema_version.last_migration_date.isoformat(),
                        'schema_checksum': schema_version.schema_checksum
                    }
                
                # Count migrations by status
                db_migrations = [m for m in self.migrations.values() if m.database_type == db_type]
                status['migration_counts'][db_type.value] = {
                    'total': len(db_migrations),
                    'completed': len([m for m in db_migrations if m.status == MigrationStatus.COMPLETED]),
                    'pending': len([m for m in db_migrations if m.status == MigrationStatus.PENDING]),
                    'failed': len([m for m in db_migrations if m.status == MigrationStatus.FAILED])
                }
                
                # Pending migrations
                pending = await self.get_pending_migrations(db_type)
                status['pending_migrations'][db_type.value] = [
                    {
                        'id': m.id,
                        'version': m.version,
                        'name': m.name,
                        'description': m.description
                    }
                    for m in pending
                ]
            
            # Failed migrations
            failed = [m for m in self.migrations.values() if m.status == MigrationStatus.FAILED]
            status['failed_migrations'] = [
                {
                    'id': m.id,
                    'version': m.version,
                    'name': m.name,
                    'database_type': m.database_type.value,
                    'error_message': m.error_message
                }
                for m in failed
            ]
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting migration status: {e}")
            return {'error': str(e)}

# Global migration manager instance
_migration_manager: Optional[DatabaseMigrationManager] = None

async def get_migration_manager(db_manager: UnifiedDatabaseManager) -> DatabaseMigrationManager:
    """Get global migration manager instance"""
    global _migration_manager
    
    if _migration_manager is None:
        _migration_manager = DatabaseMigrationManager(db_manager)
        await _migration_manager.initialize()
    
    return _migration_manager
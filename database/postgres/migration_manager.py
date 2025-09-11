#!/usr/bin/env python3
"""
Database Migration Manager for Prometheus AI Platform
Comprehensive database migration and versioning system
"""

import os
import sys
import logging
import psycopg2
import psycopg2.extras
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import hashlib
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatabaseMigrationManager:
    """Advanced database migration manager with versioning and rollback capabilities"""
    
    def __init__(self, database_url: str, migrations_path: str = "migrations"):
        self.database_url = database_url
        self.migrations_path = Path(migrations_path)
        self.connection = None
        
    def connect(self):
        """Establish database connection"""
        try:
            self.connection = psycopg2.connect(self.database_url)
            self.connection.autocommit = False
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def disconnect(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")
    
    def initialize_migration_system(self):
        """Initialize migration tracking tables"""
        try:
            with self.connection.cursor() as cursor:
                # Create schema migrations table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS schema_migrations (
                        version VARCHAR(20) PRIMARY KEY,
                        description TEXT,
                        applied_at TIMESTAMPTZ DEFAULT NOW(),
                        checksum VARCHAR(64),
                        execution_time_ms INTEGER,
                        rollback_sql TEXT
                    )
                """)
                
                # Create migration log table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS migration_log (
                        id SERIAL PRIMARY KEY,
                        migration_version VARCHAR(20),
                        status VARCHAR(20) CHECK (status IN ('started', 'completed', 'failed', 'rolled_back')),
                        executed_at TIMESTAMPTZ DEFAULT NOW(),
                        execution_time_ms INTEGER,
                        error_message TEXT,
                        executed_by VARCHAR(100) DEFAULT current_user
                    )
                """)
                
                # Create database version table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS database_version (
                        id SERIAL PRIMARY KEY,
                        version VARCHAR(20) NOT NULL,
                        release_date DATE,
                        description TEXT,
                        is_current BOOLEAN DEFAULT FALSE,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)
                
                self.connection.commit()
                logger.info("Migration system initialized successfully")
                
        except Exception as e:
            self.connection.rollback()
            logger.error(f"Failed to initialize migration system: {e}")
            raise
    
    def get_applied_migrations(self) -> List[str]:
        """Get list of applied migrations"""
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("SELECT version FROM schema_migrations ORDER BY version")
                return [row[0] for row in cursor.fetchall()]
        except psycopg2.ProgrammingError:
            # Table doesn't exist yet
            return []
    
    def get_pending_migrations(self) -> List[Dict]:
        """Get list of pending migrations"""
        applied_migrations = set(self.get_applied_migrations())
        pending_migrations = []
        
        if not self.migrations_path.exists():
            logger.warning(f"Migrations directory {self.migrations_path} does not exist")
            return pending_migrations
        
        for migration_file in sorted(self.migrations_path.glob("*.sql")):
            migration_name = migration_file.stem
            version = migration_name.split('_')[0]
            
            if version not in applied_migrations:
                with open(migration_file, 'r') as f:
                    content = f.read()
                
                pending_migrations.append({
                    'version': version,
                    'name': migration_name,
                    'file_path': migration_file,
                    'content': content,
                    'checksum': hashlib.sha256(content.encode()).hexdigest()
                })
        
        return pending_migrations
    
    def apply_migration(self, migration: Dict) -> bool:
        """Apply a single migration"""
        start_time = datetime.now()
        
        try:
            logger.info(f"Applying migration {migration['version']}: {migration['name']}")
            
            # Log migration start
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO migration_log (migration_version, status, executed_at)
                    VALUES (%s, 'started', %s)
                """, (migration['version'], start_time))
            
            # Execute migration SQL
            with self.connection.cursor() as cursor:
                cursor.execute(migration['content'])
            
            # Record successful migration
            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
            
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO schema_migrations (version, description, checksum, execution_time_ms)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (version) DO UPDATE SET
                        applied_at = NOW(),
                        checksum = EXCLUDED.checksum,
                        execution_time_ms = EXCLUDED.execution_time_ms
                """, (
                    migration['version'],
                    f"Migration {migration['name']}",
                    migration['checksum'],
                    execution_time
                ))
                
                cursor.execute("""
                    INSERT INTO migration_log (migration_version, status, execution_time_ms)
                    VALUES (%s, 'completed', %s)
                """, (migration['version'], execution_time))
            
            self.connection.commit()
            logger.info(f"Migration {migration['version']} applied successfully in {execution_time}ms")
            return True
            
        except Exception as e:
            self.connection.rollback()
            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
            
            # Log migration failure
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO migration_log (migration_version, status, execution_time_ms, error_message)
                    VALUES (%s, 'failed', %s, %s)
                """, (migration['version'], execution_time, str(e)))
            
            self.connection.commit()
            logger.error(f"Migration {migration['version']} failed: {e}")
            return False
    
    def migrate(self) -> bool:
        """Apply all pending migrations"""
        pending_migrations = self.get_pending_migrations()
        
        if not pending_migrations:
            logger.info("No pending migrations found")
            return True
        
        logger.info(f"Found {len(pending_migrations)} pending migrations")
        
        success_count = 0
        for migration in pending_migrations:
            if self.apply_migration(migration):
                success_count += 1
            else:
                logger.error(f"Migration failed, stopping at {migration['version']}")
                break
        
        logger.info(f"Applied {success_count}/{len(pending_migrations)} migrations successfully")
        return success_count == len(pending_migrations)
    
    def get_current_version(self) -> Optional[str]:
        """Get current database version"""
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    SELECT version FROM database_version 
                    WHERE is_current = TRUE 
                    ORDER BY created_at DESC 
                    LIMIT 1
                """)
                result = cursor.fetchone()
                return result[0] if result else None
        except Exception:
            return None
    
    def set_database_version(self, version: str, description: str = ""):
        """Set current database version"""
        try:
            with self.connection.cursor() as cursor:
                # Mark all versions as not current
                cursor.execute("UPDATE database_version SET is_current = FALSE")
                
                # Insert new version
                cursor.execute("""
                    INSERT INTO database_version (version, description, is_current, release_date)
                    VALUES (%s, %s, TRUE, CURRENT_DATE)
                """, (version, description))
                
            self.connection.commit()
            logger.info(f"Database version set to {version}")
            
        except Exception as e:
            self.connection.rollback()
            logger.error(f"Failed to set database version: {e}")
            raise
    
    def validate_schema_integrity(self) -> bool:
        """Validate database schema integrity"""
        try:
            with self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                # Check required schemas exist
                required_schemas = [
                    'geological_core', 'rock_mechanics', 'volcanic_systems',
                    'seismic_networks', 'geodetic_data', 'geophysical_data',
                    'geological_age', 'model_management', 'mlops',
                    'external_data', 'backup_archive', 'performance_monitoring'
                ]
                
                cursor.execute("""
                    SELECT schema_name FROM information_schema.schemata 
                    WHERE schema_name = ANY(%s)
                """, (required_schemas,))
                
                existing_schemas = [row['schema_name'] for row in cursor.fetchall()]
                missing_schemas = set(required_schemas) - set(existing_schemas)
                
                if missing_schemas:
                    logger.error(f"Missing required schemas: {missing_schemas}")
                    return False
                
                # Check required extensions
                required_extensions = ['postgis', 'timescaledb', 'pg_stat_statements']
                cursor.execute("""
                    SELECT extname FROM pg_extension 
                    WHERE extname = ANY(%s)
                """)
                
                existing_extensions = [row['extname'] for row in cursor.fetchall()]
                missing_extensions = set(required_extensions) - set(existing_extensions)
                
                if missing_extensions:
                    logger.error(f"Missing required extensions: {missing_extensions}")
                    return False
                
                logger.info("Schema integrity validation passed")
                return True
                
        except Exception as e:
            logger.error(f"Schema integrity validation failed: {e}")
            return False
    
    def generate_migration_report(self) -> Dict:
        """Generate comprehensive migration report"""
        try:
            with self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                # Get migration history
                cursor.execute("""
                    SELECT * FROM schema_migrations 
                    ORDER BY applied_at DESC
                """)
                migrations = cursor.fetchall()
                
                # Get migration log
                cursor.execute("""
                    SELECT * FROM migration_log 
                    ORDER BY executed_at DESC 
                    LIMIT 50
                """)
                migration_log = cursor.fetchall()
                
                # Get current version
                current_version = self.get_current_version()
                
                # Get pending migrations
                pending_migrations = self.get_pending_migrations()
                
                report = {
                    'current_version': current_version,
                    'applied_migrations_count': len(migrations),
                    'pending_migrations_count': len(pending_migrations),
                    'applied_migrations': [dict(m) for m in migrations],
                    'pending_migrations': [
                        {
                            'version': m['version'],
                            'name': m['name'],
                            'checksum': m['checksum']
                        } for m in pending_migrations
                    ],
                    'recent_log_entries': [dict(l) for l in migration_log],
                    'schema_integrity_valid': self.validate_schema_integrity(),
                    'generated_at': datetime.now().isoformat()
                }
                
                return report
                
        except Exception as e:
            logger.error(f"Failed to generate migration report: {e}")
            return {'error': str(e)}

def main():
    """Main migration manager CLI"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Database Migration Manager')
    parser.add_argument('--database-url', required=True, help='Database connection URL')
    parser.add_argument('--migrations-path', default='migrations', help='Path to migrations directory')
    parser.add_argument('--action', choices=['init', 'migrate', 'status', 'validate', 'report'], 
                       default='migrate', help='Action to perform')
    parser.add_argument('--version', help='Set database version')
    
    args = parser.parse_args()
    
    manager = DatabaseMigrationManager(args.database_url, args.migrations_path)
    
    try:
        manager.connect()
        
        if args.action == 'init':
            manager.initialize_migration_system()
            logger.info("Migration system initialized")
            
        elif args.action == 'migrate':
            manager.initialize_migration_system()
            success = manager.migrate()
            if success:
                logger.info("All migrations applied successfully")
                if args.version:
                    manager.set_database_version(args.version)
            else:
                logger.error("Migration failed")
                sys.exit(1)
                
        elif args.action == 'status':
            applied = manager.get_applied_migrations()
            pending = manager.get_pending_migrations()
            current_version = manager.get_current_version()
            
            print(f"Current version: {current_version or 'Unknown'}")
            print(f"Applied migrations: {len(applied)}")
            print(f"Pending migrations: {len(pending)}")
            
            if pending:
                print("\nPending migrations:")
                for migration in pending:
                    print(f"  - {migration['version']}: {migration['name']}")
                    
        elif args.action == 'validate':
            valid = manager.validate_schema_integrity()
            if valid:
                logger.info("Schema validation passed")
            else:
                logger.error("Schema validation failed")
                sys.exit(1)
                
        elif args.action == 'report':
            report = manager.generate_migration_report()
            print(json.dumps(report, indent=2, default=str))
            
    finally:
        manager.disconnect()

if __name__ == '__main__':
    main()
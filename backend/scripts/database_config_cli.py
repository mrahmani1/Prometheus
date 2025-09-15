#!/usr/bin/env python3
"""
Database Configuration Management CLI Tool

Provides command-line interface for managing database configurations,
including validation, testing, versioning, and deployment operations.
"""

import asyncio
import argparse
import json
import yaml
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from datetime import datetime

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.database_config_manager import (
    DatabaseConfigManager, 
    DatabaseType, 
    Environment,
    DatabaseConfig
)


class DatabaseConfigCLI:
    """Command-line interface for database configuration management"""
    
    def __init__(self):
        self.config_manager: Optional[DatabaseConfigManager] = None
        self.logger = logging.getLogger(__name__)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    async def initialize_manager(self, environment: str, config_dir: Optional[str] = None):
        """Initialize the database configuration manager"""
        env = Environment(environment)
        self.config_manager = DatabaseConfigManager(
            config_dir=config_dir,
            environment=env
        )
        
        success = await self.config_manager.initialize()
        if not success:
            print(f"❌ Failed to initialize configuration manager")
            sys.exit(1)
        
        print(f"✅ Configuration manager initialized for {environment} environment")
    
    async def list_configurations(self):
        """List all database configurations"""
        if not self.config_manager:
            print("❌ Configuration manager not initialized")
            return
        
        configs = self.config_manager.configs
        if not configs:
            print("📋 No database configurations found")
            return
        
        print(f"📋 Database Configurations ({len(configs)} total):")
        print("-" * 80)
        
        for name, config in configs.items():
            status_icon = "🟢" if config else "🔴"
            print(f"{status_icon} {name}")
            print(f"   Type: {config.type.value}")
            print(f"   Host: {config.host}:{config.port}")
            print(f"   Database: {config.database or 'N/A'}")
            print(f"   Version: {config.version}")
            print(f"   SSL: {'Enabled' if config.ssl_enabled else 'Disabled'}")
            print(f"   Updated: {config.updated_at.strftime('%Y-%m-%d %H:%M:%S')}")
            print()
    
    async def create_configuration(self, 
                                 name: str,
                                 db_type: str,
                                 host: str,
                                 port: int,
                                 database: Optional[str] = None,
                                 **kwargs):
        """Create a new database configuration"""
        if not self.config_manager:
            print("❌ Configuration manager not initialized")
            return
        
        try:
            database_type = DatabaseType(db_type)
            
            config = await self.config_manager.create_database_config(
                name=name,
                db_type=database_type,
                host=host,
                port=port,
                database=database,
                **kwargs
            )
            
            print(f"✅ Created database configuration: {name}")
            print(f"   Type: {config.type.value}")
            print(f"   Connection: {config.get_connection_string(include_credentials=False)}")
            print(f"   Version: {config.version}")
            
        except Exception as e:
            print(f"❌ Failed to create configuration: {e}")
    
    async def validate_configurations(self, name: Optional[str] = None):
        """Validate database configurations"""
        if not self.config_manager:
            print("❌ Configuration manager not initialized")
            return
        
        if name:
            # Validate specific configuration
            if name not in self.config_manager.configs:
                print(f"❌ Configuration '{name}' not found")
                return
            
            config = self.config_manager.configs[name]
            is_valid, errors = config.validate()
            
            if is_valid:
                print(f"✅ Configuration '{name}' is valid")
            else:
                print(f"❌ Configuration '{name}' validation failed:")
                for error in errors:
                    print(f"   • {error}")
        else:
            # Validate all configurations
            results = await self.config_manager.validate_all_configurations()
            
            valid_count = sum(1 for is_valid, _ in results.values() if is_valid)
            total_count = len(results)
            
            print(f"🔍 Validation Results: {valid_count}/{total_count} configurations valid")
            print("-" * 60)
            
            for config_name, (is_valid, errors) in results.items():
                status_icon = "✅" if is_valid else "❌"
                print(f"{status_icon} {config_name}")
                
                if not is_valid:
                    for error in errors:
                        print(f"   • {error}")
                print()
    
    async def test_connections(self, name: Optional[str] = None):
        """Test database connections"""
        if not self.config_manager:
            print("❌ Configuration manager not initialized")
            return
        
        if name:
            # Test specific configuration
            if name not in self.config_manager.configs:
                print(f"❌ Configuration '{name}' not found")
                return
            
            print(f"🔌 Testing connection for '{name}'...")
            success, error = await self.config_manager._test_config_connection(
                self.config_manager.configs[name]
            )
            
            if success:
                print(f"✅ Connection successful for '{name}'")
            else:
                print(f"❌ Connection failed for '{name}': {error}")
        else:
            # Test all connections
            results = await self.config_manager.test_all_connections()
            
            success_count = sum(1 for success, _ in results.values() if success)
            total_count = len(results)
            
            print(f"🔌 Connection Test Results: {success_count}/{total_count} successful")
            print("-" * 60)
            
            for config_name, (success, error) in results.items():
                status_icon = "✅" if success else "❌"
                print(f"{status_icon} {config_name}")
                
                if not success and error:
                    print(f"   Error: {error}")
                print()
    
    async def show_configuration(self, name: str, format: str = 'yaml'):
        """Show detailed configuration"""
        if not self.config_manager:
            print("❌ Configuration manager not initialized")
            return
        
        if name not in self.config_manager.configs:
            print(f"❌ Configuration '{name}' not found")
            return
        
        config_str = await self.config_manager.export_configuration(name, format)
        if config_str:
            print(f"📄 Configuration '{name}' ({format.upper()}):")
            print("-" * 60)
            print(config_str)
        else:
            print(f"❌ Failed to export configuration '{name}'")
    
    async def show_history(self, name: str):
        """Show configuration version history"""
        if not self.config_manager:
            print("❌ Configuration manager not initialized")
            return
        
        history = self.config_manager.get_config_history(name)
        if not history:
            print(f"📋 No version history found for '{name}'")
            return
        
        print(f"📋 Version History for '{name}' ({len(history)} versions):")
        print("-" * 80)
        
        for i, config in enumerate(reversed(history)):
            print(f"Version {config.version} (Created: {config.created_at.strftime('%Y-%m-%d %H:%M:%S')})")
            print(f"  Host: {config.host}:{config.port}")
            print(f"  Database: {config.database or 'N/A'}")
            print(f"  SSL: {'Enabled' if config.ssl_enabled else 'Disabled'}")
            if i < len(history) - 1:
                print()
    
    async def rollback_configuration(self, name: str, version: Optional[str] = None):
        """Rollback configuration to previous or specific version"""
        if not self.config_manager:
            print("❌ Configuration manager not initialized")
            return
        
        if name not in self.config_manager.configs:
            print(f"❌ Configuration '{name}' not found")
            return
        
        print(f"🔄 Rolling back configuration '{name}'...")
        
        if version:
            print(f"   Target version: {version}")
        else:
            print("   Target: Previous version")
        
        result = await self.config_manager.rollback_config(name, version)
        
        if result:
            print(f"✅ Successfully rolled back '{name}' to version {result.version}")
        else:
            print(f"❌ Failed to rollback configuration '{name}'")
    
    async def compare_versions(self, name: str, version1: str, version2: str):
        """Compare two configuration versions"""
        if not self.config_manager:
            print("❌ Configuration manager not initialized")
            return
        
        diff = await self.config_manager.get_configuration_diff(name, version1, version2)
        
        if not diff:
            print(f"❌ Could not compare versions {version1} and {version2} for '{name}'")
            return
        
        print(f"🔍 Configuration Diff for '{name}':")
        print(f"   Version {version1} → Version {version2}")
        print("-" * 60)
        
        if not diff['changes']:
            print("   No differences found")
            return
        
        for key, change in diff['changes'].items():
            print(f"   {key}:")
            print(f"     From: {change['from']}")
            print(f"     To:   {change['to']}")
            print()
    
    async def export_configuration(self, name: str, output_file: str, format: str = 'yaml'):
        """Export configuration to file"""
        if not self.config_manager:
            print("❌ Configuration manager not initialized")
            return
        
        config_str = await self.config_manager.export_configuration(name, format)
        if not config_str:
            print(f"❌ Failed to export configuration '{name}'")
            return
        
        try:
            with open(output_file, 'w') as f:
                f.write(config_str)
            
            print(f"✅ Exported configuration '{name}' to {output_file}")
            
        except Exception as e:
            print(f"❌ Failed to write to file {output_file}: {e}")
    
    async def import_configuration(self, name: str, input_file: str, format: str = 'yaml'):
        """Import configuration from file"""
        if not self.config_manager:
            print("❌ Configuration manager not initialized")
            return
        
        try:
            with open(input_file, 'r') as f:
                config_data = f.read()
            
            success = await self.config_manager.import_configuration(name, config_data, format)
            
            if success:
                print(f"✅ Successfully imported configuration '{name}' from {input_file}")
            else:
                print(f"❌ Failed to import configuration '{name}' from {input_file}")
                
        except Exception as e:
            print(f"❌ Failed to read file {input_file}: {e}")
    
    async def cleanup_versions(self, name: str, keep: int = 10):
        """Clean up old configuration versions"""
        if not self.config_manager:
            print("❌ Configuration manager not initialized")
            return
        
        removed_count = await self.config_manager.cleanup_old_versions(name, keep)
        
        if removed_count > 0:
            print(f"✅ Cleaned up {removed_count} old versions for '{name}' (keeping {keep} most recent)")
        else:
            print(f"📋 No cleanup needed for '{name}' (has {keep} or fewer versions)")


async def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Database Configuration Management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all configurations
  python database_config_cli.py --env production list

  # Create new configuration
  python database_config_cli.py --env development create \\
    --name geological_data --type postgresql --host localhost --port 5432 --database geo_db

  # Validate configurations
  python database_config_cli.py --env production validate

  # Test connections
  python database_config_cli.py --env production test

  # Show configuration
  python database_config_cli.py --env production show --name geological_data

  # Rollback configuration
  python database_config_cli.py --env production rollback --name geological_data --version 1.2.3

  # Export configuration
  python database_config_cli.py --env production export --name geological_data --output config.yaml
        """
    )
    
    parser.add_argument('--env', '--environment', 
                       choices=['development', 'testing', 'staging', 'production'],
                       default='development',
                       help='Target environment')
    
    parser.add_argument('--config-dir',
                       help='Configuration directory path')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List command
    subparsers.add_parser('list', help='List all database configurations')
    
    # Create command
    create_parser = subparsers.add_parser('create', help='Create new database configuration')
    create_parser.add_argument('--name', required=True, help='Configuration name')
    create_parser.add_argument('--type', required=True, 
                              choices=[t.value for t in DatabaseType],
                              help='Database type')
    create_parser.add_argument('--host', required=True, help='Database host')
    create_parser.add_argument('--port', type=int, required=True, help='Database port')
    create_parser.add_argument('--database', help='Database name')
    create_parser.add_argument('--ssl', action='store_true', help='Enable SSL')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate configurations')
    validate_parser.add_argument('--name', help='Specific configuration to validate')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test database connections')
    test_parser.add_argument('--name', help='Specific configuration to test')
    
    # Show command
    show_parser = subparsers.add_parser('show', help='Show configuration details')
    show_parser.add_argument('--name', required=True, help='Configuration name')
    show_parser.add_argument('--format', choices=['yaml', 'json'], default='yaml',
                            help='Output format')
    
    # History command
    history_parser = subparsers.add_parser('history', help='Show configuration version history')
    history_parser.add_argument('--name', required=True, help='Configuration name')
    
    # Rollback command
    rollback_parser = subparsers.add_parser('rollback', help='Rollback configuration')
    rollback_parser.add_argument('--name', required=True, help='Configuration name')
    rollback_parser.add_argument('--version', help='Target version (default: previous)')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare configuration versions')
    compare_parser.add_argument('--name', required=True, help='Configuration name')
    compare_parser.add_argument('--version1', required=True, help='First version')
    compare_parser.add_argument('--version2', required=True, help='Second version')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export configuration to file')
    export_parser.add_argument('--name', required=True, help='Configuration name')
    export_parser.add_argument('--output', required=True, help='Output file path')
    export_parser.add_argument('--format', choices=['yaml', 'json'], default='yaml',
                              help='Output format')
    
    # Import command
    import_parser = subparsers.add_parser('import', help='Import configuration from file')
    import_parser.add_argument('--name', required=True, help='Configuration name')
    import_parser.add_argument('--input', required=True, help='Input file path')
    import_parser.add_argument('--format', choices=['yaml', 'json'], default='yaml',
                              help='Input format')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up old configuration versions')
    cleanup_parser.add_argument('--name', required=True, help='Configuration name')
    cleanup_parser.add_argument('--keep', type=int, default=10, 
                               help='Number of versions to keep (default: 10)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize CLI
    cli = DatabaseConfigCLI()
    await cli.initialize_manager(args.env, args.config_dir)
    
    # Execute command
    try:
        if args.command == 'list':
            await cli.list_configurations()
        
        elif args.command == 'create':
            await cli.create_configuration(
                name=args.name,
                db_type=args.type,
                host=args.host,
                port=args.port,
                database=args.database,
                ssl_enabled=args.ssl
            )
        
        elif args.command == 'validate':
            await cli.validate_configurations(args.name)
        
        elif args.command == 'test':
            await cli.test_connections(args.name)
        
        elif args.command == 'show':
            await cli.show_configuration(args.name, args.format)
        
        elif args.command == 'history':
            await cli.show_history(args.name)
        
        elif args.command == 'rollback':
            await cli.rollback_configuration(args.name, args.version)
        
        elif args.command == 'compare':
            await cli.compare_versions(args.name, args.version1, args.version2)
        
        elif args.command == 'export':
            await cli.export_configuration(args.name, args.output, args.format)
        
        elif args.command == 'import':
            await cli.import_configuration(args.name, args.input, args.format)
        
        elif args.command == 'cleanup':
            await cli.cleanup_versions(args.name, args.keep)
        
    except KeyboardInterrupt:
        print("\n🛑 Operation cancelled by user")
    except Exception as e:
        print(f"❌ Command failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())
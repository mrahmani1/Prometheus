#!/usr/bin/env python3
"""
Script to Replace Hardcoded Database Configurations
Systematically replaces hardcoded database connection strings with environment variables
and standardized configuration management across the entire codebase.
"""

import os
import re
import sys
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import argparse


@dataclass
class HardcodedConfig:
    """Represents a hardcoded database configuration found in code"""
    file_path: str
    line_number: int
    line_content: str
    config_type: str
    original_value: str
    suggested_replacement: str
    context: str


class DatabaseConfigReplacer:
    """
    Replaces hardcoded database configurations with environment variables
    and standardized configuration management
    """
    
    def __init__(self, root_dir: str = ".", dry_run: bool = False):
        self.root_dir = Path(root_dir)
        self.dry_run = dry_run
        self.logger = logging.getLogger(__name__)
        
        # Patterns to detect hardcoded configurations
        self.hardcoded_patterns = {
            'localhost_postgres': r'localhost:5432',
            'localhost_redis': r'localhost:6379',
            'localhost_influxdb': r'localhost:8086',
            'localhost_neo4j': r'localhost:7687|localhost:7474',
            'localhost_mongodb': r'localhost:27017',
            'localhost_elasticsearch': r'localhost:9200',
            'localhost_kafka': r'localhost:9092',
            'hardcoded_connection_string': r'postgresql://[^/]+:[^@]+@[^/]+:\d+/\w+',
            'hardcoded_redis_url': r'redis://[^/]*:\d+',
            'hardcoded_mongo_url': r'mongodb://[^/]+:\d+',
            'hardcoded_credentials': r'(username|password|token)\s*[:=]\s*["\'][^"\']+["\']'
        }
        
        # Environment variable mappings
        self.env_var_mappings = {
            'localhost:5432': '${POSTGRES_HOST:localhost}:${POSTGRES_PORT:5432}',
            'localhost:6379': '${REDIS_HOST:localhost}:${REDIS_PORT:6379}',
            'localhost:8086': '${INFLUXDB_HOST:localhost}:${INFLUXDB_PORT:8086}',
            'localhost:7687': '${NEO4J_HOST:localhost}:${NEO4J_PORT:7687}',
            'localhost:7474': '${NEO4J_HTTP_HOST:localhost}:${NEO4J_HTTP_PORT:7474}',
            'localhost:27017': '${MONGODB_HOST:localhost}:${MONGODB_PORT:27017}',
            'localhost:9200': '${ELASTICSEARCH_HOST:localhost}:${ELASTICSEARCH_PORT:9200}',
            'localhost:9092': '${KAFKA_HOST:localhost}:${KAFKA_PORT:9092}'
        }
        
        # Files to exclude from processing
        self.exclude_patterns = {
            '*.pyc', '*.pyo', '*.pyd', '__pycache__', '.git', '.pytest_cache',
            'node_modules', '.venv', 'venv', '.env', '*.log', '*.db', '*.sqlite',
            'audit_results', '.benchmarks', 'vector_stores'
        }
        
        # Files to include for processing
        self.include_patterns = {
            '*.py', '*.js', '*.ts', '*.yaml', '*.yml', '*.json', '*.md', '*.env.example'
        }
        
        # Results tracking
        self.found_configs: List[HardcodedConfig] = []
        self.replaced_configs: List[HardcodedConfig] = []
        self.failed_replacements: List[Tuple[HardcodedConfig, str]] = []
    
    def scan_codebase(self) -> List[HardcodedConfig]:
        """Scan entire codebase for hardcoded database configurations"""
        self.logger.info(f"Scanning codebase in {self.root_dir}")
        
        for file_path in self._get_files_to_process():
            try:
                self._scan_file(file_path)
            except Exception as e:
                self.logger.warning(f"Failed to scan {file_path}: {e}")
        
        self.logger.info(f"Found {len(self.found_configs)} hardcoded configurations")
        return self.found_configs
    
    def _get_files_to_process(self) -> List[Path]:
        """Get list of files to process based on include/exclude patterns"""
        files_to_process = []
        
        for pattern in self.include_patterns:
            files_to_process.extend(self.root_dir.rglob(pattern))
        
        # Filter out excluded files
        filtered_files = []
        for file_path in files_to_process:
            should_exclude = False
            
            for exclude_pattern in self.exclude_patterns:
                if exclude_pattern in str(file_path):
                    should_exclude = True
                    break
            
            if not should_exclude and file_path.is_file():
                filtered_files.append(file_path)
        
        return filtered_files
    
    def _scan_file(self, file_path: Path):
        """Scan individual file for hardcoded configurations"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            for line_num, line in enumerate(lines, 1):
                self._scan_line(file_path, line_num, line, lines)
                
        except Exception as e:
            self.logger.warning(f"Failed to read {file_path}: {e}")
    
    def _scan_line(self, file_path: Path, line_num: int, line: str, all_lines: List[str]):
        """Scan individual line for hardcoded configurations"""
        for pattern_name, pattern in self.hardcoded_patterns.items():
            matches = re.finditer(pattern, line, re.IGNORECASE)
            
            for match in matches:
                # Skip if this looks like a comment about the pattern
                if any(comment_marker in line for comment_marker in ['#', '//', '/*', '<!--']):
                    if 'pattern' in line.lower() or 'example' in line.lower():
                        continue
                
                # Get context (3 lines before and after)
                start_line = max(0, line_num - 4)
                end_line = min(len(all_lines), line_num + 3)
                context_lines = all_lines[start_line:end_line]
                context = ''.join(context_lines).strip()
                
                # Determine suggested replacement
                original_value = match.group(0)
                suggested_replacement = self._get_suggested_replacement(
                    original_value, pattern_name, file_path
                )
                
                hardcoded_config = HardcodedConfig(
                    file_path=str(file_path),
                    line_number=line_num,
                    line_content=line.strip(),
                    config_type=pattern_name,
                    original_value=original_value,
                    suggested_replacement=suggested_replacement,
                    context=context
                )
                
                self.found_configs.append(hardcoded_config)
    
    def _get_suggested_replacement(self, original_value: str, pattern_name: str, file_path: Path) -> str:
        """Get suggested replacement for hardcoded configuration"""
        # Direct mappings
        if original_value in self.env_var_mappings:
            return self.env_var_mappings[original_value]
        
        # Pattern-based replacements
        if pattern_name == 'hardcoded_connection_string':
            return self._suggest_connection_string_replacement(original_value)
        elif pattern_name == 'hardcoded_redis_url':
            return self._suggest_redis_url_replacement(original_value)
        elif pattern_name == 'hardcoded_mongo_url':
            return self._suggest_mongo_url_replacement(original_value)
        elif pattern_name == 'hardcoded_credentials':
            return self._suggest_credential_replacement(original_value)
        
        # File-type specific replacements
        if file_path.suffix in ['.py']:
            return self._suggest_python_replacement(original_value, pattern_name)
        elif file_path.suffix in ['.yaml', '.yml']:
            return self._suggest_yaml_replacement(original_value, pattern_name)
        elif file_path.suffix in ['.js', '.ts']:
            return self._suggest_javascript_replacement(original_value, pattern_name)
        
        return f"${{{self._get_env_var_name(original_value)}}}"
    
    def _suggest_connection_string_replacement(self, connection_string: str) -> str:
        """Suggest replacement for PostgreSQL connection string"""
        # Parse connection string components
        pattern = r'postgresql://([^:]+):([^@]+)@([^:]+):(\d+)/(\w+)'
        match = re.match(pattern, connection_string)
        
        if match:
            username, password, host, port, database = match.groups()
            return (
                f"postgresql://${{POSTGRES_USERNAME:{username}}}:"
                f"${{POSTGRES_PASSWORD}}@"
                f"${{POSTGRES_HOST:{host}}}:"
                f"${{POSTGRES_PORT:{port}}}/"
                f"${{POSTGRES_DATABASE:{database}}}"
            )
        
        return "${POSTGRES_CONNECTION_STRING}"
    
    def _suggest_redis_url_replacement(self, redis_url: str) -> str:
        """Suggest replacement for Redis URL"""
        if 'localhost:6379' in redis_url:
            return redis_url.replace('localhost:6379', '${REDIS_HOST:localhost}:${REDIS_PORT:6379}')
        return "${REDIS_URL}"
    
    def _suggest_mongo_url_replacement(self, mongo_url: str) -> str:
        """Suggest replacement for MongoDB URL"""
        if 'localhost:27017' in mongo_url:
            return mongo_url.replace('localhost:27017', '${MONGODB_HOST:localhost}:${MONGODB_PORT:27017}')
        return "${MONGODB_URL}"
    
    def _suggest_credential_replacement(self, credential: str) -> str:
        """Suggest replacement for hardcoded credentials"""
        if 'username' in credential.lower():
            return 'username: ${DATABASE_USERNAME}'
        elif 'password' in credential.lower():
            return 'password: ${DATABASE_PASSWORD}'
        elif 'token' in credential.lower():
            return 'token: ${DATABASE_TOKEN}'
        return credential
    
    def _suggest_python_replacement(self, original_value: str, pattern_name: str) -> str:
        """Suggest Python-specific replacement"""
        env_var = self._get_env_var_name(original_value)
        return f"os.getenv('{env_var}', '{original_value}')"
    
    def _suggest_yaml_replacement(self, original_value: str, pattern_name: str) -> str:
        """Suggest YAML-specific replacement"""
        env_var = self._get_env_var_name(original_value)
        return f"${{{env_var}:{original_value}}}"
    
    def _suggest_javascript_replacement(self, original_value: str, pattern_name: str) -> str:
        """Suggest JavaScript-specific replacement"""
        env_var = self._get_env_var_name(original_value)
        return f"process.env.{env_var} || '{original_value}'"
    
    def _get_env_var_name(self, value: str) -> str:
        """Generate environment variable name from value"""
        if 'localhost:5432' in value:
            return 'POSTGRES_HOST'
        elif 'localhost:6379' in value:
            return 'REDIS_HOST'
        elif 'localhost:8086' in value:
            return 'INFLUXDB_HOST'
        elif 'localhost:7687' in value:
            return 'NEO4J_HOST'
        elif 'localhost:27017' in value:
            return 'MONGODB_HOST'
        elif 'localhost:9200' in value:
            return 'ELASTICSEARCH_HOST'
        elif 'localhost:9092' in value:
            return 'KAFKA_HOST'
        else:
            return 'DATABASE_HOST'
    
    def replace_configurations(self, configs: Optional[List[HardcodedConfig]] = None) -> int:
        """Replace hardcoded configurations with environment variables"""
        if configs is None:
            configs = self.found_configs
        
        self.logger.info(f"Replacing {len(configs)} hardcoded configurations")
        
        if self.dry_run:
            self.logger.info("DRY RUN MODE - No files will be modified")
            for config in configs:
                self.logger.info(f"Would replace in {config.file_path}:{config.line_number}")
                self.logger.info(f"  Original: {config.original_value}")
                self.logger.info(f"  Replacement: {config.suggested_replacement}")
            return len(configs)
        
        # Group configs by file for efficient processing
        configs_by_file = {}
        for config in configs:
            if config.file_path not in configs_by_file:
                configs_by_file[config.file_path] = []
            configs_by_file[config.file_path].append(config)
        
        replaced_count = 0
        
        for file_path, file_configs in configs_by_file.items():
            try:
                replaced_count += self._replace_in_file(file_path, file_configs)
            except Exception as e:
                self.logger.error(f"Failed to replace configurations in {file_path}: {e}")
                for config in file_configs:
                    self.failed_replacements.append((config, str(e)))
        
        self.logger.info(f"Successfully replaced {replaced_count} configurations")
        return replaced_count
    
    def _replace_in_file(self, file_path: str, configs: List[HardcodedConfig]) -> int:
        """Replace configurations in a single file"""
        # Read file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        replaced_count = 0
        
        # Sort configs by line number in reverse order to avoid line number shifts
        configs.sort(key=lambda x: x.line_number, reverse=True)
        
        for config in configs:
            # Replace the specific occurrence
            old_value = config.original_value
            new_value = config.suggested_replacement
            
            # Use more precise replacement to avoid unintended changes
            if old_value in content:
                content = content.replace(old_value, new_value, 1)
                replaced_count += 1
                self.replaced_configs.append(config)
                
                self.logger.debug(f"Replaced '{old_value}' with '{new_value}' in {file_path}")
        
        # Write back to file if changes were made
        if content != original_content:
            # Create backup
            backup_path = f"{file_path}.backup"
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(original_content)
            
            # Write updated content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            self.logger.info(f"Updated {file_path} (backup: {backup_path})")
        
        return replaced_count
    
    def generate_environment_template(self) -> str:
        """Generate environment variable template based on found configurations"""
        env_vars = set()
        
        for config in self.found_configs:
            # Extract environment variables from suggested replacements
            env_var_pattern = r'\$\{([^}:]+)(?::[^}]*)?\}'
            matches = re.findall(env_var_pattern, config.suggested_replacement)
            env_vars.update(matches)
            
            # Also check for os.getenv patterns
            getenv_pattern = r"os\.getenv\('([^']+)'"
            matches = re.findall(getenv_pattern, config.suggested_replacement)
            env_vars.update(matches)
        
        # Generate template
        template_lines = [
            "# Database Configuration Environment Variables",
            "# Generated by database configuration replacement script",
            f"# Generated on: {os.popen('date').read().strip()}",
            "",
            "# Environment",
            "ENVIRONMENT=development",
            "",
            "# Database Hosts",
        ]
        
        # Group by database type
        db_vars = {
            'POSTGRES': [],
            'REDIS': [],
            'INFLUXDB': [],
            'NEO4J': [],
            'MONGODB': [],
            'ELASTICSEARCH': [],
            'KAFKA': [],
            'OTHER': []
        }
        
        for var in sorted(env_vars):
            categorized = False
            for db_type in db_vars.keys():
                if db_type in var:
                    db_vars[db_type].append(var)
                    categorized = True
                    break
            
            if not categorized:
                db_vars['OTHER'].append(var)
        
        # Add variables by category
        for db_type, vars_list in db_vars.items():
            if vars_list:
                template_lines.append(f"")
                template_lines.append(f"# {db_type} Configuration")
                
                for var in vars_list:
                    default_value = self._get_default_value(var)
                    template_lines.append(f"{var}={default_value}")
        
        return '\n'.join(template_lines)
    
    def _get_default_value(self, env_var: str) -> str:
        """Get default value for environment variable"""
        defaults = {
            'POSTGRES_HOST': 'localhost',
            'POSTGRES_PORT': '5432',
            'POSTGRES_USERNAME': 'prometheus',
            'POSTGRES_PASSWORD': 'change_me',
            'POSTGRES_DATABASE': 'prometheus',
            'REDIS_HOST': 'localhost',
            'REDIS_PORT': '6379',
            'REDIS_PASSWORD': '',
            'INFLUXDB_HOST': 'localhost',
            'INFLUXDB_PORT': '8086',
            'INFLUXDB_TOKEN': 'change_me',
            'INFLUXDB_ORG': 'prometheus',
            'NEO4J_HOST': 'localhost',
            'NEO4J_PORT': '7687',
            'NEO4J_USERNAME': 'neo4j',
            'NEO4J_PASSWORD': 'change_me',
            'MONGODB_HOST': 'localhost',
            'MONGODB_PORT': '27017',
            'ELASTICSEARCH_HOST': 'localhost',
            'ELASTICSEARCH_PORT': '9200',
            'KAFKA_HOST': 'localhost',
            'KAFKA_PORT': '9092'
        }
        
        return defaults.get(env_var, 'change_me')
    
    def generate_report(self) -> Dict:
        """Generate comprehensive report of the replacement process"""
        return {
            'summary': {
                'total_files_scanned': len(set(config.file_path for config in self.found_configs)),
                'total_hardcoded_configs_found': len(self.found_configs),
                'total_configs_replaced': len(self.replaced_configs),
                'total_failed_replacements': len(self.failed_replacements)
            },
            'found_configurations': [
                {
                    'file_path': config.file_path,
                    'line_number': config.line_number,
                    'config_type': config.config_type,
                    'original_value': config.original_value,
                    'suggested_replacement': config.suggested_replacement
                }
                for config in self.found_configs
            ],
            'replaced_configurations': [
                {
                    'file_path': config.file_path,
                    'line_number': config.line_number,
                    'original_value': config.original_value,
                    'replacement': config.suggested_replacement
                }
                for config in self.replaced_configs
            ],
            'failed_replacements': [
                {
                    'file_path': config.file_path,
                    'line_number': config.line_number,
                    'original_value': config.original_value,
                    'error': error
                }
                for config, error in self.failed_replacements
            ]
        }


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Replace hardcoded database configurations with environment variables"
    )
    parser.add_argument(
        '--root-dir', 
        default='.', 
        help='Root directory to scan (default: current directory)'
    )
    parser.add_argument(
        '--dry-run', 
        action='store_true', 
        help='Show what would be replaced without making changes'
    )
    parser.add_argument(
        '--output-report', 
        help='Output file for replacement report (JSON format)'
    )
    parser.add_argument(
        '--output-env-template', 
        help='Output file for environment variable template'
    )
    parser.add_argument(
        '--verbose', 
        action='store_true', 
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Initialize replacer
    replacer = DatabaseConfigReplacer(
        root_dir=args.root_dir,
        dry_run=args.dry_run
    )
    
    try:
        # Scan codebase
        found_configs = replacer.scan_codebase()
        
        if not found_configs:
            print("No hardcoded database configurations found!")
            return 0
        
        print(f"Found {len(found_configs)} hardcoded database configurations")
        
        # Replace configurations
        replaced_count = replacer.replace_configurations()
        
        # Generate environment template
        if args.output_env_template:
            env_template = replacer.generate_environment_template()
            with open(args.output_env_template, 'w') as f:
                f.write(env_template)
            print(f"Environment template written to {args.output_env_template}")
        
        # Generate report
        report = replacer.generate_report()
        
        if args.output_report:
            with open(args.output_report, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Replacement report written to {args.output_report}")
        
        # Print summary
        print(f"\nSummary:")
        print(f"  Files scanned: {report['summary']['total_files_scanned']}")
        print(f"  Hardcoded configs found: {report['summary']['total_hardcoded_configs_found']}")
        print(f"  Configs replaced: {report['summary']['total_configs_replaced']}")
        print(f"  Failed replacements: {report['summary']['total_failed_replacements']}")
        
        if report['summary']['total_failed_replacements'] > 0:
            print(f"\nFailed replacements:")
            for failure in report['failed_replacements']:
                print(f"  {failure['file_path']}:{failure['line_number']} - {failure['error']}")
        
        return 0 if report['summary']['total_failed_replacements'] == 0 else 1
        
    except Exception as e:
        logging.error(f"Script failed: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
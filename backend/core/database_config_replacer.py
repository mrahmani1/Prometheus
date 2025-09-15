"""
Database Configuration Replacement Tool
Systematically replaces hardcoded database configurations with environment variables
and standardizes database configuration management across the entire codebase.
"""

import os
import re
import ast
import logging
from typing import Dict, List, Set, Tuple, Optional, Any
from pathlib import Path
from dataclasses import dataclass, field
import yaml
import json
from datetime import datetime

from backend.core.database_config_manager import (
    DatabaseConfigManager, DatabaseType, Environment, DatabaseConfig
)


@dataclass
class HardcodedConfig:
    """Represents a hardcoded database configuration found in code"""
    file_path: str
    line_number: int
    line_content: str
    config_type: str  # 'connection_string', 'host_port', 'url'
    database_type: Optional[DatabaseType] = None
    host: Optional[str] = None
    port: Optional[int] = None
    database: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    confidence: float = 0.0  # Confidence in detection (0.0 to 1.0)
    suggested_env_var: Optional[str] = None
    replacement_code: Optional[str] = None


@dataclass
class ReplacementResult:
    """Result of configuration replacement operation"""
    file_path: str
    original_content: str
    modified_content: str
    replacements_made: int
    hardcoded_configs: List[HardcodedConfig]
    success: bool
    error_message: Optional[str] = None


class DatabaseConfigReplacer:
    """
    Comprehensive tool for replacing hardcoded database configurations
    with environment variables and standardized configuration management.
    """
    
    def __init__(self, 
                 root_dir: str = ".",
                 config_manager: Optional[DatabaseConfigManager] = None,
                 dry_run: bool = False):
        self.root_dir = Path(root_dir)
        self.config_manager = config_manager
        self.dry_run = dry_run
        self.logger = logging.getLogger(__name__ + ".DatabaseConfigReplacer")
        
        # Patterns for detecting hardcoded configurations
        self.connection_patterns = self._initialize_patterns()
        
        # Environment variable mapping
        self.env_var_mapping = self._initialize_env_mapping()
        
        # Replacement statistics
        self.stats = {
            'files_scanned': 0,
            'files_modified': 0,
            'configs_found': 0,
            'configs_replaced': 0,
            'errors': 0
        }
    
    def _initialize_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Initialize regex patterns for detecting hardcoded configurations"""
        return {
            'postgresql_urls': [
                re.compile(r'postgresql://[^"\s]+', re.IGNORECASE),
                re.compile(r'postgres://[^"\s]+', re.IGNORECASE),
                re.compile(r'"postgresql://[^"]*"', re.IGNORECASE),
                re.compile(r"'postgresql://[^']*'", re.IGNORECASE),
            ],
            'redis_urls': [
                re.compile(r'redis://[^"\s]+', re.IGNORECASE),
                re.compile(r'rediss://[^"\s]+', re.IGNORECASE),
                re.compile(r'"redis://[^"]*"', re.IGNORECASE),
                re.compile(r"'redis://[^']*'", re.IGNORECASE),
            ],
            'influxdb_urls': [
                re.compile(r'http://[^"\s]*:8086[^"\s]*', re.IGNORECASE),
                re.compile(r'https://[^"\s]*:8086[^"\s]*', re.IGNORECASE),
                re.compile(r'"https?://[^"]*:8086[^"]*"', re.IGNORECASE),
                re.compile(r"'https?://[^']*:8086[^']*'", re.IGNORECASE),
            ],
            'neo4j_urls': [
                re.compile(r'bolt://[^"\s]+', re.IGNORECASE),
                re.compile(r'bolt\+s://[^"\s]+', re.IGNORECASE),
                re.compile(r'neo4j://[^"\s]+', re.IGNORECASE),
                re.compile(r'"bolt[^"]*"', re.IGNORECASE),
                re.compile(r"'bolt[^']*'", re.IGNORECASE),
            ],
            'localhost_patterns': [
                re.compile(r'localhost:5432', re.IGNORECASE),
                re.compile(r'localhost:6379', re.IGNORECASE),
                re.compile(r'localhost:8086', re.IGNORECASE),
                re.compile(r'localhost:7474', re.IGNORECASE),
                re.compile(r'localhost:7687', re.IGNORECASE),
                re.compile(r'localhost:9042', re.IGNORECASE),
                re.compile(r'127\.0\.0\.1:5432', re.IGNORECASE),
                re.compile(r'127\.0\.0\.1:6379', re.IGNORECASE),
                re.compile(r'127\.0\.0\.1:8086', re.IGNORECASE),
            ],
            'host_port_configs': [
                re.compile(r"['\"]host['\"]:\s*['\"]localhost['\"]", re.IGNORECASE),
                re.compile(r"['\"]port['\"]:\s*5432", re.IGNORECASE),
                re.compile(r"['\"]port['\"]:\s*6379", re.IGNORECASE),
                re.compile(r"['\"]port['\"]:\s*8086", re.IGNORECASE),
                re.compile(r"host\s*=\s*['\"]localhost['\"]", re.IGNORECASE),
                re.compile(r"port\s*=\s*5432", re.IGNORECASE),
                re.compile(r"port\s*=\s*6379", re.IGNORECASE),
            ],
            'database_names': [
                re.compile(r"['\"]database['\"]:\s*['\"]prometheus[^'\"]*['\"]", re.IGNORECASE),
                re.compile(r"database\s*=\s*['\"]prometheus[^'\"]*['\"]", re.IGNORECASE),
                re.compile(r"db\s*=\s*['\"]prometheus[^'\"]*['\"]", re.IGNORECASE),
            ]
        }
    
    def _initialize_env_mapping(self) -> Dict[str, Dict[str, str]]:
        """Initialize environment variable mapping for different database types"""
        return {
            'postgresql': {
                'host': 'POSTGRES_HOST',
                'port': 'POSTGRES_PORT',
                'database': 'POSTGRES_DB',
                'username': 'POSTGRES_USER',
                'password': 'POSTGRES_PASSWORD',
                'url': 'POSTGRES_URL'
            },
            'redis': {
                'host': 'REDIS_HOST',
                'port': 'REDIS_PORT',
                'password': 'REDIS_PASSWORD',
                'url': 'REDIS_URL'
            },
            'influxdb': {
                'host': 'INFLUXDB_HOST',
                'port': 'INFLUXDB_PORT',
                'token': 'INFLUXDB_TOKEN',
                'org': 'INFLUXDB_ORG',
                'bucket': 'INFLUXDB_BUCKET',
                'url': 'INFLUXDB_URL'
            },
            'neo4j': {
                'host': 'NEO4J_HOST',
                'port': 'NEO4J_PORT',
                'username': 'NEO4J_USER',
                'password': 'NEO4J_PASSWORD',
                'url': 'NEO4J_URL'
            },
            'timescaledb': {
                'host': 'TIMESCALEDB_HOST',
                'port': 'TIMESCALEDB_PORT',
                'database': 'TIMESCALEDB_DB',
                'username': 'TIMESCALEDB_USER',
                'password': 'TIMESCALEDB_PASSWORD',
                'url': 'TIMESCALEDB_URL'
            }
        }
    
    async def scan_codebase(self, 
                           file_patterns: List[str] = None,
                           exclude_patterns: List[str] = None) -> List[HardcodedConfig]:
        """Scan entire codebase for hardcoded database configurations"""
        if file_patterns is None:
            file_patterns = ['*.py', '*.yaml', '*.yml', '*.json', '*.env']
        
        if exclude_patterns is None:
            exclude_patterns = [
                '**/node_modules/**',
                '**/__pycache__/**',
                '**/venv/**',
                '**/env/**',
                '**/.git/**',
                '**/test_*',
                '**/tests/**'
            ]
        
        hardcoded_configs = []
        
        for pattern in file_patterns:
            for file_path in self.root_dir.rglob(pattern):
                # Skip excluded patterns
                if any(file_path.match(exclude) for exclude in exclude_patterns):
                    continue
                
                try:
                    configs = await self._scan_file(file_path)
                    hardcoded_configs.extend(configs)
                    self.stats['files_scanned'] += 1
                    
                except Exception as e:
                    self.logger.error(f"Error scanning file {file_path}: {e}")
                    self.stats['errors'] += 1
        
        self.stats['configs_found'] = len(hardcoded_configs)
        self.logger.info(f"Scan complete: {len(hardcoded_configs)} hardcoded configs found in {self.stats['files_scanned']} files")
        
        return hardcoded_configs
    
    async def _scan_file(self, file_path: Path) -> List[HardcodedConfig]:
        """Scan individual file for hardcoded configurations"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            # Skip binary files
            return []
        
        configs = []
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            line_configs = self._analyze_line(str(file_path), line_num, line)
            configs.extend(line_configs)
        
        return configs
    
    def _analyze_line(self, file_path: str, line_number: int, line: str) -> List[HardcodedConfig]:
        """Analyze a single line for hardcoded configurations"""
        configs = []
        
        # Skip comments and empty lines
        stripped_line = line.strip()
        if not stripped_line or stripped_line.startswith('#') or stripped_line.startswith('//'):
            return configs
        
        # Check each pattern category
        for category, patterns in self.connection_patterns.items():
            for pattern in patterns:
                matches = pattern.finditer(line)
                for match in matches:
                    config = self._create_hardcoded_config(
                        file_path, line_number, line, match, category
                    )
                    if config:
                        configs.append(config)
        
        return configs
    
    def _create_hardcoded_config(self, 
                                file_path: str, 
                                line_number: int, 
                                line: str, 
                                match: re.Match, 
                                category: str) -> Optional[HardcodedConfig]:
        """Create HardcodedConfig object from regex match"""
        matched_text = match.group(0)
        
        # Parse the matched configuration
        config = HardcodedConfig(
            file_path=file_path,
            line_number=line_number,
            line_content=line.strip(),
            config_type=category
        )
        
        # Extract database type and connection details
        if 'postgresql' in category or 'postgres' in matched_text.lower():
            config.database_type = DatabaseType.POSTGRESQL
            config = self._parse_postgresql_config(config, matched_text)
        elif 'redis' in category or 'redis' in matched_text.lower():
            config.database_type = DatabaseType.REDIS
            config = self._parse_redis_config(config, matched_text)
        elif 'influxdb' in category or ':8086' in matched_text:
            config.database_type = DatabaseType.INFLUXDB
            config = self._parse_influxdb_config(config, matched_text)
        elif 'neo4j' in category or 'bolt' in matched_text.lower():
            config.database_type = DatabaseType.NEO4J
            config = self._parse_neo4j_config(config, matched_text)
        elif 'localhost' in category:
            config = self._parse_localhost_config(config, matched_text)
        
        # Generate suggested environment variable and replacement
        if config.database_type:
            config.suggested_env_var = self._suggest_env_var(config)
            config.replacement_code = self._generate_replacement_code(config, matched_text)
            config.confidence = self._calculate_confidence(config, line)
        
        return config if config.confidence > 0.5 else None
    
    def _parse_postgresql_config(self, config: HardcodedConfig, matched_text: str) -> HardcodedConfig:
        """Parse PostgreSQL connection string"""
        # Extract components from postgresql://user:pass@host:port/database
        url_pattern = re.compile(r'postgresql://(?:([^:]+):([^@]+)@)?([^:/]+):?(\d+)?/?([^?]*)')
        match = url_pattern.search(matched_text)
        
        if match:
            config.username = match.group(1)
            config.password = match.group(2)
            config.host = match.group(3)
            config.port = int(match.group(4)) if match.group(4) else 5432
            config.database = match.group(5) if match.group(5) else None
        
        return config
    
    def _parse_redis_config(self, config: HardcodedConfig, matched_text: str) -> HardcodedConfig:
        """Parse Redis connection string"""
        # Extract components from redis://[:password@]host:port[/database]
        url_pattern = re.compile(r'redis://(?::([^@]+)@)?([^:/]+):?(\d+)?/?(\d+)?')
        match = url_pattern.search(matched_text)
        
        if match:
            config.password = match.group(1)
            config.host = match.group(2)
            config.port = int(match.group(3)) if match.group(3) else 6379
            config.database = match.group(4) if match.group(4) else None
        
        return config
    
    def _parse_influxdb_config(self, config: HardcodedConfig, matched_text: str) -> HardcodedConfig:
        """Parse InfluxDB connection string"""
        # Extract components from http(s)://host:port
        url_pattern = re.compile(r'https?://([^:/]+):?(\d+)?')
        match = url_pattern.search(matched_text)
        
        if match:
            config.host = match.group(1)
            config.port = int(match.group(2)) if match.group(2) else 8086
        
        return config
    
    def _parse_neo4j_config(self, config: HardcodedConfig, matched_text: str) -> HardcodedConfig:
        """Parse Neo4j connection string"""
        # Extract components from bolt://host:port
        url_pattern = re.compile(r'bolt(?:\+s)?://([^:/]+):?(\d+)?')
        match = url_pattern.search(matched_text)
        
        if match:
            config.host = match.group(1)
            config.port = int(match.group(2)) if match.group(2) else 7687
        
        return config
    
    def _parse_localhost_config(self, config: HardcodedConfig, matched_text: str) -> HardcodedConfig:
        """Parse localhost:port configurations"""
        port_match = re.search(r':(\d+)', matched_text)
        if port_match:
            port = int(port_match.group(1))
            config.host = 'localhost'
            config.port = port
            
            # Determine database type by port
            if port == 5432:
                config.database_type = DatabaseType.POSTGRESQL
            elif port == 6379:
                config.database_type = DatabaseType.REDIS
            elif port == 8086:
                config.database_type = DatabaseType.INFLUXDB
            elif port in [7474, 7687]:
                config.database_type = DatabaseType.NEO4J
        
        return config
    
    def _suggest_env_var(self, config: HardcodedConfig) -> str:
        """Suggest appropriate environment variable name"""
        if not config.database_type:
            return "DATABASE_URL"
        
        db_type = config.database_type.value.lower()
        mapping = self.env_var_mapping.get(db_type, {})
        
        # Determine the most appropriate env var based on what's being configured
        if 'url' in config.line_content.lower() or '://' in config.line_content:
            return mapping.get('url', f"{db_type.upper()}_URL")
        elif 'host' in config.line_content.lower():
            return mapping.get('host', f"{db_type.upper()}_HOST")
        elif 'port' in config.line_content.lower():
            return mapping.get('port', f"{db_type.upper()}_PORT")
        else:
            return mapping.get('url', f"{db_type.upper()}_URL")
    
    def _generate_replacement_code(self, config: HardcodedConfig, original_text: str) -> str:
        """Generate replacement code using environment variables"""
        env_var = config.suggested_env_var
        
        # Determine default value based on original configuration
        if config.host and config.port:
            if config.database_type == DatabaseType.POSTGRESQL:
                default = f"postgresql://{config.host}:{config.port}"
                if config.database:
                    default += f"/{config.database}"
            elif config.database_type == DatabaseType.REDIS:
                default = f"redis://{config.host}:{config.port}"
            elif config.database_type == DatabaseType.INFLUXDB:
                default = f"http://{config.host}:{config.port}"
            elif config.database_type == DatabaseType.NEO4J:
                default = f"bolt://{config.host}:{config.port}"
            else:
                default = original_text.strip('"\'')
        else:
            default = original_text.strip('"\'')
        
        # Generate os.getenv() call
        return f'os.getenv("{env_var}", "{default}")'
    
    def _calculate_confidence(self, config: HardcodedConfig, line: str) -> float:
        """Calculate confidence score for the detected configuration"""
        confidence = 0.5  # Base confidence
        
        # Increase confidence for clear database URLs
        if '://' in line:
            confidence += 0.3
        
        # Increase confidence for localhost
        if 'localhost' in line.lower():
            confidence += 0.2
        
        # Increase confidence for known ports
        known_ports = [5432, 6379, 8086, 7474, 7687, 9042]
        if config.port in known_ports:
            confidence += 0.2
        
        # Decrease confidence for comments or documentation
        if any(keyword in line.lower() for keyword in ['example', 'sample', 'demo', 'test']):
            confidence -= 0.3
        
        # Decrease confidence for variable assignments that might be intentional
        if re.search(r'(default|fallback|example).*=', line.lower()):
            confidence -= 0.2
        
        return min(1.0, max(0.0, confidence))
    
    async def replace_configurations(self, 
                                   hardcoded_configs: List[HardcodedConfig],
                                   backup: bool = True) -> List[ReplacementResult]:
        """Replace hardcoded configurations with environment variables"""
        results = []
        files_to_process = {}
        
        # Group configurations by file
        for config in hardcoded_configs:
            if config.file_path not in files_to_process:
                files_to_process[config.file_path] = []
            files_to_process[config.file_path].append(config)
        
        # Process each file
        for file_path, configs in files_to_process.items():
            try:
                result = await self._replace_file_configurations(file_path, configs, backup)
                results.append(result)
                
                if result.success:
                    self.stats['files_modified'] += 1
                    self.stats['configs_replaced'] += result.replacements_made
                else:
                    self.stats['errors'] += 1
                    
            except Exception as e:
                self.logger.error(f"Error processing file {file_path}: {e}")
                results.append(ReplacementResult(
                    file_path=file_path,
                    original_content="",
                    modified_content="",
                    replacements_made=0,
                    hardcoded_configs=configs,
                    success=False,
                    error_message=str(e)
                ))
                self.stats['errors'] += 1
        
        return results
    
    async def _replace_file_configurations(self, 
                                         file_path: str, 
                                         configs: List[HardcodedConfig],
                                         backup: bool) -> ReplacementResult:
        """Replace configurations in a single file"""
        try:
            # Read original content
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            modified_content = original_content
            replacements_made = 0
            
            # Sort configs by line number (descending) to avoid line number shifts
            sorted_configs = sorted(configs, key=lambda c: c.line_number, reverse=True)
            
            lines = modified_content.split('\n')
            
            for config in sorted_configs:
                if config.replacement_code and config.confidence > 0.7:
                    line_idx = config.line_number - 1
                    if 0 <= line_idx < len(lines):
                        original_line = lines[line_idx]
                        
                        # Add import for os if needed
                        if 'os.getenv' in config.replacement_code and not self._has_os_import(modified_content):
                            lines.insert(0, 'import os')
                            line_idx += 1  # Adjust for inserted import
                        
                        # Replace the hardcoded value with environment variable
                        modified_line = self._replace_in_line(original_line, config)
                        if modified_line != original_line:
                            lines[line_idx] = modified_line
                            replacements_made += 1
                            
                            self.logger.info(f"Replaced in {file_path}:{config.line_number}: {config.suggested_env_var}")
            
            modified_content = '\n'.join(lines)
            
            # Create backup if requested
            if backup and replacements_made > 0:
                backup_path = f"{file_path}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(original_content)
                self.logger.info(f"Created backup: {backup_path}")
            
            # Write modified content (unless dry run)
            if not self.dry_run and replacements_made > 0:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(modified_content)
            
            return ReplacementResult(
                file_path=file_path,
                original_content=original_content,
                modified_content=modified_content,
                replacements_made=replacements_made,
                hardcoded_configs=configs,
                success=True
            )
            
        except Exception as e:
            return ReplacementResult(
                file_path=file_path,
                original_content="",
                modified_content="",
                replacements_made=0,
                hardcoded_configs=configs,
                success=False,
                error_message=str(e)
            )
    
    def _has_os_import(self, content: str) -> bool:
        """Check if file already imports os module"""
        import_patterns = [
            re.compile(r'^import os$', re.MULTILINE),
            re.compile(r'^import os\s+', re.MULTILINE),
            re.compile(r'^from os import', re.MULTILINE),
        ]
        
        return any(pattern.search(content) for pattern in import_patterns)
    
    def _replace_in_line(self, line: str, config: HardcodedConfig) -> str:
        """Replace hardcoded value in a single line"""
        # Find the hardcoded value in the line
        for category, patterns in self.connection_patterns.items():
            for pattern in patterns:
                match = pattern.search(line)
                if match:
                    # Replace the matched text with environment variable call
                    original_value = match.group(0)
                    
                    # Preserve quotes if they exist
                    if original_value.startswith('"') and original_value.endswith('"'):
                        replacement = f'"{config.replacement_code}"'
                    elif original_value.startswith("'") and original_value.endswith("'"):
                        replacement = f"'{config.replacement_code}'"
                    else:
                        replacement = config.replacement_code
                    
                    return line.replace(original_value, replacement)
        
        return line
    
    def generate_environment_template(self, 
                                    hardcoded_configs: List[HardcodedConfig],
                                    environment: Environment = Environment.DEVELOPMENT) -> str:
        """Generate environment variable template file"""
        env_vars = {}
        
        # Collect all unique environment variables
        for config in hardcoded_configs:
            if config.suggested_env_var and config.confidence > 0.7:
                env_var = config.suggested_env_var
                
                # Determine default value
                if config.host and config.port:
                    if config.database_type == DatabaseType.POSTGRESQL:
                        default = f"postgresql://username:password@{config.host}:{config.port}"
                        if config.database:
                            default += f"/{config.database}"
                    elif config.database_type == DatabaseType.REDIS:
                        default = f"redis://:{config.host}:{config.port}"
                    elif config.database_type == DatabaseType.INFLUXDB:
                        default = f"http://{config.host}:{config.port}"
                    elif config.database_type == DatabaseType.NEO4J:
                        default = f"bolt://{config.host}:{config.port}"
                    else:
                        default = f"{config.host}:{config.port}"
                else:
                    default = "localhost"
                
                env_vars[env_var] = default
        
        # Generate template content
        template = f"""# Database Configuration Environment Variables
# Generated on {datetime.now().isoformat()}
# Environment: {environment.value}

"""
        
        # Group by database type
        db_groups = {}
        for env_var, default in env_vars.items():
            db_type = env_var.split('_')[0].lower()
            if db_type not in db_groups:
                db_groups[db_type] = []
            db_groups[db_type].append((env_var, default))
        
        for db_type, vars_list in sorted(db_groups.items()):
            template += f"# {db_type.upper()} Configuration\n"
            for env_var, default in sorted(vars_list):
                template += f"{env_var}={default}\n"
            template += "\n"
        
        return template
    
    def generate_report(self, 
                       hardcoded_configs: List[HardcodedConfig],
                       replacement_results: List[ReplacementResult] = None) -> Dict[str, Any]:
        """Generate comprehensive replacement report"""
        report = {
            'summary': {
                'scan_timestamp': datetime.now().isoformat(),
                'total_files_scanned': self.stats['files_scanned'],
                'total_configs_found': len(hardcoded_configs),
                'high_confidence_configs': len([c for c in hardcoded_configs if c.confidence > 0.7]),
                'files_modified': self.stats.get('files_modified', 0),
                'configs_replaced': self.stats.get('configs_replaced', 0),
                'errors': self.stats['errors']
            },
            'configurations_by_type': {},
            'configurations_by_file': {},
            'environment_variables': {},
            'replacement_results': []
        }
        
        # Group configurations by database type
        for config in hardcoded_configs:
            db_type = config.database_type.value if config.database_type else 'unknown'
            if db_type not in report['configurations_by_type']:
                report['configurations_by_type'][db_type] = []
            
            report['configurations_by_type'][db_type].append({
                'file_path': config.file_path,
                'line_number': config.line_number,
                'confidence': config.confidence,
                'suggested_env_var': config.suggested_env_var
            })
        
        # Group configurations by file
        for config in hardcoded_configs:
            file_path = config.file_path
            if file_path not in report['configurations_by_file']:
                report['configurations_by_file'][file_path] = []
            
            report['configurations_by_file'][file_path].append({
                'line_number': config.line_number,
                'database_type': config.database_type.value if config.database_type else 'unknown',
                'confidence': config.confidence,
                'line_content': config.line_content
            })
        
        # Collect environment variables
        for config in hardcoded_configs:
            if config.suggested_env_var and config.confidence > 0.7:
                report['environment_variables'][config.suggested_env_var] = {
                    'database_type': config.database_type.value if config.database_type else 'unknown',
                    'usage_count': len([c for c in hardcoded_configs if c.suggested_env_var == config.suggested_env_var])
                }
        
        # Add replacement results if provided
        if replacement_results:
            for result in replacement_results:
                report['replacement_results'].append({
                    'file_path': result.file_path,
                    'success': result.success,
                    'replacements_made': result.replacements_made,
                    'error_message': result.error_message
                })
        
        return report
    
    async def create_standardized_configs(self, 
                                        hardcoded_configs: List[HardcodedConfig]) -> Dict[str, DatabaseConfig]:
        """Create standardized database configurations from hardcoded configs"""
        if not self.config_manager:
            self.logger.warning("No config manager provided, cannot create standardized configs")
            return {}
        
        configs = {}
        
        # Group by database type and create unique configurations
        db_instances = {}
        for config in hardcoded_configs:
            if config.confidence < 0.7 or not config.database_type:
                continue
            
            key = f"{config.database_type.value}_{config.host}_{config.port}"
            if key not in db_instances:
                db_instances[key] = config
        
        # Create DatabaseConfig objects
        for key, config in db_instances.items():
            try:
                db_name = f"{config.database_type.value}_{config.host.replace('.', '_')}_{config.port}"
                
                db_config = await self.config_manager.create_database_config(
                    name=db_name,
                    db_type=config.database_type,
                    host=config.host or 'localhost',
                    port=config.port or self._get_default_port(config.database_type),
                    database=config.database
                )
                
                configs[db_name] = db_config
                self.logger.info(f"Created standardized config: {db_name}")
                
            except Exception as e:
                self.logger.error(f"Failed to create config for {key}: {e}")
        
        return configs
    
    def _get_default_port(self, db_type: DatabaseType) -> int:
        """Get default port for database type"""
        defaults = {
            DatabaseType.POSTGRESQL: 5432,
            DatabaseType.REDIS: 6379,
            DatabaseType.INFLUXDB: 8086,
            DatabaseType.NEO4J: 7687,
            DatabaseType.TIMESCALEDB: 5432,
            DatabaseType.MONGODB: 27017
        }
        return defaults.get(db_type, 5432)


# CLI interface for the replacement tool
async def main():
    """Main CLI interface for database configuration replacement"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Replace hardcoded database configurations")
    parser.add_argument("--root-dir", default=".", help="Root directory to scan")
    parser.add_argument("--dry-run", action="store_true", help="Perform dry run without making changes")
    parser.add_argument("--backup", action="store_true", default=True, help="Create backups of modified files")
    parser.add_argument("--output-report", help="Output report file path")
    parser.add_argument("--output-env", help="Output environment template file path")
    parser.add_argument("--confidence-threshold", type=float, default=0.7, help="Minimum confidence for replacements")
    
    args = parser.parse_args()
    
    # Initialize replacer
    replacer = DatabaseConfigReplacer(
        root_dir=args.root_dir,
        dry_run=args.dry_run
    )
    
    # Scan codebase
    print("Scanning codebase for hardcoded database configurations...")
    hardcoded_configs = await replacer.scan_codebase()
    
    # Filter by confidence threshold
    high_confidence_configs = [c for c in hardcoded_configs if c.confidence >= args.confidence_threshold]
    
    print(f"Found {len(hardcoded_configs)} total configurations, {len(high_confidence_configs)} above confidence threshold")
    
    # Replace configurations
    if high_confidence_configs:
        print("Replacing hardcoded configurations...")
        replacement_results = await replacer.replace_configurations(high_confidence_configs, backup=args.backup)
        
        successful_replacements = sum(r.replacements_made for r in replacement_results if r.success)
        print(f"Successfully replaced {successful_replacements} configurations")
    else:
        replacement_results = []
    
    # Generate report
    report = replacer.generate_report(hardcoded_configs, replacement_results)
    
    if args.output_report:
        with open(args.output_report, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to {args.output_report}")
    
    # Generate environment template
    if args.output_env:
        env_template = replacer.generate_environment_template(high_confidence_configs)
        with open(args.output_env, 'w') as f:
            f.write(env_template)
        print(f"Environment template saved to {args.output_env}")
    
    # Print summary
    print("\nSummary:")
    print(f"  Files scanned: {report['summary']['total_files_scanned']}")
    print(f"  Configurations found: {report['summary']['total_configs_found']}")
    print(f"  High confidence: {report['summary']['high_confidence_configs']}")
    print(f"  Files modified: {report['summary']['files_modified']}")
    print(f"  Configurations replaced: {report['summary']['configs_replaced']}")
    print(f"  Errors: {report['summary']['errors']}")


if __name__ == "__main__":
    asyncio.run(main())
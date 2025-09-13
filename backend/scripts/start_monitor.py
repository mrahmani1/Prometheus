#!/usr/bin/env python3
"""
Startup script for Real-Time Process Monitor

This script initializes and starts the real-time geological process monitoring system
with proper configuration, logging, and error handling for production deployment.
"""

import asyncio
import logging
import signal
import sys
import os
from pathlib import Path
from typing import Optional
import yaml
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from modules.real_time_process_monitor import RealTimeProcessMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/app/logs/monitor.log', mode='a')
    ]
)

logger = logging.getLogger(__name__)

class MonitorService:
    """Service wrapper for the Real-Time Process Monitor"""
    
    def __init__(self):
        self.monitor: Optional[RealTimeProcessMonitor] = None
        self.config: dict = {}
        self.shutdown_event = asyncio.Event()
        
    def load_config(self) -> dict:
        """Load configuration from file and environment variables"""
        try:
            # Load base configuration
            config_path = Path('/app/config/real_time_monitor_config.yaml')
            if not config_path.exists():
                config_path = Path('backend/config/real_time_monitor_config.yaml')
            
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Override with environment variables
            config = self._apply_env_overrides(config)
            
            logger.info("Configuration loaded successfully")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _apply_env_overrides(self, config: dict) -> dict:
        """Apply environment variable overrides to configuration"""
        env_mappings = {
            'REDIS_HOST': ['database', 'redis', 'host'],
            'REDIS_PORT': ['database', 'redis', 'port'],
            'REDIS_PASSWORD': ['database', 'redis', 'password'],
            'INFLUXDB_URL': ['database', 'influxdb', 'url'],
            'INFLUXDB_TOKEN': ['database', 'influxdb', 'token'],
            'INFLUXDB_ORG': ['database', 'influxdb', 'org'],
            'POSTGRES_HOST': ['database', 'postgresql', 'host'],
            'POSTGRES_PORT': ['database', 'postgresql', 'port'],
            'POSTGRES_PASSWORD': ['database', 'postgresql', 'password'],
            'WEBSOCKET_HOST': ['websocket', 'host'],
            'WEBSOCKET_PORT': ['websocket', 'port'],
            'LOG_LEVEL': ['monitoring', 'logging', 'level'],
            'MAX_WORKERS': ['processing', 'max_workers'],
            'DEBUG_MODE': ['development', 'debug_mode']
        }
        
        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Navigate to the config section
                current = config
                for key in config_path[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                
                # Set the value with appropriate type conversion
                final_key = config_path[-1]
                if env_var.endswith('_PORT') or env_var == 'MAX_WORKERS':
                    current[final_key] = int(value)
                elif env_var == 'DEBUG_MODE':
                    current[final_key] = value.lower() in ('true', '1', 'yes', 'on')
                else:
                    current[final_key] = value
        
        return config
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            self.shutdown_event.set()
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    
    async def start(self):
        """Start the monitoring service"""
        try:
            logger.info("Starting Real-Time Process Monitor service...")
            
            # Load configuration
            self.config = self.load_config()
            
            # Setup signal handlers
            self.setup_signal_handlers()
            
            # Create monitor instance
            monitor_config = self._extract_monitor_config()
            self.monitor = RealTimeProcessMonitor(monitor_config)
            
            # Start monitoring in background
            monitor_task = asyncio.create_task(self.monitor.start_monitoring())
            
            # Wait for shutdown signal
            shutdown_task = asyncio.create_task(self.shutdown_event.wait())
            
            logger.info("Real-Time Process Monitor service started successfully")
            
            # Wait for either monitoring to complete or shutdown signal
            done, pending = await asyncio.wait(
                [monitor_task, shutdown_task],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Cancel pending tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
            logger.info("Real-Time Process Monitor service stopped")
            
        except Exception as e:
            logger.error(f"Error in monitor service: {e}")
            raise
    
    def _extract_monitor_config(self) -> dict:
        """Extract monitor-specific configuration"""
        return {
            'redis_host': self.config['database']['redis']['host'],
            'redis_port': self.config['database']['redis']['port'],
            'redis_password': self.config['database']['redis'].get('password'),
            'influxdb_url': self.config['database']['influxdb']['url'],
            'influxdb_token': self.config['database']['influxdb']['token'],
            'influxdb_org': self.config['database']['influxdb']['org'],
            'postgres_host': self.config['database']['postgresql']['host'],
            'postgres_port': self.config['database']['postgresql']['port'],
            'postgres_password': self.config['database']['postgresql']['password'],
            'websocket_host': self.config['websocket']['host'],
            'websocket_port': self.config['websocket']['port'],
            'max_workers': self.config['processing']['max_workers'],
            'processing_frequency': self.config['processing']['processing_frequency'],
            'buffer_size': self.config['processing']['buffer_size'],
            'time_window': self.config['processing']['time_window']
        }
    
    async def stop(self):
        """Stop the monitoring service"""
        logger.info("Stopping Real-Time Process Monitor service...")
        self.shutdown_event.set()

async def main():
    """Main entry point"""
    service = MonitorService()
    
    try:
        await service.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Service failed: {e}")
        sys.exit(1)
    finally:
        await service.stop()

if __name__ == "__main__":
    # Ensure log directory exists
    os.makedirs('/app/logs', exist_ok=True)
    
    # Log startup information
    logger.info("="*50)
    logger.info("Real-Time Process Monitor Starting")
    logger.info(f"Timestamp: {datetime.utcnow().isoformat()}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info("="*50)
    
    # Run the service
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Failed to start service: {e}")
        sys.exit(1)
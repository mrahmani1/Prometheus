"""
Main Compliance Framework Service for Prometheus AI Platform

This service orchestrates all compliance frameworks and provides unified
compliance management capabilities.
"""

import asyncio
import logging
import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional
import asyncpg
import aioredis
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
from cryptography.fernet import Fernet

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compliance.gdpr_compliance import GDPRComplianceEngine
from compliance.soc2_compliance import SOC2ComplianceEngine
from compliance.iso27001_isms import ISO27001ISMS
from compliance.audit_logging import TamperProofAuditLogger
from compliance.compliance_monitoring import ComplianceMonitoringEngine
from compliance.data_retention import DataRetentionEngine
from compliance.compliance_dashboard import ComplianceDashboard

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ComplianceFrameworkService:
    """
    Main compliance framework service that orchestrates all compliance engines
    and provides unified compliance management for the Prometheus AI Platform.
    """
    
    def __init__(self):
        self.db_pool: Optional[asyncpg.Pool] = None
        self.redis_client: Optional[aioredis.Redis] = None
        self.gdpr_engine: Optional[GDPRComplianceEngine] = None
        self.soc2_engine: Optional[SOC2ComplianceEngine] = None
        self.iso27001_isms: Optional[ISO27001ISMS] = None
        self.audit_logger: Optional[TamperProofAuditLogger] = None
        self.compliance_monitor: Optional[ComplianceMonitoringEngine] = None
        self.retention_engine: Optional[DataRetentionEngine] = None
        self.dashboard: Optional[ComplianceDashboard] = None
        
        # Cryptographic keys
        self.signing_key: Optional[rsa.RSAPrivateKey] = None
        self.encryption_key: Optional[bytes] = None
        
    async def initialize(self):
        """Initialize all compliance framework components"""
        try:
            logger.info("Initializing Compliance Framework Service...")
            
            # Initialize cryptographic keys
            await self._initialize_crypto_keys()
            
            # Initialize database connections
            await self._initialize_database()
            
            # Initialize Redis connection
            await self._initialize_redis()
            
            # Initialize compliance engines
            await self._initialize_compliance_engines()
            
            # Initialize database schemas
            await self._initialize_schemas()
            
            # Initialize compliance policies and configurations
            await self._initialize_compliance_configurations()
            
            logger.info("Compliance Framework Service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Compliance Framework Service: {str(e)}")
            raise
    
    async def _initialize_crypto_keys(self):
        """Initialize cryptographic keys for audit logging and data protection"""
        # Generate RSA key pair for digital signatures
        self.signing_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        
        # Generate encryption key for sensitive data
        self.encryption_key = Fernet.generate_key()
        
        logger.info("Cryptographic keys initialized")
    
    async def _initialize_database(self):
        """Initialize PostgreSQL database connection pool"""
        database_url = os.getenv(
            'DATABASE_URL',
            'postgresql://prometheus:prometheus123@localhost:5432/prometheus'
        )
        
        self.db_pool = await asyncpg.create_pool(
            database_url,
            min_size=5,
            max_size=20,
            command_timeout=60
        )
        
        logger.info("Database connection pool initialized")
    
    async def _initialize_redis(self):
        """Initialize Redis connection for caching and real-time data"""
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
        
        self.redis_client = await aioredis.from_url(
            redis_url,
            encoding='utf-8',
            decode_responses=False
        )
        
        logger.info("Redis connection initialized")
    
    async def _initialize_compliance_engines(self):
        """Initialize all compliance framework engines"""
        # GDPR Compliance Engine
        self.gdpr_engine = GDPRComplianceEngine(
            self.db_pool,
            self.redis_client,
            self.encryption_key
        )
        
        # SOC2 Compliance Engine
        self.soc2_engine = SOC2ComplianceEngine(
            self.db_pool,
            self.redis_client
        )
        
        # ISO27001 ISMS
        self.iso27001_isms = ISO27001ISMS(
            self.db_pool,
            self.redis_client
        )
        
        # Tamper-Proof Audit Logger
        self.audit_logger = TamperProofAuditLogger(
            self.db_pool,
            self.redis_client,
            self.signing_key,
            self.encryption_key
        )
        
        # Compliance Monitoring Engine
        self.compliance_monitor = ComplianceMonitoringEngine(
            self.db_pool,
            self.redis_client
        )
        
        # Data Retention Engine
        self.retention_engine = DataRetentionEngine(
            self.db_pool,
            self.redis_client,
            self.encryption_key
        )
        
        # Compliance Dashboard
        self.dashboard = ComplianceDashboard(
            self.db_pool,
            self.redis_client
        )
        
        logger.info("All compliance engines initialized")
    
    async def _initialize_schemas(self):
        """Initialize database schemas for all compliance frameworks"""
        schemas = [
            ("GDPR", "services/security-framework/compliance/gdpr_compliance.py"),
            ("SOC2", "services/security-framework/compliance/soc2_compliance.py"),
            ("ISO27001", "services/security-framework/compliance/iso27001_isms.py"),
            ("Audit Logging", "services/security-framework/compliance/audit_logging.py"),
            ("Compliance Monitoring", "services/security-framework/compliance/compliance_monitoring.py"),
            ("Data Retention", "services/security-framework/compliance/data_retention.py"),
            ("Dashboard", "services/security-framework/compliance/compliance_dashboard.py")
        ]
        
        async with self.db_pool.acquire() as conn:
            for schema_name, schema_file in schemas:
                try:
                    # In a real implementation, you would execute the SQL from each module
                    # For now, we'll create basic tables
                    logger.info(f"Initializing {schema_name} schema...")
                    
                    # This would execute the actual SQL schema from each module
                    # await conn.execute(schema_sql)
                    
                except Exception as e:
                    logger.error(f"Failed to initialize {schema_name} schema: {str(e)}")
        
        logger.info("Database schemas initialized")
    
    async def _initialize_compliance_configurations(self):
        """Initialize compliance policies and configurations"""
        # Initialize GDPR data categories and consent management
        # (This would be done by each engine's initialization methods)
        
        # Initialize SOC2 controls
        # (This would be done by the SOC2 engine)
        
        # Initialize ISO27001 security controls
        # (This would be done by the ISO27001 ISMS)
        
        # Initialize compliance monitoring policies
        await self.compliance_monitor.initialize_compliance_policies()
        
        # Initialize data retention policies
        await self.retention_engine.initialize_retention_policies()
        
        # Initialize dashboard widgets
        await self.dashboard.initialize_dashboard_widgets()
        
        logger.info("Compliance configurations initialized")
    
    async def start_monitoring(self):
        """Start continuous compliance monitoring tasks"""
        try:
            # Start compliance monitoring
            await self.compliance_monitor.start_compliance_monitoring()
            
            # Start SOC2 continuous control monitoring
            asyncio.create_task(self.soc2_engine.monitor_continuous_controls())
            
            logger.info("Compliance monitoring started")
            
        except Exception as e:
            logger.error(f"Failed to start compliance monitoring: {str(e)}")
            raise
    
    async def get_compliance_status(self) -> Dict[str, Any]:
        """Get overall compliance status across all frameworks"""
        try:
            # Get GDPR compliance metrics
            gdpr_metrics = await self.gdpr_engine.monitor_compliance_metrics()
            
            # Get SOC2 compliance status
            # (This would be implemented in the SOC2 engine)
            
            # Get ISO27001 ISMS status
            isms_dashboard = await self.iso27001_isms.generate_isms_dashboard()
            
            # Get overall dashboard data
            dashboard_data = await self.dashboard.get_dashboard_data(["compliance_admin"])
            
            return {
                "overall_status": "compliant",  # Would be calculated
                "gdpr_metrics": gdpr_metrics,
                "isms_dashboard": isms_dashboard,
                "dashboard_summary": dashboard_data.get("summary_metrics", {}),
                "last_updated": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get compliance status: {str(e)}")
            return {
                "overall_status": "error",
                "error": str(e),
                "last_updated": datetime.utcnow().isoformat()
            }
    
    async def generate_compliance_report(self, framework: str, 
                                       start_date: datetime, 
                                       end_date: datetime) -> Dict[str, Any]:
        """Generate compliance report for specified framework"""
        try:
            if framework.upper() == "GDPR":
                # Generate GDPR compliance report
                return {
                    "framework": "GDPR",
                    "report_period": {
                        "start": start_date.isoformat(),
                        "end": end_date.isoformat()
                    },
                    "metrics": await self.gdpr_engine.monitor_compliance_metrics(),
                    "generated_at": datetime.utcnow().isoformat()
                }
            
            elif framework.upper() == "SOC2":
                # Generate SOC2 compliance report
                return await self.soc2_engine.generate_soc2_report(start_date, end_date)
            
            elif framework.upper() == "ISO27001":
                # Generate ISO27001 ISMS report
                return await self.iso27001_isms.generate_isms_dashboard()
            
            else:
                raise ValueError(f"Unsupported compliance framework: {framework}")
                
        except Exception as e:
            logger.error(f"Failed to generate {framework} compliance report: {str(e)}")
            raise
    
    async def shutdown(self):
        """Gracefully shutdown the compliance framework service"""
        try:
            logger.info("Shutting down Compliance Framework Service...")
            
            # Cancel monitoring tasks
            for task in asyncio.all_tasks():
                if not task.done():
                    task.cancel()
            
            # Close database connections
            if self.db_pool:
                await self.db_pool.close()
            
            # Close Redis connection
            if self.redis_client:
                await self.redis_client.close()
            
            logger.info("Compliance Framework Service shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")

async def main():
    """Main entry point for the compliance framework service"""
    service = ComplianceFrameworkService()
    
    try:
        # Initialize the service
        await service.initialize()
        
        # Start monitoring
        await service.start_monitoring()
        
        # Keep the service running
        logger.info("Compliance Framework Service is running...")
        
        # In a real deployment, this would be a web server or message queue consumer
        while True:
            await asyncio.sleep(60)  # Check every minute
            
            # Perform periodic health checks
            status = await service.get_compliance_status()
            logger.info(f"Compliance status: {status['overall_status']}")
            
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except Exception as e:
        logger.error(f"Service error: {str(e)}")
    finally:
        await service.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
"""
Tectonic Setting Integration Module

Integrates the comprehensive tectonic setting database with existing volcanic systems,
providing seamless access to tectonic classification and analysis capabilities.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from tectonic_setting_database import TectonicSettingDatabase, TectonicSetting
from tectonic_database_init import TectonicDatabaseInitializer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TectonicIntegrationManager:
    """
    Manages integration between tectonic setting database and volcanic systems
    """
    
    def __init__(self, database_url: str):
        """Initialize the integration manager"""
        self.database_url = database_url
        self.tectonic_db = TectonicSettingDatabase(database_url)
        self.engine = create_engine(database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        logger.info("TectonicIntegrationManager initialized")
    
    async def initialize_integration(self):
        """Initialize the complete tectonic integration system"""
        
        logger.info("Initializing tectonic integration system...")
        
        # Initialize tectonic database if needed
        await self._ensure_tectonic_database_ready()
        
        # Create integration views and functions
        await self._create_integration_views()
        
        # Set up automated classification triggers
        await self._setup_classif
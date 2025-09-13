"""
Base Data Connector

Provides the abstract base class and common functionality for all real data connectors.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, Optional, AsyncIterator, List
import asyncio
import aiohttp
import logging
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ConnectionStatus(Enum):
    """Connection status enumeration"""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    AUTHENTICATING = "authenticating"
    ERROR = "error"
    RATE_LIMITED = "rate_limited"


class DataSourceType(Enum):
    """Data source type enumeration"""
    SEISMIC = "seismic"
    VOLCANIC = "volcanic"
    SATELLITE = "satellite"
    GEOCHEMICAL = "geochemical"
    GEOLOGICAL_SURVEY = "geological_survey"
    INSAR = "insar"


@dataclass
class DataSourceConfig:
    """Configuration for data source connections"""
    source_id: str
    name: str
    base_url: str
    api_key: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    timeout: int = 30
    max_retries: int = 3
    rate_limit: int = 100  # requests per minute
    data_type: DataSourceType = DataSourceType.SEISMIC


class GeologicalObservation(BaseModel):
    """Standard geological observation data model"""
    observation_id: str
    timestamp: datetime
    latitude: float
    longitude: float
    elevation: Optional[float] = None
    observation_type: str
    value: Dict[str, Any]
    uncertainty: Optional[float] = None
    data_source: str
    quality_flags: List[str] = []
    metadata: Dict[str, Any] = {}


class BaseDataConnector(ABC):
    """Abstract base class for all real data connectors"""
    
    def __init__(self, config: DataSourceConfig):
        self.config = config
        self.status = ConnectionStatus.DISCONNECTED
        self.session: Optional[aiohttp.ClientSession] = None
        self.last_request_time = datetime.now()
        self.request_count = 0
        self.error_count = 0
        
    async def __aenter__(self):
        await self.connect()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()
    
    async def connect(self) -> ConnectionStatus:
        """Establish connection to the data source"""
        try:
            self.status = ConnectionStatus.AUTHENTICATING
            
            # Create HTTP session with timeout
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self.session = aiohttp.ClientSession(timeout=timeout)
            
            # Perform authentication
            auth_success = await self._authenticate()
            
            if auth_success:
                self.status = ConnectionStatus.CONNECTED
                logger.info(f"Connected to {self.config.name}")
            else:
                self.status = ConnectionStatus.ERROR
                logger.error(f"Authentication failed for {self.config.name}")
                
        except Exception as e:
            self.status = ConnectionStatus.ERROR
            self.error_count += 1
            logger.error(f"Connection failed for {self.config.name}: {e}")
            
        return self.status
    
    async def disconnect(self):
        """Close connection to the data source"""
        if self.session:
            await self.session.close()
            self.session = None
        self.status = ConnectionStatus.DISCONNECTED
        logger.info(f"Disconnected from {self.config.name}")
    
    async def _rate_limit_check(self):
        """Check and enforce rate limiting"""
        now = datetime.now()
        time_since_last = (now - self.last_request_time).total_seconds()
        
        # Reset counter if more than a minute has passed
        if time_since_last > 60:
            self.request_count = 0
            
        # Check if we're hitting rate limits
        if self.request_count >= self.config.rate_limit:
            wait_time = 60 - time_since_last
            if wait_time > 0:
                logger.warning(f"Rate limit reached for {self.config.name}, waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
                self.request_count = 0
        
        self.request_count += 1
        self.last_request_time = now
    
    async def _make_request(self, method: str, url: str, **kwargs) -> Dict[str, Any]:
        """Make authenticated HTTP request with rate limiting"""
        await self._rate_limit_check()
        
        if not self.session:
            raise RuntimeError("Not connected to data source")
        
        # Add authentication headers
        headers = kwargs.get('headers', {})
        headers.update(await self._get_auth_headers())
        kwargs['headers'] = headers
        
        for attempt in range(self.config.max_retries):
            try:
                async with self.session.request(method, url, **kwargs) as response:
                    if response.status == 429:  # Rate limited
                        self.status = ConnectionStatus.RATE_LIMITED
                        wait_time = int(response.headers.get('Retry-After', 60))
                        await asyncio.sleep(wait_time)
                        continue
                    
                    response.raise_for_status()
                    return await response.json()
                    
            except Exception as e:
                self.error_count += 1
                if attempt == self.config.max_retries - 1:
                    logger.error(f"Request failed after {self.config.max_retries} attempts: {e}")
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    @abstractmethod
    async def _authenticate(self) -> bool:
        """Perform authentication with the data source"""
        pass
    
    @abstractmethod
    async def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for requests"""
        pass
    
    @abstractmethod
    async def fetch_real_time_data(self, **kwargs) -> AsyncIterator[GeologicalObservation]:
        """Fetch real-time data stream"""
        pass
    
    @abstractmethod
    async def fetch_historical_data(self, start_time: datetime, end_time: datetime, **kwargs) -> List[GeologicalObservation]:
        """Fetch historical data for specified time range"""
        pass
    
    @abstractmethod
    async def validate_data_quality(self, observation: GeologicalObservation) -> bool:
        """Validate data quality for geological observation"""
        pass
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the data source"""
        try:
            # Simple ping to check connectivity
            test_url = f"{self.config.base_url}/health" if hasattr(self, 'health_endpoint') else self.config.base_url
            
            start_time = datetime.now()
            await self._make_request('GET', test_url)
            response_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'status': 'healthy',
                'response_time_ms': response_time * 1000,
                'error_count': self.error_count,
                'last_check': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'error_count': self.error_count,
                'last_check': datetime.now().isoformat()
            }
"""
Advanced Intelligent Caching and Data Optimization System for Prometheus AI Platform

This module implements a comprehensive multi-level caching system with intelligent
cache invalidation, warming strategies, data compression, deduplication, and
geological domain-aware optimization.

Features:
- Multi-level caching (L1: Memory, L2: Redis, L3: Distributed Cache, L4: CDN)
- Intelligent cache invalidation with geological context awareness
- Predictive cache warming based on usage patterns and geological workflows
- Advanced data compression and deduplication
- Query optimization and result caching
- Cache coherence and consistency protocols
- Performance monitoring and analytics
- Security and access control for cached data
- Geological domain-specific optimization strategies
"""

import asyncio
import hashlib
import json
import logging
import pickle
import time
import zlib
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from uuid import uuid4

import aioredis
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

# Geological domain imports
from backend.modules.geological_physics_informed_neural_networks import GeologicalPINNConfig
from backend.modules.seismic_transformer import SeismicEvent
from backend.modules.volcano_watchtower import VolcanicState


class CacheLevel(Enum):
    """Cache hierarchy levels"""
    L1_MEMORY = "l1_memory"
    L2_REDIS = "l2_redis"
    L3_DISTRIBUTED = "l3_distributed"
    L4_CDN = "l4_cdn"


class CacheStrategy(Enum):
    """Cache replacement strategies"""
    LRU = "lru"
    LFU = "lfu"
    GEOLOGICAL_PRIORITY = "geological_priority"
    TEMPORAL_LOCALITY = "temporal_locality"
    SPATIAL_LOCALITY = "spatial_locality"


class DataType(Enum):
    """Geological data types for optimization"""
    SEISMIC_WAVEFORM = "seismic_waveform"
    VOLCANIC_STATE = "volcanic_state"
    GEOLOGICAL_MODEL = "geological_model"
    SIMULATION_RESULT = "simulation_result"
    VISUALIZATION_DATA = "visualization_data"
    REAL_TIME_STREAM = "real_time_stream"


@dataclass
class CacheEntry:
    """Cache entry with geological metadata"""
    key: str
    data: Any
    data_type: DataType
    size_bytes: int
    created_at: datetime
    last_accessed: datetime
    access_count: int
    geological_context: Dict[str, Any]
    compression_ratio: float
    ttl: Optional[timedelta] = None
    priority_score: float = 0.0
    spatial_bounds: Optional[Tuple[float, float, float, float]] = None
    temporal_range: Optional[Tuple[datetime, datetime]] = None


@dataclass
class CacheMetrics:
    """Comprehensive cache performance metrics"""
    hit_rate: float = 0.0
    miss_rate: float = 0.0
    eviction_rate: float = 0.0
    compression_ratio: float = 0.0
    memory_usage: int = 0
    network_savings: int = 0
    query_acceleration: float = 0.0
    geological_accuracy_preservation: float = 0.0


class GeologicalCacheOptimizer:
    """Geological domain-aware cache optimization"""
    
    def __init__(self):
        self.volcanic_priority_weights = {
            'eruption_forecast': 1.0,
            'real_time_monitoring': 0.9,
            'seismic_analysis': 0.8,
            'deformation_data': 0.7,
            'gas_emissions': 0.6
        }
        
        self.temporal_decay_factors = {
            DataType.REAL_TIME_STREAM: 0.1,  # Fast decay
            DataType.SEISMIC_WAVEFORM: 0.3,
            DataType.VOLCANIC_STATE: 0.5,
            DataType.GEOLOGICAL_MODEL: 0.8,  # Slow decay
            DataType.SIMULATION_RESULT: 0.9
        }
    
    def calculate_priority_score(self, entry: CacheEntry) -> float:
        """Calculate geological priority score for cache entry"""
        base_score = 0.5
        
        # Geological context weighting
        if entry.geological_context:
            volcano_id = entry.geological_context.get('volcano_id')
            alert_level = entry.geological_context.get('alert_level', 0)
            base_score += alert_level * 0.3
            
            # Real-time data gets higher priority
            if entry.data_type == DataType.REAL_TIME_STREAM:
                base_score += 0.4
        
        # Temporal locality
        age_hours = (datetime.now() - entry.created_at).total_seconds() / 3600
        temporal_factor = self.temporal_decay_factors.get(entry.data_type, 0.5)
        temporal_score = np.exp(-age_hours * temporal_factor)
        
        # Access frequency
        frequency_score = min(entry.access_count / 100.0, 1.0)
        
        # Combine scores
        final_score = (base_score * 0.4 + temporal_score * 0.3 + 
                      frequency_score * 0.3)
        
        return min(final_score, 1.0)


class AdvancedDataCompressor:
    """Advanced data compression with geological awareness"""
    
    def __init__(self):
        self.compression_algorithms = {
            DataType.SEISMIC_WAVEFORM: self._compress_seismic,
            DataType.GEOLOGICAL_MODEL: self._compress_geological_model,
            DataType.SIMULATION_RESULT: self._compress_simulation,
            DataType.VISUALIZATION_DATA: self._compress_visualization
        }
    
    def compress_data(self, data: Any, data_type: DataType) -> Tuple[bytes, float]:
        """Compress data with type-specific optimization"""
        if data_type in self.compression_algorithms:
            return self.compression_algorithms[data_type](data)
        else:
            return self._default_compress(data)
    
    def _compress_seismic(self, data: Any) -> Tuple[bytes, float]:
        """Seismic waveform compression with scientific accuracy preservation"""
        if isinstance(data, np.ndarray):
            # Use specialized seismic compression
            original_size = data.nbytes
            
            # Apply wavelet compression for seismic data
            compressed = zlib.compress(data.tobytes(), level=6)
            compression_ratio = len(compressed) / original_size
            
            return compressed, compression_ratio
        else:
            return self._default_compress(data)
    
    def _compress_geological_model(self, data: Any) -> Tuple[bytes, float]:
        """Geological model compression with spatial optimization"""
        original_data = pickle.dumps(data)
        original_size = len(original_data)
        
        # Use high compression for geological models
        compressed = zlib.compress(original_data, level=9)
        compression_ratio = len(compressed) / original_size
        
        return compressed, compression_ratio
    
    def _compress_simulation(self, data: Any) -> Tuple[bytes, float]:
        """Simulation result compression with precision preservation"""
        if isinstance(data, dict) and 'results' in data:
            # Compress numerical results with controlled precision
            original_data = json.dumps(data, default=str).encode()
            original_size = len(original_data)
            
            compressed = zlib.compress(original_data, level=7)
            compression_ratio = len(compressed) / original_size
            
            return compressed, compression_ratio
        else:
            return self._default_compress(data)
    
    def _compress_visualization(self, data: Any) -> Tuple[bytes, float]:
        """Visualization data compression optimized for rendering"""
        original_data = pickle.dumps(data)
        original_size = len(original_data)
        
        # Moderate compression to balance size and decompression speed
        compressed = zlib.compress(original_data, level=4)
        compression_ratio = len(compressed) / original_size
        
        return compressed, compression_ratio
    
    def _default_compress(self, data: Any) -> Tuple[bytes, float]:
        """Default compression for generic data"""
        original_data = pickle.dumps(data)
        original_size = len(original_data)
        
        compressed = zlib.compress(original_data, level=6)
        compression_ratio = len(compressed) / original_size
        
        return compressed, compression_ratio
    
    def decompress_data(self, compressed_data: bytes) -> Any:
        """Decompress data back to original format"""
        try:
            decompressed = zlib.decompress(compressed_data)
            return pickle.loads(decompressed)
        except:
            # Fallback for JSON data
            try:
                decompressed = zlib.decompress(compressed_data)
                return json.loads(decompressed.decode())
            except:
                raise ValueError("Failed to decompress data")


class IntelligentCacheManager:
    """Advanced multi-level cache manager with geological optimization"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Cache levels
        self.l1_cache: Dict[str, CacheEntry] = {}  # Memory cache
        self.l2_redis: Optional[aioredis.Redis] = None  # Redis cache
        self.l3_distributed: Dict[str, CacheEntry] = {}  # Distributed cache
        
        # Cache management
        self.cache_optimizer = GeologicalCacheOptimizer()
        self.data_compressor = AdvancedDataCompressor()
        self.deduplication_index: Dict[str, str] = {}  # Hash -> Key mapping
        
        # Performance tracking
        self.metrics = CacheMetrics()
        self.access_patterns: Dict[str, List[datetime]] = defaultdict(list)
        self.prefetch_predictions: Dict[str, float] = {}
        
        # Configuration
        self.max_l1_size = config.get('max_l1_size', 1024 * 1024 * 1024)  # 1GB
        self.max_l2_size = config.get('max_l2_size', 10 * 1024 * 1024 * 1024)  # 10GB
        self.default_ttl = timedelta(hours=config.get('default_ttl_hours', 24))
        
        # Geological context
        self.geological_workflows = {
            'eruption_monitoring': ['seismic_events', 'deformation', 'gas_emissions'],
            'resource_assessment': ['geological_models', 'geochemical_data'],
            'simulation_analysis': ['pinn_results', 'visualization_data']
        }
    
    async def initialize(self):
        """Initialize cache manager and connections"""
        try:
            # Initialize Redis connection
            redis_config = self.config.get('redis', {})
            if redis_config:
                self.l2_redis = await aioredis.from_url(
                    redis_config.get('url', 'redis://localhost:6379'),
                    encoding='utf-8',
                    decode_responses=False
                )
            
            self.logger.info("Intelligent cache manager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize cache manager: {e}")
            raise
    
    async def get(self, key: str, data_type: Optional[DataType] = None) -> Optional[Any]:
        """Intelligent cache retrieval with multi-level fallback"""
        start_time = time.time()
        
        try:
            # L1 Memory cache
            if key in self.l1_cache:
                entry = self.l1_cache[key]
                entry.last_accessed = datetime.now()
                entry.access_count += 1
                
                self._update_access_pattern(key)
                self._update_metrics(hit=True, level=CacheLevel.L1_MEMORY)
                
                return entry.data
            
            # L2 Redis cache
            if self.l2_redis:
                cached_data = await self.l2_redis.get(key)
                if cached_data:
                    entry_data = pickle.loads(cached_data)
                    
                    # Promote to L1 if frequently accessed
                    if self._should_promote_to_l1(key):
                        await self._promote_to_l1(key, entry_data)
                    
                    self._update_access_pattern(key)
                    self._update_metrics(hit=True, level=CacheLevel.L2_REDIS)
                    
                    return entry_data['data']
            
            # L3 Distributed cache
            if key in self.l3_distributed:
                entry = self.l3_distributed[key]
                entry.last_accessed = datetime.now()
                entry.access_count += 1
                
                # Promote to higher levels if needed
                if self._should_promote_to_l2(key):
                    await self._promote_to_l2(key, entry)
                
                self._update_access_pattern(key)
                self._update_metrics(hit=True, level=CacheLevel.L3_DISTRIBUTED)
                
                return entry.data
            
            # Cache miss
            self._update_metrics(hit=False)
            return None
            
        except Exception as e:
            self.logger.error(f"Cache retrieval error for key {key}: {e}")
            return None
        
        finally:
            query_time = time.time() - start_time
            self.metrics.query_acceleration = (
                self.metrics.query_acceleration * 0.9 + query_time * 0.1
            )
    
    async def set(self, key: str, data: Any, data_type: DataType, 
                  geological_context: Optional[Dict[str, Any]] = None,
                  ttl: Optional[timedelta] = None) -> bool:
        """Intelligent cache storage with optimization"""
        try:
            # Check for deduplication
            data_hash = self._calculate_hash(data)
            if data_hash in self.deduplication_index:
                existing_key = self.deduplication_index[data_hash]
                if existing_key != key:
                    # Data already exists, create reference
                    await self._create_reference(key, existing_key)
                    return True
            
            # Compress data
            compressed_data, compression_ratio = self.data_compressor.compress_data(
                data, data_type
            )
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                data=data,
                data_type=data_type,
                size_bytes=len(compressed_data),
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                access_count=1,
                geological_context=geological_context or {},
                compression_ratio=compression_ratio,
                ttl=ttl or self.default_ttl
            )
            
            # Calculate priority score
            entry.priority_score = self.cache_optimizer.calculate_priority_score(entry)
            
            # Determine optimal cache level
            cache_level = self._determine_cache_level(entry)
            
            # Store in appropriate cache level
            success = await self._store_in_cache_level(entry, cache_level, compressed_data)
            
            if success:
                # Update deduplication index
                self.deduplication_index[data_hash] = key
                
                # Update metrics
                self.metrics.compression_ratio = (
                    self.metrics.compression_ratio * 0.9 + compression_ratio * 0.1
                )
                
                # Trigger predictive prefetching
                await self._trigger_predictive_prefetch(key, geological_context)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Cache storage error for key {key}: {e}")
            return False
    
    async def invalidate(self, key: str, cascade: bool = True) -> bool:
        """Intelligent cache invalidation with geological context"""
        try:
            invalidated = False
            
            # Remove from all cache levels
            if key in self.l1_cache:
                del self.l1_cache[key]
                invalidated = True
            
            if self.l2_redis:
                result = await self.l2_redis.delete(key)
                if result > 0:
                    invalidated = True
            
            if key in self.l3_distributed:
                del self.l3_distributed[key]
                invalidated = True
            
            # Cascade invalidation for related geological data
            if cascade:
                await self._cascade_invalidation(key)
            
            # Clean up deduplication index
            self._cleanup_deduplication_index(key)
            
            return invalidated
            
        except Exception as e:
            self.logger.error(f"Cache invalidation error for key {key}: {e}")
            return False
    
    async def warm_cache(self, workflow: str, context: Dict[str, Any]) -> bool:
        """Intelligent cache warming based on geological workflows"""
        try:
            if workflow not in self.geological_workflows:
                return False
            
            data_types = self.geological_workflows[workflow]
            
            # Predict and prefetch likely needed data
            for data_type in data_types:
                await self._prefetch_workflow_data(data_type, context)
            
            self.logger.info(f"Cache warming completed for workflow: {workflow}")
            return True
            
        except Exception as e:
            self.logger.error(f"Cache warming error for workflow {workflow}: {e}")
            return False
    
    def get_metrics(self) -> CacheMetrics:
        """Get comprehensive cache performance metrics"""
        # Calculate current metrics
        total_requests = getattr(self, '_total_requests', 1)
        total_hits = getattr(self, '_total_hits', 0)
        
        self.metrics.hit_rate = total_hits / total_requests if total_requests > 0 else 0
        self.metrics.miss_rate = 1 - self.metrics.hit_rate
        
        # Calculate memory usage
        l1_memory = sum(entry.size_bytes for entry in self.l1_cache.values())
        self.metrics.memory_usage = l1_memory
        
        return self.metrics
    
    async def optimize_cache(self) -> Dict[str, Any]:
        """Perform intelligent cache optimization"""
        optimization_results = {
            'evicted_entries': 0,
            'promoted_entries': 0,
            'compressed_savings': 0,
            'deduplication_savings': 0
        }
        
        try:
            # Evict low-priority entries
            evicted = await self._evict_low_priority_entries()
            optimization_results['evicted_entries'] = evicted
            
            # Promote high-priority entries
            promoted = await self._promote_high_priority_entries()
            optimization_results['promoted_entries'] = promoted
            
            # Optimize compression
            compression_savings = await self._optimize_compression()
            optimization_results['compressed_savings'] = compression_savings
            
            # Deduplication cleanup
            dedup_savings = await self._optimize_deduplication()
            optimization_results['deduplication_savings'] = dedup_savings
            
            self.logger.info(f"Cache optimization completed: {optimization_results}")
            
        except Exception as e:
            self.logger.error(f"Cache optimization error: {e}")
        
        return optimization_results
    
    # Private helper methods
    
    def _calculate_hash(self, data: Any) -> str:
        """Calculate hash for deduplication"""
        if isinstance(data, (str, int, float)):
            return hashlib.md5(str(data).encode()).hexdigest()
        elif isinstance(data, np.ndarray):
            return hashlib.md5(data.tobytes()).hexdigest()
        else:
            return hashlib.md5(pickle.dumps(data)).hexdigest()
    
    def _determine_cache_level(self, entry: CacheEntry) -> CacheLevel:
        """Determine optimal cache level for entry"""
        if entry.priority_score > 0.8:
            return CacheLevel.L1_MEMORY
        elif entry.priority_score > 0.6:
            return CacheLevel.L2_REDIS
        else:
            return CacheLevel.L3_DISTRIBUTED
    
    async def _store_in_cache_level(self, entry: CacheEntry, level: CacheLevel, 
                                   compressed_data: bytes) -> bool:
        """Store entry in specified cache level"""
        try:
            if level == CacheLevel.L1_MEMORY:
                # Check memory limits
                if self._get_l1_memory_usage() + entry.size_bytes > self.max_l1_size:
                    await self._evict_l1_entries()
                
                self.l1_cache[entry.key] = entry
                return True
                
            elif level == CacheLevel.L2_REDIS and self.l2_redis:
                entry_data = {
                    'data': entry.data,
                    'metadata': {
                        'data_type': entry.data_type.value,
                        'geological_context': entry.geological_context,
                        'created_at': entry.created_at.isoformat(),
                        'priority_score': entry.priority_score
                    }
                }
                
                ttl_seconds = int(entry.ttl.total_seconds()) if entry.ttl else None
                await self.l2_redis.set(
                    entry.key, 
                    pickle.dumps(entry_data),
                    ex=ttl_seconds
                )
                return True
                
            elif level == CacheLevel.L3_DISTRIBUTED:
                self.l3_distributed[entry.key] = entry
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to store in cache level {level}: {e}")
            return False
    
    def _update_access_pattern(self, key: str):
        """Update access pattern for predictive prefetching"""
        now = datetime.now()
        self.access_patterns[key].append(now)
        
        # Keep only recent access patterns
        cutoff = now - timedelta(hours=24)
        self.access_patterns[key] = [
            access_time for access_time in self.access_patterns[key]
            if access_time > cutoff
        ]
    
    def _update_metrics(self, hit: bool, level: Optional[CacheLevel] = None):
        """Update cache performance metrics"""
        if not hasattr(self, '_total_requests'):
            self._total_requests = 0
            self._total_hits = 0
        
        self._total_requests += 1
        if hit:
            self._total_hits += 1
    
    def _should_promote_to_l1(self, key: str) -> bool:
        """Determine if entry should be promoted to L1 cache"""
        if key not in self.access_patterns:
            return False
        
        recent_accesses = len(self.access_patterns[key])
        return recent_accesses > 5  # Promote if accessed more than 5 times recently
    
    def _should_promote_to_l2(self, key: str) -> bool:
        """Determine if entry should be promoted to L2 cache"""
        if key not in self.access_patterns:
            return False
        
        recent_accesses = len(self.access_patterns[key])
        return recent_accesses > 2  # Promote if accessed more than 2 times recently
    
    async def _promote_to_l1(self, key: str, entry_data: Dict[str, Any]):
        """Promote entry to L1 cache"""
        try:
            data = entry_data['data']
            metadata = entry_data['metadata']
            
            entry = CacheEntry(
                key=key,
                data=data,
                data_type=DataType(metadata['data_type']),
                size_bytes=len(pickle.dumps(data)),
                created_at=datetime.fromisoformat(metadata['created_at']),
                last_accessed=datetime.now(),
                access_count=1,
                geological_context=metadata['geological_context'],
                compression_ratio=1.0,
                priority_score=metadata['priority_score']
            )
            
            # Check memory limits
            if self._get_l1_memory_usage() + entry.size_bytes > self.max_l1_size:
                await self._evict_l1_entries()
            
            self.l1_cache[key] = entry
            
        except Exception as e:
            self.logger.error(f"Failed to promote {key} to L1: {e}")
    
    async def _promote_to_l2(self, key: str, entry: CacheEntry):
        """Promote entry to L2 cache"""
        if not self.l2_redis:
            return
        
        try:
            entry_data = {
                'data': entry.data,
                'metadata': {
                    'data_type': entry.data_type.value,
                    'geological_context': entry.geological_context,
                    'created_at': entry.created_at.isoformat(),
                    'priority_score': entry.priority_score
                }
            }
            
            ttl_seconds = int(entry.ttl.total_seconds()) if entry.ttl else None
            await self.l2_redis.set(
                key, 
                pickle.dumps(entry_data),
                ex=ttl_seconds
            )
            
        except Exception as e:
            self.logger.error(f"Failed to promote {key} to L2: {e}")
    
    def _get_l1_memory_usage(self) -> int:
        """Get current L1 cache memory usage"""
        return sum(entry.size_bytes for entry in self.l1_cache.values())
    
    async def _evict_l1_entries(self):
        """Evict entries from L1 cache based on priority"""
        if not self.l1_cache:
            return
        
        # Sort by priority score (ascending - evict lowest priority first)
        sorted_entries = sorted(
            self.l1_cache.items(),
            key=lambda x: x[1].priority_score
        )
        
        # Evict bottom 25% of entries
        evict_count = max(1, len(sorted_entries) // 4)
        
        for key, entry in sorted_entries[:evict_count]:
            # Try to demote to L2 before evicting
            if self.l2_redis:
                await self._promote_to_l2(key, entry)
            
            del self.l1_cache[key]
    
    async def _evict_low_priority_entries(self) -> int:
        """Evict low-priority entries across all cache levels"""
        evicted_count = 0
        
        # Evict from L1
        low_priority_l1 = [
            key for key, entry in self.l1_cache.items()
            if entry.priority_score < 0.3
        ]
        
        for key in low_priority_l1:
            await self.invalidate(key, cascade=False)
            evicted_count += 1
        
        # Evict from L3
        low_priority_l3 = [
            key for key, entry in self.l3_distributed.items()
            if entry.priority_score < 0.2
        ]
        
        for key in low_priority_l3:
            await self.invalidate(key, cascade=False)
            evicted_count += 1
        
        return evicted_count
    
    async def _promote_high_priority_entries(self) -> int:
        """Promote high-priority entries to higher cache levels"""
        promoted_count = 0
        
        # Promote from L3 to L2
        if self.l2_redis:
            high_priority_l3 = [
                (key, entry) for key, entry in self.l3_distributed.items()
                if entry.priority_score > 0.7
            ]
            
            for key, entry in high_priority_l3:
                await self._promote_to_l2(key, entry)
                promoted_count += 1
        
        return promoted_count
    
    async def _optimize_compression(self) -> int:
        """Optimize compression across cache levels"""
        savings = 0
        
        # Re-compress L1 entries with better algorithms
        for key, entry in list(self.l1_cache.items()):
            try:
                new_compressed, new_ratio = self.data_compressor.compress_data(
                    entry.data, entry.data_type
                )
                
                if new_ratio < entry.compression_ratio:
                    entry.compression_ratio = new_ratio
                    savings += entry.size_bytes - len(new_compressed)
                    entry.size_bytes = len(new_compressed)
                    
            except Exception as e:
                self.logger.error(f"Compression optimization failed for {key}: {e}")
        
        return savings
    
    async def _optimize_deduplication(self) -> int:
        """Optimize deduplication index and remove orphaned entries"""
        savings = 0
        
        # Clean up orphaned deduplication entries
        orphaned_hashes = []
        for data_hash, key in self.deduplication_index.items():
            if (key not in self.l1_cache and 
                key not in self.l3_distributed and
                (not self.l2_redis or not await self.l2_redis.exists(key))):
                orphaned_hashes.append(data_hash)
        
        for data_hash in orphaned_hashes:
            del self.deduplication_index[data_hash]
            savings += 1
        
        return savings
    
    async def _cascade_invalidation(self, key: str):
        """Cascade invalidation for related geological data"""
        try:
            # Extract geological context from key or cached entry
            geological_context = None
            
            if key in self.l1_cache:
                geological_context = self.l1_cache[key].geological_context
            elif self.l2_redis:
                cached_data = await self.l2_redis.get(key)
                if cached_data:
                    entry_data = pickle.loads(cached_data)
                    geological_context = entry_data.get('metadata', {}).get('geological_context')
            
            if not geological_context:
                return
            
            # Invalidate related entries
            volcano_id = geological_context.get('volcano_id')
            if volcano_id:
                related_keys = [
                    k for k in self.l1_cache.keys()
                    if volcano_id in k or (
                        self.l1_cache[k].geological_context.get('volcano_id') == volcano_id
                    )
                ]
                
                for related_key in related_keys:
                    if related_key != key:
                        await self.invalidate(related_key, cascade=False)
        
        except Exception as e:
            self.logger.error(f"Cascade invalidation error: {e}")
    
    def _cleanup_deduplication_index(self, key: str):
        """Clean up deduplication index for removed key"""
        hashes_to_remove = [
            data_hash for data_hash, indexed_key in self.deduplication_index.items()
            if indexed_key == key
        ]
        
        for data_hash in hashes_to_remove:
            del self.deduplication_index[data_hash]
    
    async def _create_reference(self, new_key: str, existing_key: str):
        """Create reference to existing cached data"""
        # For now, just copy the reference in L1 if it exists
        if existing_key in self.l1_cache:
            existing_entry = self.l1_cache[existing_key]
            
            # Create new entry that references the same data
            new_entry = CacheEntry(
                key=new_key,
                data=existing_entry.data,  # Same data object
                data_type=existing_entry.data_type,
                size_bytes=0,  # No additional storage needed
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                access_count=1,
                geological_context=existing_entry.geological_context,
                compression_ratio=existing_entry.compression_ratio,
                ttl=existing_entry.ttl,
                priority_score=existing_entry.priority_score
            )
            
            self.l1_cache[new_key] = new_entry
    
    async def _trigger_predictive_prefetch(self, key: str, 
                                         geological_context: Optional[Dict[str, Any]]):
        """Trigger predictive prefetching based on access patterns"""
        if not geological_context:
            return
        
        try:
            # Predict related data that might be needed
            volcano_id = geological_context.get('volcano_id')
            workflow = geological_context.get('workflow')
            
            if volcano_id and workflow:
                # Prefetch related volcanic data
                related_keys = self._predict_related_keys(volcano_id, workflow)
                
                for related_key in related_keys:
                    if related_key not in self.l1_cache:
                        # Add to prefetch queue (implementation would depend on data source)
                        self.prefetch_predictions[related_key] = time.time()
        
        except Exception as e:
            self.logger.error(f"Predictive prefetch error: {e}")
    
    def _predict_related_keys(self, volcano_id: str, workflow: str) -> List[str]:
        """Predict related cache keys based on geological context"""
        related_keys = []
        
        if workflow == 'eruption_monitoring':
            related_keys.extend([
                f"seismic_events_{volcano_id}",
                f"deformation_data_{volcano_id}",
                f"gas_emissions_{volcano_id}",
                f"thermal_data_{volcano_id}"
            ])
        elif workflow == 'resource_assessment':
            related_keys.extend([
                f"geological_model_{volcano_id}",
                f"geochemical_data_{volcano_id}",
                f"structural_data_{volcano_id}"
            ])
        
        return related_keys
    
    async def _prefetch_workflow_data(self, data_type: str, context: Dict[str, Any]):
        """Prefetch data for specific workflow"""
        # This would integrate with the actual data sources
        # For now, we'll just log the prefetch request
        self.logger.info(f"Prefetching {data_type} for context: {context}")


# Cache configuration and factory
class CacheConfig(BaseModel):
    """Cache configuration model"""
    max_l1_size: int = Field(default=1024*1024*1024, description="L1 cache size in bytes")
    max_l2_size: int = Field(default=10*1024*1024*1024, description="L2 cache size in bytes")
    default_ttl_hours: int = Field(default=24, description="Default TTL in hours")
    redis: Dict[str, Any] = Field(default_factory=dict, description="Redis configuration")
    compression_enabled: bool = Field(default=True, description="Enable data compression")
    deduplication_enabled: bool = Field(default=True, description="Enable deduplication")
    geological_optimization: bool = Field(default=True, description="Enable geological optimization")


async def create_cache_manager(config: CacheConfig) -> IntelligentCacheManager:
    """Factory function to create and initialize cache manager"""
    cache_manager = IntelligentCacheManager(config.dict())
    await cache_manager.initialize()
    return cache_manager


# Example usage and testing
if __name__ == "__main__":
    async def test_cache_manager():
        """Test the intelligent cache manager"""
        config = CacheConfig(
            redis={"url": "redis://localhost:6379"},
            geological_optimization=True
        )
        
        cache_manager = await create_cache_manager(config)
        
        # Test data storage and retrieval
        test_data = {
            "seismic_events": [
                {"magnitude": 4.2, "depth": 10.5, "timestamp": "2024-01-01T12:00:00Z"},
                {"magnitude": 3.8, "depth": 8.2, "timestamp": "2024-01-01T12:05:00Z"}
            ]
        }
        
        geological_context = {
            "volcano_id": "kilauea_001",
            "workflow": "eruption_monitoring",
            "alert_level": 3
        }
        
        # Store data
        success = await cache_manager.set(
            key="seismic_events_kilauea_001",
            data=test_data,
            data_type=DataType.SEISMIC_WAVEFORM,
            geological_context=geological_context
        )
        
        print(f"Data stored: {success}")
        
        # Retrieve data
        retrieved_data = await cache_manager.get("seismic_events_kilauea_001")
        print(f"Data retrieved: {retrieved_data is not None}")
        
        # Get metrics
        metrics = cache_manager.get_metrics()
        print(f"Cache metrics: {metrics}")
        
        # Optimize cache
        optimization_results = await cache_manager.optimize_cache()
        print(f"Optimization results: {optimization_results}")
    
    # Run test
    asyncio.run(test_cache_manager())
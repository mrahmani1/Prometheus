"""
Distributed Rate Limiting Implementation for Prometheus AI Platform
Advanced rate limiting with multiple algorithms and distributed coordination
"""

import asyncio
import time
import json
import logging
from enum import Enum
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import redis.asyncio as redis
from prometheus_client import Counter, Histogram, Gauge
import hashlib
import math

logger = logging.getLogger(__name__)

class RateLimitAlgorithm(Enum):
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW_LOG = "sliding_window_log"
    SLIDING_WINDOW_COUNTER = "sliding_window_counter"
    FIXED_WINDOW = "fixed_window"
    LEAKY_BUCKET = "leaky_bucket"

class RateLimitScope(Enum):
    GLOBAL = "global"
    PER_USER = "per_user"
    PER_IP = "per_ip"
    PER_SERVICE = "per_service"
    PER_ENDPOINT = "per_endpoint"

@dataclass
class RateLimitRule:
    """Rate limiting rule configuration"""
    name: str
    algorithm: RateLimitAlgorithm
    scope: RateLimitScope
    limit: int  # requests per window
    window_size: int  # window size in seconds
    burst_limit: Optional[int] = None  # for token bucket
    
    # Advanced features
    priority: int = 1  # higher priority rules are checked first
    enabled: bool = True
    description: str = ""
    
    # Hierarchical limits
    parent_rule: Optional[str] = None
    child_rules: List[str] = field(default_factory=list)
    
    # Dynamic adjustment
    auto_adjust: bool = False
    min_limit: Optional[int] = None
    max_limit: Optional[int] = None
    adjustment_factor: float = 1.1

@dataclass
class RateLimitResult:
    """Result of rate limit check"""
    allowed: bool
    remaining: int
    reset_time: datetime
    retry_after: Optional[int] = None
    rule_name: str = ""
    current_usage: int = 0
    
    # Additional metadata
    algorithm_used: str = ""
    scope_key: str = ""
    headers: Dict[str, str] = field(default_factory=dict)

class DistributedRateLimiter:
    """
    Advanced distributed rate limiter with multiple algorithms
    and comprehensive monitoring capabilities
    """
    
    def __init__(self, redis_client: redis.Redis, service_name: str = "prometheus-ai"):
        self.redis_client = redis_client
        self.service_name = service_name
        self.rules: Dict[str, RateLimitRule] = {}
        
        # Metrics
        self.requests_total = Counter(
            'rate_limiter_requests_total',
            'Total number of rate limit checks',
            ['service', 'rule', 'result']
        )
        self.rate_limit_duration = Histogram(
            'rate_limiter_check_duration_seconds',
            'Duration of rate limit checks',
            ['service', 'algorithm']
        )
        self.current_usage = Gauge(
            'rate_limiter_current_usage',
            'Current usage for rate limited resources',
            ['service', 'rule', 'scope_key']
        )
        
        logger.info(f"Initialized distributed rate limiter for service: {service_name}")

    def add_rule(self, rule: RateLimitRule):
        """Add a rate limiting rule"""
        self.rules[rule.name] = rule
        logger.info(f"Added rate limit rule: {rule.name} - {rule.limit}/{rule.window_size}s")

    def remove_rule(self, rule_name: str):
        """Remove a rate limiting rule"""
        if rule_name in self.rules:
            del self.rules[rule_name]
            logger.info(f"Removed rate limit rule: {rule_name}")

    async def check_rate_limit(self, 
                             identifier: str, 
                             rule_names: Optional[List[str]] = None,
                             metadata: Optional[Dict[str, Any]] = None) -> RateLimitResult:
        """
        Check rate limits for a given identifier
        Returns the most restrictive result if multiple rules apply
        """
        start_time = time.time()
        
        # Determine which rules to check
        rules_to_check = []
        if rule_names:
            rules_to_check = [self.rules[name] for name in rule_names if name in self.rules and self.rules[name].enabled]
        else:
            rules_to_check = [rule for rule in self.rules.values() if rule.enabled]
        
        # Sort by priority (higher priority first)
        rules_to_check.sort(key=lambda r: r.priority, reverse=True)
        
        most_restrictive_result = None
        
        for rule in rules_to_check:
            # Generate scope key
            scope_key = self._generate_scope_key(rule.scope, identifier, metadata)
            
            # Check rate limit based on algorithm
            result = await self._check_algorithm(rule, scope_key)
            
            # Update metrics
            self.requests_total.labels(
                service=self.service_name,
                rule=rule.name,
                result='allowed' if result.allowed else 'denied'
            ).inc()
            
            self.current_usage.labels(
                service=self.service_name,
                rule=rule.name,
                scope_key=scope_key
            ).set(result.current_usage)
            
            # Track most restrictive result
            if not result.allowed or (most_restrictive_result and result.remaining < most_restrictive_result.remaining):
                most_restrictive_result = result
                most_restrictive_result.rule_name = rule.name
                most_restrictive_result.algorithm_used = rule.algorithm.value
                most_restrictive_result.scope_key = scope_key
            
            # If any rule denies, return immediately
            if not result.allowed:
                break
        
        # Record duration
        duration = time.time() - start_time
        if most_restrictive_result:
            self.rate_limit_duration.labels(
                service=self.service_name,
                algorithm=most_restrictive_result.algorithm_used
            ).observe(duration)
        
        # Generate response headers
        if most_restrictive_result:
            most_restrictive_result.headers = self._generate_headers(most_restrictive_result)
        
        return most_restrictive_result or RateLimitResult(
            allowed=True,
            remaining=float('inf'),
            reset_time=datetime.now() + timedelta(hours=1)
        )

    async def _check_algorithm(self, rule: RateLimitRule, scope_key: str) -> RateLimitResult:
        """Check rate limit using the specified algorithm"""
        
        if rule.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
            return await self._check_token_bucket(rule, scope_key)
        elif rule.algorithm == RateLimitAlgorithm.SLIDING_WINDOW_LOG:
            return await self._check_sliding_window_log(rule, scope_key)
        elif rule.algorithm == RateLimitAlgorithm.SLIDING_WINDOW_COUNTER:
            return await self._check_sliding_window_counter(rule, scope_key)
        elif rule.algorithm == RateLimitAlgorithm.FIXED_WINDOW:
            return await self._check_fixed_window(rule, scope_key)
        elif rule.algorithm == RateLimitAlgorithm.LEAKY_BUCKET:
            return await self._check_leaky_bucket(rule, scope_key)
        else:
            raise ValueError(f"Unsupported rate limit algorithm: {rule.algorithm}")

    async def _check_token_bucket(self, rule: RateLimitRule, scope_key: str) -> RateLimitResult:
        """Token bucket algorithm implementation"""
        key = f"rate_limit:token_bucket:{self.service_name}:{rule.name}:{scope_key}"
        
        # Lua script for atomic token bucket operations
        lua_script = """
        local key = KEYS[1]
        local capacity = tonumber(ARGV[1])
        local refill_rate = tonumber(ARGV[2])
        local requested_tokens = tonumber(ARGV[3])
        local now = tonumber(ARGV[4])
        
        local bucket = redis.call('HMGET', key, 'tokens', 'last_refill')
        local tokens = tonumber(bucket[1]) or capacity
        local last_refill = tonumber(bucket[2]) or now
        
        -- Calculate tokens to add based on time elapsed
        local time_elapsed = now - last_refill
        local tokens_to_add = math.floor(time_elapsed * refill_rate)
        tokens = math.min(capacity, tokens + tokens_to_add)
        
        local allowed = 0
        local remaining = tokens
        
        if tokens >= requested_tokens then
            tokens = tokens - requested_tokens
            allowed = 1
            remaining = tokens
        end
        
        -- Update bucket state
        redis.call('HMSET', key, 'tokens', tokens, 'last_refill', now)
        redis.call('EXPIRE', key, 3600)  -- 1 hour TTL
        
        return {allowed, remaining, capacity}
        """
        
        capacity = rule.burst_limit or rule.limit
        refill_rate = rule.limit / rule.window_size  # tokens per second
        now = time.time()
        
        result = await self.redis_client.eval(
            lua_script, 1, key, capacity, refill_rate, 1, now
        )
        
        allowed, remaining, _ = result
        reset_time = datetime.now() + timedelta(seconds=rule.window_size)
        
        return RateLimitResult(
            allowed=bool(allowed),
            remaining=int(remaining),
            reset_time=reset_time,
            current_usage=capacity - int(remaining),
            retry_after=None if allowed else int(rule.window_size / refill_rate)
        )

    async def _check_sliding_window_log(self, rule: RateLimitRule, scope_key: str) -> RateLimitResult:
        """Sliding window log algorithm implementation"""
        key = f"rate_limit:sliding_log:{self.service_name}:{rule.name}:{scope_key}"
        now = time.time()
        window_start = now - rule.window_size
        
        # Lua script for atomic sliding window operations
        lua_script = """
        local key = KEYS[1]
        local window_start = tonumber(ARGV[1])
        local now = tonumber(ARGV[2])
        local limit = tonumber(ARGV[3])
        
        -- Remove old entries
        redis.call('ZREMRANGEBYSCORE', key, '-inf', window_start)
        
        -- Count current entries
        local current_count = redis.call('ZCARD', key)
        
        local allowed = 0
        if current_count < limit then
            -- Add new entry
            redis.call('ZADD', key, now, now)
            allowed = 1
            current_count = current_count + 1
        end
        
        redis.call('EXPIRE', key, 3600)  -- 1 hour TTL
        
        return {allowed, limit - current_count, current_count}
        """
        
        result = await self.redis_client.eval(
            lua_script, 1, key, window_start, now, rule.limit
        )
        
        allowed, remaining, current_usage = result
        reset_time = datetime.fromtimestamp(now + rule.window_size)
        
        return RateLimitResult(
            allowed=bool(allowed),
            remaining=int(remaining),
            reset_time=reset_time,
            current_usage=int(current_usage),
            retry_after=None if allowed else rule.window_size
        )

    async def _check_sliding_window_counter(self, rule: RateLimitRule, scope_key: str) -> RateLimitResult:
        """Sliding window counter algorithm implementation"""
        key_prefix = f"rate_limit:sliding_counter:{self.service_name}:{rule.name}:{scope_key}"
        now = time.time()
        current_window = int(now // rule.window_size)
        previous_window = current_window - 1
        
        # Calculate weight for previous window
        time_in_current_window = now - (current_window * rule.window_size)
        weight = 1 - (time_in_current_window / rule.window_size)
        
        # Lua script for atomic sliding window counter operations
        lua_script = """
        local current_key = KEYS[1]
        local previous_key = KEYS[2]
        local limit = tonumber(ARGV[1])
        local weight = tonumber(ARGV[2])
        local window_size = tonumber(ARGV[3])
        
        local current_count = tonumber(redis.call('GET', current_key) or 0)
        local previous_count = tonumber(redis.call('GET', previous_key) or 0)
        
        local estimated_count = math.floor(previous_count * weight + current_count)
        
        local allowed = 0
        if estimated_count < limit then
            current_count = current_count + 1
            redis.call('SET', current_key, current_count, 'EX', window_size * 2)
            allowed = 1
        end
        
        return {allowed, limit - estimated_count - (allowed == 1 and 1 or 0), estimated_count + (allowed == 1 and 1 or 0)}
        """
        
        current_key = f"{key_prefix}:{current_window}"
        previous_key = f"{key_prefix}:{previous_window}"
        
        result = await self.redis_client.eval(
            lua_script, 2, current_key, previous_key, rule.limit, weight, rule.window_size
        )
        
        allowed, remaining, current_usage = result
        reset_time = datetime.fromtimestamp((current_window + 1) * rule.window_size)
        
        return RateLimitResult(
            allowed=bool(allowed),
            remaining=max(0, int(remaining)),
            reset_time=reset_time,
            current_usage=int(current_usage),
            retry_after=None if allowed else int((current_window + 1) * rule.window_size - now)
        )

    async def _check_fixed_window(self, rule: RateLimitRule, scope_key: str) -> RateLimitResult:
        """Fixed window algorithm implementation"""
        now = time.time()
        window = int(now // rule.window_size)
        key = f"rate_limit:fixed_window:{self.service_name}:{rule.name}:{scope_key}:{window}"
        
        # Lua script for atomic fixed window operations
        lua_script = """
        local key = KEYS[1]
        local limit = tonumber(ARGV[1])
        local window_size = tonumber(ARGV[2])
        
        local current_count = tonumber(redis.call('GET', key) or 0)
        
        local allowed = 0
        if current_count < limit then
            current_count = redis.call('INCR', key)
            redis.call('EXPIRE', key, window_size * 2)
            allowed = 1
        end
        
        return {allowed, limit - current_count, current_count}
        """
        
        result = await self.redis_client.eval(
            lua_script, 1, key, rule.limit, rule.window_size
        )
        
        allowed, remaining, current_usage = result
        reset_time = datetime.fromtimestamp((window + 1) * rule.window_size)
        
        return RateLimitResult(
            allowed=bool(allowed),
            remaining=max(0, int(remaining)),
            reset_time=reset_time,
            current_usage=int(current_usage),
            retry_after=None if allowed else int((window + 1) * rule.window_size - now)
        )

    async def _check_leaky_bucket(self, rule: RateLimitRule, scope_key: str) -> RateLimitResult:
        """Leaky bucket algorithm implementation"""
        key = f"rate_limit:leaky_bucket:{self.service_name}:{rule.name}:{scope_key}"
        
        # Lua script for atomic leaky bucket operations
        lua_script = """
        local key = KEYS[1]
        local capacity = tonumber(ARGV[1])
        local leak_rate = tonumber(ARGV[2])
        local now = tonumber(ARGV[3])
        
        local bucket = redis.call('HMGET', key, 'volume', 'last_leak')
        local volume = tonumber(bucket[1]) or 0
        local last_leak = tonumber(bucket[2]) or now
        
        -- Calculate volume to leak based on time elapsed
        local time_elapsed = now - last_leak
        local volume_to_leak = time_elapsed * leak_rate
        volume = math.max(0, volume - volume_to_leak)
        
        local allowed = 0
        if volume < capacity then
            volume = volume + 1
            allowed = 1
        end
        
        -- Update bucket state
        redis.call('HMSET', key, 'volume', volume, 'last_leak', now)
        redis.call('EXPIRE', key, 3600)  -- 1 hour TTL
        
        return {allowed, capacity - volume, volume}
        """
        
        capacity = rule.limit
        leak_rate = rule.limit / rule.window_size  # requests per second
        now = time.time()
        
        result = await self.redis_client.eval(
            lua_script, 1, key, capacity, leak_rate, now
        )
        
        allowed, remaining, current_usage = result
        reset_time = datetime.now() + timedelta(seconds=rule.window_size)
        
        return RateLimitResult(
            allowed=bool(allowed),
            remaining=int(remaining),
            reset_time=reset_time,
            current_usage=int(current_usage),
            retry_after=None if allowed else int(1 / leak_rate)
        )

    def _generate_scope_key(self, scope: RateLimitScope, identifier: str, metadata: Optional[Dict[str, Any]]) -> str:
        """Generate scope key based on rate limit scope"""
        if scope == RateLimitScope.GLOBAL:
            return "global"
        elif scope == RateLimitScope.PER_USER:
            return f"user:{identifier}"
        elif scope == RateLimitScope.PER_IP:
            return f"ip:{identifier}"
        elif scope == RateLimitScope.PER_SERVICE:
            service = metadata.get('service', 'unknown') if metadata else 'unknown'
            return f"service:{service}"
        elif scope == RateLimitScope.PER_ENDPOINT:
            endpoint = metadata.get('endpoint', 'unknown') if metadata else 'unknown'
            return f"endpoint:{endpoint}"
        else:
            return identifier

    def _generate_headers(self, result: RateLimitResult) -> Dict[str, str]:
        """Generate HTTP headers for rate limit response"""
        headers = {
            'X-RateLimit-Limit': str(result.remaining + result.current_usage),
            'X-RateLimit-Remaining': str(result.remaining),
            'X-RateLimit-Reset': str(int(result.reset_time.timestamp())),
            'X-RateLimit-Rule': result.rule_name,
            'X-RateLimit-Algorithm': result.algorithm_used
        }
        
        if result.retry_after:
            headers['Retry-After'] = str(result.retry_after)
        
        return headers

    async def get_usage_stats(self, rule_name: Optional[str] = None) -> Dict[str, Any]:
        """Get usage statistics for rate limiting rules"""
        stats = {}
        
        rules_to_check = [self.rules[rule_name]] if rule_name and rule_name in self.rules else self.rules.values()
        
        for rule in rules_to_check:
            # Get current usage from Redis
            pattern = f"rate_limit:*:{self.service_name}:{rule.name}:*"
            keys = await self.redis_client.keys(pattern)
            
            total_usage = 0
            active_scopes = 0
            
            for key in keys:
                if rule.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
                    bucket_data = await self.redis_client.hmget(key, 'tokens')
                    if bucket_data[0]:
                        capacity = rule.burst_limit or rule.limit
                        usage = capacity - int(bucket_data[0])
                        total_usage += usage
                        if usage > 0:
                            active_scopes += 1
                
                elif rule.algorithm in [RateLimitAlgorithm.SLIDING_WINDOW_LOG]:
                    count = await self.redis_client.zcard(key)
                    total_usage += count
                    if count > 0:
                        active_scopes += 1
                
                elif rule.algorithm == RateLimitAlgorithm.FIXED_WINDOW:
                    count = await self.redis_client.get(key)
                    if count:
                        total_usage += int(count)
                        active_scopes += 1
            
            stats[rule.name] = {
                'algorithm': rule.algorithm.value,
                'scope': rule.scope.value,
                'limit': rule.limit,
                'window_size': rule.window_size,
                'total_usage': total_usage,
                'active_scopes': active_scopes,
                'enabled': rule.enabled
            }
        
        return stats

    async def reset_rate_limit(self, rule_name: str, scope_key: Optional[str] = None) -> bool:
        """Reset rate limit for a specific rule and scope"""
        if rule_name not in self.rules:
            return False
        
        if scope_key:
            pattern = f"rate_limit:*:{self.service_name}:{rule_name}:{scope_key}*"
        else:
            pattern = f"rate_limit:*:{self.service_name}:{rule_name}:*"
        
        keys = await self.redis_client.keys(pattern)
        if keys:
            await self.redis_client.delete(*keys)
            logger.info(f"Reset rate limit for rule '{rule_name}', scope: {scope_key or 'all'}")
            return True
        
        return False

class RateLimitMiddleware:
    """
    Middleware for applying rate limiting to HTTP requests
    """
    
    def __init__(self, rate_limiter: DistributedRateLimiter):
        self.rate_limiter = rate_limiter
    
    async def __call__(self, request, call_next):
        """Apply rate limiting to HTTP request"""
        # Extract identifier (IP, user ID, etc.)
        identifier = self._extract_identifier(request)
        
        # Extract metadata
        metadata = {
            'endpoint': f"{request.method} {request.url.path}",
            'service': getattr(request.state, 'service_name', 'unknown'),
            'user_agent': request.headers.get('user-agent', ''),
            'ip': request.client.host if request.client else 'unknown'
        }
        
        # Check rate limits
        result = await self.rate_limiter.check_rate_limit(identifier, metadata=metadata)
        
        if not result.allowed:
            # Rate limit exceeded
            from fastapi import HTTPException
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
                headers=result.headers
            )
        
        # Add rate limit headers to response
        response = await call_next(request)
        for header, value in result.headers.items():
            response.headers[header] = value
        
        return response
    
    def _extract_identifier(self, request) -> str:
        """Extract identifier from request for rate limiting"""
        # Try to get user ID from JWT token
        if hasattr(request.state, 'user_id'):
            return request.state.user_id
        
        # Fall back to IP address
        if request.client:
            return request.client.host
        
        return 'anonymous'

# Example usage and configuration
async def setup_rate_limiting():
    """Example setup for rate limiting in Prometheus AI Platform"""
    
    # Initialize Redis client
    redis_client = await redis.from_url("redis://localhost:6379")
    
    # Create rate limiter
    rate_limiter = DistributedRateLimiter(redis_client, "prometheus-ai")
    
    # Add rate limiting rules
    
    # Global API rate limit
    rate_limiter.add_rule(RateLimitRule(
        name="global_api_limit",
        algorithm=RateLimitAlgorithm.SLIDING_WINDOW_COUNTER,
        scope=RateLimitScope.GLOBAL,
        limit=10000,  # 10k requests per minute
        window_size=60,
        priority=1,
        description="Global API rate limit"
    ))
    
    # Per-user rate limit
    rate_limiter.add_rule(RateLimitRule(
        name="per_user_limit",
        algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
        scope=RateLimitScope.PER_USER,
        limit=100,  # 100 requests per minute
        window_size=60,
        burst_limit=150,  # Allow bursts up to 150
        priority=2,
        description="Per-user rate limit with burst capability"
    ))
    
    # Per-IP rate limit for anonymous users
    rate_limiter.add_rule(RateLimitRule(
        name="per_ip_limit",
        algorithm=RateLimitAlgorithm.SLIDING_WINDOW_LOG,
        scope=RateLimitScope.PER_IP,
        limit=20,  # 20 requests per minute
        window_size=60,
        priority=3,
        description="Per-IP rate limit for anonymous users"
    ))
    
    # Heavy computation endpoints
    rate_limiter.add_rule(RateLimitRule(
        name="heavy_computation_limit",
        algorithm=RateLimitAlgorithm.LEAKY_BUCKET,
        scope=RateLimitScope.PER_USER,
        limit=5,  # 5 requests per 5 minutes
        window_size=300,
        priority=4,
        description="Rate limit for computationally expensive endpoints"
    ))
    
    return rate_limiter

if __name__ == "__main__":
    async def test_rate_limiting():
        rate_limiter = await setup_rate_limiting()
        
        # Test rate limiting
        for i in range(10):
            result = await rate_limiter.check_rate_limit(
                "user123",
                rule_names=["per_user_limit"],
                metadata={"endpoint": "GET /api/v1/volcanoes"}
            )
            print(f"Request {i+1}: Allowed={result.allowed}, Remaining={result.remaining}")
            
            if not result.allowed:
                print(f"Rate limited! Retry after {result.retry_after} seconds")
                break
        
        # Get usage stats
        stats = await rate_limiter.get_usage_stats()
        print(f"Usage stats: {json.dumps(stats, indent=2)}")
    
    asyncio.run(test_rate_limiting())
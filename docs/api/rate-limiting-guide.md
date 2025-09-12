# Rate Limiting Guide - Prometheus AI Platform

## Overview

The Prometheus AI Platform implements intelligent rate limiting to ensure fair usage, prevent abuse, and maintain optimal performance for all users. This guide covers rate limiting policies, monitoring, and best practices for handling limits.

## Rate Limiting Policies

### User-Based Limits

| User Type | Requests/Hour | Requests/Minute | Burst Limit |
|-----------|---------------|-----------------|-------------|
| **Free Tier** | 100 | 10 | 20 |
| **Researcher** | 1,000 | 50 | 100 |
| **Premium** | 5,000 | 200 | 500 |
| **Enterprise** | 25,000 | 1,000 | 2,000 |
| **Service Account** | 50,000 | 2,000 | 5,000 |

### Endpoint-Specific Limits

#### Computational Endpoints (Higher Cost)

| Endpoint | Free | Researcher | Premium | Enterprise |
|----------|------|------------|---------|------------|
| `/analyze/geological` | 10/hour | 100/hour | 500/hour | 2,000/hour |
| `/simulate/physics` | 5/hour | 50/hour | 250/hour | 1,000/hour |
| `/forecast/eruption` | 20/hour | 200/hour | 1,000/hour | 5,000/hour |

#### Monitoring Endpoints (Lower Cost)

| Endpoint | Free | Researcher | Premium | Enterprise |
|----------|------|------------|---------|------------|
| `/monitor/volcano/*` | 50/hour | 500/hour | 2,500/hour | 10,000/hour |
| `/health` | Unlimited | Unlimited | Unlimited | Unlimited |
| `/metrics` | 100/hour | 1,000/hour | 5,000/hour | 25,000/hour |

### Geographic Rate Limiting

Different limits may apply based on geographic regions:

- **North America/Europe:** Standard limits
- **Asia-Pacific:** 80% of standard limits
- **Other Regions:** 60% of standard limits

## Rate Limit Headers

Every API response includes rate limiting information:

```http
HTTP/1.1 200 OK
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1641024000
X-RateLimit-Window: 3600
X-RateLimit-Policy: user-based
X-RateLimit-Scope: user:123
```

### Header Descriptions

- **X-RateLimit-Limit:** Maximum requests allowed in the current window
- **X-RateLimit-Remaining:** Requests remaining in the current window
- **X-RateLimit-Reset:** Unix timestamp when the window resets
- **X-RateLimit-Window:** Window duration in seconds
- **X-RateLimit-Policy:** Type of rate limiting applied
- **X-RateLimit-Scope:** Scope of the rate limit (user, IP, API key)

## Handling Rate Limits

### 429 Too Many Requests Response

When rate limits are exceeded:

```http
HTTP/1.1 429 Too Many Requests
Content-Type: application/json
Retry-After: 60
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1641024000

{
  "error": "Rate limit exceeded",
  "code": "RATE_LIMIT_EXCEEDED",
  "message": "You have exceeded the rate limit of 1000 requests per hour",
  "retry_after": 60,
  "limit_type": "user_hourly",
  "reset_time": "2025-01-09T11:00:00Z"
}
```

### Best Practices for Handling Limits

#### 1. Exponential Backoff

```python
import time
import random
import requests

def make_request_with_backoff(url, headers, max_retries=5):
    """Make request with exponential backoff on rate limits"""
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers)
            
            if response.status_code == 429:
                # Get retry-after header or calculate backoff
                retry_after = int(response.headers.get('Retry-After', 0))
                
                if retry_after > 0:
                    # Use server-provided retry time
                    wait_time = retry_after
                else:
                    # Calculate exponential backoff with jitter
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                
                print(f"Rate limited. Waiting {wait_time:.2f} seconds...")
                time.sleep(wait_time)
                continue
            
            # Success or other error
            return response
            
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise e
            
            # Backoff on network errors too
            wait_time = (2 ** attempt) + random.uniform(0, 1)
            time.sleep(wait_time)
    
    raise Exception(f"Max retries ({max_retries}) exceeded")

# Usage example
try:
    response = make_request_with_backoff(
        'https://api.prometheus-ai.com/api/v1/helios/analyze/geological',
        headers={'Authorization': 'Bearer your_token'}
    )
    print("Request successful:", response.json())
except Exception as e:
    print("Request failed:", e)
```

#### 2. Rate Limit Monitoring

```python
class RateLimitMonitor:
    """Monitor and track rate limit usage"""
    
    def __init__(self):
        self.limits = {}
        self.usage = {}
    
    def update_from_headers(self, headers):
        """Update rate limit info from response headers"""
        limit = int(headers.get('X-RateLimit-Limit', 0))
        remaining = int(headers.get('X-RateLimit-Remaining', 0))
        reset_time = int(headers.get('X-RateLimit-Reset', 0))
        scope = headers.get('X-RateLimit-Scope', 'default')
        
        self.limits[scope] = {
            'limit': limit,
            'remaining': remaining,
            'reset_time': reset_time,
            'usage_percent': ((limit - remaining) / limit * 100) if limit > 0 else 0
        }
    
    def should_throttle(self, scope='default', threshold=90):
        """Check if we should throttle requests"""
        if scope not in self.limits:
            return False
        
        return self.limits[scope]['usage_percent'] > threshold
    
    def time_until_reset(self, scope='default'):
        """Get seconds until rate limit resets"""
        if scope not in self.limits:
            return 0
        
        reset_time = self.limits[scope]['reset_time']
        return max(0, reset_time - time.time())
    
    def get_status(self):
        """Get current rate limit status"""
        return {
            scope: {
                'limit': info['limit'],
                'remaining': info['remaining'],
                'usage_percent': round(info['usage_percent'], 1),
                'reset_in_seconds': self.time_until_reset(scope)
            }
            for scope, info in self.limits.items()
        }

# Usage example
monitor = RateLimitMonitor()

def make_monitored_request(url, headers):
    """Make request with rate limit monitoring"""
    
    # Check if we should throttle
    if monitor.should_throttle(threshold=85):
        wait_time = monitor.time_until_reset()
        if wait_time > 0:
            print(f"Approaching rate limit. Waiting {wait_time} seconds...")
            time.sleep(wait_time)
    
    response = requests.get(url, headers=headers)
    
    # Update monitoring info
    monitor.update_from_headers(response.headers)
    
    # Log current status
    status = monitor.get_status()
    print(f"Rate limit status: {status}")
    
    return response
```

#### 3. Request Queuing and Batching

```python
import asyncio
from collections import deque
from datetime import datetime, timedelta

class RequestQueue:
    """Queue requests to respect rate limits"""
    
    def __init__(self, requests_per_minute=50):
        self.requests_per_minute = requests_per_minute
        self.request_times = deque()
        self.queue = asyncio.Queue()
        self.processing = False
    
    async def add_request(self, request_func, *args, **kwargs):
        """Add request to queue"""
        await self.queue.put((request_func, args, kwargs))
        
        if not self.processing:
            asyncio.create_task(self.process_queue())
    
    async def process_queue(self):
        """Process queued requests respecting rate limits"""
        self.processing = True
        
        while not self.queue.empty():
            # Clean old request times
            now = datetime.now()
            minute_ago = now - timedelta(minutes=1)
            
            while self.request_times and self.request_times[0] < minute_ago:
                self.request_times.popleft()
            
            # Check if we can make a request
            if len(self.request_times) >= self.requests_per_minute:
                # Wait until we can make another request
                sleep_time = 60 - (now - self.request_times[0]).total_seconds()
                await asyncio.sleep(max(0, sleep_time))
                continue
            
            # Make the request
            request_func, args, kwargs = await self.queue.get()
            
            try:
                result = await request_func(*args, **kwargs)
                self.request_times.append(now)
                print(f"Request completed: {result}")
            except Exception as e:
                print(f"Request failed: {e}")
            
            # Small delay between requests
            await asyncio.sleep(0.1)
        
        self.processing = False

# Usage example
async def api_request(endpoint, data):
    """Example API request function"""
    # Simulate API call
    await asyncio.sleep(0.5)
    return f"Response from {endpoint}"

async def main():
    queue = RequestQueue(requests_per_minute=30)
    
    # Add multiple requests
    for i in range(50):
        await queue.add_request(
            api_request, 
            f"/api/endpoint/{i}", 
            {"data": f"request_{i}"}
        )
    
    # Wait for all requests to complete
    while queue.processing or not queue.queue.empty():
        await asyncio.sleep(1)

# Run example
# asyncio.run(main())
```

## Advanced Rate Limiting Features

### 1. Burst Allowance

The platform allows short bursts above the sustained rate:

```python
def check_burst_allowance(headers):
    """Check if burst capacity is available"""
    burst_limit = int(headers.get('X-RateLimit-Burst', 0))
    burst_remaining = int(headers.get('X-RateLimit-Burst-Remaining', 0))
    
    return {
        'burst_available': burst_remaining > 0,
        'burst_capacity': burst_limit,
        'burst_remaining': burst_remaining
    }
```

### 2. Adaptive Rate Limiting

The system adjusts limits based on:

- **System Load:** Lower limits during high load
- **User Behavior:** Higher limits for well-behaved clients
- **Time of Day:** Different limits for peak/off-peak hours
- **Geographic Region:** Adjusted for regional capacity

### 3. Priority Queuing

Requests are prioritized based on:

1. **Emergency Alerts:** Volcanic eruption warnings
2. **Real-time Monitoring:** Active monitoring systems
3. **Interactive Analysis:** User-initiated analysis
4. **Batch Processing:** Background computations

## Rate Limit Optimization

### 1. Caching Strategies

```python
import hashlib
import json
from functools import wraps

def cache_response(ttl_seconds=300):
    """Cache API responses to reduce rate limit usage"""
    cache = {}
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function arguments
            cache_key = hashlib.md5(
                json.dumps([args, kwargs], sort_keys=True).encode()
            ).hexdigest()
            
            # Check cache
            if cache_key in cache:
                cached_data, timestamp = cache[cache_key]
                if time.time() - timestamp < ttl_seconds:
                    print(f"Cache hit for {func.__name__}")
                    return cached_data
            
            # Make API call
            result = func(*args, **kwargs)
            
            # Cache result
            cache[cache_key] = (result, time.time())
            
            return result
        
        return wrapper
    return decorator

@cache_response(ttl_seconds=600)  # Cache for 10 minutes
def get_volcano_status(volcano_id):
    """Get volcano status with caching"""
    response = requests.get(
        f'https://api.prometheus-ai.com/api/v1/volcano/{volcano_id}/status',
        headers={'Authorization': f'Bearer {token}'}
    )
    return response.json()
```

### 2. Request Optimization

```python
def optimize_geological_analysis(data_requests):
    """Batch multiple data requests into single API call"""
    
    # Group requests by spatial proximity
    grouped_requests = group_by_spatial_proximity(data_requests)
    
    results = []
    for group in grouped_requests:
        # Combine spatial extents
        combined_extent = merge_spatial_extents([req.spatial_extent for req in group])
        
        # Make single API call for the group
        response = requests.post(
            'https://api.prometheus-ai.com/api/v1/helios/analyze/geological',
            headers={'Authorization': f'Bearer {token}'},
            json={
                'data_type': group[0].data_type,
                'spatial_extent': combined_extent,
                'parameters': merge_parameters([req.parameters for req in group])
            }
        )
        
        # Split results back to individual requests
        group_results = split_analysis_results(response.json(), group)
        results.extend(group_results)
    
    return results

def group_by_spatial_proximity(requests, max_distance_km=50):
    """Group requests by spatial proximity"""
    # Implementation for spatial clustering
    pass

def merge_spatial_extents(extents):
    """Merge multiple spatial extents into one"""
    min_lat = min(ext['min_lat'] for ext in extents)
    max_lat = max(ext['max_lat'] for ext in extents)
    min_lon = min(ext['min_lon'] for ext in extents)
    max_lon = max(ext['max_lon'] for ext in extents)
    
    return {
        'min_lat': min_lat,
        'max_lat': max_lat,
        'min_lon': min_lon,
        'max_lon': max_lon
    }
```

## Monitoring and Alerting

### 1. Rate Limit Dashboard

```python
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class RateLimitDashboard:
    """Dashboard for monitoring rate limit usage"""
    
    def __init__(self):
        self.usage_history = []
    
    def log_usage(self, headers):
        """Log rate limit usage"""
        self.usage_history.append({
            'timestamp': datetime.now(),
            'limit': int(headers.get('X-RateLimit-Limit', 0)),
            'remaining': int(headers.get('X-RateLimit-Remaining', 0)),
            'usage_percent': self.calculate_usage_percent(headers)
        })
    
    def calculate_usage_percent(self, headers):
        """Calculate usage percentage"""
        limit = int(headers.get('X-RateLimit-Limit', 0))
        remaining = int(headers.get('X-RateLimit-Remaining', 0))
        
        if limit == 0:
            return 0
        
        return ((limit - remaining) / limit) * 100
    
    def plot_usage_trend(self, hours=24):
        """Plot rate limit usage trend"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_data = [
            entry for entry in self.usage_history 
            if entry['timestamp'] > cutoff_time
        ]
        
        if not recent_data:
            print("No data available for the specified time period")
            return
        
        timestamps = [entry['timestamp'] for entry in recent_data]
        usage_percentages = [entry['usage_percent'] for entry in recent_data]
        
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, usage_percentages, marker='o')
        plt.title(f'Rate Limit Usage - Last {hours} Hours')
        plt.xlabel('Time')
        plt.ylabel('Usage Percentage (%)')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def get_usage_stats(self, hours=24):
        """Get usage statistics"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_data = [
            entry for entry in self.usage_history 
            if entry['timestamp'] > cutoff_time
        ]
        
        if not recent_data:
            return None
        
        usage_percentages = [entry['usage_percent'] for entry in recent_data]
        
        return {
            'average_usage': sum(usage_percentages) / len(usage_percentages),
            'max_usage': max(usage_percentages),
            'min_usage': min(usage_percentages),
            'current_usage': usage_percentages[-1] if usage_percentages else 0,
            'data_points': len(recent_data)
        }
```

### 2. Alerting System

```python
import smtplib
from email.mime.text import MimeText

class RateLimitAlerting:
    """Alert system for rate limit monitoring"""
    
    def __init__(self, email_config):
        self.email_config = email_config
        self.alert_thresholds = {
            'warning': 75,    # 75% usage
            'critical': 90,   # 90% usage
            'emergency': 95   # 95% usage
        }
        self.last_alerts = {}
    
    def check_and_alert(self, usage_percent, scope='default'):
        """Check usage and send alerts if needed"""
        current_time = datetime.now()
        
        # Determine alert level
        alert_level = None
        if usage_percent >= self.alert_thresholds['emergency']:
            alert_level = 'emergency'
        elif usage_percent >= self.alert_thresholds['critical']:
            alert_level = 'critical'
        elif usage_percent >= self.alert_thresholds['warning']:
            alert_level = 'warning'
        
        if not alert_level:
            return
        
        # Check if we've already sent this alert recently
        alert_key = f"{scope}:{alert_level}"
        if alert_key in self.last_alerts:
            time_since_last = current_time - self.last_alerts[alert_key]
            if time_since_last < timedelta(minutes=15):  # Don't spam alerts
                return
        
        # Send alert
        self.send_alert(alert_level, usage_percent, scope)
        self.last_alerts[alert_key] = current_time
    
    def send_alert(self, level, usage_percent, scope):
        """Send rate limit alert"""
        subject = f"Rate Limit Alert - {level.upper()}"
        
        message = f"""
Rate Limit Alert - Prometheus AI Platform

Alert Level: {level.upper()}
Scope: {scope}
Current Usage: {usage_percent:.1f}%
Threshold: {self.alert_thresholds[level]}%
Time: {datetime.now().isoformat()}

Recommended Actions:
- Implement request throttling
- Review and optimize API usage patterns
- Consider upgrading to higher rate limit tier
- Check for potential abuse or runaway processes

Dashboard: https://dashboard.prometheus-ai.com/rate-limits
Support: support@prometheus-ai.com
"""
        
        try:
            msg = MimeText(message)
            msg['Subject'] = subject
            msg['From'] = self.email_config['from']
            msg['To'] = self.email_config['to']
            
            with smtplib.SMTP(self.email_config['smtp_server']) as server:
                server.starttls()
                server.login(self.email_config['username'], self.email_config['password'])
                server.send_message(msg)
            
            print(f"Alert sent: {subject}")
            
        except Exception as e:
            print(f"Failed to send alert: {e}")
```

## Upgrading Rate Limits

### 1. Temporary Limit Increases

Request temporary increases for special events:

```bash
curl -X POST "https://api.prometheus-ai.com/auth/rate-limit/increase" \
  -H "Authorization: Bearer <admin_token>" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "multiplier": 2.0,
    "duration_hours": 24,
    "reason": "Emergency volcanic monitoring - Mount Vesuvius activity"
  }'
```

### 2. Plan Upgrades

Upgrade your plan for permanent increases:

```bash
curl -X POST "https://api.prometheus-ai.com/billing/upgrade" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "plan": "premium",
    "billing_cycle": "monthly"
  }'
```

## Troubleshooting

### Common Issues

1. **Unexpected Rate Limits**
   - Check for multiple API keys from same account
   - Verify geographic rate limiting policies
   - Review shared IP address limitations

2. **Inconsistent Limits**
   - Different endpoints have different limits
   - Burst vs. sustained rate differences
   - Time-based limit variations

3. **Rate Limit Not Resetting**
   - Check system clock synchronization
   - Verify timezone handling
   - Contact support for backend issues

### Debug Tools

```python
def debug_rate_limits(response):
    """Debug rate limit information"""
    headers = response.headers
    
    print("=== Rate Limit Debug Info ===")
    print(f"Status Code: {response.status_code}")
    print(f"Limit: {headers.get('X-RateLimit-Limit', 'Not set')}")
    print(f"Remaining: {headers.get('X-RateLimit-Remaining', 'Not set')}")
    print(f"Reset: {headers.get('X-RateLimit-Reset', 'Not set')}")
    print(f"Window: {headers.get('X-RateLimit-Window', 'Not set')}")
    print(f"Policy: {headers.get('X-RateLimit-Policy', 'Not set')}")
    print(f"Scope: {headers.get('X-RateLimit-Scope', 'Not set')}")
    
    if response.status_code == 429:
        print(f"Retry After: {headers.get('Retry-After', 'Not set')}")
        print("Rate limit exceeded!")
    
    print("============================")
```

## Support

For rate limiting issues:

- **Documentation:** https://docs.prometheus-ai.com/rate-limits
- **Support Email:** rate-limits@prometheus-ai.com
- **Upgrade Requests:** billing@prometheus-ai.com
- **Emergency Increases:** emergency@prometheus-ai.com
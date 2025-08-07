# Prometheus Remote Monitoring & Debugging System - Deployment Guide

## Overview
This guide provides complete instructions for deploying the Prometheus remote monitoring and debugging system for http://20.80.89.29:8080/.

## System Components

### 1. Client-Side Monitoring Agent
- **File**: `remote_monitor.js`
- **Purpose**: Captures and reports errors, performance metrics, and system health
- **Features**: 
  - JavaScript error capture with stack traces
  - Network request monitoring
  - Performance metrics collection
  - Console log capture
  - Automatic error recovery

### 2. Global Function Utility
- **File**: `global_functions_utility.js`
- **Purpose**: Ensures critical functions are available globally
- **Features**: Fallback implementations for missing functions

### 3. Automatic Recovery System
- **File**: `automatic_recovery_system.js`
- **Purpose**: Advanced self-healing mechanisms
- **Features**: Health monitoring, automatic recovery, resource fallbacks

### 4. Debug Dashboard
- **File**: `debug_dashboard.html`
- **Purpose**: Real-time error visualization and monitoring
- **Features**: Live error feed, charts, system statistics

### 5. Error Collection Server
- **File**: `error_collection_server.py`
- **Purpose**: Backend service for collecting and storing errors
- **Features**: REST API, SQLite database, statistics

## Installation Instructions

### Step 1: Server Setup

#### 1.1 Install Python Dependencies
```bash
cd /path/to/prometheus
pip install -r requirements-monitoring.txt
```

#### 1.2 Start the Error Collection Server
```bash
# Development mode
python error_collection_server.py --debug

# Production mode
python error_collection_server.py --host 20.80.89.29 --port 8080

# With custom database location
python error_collection_server.py --db-path /path/to/errors.db
```

#### 1.3 Verify Server is Running
```bash
curl http://20.80.89.29:8080/api/health
```
Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2024-08-06T...",
  "version": "1.0.0"
}
```

### Step 2: Client-Side Integration

#### 2.1 Add Monitoring Scripts to HTML Files
Add these script tags to all HTML files that need monitoring:

```html
<!-- Add before closing </head> tag -->
<script src="remote_monitor.js"></script>
<script src="global_functions_utility.js"></script>
<script src="automatic_recovery_system.js"></script>
```

#### 2.2 Update Existing HTML Files
The following files need to be updated:

1. **volcanic_system_simulator.html**
2. **advanced_tectonic_simulation.html**
3. **advanced_insar_analysis.html**
4. **earthquake_stress_visualization.html**
5. **test_integration.html**

Add the monitoring scripts after the existing script inclusions:

```html
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
<script src="webgl_fix_and_minimize.js"></script>
<script src="holocene_volcanoes.js"></script>
<!-- Add monitoring scripts here -->
<script src="remote_monitor.js"></script>
<script src="global_functions_utility.js"></script>
<script src="automatic_recovery_system.js"></script>
```

#### 2.3 Fixed holocene_volcanoes.js
The `holocene_volcanoes.js` file has been updated to prevent redeclaration errors. No additional changes needed.

### Step 3: Dashboard Access

#### 3.1 Access Debug Dashboard
Open in browser: http://20.80.89.29:8080/

The dashboard provides:
- Real-time error monitoring
- System health statistics
- Error trend visualization
- Performance metrics
- System logs

#### 3.2 Dashboard Features
- **Error Filtering**: Filter by severity, type, or text search
- **Export Data**: Export error data as JSON
- **Pause/Resume**: Control real-time updates
- **Clear Errors**: Reset error history

### Step 4: Configuration Options

#### 4.1 Monitor Configuration
Edit `remote_monitor.js` to customize:

```javascript
window.prometheusMonitor = new PrometheusRemoteMonitor({
    endpoint: 'http://20.80.89.29:8080/api/errors',  // Server endpoint
    maxErrors: 1000,                                  // Buffer size
    batchSize: 10,                                    // Batch upload size
    flushInterval: 30000,                            // Upload frequency (ms)
    retryAttempts: 3,                                // Network retry attempts
    enableRecovery: true,                            // Enable auto-recovery
    debugMode: true,                                 // Enable debug logging
    userId: 'admin'                                  // User identification
});
```

#### 4.2 Recovery System Configuration
Edit `automatic_recovery_system.js`:

```javascript
window.prometheusRecovery = new PrometheusRecoverySystem({
    maxRecoveryAttempts: 3,        // Max recovery tries
    recoveryDelay: 1000,           // Delay between attempts (ms)
    healthCheckInterval: 30000,     // Health check frequency (ms)
    enableAutoReload: true,        // Enable page reload as last resort
    debugMode: true                // Enable debug logging
});
```

### Step 5: API Endpoints

The error collection server provides these endpoints:

#### 5.1 Error Collection
```
POST /api/errors
Content-Type: application/json

{
  "errors": [...],
  "metadata": {
    "sessionId": "session_abc123",
    "userId": "user123",
    "userAgent": "...",
    "timestamp": "2024-08-06T..."
  }
}
```

#### 5.2 Error Retrieval
```
GET /api/errors?limit=100&offset=0&severity=high&since=2024-08-06T00:00:00Z
```

#### 5.3 System Statistics
```
GET /api/stats
```

#### 5.4 Health Check
```
GET /api/health
```

### Step 6: Monitoring Verification

#### 6.1 Test Error Capture
Add this to any monitored page to test error capture:

```javascript
// Test JavaScript error
setTimeout(() => {
    throw new Error('Test error for monitoring system');
}, 5000);

// Test custom error
window.prometheusMonitor.addCustomError('test_error', 'This is a test error');
```

#### 6.2 Verify Dashboard Updates
1. Open http://20.80.89.29:8080/
2. Trigger test errors
3. Verify errors appear in dashboard within 30 seconds

#### 6.3 Check Server Logs
```bash
tail -f prometheus_errors.log
```

### Step 7: Production Deployment

#### 7.1 Use Production WSGI Server
```bash
# Install Gunicorn
pip install gunicorn

# Run with Gunicorn
gunicorn -w 4 -b 20.80.89.29:8080 error_collection_server:app
```

#### 7.2 Setup Process Management
Create systemd service file `/etc/systemd/system/prometheus-monitoring.service`:

```ini
[Unit]
Description=Prometheus Monitoring Server
After=network.target

[Service]
Type=exec
User=prometheus
WorkingDirectory=/path/to/prometheus
ExecStart=/usr/bin/gunicorn -w 4 -b 20.80.89.29:8080 error_collection_server:app
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable prometheus-monitoring
sudo systemctl start prometheus-monitoring
sudo systemctl status prometheus-monitoring
```

#### 7.3 Setup Log Rotation
Create `/etc/logrotate.d/prometheus-monitoring`:

```
/path/to/prometheus/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 prometheus prometheus
}
```

### Step 8: Security Considerations

#### 8.1 Network Security
- Configure firewall to allow only necessary traffic
- Use HTTPS in production (configure reverse proxy)
- Implement rate limiting for API endpoints

#### 8.2 Data Privacy
- Sanitize sensitive data before logging
- Implement data retention policies
- Consider GDPR compliance for user data

#### 8.3 Access Control
- Implement authentication for dashboard access
- Use environment variables for sensitive configuration
- Regular security updates

### Step 9: Maintenance

#### 9.1 Database Maintenance
The server automatically cleans up:
- Errors older than 30 days
- Sessions older than 7 days

#### 9.2 Log Monitoring
Monitor these log files:
- Application logs: `prometheus_monitoring.log`
- Error logs: `error_collection.log`
- System logs: `/var/log/syslog`

#### 9.3 Health Monitoring
Set up external monitoring for:
- Server availability
- Database size
- Memory usage
- CPU usage

### Step 10: Troubleshooting

#### 10.1 Common Issues

**Error: "HOLOCENE_VOLCANOES is not defined"**
- Solution: Ensure `holocene_volcanoes.js` is loaded before other scripts
- The updated file includes existence checks

**Error: "switchView is not a function"**
- Solution: `global_functions_utility.js` provides fallback implementations
- Verify script load order

**Dashboard not showing errors**
- Check network connectivity to server
- Verify API endpoint configuration
- Check browser console for errors

**High memory usage**
- The recovery system includes memory cleanup
- Check for resource leaks in Three.js scenes
- Monitor browser memory usage

#### 10.2 Debug Commands

Test function availability:
```javascript
window.debugGlobalFunctions();
```

Check monitoring status:
```javascript
console.log(window.prometheusMonitor.getErrorStats());
```

Check recovery system health:
```javascript
console.log(window.prometheusRecovery.getSystemHealth());
```

Force health check:
```javascript
window.prometheusRecovery.forceRecoveryCheck();
```

#### 10.3 Manual Recovery

If automatic recovery fails:

1. **Reload page**: `window.location.reload()`
2. **Clear cache**: Hard refresh (Ctrl+Shift+R)
3. **Reset monitoring**: Reinitialize monitoring scripts
4. **Check server**: Verify error collection server is running

### Step 11: Performance Optimization

#### 11.1 Client-Side
- Error buffer size: Reduce for lower memory usage
- Flush interval: Increase for better performance
- Debug mode: Disable in production

#### 11.2 Server-Side
- Database indexing: Already optimized for common queries
- Connection pooling: Use for high-traffic scenarios
- Caching: Implement Redis for session data

### Step 12: Monitoring Metrics

#### 12.1 Key Metrics to Track
- Error rate per page
- User session duration
- Recovery success rate
- Network failure rate
- Memory usage trends

#### 12.2 Alerting Setup
Configure alerts for:
- Error rate > 10 errors/minute
- Memory usage > 80%
- Network connectivity issues
- Server downtime

## Conclusion

The Prometheus Remote Monitoring & Debugging System provides comprehensive error tracking, automatic recovery, and real-time monitoring for the Prometheus platform. Following this deployment guide ensures robust monitoring and improved system reliability.

For additional support or questions, refer to the system logs and debug utilities provided with each component.
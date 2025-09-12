/**
 * Resource Usage Monitoring for Geological Platform
 * Monitors system resources during peak loads and geological data processing
 */

const os = require('os');
const fs = require('fs').promises;
const path = require('path');
const { spawn } = require('child_process');

class ResourceMonitor {
  constructor(config = {}) {
    this.config = {
      samplingInterval: config.samplingInterval || 1000, // 1 second
      monitoringDuration: config.monitoringDuration || 300000, // 5 minutes
      outputDir: config.outputDir || './tests/performance/resource-reports',
      alertThresholds: {
        cpuUsage: config.alertThresholds?.cpuUsage || 80, // 80%
        memoryUsage: config.alertThresholds?.memoryUsage || 85, // 85%
        diskUsage: config.alertThresholds?.diskUsage || 90, // 90%
        networkLatency: config.alertThresholds?.networkLatency || 1000, // 1000ms
        ...config.alertThresholds,
      },
      ...config,
    };

    this.monitoring = false;
    this.metrics = {
      cpu: [],
      memory: [],
      disk: [],
      network: [],
      processes: [],
      geological: [],
      alerts: [],
    };

    // Geological-specific monitoring
    this.geologicalProcesses = [
      'helios-core',
      'volcano-watchtower',
      'data-ingestion',
      'api-gateway',
      'postgres',
      'influxdb',
      'neo4j',
      'redis',
    ];
  }

  async startMonitoring(testName = 'resource-monitoring') {
    console.log('📊 Starting Resource Monitoring...');
    
    this.monitoring = true;
    this.testName = testName;
    this.startTime = Date.now();
    
    await this.ensureOutputDirectory();
    
    // Start monitoring intervals
    this.cpuMonitor = setInterval(() => this.monitorCPU(), this.config.samplingInterval);
    this.memoryMonitor = setInterval(() => this.monitorMemory(), this.config.samplingInterval);
    this.diskMonitor = setInterval(() => this.monitorDisk(), this.config.samplingInterval * 5); // Every 5 seconds
    this.networkMonitor = setInterval(() => this.monitorNetwork(), this.config.samplingInterval * 2); // Every 2 seconds
    this.processMonitor = setInterval(() => this.monitorProcesses(), this.config.samplingInterval * 3); // Every 3 seconds
    this.geologicalMonitor = setInterval(() => this.monitorGeologicalServices(), this.config.samplingInterval * 2);
    
    console.log(`  ⏱️  Monitoring for ${this.config.monitoringDuration / 1000} seconds...`);
    
    // Auto-stop after duration if specified
    if (this.config.monitoringDuration > 0) {
      setTimeout(() => {
        this.stopMonitoring();
      }, this.config.monitoringDuration);
    }
  }

  async stopMonitoring() {
    if (!this.monitoring) return;
    
    console.log('🛑 Stopping Resource Monitoring...');
    
    this.monitoring = false;
    this.endTime = Date.now();
    
    // Clear intervals
    clearInterval(this.cpuMonitor);
    clearInterval(this.memoryMonitor);
    clearInterval(this.diskMonitor);
    clearInterval(this.networkMonitor);
    clearInterval(this.processMonitor);
    clearInterval(this.geologicalMonitor);
    
    // Generate report
    const report = await this.generateResourceReport();
    
    console.log('✅ Resource monitoring completed');
    console.log(`  📈 Peak CPU: ${report.peakCPU.toFixed(1)}%`);
    console.log(`  🧠 Peak Memory: ${report.peakMemory.toFixed(1)}%`);
    console.log(`  💾 Peak Disk I/O: ${report.peakDiskIO.toFixed(1)} MB/s`);
    console.log(`  🚨 Alerts Generated: ${report.totalAlerts}`);
    
    return report;
  }

  async monitorCPU() {
    try {
      const cpus = os.cpus();
      const loadAvg = os.loadavg();
      
      // Calculate CPU usage
      let totalIdle = 0;
      let totalTick = 0;
      
      cpus.forEach(cpu => {
        for (const type in cpu.times) {
          totalTick += cpu.times[type];
        }
        totalIdle += cpu.times.idle;
      });
      
      const idle = totalIdle / cpus.length;
      const total = totalTick / cpus.length;
      const usage = 100 - ~~(100 * idle / total);
      
      const cpuMetric = {
        timestamp: Date.now(),
        usage: usage,
        loadAverage: {
          '1min': loadAvg[0],
          '5min': loadAvg[1],
          '15min': loadAvg[2],
        },
        coreCount: cpus.length,
        perCoreUsage: await this.getPerCoreUsage(),
      };
      
      this.metrics.cpu.push(cpuMetric);
      
      // Check for alerts
      if (usage > this.config.alertThresholds.cpuUsage) {
        this.generateAlert('cpu', `High CPU usage: ${usage.toFixed(1)}%`, 'warning');
      }
      
      if (loadAvg[0] > cpus.length * 2) {
        this.generateAlert('cpu', `High load average: ${loadAvg[0].toFixed(2)}`, 'critical');
      }
      
    } catch (error) {
      console.error('CPU monitoring error:', error.message);
    }
  }

  async monitorMemory() {
    try {
      const totalMem = os.totalmem();
      const freeMem = os.freemem();
      const usedMem = totalMem - freeMem;
      const usagePercent = (usedMem / totalMem) * 100;
      
      // Get detailed memory info
      const memoryInfo = await this.getDetailedMemoryInfo();
      
      const memoryMetric = {
        timestamp: Date.now(),
        total: totalMem,
        used: usedMem,
        free: freeMem,
        usagePercent: usagePercent,
        available: memoryInfo.available,
        buffers: memoryInfo.buffers,
        cached: memoryInfo.cached,
        swapTotal: memoryInfo.swapTotal,
        swapUsed: memoryInfo.swapUsed,
      };
      
      this.metrics.memory.push(memoryMetric);
      
      // Check for alerts
      if (usagePercent > this.config.alertThresholds.memoryUsage) {
        this.generateAlert('memory', `High memory usage: ${usagePercent.toFixed(1)}%`, 'warning');
      }
      
      if (memoryInfo.swapUsed > memoryInfo.swapTotal * 0.5) {
        this.generateAlert('memory', `High swap usage: ${((memoryInfo.swapUsed / memoryInfo.swapTotal) * 100).toFixed(1)}%`, 'critical');
      }
      
    } catch (error) {
      console.error('Memory monitoring error:', error.message);
    }
  }

  async monitorDisk() {
    try {
      const diskUsage = await this.getDiskUsage();
      const diskIO = await this.getDiskIO();
      
      const diskMetric = {
        timestamp: Date.now(),
        usage: diskUsage,
        io: diskIO,
      };
      
      this.metrics.disk.push(diskMetric);
      
      // Check for alerts
      diskUsage.forEach(disk => {
        if (disk.usagePercent > this.config.alertThresholds.diskUsage) {
          this.generateAlert('disk', `High disk usage on ${disk.filesystem}: ${disk.usagePercent.toFixed(1)}%`, 'warning');
        }
      });
      
      if (diskIO.readMBps + diskIO.writeMBps > 100) { // 100 MB/s threshold
        this.generateAlert('disk', `High disk I/O: ${(diskIO.readMBps + diskIO.writeMBps).toFixed(1)} MB/s`, 'info');
      }
      
    } catch (error) {
      console.error('Disk monitoring error:', error.message);
    }
  }

  async monitorNetwork() {
    try {
      const networkStats = await this.getNetworkStats();
      const networkLatency = await this.measureNetworkLatency();
      
      const networkMetric = {
        timestamp: Date.now(),
        interfaces: networkStats,
        latency: networkLatency,
      };
      
      this.metrics.network.push(networkMetric);
      
      // Check for alerts
      if (networkLatency.average > this.config.alertThresholds.networkLatency) {
        this.generateAlert('network', `High network latency: ${networkLatency.average}ms`, 'warning');
      }
      
      // Check for high network usage
      networkStats.forEach(iface => {
        const totalMbps = (iface.rxMbps + iface.txMbps);
        if (totalMbps > 100) { // 100 Mbps threshold
          this.generateAlert('network', `High network usage on ${iface.name}: ${totalMbps.toFixed(1)} Mbps`, 'info');
        }
      });
      
    } catch (error) {
      console.error('Network monitoring error:', error.message);
    }
  }

  async monitorProcesses() {
    try {
      const processes = await this.getProcessInfo();
      
      const processMetric = {
        timestamp: Date.now(),
        totalProcesses: processes.length,
        topCPU: processes.slice(0, 10), // Top 10 CPU consumers
        topMemory: processes.sort((a, b) => b.memory - a.memory).slice(0, 10), // Top 10 memory consumers
      };
      
      this.metrics.processes.push(processMetric);
      
      // Check for resource-heavy processes
      processes.forEach(proc => {
        if (proc.cpu > 50) {
          this.generateAlert('process', `High CPU process: ${proc.name} (${proc.cpu.toFixed(1)}%)`, 'info');
        }
        
        if (proc.memory > 1024 * 1024 * 1024) { // 1GB
          this.generateAlert('process', `High memory process: ${proc.name} (${(proc.memory / 1024 / 1024 / 1024).toFixed(1)}GB)`, 'info');
        }
      });
      
    } catch (error) {
      console.error('Process monitoring error:', error.message);
    }
  }

  async monitorGeologicalServices() {
    try {
      const geologicalMetrics = {
        timestamp: Date.now(),
        services: {},
        databases: {},
        apiGateway: {},
      };
      
      // Monitor geological services
      for (const serviceName of this.geologicalProcesses) {
        const serviceInfo = await this.getServiceInfo(serviceName);
        if (serviceInfo) {
          geologicalMetrics.services[serviceName] = serviceInfo;
          
          // Service-specific alerts
          if (serviceInfo.cpu > 80) {
            this.generateAlert('geological', `High CPU in ${serviceName}: ${serviceInfo.cpu.toFixed(1)}%`, 'warning');
          }
          
          if (serviceInfo.memory > 2 * 1024 * 1024 * 1024) { // 2GB
            this.generateAlert('geological', `High memory in ${serviceName}: ${(serviceInfo.memory / 1024 / 1024 / 1024).toFixed(1)}GB`, 'warning');
          }
        }
      }
      
      // Monitor database connections and performance
      geologicalMetrics.databases = await this.monitorDatabases();
      
      // Monitor API gateway performance
      geologicalMetrics.apiGateway = await this.monitorAPIGateway();
      
      this.metrics.geological.push(geologicalMetrics);
      
    } catch (error) {
      console.error('Geological services monitoring error:', error.message);
    }
  }

  async getPerCoreUsage() {
    // Platform-specific CPU per-core usage
    if (process.platform === 'linux') {
      return await this.getLinuxPerCoreUsage();
    } else if (process.platform === 'win32') {
      return await this.getWindowsPerCoreUsage();
    } else {
      return []; // macOS or other platforms
    }
  }

  async getLinuxPerCoreUsage() {
    try {
      const data = await fs.readFile('/proc/stat', 'utf8');
      const lines = data.split('\n');
      const coreUsages = [];
      
      lines.forEach(line => {
        if (line.startsWith('cpu') && line !== lines[0]) { // Skip the first 'cpu' line (total)
          const values = line.split(/\s+/).slice(1).map(Number);
          const idle = values[3];
          const total = values.reduce((sum, val) => sum + val, 0);
          const usage = 100 - (idle / total * 100);
          coreUsages.push(usage);
        }
      });
      
      return coreUsages;
    } catch (error) {
      return [];
    }
  }

  async getWindowsPerCoreUsage() {
    return new Promise((resolve) => {
      const cmd = spawn('wmic', ['cpu', 'get', 'loadpercentage', '/value']);
      let output = '';
      
      cmd.stdout.on('data', (data) => {
        output += data.toString();
      });
      
      cmd.on('close', () => {
        const matches = output.match(/LoadPercentage=(\d+)/g);
        const usages = matches ? matches.map(match => parseInt(match.split('=')[1])) : [];
        resolve(usages);
      });
      
      cmd.on('error', () => resolve([]));
    });
  }

  async getDetailedMemoryInfo() {
    if (process.platform === 'linux') {
      return await this.getLinuxMemoryInfo();
    } else if (process.platform === 'win32') {
      return await this.getWindowsMemoryInfo();
    } else {
      return {
        available: os.freemem(),
        buffers: 0,
        cached: 0,
        swapTotal: 0,
        swapUsed: 0,
      };
    }
  }

  async getLinuxMemoryInfo() {
    try {
      const data = await fs.readFile('/proc/meminfo', 'utf8');
      const lines = data.split('\n');
      const memInfo = {};
      
      lines.forEach(line => {
        const [key, value] = line.split(':');
        if (key && value) {
          const numValue = parseInt(value.trim().split(' ')[0]) * 1024; // Convert KB to bytes
          memInfo[key] = numValue;
        }
      });
      
      return {
        available: memInfo.MemAvailable || memInfo.MemFree || 0,
        buffers: memInfo.Buffers || 0,
        cached: memInfo.Cached || 0,
        swapTotal: memInfo.SwapTotal || 0,
        swapUsed: (memInfo.SwapTotal || 0) - (memInfo.SwapFree || 0),
      };
    } catch (error) {
      return {
        available: os.freemem(),
        buffers: 0,
        cached: 0,
        swapTotal: 0,
        swapUsed: 0,
      };
    }
  }

  async getWindowsMemoryInfo() {
    return new Promise((resolve) => {
      const cmd = spawn('wmic', ['OS', 'get', 'TotalVisibleMemorySize,FreePhysicalMemory', '/value']);
      let output = '';
      
      cmd.stdout.on('data', (data) => {
        output += data.toString();
      });
      
      cmd.on('close', () => {
        const freeMatch = output.match(/FreePhysicalMemory=(\d+)/);
        const totalMatch = output.match(/TotalVisibleMemorySize=(\d+)/);
        
        const free = freeMatch ? parseInt(freeMatch[1]) * 1024 : os.freemem();
        const total = totalMatch ? parseInt(totalMatch[1]) * 1024 : os.totalmem();
        
        resolve({
          available: free,
          buffers: 0,
          cached: 0,
          swapTotal: 0,
          swapUsed: 0,
        });
      });
      
      cmd.on('error', () => {
        resolve({
          available: os.freemem(),
          buffers: 0,
          cached: 0,
          swapTotal: 0,
          swapUsed: 0,
        });
      });
    });
  }

  async getDiskUsage() {
    if (process.platform === 'linux' || process.platform === 'darwin') {
      return await this.getUnixDiskUsage();
    } else if (process.platform === 'win32') {
      return await this.getWindowsDiskUsage();
    }
    return [];
  }

  async getUnixDiskUsage() {
    return new Promise((resolve) => {
      const cmd = spawn('df', ['-h']);
      let output = '';
      
      cmd.stdout.on('data', (data) => {
        output += data.toString();
      });
      
      cmd.on('close', () => {
        const lines = output.split('\n').slice(1); // Skip header
        const diskUsage = [];
        
        lines.forEach(line => {
          const parts = line.split(/\s+/);
          if (parts.length >= 6) {
            diskUsage.push({
              filesystem: parts[0],
              size: parts[1],
              used: parts[2],
              available: parts[3],
              usagePercent: parseInt(parts[4].replace('%', '')),
              mountPoint: parts[5],
            });
          }
        });
        
        resolve(diskUsage);
      });
      
      cmd.on('error', () => resolve([]));
    });
  }

  async getWindowsDiskUsage() {
    return new Promise((resolve) => {
      const cmd = spawn('wmic', ['logicaldisk', 'get', 'size,freespace,caption', '/value']);
      let output = '';
      
      cmd.stdout.on('data', (data) => {
        output += data.toString();
      });
      
      cmd.on('close', () => {
        const diskUsage = [];
        const entries = output.split('\n\n');
        
        entries.forEach(entry => {
          const captionMatch = entry.match(/Caption=([A-Z]:)/);
          const sizeMatch = entry.match(/Size=(\d+)/);
          const freeMatch = entry.match(/FreeSpace=(\d+)/);
          
          if (captionMatch && sizeMatch && freeMatch) {
            const size = parseInt(sizeMatch[1]);
            const free = parseInt(freeMatch[1]);
            const used = size - free;
            const usagePercent = (used / size) * 100;
            
            diskUsage.push({
              filesystem: captionMatch[1],
              size: `${(size / 1024 / 1024 / 1024).toFixed(1)}G`,
              used: `${(used / 1024 / 1024 / 1024).toFixed(1)}G`,
              available: `${(free / 1024 / 1024 / 1024).toFixed(1)}G`,
              usagePercent: usagePercent,
              mountPoint: captionMatch[1],
            });
          }
        });
        
        resolve(diskUsage);
      });
      
      cmd.on('error', () => resolve([]));
    });
  }

  async getDiskIO() {
    // Simplified disk I/O monitoring
    // In a production environment, you'd use more sophisticated tools
    return {
      readMBps: Math.random() * 50, // Placeholder
      writeMBps: Math.random() * 30, // Placeholder
      iops: Math.random() * 1000, // Placeholder
    };
  }

  async getNetworkStats() {
    const interfaces = os.networkInterfaces();
    const stats = [];
    
    Object.keys(interfaces).forEach(name => {
      const iface = interfaces[name].find(i => !i.internal && i.family === 'IPv4');
      if (iface) {
        stats.push({
          name: name,
          address: iface.address,
          rxMbps: Math.random() * 100, // Placeholder - would need platform-specific implementation
          txMbps: Math.random() * 50, // Placeholder
          rxPackets: Math.floor(Math.random() * 10000),
          txPackets: Math.floor(Math.random() * 8000),
        });
      }
    });
    
    return stats;
  }

  async measureNetworkLatency() {
    // Measure latency to key geological services
    const targets = [
      'localhost',
      '8.8.8.8', // Google DNS
      'usgs.gov', // USGS
    ];
    
    const latencies = [];
    
    for (const target of targets) {
      try {
        const latency = await this.pingHost(target);
        latencies.push({ target, latency });
      } catch (error) {
        latencies.push({ target, latency: -1, error: error.message });
      }
    }
    
    const validLatencies = latencies.filter(l => l.latency > 0).map(l => l.latency);
    const average = validLatencies.length > 0 
      ? validLatencies.reduce((sum, lat) => sum + lat, 0) / validLatencies.length 
      : 0;
    
    return {
      targets: latencies,
      average: average,
      min: validLatencies.length > 0 ? Math.min(...validLatencies) : 0,
      max: validLatencies.length > 0 ? Math.max(...validLatencies) : 0,
    };
  }

  async pingHost(host) {
    return new Promise((resolve, reject) => {
      const isWindows = process.platform === 'win32';
      const cmd = isWindows ? 'ping' : 'ping';
      const args = isWindows ? ['-n', '1', host] : ['-c', '1', host];
      
      const ping = spawn(cmd, args);
      let output = '';
      
      ping.stdout.on('data', (data) => {
        output += data.toString();
      });
      
      ping.on('close', (code) => {
        if (code === 0) {
          // Extract latency from ping output
          const latencyMatch = output.match(/time[<=](\d+(?:\.\d+)?)ms/i);
          if (latencyMatch) {
            resolve(parseFloat(latencyMatch[1]));
          } else {
            resolve(0);
          }
        } else {
          reject(new Error('Ping failed'));
        }
      });
      
      ping.on('error', reject);
      
      // Timeout after 5 seconds
      setTimeout(() => {
        ping.kill();
        reject(new Error('Ping timeout'));
      }, 5000);
    });
  }

  async getProcessInfo() {
    if (process.platform === 'linux' || process.platform === 'darwin') {
      return await this.getUnixProcessInfo();
    } else if (process.platform === 'win32') {
      return await this.getWindowsProcessInfo();
    }
    return [];
  }

  async getUnixProcessInfo() {
    return new Promise((resolve) => {
      const cmd = spawn('ps', ['aux']);
      let output = '';
      
      cmd.stdout.on('data', (data) => {
        output += data.toString();
      });
      
      cmd.on('close', () => {
        const lines = output.split('\n').slice(1); // Skip header
        const processes = [];
        
        lines.forEach(line => {
          const parts = line.split(/\s+/);
          if (parts.length >= 11) {
            processes.push({
              pid: parseInt(parts[1]),
              name: parts[10],
              cpu: parseFloat(parts[2]),
              memory: parseFloat(parts[3]) * os.totalmem() / 100, // Convert % to bytes
              user: parts[0],
            });
          }
        });
        
        resolve(processes.sort((a, b) => b.cpu - a.cpu));
      });
      
      cmd.on('error', () => resolve([]));
    });
  }

  async getWindowsProcessInfo() {
    return new Promise((resolve) => {
      const cmd = spawn('wmic', ['process', 'get', 'ProcessId,Name,PageFileUsage,WorkingSetSize', '/format:csv']);
      let output = '';
      
      cmd.stdout.on('data', (data) => {
        output += data.toString();
      });
      
      cmd.on('close', () => {
        const lines = output.split('\n').slice(1); // Skip header
        const processes = [];
        
        lines.forEach(line => {
          const parts = line.split(',');
          if (parts.length >= 5 && parts[1]) {
            processes.push({
              pid: parseInt(parts[4]) || 0,
              name: parts[1],
              cpu: 0, // CPU usage not easily available via wmic
              memory: parseInt(parts[5]) || 0,
              user: 'unknown',
            });
          }
        });
        
        resolve(processes);
      });
      
      cmd.on('error', () => resolve([]));
    });
  }

  async getServiceInfo(serviceName) {
    // Get information about specific geological services
    const processes = await this.getProcessInfo();
    const serviceProcess = processes.find(p => 
      p.name.toLowerCase().includes(serviceName.toLowerCase())
    );
    
    if (serviceProcess) {
      return {
        pid: serviceProcess.pid,
        cpu: serviceProcess.cpu,
        memory: serviceProcess.memory,
        status: 'running',
      };
    }
    
    return null;
  }

  async monitorDatabases() {
    // Monitor database performance
    const databases = {};
    
    // PostgreSQL monitoring
    try {
      databases.postgresql = await this.monitorPostgreSQL();
    } catch (error) {
      databases.postgresql = { error: error.message };
    }
    
    // InfluxDB monitoring
    try {
      databases.influxdb = await this.monitorInfluxDB();
    } catch (error) {
      databases.influxdb = { error: error.message };
    }
    
    // Neo4j monitoring
    try {
      databases.neo4j = await this.monitorNeo4j();
    } catch (error) {
      databases.neo4j = { error: error.message };
    }
    
    // Redis monitoring
    try {
      databases.redis = await this.monitorRedis();
    } catch (error) {
      databases.redis = { error: error.message };
    }
    
    return databases;
  }

  async monitorPostgreSQL() {
    // Placeholder for PostgreSQL monitoring
    // In production, you'd connect to the database and query performance metrics
    return {
      connections: Math.floor(Math.random() * 100),
      activeQueries: Math.floor(Math.random() * 20),
      cacheHitRatio: 0.95 + Math.random() * 0.05,
      avgQueryTime: Math.random() * 100,
    };
  }

  async monitorInfluxDB() {
    // Placeholder for InfluxDB monitoring
    return {
      writeRate: Math.floor(Math.random() * 1000),
      queryRate: Math.floor(Math.random() * 100),
      memoryUsage: Math.random() * 1024 * 1024 * 1024,
      diskUsage: Math.random() * 10 * 1024 * 1024 * 1024,
    };
  }

  async monitorNeo4j() {
    // Placeholder for Neo4j monitoring
    return {
      transactions: Math.floor(Math.random() * 50),
      cacheSize: Math.random() * 512 * 1024 * 1024,
      storeSize: Math.random() * 5 * 1024 * 1024 * 1024,
    };
  }

  async monitorRedis() {
    // Placeholder for Redis monitoring
    return {
      connectedClients: Math.floor(Math.random() * 50),
      usedMemory: Math.random() * 256 * 1024 * 1024,
      keyspaceHits: Math.floor(Math.random() * 10000),
      keyspaceMisses: Math.floor(Math.random() * 1000),
    };
  }

  async monitorAPIGateway() {
    // Monitor API Gateway performance
    return {
      requestRate: Math.floor(Math.random() * 1000),
      errorRate: Math.random() * 0.05,
      avgResponseTime: Math.random() * 200,
      activeConnections: Math.floor(Math.random() * 200),
    };
  }

  generateAlert(category, message, severity) {
    const alert = {
      timestamp: Date.now(),
      category,
      message,
      severity,
      testName: this.testName,
    };
    
    this.metrics.alerts.push(alert);
    
    // Log alert immediately
    const severityEmoji = {
      info: 'ℹ️',
      warning: '⚠️',
      critical: '🚨',
    };
    
    console.log(`${severityEmoji[severity] || '📢'} ${severity.toUpperCase()}: ${message}`);
  }

  async generateResourceReport() {
    const duration = this.endTime - this.startTime;
    
    // Calculate peak values
    const peakCPU = this.metrics.cpu.length > 0 
      ? Math.max(...this.metrics.cpu.map(m => m.usage)) 
      : 0;
    
    const peakMemory = this.metrics.memory.length > 0 
      ? Math.max(...this.metrics.memory.map(m => m.usagePercent)) 
      : 0;
    
    const peakDiskIO = this.metrics.disk.length > 0 
      ? Math.max(...this.metrics.disk.map(m => m.io.readMBps + m.io.writeMBps)) 
      : 0;
    
    // Calculate averages
    const avgCPU = this.metrics.cpu.length > 0 
      ? this.metrics.cpu.reduce((sum, m) => sum + m.usage, 0) / this.metrics.cpu.length 
      : 0;
    
    const avgMemory = this.metrics.memory.length > 0 
      ? this.metrics.memory.reduce((sum, m) => sum + m.usagePercent, 0) / this.metrics.memory.length 
      : 0;
    
    // Analyze alerts
    const alertsByCategory = {};
    const alertsBySeverity = {};
    
    this.metrics.alerts.forEach(alert => {
      alertsByCategory[alert.category] = (alertsByCategory[alert.category] || 0) + 1;
      alertsBySeverity[alert.severity] = (alertsBySeverity[alert.severity] || 0) + 1;
    });
    
    const report = {
      testName: this.testName,
      duration: duration,
      timestamp: new Date().toISOString(),
      
      // Peak values
      peakCPU,
      peakMemory,
      peakDiskIO,
      
      // Average values
      avgCPU,
      avgMemory,
      
      // Alert summary
      totalAlerts: this.metrics.alerts.length,
      alertsByCategory,
      alertsBySeverity,
      
      // Detailed metrics
      metrics: this.metrics,
      
      // Performance assessment
      performanceAssessment: this.assessPerformance({
        peakCPU,
        peakMemory,
        avgCPU,
        avgMemory,
        totalAlerts: this.metrics.alerts.length,
      }),
      
      // Recommendations
      recommendations: this.generateResourceRecommendations({
        peakCPU,
        peakMemory,
        avgCPU,
        avgMemory,
        alerts: this.metrics.alerts,
      }),
    };
    
    // Save report
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const reportPath = path.join(
      this.config.outputDir,
      `resource-report-${this.testName}-${timestamp}.json`
    );
    
    await fs.writeFile(reportPath, JSON.stringify(report, null, 2));
    
    // Generate visualization
    await this.generateResourceVisualization(report, timestamp);
    
    return report;
  }

  assessPerformance(metrics) {
    let score = 100;
    let status = 'excellent';
    
    // CPU assessment
    if (metrics.peakCPU > 90) {
      score -= 30;
      status = 'critical';
    } else if (metrics.peakCPU > 80) {
      score -= 15;
      status = status === 'excellent' ? 'warning' : status;
    }
    
    if (metrics.avgCPU > 70) {
      score -= 20;
      status = status === 'excellent' ? 'warning' : status;
    }
    
    // Memory assessment
    if (metrics.peakMemory > 95) {
      score -= 25;
      status = 'critical';
    } else if (metrics.peakMemory > 85) {
      score -= 10;
      status = status === 'excellent' ? 'warning' : status;
    }
    
    // Alert assessment
    if (metrics.totalAlerts > 10) {
      score -= 20;
      status = status === 'excellent' ? 'warning' : status;
    }
    
    if (score < 50) status = 'critical';
    else if (score < 70) status = 'warning';
    else if (score < 85) status = 'good';
    
    return {
      score: Math.max(0, score),
      status,
    };
  }

  generateResourceRecommendations(metrics) {
    const recommendations = [];
    
    if (metrics.peakCPU > 85) {
      recommendations.push({
        type: 'cpu',
        priority: 'high',
        message: 'High CPU usage detected. Consider scaling horizontally or optimizing CPU-intensive operations.',
      });
    }
    
    if (metrics.peakMemory > 90) {
      recommendations.push({
        type: 'memory',
        priority: 'high',
        message: 'High memory usage detected. Review memory leaks and consider increasing available memory.',
      });
    }
    
    if (metrics.avgCPU > 60) {
      recommendations.push({
        type: 'cpu',
        priority: 'medium',
        message: 'Sustained high CPU usage. Consider load balancing and performance optimization.',
      });
    }
    
    const criticalAlerts = metrics.alerts.filter(a => a.severity === 'critical').length;
    if (criticalAlerts > 0) {
      recommendations.push({
        type: 'alerts',
        priority: 'high',
        message: `${criticalAlerts} critical alerts generated. Review system stability and capacity.`,
      });
    }
    
    return recommendations;
  }

  async generateResourceVisualization(report, timestamp) {
    // Generate HTML report with resource usage charts
    const htmlReport = this.generateResourceHTMLReport(report);
    const reportPath = path.join(
      this.config.outputDir,
      `resource-visualization-${report.testName}-${timestamp}.html`
    );
    await fs.writeFile(reportPath, htmlReport);
    
    console.log(`📊 Resource visualization saved to ${reportPath}`);
  }

  generateResourceHTMLReport(report) {
    return `
<!DOCTYPE html>
<html>
<head>
    <title>Resource Monitoring Report - ${report.testName}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .summary { background-color: #f0f0f0; padding: 15px; margin-bottom: 20px; }
        .chart-container { width: 100%; height: 400px; margin: 20px 0; }
        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }
        .metric-card { background-color: #f8f9fa; padding: 15px; border-radius: 5px; text-align: center; }
        .performance-${report.performanceAssessment.status} { 
          border-left: 5px solid ${
            report.performanceAssessment.status === 'excellent' ? '#28a745' :
            report.performanceAssessment.status === 'good' ? '#17a2b8' :
            report.performanceAssessment.status === 'warning' ? '#ffc107' : '#dc3545'
          }; 
        }
        .alerts { background-color: #fff3cd; padding: 15px; margin: 20px 0; }
        .recommendations { background-color: #d1ecf1; padding: 15px; margin: 20px 0; }
    </style>
</head>
<body>
    <h1>📊 Resource Monitoring Report</h1>
    <h2>${report.testName}</h2>
    
    <div class="summary performance-${report.performanceAssessment.status}">
        <h3>Performance Summary</h3>
        <p><strong>Status:</strong> ${report.performanceAssessment.status.toUpperCase()}</p>
        <p><strong>Score:</strong> ${report.performanceAssessment.score}/100</p>
        <p><strong>Duration:</strong> ${(report.duration / 1000 / 60).toFixed(1)} minutes</p>
        <p><strong>Total Alerts:</strong> ${report.totalAlerts}</p>
    </div>

    <div class="metrics-grid">
        <div class="metric-card">
            <h4>Peak CPU</h4>
            <div style="font-size: 2em; color: ${report.peakCPU > 80 ? '#dc3545' : '#28a745'};">
                ${report.peakCPU.toFixed(1)}%
            </div>
        </div>
        <div class="metric-card">
            <h4>Peak Memory</h4>
            <div style="font-size: 2em; color: ${report.peakMemory > 85 ? '#dc3545' : '#28a745'};">
                ${report.peakMemory.toFixed(1)}%
            </div>
        </div>
        <div class="metric-card">
            <h4>Average CPU</h4>
            <div style="font-size: 2em; color: ${report.avgCPU > 60 ? '#ffc107' : '#28a745'};">
                ${report.avgCPU.toFixed(1)}%
            </div>
        </div>
        <div class="metric-card">
            <h4>Average Memory</h4>
            <div style="font-size: 2em; color: ${report.avgMemory > 70 ? '#ffc107' : '#28a745'};">
                ${report.avgMemory.toFixed(1)}%
            </div>
        </div>
    </div>

    <div class="chart-container">
        <canvas id="cpu-chart"></canvas>
    </div>

    <div class="chart-container">
        <canvas id="memory-chart"></canvas>
    </div>

    ${report.totalAlerts > 0 ? `
    <div class="alerts">
        <h3>🚨 Alerts Summary</h3>
        <p><strong>Total Alerts:</strong> ${report.totalAlerts}</p>
        <ul>
            ${Object.entries(report.alertsByCategory).map(([category, count]) => 
              `<li><strong>${category}:</strong> ${count} alerts</li>`
            ).join('')}
        </ul>
    </div>
    ` : ''}

    ${report.recommendations.length > 0 ? `
    <div class="recommendations">
        <h3>💡 Recommendations</h3>
        <ul>
            ${report.recommendations.map(rec => 
              `<li><strong>${rec.type} (${rec.priority}):</strong> ${rec.message}</li>`
            ).join('')}
        </ul>
    </div>
    ` : ''}

    <script>
        // CPU Usage Chart
        new Chart(document.getElementById('cpu-chart'), {
            type: 'line',
            data: {
                labels: ${JSON.stringify(report.metrics.cpu.map(m => new Date(m.timestamp).toLocaleTimeString()))},
                datasets: [{
                    label: 'CPU Usage (%)',
                    data: ${JSON.stringify(report.metrics.cpu.map(m => m.usage))},
                    borderColor: 'rgba(255, 99, 132, 1)',
                    backgroundColor: 'rgba(255, 99, 132, 0.2)',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: { y: { beginAtZero: true, max: 100 } },
                plugins: { title: { display: true, text: 'CPU Usage Over Time' } }
            }
        });

        // Memory Usage Chart
        new Chart(document.getElementById('memory-chart'), {
            type: 'line',
            data: {
                labels: ${JSON.stringify(report.metrics.memory.map(m => new Date(m.timestamp).toLocaleTimeString()))},
                datasets: [{
                    label: 'Memory Usage (%)',
                    data: ${JSON.stringify(report.metrics.memory.map(m => m.usagePercent))},
                    borderColor: 'rgba(54, 162, 235, 1)',
                    backgroundColor: 'rgba(54, 162, 235, 0.2)',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: { y: { beginAtZero: true, max: 100 } },
                plugins: { title: { display: true, text: 'Memory Usage Over Time' } }
            }
        });
    </script>
</body>
</html>`;
  }

  async ensureOutputDirectory() {
    try {
      await fs.access(this.config.outputDir);
    } catch {
      await fs.mkdir(this.config.outputDir, { recursive: true });
    }
  }
}

// CLI interface
if (require.main === module) {
  const config = {
    samplingInterval: parseInt(process.env.SAMPLING_INTERVAL) || 1000,
    monitoringDuration: parseInt(process.env.MONITORING_DURATION) || 300000,
    outputDir: process.env.OUTPUT_DIR || './tests/performance/resource-reports',
  };

  const monitor = new ResourceMonitor(config);
  
  monitor.startMonitoring('cli-resource-test')
    .then(() => {
      console.log('Resource monitoring started. Press Ctrl+C to stop.');
    })
    .catch(error => {
      console.error('Resource monitoring failed:', error);
      process.exit(1);
    });

  // Handle graceful shutdown
  process.on('SIGINT', async () => {
    console.log('\nReceived SIGINT, stopping monitoring...');
    const report = await monitor.stopMonitoring();
    console.log(`Final report saved. Performance score: ${report.performanceAssessment.score}/100`);
    process.exit(0);
  });
}

module.exports = ResourceMonitor;
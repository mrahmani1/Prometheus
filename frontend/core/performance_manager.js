/**
 * Performance Manager
 * Advanced performance optimization system with WebWorkers, rendering optimizations, and progressive loading
 */

window.PERFORMANCE_MANAGER = (function() {
    'use strict';

    let workers = new Map();
    let performanceMetrics = {
        frameRate: 0,
        memoryUsage: 0,
        renderTime: 0,
        computeTime: 0,
        loadTime: 0
    };
    let lodLevels = new Map();
    let virtualScrollers = new Map();
    let progressiveLoaders = new Map();
    let performanceObserver = null;

    const WORKER_TASKS = {
        SEISMIC_PROCESSING: 'seismic-processing',
        TOMOGRAPHY_CALCULATION: 'tomography-calculation',
        RISK_ANALYSIS: 'risk-analysis',
        DATA_TRANSFORMATION: 'data-transformation',
        MESH_GENERATION: 'mesh-generation',
        MONTE_CARLO: 'monte-carlo',
        FFT_PROCESSING: 'fft-processing',
        CLUSTERING: 'clustering'
    };

    const LOD_LEVELS = {
        HIGH: { detail: 1.0, distance: 0, vertices: Infinity },
        MEDIUM: { detail: 0.5, distance: 50, vertices: 50000 },
        LOW: { detail: 0.25, distance: 100, vertices: 10000 },
        ULTRA_LOW: { detail: 0.1, distance: 200, vertices: 1000 }
    };

    const PERFORMANCE_THRESHOLDS = {
        TARGET_FPS: 60,
        MIN_FPS: 30,
        MAX_MEMORY_MB: 512,
        MAX_RENDER_TIME_MS: 16.67, // 60 FPS target
        THROTTLE_THRESHOLD: 0.8
    };

    // Initialize performance management system
    function init() {
        console.log('⚡ Initializing Performance Manager...');
        
        setupWebWorkers();
        initializePerformanceMonitoring();
        setupLevelOfDetail();
        initializeVirtualScrolling();
        setupProgressiveLoading();
        createPerformanceUI();
        
        console.log('✅ Performance Manager initialized');
    }

    function setupWebWorkers() {
        console.log('👥 Setting up WebWorkers...');
        
        // Create worker pool for different task types
        Object.values(WORKER_TASKS).forEach(taskType => {
            createWorkerPool(taskType, getOptimalWorkerCount(taskType));
        });
        
        // Setup worker message handling
        setupWorkerCommunication();
    }

    function createWorkerPool(taskType, count) {
        const pool = {
            available: [],
            busy: new Set(),
            queue: [],
            stats: { tasksCompleted: 0, totalTime: 0 }
        };
        
        for (let i = 0; i < count; i++) {
            const worker = createWorker(taskType);
            if (worker) {
                pool.available.push(worker);
            }
        }
        
        workers.set(taskType, pool);
        console.log(`🔧 Created ${count} workers for ${taskType}`);
    }

    function createWorker(taskType) {
        try {
            // Create worker with inline script to avoid CORS issues
            const workerScript = generateWorkerScript(taskType);
            const blob = new Blob([workerScript], { type: 'application/javascript' });
            const workerUrl = URL.createObjectURL(blob);
            
            const worker = new Worker(workerUrl);
            worker.taskType = taskType;
            worker.id = Math.random().toString(36).substr(2, 9);
            
            worker.onmessage = handleWorkerMessage;
            worker.onerror = handleWorkerError;
            
            return worker;
        } catch (error) {
            console.warn(`Failed to create worker for ${taskType}:`, error);
            return null;
        }
    }

    function generateWorkerScript(taskType) {
        const commonFunctions = `
            // Common mathematical functions
            function gaussianKernel(x, sigma) {
                return Math.exp(-(x * x) / (2 * sigma * sigma)) / (sigma * Math.sqrt(2 * Math.PI));
            }
            
            function fastFourierTransform(signal) {
                const N = signal.length;
                if (N <= 1) return signal;
                
                const even = fastFourierTransform(signal.filter((_, i) => i % 2 === 0));
                const odd = fastFourierTransform(signal.filter((_, i) => i % 2 === 1));
                
                const result = new Array(N);
                for (let k = 0; k < N / 2; k++) {
                    const t = Math.exp(-2 * Math.PI * k / N) * odd[k];
                    result[k] = even[k] + t;
                    result[k + N / 2] = even[k] - t;
                }
                
                return result;
            }
            
            function kmeansClustering(data, k, maxIterations = 100) {
                const centroids = [];
                for (let i = 0; i < k; i++) {
                    centroids.push(data[Math.floor(Math.random() * data.length)]);
                }
                
                for (let iter = 0; iter < maxIterations; iter++) {
                    const clusters = Array(k).fill().map(() => []);
                    
                    // Assign points to nearest centroid
                    data.forEach(point => {
                        let minDist = Infinity;
                        let clusterIndex = 0;
                        
                        centroids.forEach((centroid, i) => {
                            const dist = euclideanDistance(point, centroid);
                            if (dist < minDist) {
                                minDist = dist;
                                clusterIndex = i;
                            }
                        });
                        
                        clusters[clusterIndex].push(point);
                    });
                    
                    // Update centroids
                    let changed = false;
                    for (let i = 0; i < k; i++) {
                        if (clusters[i].length > 0) {
                            const newCentroid = calculateCentroid(clusters[i]);
                            if (euclideanDistance(centroids[i], newCentroid) > 0.001) {
                                centroids[i] = newCentroid;
                                changed = true;
                            }
                        }
                    }
                    
                    if (!changed) break;
                }
                
                return centroids;
            }
            
            function euclideanDistance(a, b) {
                return Math.sqrt(a.reduce((sum, val, i) => sum + Math.pow(val - b[i], 2), 0));
            }
            
            function calculateCentroid(points) {
                const dimensions = points[0].length;
                const centroid = new Array(dimensions).fill(0);
                
                points.forEach(point => {
                    point.forEach((val, i) => {
                        centroid[i] += val;
                    });
                });
                
                return centroid.map(val => val / points.length);
            }
        `;

        const taskSpecificCode = {
            [WORKER_TASKS.SEISMIC_PROCESSING]: `
                function processSeismicData(data) {
                    const { waveforms, sampleRate } = data;
                    const processed = [];
                    
                    waveforms.forEach(waveform => {
                        // Apply bandpass filter
                        const filtered = bandpassFilter(waveform, 1, 30, sampleRate);
                        
                        // Calculate STA/LTA for event detection
                        const staLta = calculateSTALTA(filtered, sampleRate);
                        
                        // Find P and S wave arrivals
                        const arrivals = detectArrivals(filtered, staLta);
                        
                        processed.push({
                            filtered: filtered,
                            staLta: staLta,
                            arrivals: arrivals
                        });
                    });
                    
                    return processed;
                }
                
                function bandpassFilter(signal, lowFreq, highFreq, sampleRate) {
                    // Simple butterworth filter implementation
                    const nyquist = sampleRate / 2;
                    const lowNorm = lowFreq / nyquist;
                    const highNorm = highFreq / nyquist;
                    
                    return signal.map((sample, i) => {
                        if (i === 0) return sample;
                        // Simplified filter - in practice use proper filter coefficients
                        return sample * 0.8 + signal[i-1] * 0.2;
                    });
                }
                
                function calculateSTALTA(signal, sampleRate) {
                    const staWindow = Math.floor(0.5 * sampleRate); // 0.5 second STA
                    const ltaWindow = Math.floor(10 * sampleRate); // 10 second LTA
                    const ratio = [];
                    
                    for (let i = ltaWindow; i < signal.length; i++) {
                        const sta = signal.slice(i - staWindow, i).reduce((a, b) => a + b*b, 0) / staWindow;
                        const lta = signal.slice(i - ltaWindow, i).reduce((a, b) => a + b*b, 0) / ltaWindow;
                        ratio.push(lta > 0 ? sta / lta : 0);
                    }
                    
                    return ratio;
                }
                
                function detectArrivals(signal, staLta) {
                    const threshold = 3.0;
                    const arrivals = [];
                    
                    for (let i = 1; i < staLta.length; i++) {
                        if (staLta[i] > threshold && staLta[i-1] <= threshold) {
                            arrivals.push({ type: 'P', time: i, confidence: staLta[i] });
                        }
                    }
                    
                    return arrivals;
                }
            `,
            
            [WORKER_TASKS.TOMOGRAPHY_CALCULATION]: `
                function calculateTomography(data) {
                    const { rays, grid, iterations } = data;
                    let velocity = new Float32Array(grid.nx * grid.ny * grid.nz);
                    
                    // Initialize with reference model
                    velocity.fill(6.0); // km/s
                    
                    // Iterative inversion using LSQR
                    for (let iter = 0; iter < iterations; iter++) {
                        const residuals = calculateResiduals(rays, velocity, grid);
                        const update = calculateUpdate(residuals, rays, grid);
                        
                        // Apply damping and smoothing
                        for (let i = 0; i < velocity.length; i++) {
                            velocity[i] += update[i] * 0.1; // damping factor
                        }
                        
                        // Apply smoothing
                        velocity = applySmoothingFilter(velocity, grid);
                    }
                    
                    return velocity;
                }
                
                function calculateResiduals(rays, velocity, grid) {
                    return rays.map(ray => {
                        const predicted = calculateTravelTime(ray.path, velocity, grid);
                        return ray.observed - predicted;
                    });
                }
                
                function calculateTravelTime(path, velocity, grid) {
                    let travelTime = 0;
                    for (let i = 0; i < path.length - 1; i++) {
                        const distance = euclideanDistance(path[i], path[i + 1]);
                        const vel = interpolateVelocity(path[i], velocity, grid);
                        travelTime += distance / vel;
                    }
                    return travelTime;
                }
                
                function applySmoothingFilter(velocity, grid) {
                    const smoothed = new Float32Array(velocity.length);
                    const { nx, ny, nz } = grid;
                    
                    for (let k = 1; k < nz - 1; k++) {
                        for (let j = 1; j < ny - 1; j++) {
                            for (let i = 1; i < nx - 1; i++) {
                                const idx = i + j * nx + k * nx * ny;
                                let sum = velocity[idx] * 6; // center weight
                                
                                // Add neighbors
                                sum += velocity[idx - 1] + velocity[idx + 1]; // x neighbors
                                sum += velocity[idx - nx] + velocity[idx + nx]; // y neighbors
                                sum += velocity[idx - nx * ny] + velocity[idx + nx * ny]; // z neighbors
                                
                                smoothed[idx] = sum / 12;
                            }
                        }
                    }
                    
                    return smoothed;
                }
            `,
            
            [WORKER_TASKS.RISK_ANALYSIS]: `
                function performRiskAnalysis(data) {
                    const { hazards, vulnerabilities, exposures, iterations } = data;
                    const results = [];
                    
                    for (let i = 0; i < iterations; i++) {
                        const scenario = generateRandomScenario(hazards);
                        const risk = calculateScenarioRisk(scenario, vulnerabilities, exposures);
                        results.push(risk);
                    }
                    
                    return analyzeRiskDistribution(results);
                }
                
                function generateRandomScenario(hazards) {
                    const scenario = {};
                    
                    if (hazards.seismic) {
                        scenario.magnitude = hazards.seismic.minMag + 
                                           Math.random() * (hazards.seismic.maxMag - hazards.seismic.minMag);
                        scenario.depth = 5 + Math.random() * 25; // 5-30 km
                    }
                    
                    if (hazards.volcanic) {
                        scenario.vei = Math.floor(Math.random() * 6) + 1; // VEI 1-6
                        scenario.windSpeed = 5 + Math.random() * 45; // 5-50 m/s
                    }
                    
                    return scenario;
                }
                
                function calculateScenarioRisk(scenario, vulnerabilities, exposures) {
                    let totalRisk = 0;
                    
                    if (scenario.magnitude) {
                        const pga = Math.exp(scenario.magnitude - 3.5) / scenario.depth;
                        const seismicRisk = pga * vulnerabilities.building * exposures.population;
                        totalRisk += seismicRisk;
                    }
                    
                    if (scenario.vei) {
                        const volcanicRisk = Math.pow(2, scenario.vei) * vulnerabilities.ash * exposures.agriculture;
                        totalRisk += volcanicRisk;
                    }
                    
                    return totalRisk;
                }
                
                function analyzeRiskDistribution(results) {
                    results.sort((a, b) => a - b);
                    const length = results.length;
                    
                    return {
                        mean: results.reduce((a, b) => a + b, 0) / length,
                        median: results[Math.floor(length / 2)],
                        p95: results[Math.floor(0.95 * length)],
                        p99: results[Math.floor(0.99 * length)],
                        max: results[length - 1],
                        min: results[0]
                    };
                }
            `,
            
            [WORKER_TASKS.MESH_GENERATION]: `
                function generateMesh(data) {
                    const { points, triangulate, simplify } = data;
                    
                    let mesh;
                    if (triangulate) {
                        mesh = delaunayTriangulation(points);
                    } else {
                        mesh = generateRegularMesh(points);
                    }
                    
                    if (simplify) {
                        mesh = simplifyMesh(mesh, simplify.tolerance);
                    }
                    
                    return mesh;
                }
                
                function delaunayTriangulation(points) {
                    // Simplified Delaunay triangulation
                    const triangles = [];
                    
                    for (let i = 0; i < points.length - 2; i++) {
                        for (let j = i + 1; j < points.length - 1; j++) {
                            for (let k = j + 1; k < points.length; k++) {
                                if (isValidTriangle(points[i], points[j], points[k])) {
                                    triangles.push([i, j, k]);
                                }
                            }
                        }
                    }
                    
                    return { vertices: points, triangles: triangles };
                }
                
                function isValidTriangle(a, b, c) {
                    const area = Math.abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]));
                    return area > 0.001; // Minimum area threshold
                }
                
                function simplifyMesh(mesh, tolerance) {
                    // Quadric error simplification
                    const simplified = { vertices: [], triangles: [] };
                    const vertexMap = new Map();
                    
                    mesh.vertices.forEach((vertex, i) => {
                        const key = Math.floor(vertex[0] / tolerance) + ',' + 
                                   Math.floor(vertex[1] / tolerance) + ',' + 
                                   Math.floor(vertex[2] / tolerance);
                        
                        if (!vertexMap.has(key)) {
                            vertexMap.set(key, simplified.vertices.length);
                            simplified.vertices.push(vertex);
                        }
                    });
                    
                    mesh.triangles.forEach(triangle => {
                        const newTriangle = triangle.map(i => {
                            const vertex = mesh.vertices[i];
                            const key = Math.floor(vertex[0] / tolerance) + ',' + 
                                       Math.floor(vertex[1] / tolerance) + ',' + 
                                       Math.floor(vertex[2] / tolerance);
                            return vertexMap.get(key);
                        });
                        
                        if (newTriangle[0] !== newTriangle[1] && 
                            newTriangle[1] !== newTriangle[2] && 
                            newTriangle[0] !== newTriangle[2]) {
                            simplified.triangles.push(newTriangle);
                        }
                    });
                    
                    return simplified;
                }
            `
        };

        return `
            ${commonFunctions}
            ${taskSpecificCode[taskType] || ''}
            
            self.onmessage = function(e) {
                const { id, task, data } = e.data;
                const startTime = performance.now();
                
                try {
                    let result;
                    
                    switch(task) {
                        case 'seismic-processing':
                            result = processSeismicData(data);
                            break;
                        case 'tomography-calculation':
                            result = calculateTomography(data);
                            break;
                        case 'risk-analysis':
                            result = performRiskAnalysis(data);
                            break;
                        case 'mesh-generation':
                            result = generateMesh(data);
                            break;
                        case 'monte-carlo':
                            result = performRiskAnalysis(data);
                            break;
                        case 'fft-processing':
                            result = fastFourierTransform(data.signal);
                            break;
                        case 'clustering':
                            result = kmeansClustering(data.points, data.k);
                            break;
                        default:
                            throw new Error('Unknown task: ' + task);
                    }
                    
                    const endTime = performance.now();
                    
                    self.postMessage({
                        id: id,
                        task: task,
                        result: result,
                        processingTime: endTime - startTime,
                        success: true
                    });
                } catch (error) {
                    self.postMessage({
                        id: id,
                        task: task,
                        error: error.message,
                        success: false
                    });
                }
            };
        `;
    }

    function getOptimalWorkerCount(taskType) {
        const cores = navigator.hardwareConcurrency || 4;
        
        // Adjust worker count based on task type
        switch (taskType) {
            case WORKER_TASKS.SEISMIC_PROCESSING:
            case WORKER_TASKS.TOMOGRAPHY_CALCULATION:
                return Math.min(cores, 4); // CPU intensive
            case WORKER_TASKS.RISK_ANALYSIS:
            case WORKER_TASKS.MONTE_CARLO:
                return Math.min(cores - 1, 6); // Leave one core for UI
            default:
                return Math.min(cores / 2, 2); // Conservative for other tasks
        }
    }

    function setupWorkerCommunication() {
        // Global message handler for all workers
        function handleWorkerMessage(event) {
            const { id, task, result, error, processingTime, success } = event.data;
            
            if (success) {
                console.log(`✅ Worker completed ${task} in ${processingTime.toFixed(2)}ms`);
                
                // Update performance metrics
                performanceMetrics.computeTime = processingTime;
                
                // Return worker to available pool
                const pool = workers.get(this.taskType);
                if (pool) {
                    pool.busy.delete(this);
                    pool.available.push(this);
                    pool.stats.tasksCompleted++;
                    pool.stats.totalTime += processingTime;
                    
                    // Process queued tasks
                    processWorkerQueue(this.taskType);
                }
                
                // Emit completion event
                document.dispatchEvent(new CustomEvent('workerTaskComplete', {
                    detail: { id, task, result, processingTime }
                }));
            } else {
                console.error(`❌ Worker failed ${task}:`, error);
                
                // Return worker to available pool
                const pool = workers.get(this.taskType);
                if (pool) {
                    pool.busy.delete(this);
                    pool.available.push(this);
                }
                
                // Emit error event
                document.dispatchEvent(new CustomEvent('workerTaskError', {
                    detail: { id, task, error }
                }));
            }
        }
        
        function handleWorkerError(error) {
            console.error('Worker error:', error);
            
            // Remove failed worker from pool
            const pool = workers.get(this.taskType);
            if (pool) {
                pool.busy.delete(this);
                const index = pool.available.indexOf(this);
                if (index > -1) {
                    pool.available.splice(index, 1);
                }
                
                // Create replacement worker
                const newWorker = createWorker(this.taskType);
                if (newWorker) {
                    pool.available.push(newWorker);
                }
            }
        }
        
        // Assign handlers to window for access in worker creation
        window.handleWorkerMessage = handleWorkerMessage;
        window.handleWorkerError = handleWorkerError;
    }

    function executeTask(taskType, taskData, options = {}) {
        return new Promise((resolve, reject) => {
            const pool = workers.get(taskType);
            if (!pool) {
                reject(new Error(`No worker pool for task type: ${taskType}`));
                return;
            }
            
            const taskId = Math.random().toString(36).substr(2, 9);
            const task = {
                id: taskId,
                task: taskType,
                data: taskData,
                options: options,
                resolve: resolve,
                reject: reject,
                startTime: performance.now()
            };
            
            if (pool.available.length > 0) {
                assignTaskToWorker(taskType, task);
            } else {
                pool.queue.push(task);
                console.log(`📋 Task ${taskId} queued for ${taskType}`);
            }
        });
    }

    function assignTaskToWorker(taskType, task) {
        const pool = workers.get(taskType);
        const worker = pool.available.pop();
        
        pool.busy.add(worker);
        
        // Setup one-time listeners for this specific task
        const originalOnMessage = worker.onmessage;
        worker.onmessage = function(event) {
            if (event.data.id === task.id) {
                // Restore original handler
                worker.onmessage = originalOnMessage;
                
                if (event.data.success) {
                    task.resolve(event.data.result);
                } else {
                    task.reject(new Error(event.data.error));
                }
                
                // Call original handler for pool management
                originalOnMessage.call(this, event);
            }
        };
        
        worker.postMessage({
            id: task.id,
            task: task.task,
            data: task.data
        });
        
        console.log(`🚀 Task ${task.id} assigned to worker ${worker.id}`);
    }

    function processWorkerQueue(taskType) {
        const pool = workers.get(taskType);
        if (pool && pool.queue.length > 0 && pool.available.length > 0) {
            const task = pool.queue.shift();
            assignTaskToWorker(taskType, task);
        }
    }

    function initializePerformanceMonitoring() {
        console.log('📊 Setting up performance monitoring...');
        
        // Performance Observer for measuring various metrics
        if ('PerformanceObserver' in window) {
            performanceObserver = new PerformanceObserver((list) => {
                const entries = list.getEntries();
                entries.forEach(entry => {
                    switch (entry.entryType) {
                        case 'measure':
                            performanceMetrics.renderTime = entry.duration;
                            break;
                        case 'navigation':
                            performanceMetrics.loadTime = entry.loadEventEnd - entry.loadEventStart;
                            break;
                    }
                });
            });
            
            performanceObserver.observe({ entryTypes: ['measure', 'navigation'] });
        }
        
        // Frame rate monitoring
        let frameCount = 0;
        let lastFrameTime = performance.now();
        
        function measureFrameRate() {
            const currentTime = performance.now();
            frameCount++;
            
            if (currentTime - lastFrameTime >= 1000) {
                performanceMetrics.frameRate = frameCount;
                frameCount = 0;
                lastFrameTime = currentTime;
                
                // Trigger throttling if performance is poor
                if (performanceMetrics.frameRate < PERFORMANCE_THRESHOLDS.MIN_FPS) {
                    triggerPerformanceThrottling();
                }
            }
            
            requestAnimationFrame(measureFrameRate);
        }
        measureFrameRate();
        
        // Memory monitoring
        if ('memory' in performance) {
            setInterval(() => {
                performanceMetrics.memoryUsage = performance.memory.usedJSHeapSize / (1024 * 1024);
                
                if (performanceMetrics.memoryUsage > PERFORMANCE_THRESHOLDS.MAX_MEMORY_MB) {
                    triggerMemoryCleanup();
                }
            }, 5000);
        }
    }

    function setupLevelOfDetail() {
        console.log('🎯 Setting up Level of Detail system...');
        
        // Camera distance-based LOD for 3D objects
        function updateLOD(camera, objects) {
            objects.forEach(obj => {
                if (!obj.position || !obj.userData) return;
                
                const distance = camera.position.distanceTo(obj.position);
                let lodLevel = LOD_LEVELS.HIGH;
                
                if (distance > LOD_LEVELS.ULTRA_LOW.distance) {
                    lodLevel = LOD_LEVELS.ULTRA_LOW;
                } else if (distance > LOD_LEVELS.LOW.distance) {
                    lodLevel = LOD_LEVELS.LOW;
                } else if (distance > LOD_LEVELS.MEDIUM.distance) {
                    lodLevel = LOD_LEVELS.MEDIUM;
                }
                
                // Apply LOD
                if (obj.userData.lodLevel !== lodLevel) {
                    applyLODToObject(obj, lodLevel);
                    obj.userData.lodLevel = lodLevel;
                }
            });
        }
        
        // Store LOD update function globally
        window.updateLOD = updateLOD;
        
        // Automatic LOD updates for registered scenes
        setInterval(() => {
            document.dispatchEvent(new CustomEvent('lodUpdate'));
        }, 100);
    }

    function applyLODToObject(obj, lodLevel) {
        if (obj.material) {
            // Adjust material quality
            if (obj.material.map) {
                obj.material.map.minFilter = lodLevel.detail > 0.5 ? 
                    THREE.LinearMipMapLinearFilter : THREE.LinearFilter;
            }
            
            // Adjust wireframe for very low LOD
            obj.material.wireframe = lodLevel.detail < 0.2;
        }
        
        if (obj.geometry && obj.geometry.attributes) {
            // Simplify geometry by skipping vertices
            const originalVertices = obj.geometry.attributes.position.array;
            const step = Math.max(1, Math.floor(1 / lodLevel.detail));
            
            if (step > 1 && !obj.userData.originalGeometry) {
                obj.userData.originalGeometry = obj.geometry.clone();
            }
            
            if (step > 1) {
                const simplifiedVertices = [];
                for (let i = 0; i < originalVertices.length; i += step * 3) {
                    simplifiedVertices.push(
                        originalVertices[i],
                        originalVertices[i + 1],
                        originalVertices[i + 2]
                    );
                }
                obj.geometry.attributes.position.array = new Float32Array(simplifiedVertices);
                obj.geometry.attributes.position.needsUpdate = true;
            }
        }
        
        // Adjust visibility based on distance
        obj.visible = lodLevel.detail > 0.05;
    }

    function initializeVirtualScrolling() {
        console.log('📜 Setting up virtual scrolling...');
        
        class VirtualScroller {
            constructor(container, options = {}) {
                this.container = container;
                this.options = {
                    itemHeight: options.itemHeight || 50,
                    bufferSize: options.bufferSize || 10,
                    renderItem: options.renderItem || this.defaultRenderItem,
                    ...options
                };
                
                this.data = [];
                this.visibleRange = { start: 0, end: 0 };
                this.scrollTop = 0;
                
                this.setupScrollContainer();
                this.bindEvents();
            }
            
            setupScrollContainer() {
                this.viewport = document.createElement('div');
                this.viewport.style.cssText = `
                    height: 100%;
                    overflow-y: auto;
                    position: relative;
                `;
                
                this.content = document.createElement('div');
                this.content.style.cssText = `
                    position: relative;
                `;
                
                this.viewport.appendChild(this.content);
                this.container.appendChild(this.viewport);
            }
            
            bindEvents() {
                this.viewport.addEventListener('scroll', () => {
                    this.scrollTop = this.viewport.scrollTop;
                    this.updateVisibleRange();
                    this.renderVisibleItems();
                });
                
                // Handle resize
                window.addEventListener('resize', () => this.refresh());
            }
            
            setData(data) {
                this.data = data;
                this.content.style.height = `${data.length * this.options.itemHeight}px`;
                this.updateVisibleRange();
                this.renderVisibleItems();
            }
            
            updateVisibleRange() {
                const viewportHeight = this.viewport.clientHeight;
                const start = Math.floor(this.scrollTop / this.options.itemHeight);
                const end = Math.min(
                    this.data.length,
                    start + Math.ceil(viewportHeight / this.options.itemHeight) + this.options.bufferSize
                );
                
                this.visibleRange = { 
                    start: Math.max(0, start - this.options.bufferSize), 
                    end 
                };
            }
            
            renderVisibleItems() {
                // Clear existing items
                this.content.innerHTML = '';
                
                for (let i = this.visibleRange.start; i < this.visibleRange.end; i++) {
                    const item = this.options.renderItem(this.data[i], i);
                    item.style.position = 'absolute';
                    item.style.top = `${i * this.options.itemHeight}px`;
                    item.style.width = '100%';
                    item.style.height = `${this.options.itemHeight}px`;
                    
                    this.content.appendChild(item);
                }
            }
            
            defaultRenderItem(data, index) {
                const item = document.createElement('div');
                item.textContent = `Item ${index}: ${JSON.stringify(data)}`;
                return item;
            }
            
            refresh() {
                this.updateVisibleRange();
                this.renderVisibleItems();
            }
            
            scrollToIndex(index) {
                this.viewport.scrollTop = index * this.options.itemHeight;
            }
        }
        
        // Make VirtualScroller available globally
        window.VirtualScroller = VirtualScroller;
    }

    function setupProgressiveLoading() {
        console.log('📈 Setting up progressive loading...');
        
        class ProgressiveLoader {
            constructor(options = {}) {
                this.options = {
                    chunkSize: options.chunkSize || 1000,
                    delay: options.delay || 50,
                    priority: options.priority || 'normal',
                    ...options
                };
                
                this.queue = [];
                this.loading = false;
                this.loadedCount = 0;
                this.totalCount = 0;
            }
            
            load(dataSource, processor) {
                return new Promise((resolve, reject) => {
                    this.queue.push({
                        dataSource,
                        processor,
                        resolve,
                        reject,
                        priority: this.options.priority
                    });
                    
                    if (!this.loading) {
                        this.processQueue();
                    }
                });
            }
            
            async processQueue() {
                if (this.queue.length === 0) return;
                
                this.loading = true;
                
                // Sort queue by priority
                this.queue.sort((a, b) => {
                    const priorities = { high: 3, normal: 2, low: 1 };
                    return priorities[b.priority] - priorities[a.priority];
                });
                
                const task = this.queue.shift();
                
                try {
                    const result = await this.processTask(task);
                    task.resolve(result);
                } catch (error) {
                    task.reject(error);
                }
                
                // Process next task after delay
                setTimeout(() => {
                    this.loading = false;
                    this.processQueue();
                }, this.options.delay);
            }
            
            async processTask(task) {
                const { dataSource, processor } = task;
                const results = [];
                
                // Determine data size
                let data;
                if (typeof dataSource === 'function') {
                    data = await dataSource();
                } else if (Array.isArray(dataSource)) {
                    data = dataSource;
                } else {
                    throw new Error('Invalid data source');
                }
                
                this.totalCount = data.length;
                this.loadedCount = 0;
                
                // Process in chunks
                for (let i = 0; i < data.length; i += this.options.chunkSize) {
                    const chunk = data.slice(i, i + this.options.chunkSize);
                    
                    // Process chunk (potentially in worker)
                    const processedChunk = await processor(chunk, i);
                    results.push(...processedChunk);
                    
                    this.loadedCount = Math.min(this.loadedCount + chunk.length, this.totalCount);
                    
                    // Emit progress event
                    document.dispatchEvent(new CustomEvent('progressiveLoadingProgress', {
                        detail: {
                            loaded: this.loadedCount,
                            total: this.totalCount,
                            progress: this.loadedCount / this.totalCount
                        }
                    }));
                    
                    // Yield to browser for rendering
                    await new Promise(resolve => setTimeout(resolve, 1));
                }
                
                return results;
            }
        }
        
        // Make ProgressiveLoader available globally
        window.ProgressiveLoader = ProgressiveLoader;
        
        // Default progressive loader instance
        window.defaultProgressiveLoader = new ProgressiveLoader();
    }

    function triggerPerformanceThrottling() {
        console.warn('⚠️ Performance throttling activated');
        
        // Reduce LOD quality
        Object.values(LOD_LEVELS).forEach(level => {
            level.detail *= PERFORMANCE_THRESHOLDS.THROTTLE_THRESHOLD;
        });
        
        // Increase worker task delays
        Object.values(workers).forEach(pool => {
            pool.delay = (pool.delay || 0) + 10;
        });
        
        // Emit throttling event
        document.dispatchEvent(new CustomEvent('performanceThrottling', {
            detail: { frameRate: performanceMetrics.frameRate }
        }));
    }

    function triggerMemoryCleanup() {
        console.warn('🧹 Memory cleanup triggered');
        
        // Clear worker queues for non-critical tasks
        [WORKER_TASKS.DATA_TRANSFORMATION, WORKER_TASKS.CLUSTERING].forEach(taskType => {
            const pool = workers.get(taskType);
            if (pool) {
                pool.queue = pool.queue.filter(task => task.options.priority === 'high');
            }
        });
        
        // Clear caches
        if (window.CACHE_SYSTEM && window.CACHE_SYSTEM.cleanup) {
            window.CACHE_SYSTEM.cleanup();
        }
        
        // Force garbage collection if available
        if (window.gc) {
            window.gc();
        }
        
        document.dispatchEvent(new CustomEvent('memoryCleanup', {
            detail: { memoryUsage: performanceMetrics.memoryUsage }
        }));
    }

    function createPerformanceUI() {
        const ui = document.createElement('div');
        ui.id = 'performance-monitor';
        ui.innerHTML = `
            <div class="performance-panel">
                <h4>⚡ Performance Monitor</h4>
                <div class="metrics">
                    <div class="metric">
                        <span>FPS:</span>
                        <span id="fps-value">--</span>
                    </div>
                    <div class="metric">
                        <span>Memory:</span>
                        <span id="memory-value">--</span>
                    </div>
                    <div class="metric">
                        <span>Render Time:</span>
                        <span id="render-time-value">--</span>
                    </div>
                    <div class="metric">
                        <span>Workers:</span>
                        <span id="worker-count">--</span>
                    </div>
                </div>
                <div class="controls">
                    <button id="optimize-performance">🚀 Optimize</button>
                    <button id="reset-performance">🔄 Reset</button>
                </div>
            </div>
        `;
        
        ui.style.cssText = `
            position: fixed;
            top: 10px;
            right: 10px;
            width: 250px;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            border-radius: 8px;
            padding: 15px;
            font-family: monospace;
            font-size: 12px;
            z-index: 10001;
            display: none;
        `;
        
        document.body.appendChild(ui);
        
        // Update metrics display
        setInterval(() => {
            document.getElementById('fps-value').textContent = performanceMetrics.frameRate;
            document.getElementById('memory-value').textContent = 
                performanceMetrics.memoryUsage ? `${performanceMetrics.memoryUsage.toFixed(1)} MB` : '--';
            document.getElementById('render-time-value').textContent = 
                `${performanceMetrics.renderTime.toFixed(2)} ms`;
            
            let activeWorkers = 0;
            workers.forEach(pool => activeWorkers += pool.busy.size);
            document.getElementById('worker-count').textContent = `${activeWorkers}/${getTotalWorkerCount()}`;
        }, 1000);
        
        // Control buttons
        document.getElementById('optimize-performance').addEventListener('click', optimizePerformance);
        document.getElementById('reset-performance').addEventListener('click', resetPerformanceSettings);
    }

    function getTotalWorkerCount() {
        let total = 0;
        workers.forEach(pool => total += pool.available.length + pool.busy.size);
        return total;
    }

    function optimizePerformance() {
        console.log('🚀 Optimizing performance...');
        
        // Reduce LOD detail for better performance
        Object.values(LOD_LEVELS).forEach(level => {
            level.detail *= 0.8;
        });
        
        // Clear non-essential worker queues
        workers.forEach(pool => {
            pool.queue = pool.queue.filter(task => 
                task.options.priority === 'high' || task.options.essential
            );
        });
        
        // Trigger memory cleanup
        triggerMemoryCleanup();
        
        console.log('✅ Performance optimization complete');
    }

    function resetPerformanceSettings() {
        console.log('🔄 Resetting performance settings...');
        
        // Reset LOD levels
        Object.assign(LOD_LEVELS, {
            HIGH: { detail: 1.0, distance: 0, vertices: Infinity },
            MEDIUM: { detail: 0.5, distance: 50, vertices: 50000 },
            LOW: { detail: 0.25, distance: 100, vertices: 10000 },
            ULTRA_LOW: { detail: 0.1, distance: 200, vertices: 1000 }
        });
        
        console.log('✅ Performance settings reset');
    }

    function showPerformanceMonitor() {
        const monitor = document.getElementById('performance-monitor');
        if (monitor) {
            monitor.style.display = 'block';
        }
    }

    function hidePerformanceMonitor() {
        const monitor = document.getElementById('performance-monitor');
        if (monitor) {
            monitor.style.display = 'none';
        }
    }

    // Keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        if (e.ctrlKey && e.altKey && e.key === 'P') {
            e.preventDefault();
            showPerformanceMonitor();
        }
        if (e.key === 'Escape') {
            hidePerformanceMonitor();
        }
    });

    // Public API
    return {
        init: init,
        executeTask: executeTask,
        getPerformanceMetrics: () => ({ ...performanceMetrics }),
        showPerformanceMonitor: showPerformanceMonitor,
        hidePerformanceMonitor: hidePerformanceMonitor,
        optimizePerformance: optimizePerformance,
        resetPerformanceSettings: resetPerformanceSettings,
        VirtualScroller: () => window.VirtualScroller,
        ProgressiveLoader: () => window.ProgressiveLoader,
        WORKER_TASKS: WORKER_TASKS,
        LOD_LEVELS: LOD_LEVELS
    };

})();

// Auto-initialize
document.addEventListener('DOMContentLoaded', function() {
    window.PERFORMANCE_MANAGER.init();
    console.log('⚡ Performance Manager loaded');
    console.log('🎮 Press Ctrl+Alt+P to open Performance Monitor');
});
/**
 * Advanced Cache System
 * IndexedDB-based caching for large datasets with compression and intelligent eviction
 */

window.CACHE_SYSTEM = (function() {
    'use strict';

    let db = null;
    let cacheStats = {
        hits: 0,
        misses: 0,
        totalSize: 0,
        itemCount: 0,
        compressionRatio: 0
    };
    
    const CACHE_CONFIG = {
        dbName: 'PrometheusCache',
        version: 1,
        stores: {
            seismic: 'seismic_data',
            tomography: 'tomography_models',
            hazard: 'hazard_maps',
            risk: 'risk_assessments',
            mesh: 'mesh_data',
            images: 'image_cache',
            general: 'general_cache'
        },
        maxSize: 500 * 1024 * 1024, // 500MB
        maxAge: 7 * 24 * 60 * 60 * 1000, // 7 days
        compressionThreshold: 1024, // 1KB
        cleanupInterval: 60 * 60 * 1000 // 1 hour
    };

    const COMPRESSION_TYPES = {
        NONE: 'none',
        LZ_STRING: 'lz_string',
        GZIP: 'gzip'
    };

    // Initialize the cache system
    function init() {
        console.log('💾 Initializing Advanced Cache System...');
        
        return initializeDatabase()
            .then(() => {
                setupCleanupTimer();
                setupStorageQuotaMonitoring();
                createCacheUI();
                console.log('✅ Cache System initialized');
                return true;
            })
            .catch(error => {
                console.error('❌ Cache System initialization failed:', error);
                return false;
            });
    }

    function initializeDatabase() {
        return new Promise((resolve, reject) => {
            if (!window.indexedDB) {
                reject(new Error('IndexedDB not supported'));
                return;
            }

            const request = indexedDB.open(CACHE_CONFIG.dbName, CACHE_CONFIG.version);
            
            request.onerror = () => reject(request.error);
            request.onsuccess = () => {
                db = request.result;
                resolve(db);
            };
            
            request.onupgradeneeded = (event) => {
                const database = event.target.result;
                
                // Create object stores
                Object.values(CACHE_CONFIG.stores).forEach(storeName => {
                    if (!database.objectStoreNames.contains(storeName)) {
                        const store = database.createObjectStore(storeName, { keyPath: 'key' });
                        store.createIndex('timestamp', 'timestamp', { unique: false });
                        store.createIndex('size', 'size', { unique: false });
                        store.createIndex('accessCount', 'accessCount', { unique: false });
                        store.createIndex('priority', 'priority', { unique: false });
                    }
                });
                
                console.log('🔧 Cache database schema created/updated');
            };
        });
    }

    function set(key, data, options = {}) {
        return new Promise(async (resolve, reject) => {
            if (!db) {
                reject(new Error('Cache database not initialized'));
                return;
            }

            try {
                const storeName = options.store || CACHE_CONFIG.stores.general;
                const priority = options.priority || 'normal';
                const maxAge = options.maxAge || CACHE_CONFIG.maxAge;
                
                // Prepare data for storage
                const preparedData = await prepareDataForStorage(data, options);
                
                const cacheItem = {
                    key: key,
                    data: preparedData.data,
                    metadata: {
                        timestamp: Date.now(),
                        size: preparedData.size,
                        originalSize: preparedData.originalSize,
                        compression: preparedData.compression,
                        maxAge: maxAge,
                        priority: priority,
                        accessCount: 0,
                        mimeType: options.mimeType || 'application/json',
                        tags: options.tags || []
                    }
                };

                const transaction = db.transaction([storeName], 'readwrite');
                const store = transaction.objectStore(storeName);
                
                const request = store.put(cacheItem);
                
                request.onsuccess = () => {
                    updateCacheStats(preparedData.size, 1);
                    console.log(`💾 Cached item: ${key} (${formatBytes(preparedData.size)})`);
                    resolve(true);
                };
                
                request.onerror = () => reject(request.error);
                
                // Handle quota exceeded
                transaction.onabort = () => {
                    if (request.error && request.error.name === 'QuotaExceededError') {
                        console.warn('💾 Cache quota exceeded, performing cleanup...');
                        performEmergencyCleanup().then(() => {
                            // Retry the operation
                            set(key, data, options).then(resolve).catch(reject);
                        });
                    } else {
                        reject(request.error);
                    }
                };
                
            } catch (error) {
                reject(error);
            }
        });
    }

    function get(key, options = {}) {
        return new Promise((resolve, reject) => {
            if (!db) {
                reject(new Error('Cache database not initialized'));
                return;
            }

            const storeName = options.store || CACHE_CONFIG.stores.general;
            const transaction = db.transaction([storeName], 'readonly');
            const store = transaction.objectStore(storeName);
            
            const request = store.get(key);
            
            request.onsuccess = async () => {
                const result = request.result;
                
                if (!result) {
                    cacheStats.misses++;
                    resolve(null);
                    return;
                }
                
                // Check if item has expired
                if (isExpired(result)) {
                    // Remove expired item
                    remove(key, { store: storeName });
                    cacheStats.misses++;
                    resolve(null);
                    return;
                }
                
                try {
                    // Decompress data if needed
                    const decompressedData = await decompressData(result.data, result.metadata.compression);
                    
                    // Update access statistics
                    updateAccessCount(key, storeName);
                    cacheStats.hits++;
                    
                    console.log(`💾 Cache hit: ${key}`);
                    resolve(decompressedData);
                    
                } catch (error) {
                    console.error('💾 Cache decompression failed:', error);
                    cacheStats.misses++;
                    resolve(null);
                }
            };
            
            request.onerror = () => {
                cacheStats.misses++;
                reject(request.error);
            };
        });
    }

    function remove(key, options = {}) {
        return new Promise((resolve, reject) => {
            if (!db) {
                reject(new Error('Cache database not initialized'));
                return;
            }

            const storeName = options.store || CACHE_CONFIG.stores.general;
            const transaction = db.transaction([storeName], 'readwrite');
            const store = transaction.objectStore(storeName);
            
            // Get item size before deletion for stats update
            const getRequest = store.get(key);
            getRequest.onsuccess = () => {
                const item = getRequest.result;
                const size = item ? item.metadata.size : 0;
                
                const deleteRequest = store.delete(key);
                deleteRequest.onsuccess = () => {
                    updateCacheStats(-size, -1);
                    console.log(`💾 Removed from cache: ${key}`);
                    resolve(true);
                };
                deleteRequest.onerror = () => reject(deleteRequest.error);
            };
            getRequest.onerror = () => reject(getRequest.error);
        });
    }

    function clear(options = {}) {
        return new Promise((resolve, reject) => {
            if (!db) {
                reject(new Error('Cache database not initialized'));
                return;
            }

            const storesToClear = options.store ? [options.store] : Object.values(CACHE_CONFIG.stores);
            const promises = [];

            storesToClear.forEach(storeName => {
                const promise = new Promise((storeResolve, storeReject) => {
                    const transaction = db.transaction([storeName], 'readwrite');
                    const store = transaction.objectStore(storeName);
                    const request = store.clear();
                    
                    request.onsuccess = () => storeResolve();
                    request.onerror = () => storeReject(request.error);
                });
                promises.push(promise);
            });

            Promise.all(promises).then(() => {
                if (!options.store) {
                    // Reset stats if clearing all stores
                    cacheStats = {
                        hits: 0,
                        misses: 0,
                        totalSize: 0,
                        itemCount: 0,
                        compressionRatio: 0
                    };
                }
                console.log('💾 Cache cleared');
                resolve(true);
            }).catch(reject);
        });
    }

    function has(key, options = {}) {
        return get(key, options).then(result => result !== null);
    }

    function keys(options = {}) {
        return new Promise((resolve, reject) => {
            if (!db) {
                reject(new Error('Cache database not initialized'));
                return;
            }

            const storeName = options.store || CACHE_CONFIG.stores.general;
            const transaction = db.transaction([storeName], 'readonly');
            const store = transaction.objectStore(storeName);
            const request = store.getAllKeys();
            
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        });
    }

    function size(options = {}) {
        return new Promise((resolve, reject) => {
            if (!db) {
                reject(new Error('Cache database not initialized'));
                return;
            }

            const storeName = options.store || CACHE_CONFIG.stores.general;
            const transaction = db.transaction([storeName], 'readonly');
            const store = transaction.objectStore(storeName);
            const request = store.count();
            
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        });
    }

    async function prepareDataForStorage(data, options) {
        let serializedData;
        let originalSize;
        let compression = COMPRESSION_TYPES.NONE;

        // Serialize data
        if (typeof data === 'string') {
            serializedData = data;
        } else if (data instanceof ArrayBuffer) {
            serializedData = data;
        } else if (data instanceof Blob) {
            serializedData = await blobToArrayBuffer(data);
        } else {
            serializedData = JSON.stringify(data);
        }

        originalSize = getDataSize(serializedData);

        // Compress if data is large enough
        if (originalSize > CACHE_CONFIG.compressionThreshold && !options.noCompress) {
            try {
                if (typeof LZString !== 'undefined') {
                    const compressed = LZString.compress(serializedData);
                    if (compressed && compressed.length < serializedData.length) {
                        serializedData = compressed;
                        compression = COMPRESSION_TYPES.LZ_STRING;
                    }
                }
            } catch (error) {
                console.warn('💾 Compression failed, storing uncompressed:', error);
            }
        }

        const finalSize = getDataSize(serializedData);

        return {
            data: serializedData,
            size: finalSize,
            originalSize: originalSize,
            compression: compression
        };
    }

    async function decompressData(data, compressionType) {
        switch (compressionType) {
            case COMPRESSION_TYPES.LZ_STRING:
                if (typeof LZString !== 'undefined') {
                    return LZString.decompress(data);
                }
                break;
            case COMPRESSION_TYPES.NONE:
            default:
                return data;
        }
        return data;
    }

    function blobToArrayBuffer(blob) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result);
            reader.onerror = reject;
            reader.readAsArrayBuffer(blob);
        });
    }

    function getDataSize(data) {
        if (typeof data === 'string') {
            return new Blob([data]).size;
        } else if (data instanceof ArrayBuffer) {
            return data.byteLength;
        } else if (data && data.length !== undefined) {
            return data.length;
        }
        return 0;
    }

    function isExpired(cacheItem) {
        const now = Date.now();
        const age = now - cacheItem.metadata.timestamp;
        return age > cacheItem.metadata.maxAge;
    }

    function updateAccessCount(key, storeName) {
        const transaction = db.transaction([storeName], 'readwrite');
        const store = transaction.objectStore(storeName);
        
        const request = store.get(key);
        request.onsuccess = () => {
            const item = request.result;
            if (item) {
                item.metadata.accessCount++;
                item.metadata.lastAccessed = Date.now();
                store.put(item);
            }
        };
    }

    function updateCacheStats(sizeChange, countChange) {
        cacheStats.totalSize += sizeChange;
        cacheStats.itemCount += countChange;
        
        // Update compression ratio
        if (cacheStats.itemCount > 0) {
            // This is a simplified calculation
            cacheStats.compressionRatio = (cacheStats.totalSize / cacheStats.itemCount) * 0.7;
        }
    }

    function setupCleanupTimer() {
        setInterval(() => {
            performCleanup();
        }, CACHE_CONFIG.cleanupInterval);
    }

    function performCleanup() {
        console.log('🧹 Performing cache cleanup...');
        
        const promises = Object.values(CACHE_CONFIG.stores).map(storeName => {
            return cleanupStore(storeName);
        });
        
        Promise.all(promises).then(() => {
            console.log('✅ Cache cleanup completed');
        }).catch(error => {
            console.error('❌ Cache cleanup failed:', error);
        });
    }

    function cleanupStore(storeName) {
        return new Promise((resolve, reject) => {
            const transaction = db.transaction([storeName], 'readwrite');
            const store = transaction.objectStore(storeName);
            const request = store.openCursor();
            
            const itemsToDelete = [];
            let totalSize = 0;
            
            request.onsuccess = (event) => {
                const cursor = event.target.result;
                if (cursor) {
                    const item = cursor.value;
                    
                    // Check if expired
                    if (isExpired(item)) {
                        itemsToDelete.push(item.key);
                    } else {
                        totalSize += item.metadata.size;
                    }
                    
                    cursor.continue();
                } else {
                    // Delete expired items
                    const deletePromises = itemsToDelete.map(key => {
                        return new Promise((deleteResolve) => {
                            const deleteRequest = store.delete(key);
                            deleteRequest.onsuccess = () => deleteResolve();
                            deleteRequest.onerror = () => deleteResolve(); // Continue even if delete fails
                        });
                    });
                    
                    Promise.all(deletePromises).then(() => {
                        console.log(`🧹 Cleaned ${itemsToDelete.length} expired items from ${storeName}`);
                        resolve();
                    });
                }
            };
            
            request.onerror = () => reject(request.error);
        });
    }

    function performEmergencyCleanup() {
        console.log('🚨 Performing emergency cache cleanup...');
        
        return new Promise((resolve, reject) => {
            const promises = Object.values(CACHE_CONFIG.stores).map(storeName => {
                return emergencyCleanupStore(storeName);
            });
            
            Promise.all(promises).then(() => {
                console.log('✅ Emergency cleanup completed');
                resolve();
            }).catch(reject);
        });
    }

    function emergencyCleanupStore(storeName) {
        return new Promise((resolve, reject) => {
            const transaction = db.transaction([storeName], 'readwrite');
            const store = transaction.objectStore(storeName);
            const index = store.index('accessCount');
            const request = index.openCursor(null, 'next'); // Least accessed first
            
            let deletedCount = 0;
            const maxToDelete = 50; // Delete up to 50 least accessed items
            
            request.onsuccess = (event) => {
                const cursor = event.target.result;
                if (cursor && deletedCount < maxToDelete) {
                    const item = cursor.value;
                    
                    // Skip high priority items
                    if (item.metadata.priority !== 'high') {
                        const deleteRequest = store.delete(item.key);
                        deleteRequest.onsuccess = () => {
                            deletedCount++;
                            updateCacheStats(-item.metadata.size, -1);
                        };
                    }
                    
                    cursor.continue();
                } else {
                    console.log(`🚨 Emergency cleanup deleted ${deletedCount} items from ${storeName}`);
                    resolve();
                }
            };
            
            request.onerror = () => reject(request.error);
        });
    }

    function setupStorageQuotaMonitoring() {
        if ('storage' in navigator && 'estimate' in navigator.storage) {
            setInterval(async () => {
                try {
                    const estimate = await navigator.storage.estimate();
                    const usedPercent = (estimate.usage / estimate.quota) * 100;
                    
                    if (usedPercent > 80) {
                        console.warn(`💾 Storage quota ${usedPercent.toFixed(1)}% full`);
                        
                        if (usedPercent > 90) {
                            performEmergencyCleanup();
                        }
                    }
                } catch (error) {
                    console.warn('💾 Storage quota monitoring failed:', error);
                }
            }, 5 * 60 * 1000); // Check every 5 minutes
        }
    }

    function cacheSeismicData(key, data, options = {}) {
        return set(key, data, { ...options, store: CACHE_CONFIG.stores.seismic, priority: 'high' });
    }

    function cacheTomographyModel(key, data, options = {}) {
        return set(key, data, { ...options, store: CACHE_CONFIG.stores.tomography, priority: 'normal' });
    }

    function cacheHazardMap(key, data, options = {}) {
        return set(key, data, { ...options, store: CACHE_CONFIG.stores.hazard, priority: 'normal' });
    }

    function cacheRiskAssessment(key, data, options = {}) {
        return set(key, data, { ...options, store: CACHE_CONFIG.stores.risk, priority: 'normal' });
    }

    function cacheMeshData(key, data, options = {}) {
        return set(key, data, { ...options, store: CACHE_CONFIG.stores.mesh, priority: 'low' });
    }

    function cacheImage(key, data, options = {}) {
        return set(key, data, { ...options, store: CACHE_CONFIG.stores.images, priority: 'low' });
    }

    function getStats() {
        return { ...cacheStats };
    }

    function formatBytes(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    function createCacheUI() {
        const ui = document.createElement('div');
        ui.id = 'cache-monitor';
        ui.innerHTML = `
            <div class="cache-panel">
                <h4>💾 Cache System</h4>
                <div class="cache-stats">
                    <div class="stat">
                        <span>Hit Rate:</span>
                        <span id="cache-hit-rate">--</span>
                    </div>
                    <div class="stat">
                        <span>Total Size:</span>
                        <span id="cache-total-size">--</span>
                    </div>
                    <div class="stat">
                        <span>Items:</span>
                        <span id="cache-item-count">--</span>
                    </div>
                    <div class="stat">
                        <span>Compression:</span>
                        <span id="cache-compression">--</span>
                    </div>
                </div>
                <div class="cache-controls">
                    <button id="cache-cleanup">🧹 Cleanup</button>
                    <button id="cache-clear">🗑️ Clear All</button>
                    <button id="cache-stats-reset">🔄 Reset Stats</button>
                </div>
                <div class="cache-stores">
                    <h5>Cache Stores</h5>
                    <div id="cache-store-list"></div>
                </div>
            </div>
        `;
        
        ui.style.cssText = `
            position: fixed;
            top: 70px;
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
        
        // Update cache statistics display
        setInterval(updateCacheDisplay, 2000);
        
        // Control buttons
        document.getElementById('cache-cleanup').addEventListener('click', performCleanup);
        document.getElementById('cache-clear').addEventListener('click', () => {
            if (confirm('Clear all cached data?')) {
                clear();
            }
        });
        document.getElementById('cache-stats-reset').addEventListener('click', () => {
            cacheStats.hits = 0;
            cacheStats.misses = 0;
        });
    }

    function updateCacheDisplay() {
        const hitRate = cacheStats.hits + cacheStats.misses > 0 ? 
            (cacheStats.hits / (cacheStats.hits + cacheStats.misses) * 100).toFixed(1) + '%' : '--';
        
        document.getElementById('cache-hit-rate').textContent = hitRate;
        document.getElementById('cache-total-size').textContent = formatBytes(cacheStats.totalSize);
        document.getElementById('cache-item-count').textContent = cacheStats.itemCount;
        document.getElementById('cache-compression').textContent = 
            cacheStats.compressionRatio > 0 ? (cacheStats.compressionRatio * 100).toFixed(1) + '%' : '--';
        
        // Update store information
        updateStoreDisplay();
    }

    async function updateStoreDisplay() {
        const storeList = document.getElementById('cache-store-list');
        if (!storeList) return;
        
        storeList.innerHTML = '';
        
        for (const [name, storeName] of Object.entries(CACHE_CONFIG.stores)) {
            try {
                const count = await size({ store: storeName });
                const item = document.createElement('div');
                item.innerHTML = `<span>${name}:</span> <span>${count} items</span>`;
                item.style.cssText = 'display: flex; justify-content: space-between; margin: 2px 0;';
                storeList.appendChild(item);
            } catch (error) {
                // Store might not exist yet
            }
        }
    }

    function showCacheMonitor() {
        const monitor = document.getElementById('cache-monitor');
        if (monitor) {
            monitor.style.display = 'block';
            updateCacheDisplay();
        }
    }

    function hideCacheMonitor() {
        const monitor = document.getElementById('cache-monitor');
        if (monitor) {
            monitor.style.display = 'none';
        }
    }

    // Keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        if (e.ctrlKey && e.altKey && e.key === 'C') {
            e.preventDefault();
            showCacheMonitor();
        }
        if (e.key === 'Escape') {
            hideCacheMonitor();
        }
    });

    // Cleanup function for memory management
    function cleanup() {
        return performEmergencyCleanup();
    }

    // Public API
    return {
        init: init,
        set: set,
        get: get,
        remove: remove,
        clear: clear,
        has: has,
        keys: keys,
        size: size,
        cleanup: cleanup,
        getStats: getStats,
        showCacheMonitor: showCacheMonitor,
        hideCacheMonitor: hideCacheMonitor,
        // Specialized cache methods
        cacheSeismicData: cacheSeismicData,
        cacheTomographyModel: cacheTomographyModel,
        cacheHazardMap: cacheHazardMap,
        cacheRiskAssessment: cacheRiskAssessment,
        cacheMeshData: cacheMeshData,
        cacheImage: cacheImage,
        // Configuration
        CACHE_CONFIG: CACHE_CONFIG,
        COMPRESSION_TYPES: COMPRESSION_TYPES
    };

})();

// Auto-initialize
document.addEventListener('DOMContentLoaded', function() {
    window.CACHE_SYSTEM.init().then(success => {
        if (success) {
            console.log('💾 Advanced Cache System loaded');
            console.log('🎮 Press Ctrl+Alt+C to open Cache Monitor');
        }
    });
});
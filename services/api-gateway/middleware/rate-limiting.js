const rateLimit = require('express-rate-limit');
const RedisStore = require('rate-limit-redis');
const redis = require('redis');
const winston = require('winston');
const crypto = require('crypto');

// Logger configuration
const logger = winston.createLogger({
    level: 'info',
    format: winston.format.combine(
        winston.format.timestamp(),
        winston.format.errors({ stack: true }),
        winston.format.json()
    ),
    transports: [
        new winston.transports.Console(),
        new winston.transports.File({ filename: 'logs/rate-limiting.log' })
    ]
});

// Redis client for distributed rate limiting
const redisClient = redis.createClient({
    host: process.env.REDIS_HOST || 'localhost',
    port: process.env.REDIS_PORT || 6379,
    password: process.env.REDIS_PASSWORD,
    db: 1 // Use database 1 for rate limiting
});

redisClient.on('error', (err) => {
    logger.error('Redis connection error for rate limiting:', err);
});

redisClient.on('connect', () => {
    logger.info('Connected to Redis for rate limiting');
});

// Usage tracking storage
const usageStats = new Map();

// Rate limiting configurations for different API tiers
const RATE_LIMIT_CONFIGS = {
    // Free tier - basic access
    free: {
        requests: 100,
        window: 15 * 60 * 1000, // 15 minutes
        message: 'Free tier rate limit exceeded. Upgrade for higher limits.',
        skipSuccessfulRequests: false,
        skipFailedRequests: false
    },
    
    // Basic tier - research institutions
    basic: {
        requests: 1000,
        window: 15 * 60 * 1000, // 15 minutes
        message: 'Basic tier rate limit exceeded. Contact support for higher limits.',
        skipSuccessfulRequests: false,
        skipFailedRequests: false
    },
    
    // Premium tier - commercial users
    premium: {
        requests: 10000,
        window: 15 * 60 * 1000, // 15 minutes
        message: 'Premium tier rate limit exceeded. Contact support.',
        skipSuccessfulRequests: false,
        skipFailedRequests: false
    },
    
    // Enterprise tier - unlimited within reason
    enterprise: {
        requests: 100000,
        window: 15 * 60 * 1000, // 15 minutes
        message: 'Enterprise tier rate limit exceeded. Contact your account manager.',
        skipSuccessfulRequests: false,
        skipFailedRequests: false
    },
    
    // Special limits for different endpoints
    data_upload: {
        requests: 50,
        window: 60 * 1000, // 1 minute
        message: 'Data upload rate limit exceeded. Please wait before uploading more data.',
        skipSuccessfulRequests: false,
        skipFailedRequests: true
    },
    
    simulation_create: {
        requests: 10,
        window: 60 * 60 * 1000, // 1 hour
        message: 'Simulation creation rate limit exceeded. Please wait before creating more simulations.',
        skipSuccessfulRequests: false,
        skipFailedRequests: true
    },
    
    monitoring_alerts: {
        requests: 1000,
        window: 60 * 1000, // 1 minute
        message: 'Monitoring alerts rate limit exceeded.',
        skipSuccessfulRequests: true,
        skipFailedRequests: false
    }
};

// Function to determine user tier based on API key or JWT token
const getUserTier = (req) => {
    // Check for API key tier
    if (req.apiKeyHash) {
        // In production, this would be looked up from database
        const tierMapping = {
            [process.env.USGS_API_KEY_HASH]: 'enterprise',
            [process.env.NASA_API_KEY_HASH]: 'enterprise',
            [process.env.ESA_API_KEY_HASH]: 'premium',
            [process.env.PARTNER_API_KEY_HASH]: 'basic'
        };
        return tierMapping[req.apiKeyHash] || 'free';
    }
    
    // Check for JWT token tier
    if (req.user && req.user.tier) {
        return req.user.tier;
    }
    
    // Default to free tier
    return 'free';
};

// Function to get rate limit key
const getRateLimitKey = (req, endpoint = 'general') => {
    let identifier;
    
    if (req.apiKeyHash) {
        identifier = `api_key:${req.apiKeyHash}`;
    } else if (req.user && req.user.id) {
        identifier = `user:${req.user.id}`;
    } else {
        identifier = `ip:${req.ip}`;
    }
    
    return `rate_limit:${endpoint}:${identifier}`;
};

// Usage tracking middleware
const trackUsage = (endpoint, tier) => {
    return (req, res, next) => {
        const startTime = Date.now();
        const originalSend = res.send;
        
        res.send = function(data) {
            const endTime = Date.now();
            const responseTime = endTime - startTime;
            
            // Track usage statistics
            const key = getRateLimitKey(req, endpoint);
            const usageKey = `usage:${key}:${new Date().toISOString().split('T')[0]}`; // Daily usage
            
            const usageData = {
                endpoint,
                tier,
                timestamp: new Date(),
                responseTime,
                statusCode: res.statusCode,
                requestSize: req.get('content-length') || 0,
                responseSize: Buffer.byteLength(data || ''),
                userAgent: req.get('user-agent'),
                ip: req.ip
            };
            
            // Store in Redis for persistence
            redisClient.hincrby(usageKey, 'requests', 1);
            redisClient.hincrby(usageKey, 'responseTime', responseTime);
            redisClient.hincrby(usageKey, 'requestSize', usageData.requestSize);
            redisClient.hincrby(usageKey, 'responseSize', usageData.responseSize);
            redisClient.expire(usageKey, 30 * 24 * 60 * 60); // Keep for 30 days
            
            // Store in memory for quick access
            if (!usageStats.has(usageKey)) {
                usageStats.set(usageKey, {
                    requests: 0,
                    totalResponseTime: 0,
                    totalRequestSize: 0,
                    totalResponseSize: 0,
                    errors: 0
                });
            }
            
            const stats = usageStats.get(usageKey);
            stats.requests++;
            stats.totalResponseTime += responseTime;
            stats.totalRequestSize += usageData.requestSize;
            stats.totalResponseSize += usageData.responseSize;
            
            if (res.statusCode >= 400) {
                stats.errors++;
                redisClient.hincrby(usageKey, 'errors', 1);
            }
            
            // Log usage for monitoring
            logger.info('API Usage', {
                endpoint,
                tier,
                key: key.substring(0, 20) + '...',
                responseTime,
                statusCode: res.statusCode,
                requestSize: usageData.requestSize,
                responseSize: usageData.responseSize
            });
            
            originalSend.call(this, data);
        };
        
        next();
    };
};

// Create rate limiter with custom configuration
const createRateLimiter = (configName, endpoint = 'general') => {
    const config = RATE_LIMIT_CONFIGS[configName];
    
    if (!config) {
        throw new Error(`Rate limit configuration '${configName}' not found`);
    }
    
    return rateLimit({
        store: new RedisStore({
            client: redisClient,
            prefix: `rl:${endpoint}:`
        }),
        windowMs: config.window,
        max: (req) => {
            const tier = getUserTier(req);
            const tierConfig = RATE_LIMIT_CONFIGS[tier];
            return tierConfig ? tierConfig.requests : config.requests;
        },
        message: (req) => {
            const tier = getUserTier(req);
            const tierConfig = RATE_LIMIT_CONFIGS[tier];
            return {
                error: 'rate_limit_exceeded',
                message: tierConfig ? tierConfig.message : config.message,
                tier: tier,
                limit: tierConfig ? tierConfig.requests : config.requests,
                window: config.window / 1000,
                retryAfter: Math.ceil(config.window / 1000)
            };
        },
        standardHeaders: true,
        legacyHeaders: false,
        keyGenerator: (req) => getRateLimitKey(req, endpoint),
        skipSuccessfulRequests: config.skipSuccessfulRequests,
        skipFailedRequests: config.skipFailedRequests,
        handler: (req, res, next) => {
            const tier = getUserTier(req);
            const key = getRateLimitKey(req, endpoint);
            
            logger.warn('Rate limit exceeded', {
                endpoint,
                tier,
                key: key.substring(0, 20) + '...',
                ip: req.ip,
                userAgent: req.get('user-agent')
            });
            
            // Track rate limit violations
            const violationKey = `violations:${key}:${new Date().toISOString().split('T')[0]}`;
            redisClient.incr(violationKey);
            redisClient.expire(violationKey, 24 * 60 * 60); // Keep for 24 hours
            
            const tierConfig = RATE_LIMIT_CONFIGS[tier] || config;
            res.status(429).json({
                error: 'rate_limit_exceeded',
                message: tierConfig.message,
                tier: tier,
                limit: tierConfig.requests,
                window: config.window / 1000,
                retryAfter: Math.ceil(config.window / 1000)
            });
        },
        onLimitReached: (req, res, options) => {
            const tier = getUserTier(req);
            const key = getRateLimitKey(req, endpoint);
            
            logger.warn('Rate limit reached', {
                endpoint,
                tier,
                key: key.substring(0, 20) + '...',
                limit: options.max,
                window: options.windowMs
            });
        }
    });
};

// Predefined rate limiters for common use cases
const rateLimiters = {
    general: createRateLimiter('free', 'general'),
    dataUpload: createRateLimiter('data_upload', 'data_upload'),
    simulationCreate: createRateLimiter('simulation_create', 'simulation_create'),
    monitoringAlerts: createRateLimiter('monitoring_alerts', 'monitoring_alerts')
};

// Dynamic rate limiter that adapts based on user tier
const dynamicRateLimit = (endpoint = 'general') => {
    return (req, res, next) => {
        const tier = getUserTier(req);
        const config = RATE_LIMIT_CONFIGS[tier];
        
        if (!config) {
            return next();
        }
        
        const limiter = createRateLimiter(tier, endpoint);
        limiter(req, res, next);
    };
};

// Usage analytics endpoint
const getUsageAnalytics = async (req, res) => {
    try {
        const { period = 'day', endpoint, tier } = req.query;
        const key = getRateLimitKey(req, endpoint);
        
        let dateKey;
        switch (period) {
            case 'hour':
                dateKey = new Date().toISOString().split(':')[0];
                break;
            case 'day':
                dateKey = new Date().toISOString().split('T')[0];
                break;
            case 'month':
                dateKey = new Date().toISOString().split('-').slice(0, 2).join('-');
                break;
            default:
                dateKey = new Date().toISOString().split('T')[0];
        }
        
        const usageKey = `usage:${key}:${dateKey}`;
        const usage = await redisClient.hgetall(usageKey);
        
        const analytics = {
            period,
            endpoint,
            tier: getUserTier(req),
            usage: {
                requests: parseInt(usage.requests || 0),
                averageResponseTime: usage.requests ? 
                    Math.round(parseInt(usage.responseTime || 0) / parseInt(usage.requests)) : 0,
                totalRequestSize: parseInt(usage.requestSize || 0),
                totalResponseSize: parseInt(usage.responseSize || 0),
                errors: parseInt(usage.errors || 0),
                errorRate: usage.requests ? 
                    (parseInt(usage.errors || 0) / parseInt(usage.requests) * 100).toFixed(2) + '%' : '0%'
            },
            limits: RATE_LIMIT_CONFIGS[getUserTier(req)] || RATE_LIMIT_CONFIGS.free
        };
        
        res.json(analytics);
    } catch (error) {
        logger.error('Error retrieving usage analytics:', error);
        res.status(500).json({ error: 'Failed to retrieve usage analytics' });
    }
};

// Quota management
const checkQuota = (quotaType) => {
    return async (req, res, next) => {
        try {
            const tier = getUserTier(req);
            const key = getRateLimitKey(req, quotaType);
            const quotaKey = `quota:${key}:${new Date().toISOString().split('-').slice(0, 2).join('-')}`; // Monthly quota
            
            // Define monthly quotas by tier
            const monthlyQuotas = {
                free: { requests: 10000, data_upload: 100, simulations: 10 },
                basic: { requests: 100000, data_upload: 1000, simulations: 100 },
                premium: { requests: 1000000, data_upload: 10000, simulations: 1000 },
                enterprise: { requests: 10000000, data_upload: 100000, simulations: 10000 }
            };
            
            const quota = monthlyQuotas[tier] || monthlyQuotas.free;
            const currentUsage = await redisClient.get(quotaKey) || 0;
            
            if (parseInt(currentUsage) >= quota[quotaType]) {
                logger.warn('Monthly quota exceeded', {
                    quotaType,
                    tier,
                    key: key.substring(0, 20) + '...',
                    currentUsage,
                    quota: quota[quotaType]
                });
                
                return res.status(429).json({
                    error: 'quota_exceeded',
                    message: `Monthly ${quotaType} quota exceeded for ${tier} tier`,
                    tier,
                    quota: quota[quotaType],
                    currentUsage: parseInt(currentUsage),
                    resetDate: new Date(new Date().getFullYear(), new Date().getMonth() + 1, 1)
                });
            }
            
            // Increment quota usage
            await redisClient.incr(quotaKey);
            await redisClient.expire(quotaKey, 31 * 24 * 60 * 60); // Expire at end of month
            
            next();
        } catch (error) {
            logger.error('Error checking quota:', error);
            next(); // Continue on error to avoid blocking requests
        }
    };
};

// Health check for rate limiting system
const healthCheck = async (req, res) => {
    try {
        // Test Redis connection
        await redisClient.ping();
        
        // Get some basic stats
        const stats = {
            status: 'healthy',
            redis: 'connected',
            activeConfigs: Object.keys(RATE_LIMIT_CONFIGS).length,
            memoryUsage: process.memoryUsage(),
            timestamp: new Date().toISOString()
        };
        
        res.json(stats);
    } catch (error) {
        logger.error('Rate limiting health check failed:', error);
        res.status(500).json({
            status: 'unhealthy',
            error: error.message,
            timestamp: new Date().toISOString()
        });
    }
};

module.exports = {
    rateLimiters,
    createRateLimiter,
    dynamicRateLimit,
    trackUsage,
    getUserTier,
    getRateLimitKey,
    getUsageAnalytics,
    checkQuota,
    healthCheck,
    RATE_LIMIT_CONFIGS
};
const express = require('express');
const rateLimit = require('express-rate-limit');
const helmet = require('helmet');
const cors = require('cors');
const jwt = require('jsonwebtoken');
const crypto = require('crypto');
const { body, validationResult, query } = require('express-validator');
const winston = require('winston');

const router = express.Router();

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
        new winston.transports.File({ filename: 'logs/external-api.log' })
    ]
});

// Security middleware
router.use(helmet({
    contentSecurityPolicy: {
        directives: {
            defaultSrc: ["'self'"],
            styleSrc: ["'self'", "'unsafe-inline'"],
            scriptSrc: ["'self'"],
            imgSrc: ["'self'", "data:", "https:"],
        },
    },
    hsts: {
        maxAge: 31536000,
        includeSubDomains: true,
        preload: true
    }
}));

// CORS configuration for external APIs
const corsOptions = {
    origin: function (origin, callback) {
        // Allow requests from approved domains
        const allowedOrigins = [
            'https://usgs.gov',
            'https://nasa.gov',
            'https://esa.int',
            'https://partner-institutions.edu',
            process.env.FRONTEND_URL
        ];
        
        if (!origin || allowedOrigins.includes(origin)) {
            callback(null, true);
        } else {
            callback(new Error('Not allowed by CORS'));
        }
    },
    credentials: true,
    methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization', 'X-API-Key', 'X-Request-ID']
};

router.use(cors(corsOptions));

// Rate limiting for external APIs
const createRateLimit = (windowMs, max, message) => rateLimit({
    windowMs,
    max,
    message: { error: message },
    standardHeaders: true,
    legacyHeaders: false,
    handler: (req, res) => {
        logger.warn(`Rate limit exceeded for IP: ${req.ip}, endpoint: ${req.path}`);
        res.status(429).json({ error: message });
    }
});

// Different rate limits for different endpoints
const generalRateLimit = createRateLimit(15 * 60 * 1000, 100, 'Too many requests, please try again later');
const dataRateLimit = createRateLimit(60 * 1000, 10, 'Data request rate limit exceeded');
const uploadRateLimit = createRateLimit(60 * 1000, 5, 'Upload rate limit exceeded');

// API Key authentication middleware
const authenticateAPIKey = async (req, res, next) => {
    try {
        const apiKey = req.headers['x-api-key'];
        
        if (!apiKey) {
            return res.status(401).json({ error: 'API key required' });
        }

        // Validate API key format and check against database
        const keyHash = crypto.createHash('sha256').update(apiKey).digest('hex');
        
        // TODO: Replace with actual database lookup
        const validKeys = new Set([
            process.env.USGS_API_KEY_HASH,
            process.env.NASA_API_KEY_HASH,
            process.env.ESA_API_KEY_HASH,
            process.env.PARTNER_API_KEY_HASH
        ]);

        if (!validKeys.has(keyHash)) {
            logger.warn(`Invalid API key attempt from IP: ${req.ip}`);
            return res.status(401).json({ error: 'Invalid API key' });
        }

        // Add API key info to request for logging
        req.apiKeyHash = keyHash;
        req.requestId = crypto.randomUUID();
        
        logger.info(`API request authenticated`, {
            requestId: req.requestId,
            endpoint: req.path,
            method: req.method,
            ip: req.ip,
            userAgent: req.get('User-Agent')
        });

        next();
    } catch (error) {
        logger.error('API key authentication error:', error);
        res.status(500).json({ error: 'Authentication error' });
    }
};

// OAuth2 JWT authentication middleware
const authenticateJWT = (req, res, next) => {
    const authHeader = req.headers.authorization;
    const token = authHeader && authHeader.split(' ')[1];

    if (!token) {
        return res.status(401).json({ error: 'Access token required' });
    }

    jwt.verify(token, process.env.JWT_SECRET, (err, user) => {
        if (err) {
            logger.warn(`Invalid JWT token from IP: ${req.ip}`);
            return res.status(403).json({ error: 'Invalid or expired token' });
        }

        req.user = user;
        req.requestId = crypto.randomUUID();
        
        logger.info(`JWT request authenticated`, {
            requestId: req.requestId,
            userId: user.id,
            endpoint: req.path,
            method: req.method,
            ip: req.ip
        });

        next();
    });
};

// Request validation middleware
const validateRequest = (req, res, next) => {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
        logger.warn(`Request validation failed`, {
            requestId: req.requestId,
            errors: errors.array(),
            endpoint: req.path
        });
        return res.status(400).json({
            error: 'Validation failed',
            details: errors.array()
        });
    }
    next();
};

// Geological Data API Endpoints

/**
 * @swagger
 * /api/external/v1/geological-data:
 *   get:
 *     summary: Retrieve geological data
 *     tags: [Geological Data]
 *     security:
 *       - ApiKeyAuth: []
 *     parameters:
 *       - in: query
 *         name: type
 *         schema:
 *           type: string
 *           enum: [seismic, deformation, geochemical, thermal]
 *         required: true
 *         description: Type of geological data
 *       - in: query
 *         name: start_time
 *         schema:
 *           type: string
 *           format: date-time
 *         description: Start time for data query
 *       - in: query
 *         name: end_time
 *         schema:
 *           type: string
 *           format: date-time
 *         description: End time for data query
 *       - in: query
 *         name: bbox
 *         schema:
 *           type: string
 *         description: Bounding box (min_lon,min_lat,max_lon,max_lat)
 *       - in: query
 *         name: limit
 *         schema:
 *           type: integer
 *           minimum: 1
 *           maximum: 1000
 *         description: Maximum number of records to return
 *     responses:
 *       200:
 *         description: Geological data retrieved successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 data:
 *                   type: array
 *                   items:
 *                     $ref: '#/components/schemas/GeologicalData'
 *                 metadata:
 *                   $ref: '#/components/schemas/DataMetadata'
 */
router.get('/geological-data',
    generalRateLimit,
    authenticateAPIKey,
    [
        query('type').isIn(['seismic', 'deformation', 'geochemical', 'thermal']),
        query('start_time').optional().isISO8601(),
        query('end_time').optional().isISO8601(),
        query('bbox').optional().matches(/^-?\d+\.?\d*,-?\d+\.?\d*,-?\d+\.?\d*,-?\d+\.?\d*$/),
        query('limit').optional().isInt({ min: 1, max: 1000 })
    ],
    validateRequest,
    async (req, res) => {
        try {
            const { type, start_time, end_time, bbox, limit = 100 } = req.query;
            
            // Parse bounding box if provided
            let boundingBox = null;
            if (bbox) {
                const coords = bbox.split(',').map(parseFloat);
                boundingBox = {
                    min_longitude: coords[0],
                    min_latitude: coords[1],
                    max_longitude: coords[2],
                    max_latitude: coords[3]
                };
            }

            // Query geological data service
            const geologicalDataService = req.app.get('geologicalDataService');
            const data = await geologicalDataService.queryData({
                type,
                startTime: start_time,
                endTime: end_time,
                boundingBox,
                limit: parseInt(limit)
            });

            const response = {
                data: data.records,
                metadata: {
                    total_count: data.totalCount,
                    returned_count: data.records.length,
                    query_time: new Date().toISOString(),
                    data_type: type,
                    time_range: {
                        start: start_time,
                        end: end_time
                    },
                    spatial_bounds: boundingBox
                }
            };

            logger.info(`Geological data query completed`, {
                requestId: req.requestId,
                dataType: type,
                recordCount: data.records.length,
                queryTime: response.metadata.query_time
            });

            res.json(response);
        } catch (error) {
            logger.error(`Geological data query error:`, {
                requestId: req.requestId,
                error: error.message,
                stack: error.stack
            });
            res.status(500).json({ error: 'Internal server error' });
        }
    }
);

/**
 * @swagger
 * /api/external/v1/geological-data:
 *   post:
 *     summary: Submit geological data
 *     tags: [Geological Data]
 *     security:
 *       - ApiKeyAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             properties:
 *               data_type:
 *                 type: string
 *                 enum: [seismic, deformation, geochemical, thermal]
 *               source:
 *                 type: string
 *               records:
 *                 type: array
 *                 items:
 *                   $ref: '#/components/schemas/GeologicalDataRecord'
 *     responses:
 *       201:
 *         description: Data submitted successfully
 */
router.post('/geological-data',
    uploadRateLimit,
    authenticateAPIKey,
    [
        body('data_type').isIn(['seismic', 'deformation', 'geochemical', 'thermal']),
        body('source').isString().isLength({ min: 1, max: 100 }),
        body('records').isArray({ min: 1, max: 1000 }),
        body('records.*.timestamp').isISO8601(),
        body('records.*.location.latitude').isFloat({ min: -90, max: 90 }),
        body('records.*.location.longitude').isFloat({ min: -180, max: 180 })
    ],
    validateRequest,
    async (req, res) => {
        try {
            const { data_type, source, records } = req.body;

            // Validate and sanitize data
            const sanitizedRecords = records.map(record => ({
                ...record,
                timestamp: new Date(record.timestamp),
                source: source,
                data_type: data_type,
                submitted_at: new Date(),
                submitted_by: req.apiKeyHash
            }));

            // Submit to data ingestion service
            const dataIngestionService = req.app.get('dataIngestionService');
            const result = await dataIngestionService.ingestData({
                dataType: data_type,
                source: source,
                records: sanitizedRecords
            });

            logger.info(`Data submission completed`, {
                requestId: req.requestId,
                dataType: data_type,
                source: source,
                recordCount: records.length,
                ingestionId: result.ingestionId
            });

            res.status(201).json({
                message: 'Data submitted successfully',
                ingestion_id: result.ingestionId,
                processed_count: result.processedCount,
                rejected_count: result.rejectedCount
            });
        } catch (error) {
            logger.error(`Data submission error:`, {
                requestId: req.requestId,
                error: error.message,
                stack: error.stack
            });
            res.status(500).json({ error: 'Data submission failed' });
        }
    }
);

// Simulation API Endpoints

/**
 * @swagger
 * /api/external/v1/simulations:
 *   post:
 *     summary: Create a new geological simulation
 *     tags: [Simulations]
 *     security:
 *       - BearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             $ref: '#/components/schemas/SimulationRequest'
 *     responses:
 *       202:
 *         description: Simulation created and queued
 */
router.post('/simulations',
    generalRateLimit,
    authenticateJWT,
    [
        body('simulation_type').isIn(['long_term_geological', 'volcanic_monitoring', 'earthquake_forecasting']),
        body('parameters').isObject(),
        body('time_span').optional().isObject(),
        body('spatial_domain').optional().isObject()
    ],
    validateRequest,
    async (req, res) => {
        try {
            const simulationRequest = {
                ...req.body,
                user_id: req.user.id,
                created_at: new Date(),
                status: 'queued'
            };

            const simulationService = req.app.get('simulationService');
            const simulation = await simulationService.createSimulation(simulationRequest);

            logger.info(`Simulation created`, {
                requestId: req.requestId,
                simulationId: simulation.id,
                userId: req.user.id,
                simulationType: req.body.simulation_type
            });

            res.status(202).json({
                simulation_id: simulation.id,
                status: simulation.status,
                estimated_completion: simulation.estimatedCompletion,
                message: 'Simulation queued successfully'
            });
        } catch (error) {
            logger.error(`Simulation creation error:`, {
                requestId: req.requestId,
                userId: req.user.id,
                error: error.message
            });
            res.status(500).json({ error: 'Simulation creation failed' });
        }
    }
);

/**
 * @swagger
 * /api/external/v1/simulations/{id}:
 *   get:
 *     summary: Get simulation status and results
 *     tags: [Simulations]
 *     security:
 *       - BearerAuth: []
 *     parameters:
 *       - in: path
 *         name: id
 *         required: true
 *         schema:
 *           type: string
 *         description: Simulation ID
 *     responses:
 *       200:
 *         description: Simulation details retrieved
 */
router.get('/simulations/:id',
    generalRateLimit,
    authenticateJWT,
    async (req, res) => {
        try {
            const { id } = req.params;
            
            const simulationService = req.app.get('simulationService');
            const simulation = await simulationService.getSimulation(id, req.user.id);

            if (!simulation) {
                return res.status(404).json({ error: 'Simulation not found' });
            }

            logger.info(`Simulation retrieved`, {
                requestId: req.requestId,
                simulationId: id,
                userId: req.user.id,
                status: simulation.status
            });

            res.json({
                simulation_id: simulation.id,
                status: simulation.status,
                progress: simulation.progress,
                created_at: simulation.createdAt,
                completed_at: simulation.completedAt,
                results: simulation.results,
                error_message: simulation.errorMessage
            });
        } catch (error) {
            logger.error(`Simulation retrieval error:`, {
                requestId: req.requestId,
                simulationId: req.params.id,
                userId: req.user.id,
                error: error.message
            });
            res.status(500).json({ error: 'Failed to retrieve simulation' });
        }
    }
);

// Real-time Monitoring API Endpoints

/**
 * @swagger
 * /api/external/v1/monitoring/alerts:
 *   get:
 *     summary: Get current volcanic alerts
 *     tags: [Monitoring]
 *     security:
 *       - ApiKeyAuth: []
 *     parameters:
 *       - in: query
 *         name: volcano_id
 *         schema:
 *           type: string
 *         description: Specific volcano ID
 *       - in: query
 *         name: alert_level
 *         schema:
 *           type: integer
 *           minimum: 0
 *           maximum: 5
 *         description: Minimum alert level
 *     responses:
 *       200:
 *         description: Current alerts retrieved
 */
router.get('/monitoring/alerts',
    dataRateLimit,
    authenticateAPIKey,
    [
        query('volcano_id').optional().isString(),
        query('alert_level').optional().isInt({ min: 0, max: 5 })
    ],
    validateRequest,
    async (req, res) => {
        try {
            const { volcano_id, alert_level } = req.query;

            const monitoringService = req.app.get('monitoringService');
            const alerts = await monitoringService.getCurrentAlerts({
                volcanoId: volcano_id,
                minAlertLevel: alert_level ? parseInt(alert_level) : 0
            });

            logger.info(`Monitoring alerts retrieved`, {
                requestId: req.requestId,
                volcanoId: volcano_id,
                alertCount: alerts.length
            });

            res.json({
                alerts: alerts,
                timestamp: new Date().toISOString(),
                total_count: alerts.length
            });
        } catch (error) {
            logger.error(`Monitoring alerts error:`, {
                requestId: req.requestId,
                error: error.message
            });
            res.status(500).json({ error: 'Failed to retrieve alerts' });
        }
    }
);

// Health check endpoint
router.get('/health', (req, res) => {
    res.json({
        status: 'healthy',
        timestamp: new Date().toISOString(),
        version: process.env.API_VERSION || '1.0.0'
    });
});

// Error handling middleware
router.use((error, req, res, next) => {
    logger.error(`Unhandled error in external API:`, {
        requestId: req.requestId,
        error: error.message,
        stack: error.stack,
        endpoint: req.path,
        method: req.method
    });

    res.status(500).json({
        error: 'Internal server error',
        request_id: req.requestId
    });
});

module.exports = router;
const express = require('express');
const helmet = require('helmet');
const cors = require('cors');
const compression = require('compression');
const morgan = require('morgan');
const winston = require('winston');
const swaggerUi = require('swagger-ui-express');
const swaggerJsdoc = require('swagger-jsdoc');
const { createGraphQLServer } = require('./api/external/graphql_api');
const restAPI = require('./api/external/rest_api');
const oauth2Router = require('./auth/oauth2');
const { rateLimiters, dynamicRateLimit, trackUsage, getUsageAnalytics, checkQuota, healthCheck } = require('./middleware/rate-limiting');
const { encryptionService, encryptionMiddleware } = require('./security/encryption');

// Initialize Express app
const app = express();
const PORT = process.env.PORT || 8080;

// Logger configuration
const logger = winston.createLogger({
    level: process.env.LOG_LEVEL || 'info',
    format: winston.format.combine(
        winston.format.timestamp(),
        winston.format.errors({ stack: true }),
        winston.format.json()
    ),
    transports: [
        new winston.transports.Console({
            format: winston.format.combine(
                winston.format.colorize(),
                winston.format.simple()
            )
        }),
        new winston.transports.File({ filename: 'logs/api-gateway.log' }),
        new winston.transports.File({ 
            filename: 'logs/api-gateway-error.log', 
            level: 'error' 
        })
    ]
});

// Swagger configuration
const swaggerOptions = {
    definition: {
        openapi: '3.0.0',
        info: {
            title: 'Prometheus AI Geodynamic Platform API',
            version: '1.0.0',
            description: 'External API for geological data integration and simulation services',
            contact: {
                name: 'Prometheus AI Team',
                email: 'api-support@prometheus-ai.com',
                url: 'https://prometheus-ai.com'
            },
            license: {
                name: 'MIT',
                url: 'https://opensource.org/licenses/MIT'
            }
        },
        servers: [
            {
                url: process.env.API_BASE_URL || 'http://localhost:8080',
                description: 'Development server'
            },
            {
                url: 'https://api.prometheus-ai.com',
                description: 'Production server'
            }
        ],
        components: {
            securitySchemes: {
                BearerAuth: {
                    type: 'http',
                    scheme: 'bearer',
                    bearerFormat: 'JWT'
                },
                ApiKeyAuth: {
                    type: 'apiKey',
                    in: 'header',
                    name: 'X-API-Key'
                },
                OAuth2: {
                    type: 'oauth2',
                    flows: {
                        authorizationCode: {
                            authorizationUrl: '/oauth2/authorize',
                            tokenUrl: '/oauth2/token',
                            scopes: {
                                'read:geological-data': 'Read geological data',
                                'write:geological-data': 'Write geological data',
                                'read:simulations': 'Read simulation results',
                                'write:simulations': 'Create and manage simulations',
                                'read:monitoring': 'Read monitoring data',
                                'write:monitoring': 'Write monitoring data',
                                'admin:system': 'System administration'
                            }
                        }
                    }
                }
            },
            schemas: {
                GeologicalData: {
                    type: 'object',
                    properties: {
                        id: { type: 'string' },
                        timestamp: { type: 'string', format: 'date-time' },
                        location: { $ref: '#/components/schemas/Location' },
                        dataType: { type: 'string', enum: ['seismic', 'deformation', 'geochemical', 'thermal'] },
                        measurements: { type: 'object' },
                        quality: { type: 'string' },
                        source: { type: 'string' }
                    }
                },
                Location: {
                    type: 'object',
                    properties: {
                        latitude: { type: 'number', minimum: -90, maximum: 90 },
                        longitude: { type: 'number', minimum: -180, maximum: 180 },
                        elevation: { type: 'number' },
                        depth: { type: 'number' }
                    },
                    required: ['latitude', 'longitude']
                },
                DataMetadata: {
                    type: 'object',
                    properties: {
                        totalCount: { type: 'integer' },
                        queryTime: { type: 'string', format: 'date-time' },
                        dataSource: { type: 'string' },
                        qualityMetrics: { type: 'object' }
                    }
                },
                SimulationRequest: {
                    type: 'object',
                    properties: {
                        simulationType: { 
                            type: 'string', 
                            enum: ['long_term_geological', 'volcanic_monitoring', 'earthquake_forecasting'] 
                        },
                        parameters: { type: 'object' },
                        timeSpan: { type: 'object' },
                        spatialDomain: { type: 'object' }
                    },
                    required: ['simulationType', 'parameters']
                }
            }
        }
    },
    apis: [
        './api/external/*.js',
        './auth/*.js'
    ]
};

const swaggerSpec = swaggerJsdoc(swaggerOptions);

// Security middleware
app.use(helmet({
    contentSecurityPolicy: {
        directives: {
            defaultSrc: ["'self'"],
            styleSrc: ["'self'", "'unsafe-inline'", "https://fonts.googleapis.com"],
            fontSrc: ["'self'", "https://fonts.gstatic.com"],
            scriptSrc: ["'self'"],
            imgSrc: ["'self'", "data:", "https:"],
            connectSrc: ["'self'", "wss:", "ws:"]
        }
    },
    crossOriginEmbedderPolicy: false
}));

// CORS configuration
const corsOptions = {
    origin: function (origin, callback) {
        const allowedOrigins = [
            'https://prometheus-ai.com',
            'https://app.prometheus-ai.com',
            'https://usgs.gov',
            'https://nasa.gov',
            'https://esa.int',
            process.env.FRONTEND_URL
        ];
        
        // Allow requests with no origin (mobile apps, etc.)
        if (!origin) return callback(null, true);
        
        if (allowedOrigins.includes(origin)) {
            callback(null, true);
        } else {
            callback(new Error('Not allowed by CORS'));
        }
    },
    credentials: true,
    methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
    allowedHeaders: [
        'Content-Type', 
        'Authorization', 
        'X-API-Key', 
        'X-Request-ID',
        'Accept-Encoding',
        'Content-Encoding'
    ]
};

app.use(cors(corsOptions));

// Compression middleware
app.use(compression({
    filter: (req, res) => {
        // Don't compress if already encrypted
        if (res.getHeader('Content-Encoding') === 'encrypted') {
            return false;
        }
        return compression.filter(req, res);
    }
}));

// Request parsing middleware
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// Request logging
app.use(morgan('combined', {
    stream: {
        write: (message) => logger.info(message.trim())
    }
}));

// Request ID middleware
app.use((req, res, next) => {
    req.requestId = req.headers['x-request-id'] || require('crypto').randomUUID();
    res.setHeader('X-Request-ID', req.requestId);
    next();
});

// Encryption middleware
const encryption = encryptionMiddleware(encryptionService);
app.use(encryption.decryptRequest);
app.use(encryption.encryptResponse);

// Service dependencies (would be injected in production)
const mockServices = {
    geologicalDataService: {
        queryData: async (params) => ({
            records: [],
            totalCount: 0,
            qualityMetrics: {},
            processingInfo: {}
        })
    },
    dataIngestionService: {
        ingestData: async (data) => ({
            success: true,
            ingestionId: require('crypto').randomUUID(),
            processedCount: data.records.length,
            rejectedCount: 0,
            message: 'Data ingested successfully'
        })
    },
    simulationService: {
        createSimulation: async (params) => ({
            id: require('crypto').randomUUID(),
            status: 'queued',
            estimatedCompletion: new Date(Date.now() + 3600000),
            createdAt: new Date()
        }),
        getSimulation: async (id, userId) => ({
            id,
            status: 'completed',
            progress: 100,
            results: { message: 'Simulation completed successfully' },
            createdAt: new Date(),
            completedAt: new Date()
        }),
        getUserSimulations: async (params) => ({
            simulations: [],
            totalCount: 0
        })
    },
    monitoringService: {
        getCurrentAlerts: async (params) => [],
        getAlerts: async (params) => [],
        acknowledgeAlert: async (alertId, userId) => true
    },
    resourceService: {
        getMineralResources: async (params) => [],
        getGeothermalResources: async (params) => []
    },
    seismicService: {
        getEvents: async (params) => []
    }
};

// Attach services to app
app.set('services', mockServices);
app.set('geologicalDataService', mockServices.geologicalDataService);
app.set('dataIngestionService', mockServices.dataIngestionService);
app.set('simulationService', mockServices.simulationService);
app.set('monitoringService', mockServices.monitoringService);

// API Documentation
app.use('/api-docs', swaggerUi.serve, swaggerUi.setup(swaggerSpec, {
    explorer: true,
    customCss: '.swagger-ui .topbar { display: none }',
    customSiteTitle: 'Prometheus AI API Documentation'
}));

// Serve OpenAPI spec
app.get('/api-docs.json', (req, res) => {
    res.setHeader('Content-Type', 'application/json');
    res.send(swaggerSpec);
});

// Health check endpoints
app.get('/health', (req, res) => {
    res.json({
        status: 'healthy',
        timestamp: new Date().toISOString(),
        version: process.env.API_VERSION || '1.0.0',
        uptime: process.uptime(),
        environment: process.env.NODE_ENV || 'development'
    });
});

app.get('/ready', (req, res) => {
    // Check if all services are ready
    const checks = {
        database: true, // Would check actual database connection
        redis: true,    // Would check Redis connection
        services: true  // Would check service dependencies
    };
    
    const isReady = Object.values(checks).every(check => check);
    
    res.status(isReady ? 200 : 503).json({
        status: isReady ? 'ready' : 'not ready',
        checks,
        timestamp: new Date().toISOString()
    });
});

// Rate limiting health check
app.get('/health/rate-limiting', healthCheck);

// Usage analytics endpoint
app.get('/analytics/usage', 
    rateLimiters.general,
    trackUsage('analytics', 'basic'),
    getUsageAnalytics
);

// OAuth2 endpoints
app.use('/oauth2', oauth2Router);
app.use('/.well-known', oauth2Router);

// External REST API endpoints
app.use('/api/external/v1', 
    dynamicRateLimit('external_api'),
    trackUsage('external_api'),
    restAPI
);

// GraphQL endpoint
const createGraphQLEndpoint = async () => {
    const server = createGraphQLServer();
    await server.start();
    server.applyMiddleware({ 
        app, 
        path: '/api/external/graphql',
        cors: corsOptions
    });
    
    logger.info('GraphQL endpoint configured at /api/external/graphql');
};

// API versioning middleware
app.use('/api/v1', (req, res, next) => {
    res.setHeader('API-Version', '1.0.0');
    next();
});

// Root endpoint
app.get('/', (req, res) => {
    res.json({
        name: 'Prometheus AI Geodynamic Platform API Gateway',
        version: process.env.API_VERSION || '1.0.0',
        description: 'External API for geological data integration and simulation services',
        documentation: '/api-docs',
        endpoints: {
            rest: '/api/external/v1',
            graphql: '/api/external/graphql',
            oauth2: '/oauth2',
            health: '/health',
            analytics: '/analytics/usage'
        },
        timestamp: new Date().toISOString()
    });
});

// 404 handler
app.use('*', (req, res) => {
    logger.warn(`404 - Route not found: ${req.method} ${req.originalUrl}`, {
        ip: req.ip,
        userAgent: req.get('User-Agent')
    });
    
    res.status(404).json({
        error: 'Not Found',
        message: 'The requested endpoint does not exist',
        path: req.originalUrl,
        method: req.method,
        timestamp: new Date().toISOString()
    });
});

// Global error handler
app.use((error, req, res, next) => {
    logger.error('Unhandled error:', {
        error: error.message,
        stack: error.stack,
        requestId: req.requestId,
        path: req.path,
        method: req.method,
        ip: req.ip
    });

    // Don't expose internal errors in production
    const isDevelopment = process.env.NODE_ENV !== 'production';
    
    res.status(error.status || 500).json({
        error: 'Internal Server Error',
        message: isDevelopment ? error.message : 'An unexpected error occurred',
        requestId: req.requestId,
        timestamp: new Date().toISOString(),
        ...(isDevelopment && { stack: error.stack })
    });
});

// Graceful shutdown handling
const gracefulShutdown = (signal) => {
    logger.info(`Received ${signal}, starting graceful shutdown...`);
    
    server.close(() => {
        logger.info('HTTP server closed');
        
        // Close database connections, cleanup resources, etc.
        process.exit(0);
    });
    
    // Force shutdown after 30 seconds
    setTimeout(() => {
        logger.error('Forced shutdown after timeout');
        process.exit(1);
    }, 30000);
};

process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));
process.on('SIGINT', () => gracefulShutdown('SIGINT'));

// Start server
const startServer = async () => {
    try {
        // Initialize GraphQL endpoint
        await createGraphQLEndpoint();
        
        // Start HTTP server
        const server = app.listen(PORT, '0.0.0.0', () => {
            logger.info(`Prometheus AI API Gateway started`, {
                port: PORT,
                environment: process.env.NODE_ENV || 'development',
                pid: process.pid,
                endpoints: {
                    health: `http://localhost:${PORT}/health`,
                    docs: `http://localhost:${PORT}/api-docs`,
                    rest: `http://localhost:${PORT}/api/external/v1`,
                    graphql: `http://localhost:${PORT}/api/external/graphql`,
                    oauth2: `http://localhost:${PORT}/oauth2`
                }
            });
        });
        
        // Store server reference for graceful shutdown
        global.server = server;
        
        return server;
    } catch (error) {
        logger.error('Failed to start server:', error);
        process.exit(1);
    }
};

// Start the server if this file is run directly
if (require.main === module) {
    startServer();
}

module.exports = { app, startServer };
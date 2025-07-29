const { ApolloServer } = require('apollo-server-express');
const { gql } = require('apollo-server-express');
const { GraphQLScalarType } = require('graphql');
const { Kind } = require('graphql/language');
const jwt = require('jsonwebtoken');
const crypto = require('crypto');
const winston = require('winston');
const rateLimit = require('express-rate-limit');

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
        new winston.transports.File({ filename: 'logs/graphql-api.log' })
    ]
});

// Custom scalar types
const DateTimeType = new GraphQLScalarType({
    name: 'DateTime',
    description: 'Date custom scalar type',
    serialize(value) {
        return value instanceof Date ? value.toISOString() : null;
    },
    parseValue(value) {
        return new Date(value);
    },
    parseLiteral(ast) {
        if (ast.kind === Kind.STRING) {
            return new Date(ast.value);
        }
        return null;
    },
});

const JSONType = new GraphQLScalarType({
    name: 'JSON',
    description: 'JSON custom scalar type',
    serialize(value) {
        return value;
    },
    parseValue(value) {
        return value;
    },
    parseLiteral(ast) {
        if (ast.kind === Kind.STRING) {
            return JSON.parse(ast.value);
        }
        return null;
    },
});

// GraphQL Schema Definition
const typeDefs = gql`
    scalar DateTime
    scalar JSON

    type Query {
        # Geological Data Queries
        geologicalData(
            type: GeologicalDataType!
            startTime: DateTime
            endTime: DateTime
            boundingBox: BoundingBoxInput
            limit: Int = 100
            offset: Int = 0
        ): GeologicalDataResponse!

        seismicEvents(
            startTime: DateTime
            endTime: DateTime
            minMagnitude: Float
            maxMagnitude: Float
            boundingBox: BoundingBoxInput
            limit: Int = 100
        ): [SeismicEvent!]!

        deformationData(
            startTime: DateTime
            endTime: DateTime
            measurementType: String
            boundingBox: BoundingBoxInput
            limit: Int = 100
        ): [DeformationMeasurement!]!

        geochemicalSamples(
            startTime: DateTime
            endTime: DateTime
            sampleType: String
            boundingBox: BoundingBoxInput
            limit: Int = 100
        ): [GeochemicalSample!]!

        # Simulation Queries
        simulation(id: ID!): Simulation
        simulations(
            status: SimulationStatus
            type: SimulationType
            limit: Int = 20
            offset: Int = 0
        ): SimulationConnection!

        # Monitoring Queries
        volcanicAlerts(
            volcanoId: String
            minAlertLevel: Int
            active: Boolean = true
        ): [VolcanicAlert!]!

        realTimeStreams(
            dataType: String
            volcanoId: String
        ): [RealTimeStream!]!

        # Resource Prediction Queries
        mineralResources(
            resourceType: String
            confidenceThreshold: Float = 0.7
            boundingBox: BoundingBoxInput
        ): [MineralResource!]!

        geothermalResources(
            minTemperature: Float
            boundingBox: BoundingBoxInput
        ): [GeothermalResource!]!
    }

    type Mutation {
        # Data Submission
        submitGeologicalData(input: GeologicalDataInput!): DataSubmissionResult!
        submitSeismicEvent(input: SeismicEventInput!): DataSubmissionResult!
        submitDeformationData(input: DeformationDataInput!): DataSubmissionResult!

        # Simulation Management
        createSimulation(input: SimulationInput!): Simulation!
        cancelSimulation(id: ID!): Boolean!

        # Alert Management
        acknowledgeAlert(alertId: ID!): Boolean!
        updateAlertStatus(alertId: ID!, status: AlertStatus!): VolcanicAlert!
    }

    type Subscription {
        # Real-time data streams
        seismicEventStream(volcanoId: String): SeismicEvent!
        deformationStream(volcanoId: String): DeformationMeasurement!
        alertStream(volcanoId: String, minAlertLevel: Int): VolcanicAlert!
        simulationProgress(simulationId: ID!): SimulationProgress!
    }

    # Enums
    enum GeologicalDataType {
        SEISMIC
        DEFORMATION
        GEOCHEMICAL
        THERMAL
        MAGNETIC
        GRAVITY
    }

    enum SimulationType {
        LONG_TERM_GEOLOGICAL
        VOLCANIC_MONITORING
        EARTHQUAKE_FORECASTING
        RESOURCE_PREDICTION
        HAZARD_ASSESSMENT
    }

    enum SimulationStatus {
        QUEUED
        RUNNING
        COMPLETED
        FAILED
        CANCELLED
    }

    enum AlertStatus {
        ACTIVE
        ACKNOWLEDGED
        RESOLVED
        ESCALATED
    }

    # Input Types
    input BoundingBoxInput {
        minLatitude: Float!
        maxLatitude: Float!
        minLongitude: Float!
        maxLongitude: Float!
    }

    input LocationInput {
        latitude: Float!
        longitude: Float!
        elevation: Float
        depth: Float
    }

    input GeologicalDataInput {
        dataType: GeologicalDataType!
        source: String!
        records: [GeologicalDataRecordInput!]!
    }

    input GeologicalDataRecordInput {
        timestamp: DateTime!
        location: LocationInput!
        measurements: JSON!
        quality: String
        metadata: JSON
    }

    input SeismicEventInput {
        timestamp: DateTime!
        location: LocationInput!
        magnitude: Float!
        depth: Float!
        eventType: String
        waveformData: String
    }

    input DeformationDataInput {
        timestamp: DateTime!
        location: LocationInput!
        displacementVector: [Float!]!
        measurementType: String!
        uncertainty: [Float!]
    }

    input SimulationInput {
        type: SimulationType!
        parameters: JSON!
        timeSpan: TimeSpanInput
        spatialDomain: SpatialDomainInput
        outputFormat: String = "JSON"
        priority: Int = 5
    }

    input TimeSpanInput {
        startTime: DateTime!
        endTime: DateTime!
        resolution: String
    }

    input SpatialDomainInput {
        boundingBox: BoundingBoxInput!
        resolution: Float
        coordinateSystem: String = "WGS84"
    }

    # Object Types
    type GeologicalDataResponse {
        data: [GeologicalDataRecord!]!
        metadata: DataMetadata!
        pagination: PaginationInfo!
    }

    type GeologicalDataRecord {
        id: ID!
        timestamp: DateTime!
        location: Location!
        dataType: GeologicalDataType!
        measurements: JSON!
        quality: String
        source: String!
        metadata: JSON
        createdAt: DateTime!
    }

    type Location {
        latitude: Float!
        longitude: Float!
        elevation: Float
        depth: Float
        coordinateSystem: String!
    }

    type DataMetadata {
        totalCount: Int!
        queryTime: DateTime!
        dataSource: String!
        qualityMetrics: JSON
        processingInfo: JSON
    }

    type PaginationInfo {
        hasNextPage: Boolean!
        hasPreviousPage: Boolean!
        totalPages: Int!
        currentPage: Int!
        pageSize: Int!
    }

    type SeismicEvent {
        id: ID!
        timestamp: DateTime!
        location: Location!
        magnitude: Float!
        depth: Float!
        eventType: String
        waveformData: String
        confidence: Float
        source: String!
        createdAt: DateTime!
    }

    type DeformationMeasurement {
        id: ID!
        timestamp: DateTime!
        location: Location!
        displacementVector: [Float!]!
        measurementType: String!
        uncertainty: [Float!]
        referenceFrame: String
        source: String!
        createdAt: DateTime!
    }

    type GeochemicalSample {
        id: ID!
        timestamp: DateTime!
        location: Location!
        sampleType: String!
        composition: JSON!
        isotopeRatios: JSON
        temperature: Float
        pressure: Float
        source: String!
        createdAt: DateTime!
    }

    type Simulation {
        id: ID!
        type: SimulationType!
        status: SimulationStatus!
        parameters: JSON!
        progress: Float
        results: JSON
        errorMessage: String
        estimatedCompletion: DateTime
        createdAt: DateTime!
        completedAt: DateTime
        userId: String!
    }

    type SimulationConnection {
        edges: [SimulationEdge!]!
        pageInfo: PageInfo!
        totalCount: Int!
    }

    type SimulationEdge {
        node: Simulation!
        cursor: String!
    }

    type PageInfo {
        hasNextPage: Boolean!
        hasPreviousPage: Boolean!
        startCursor: String
        endCursor: String
    }

    type SimulationProgress {
        simulationId: ID!
        progress: Float!
        stage: String!
        message: String
        timestamp: DateTime!
    }

    type VolcanicAlert {
        id: ID!
        volcanoId: String!
        alertLevel: Int!
        description: String!
        recommendedActions: [String!]!
        validityPeriod: TimeRange!
        confidence: Float!
        status: AlertStatus!
        createdAt: DateTime!
        updatedAt: DateTime!
    }

    type TimeRange {
        start: DateTime!
        end: DateTime
    }

    type RealTimeStream {
        id: ID!
        dataType: String!
        volcanoId: String
        isActive: Boolean!
        lastUpdate: DateTime!
        sampleRate: Float
        dataQuality: String
    }

    type MineralResource {
        id: ID!
        resourceType: String!
        location: Location!
        estimatedVolume: Float
        confidence: Float!
        geologicalContext: JSON!
        economicViability: Float
        discoveryDate: DateTime!
    }

    type GeothermalResource {
        id: ID!
        location: Location!
        estimatedTemperature: Float!
        depth: Float!
        permeability: Float
        sustainability: Float
        confidence: Float!
        economicViability: Float
    }

    type DataSubmissionResult {
        success: Boolean!
        ingestionId: String!
        processedCount: Int!
        rejectedCount: Int!
        errors: [String!]
        message: String!
    }
`;

// GraphQL Resolvers
const resolvers = {
    DateTime: DateTimeType,
    JSON: JSONType,

    Query: {
        geologicalData: async (parent, args, context) => {
            try {
                const { type, startTime, endTime, boundingBox, limit, offset } = args;
                
                // Check authentication
                if (!context.user && !context.apiKey) {
                    throw new Error('Authentication required');
                }

                const geologicalDataService = context.services.geologicalDataService;
                const result = await geologicalDataService.queryData({
                    type: type.toLowerCase(),
                    startTime,
                    endTime,
                    boundingBox,
                    limit,
                    offset
                });

                logger.info(`GraphQL geological data query`, {
                    requestId: context.requestId,
                    type,
                    recordCount: result.records.length,
                    userId: context.user?.id
                });

                return {
                    data: result.records,
                    metadata: {
                        totalCount: result.totalCount,
                        queryTime: new Date(),
                        dataSource: 'prometheus-ai',
                        qualityMetrics: result.qualityMetrics,
                        processingInfo: result.processingInfo
                    },
                    pagination: {
                        hasNextPage: offset + limit < result.totalCount,
                        hasPreviousPage: offset > 0,
                        totalPages: Math.ceil(result.totalCount / limit),
                        currentPage: Math.floor(offset / limit) + 1,
                        pageSize: limit
                    }
                };
            } catch (error) {
                logger.error(`GraphQL geological data query error:`, {
                    requestId: context.requestId,
                    error: error.message,
                    args
                });
                throw error;
            }
        },

        seismicEvents: async (parent, args, context) => {
            const { startTime, endTime, minMagnitude, maxMagnitude, boundingBox, limit } = args;
            
            const seismicService = context.services.seismicService;
            const events = await seismicService.getEvents({
                startTime,
                endTime,
                minMagnitude,
                maxMagnitude,
                boundingBox,
                limit
            });

            return events;
        },

        simulation: async (parent, { id }, context) => {
            if (!context.user) {
                throw new Error('Authentication required');
            }

            const simulationService = context.services.simulationService;
            const simulation = await simulationService.getSimulation(id, context.user.id);
            
            if (!simulation) {
                throw new Error('Simulation not found');
            }

            return simulation;
        },

        simulations: async (parent, args, context) => {
            if (!context.user) {
                throw new Error('Authentication required');
            }

            const { status, type, limit, offset } = args;
            const simulationService = context.services.simulationService;
            
            const result = await simulationService.getUserSimulations({
                userId: context.user.id,
                status,
                type,
                limit,
                offset
            });

            return {
                edges: result.simulations.map((sim, index) => ({
                    node: sim,
                    cursor: Buffer.from(`${offset + index}`).toString('base64')
                })),
                pageInfo: {
                    hasNextPage: offset + limit < result.totalCount,
                    hasPreviousPage: offset > 0,
                    startCursor: result.simulations.length > 0 ? 
                        Buffer.from(`${offset}`).toString('base64') : null,
                    endCursor: result.simulations.length > 0 ? 
                        Buffer.from(`${offset + result.simulations.length - 1}`).toString('base64') : null
                },
                totalCount: result.totalCount
            };
        },

        volcanicAlerts: async (parent, args, context) => {
            const { volcanoId, minAlertLevel, active } = args;
            
            const monitoringService = context.services.monitoringService;
            const alerts = await monitoringService.getAlerts({
                volcanoId,
                minAlertLevel,
                active
            });

            return alerts;
        },

        mineralResources: async (parent, args, context) => {
            const { resourceType, confidenceThreshold, boundingBox } = args;
            
            const resourceService = context.services.resourceService;
            const resources = await resourceService.getMineralResources({
                resourceType,
                confidenceThreshold,
                boundingBox
            });

            return resources;
        }
    },

    Mutation: {
        submitGeologicalData: async (parent, { input }, context) => {
            if (!context.apiKey && !context.user) {
                throw new Error('Authentication required');
            }

            const dataIngestionService = context.services.dataIngestionService;
            const result = await dataIngestionService.ingestData({
                dataType: input.dataType.toLowerCase(),
                source: input.source,
                records: input.records,
                submittedBy: context.user?.id || context.apiKey
            });

            logger.info(`GraphQL data submission`, {
                requestId: context.requestId,
                dataType: input.dataType,
                recordCount: input.records.length,
                ingestionId: result.ingestionId
            });

            return {
                success: result.success,
                ingestionId: result.ingestionId,
                processedCount: result.processedCount,
                rejectedCount: result.rejectedCount,
                errors: result.errors || [],
                message: result.message
            };
        },

        createSimulation: async (parent, { input }, context) => {
            if (!context.user) {
                throw new Error('Authentication required');
            }

            const simulationService = context.services.simulationService;
            const simulation = await simulationService.createSimulation({
                ...input,
                userId: context.user.id
            });

            logger.info(`GraphQL simulation created`, {
                requestId: context.requestId,
                simulationId: simulation.id,
                userId: context.user.id,
                type: input.type
            });

            return simulation;
        },

        acknowledgeAlert: async (parent, { alertId }, context) => {
            if (!context.user) {
                throw new Error('Authentication required');
            }

            const monitoringService = context.services.monitoringService;
            const success = await monitoringService.acknowledgeAlert(alertId, context.user.id);

            return success;
        }
    },

    Subscription: {
        seismicEventStream: {
            subscribe: async (parent, { volcanoId }, context) => {
                if (!context.user && !context.apiKey) {
                    throw new Error('Authentication required');
                }

                const pubsub = context.pubsub;
                return pubsub.asyncIterator([`SEISMIC_EVENT_${volcanoId || 'ALL'}`]);
            }
        },

        alertStream: {
            subscribe: async (parent, { volcanoId, minAlertLevel }, context) => {
                if (!context.user && !context.apiKey) {
                    throw new Error('Authentication required');
                }

                const pubsub = context.pubsub;
                return pubsub.asyncIterator([`VOLCANIC_ALERT_${volcanoId || 'ALL'}`]);
            }
        },

        simulationProgress: {
            subscribe: async (parent, { simulationId }, context) => {
                if (!context.user) {
                    throw new Error('Authentication required');
                }

                // Verify user owns the simulation
                const simulationService = context.services.simulationService;
                const simulation = await simulationService.getSimulation(simulationId, context.user.id);
                
                if (!simulation) {
                    throw new Error('Simulation not found or access denied');
                }

                const pubsub = context.pubsub;
                return pubsub.asyncIterator([`SIMULATION_PROGRESS_${simulationId}`]);
            }
        }
    }
};

// Authentication context function
const createContext = async ({ req, connection }) => {
    let user = null;
    let apiKey = null;
    const requestId = crypto.randomUUID();

    if (connection) {
        // WebSocket connection for subscriptions
        const token = connection.context.authorization;
        if (token) {
            try {
                user = jwt.verify(token.replace('Bearer ', ''), process.env.JWT_SECRET);
            } catch (error) {
                logger.warn(`Invalid JWT in WebSocket connection: ${error.message}`);
            }
        }
    } else {
        // HTTP request
        const authHeader = req.headers.authorization;
        const apiKeyHeader = req.headers['x-api-key'];

        if (authHeader) {
            const token = authHeader.replace('Bearer ', '');
            try {
                user = jwt.verify(token, process.env.JWT_SECRET);
            } catch (error) {
                logger.warn(`Invalid JWT token: ${error.message}`);
            }
        } else if (apiKeyHeader) {
            // Validate API key
            const keyHash = crypto.createHash('sha256').update(apiKeyHeader).digest('hex');
            const validKeys = new Set([
                process.env.USGS_API_KEY_HASH,
                process.env.NASA_API_KEY_HASH,
                process.env.ESA_API_KEY_HASH,
                process.env.PARTNER_API_KEY_HASH
            ]);

            if (validKeys.has(keyHash)) {
                apiKey = keyHash;
            }
        }
    }

    return {
        user,
        apiKey,
        requestId,
        services: req?.app?.get('services') || {},
        pubsub: req?.app?.get('pubsub')
    };
};

// Create Apollo Server
const createGraphQLServer = () => {
    return new ApolloServer({
        typeDefs,
        resolvers,
        context: createContext,
        introspection: process.env.NODE_ENV !== 'production',
        playground: process.env.NODE_ENV !== 'production',
        formatError: (error) => {
            logger.error(`GraphQL Error: ${error.message}`, {
                error: error.message,
                stack: error.stack,
                path: error.path,
                source: error.source
            });

            // Don't expose internal errors in production
            if (process.env.NODE_ENV === 'production') {
                return new Error('Internal server error');
            }

            return error;
        },
        plugins: [
            {
                requestDidStart() {
                    return {
                        didResolveOperation(requestContext) {
                            logger.info(`GraphQL Operation: ${requestContext.request.operationName}`, {
                                operationName: requestContext.request.operationName,
                                query: requestContext.request.query,
                                variables: requestContext.request.variables
                            });
                        },
                        didEncounterErrors(requestContext) {
                            logger.error(`GraphQL Errors:`, {
                                errors: requestContext.errors.map(error => ({
                                    message: error.message,
                                    path: error.path
                                }))
                            });
                        }
                    };
                }
            }
        ]
    });
};

module.exports = {
    createGraphQLServer,
    typeDefs,
    resolvers
};
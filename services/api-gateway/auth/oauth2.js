const express = require('express');
const jwt = require('jsonwebtoken');
const crypto = require('crypto');
const bcrypt = require('bcrypt');
const rateLimit = require('express-rate-limit');
const { body, validationResult } = require('express-validator');
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
        new winston.transports.File({ filename: 'logs/oauth2.log' })
    ]
});

// Rate limiting for authentication endpoints
const authRateLimit = rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 5, // limit each IP to 5 requests per windowMs
    message: { error: 'Too many authentication attempts, please try again later' },
    standardHeaders: true,
    legacyHeaders: false,
    handler: (req, res) => {
        logger.warn(`Authentication rate limit exceeded for IP: ${req.ip}`);
        res.status(429).json({ error: 'Too many authentication attempts, please try again later' });
    }
});

const tokenRateLimit = rateLimit({
    windowMs: 60 * 1000, // 1 minute
    max: 10, // limit each IP to 10 token requests per minute
    message: { error: 'Too many token requests, please try again later' }
});

// OAuth2 Configuration
const OAUTH2_CONFIG = {
    accessTokenExpiry: '1h',
    refreshTokenExpiry: '7d',
    authorizationCodeExpiry: 10 * 60 * 1000, // 10 minutes
    supportedGrantTypes: ['authorization_code', 'refresh_token', 'client_credentials'],
    supportedScopes: [
        'read:geological-data',
        'write:geological-data',
        'read:simulations',
        'write:simulations',
        'read:monitoring',
        'write:monitoring',
        'admin:system'
    ]
};

// In-memory stores (replace with database in production)
const authorizationCodes = new Map();
const refreshTokens = new Map();
const clients = new Map();
const users = new Map();

// Initialize default clients and users
const initializeDefaults = () => {
    // Default OAuth2 clients
    clients.set('usgs-client', {
        id: 'usgs-client',
        secret: crypto.createHash('sha256').update('usgs-secret-key').digest('hex'),
        name: 'USGS Data Integration',
        redirectUris: ['https://usgs.gov/oauth/callback'],
        scopes: ['read:geological-data', 'write:geological-data'],
        grantTypes: ['authorization_code', 'refresh_token']
    });

    clients.set('nasa-client', {
        id: 'nasa-client',
        secret: crypto.createHash('sha256').update('nasa-secret-key').digest('hex'),
        name: 'NASA Earth Science Division',
        redirectUris: ['https://nasa.gov/oauth/callback'],
        scopes: ['read:geological-data', 'read:simulations'],
        grantTypes: ['authorization_code', 'refresh_token']
    });

    clients.set('research-client', {
        id: 'research-client',
        secret: crypto.createHash('sha256').update('research-secret-key').digest('hex'),
        name: 'Research Institution Access',
        redirectUris: ['https://research.edu/oauth/callback'],
        scopes: ['read:geological-data', 'read:simulations', 'write:simulations'],
        grantTypes: ['authorization_code', 'refresh_token', 'client_credentials']
    });

    // Default system users
    users.set('admin@prometheus-ai.com', {
        id: 'admin-user-1',
        email: 'admin@prometheus-ai.com',
        passwordHash: bcrypt.hashSync('admin-password', 10),
        scopes: ['admin:system'],
        active: true,
        createdAt: new Date()
    });

    users.set('researcher@university.edu', {
        id: 'researcher-user-1',
        email: 'researcher@university.edu',
        passwordHash: bcrypt.hashSync('researcher-password', 10),
        scopes: ['read:geological-data', 'read:simulations', 'write:simulations'],
        active: true,
        createdAt: new Date()
    });
};

initializeDefaults();

// Utility functions
const generateToken = (length = 32) => {
    return crypto.randomBytes(length).toString('hex');
};

const validateClient = async (clientId, clientSecret) => {
    const client = clients.get(clientId);
    if (!client) return null;

    const secretHash = crypto.createHash('sha256').update(clientSecret).digest('hex');
    if (client.secret !== secretHash) return null;

    return client;
};

const validateUser = async (email, password) => {
    const user = users.get(email);
    if (!user || !user.active) return null;

    const isValidPassword = await bcrypt.compare(password, user.passwordHash);
    if (!isValidPassword) return null;

    return user;
};

const generateJWT = (payload, expiresIn) => {
    return jwt.sign(payload, process.env.JWT_SECRET, { expiresIn });
};

const validateScopes = (requestedScopes, allowedScopes) => {
    if (!requestedScopes) return [];
    
    const requested = requestedScopes.split(' ');
    return requested.filter(scope => allowedScopes.includes(scope));
};

// OAuth2 Authorization Endpoint
/**
 * @swagger
 * /oauth2/authorize:
 *   get:
 *     summary: OAuth2 authorization endpoint
 *     tags: [OAuth2]
 *     parameters:
 *       - in: query
 *         name: response_type
 *         required: true
 *         schema:
 *           type: string
 *           enum: [code]
 *       - in: query
 *         name: client_id
 *         required: true
 *         schema:
 *           type: string
 *       - in: query
 *         name: redirect_uri
 *         required: true
 *         schema:
 *           type: string
 *       - in: query
 *         name: scope
 *         schema:
 *           type: string
 *       - in: query
 *         name: state
 *         schema:
 *           type: string
 *     responses:
 *       302:
 *         description: Redirect to authorization page or callback
 */
router.get('/authorize', [
    body('response_type').equals('code'),
    body('client_id').isString().notEmpty(),
    body('redirect_uri').isURL(),
    body('scope').optional().isString(),
    body('state').optional().isString()
], (req, res) => {
    try {
        const { response_type, client_id, redirect_uri, scope, state } = req.query;

        // Validate client
        const client = clients.get(client_id);
        if (!client) {
            logger.warn(`Invalid client_id in authorization request: ${client_id}`);
            return res.status(400).json({ error: 'invalid_client' });
        }

        // Validate redirect URI
        if (!client.redirectUris.includes(redirect_uri)) {
            logger.warn(`Invalid redirect_uri for client ${client_id}: ${redirect_uri}`);
            return res.status(400).json({ error: 'invalid_redirect_uri' });
        }

        // Validate scopes
        const requestedScopes = validateScopes(scope, client.scopes);

        // Store authorization request
        const authRequest = {
            clientId: client_id,
            redirectUri: redirect_uri,
            scopes: requestedScopes,
            state,
            createdAt: new Date()
        };

        // In a real implementation, redirect to login page
        // For this example, we'll simulate user consent
        const authCode = generateToken();
        authorizationCodes.set(authCode, {
            ...authRequest,
            userId: 'researcher-user-1', // Simulated authenticated user
            expiresAt: new Date(Date.now() + OAUTH2_CONFIG.authorizationCodeExpiry)
        });

        logger.info(`Authorization code generated`, {
            clientId: client_id,
            userId: 'researcher-user-1',
            scopes: requestedScopes,
            authCode: authCode.substring(0, 8) + '...'
        });

        // Redirect back to client with authorization code
        const redirectUrl = new URL(redirect_uri);
        redirectUrl.searchParams.set('code', authCode);
        if (state) redirectUrl.searchParams.set('state', state);

        res.redirect(redirectUrl.toString());
    } catch (error) {
        logger.error(`Authorization endpoint error:`, error);
        res.status(500).json({ error: 'server_error' });
    }
});

// OAuth2 Token Endpoint
/**
 * @swagger
 * /oauth2/token:
 *   post:
 *     summary: OAuth2 token endpoint
 *     tags: [OAuth2]
 *     requestBody:
 *       required: true
 *       content:
 *         application/x-www-form-urlencoded:
 *           schema:
 *             type: object
 *             properties:
 *               grant_type:
 *                 type: string
 *                 enum: [authorization_code, refresh_token, client_credentials]
 *               code:
 *                 type: string
 *               redirect_uri:
 *                 type: string
 *               client_id:
 *                 type: string
 *               client_secret:
 *                 type: string
 *               refresh_token:
 *                 type: string
 *               scope:
 *                 type: string
 *     responses:
 *       200:
 *         description: Access token issued
 */
router.post('/token',
    tokenRateLimit,
    [
        body('grant_type').isIn(OAUTH2_CONFIG.supportedGrantTypes),
        body('client_id').isString().notEmpty(),
        body('client_secret').isString().notEmpty()
    ],
    async (req, res) => {
        try {
            const errors = validationResult(req);
            if (!errors.isEmpty()) {
                return res.status(400).json({ error: 'invalid_request', error_description: 'Validation failed' });
            }

            const { grant_type, code, redirect_uri, client_id, client_secret, refresh_token, scope } = req.body;

            // Validate client credentials
            const client = await validateClient(client_id, client_secret);
            if (!client) {
                logger.warn(`Invalid client credentials: ${client_id}`);
                return res.status(401).json({ error: 'invalid_client' });
            }

            // Check if grant type is supported by client
            if (!client.grantTypes.includes(grant_type)) {
                return res.status(400).json({ error: 'unsupported_grant_type' });
            }

            let userId, userScopes;

            switch (grant_type) {
                case 'authorization_code':
                    // Validate authorization code
                    const authData = authorizationCodes.get(code);
                    if (!authData || authData.expiresAt < new Date() || authData.clientId !== client_id) {
                        return res.status(400).json({ error: 'invalid_grant' });
                    }

                    if (authData.redirectUri !== redirect_uri) {
                        return res.status(400).json({ error: 'invalid_grant' });
                    }

                    userId = authData.userId;
                    userScopes = authData.scopes;

                    // Remove used authorization code
                    authorizationCodes.delete(code);
                    break;

                case 'refresh_token':
                    // Validate refresh token
                    const refreshData = refreshTokens.get(refresh_token);
                    if (!refreshData || refreshData.clientId !== client_id) {
                        return res.status(400).json({ error: 'invalid_grant' });
                    }

                    userId = refreshData.userId;
                    userScopes = refreshData.scopes;

                    // Remove used refresh token
                    refreshTokens.delete(refresh_token);
                    break;

                case 'client_credentials':
                    // Client credentials grant - no user context
                    userId = null;
                    userScopes = validateScopes(scope, client.scopes);
                    break;

                default:
                    return res.status(400).json({ error: 'unsupported_grant_type' });
            }

            // Generate tokens
            const accessToken = generateJWT({
                sub: userId,
                client_id: client_id,
                scope: userScopes.join(' '),
                iat: Math.floor(Date.now() / 1000),
                type: 'access_token'
            }, OAUTH2_CONFIG.accessTokenExpiry);

            const newRefreshToken = generateToken();
            refreshTokens.set(newRefreshToken, {
                clientId: client_id,
                userId: userId,
                scopes: userScopes,
                createdAt: new Date()
            });

            logger.info(`Access token issued`, {
                clientId: client_id,
                userId: userId,
                grantType: grant_type,
                scopes: userScopes
            });

            res.json({
                access_token: accessToken,
                token_type: 'Bearer',
                expires_in: 3600, // 1 hour
                refresh_token: newRefreshToken,
                scope: userScopes.join(' ')
            });

        } catch (error) {
            logger.error(`Token endpoint error:`, error);
            res.status(500).json({ error: 'server_error' });
        }
    }
);

// Token Introspection Endpoint (RFC 7662)
/**
 * @swagger
 * /oauth2/introspect:
 *   post:
 *     summary: Token introspection endpoint
 *     tags: [OAuth2]
 *     requestBody:
 *       required: true
 *       content:
 *         application/x-www-form-urlencoded:
 *           schema:
 *             type: object
 *             properties:
 *               token:
 *                 type: string
 *               token_type_hint:
 *                 type: string
 *                 enum: [access_token, refresh_token]
 *     responses:
 *       200:
 *         description: Token introspection result
 */
router.post('/introspect',
    authRateLimit,
    [
        body('token').isString().notEmpty(),
        body('token_type_hint').optional().isIn(['access_token', 'refresh_token'])
    ],
    async (req, res) => {
        try {
            const { token, token_type_hint } = req.body;

            // Basic client authentication (should be enhanced)
            const authHeader = req.headers.authorization;
            if (!authHeader || !authHeader.startsWith('Basic ')) {
                return res.status(401).json({ error: 'invalid_client' });
            }

            try {
                // Verify JWT token
                const decoded = jwt.verify(token, process.env.JWT_SECRET);
                
                if (decoded.type !== 'access_token') {
                    return res.json({ active: false });
                }

                res.json({
                    active: true,
                    sub: decoded.sub,
                    client_id: decoded.client_id,
                    scope: decoded.scope,
                    exp: decoded.exp,
                    iat: decoded.iat
                });

            } catch (jwtError) {
                // Token is invalid or expired
                res.json({ active: false });
            }

        } catch (error) {
            logger.error(`Token introspection error:`, error);
            res.status(500).json({ error: 'server_error' });
        }
    }
);

// Token Revocation Endpoint (RFC 7009)
/**
 * @swagger
 * /oauth2/revoke:
 *   post:
 *     summary: Token revocation endpoint
 *     tags: [OAuth2]
 *     requestBody:
 *       required: true
 *       content:
 *         application/x-www-form-urlencoded:
 *           schema:
 *             type: object
 *             properties:
 *               token:
 *                 type: string
 *               token_type_hint:
 *                 type: string
 *                 enum: [access_token, refresh_token]
 *     responses:
 *       200:
 *         description: Token revoked successfully
 */
router.post('/revoke',
    authRateLimit,
    [
        body('token').isString().notEmpty(),
        body('token_type_hint').optional().isIn(['access_token', 'refresh_token'])
    ],
    async (req, res) => {
        try {
            const { token, token_type_hint } = req.body;

            // Basic client authentication
            const authHeader = req.headers.authorization;
            if (!authHeader || !authHeader.startsWith('Basic ')) {
                return res.status(401).json({ error: 'invalid_client' });
            }

            // If it's a refresh token, remove it from storage
            if (token_type_hint === 'refresh_token' || refreshTokens.has(token)) {
                refreshTokens.delete(token);
                logger.info(`Refresh token revoked: ${token.substring(0, 8)}...`);
            }

            // For access tokens, we would typically add them to a blacklist
            // Since JWTs are stateless, we'll log the revocation
            logger.info(`Token revocation requested: ${token.substring(0, 8)}...`);

            res.status(200).json({ message: 'Token revoked successfully' });

        } catch (error) {
            logger.error(`Token revocation error:`, error);
            res.status(500).json({ error: 'server_error' });
        }
    }
);

// OAuth2 Discovery Endpoint (RFC 8414)
/**
 * @swagger
 * /.well-known/oauth-authorization-server:
 *   get:
 *     summary: OAuth2 authorization server metadata
 *     tags: [OAuth2]
 *     responses:
 *       200:
 *         description: Authorization server metadata
 */
router.get('/.well-known/oauth-authorization-server', (req, res) => {
    const baseUrl = `${req.protocol}://${req.get('host')}`;
    
    res.json({
        issuer: baseUrl,
        authorization_endpoint: `${baseUrl}/oauth2/authorize`,
        token_endpoint: `${baseUrl}/oauth2/token`,
        introspection_endpoint: `${baseUrl}/oauth2/introspect`,
        revocation_endpoint: `${baseUrl}/oauth2/revoke`,
        response_types_supported: ['code'],
        grant_types_supported: OAUTH2_CONFIG.supportedGrantTypes,
        token_endpoint_auth_methods_supported: ['client_secret_basic', 'client_secret_post'],
        scopes_supported: OAUTH2_CONFIG.supportedScopes,
        code_challenge_methods_supported: ['S256'],
        introspection_endpoint_auth_methods_supported: ['client_secret_basic'],
        revocation_endpoint_auth_methods_supported: ['client_secret_basic']
    });
});

// Client Management Endpoints (for administrative purposes)

/**
 * @swagger
 * /oauth2/clients:
 *   post:
 *     summary: Register a new OAuth2 client
 *     tags: [OAuth2 Admin]
 *     security:
 *       - BearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             properties:
 *               name:
 *                 type: string
 *               redirect_uris:
 *                 type: array
 *                 items:
 *                   type: string
 *               scopes:
 *                 type: array
 *                 items:
 *                   type: string
 *               grant_types:
 *                 type: array
 *                 items:
 *                   type: string
 *     responses:
 *       201:
 *         description: Client registered successfully
 */
router.post('/clients',
    authRateLimit,
    [
        body('name').isString().isLength({ min: 1, max: 100 }),
        body('redirect_uris').isArray({ min: 1 }),
        body('redirect_uris.*').isURL(),
        body('scopes').isArray(),
        body('scopes.*').isIn(OAUTH2_CONFIG.supportedScopes),
        body('grant_types').isArray(),
        body('grant_types.*').isIn(OAUTH2_CONFIG.supportedGrantTypes)
    ],
    async (req, res) => {
        try {
            // Verify admin access
            const authHeader = req.headers.authorization;
            if (!authHeader) {
                return res.status(401).json({ error: 'Authentication required' });
            }

            const token = authHeader.replace('Bearer ', '');
            const decoded = jwt.verify(token, process.env.JWT_SECRET);
            
            if (!decoded.scope || !decoded.scope.includes('admin:system')) {
                return res.status(403).json({ error: 'Insufficient permissions' });
            }

            const { name, redirect_uris, scopes, grant_types } = req.body;

            // Generate client credentials
            const clientId = `client-${generateToken(16)}`;
            const clientSecret = generateToken(32);
            const clientSecretHash = crypto.createHash('sha256').update(clientSecret).digest('hex');

            const client = {
                id: clientId,
                secret: clientSecretHash,
                name,
                redirectUris: redirect_uris,
                scopes,
                grantTypes: grant_types,
                createdAt: new Date(),
                createdBy: decoded.sub
            };

            clients.set(clientId, client);

            logger.info(`OAuth2 client registered`, {
                clientId,
                name,
                createdBy: decoded.sub,
                scopes,
                grantTypes: grant_types
            });

            res.status(201).json({
                client_id: clientId,
                client_secret: clientSecret, // Only returned once
                name,
                redirect_uris,
                scopes,
                grant_types,
                created_at: client.createdAt
            });

        } catch (error) {
            logger.error(`Client registration error:`, error);
            res.status(500).json({ error: 'server_error' });
        }
    }
);

// Cleanup expired tokens (should be run periodically)
const cleanupExpiredTokens = () => {
    const now = new Date();
    
    // Clean up expired authorization codes
    for (const [code, data] of authorizationCodes.entries()) {
        if (data.expiresAt < now) {
            authorizationCodes.delete(code);
        }
    }

    // Clean up old refresh tokens (older than 7 days)
    const sevenDaysAgo = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
    for (const [token, data] of refreshTokens.entries()) {
        if (data.createdAt < sevenDaysAgo) {
            refreshTokens.delete(token);
        }
    }
};

// Run cleanup every hour
setInterval(cleanupExpiredTokens, 60 * 60 * 1000);

module.exports = router;
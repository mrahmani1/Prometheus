const crypto = require('crypto');
const fs = require('fs');
const path = require('path');
const winston = require('winston');
const zlib = require('zlib');

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
        new winston.transports.File({ filename: 'logs/encryption.log' })
    ]
});

// Encryption configuration
const ENCRYPTION_CONFIG = {
    algorithm: 'aes-256-gcm',
    keyLength: 32, // 256 bits
    ivLength: 16,  // 128 bits
    tagLength: 16, // 128 bits
    saltLength: 32, // 256 bits
    iterations: 100000, // PBKDF2 iterations
    hashAlgorithm: 'sha256'
};

// Key derivation configuration
const KEY_DERIVATION = {
    algorithm: 'pbkdf2',
    iterations: 100000,
    keyLength: 32,
    digest: 'sha256'
};

// Digital signature configuration
const SIGNATURE_CONFIG = {
    algorithm: 'RSA-SHA256',
    keySize: 2048,
    padding: crypto.constants.RSA_PKCS1_PSS_PADDING,
    saltLength: crypto.constants.RSA_PSS_SALTLEN_DIGEST
};

class EncryptionService {
    constructor() {
        this.masterKey = this.loadOrGenerateMasterKey();
        this.keyPairs = this.loadOrGenerateKeyPairs();
    }

    // Load or generate master encryption key
    loadOrGenerateMasterKey() {
        const keyPath = path.join(process.cwd(), 'keys', 'master.key');
        
        try {
            if (fs.existsSync(keyPath)) {
                const keyData = fs.readFileSync(keyPath);
                logger.info('Master key loaded from file');
                return keyData;
            }
        } catch (error) {
            logger.warn('Could not load master key from file:', error.message);
        }

        // Generate new master key
        const masterKey = crypto.randomBytes(ENCRYPTION_CONFIG.keyLength);
        
        try {
            // Ensure keys directory exists
            const keysDir = path.dirname(keyPath);
            if (!fs.existsSync(keysDir)) {
                fs.mkdirSync(keysDir, { recursive: true });
            }
            
            fs.writeFileSync(keyPath, masterKey, { mode: 0o600 });
            logger.info('New master key generated and saved');
        } catch (error) {
            logger.error('Could not save master key to file:', error.message);
        }

        return masterKey;
    }

    // Load or generate RSA key pairs for digital signatures
    loadOrGenerateKeyPairs() {
        const privateKeyPath = path.join(process.cwd(), 'keys', 'private.pem');
        const publicKeyPath = path.join(process.cwd(), 'keys', 'public.pem');

        try {
            if (fs.existsSync(privateKeyPath) && fs.existsSync(publicKeyPath)) {
                const privateKey = fs.readFileSync(privateKeyPath, 'utf8');
                const publicKey = fs.readFileSync(publicKeyPath, 'utf8');
                logger.info('RSA key pair loaded from files');
                return { privateKey, publicKey };
            }
        } catch (error) {
            logger.warn('Could not load RSA key pair from files:', error.message);
        }

        // Generate new key pair
        const { privateKey, publicKey } = crypto.generateKeyPairSync('rsa', {
            modulusLength: SIGNATURE_CONFIG.keySize,
            publicKeyEncoding: {
                type: 'spki',
                format: 'pem'
            },
            privateKeyEncoding: {
                type: 'pkcs8',
                format: 'pem'
            }
        });

        try {
            // Ensure keys directory exists
            const keysDir = path.dirname(privateKeyPath);
            if (!fs.existsSync(keysDir)) {
                fs.mkdirSync(keysDir, { recursive: true });
            }

            fs.writeFileSync(privateKeyPath, privateKey, { mode: 0o600 });
            fs.writeFileSync(publicKeyPath, publicKey, { mode: 0o644 });
            logger.info('New RSA key pair generated and saved');
        } catch (error) {
            logger.error('Could not save RSA key pair to files:', error.message);
        }

        return { privateKey, publicKey };
    }

    // Derive encryption key from password and salt
    deriveKey(password, salt) {
        return crypto.pbkdf2Sync(
            password,
            salt,
            KEY_DERIVATION.iterations,
            KEY_DERIVATION.keyLength,
            KEY_DERIVATION.digest
        );
    }

    // Encrypt data with AES-256-GCM
    encrypt(data, password = null) {
        try {
            // Convert data to buffer if it's not already
            const dataBuffer = Buffer.isBuffer(data) ? data : Buffer.from(JSON.stringify(data), 'utf8');
            
            // Generate random salt and IV
            const salt = crypto.randomBytes(ENCRYPTION_CONFIG.saltLength);
            const iv = crypto.randomBytes(ENCRYPTION_CONFIG.ivLength);
            
            // Derive key from password or use master key
            const key = password ? this.deriveKey(password, salt) : this.masterKey;
            
            // Create cipher
            const cipher = crypto.createCipher(ENCRYPTION_CONFIG.algorithm, key);
            cipher.setAAD(salt); // Use salt as additional authenticated data
            
            // Encrypt data
            const encrypted = Buffer.concat([
                cipher.update(dataBuffer),
                cipher.final()
            ]);
            
            // Get authentication tag
            const tag = cipher.getAuthTag();
            
            // Combine all components
            const result = {
                encrypted: encrypted.toString('base64'),
                iv: iv.toString('base64'),
                tag: tag.toString('base64'),
                salt: salt.toString('base64'),
                algorithm: ENCRYPTION_CONFIG.algorithm
            };
            
            logger.debug('Data encrypted successfully', {
                dataSize: dataBuffer.length,
                encryptedSize: encrypted.length,
                algorithm: ENCRYPTION_CONFIG.algorithm
            });
            
            return result;
        } catch (error) {
            logger.error('Encryption failed:', error);
            throw new Error('Encryption failed');
        }
    }

    // Decrypt data with AES-256-GCM
    decrypt(encryptedData, password = null) {
        try {
            const { encrypted, iv, tag, salt, algorithm } = encryptedData;
            
            // Validate algorithm
            if (algorithm !== ENCRYPTION_CONFIG.algorithm) {
                throw new Error('Unsupported encryption algorithm');
            }
            
            // Convert from base64
            const encryptedBuffer = Buffer.from(encrypted, 'base64');
            const ivBuffer = Buffer.from(iv, 'base64');
            const tagBuffer = Buffer.from(tag, 'base64');
            const saltBuffer = Buffer.from(salt, 'base64');
            
            // Derive key from password or use master key
            const key = password ? this.deriveKey(password, saltBuffer) : this.masterKey;
            
            // Create decipher
            const decipher = crypto.createDecipher(algorithm, key);
            decipher.setAAD(saltBuffer);
            decipher.setAuthTag(tagBuffer);
            
            // Decrypt data
            const decrypted = Buffer.concat([
                decipher.update(encryptedBuffer),
                decipher.final()
            ]);
            
            logger.debug('Data decrypted successfully', {
                encryptedSize: encryptedBuffer.length,
                decryptedSize: decrypted.length
            });
            
            return decrypted;
        } catch (error) {
            logger.error('Decryption failed:', error);
            throw new Error('Decryption failed');
        }
    }

    // Encrypt and compress data for efficient transmission
    encryptAndCompress(data, password = null) {
        try {
            // Convert to JSON if not already a string or buffer
            const jsonData = typeof data === 'string' ? data : JSON.stringify(data);
            
            // Compress data first
            const compressed = zlib.gzipSync(jsonData);
            
            // Then encrypt
            const encrypted = this.encrypt(compressed, password);
            
            // Add compression metadata
            encrypted.compressed = true;
            encrypted.originalSize = Buffer.byteLength(jsonData, 'utf8');
            encrypted.compressedSize = compressed.length;
            
            logger.debug('Data encrypted and compressed', {
                originalSize: encrypted.originalSize,
                compressedSize: encrypted.compressedSize,
                compressionRatio: (encrypted.compressedSize / encrypted.originalSize * 100).toFixed(2) + '%'
            });
            
            return encrypted;
        } catch (error) {
            logger.error('Encrypt and compress failed:', error);
            throw new Error('Encrypt and compress failed');
        }
    }

    // Decrypt and decompress data
    decryptAndDecompress(encryptedData, password = null) {
        try {
            // Decrypt data
            const decrypted = this.decrypt(encryptedData, password);
            
            // Decompress if it was compressed
            if (encryptedData.compressed) {
                const decompressed = zlib.gunzipSync(decrypted);
                return decompressed.toString('utf8');
            }
            
            return decrypted.toString('utf8');
        } catch (error) {
            logger.error('Decrypt and decompress failed:', error);
            throw new Error('Decrypt and decompress failed');
        }
    }

    // Create digital signature
    sign(data) {
        try {
            const dataBuffer = Buffer.isBuffer(data) ? data : Buffer.from(JSON.stringify(data), 'utf8');
            
            const signature = crypto.sign(
                SIGNATURE_CONFIG.algorithm,
                dataBuffer,
                {
                    key: this.keyPairs.privateKey,
                    padding: SIGNATURE_CONFIG.padding,
                    saltLength: SIGNATURE_CONFIG.saltLength
                }
            );
            
            logger.debug('Data signed successfully', {
                dataSize: dataBuffer.length,
                signatureSize: signature.length
            });
            
            return {
                signature: signature.toString('base64'),
                algorithm: SIGNATURE_CONFIG.algorithm,
                timestamp: new Date().toISOString()
            };
        } catch (error) {
            logger.error('Signing failed:', error);
            throw new Error('Signing failed');
        }
    }

    // Verify digital signature
    verify(data, signatureData) {
        try {
            const { signature, algorithm, timestamp } = signatureData;
            
            // Validate algorithm
            if (algorithm !== SIGNATURE_CONFIG.algorithm) {
                throw new Error('Unsupported signature algorithm');
            }
            
            const dataBuffer = Buffer.isBuffer(data) ? data : Buffer.from(JSON.stringify(data), 'utf8');
            const signatureBuffer = Buffer.from(signature, 'base64');
            
            const isValid = crypto.verify(
                algorithm,
                dataBuffer,
                {
                    key: this.keyPairs.publicKey,
                    padding: SIGNATURE_CONFIG.padding,
                    saltLength: SIGNATURE_CONFIG.saltLength
                },
                signatureBuffer
            );
            
            logger.debug('Signature verification completed', {
                isValid,
                algorithm,
                timestamp
            });
            
            return isValid;
        } catch (error) {
            logger.error('Signature verification failed:', error);
            return false;
        }
    }

    // Generate secure hash
    hash(data, algorithm = 'sha256') {
        try {
            const dataBuffer = Buffer.isBuffer(data) ? data : Buffer.from(JSON.stringify(data), 'utf8');
            const hash = crypto.createHash(algorithm);
            hash.update(dataBuffer);
            
            const result = {
                hash: hash.digest('hex'),
                algorithm,
                timestamp: new Date().toISOString()
            };
            
            logger.debug('Hash generated', {
                algorithm,
                dataSize: dataBuffer.length,
                hashLength: result.hash.length
            });
            
            return result;
        } catch (error) {
            logger.error('Hashing failed:', error);
            throw new Error('Hashing failed');
        }
    }

    // Generate secure random token
    generateToken(length = 32) {
        return crypto.randomBytes(length).toString('hex');
    }

    // Generate API key with metadata
    generateAPIKey(metadata = {}) {
        const key = this.generateToken(32);
        const keyData = {
            key,
            hash: crypto.createHash('sha256').update(key).digest('hex'),
            created: new Date().toISOString(),
            metadata
        };
        
        logger.info('API key generated', {
            keyHash: keyData.hash.substring(0, 16) + '...',
            metadata
        });
        
        return keyData;
    }

    // Encrypt sensitive configuration
    encryptConfig(config, password) {
        try {
            const encrypted = this.encryptAndCompress(config, password);
            
            // Add integrity check
            const integrity = this.hash(JSON.stringify(config));
            encrypted.integrity = integrity;
            
            return encrypted;
        } catch (error) {
            logger.error('Config encryption failed:', error);
            throw new Error('Config encryption failed');
        }
    }

    // Decrypt sensitive configuration
    decryptConfig(encryptedConfig, password) {
        try {
            const decrypted = this.decryptAndDecompress(encryptedConfig, password);
            const config = JSON.parse(decrypted);
            
            // Verify integrity if available
            if (encryptedConfig.integrity) {
                const currentIntegrity = this.hash(JSON.stringify(config));
                if (currentIntegrity.hash !== encryptedConfig.integrity.hash) {
                    throw new Error('Config integrity check failed');
                }
            }
            
            return config;
        } catch (error) {
            logger.error('Config decryption failed:', error);
            throw new Error('Config decryption failed');
        }
    }

    // Get public key for external parties
    getPublicKey() {
        return this.keyPairs.publicKey;
    }

    // Rotate encryption keys
    rotateKeys() {
        try {
            // Generate new key pairs
            const newKeyPairs = crypto.generateKeyPairSync('rsa', {
                modulusLength: SIGNATURE_CONFIG.keySize,
                publicKeyEncoding: {
                    type: 'spki',
                    format: 'pem'
                },
                privateKeyEncoding: {
                    type: 'pkcs8',
                    format: 'pem'
                }
            });

            // Generate new master key
            const newMasterKey = crypto.randomBytes(ENCRYPTION_CONFIG.keyLength);

            // Backup old keys
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
            const backupDir = path.join(process.cwd(), 'keys', 'backup', timestamp);
            
            if (!fs.existsSync(backupDir)) {
                fs.mkdirSync(backupDir, { recursive: true });
            }

            // Save backup of old keys
            fs.writeFileSync(path.join(backupDir, 'master.key'), this.masterKey);
            fs.writeFileSync(path.join(backupDir, 'private.pem'), this.keyPairs.privateKey);
            fs.writeFileSync(path.join(backupDir, 'public.pem'), this.keyPairs.publicKey);

            // Update current keys
            this.masterKey = newMasterKey;
            this.keyPairs = newKeyPairs;

            // Save new keys
            const keysDir = path.join(process.cwd(), 'keys');
            fs.writeFileSync(path.join(keysDir, 'master.key'), newMasterKey, { mode: 0o600 });
            fs.writeFileSync(path.join(keysDir, 'private.pem'), newKeyPairs.privateKey, { mode: 0o600 });
            fs.writeFileSync(path.join(keysDir, 'public.pem'), newKeyPairs.publicKey, { mode: 0o644 });

            logger.info('Encryption keys rotated successfully', {
                backupLocation: backupDir,
                timestamp
            });

            return {
                success: true,
                backupLocation: backupDir,
                newPublicKey: newKeyPairs.publicKey
            };
        } catch (error) {
            logger.error('Key rotation failed:', error);
            throw new Error('Key rotation failed');
        }
    }
}

// Middleware for automatic request/response encryption
const encryptionMiddleware = (encryptionService) => {
    return {
        // Middleware to decrypt incoming encrypted requests
        decryptRequest: (req, res, next) => {
            try {
                if (req.headers['content-encoding'] === 'encrypted') {
                    const encryptedData = req.body;
                    const decrypted = encryptionService.decryptAndDecompress(encryptedData);
                    req.body = JSON.parse(decrypted);
                    
                    logger.debug('Request decrypted', {
                        originalSize: encryptedData.originalSize,
                        compressedSize: encryptedData.compressedSize
                    });
                }
                next();
            } catch (error) {
                logger.error('Request decryption failed:', error);
                res.status(400).json({ error: 'Invalid encrypted request' });
            }
        },

        // Middleware to encrypt outgoing responses
        encryptResponse: (req, res, next) => {
            const originalSend = res.send;
            
            res.send = function(data) {
                try {
                    // Check if client accepts encrypted responses
                    if (req.headers['accept-encoding'] && req.headers['accept-encoding'].includes('encrypted')) {
                        const encrypted = encryptionService.encryptAndCompress(data);
                        
                        res.setHeader('Content-Encoding', 'encrypted');
                        res.setHeader('Content-Type', 'application/json');
                        
                        logger.debug('Response encrypted', {
                            originalSize: encrypted.originalSize,
                            compressedSize: encrypted.compressedSize
                        });
                        
                        return originalSend.call(this, JSON.stringify(encrypted));
                    }
                    
                    return originalSend.call(this, data);
                } catch (error) {
                    logger.error('Response encryption failed:', error);
                    return originalSend.call(this, data);
                }
            };
            
            next();
        }
    };
};

// Create singleton instance
const encryptionService = new EncryptionService();

module.exports = {
    EncryptionService,
    encryptionService,
    encryptionMiddleware,
    ENCRYPTION_CONFIG,
    SIGNATURE_CONFIG
};
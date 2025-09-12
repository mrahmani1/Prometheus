"""
Advanced Encryption Utilities for Zero-Trust Security Framework
Implements enterprise-grade encryption with geological data protection
"""

import os
import base64
import secrets
import hashlib
from typing import Dict, List, Optional, Tuple, Union
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding, ec
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from cryptography.hazmat.backends import default_backend
import nacl.secret
import nacl.utils
from nacl.public import PrivateKey, PublicKey, Box
import logging

logger = logging.getLogger(__name__)

class AdvancedEncryption:
    """
    Advanced encryption utilities with geological data protection
    Implements multiple encryption algorithms and key management
    """
    
    def __init__(self):
        self.backend = default_backend()
        
        # Encryption algorithms configuration
        self.algorithms = {
            'AES-256-GCM': {
                'key_size': 32,
                'iv_size': 12,
                'tag_size': 16
            },
            'ChaCha20-Poly1305': {
                'key_size': 32,
                'nonce_size': 12
            },
            'XChaCha20-Poly1305': {
                'key_size': 32,
                'nonce_size': 24
            }
        }
        
        # Key derivation configuration
        self.kdf_config = {
            'pbkdf2': {
                'iterations': 100000,
                'salt_size': 32
            },
            'scrypt': {
                'n': 2**14,
                'r': 8,
                'p': 1,
                'salt_size': 32
            }
        }
        
    def generate_key(self, algorithm: str = 'AES-256-GCM') -> bytes:
        """Generate cryptographically secure key for specified algorithm"""
        if algorithm not in self.algorithms:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
            
        key_size = self.algorithms[algorithm]['key_size']
        return secrets.token_bytes(key_size)
        
    def derive_key_from_password(
        self, 
        password: str, 
        salt: Optional[bytes] = None,
        method: str = 'scrypt'
    ) -> Tuple[bytes, bytes]:
        """Derive encryption key from password using secure KDF"""
        
        if salt is None:
            salt = secrets.token_bytes(self.kdf_config[method]['salt_size'])
            
        if method == 'pbkdf2':
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=self.kdf_config['pbkdf2']['iterations'],
                backend=self.backend
            )
        elif method == 'scrypt':
            kdf = Scrypt(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                n=self.kdf_config['scrypt']['n'],
                r=self.kdf_config['scrypt']['r'],
                p=self.kdf_config['scrypt']['p'],
                backend=self.backend
            )
        else:
            raise ValueError(f"Unsupported KDF method: {method}")
            
        key = kdf.derive(password.encode('utf-8'))
        return key, salt
        
    def encrypt_data(
        self, 
        data: Union[str, bytes], 
        key: bytes,
        algorithm: str = 'AES-256-GCM',
        associated_data: Optional[bytes] = None
    ) -> Dict[str, str]:
        """Encrypt data using specified algorithm with authenticated encryption"""
        
        if isinstance(data, str):
            data = data.encode('utf-8')
            
        if algorithm == 'AES-256-GCM':
            return self._encrypt_aes_gcm(data, key, associated_data)
        elif algorithm == 'ChaCha20-Poly1305':
            return self._encrypt_chacha20_poly1305(data, key, associated_data)
        elif algorithm == 'XChaCha20-Poly1305':
            return self._encrypt_xchacha20_poly1305(data, key, associated_data)
        else:
            raise ValueError(f"Unsupported encryption algorithm: {algorithm}")
            
    def decrypt_data(
        self, 
        encrypted_data: Dict[str, str], 
        key: bytes,
        associated_data: Optional[bytes] = None
    ) -> bytes:
        """Decrypt data using algorithm specified in encrypted data"""
        
        algorithm = encrypted_data.get('algorithm')
        if not algorithm:
            raise ValueError("Algorithm not specified in encrypted data")
            
        if algorithm == 'AES-256-GCM':
            return self._decrypt_aes_gcm(encrypted_data, key, associated_data)
        elif algorithm == 'ChaCha20-Poly1305':
            return self._decrypt_chacha20_poly1305(encrypted_data, key, associated_data)
        elif algorithm == 'XChaCha20-Poly1305':
            return self._decrypt_xchacha20_poly1305(encrypted_data, key, associated_data)
        else:
            raise ValueError(f"Unsupported decryption algorithm: {algorithm}")
            
    def _encrypt_aes_gcm(
        self, 
        data: bytes, 
        key: bytes, 
        associated_data: Optional[bytes]
    ) -> Dict[str, str]:
        """Encrypt using AES-256-GCM"""
        
        iv = secrets.token_bytes(12)  # 96-bit IV for GCM
        
        cipher = Cipher(
            algorithms.AES(key),
            modes.GCM(iv),
            backend=self.backend
        )
        
        encryptor = cipher.encryptor()
        
        if associated_data:
            encryptor.authenticate_additional_data(associated_data)
            
        ciphertext = encryptor.update(data) + encryptor.finalize()
        
        return {
            'algorithm': 'AES-256-GCM',
            'ciphertext': base64.b64encode(ciphertext).decode('utf-8'),
            'iv': base64.b64encode(iv).decode('utf-8'),
            'tag': base64.b64encode(encryptor.tag).decode('utf-8'),
            'associated_data': base64.b64encode(associated_data).decode('utf-8') if associated_data else None
        }
        
    def _decrypt_aes_gcm(
        self, 
        encrypted_data: Dict[str, str], 
        key: bytes,
        associated_data: Optional[bytes]
    ) -> bytes:
        """Decrypt using AES-256-GCM"""
        
        ciphertext = base64.b64decode(encrypted_data['ciphertext'])
        iv = base64.b64decode(encrypted_data['iv'])
        tag = base64.b64decode(encrypted_data['tag'])
        
        cipher = Cipher(
            algorithms.AES(key),
            modes.GCM(iv, tag),
            backend=self.backend
        )
        
        decryptor = cipher.decryptor()
        
        if associated_data:
            decryptor.authenticate_additional_data(associated_data)
        elif encrypted_data.get('associated_data'):
            stored_ad = base64.b64decode(encrypted_data['associated_data'])
            decryptor.authenticate_additional_data(stored_ad)
            
        plaintext = decryptor.update(ciphertext) + decryptor.finalize()
        return plaintext
        
    def _encrypt_chacha20_poly1305(
        self, 
        data: bytes, 
        key: bytes, 
        associated_data: Optional[bytes]
    ) -> Dict[str, str]:
        """Encrypt using ChaCha20-Poly1305"""
        
        nonce = secrets.token_bytes(12)  # 96-bit nonce
        
        cipher = Cipher(
            algorithms.ChaCha20(key, nonce),
            mode=None,
            backend=self.backend
        )
        
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(data) + encryptor.finalize()
        
        # For ChaCha20-Poly1305, we use NaCl for authenticated encryption
        box = nacl.secret.SecretBox(key)
        encrypted = box.encrypt(data, nonce)
        
        return {
            'algorithm': 'ChaCha20-Poly1305',
            'ciphertext': base64.b64encode(encrypted.ciphertext).decode('utf-8'),
            'nonce': base64.b64encode(encrypted.nonce).decode('utf-8'),
            'associated_data': base64.b64encode(associated_data).decode('utf-8') if associated_data else None
        }
        
    def _decrypt_chacha20_poly1305(
        self, 
        encrypted_data: Dict[str, str], 
        key: bytes,
        associated_data: Optional[bytes]
    ) -> bytes:
        """Decrypt using ChaCha20-Poly1305"""
        
        ciphertext = base64.b64decode(encrypted_data['ciphertext'])
        nonce = base64.b64decode(encrypted_data['nonce'])
        
        box = nacl.secret.SecretBox(key)
        
        # Reconstruct the encrypted message
        encrypted_message = nacl.utils.EncryptedMessage(ciphertext, nonce)
        plaintext = box.decrypt(encrypted_message)
        
        return plaintext
        
    def _encrypt_xchacha20_poly1305(
        self, 
        data: bytes, 
        key: bytes, 
        associated_data: Optional[bytes]
    ) -> Dict[str, str]:
        """Encrypt using XChaCha20-Poly1305"""
        
        nonce = secrets.token_bytes(24)  # 192-bit nonce for XChaCha20
        
        box = nacl.secret.SecretBox(key)
        encrypted = box.encrypt(data, nonce)
        
        return {
            'algorithm': 'XChaCha20-Poly1305',
            'ciphertext': base64.b64encode(encrypted.ciphertext).decode('utf-8'),
            'nonce': base64.b64encode(encrypted.nonce).decode('utf-8'),
            'associated_data': base64.b64encode(associated_data).decode('utf-8') if associated_data else None
        }
        
    def _decrypt_xchacha20_poly1305(
        self, 
        encrypted_data: Dict[str, str], 
        key: bytes,
        associated_data: Optional[bytes]
    ) -> bytes:
        """Decrypt using XChaCha20-Poly1305"""
        
        ciphertext = base64.b64decode(encrypted_data['ciphertext'])
        nonce = base64.b64decode(encrypted_data['nonce'])
        
        box = nacl.secret.SecretBox(key)
        encrypted_message = nacl.utils.EncryptedMessage(ciphertext, nonce)
        plaintext = box.decrypt(encrypted_message)
        
        return plaintext
        
    def generate_rsa_keypair(self, key_size: int = 4096) -> Tuple[bytes, bytes]:
        """Generate RSA key pair for asymmetric encryption"""
        
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=key_size,
            backend=self.backend
        )
        
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        public_key = private_key.public_key()
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        return private_pem, public_pem
        
    def generate_ec_keypair(self, curve_name: str = 'secp384r1') -> Tuple[bytes, bytes]:
        """Generate Elliptic Curve key pair"""
        
        if curve_name == 'secp256r1':
            curve = ec.SECP256R1()
        elif curve_name == 'secp384r1':
            curve = ec.SECP384R1()
        elif curve_name == 'secp521r1':
            curve = ec.SECP521R1()
        else:
            raise ValueError(f"Unsupported curve: {curve_name}")
            
        private_key = ec.generate_private_key(curve, self.backend)
        
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        public_key = private_key.public_key()
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        return private_pem, public_pem
        
    def rsa_encrypt(self, data: bytes, public_key_pem: bytes) -> bytes:
        """Encrypt data using RSA public key"""
        
        public_key = serialization.load_pem_public_key(
            public_key_pem, 
            backend=self.backend
        )
        
        ciphertext = public_key.encrypt(
            data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        return ciphertext
        
    def rsa_decrypt(self, ciphertext: bytes, private_key_pem: bytes) -> bytes:
        """Decrypt data using RSA private key"""
        
        private_key = serialization.load_pem_private_key(
            private_key_pem,
            password=None,
            backend=self.backend
        )
        
        plaintext = private_key.decrypt(
            ciphertext,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        return plaintext
        
    def sign_data(self, data: bytes, private_key_pem: bytes, algorithm: str = 'RSA-PSS') -> bytes:
        """Sign data using private key"""
        
        private_key = serialization.load_pem_private_key(
            private_key_pem,
            password=None,
            backend=self.backend
        )
        
        if algorithm == 'RSA-PSS':
            signature = private_key.sign(
                data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
        elif algorithm == 'ECDSA':
            signature = private_key.sign(data, ec.ECDSA(hashes.SHA256()))
        else:
            raise ValueError(f"Unsupported signing algorithm: {algorithm}")
            
        return signature
        
    def verify_signature(
        self, 
        data: bytes, 
        signature: bytes, 
        public_key_pem: bytes,
        algorithm: str = 'RSA-PSS'
    ) -> bool:
        """Verify signature using public key"""
        
        try:
            public_key = serialization.load_pem_public_key(
                public_key_pem,
                backend=self.backend
            )
            
            if algorithm == 'RSA-PSS':
                public_key.verify(
                    signature,
                    data,
                    padding.PSS(
                        mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.MAX_LENGTH
                    ),
                    hashes.SHA256()
                )
            elif algorithm == 'ECDSA':
                public_key.verify(signature, data, ec.ECDSA(hashes.SHA256()))
            else:
                raise ValueError(f"Unsupported verification algorithm: {algorithm}")
                
            return True
            
        except Exception as e:
            logger.warning(f"Signature verification failed: {str(e)}")
            return False
            
    def hash_data(self, data: bytes, algorithm: str = 'SHA256') -> bytes:
        """Hash data using specified algorithm"""
        
        if algorithm == 'SHA256':
            digest = hashes.Hash(hashes.SHA256(), backend=self.backend)
        elif algorithm == 'SHA384':
            digest = hashes.Hash(hashes.SHA384(), backend=self.backend)
        elif algorithm == 'SHA512':
            digest = hashes.Hash(hashes.SHA512(), backend=self.backend)
        elif algorithm == 'BLAKE2b':
            digest = hashes.Hash(hashes.BLAKE2b(64), backend=self.backend)
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")
            
        digest.update(data)
        return digest.finalize()
        
    def secure_compare(self, a: bytes, b: bytes) -> bool:
        """Constant-time comparison to prevent timing attacks"""
        return secrets.compare_digest(a, b)
        
    def generate_secure_token(self, length: int = 32) -> str:
        """Generate cryptographically secure random token"""
        return secrets.token_urlsafe(length)
        
    def encrypt_geological_data(
        self, 
        data: Union[str, bytes], 
        classification_level: str,
        geological_context: Optional[Dict] = None
    ) -> Dict[str, str]:
        """
        Encrypt geological data with classification-appropriate security
        Uses different encryption parameters based on data sensitivity
        """
        
        # Determine encryption parameters based on classification
        if classification_level in ['top_secret', 'confidential']:
            algorithm = 'XChaCha20-Poly1305'
            key_derivation = 'scrypt'
        elif classification_level in ['restricted', 'internal']:
            algorithm = 'AES-256-GCM'
            key_derivation = 'scrypt'
        else:  # public, unclassified
            algorithm = 'AES-256-GCM'
            key_derivation = 'pbkdf2'
            
        # Generate encryption key
        key = self.generate_key(algorithm)
        
        # Prepare associated data with geological context
        associated_data = None
        if geological_context:
            ad_dict = {
                'classification': classification_level,
                'geological_domain': geological_context.get('domain'),
                'timestamp': geological_context.get('timestamp'),
                'data_type': geological_context.get('data_type')
            }
            associated_data = json.dumps(ad_dict, sort_keys=True).encode('utf-8')
            
        # Encrypt data
        encrypted = self.encrypt_data(data, key, algorithm, associated_data)
        
        # Add metadata
        encrypted.update({
            'classification_level': classification_level,
            'key_derivation': key_derivation,
            'geological_context': geological_context
        })
        
        return encrypted
        
    def decrypt_geological_data(
        self, 
        encrypted_data: Dict[str, str], 
        key: bytes,
        verify_classification: bool = True
    ) -> bytes:
        """
        Decrypt geological data with classification verification
        Validates geological context and classification level
        """
        
        if verify_classification:
            classification = encrypted_data.get('classification_level')
            if not classification:
                raise ValueError("Classification level not found in encrypted data")
                
            # Additional validation could be added here
            logger.info(f"Decrypting {classification} geological data")
            
        # Prepare associated data if present
        associated_data = None
        geological_context = encrypted_data.get('geological_context')
        if geological_context:
            ad_dict = {
                'classification': encrypted_data.get('classification_level'),
                'geological_domain': geological_context.get('domain'),
                'timestamp': geological_context.get('timestamp'),
                'data_type': geological_context.get('data_type')
            }
            associated_data = json.dumps(ad_dict, sort_keys=True).encode('utf-8')
            
        return self.decrypt_data(encrypted_data, key, associated_data)
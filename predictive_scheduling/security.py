"""
Security utilities for predictive scheduling framework.

This module provides secure handling of API keys, input validation,
and path sanitization to prevent security vulnerabilities.
"""

import os
import re
import secrets
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class SecurityError(Exception):
    """Exception raised for security-related errors."""
    pass


class APIKeyManager:
    """
    Secure API key management.
    
    This class provides secure storage and retrieval of API keys using
    environment variables and optional keyring integration.
    """
    
    @staticmethod
    def get_api_key(service_name: str, username: Optional[str] = None) -> Optional[str]:
        """
        Retrieve API key securely.
        
        Priority order:
        1. Environment variable (SERVICE_NAME_API_KEY)
        2. Keyring (if available)
        3. Return None
        
        Args:
            service_name: Name of the service (e.g., 'openai', 'deepseek')
            username: Optional username for keyring lookup
            
        Returns:
            API key if found, None otherwise
        """
        # Try environment variable first
        env_var_name = f"{service_name.upper()}_API_KEY"
        api_key = os.environ.get(env_var_name)
        
        if api_key:
            logger.debug(f"Found API key for {service_name} in environment")
            return api_key
        
        # Try keyring if available
        try:
            import keyring
            if username is None:
                username = service_name
            
            api_key = keyring.get_password(service_name, username)
            if api_key:
                logger.debug(f"Found API key for {service_name} in keyring")
                return api_key
        except ImportError:
            logger.debug("Keyring not available for API key storage")
        except Exception as e:
            logger.warning(f"Error accessing keyring: {e}")
        
        logger.warning(f"No API key found for {service_name}")
        return None
    
    @staticmethod
    def set_api_key(service_name: str, api_key: str, username: Optional[str] = None) -> bool:
        """
        Store API key securely using keyring.
        
        Args:
            service_name: Name of the service
            api_key: API key to store
            username: Optional username for keyring storage
            
        Returns:
            True if stored successfully, False otherwise
        """
        try:
            import keyring
            if username is None:
                username = service_name
            
            keyring.set_password(service_name, username, api_key)
            logger.info(f"Stored API key for {service_name} in keyring")
            return True
        except ImportError:
            logger.error("Keyring not available for API key storage")
            return False
        except Exception as e:
            logger.error(f"Error storing API key: {e}")
            return False
    
    @staticmethod
    def validate_api_key_format(api_key: str, service_name: str) -> bool:
        """
        Validate API key format for known services.
        
        Args:
            api_key: API key to validate
            service_name: Name of the service
            
        Returns:
            True if format is valid, False otherwise
        """
        if not api_key or not isinstance(api_key, str):
            return False
        
        # Define patterns for known services
        patterns = {
            'openai': r'^sk-[a-zA-Z0-9]{48}$',
            'anthropic': r'^sk-ant-[a-zA-Z0-9\-_]{95}$',
            'deepseek': r'^sk-[a-zA-Z0-9]{48}$',  # Similar to OpenAI
        }
        
        pattern = patterns.get(service_name.lower())
        if pattern:
            return bool(re.match(pattern, api_key))
        
        # For unknown services, just check it's not obviously invalid
        return len(api_key) >= 10 and not api_key.startswith('token-')


class PathValidator:
    """
    Secure path validation and sanitization.
    
    This class provides utilities to prevent path traversal attacks
    and ensure file operations are performed within allowed directories.
    """
    
    def __init__(self, allowed_directories: Optional[List[str]] = None):
        """
        Initialize path validator.
        
        Args:
            allowed_directories: List of allowed base directories
        """
        self.allowed_directories = []
        if allowed_directories:
            for directory in allowed_directories:
                self.allowed_directories.append(Path(directory).resolve())
    
    def validate_path(self, path: str, must_exist: bool = False) -> Path:
        """
        Validate and sanitize a file path.
        
        Args:
            path: Path to validate
            must_exist: Whether the path must exist
            
        Returns:
            Validated Path object
            
        Raises:
            SecurityError: If path is invalid or outside allowed directories
        """
        try:
            # Convert to Path and resolve
            path_obj = Path(path).resolve()
        except (OSError, ValueError) as e:
            raise SecurityError(f"Invalid path: {e}")
        
        # Check if path exists if required
        if must_exist and not path_obj.exists():
            raise SecurityError(f"Path does not exist: {path_obj}")
        
        # Check against allowed directories
        if self.allowed_directories:
            is_allowed = False
            for allowed_dir in self.allowed_directories:
                try:
                    path_obj.relative_to(allowed_dir)
                    is_allowed = True
                    break
                except ValueError:
                    continue
            
            if not is_allowed:
                raise SecurityError(
                    f"Path {path_obj} is not within allowed directories: {self.allowed_directories}"
                )
        
        return path_obj
    
    def sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename to prevent directory traversal.
        
        Args:
            filename: Filename to sanitize
            
        Returns:
            Sanitized filename
        """
        # Remove path separators and dangerous characters
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        filename = re.sub(r'\.+', '.', filename)  # Replace multiple dots
        filename = filename.strip('. ')  # Remove leading/trailing dots and spaces
        
        # Prevent reserved names on Windows
        reserved_names = {
            'CON', 'PRN', 'AUX', 'NUL',
            'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
            'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
        }
        
        if filename.upper() in reserved_names:
            filename = f"_{filename}"
        
        return filename


class InputValidator:
    """
    Input validation utilities for user-provided data.
    """
    
    @staticmethod
    def validate_model_name(model_name: str) -> bool:
        """
        Validate model name format.
        
        Args:
            model_name: Model name to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not model_name or not isinstance(model_name, str):
            return False
        
        # Allow alphanumeric, hyphens, underscores, forward slashes, and dots
        pattern = r'^[a-zA-Z0-9\-_/.]+$'
        return bool(re.match(pattern, model_name)) and len(model_name) <= 200
    
    @staticmethod
    def validate_url(url: str) -> bool:
        """
        Validate URL format for API endpoints.
        
        Args:
            url: URL to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not url or not isinstance(url, str):
            return False
        
        # Simple URL validation - allow http/https with reasonable hosts
        pattern = r'^https?://[a-zA-Z0-9\-._~:/?#[\]@!$&\'()*+,;=%]+$'
        
        if not re.match(pattern, url):
            return False
        
        # Block localhost and private IPs in production
        if 'localhost' in url.lower() or '127.0.0.1' in url:
            logger.warning("Localhost URLs detected - ensure this is intended")
        
        return len(url) <= 2048
    
    @staticmethod
    def validate_json_safe_string(text: str, max_length: int = 10000) -> bool:
        """
        Validate that string is safe for JSON serialization.
        
        Args:
            text: Text to validate
            max_length: Maximum allowed length
            
        Returns:
            True if safe, False otherwise
        """
        if not isinstance(text, str):
            return False
        
        if len(text) > max_length:
            return False
        
        # Check for control characters (except common whitespace)
        for char in text:
            if ord(char) < 32 and char not in '\t\n\r':
                return False
        
        return True


class SecureRandomGenerator:
    """Utilities for generating secure random values."""
    
    @staticmethod
    def generate_token(length: int = 32) -> str:
        """
        Generate cryptographically secure random token.
        
        Args:
            length: Length of token in bytes
            
        Returns:
            Hex-encoded random token
        """
        return secrets.token_hex(length)
    
    @staticmethod
    def generate_session_id() -> str:
        """Generate secure session ID."""
        return secrets.token_urlsafe(32)
    
    @staticmethod
    def hash_string(text: str, salt: Optional[str] = None) -> str:
        """
        Generate secure hash of string.
        
        Args:
            text: Text to hash
            salt: Optional salt (generated if not provided)
            
        Returns:
            Hex-encoded hash
        """
        if salt is None:
            salt = secrets.token_hex(16)
        
        hash_obj = hashlib.sha256()
        hash_obj.update((text + salt).encode('utf-8'))
        return hash_obj.hexdigest()


def create_secure_config(base_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create secure configuration by validating and sanitizing input config.
    
    Args:
        base_config: Base configuration dictionary
        
    Returns:
        Validated and secured configuration
        
    Raises:
        SecurityError: If configuration contains security issues
    """
    secure_config = base_config.copy()
    
    # Validate API keys
    if 'api_key' in secure_config:
        api_key = secure_config['api_key']
        if api_key and api_key.startswith('token-'):
            raise SecurityError("Default/example API key detected - please use a real API key")
    
    # Validate model names
    if 'model_name' in secure_config:
        model_name = secure_config['model_name']
        if not InputValidator.validate_model_name(model_name):
            raise SecurityError(f"Invalid model name: {model_name}")
    
    # Validate URLs
    if 'base_url' in secure_config:
        base_url = secure_config['base_url']
        if not InputValidator.validate_url(base_url):
            raise SecurityError(f"Invalid base URL: {base_url}")
    
    # Validate file paths
    path_validator = PathValidator()
    for key in ['output_dir', 'cache_dir', 'data_dir']:
        if key in secure_config and secure_config[key]:
            try:
                secure_config[key] = str(path_validator.validate_path(secure_config[key]))
            except SecurityError as e:
                logger.warning(f"Path validation warning for {key}: {e}")
    
    return secure_config
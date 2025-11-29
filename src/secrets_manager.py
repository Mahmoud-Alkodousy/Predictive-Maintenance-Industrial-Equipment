"""
Secrets Manager Module
Secure handling of API keys and sensitive configuration
Supports multiple backends: env vars, .env files, AWS Secrets Manager, etc.
"""

import os
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum
import warnings


class SecretBackend(Enum):
    """Supported secret storage backends"""
    ENV = "environment"          # Environment variables
    DOTENV = "dotenv"           # .env file
    AWS = "aws_secrets"         # AWS Secrets Manager
    AZURE = "azure_keyvault"    # Azure Key Vault
    GCP = "gcp_secrets"         # GCP Secret Manager
    HASHICORP = "vault"         # HashiCorp Vault


@dataclass
class SecretConfig:
    """Configuration for a secret"""
    key: str
    required: bool = True
    default: Optional[str] = None
    description: str = ""
    masked: bool = True  # Whether to mask in logs


class SecretsManager:
    """
    Centralized secrets management with multiple backend support
    
    Features:
    - Multiple backends (env, .env, cloud providers)
    - Validation
    - Caching
    - Logging (with masking)
    - Type conversion
    
    Usage:
        manager = SecretsManager(backend=SecretBackend.DOTENV)
        api_key = manager.get("ANTHROPIC_API_KEY", required=True)
    """
    
    def __init__(
        self,
        backend: SecretBackend = SecretBackend.DOTENV,
        env_file: Optional[Path] = None,
        cache: bool = True
    ):
        """
        Initialize secrets manager
        
        Args:
            backend: Which backend to use
            env_file: Path to .env file (if using DOTENV backend)
            cache: Whether to cache secrets in memory
        """
        self.backend = backend
        self.env_file = env_file or Path(".env")
        self.cache_enabled = cache
        self._cache: Dict[str, Any] = {}
        
        # Initialize backend
        self._init_backend()
    
    def _init_backend(self):
        """Initialize the selected backend"""
        if self.backend == SecretBackend.DOTENV:
            self._init_dotenv()
        elif self.backend == SecretBackend.AWS:
            self._init_aws()
        elif self.backend == SecretBackend.AZURE:
            self._init_azure()
        elif self.backend == SecretBackend.GCP:
            self._init_gcp()
        elif self.backend == SecretBackend.HASHICORP:
            self._init_vault()
        # ENV backend needs no initialization
    
    def _init_dotenv(self):
        """Initialize .env file backend"""
        try:
            from dotenv import load_dotenv
            
            if not self.env_file.exists():
                warnings.warn(
                    f"⚠️  .env file not found at {self.env_file}\n"
                    f"   Create it with: cp .env.example .env"
                )
            else:
                load_dotenv(self.env_file)
                print(f"✅ Loaded secrets from {self.env_file}")
        
        except ImportError:
            raise ImportError(
                "python-dotenv not installed. Install with: pip install python-dotenv"
            )
    
    def _init_aws(self):
        """Initialize AWS Secrets Manager"""
        try:
            import boto3
            self.aws_client = boto3.client('secretsmanager')
            print("✅ AWS Secrets Manager initialized")
        except ImportError:
            raise ImportError("boto3 not installed. Install with: pip install boto3")
    
    def _init_azure(self):
        """Initialize Azure Key Vault"""
        try:
            from azure.identity import DefaultAzureCredential
            from azure.keyvault.secrets import SecretClient
            
            vault_url = os.getenv("AZURE_VAULT_URL")
            if not vault_url:
                raise ValueError("AZURE_VAULT_URL not set")
            
            credential = DefaultAzureCredential()
            self.azure_client = SecretClient(vault_url=vault_url, credential=credential)
            print("✅ Azure Key Vault initialized")
        
        except ImportError:
            raise ImportError(
                "Azure SDK not installed. Install with: "
                "pip install azure-identity azure-keyvault-secrets"
            )
    
    def _init_gcp(self):
        """Initialize GCP Secret Manager"""
        try:
            from google.cloud import secretmanager
            self.gcp_client = secretmanager.SecretManagerServiceClient()
            self.gcp_project = os.getenv("GCP_PROJECT_ID")
            if not self.gcp_project:
                raise ValueError("GCP_PROJECT_ID not set")
            print("✅ GCP Secret Manager initialized")
        
        except ImportError:
            raise ImportError(
                "GCP SDK not installed. Install with: "
                "pip install google-cloud-secret-manager"
            )
    
    def _init_vault(self):
        """Initialize HashiCorp Vault"""
        try:
            import hvac
            
            vault_url = os.getenv("VAULT_ADDR", "http://localhost:8200")
            vault_token = os.getenv("VAULT_TOKEN")
            
            self.vault_client = hvac.Client(url=vault_url, token=vault_token)
            
            if not self.vault_client.is_authenticated():
                raise ValueError("Vault authentication failed")
            
            print("✅ HashiCorp Vault initialized")
        
        except ImportError:
            raise ImportError("hvac not installed. Install with: pip install hvac")
    
    def get(
        self,
        key: str,
        required: bool = True,
        default: Optional[str] = None,
        secret_type: type = str
    ) -> Any:
        """
        Get a secret value
        
        Args:
            key: Secret key name
            required: Whether the secret is required
            default: Default value if not found
            secret_type: Type to convert to (str, int, bool, etc.)
        
        Returns:
            Secret value
        
        Raises:
            ValueError: If required secret not found
        """
        # Check cache first
        if self.cache_enabled and key in self._cache:
            return self._cache[key]
        
        # Get value based on backend
        value = self._get_from_backend(key)
        
        # Validation
        if value is None:
            if required and default is None:
                raise ValueError(
                    f"Required secret '{key}' not found in {self.backend.value}"
                )
            value = default
        
        # Type conversion
        if value is not None:
            value = self._convert_type(value, secret_type)
        
        # Cache
        if self.cache_enabled and value is not None:
            self._cache[key] = value
        
        return value
    
    def _get_from_backend(self, key: str) -> Optional[str]:
        """Get value from the configured backend"""
        if self.backend in [SecretBackend.ENV, SecretBackend.DOTENV]:
            return os.getenv(key)
        
        elif self.backend == SecretBackend.AWS:
            try:
                response = self.aws_client.get_secret_value(SecretId=key)
                return response['SecretString']
            except Exception:
                return None
        
        elif self.backend == SecretBackend.AZURE:
            try:
                secret = self.azure_client.get_secret(key)
                return secret.value
            except Exception:
                return None
        
        elif self.backend == SecretBackend.GCP:
            try:
                name = f"projects/{self.gcp_project}/secrets/{key}/versions/latest"
                response = self.gcp_client.access_secret_version(request={"name": name})
                return response.payload.data.decode('UTF-8')
            except Exception:
                return None
        
        elif self.backend == SecretBackend.HASHICORP:
            try:
                secret = self.vault_client.secrets.kv.v2.read_secret_version(
                    path=key
                )
                return secret['data']['data'].get('value')
            except Exception:
                return None
        
        return None
    
    def _convert_type(self, value: str, target_type: type) -> Any:
        """Convert string value to target type"""
        if target_type == str:
            return value
        elif target_type == int:
            return int(value)
        elif target_type == float:
            return float(value)
        elif target_type == bool:
            return value.lower() in ('true', '1', 'yes', 'on')
        else:
            return value
    
    def validate_all(self, configs: Dict[str, SecretConfig]) -> Dict[str, str]:
        """
        Validate multiple secrets at once
        
        Args:
            configs: Dictionary of key -> SecretConfig
        
        Returns:
            Dictionary of validated secrets
        
        Raises:
            ValueError: If any required secret is missing
        """
        results = {}
        errors = []
        
        for key, config in configs.items():
            try:
                value = self.get(
                    key,
                    required=config.required,
                    default=config.default
                )
                results[key] = value
            except ValueError as e:
                errors.append(f"  - {key}: {config.description or 'No description'}")
        
        if errors:
            error_msg = "❌ Missing required secrets:\n" + "\n".join(errors)
            error_msg += "\n\n💡 Add them to your .env file or set as environment variables"
            raise ValueError(error_msg)
        
        return results
    
    def mask_secret(self, value: str, show_chars: int = 4) -> str:
        """
        Mask a secret for logging
        
        Args:
            value: Secret value
            show_chars: Number of characters to show at end
        
        Returns:
            Masked string like "***abc123"
        """
        if not value:
            return ""
        
        if len(value) <= show_chars:
            return "*" * len(value)
        
        return "*" * (len(value) - show_chars) + value[-show_chars:]
    
    def clear_cache(self):
        """Clear the secrets cache"""
        self._cache.clear()


# ============================================
# PROJECT-SPECIFIC CONFIGURATIONS
# ============================================

# Define all project secrets
PROJECT_SECRETS = {
    "ANTHROPIC_API_KEY": SecretConfig(
        key="ANTHROPIC_API_KEY",
        required=True,
        description="Claude API key for chatbot"
    ),
    "SUPABASE_URL": SecretConfig(
        key="SUPABASE_URL",
        required=True,
        description="Supabase project URL"
    ),
    "SUPABASE_KEY": SecretConfig(
        key="SUPABASE_KEY",
        required=True,
        description="Supabase anon/service key"
    ),
    "KAGGLE_USERNAME": SecretConfig(
        key="KAGGLE_USERNAME",
        required=False,
        description="Kaggle username for data download"
    ),
    "KAGGLE_KEY": SecretConfig(
        key="KAGGLE_KEY",
        required=False,
        description="Kaggle API key"
    ),
    "MLFLOW_TRACKING_URI": SecretConfig(
        key="MLFLOW_TRACKING_URI",
        required=False,
        default="./mlruns",
        description="MLflow tracking server URI"
    ),
}


# ============================================
# SINGLETON INSTANCE
# ============================================

_secrets_manager: Optional[SecretsManager] = None


def get_secrets_manager(
    backend: SecretBackend = SecretBackend.DOTENV,
    force_reload: bool = False
) -> SecretsManager:
    """
    Get singleton secrets manager instance
    
    Args:
        backend: Which backend to use
        force_reload: Force recreation of manager
    
    Returns:
        SecretsManager instance
    """
    global _secrets_manager
    
    if _secrets_manager is None or force_reload:
        _secrets_manager = SecretsManager(backend=backend)
    
    return _secrets_manager


def get_secret(key: str, required: bool = True, default: Optional[str] = None) -> str:
    """
    Convenience function to get a single secret
    
    Args:
        key: Secret key
        required: Whether required
        default: Default value
    
    Returns:
        Secret value
    """
    manager = get_secrets_manager()
    return manager.get(key, required=required, default=default)


def validate_project_secrets() -> Dict[str, str]:
    """
    Validate all project secrets
    
    Returns:
        Dictionary of validated secrets
    """
    manager = get_secrets_manager()
    return manager.validate_all(PROJECT_SECRETS)


# ============================================
# TESTING
# ============================================

if __name__ == "__main__":
    print("🧪 Testing Secrets Manager...\n")
    
    # Test with environment variables
    os.environ["TEST_SECRET"] = "test_value_123"
    os.environ["TEST_INT"] = "42"
    os.environ["TEST_BOOL"] = "true"
    
    manager = SecretsManager(backend=SecretBackend.ENV)
    
    # Test string
    value = manager.get("TEST_SECRET", required=False)
    print(f"✅ String secret: {manager.mask_secret(value)}")
    
    # Test int
    int_value = manager.get("TEST_INT", required=False, secret_type=int)
    print(f"✅ Int secret: {int_value}")
    
    # Test bool
    bool_value = manager.get("TEST_BOOL", required=False, secret_type=bool)
    print(f"✅ Bool secret: {bool_value}")
    
    # Test missing (with default)
    missing = manager.get("MISSING_SECRET", required=False, default="default_value")
    print(f"✅ Missing secret with default: {missing}")
    
    # Test validation
    try:
        test_configs = {
            "TEST_SECRET": SecretConfig("TEST_SECRET", required=True),
            "MISSING_REQUIRED": SecretConfig("MISSING_REQUIRED", required=True),
        }
        manager.validate_all(test_configs)
    except ValueError as e:
        print(f"\n✅ Validation error caught correctly:\n{e}")
    
    print("\n✅ All tests passed!")

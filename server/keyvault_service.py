"""
BridgeCast AI — Azure Key Vault Service
Retrieves and sets secrets in Azure Key Vault. Can bootstrap all Azure
service keys from Key Vault into environment variables at startup.
"""

import os
import logging
from typing import Optional

from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Map of Key Vault secret names -> environment variable names
# ---------------------------------------------------------------------------

SECRET_ENV_MAP = {
    "azure-speech-key": "AZURE_SPEECH_KEY",
    "azure-speech-region": "AZURE_SPEECH_REGION",
    "azure-openai-endpoint": "AZURE_OPENAI_ENDPOINT",
    "azure-openai-key": "AZURE_OPENAI_KEY",
    "azure-openai-deployment": "AZURE_OPENAI_DEPLOYMENT",
    "azure-translator-key": "AZURE_TRANSLATOR_KEY",
    "azure-translator-endpoint": "AZURE_TRANSLATOR_ENDPOINT",
    "azure-translator-region": "AZURE_TRANSLATOR_REGION",
    "azure-content-safety-endpoint": "AZURE_CONTENT_SAFETY_ENDPOINT",
    "azure-content-safety-key": "AZURE_CONTENT_SAFETY_KEY",
    "azure-cosmos-endpoint": "AZURE_COSMOS_ENDPOINT",
    "azure-cosmos-key": "AZURE_COSMOS_KEY",
    "azure-communication-connection-string": "AZURE_COMMUNICATION_CONNECTION_STRING",
    "azure-appinsights-connection-string": "AZURE_APPINSIGHTS_CONNECTION_STRING",
}

# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

_client: Optional[SecretClient] = None


def _get_client() -> SecretClient:
    """Return (or create) a SecretClient using DefaultAzureCredential."""
    global _client
    if _client is not None:
        return _client

    vault_url = os.environ.get("AZURE_KEYVAULT_URL")
    if not vault_url:
        raise EnvironmentError(
            "AZURE_KEYVAULT_URL must be set (e.g. https://my-vault.vault.azure.net/)."
        )

    credential = DefaultAzureCredential()
    _client = SecretClient(vault_url=vault_url, credential=credential)
    logger.info("Key Vault client initialised: %s", vault_url)
    return _client


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_secret(name: str) -> Optional[str]:
    """Retrieve a single secret value from Key Vault.

    Parameters
    ----------
    name : str
        The secret name (e.g. "azure-speech-key").

    Returns
    -------
    str or None
        The secret value, or None if not found.
    """
    try:
        client = _get_client()
        secret = client.get_secret(name)
        logger.debug("Secret retrieved: %s", name)
        return secret.value
    except Exception as exc:
        logger.error("Failed to get secret '%s': %s", name, exc)
        return None


def set_secret(name: str, value: str) -> bool:
    """Store a secret in Key Vault.

    Parameters
    ----------
    name : str
        The secret name.
    value : str
        The secret value.

    Returns
    -------
    bool
        True on success, False on failure.
    """
    try:
        client = _get_client()
        client.set_secret(name, value)
        logger.info("Secret stored: %s", name)
        return True
    except Exception as exc:
        logger.error("Failed to set secret '%s': %s", name, exc)
        return False


def load_all_secrets() -> int:
    """Load all known Azure secrets from Key Vault into environment variables.

    Only sets variables that are not already set in the environment,
    so .env overrides take precedence.

    Returns
    -------
    int
        Number of secrets successfully loaded.
    """
    loaded = 0

    for secret_name, env_var in SECRET_ENV_MAP.items():
        # Skip if already set in environment
        if os.environ.get(env_var):
            logger.debug("Skipping %s — already set in environment", env_var)
            continue

        value = get_secret(secret_name)
        if value:
            os.environ[env_var] = value
            loaded += 1
            logger.info("Loaded %s from Key Vault -> %s", secret_name, env_var)
        else:
            logger.debug("Secret '%s' not found in Key Vault", secret_name)

    logger.info("Loaded %d secrets from Key Vault", loaded)
    return loaded

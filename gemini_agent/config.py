"""
Gemini Agent Configuration
This file contains configuration for the Gemini AI agent integration.
"""

import os
from typing import Optional

# ============================================================================
# IMPORTANT: Place your Gemini API key here
# ============================================================================
# To get your Gemini API key:
# 1. Go to https://ai.google.dev/
# 2. Click "Get API Key" button
# 3. Create a new API key in Google Cloud Console
# 4. Copy the key and paste it below or set as environment variable
# ============================================================================

GEMINI_API_KEY: Optional[str] = os.getenv(
    'GEMINI_API_KEY',
)

# Model configuration
GEMINI_MODEL = 'gemini-2.0-flash'  # Latest Gemini model
GEMINI_API_VERSION = 'v1beta'

# Service URLs
COVID_SERVICE_URL = os.getenv('COVID_SERVICE_URL', 'http://localhost:8000')
CHURN_SERVICE_URL = os.getenv('CHURN_SERVICE_URL', 'http://localhost:8001')

# Agent configuration
AGENT_CONFIG = {
    'temperature': 0.7,
    'max_output_tokens': 1024,
    'top_p': 0.95,
    'top_k': 40,
}

# Intent classification thresholds
INTENT_CONFIDENCE_THRESHOLD = 0.3
UNKNOWN_INTENT_THRESHOLD = 0.5

# Request timeouts (in seconds)
REQUEST_TIMEOUT = 30
SERVICE_CALL_TIMEOUT = 30

# Logging configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Feature flags
ENABLE_PARAMETER_EXTRACTION = True
ENABLE_RESPONSE_SYNTHESIS = True
ENABLE_CONFIDENCE_SCORING = True

# Cache configuration
ENABLE_RESPONSE_CACHE = True
CACHE_TTL_SECONDS = 3600  # 1 hour

# Monitoring and metrics
ENABLE_METRICS = True
METRICS_PORT = 8002

# Development mode
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'

def validate_config() -> bool:
    """
    Validate that all required configuration is set.
    
    Returns:
        bool: True if configuration is valid, False otherwise
    """
    if not GEMINI_API_KEY:
        print("⚠️  WARNING: GEMINI_API_KEY is not set!")
        print("   Please set your Gemini API key in gemini_agent/config.py")
        print("   or set the GEMINI_API_KEY environment variable")
        return False
    
    return True

def get_config_summary() -> dict:
    """
    Get a summary of the current configuration.
    
    Returns:
        dict: Configuration summary
    """
    return {
        'gemini_model': GEMINI_MODEL,
        'covid_service_url': COVID_SERVICE_URL,
        'churn_service_url': CHURN_SERVICE_URL,
        'intent_confidence_threshold': INTENT_CONFIDENCE_THRESHOLD,
        'request_timeout': REQUEST_TIMEOUT,
        'debug_mode': DEBUG,
        'api_key_configured': bool(GEMINI_API_KEY and GEMINI_API_KEY != 'zakaria place your api here'),
    }

if __name__ == '__main__':
    print("Gemini Agent Configuration")
    print("=" * 50)
    config_summary = get_config_summary()
    for key, value in config_summary.items():
        print(f"{key}: {value}")
    print("=" * 50)
    
    if not validate_config():
        print("\n❌ Configuration validation failed!")
        exit(1)
    else:
        print("\n✅ Configuration is valid!")

"""
Configuration settings for the RKLLM server.
This file centralizes all configurable parameters to make them easier to manage.
"""

# Model Configuration
MODEL_PATH = "../models/unsloth-gemma-3-1b-it-w8a8.rkllm"
LIBRARY_PATH = "./src/librkllmrt.so"  # Path to the RKLLM runtime library

# Server Configuration
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 1306
API_BASE_PATH = "/v1"
API_KEY = "anything"  # Default API key for authentication (can be any string)

# Model Parameters
MAX_CONTEXT_LENGTH = 16384
MAX_NEW_TOKENS = 8192
N_KEEP = 32
IS_ASYNC = False

# System Prompt (if any)
SYSTEM_PROMPT = "You are a helpful assistant."

# Debug Configuration
DEBUG_MODE = False
LOG_LEVEL = 0  # 0: Error, 1: Warning, 2: Info, 3: Debug

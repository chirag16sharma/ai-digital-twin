"""
Project configuration settings.

This module centralizes project-wide constants, directory paths,
dataset locations, and default configuration values.

Keeping configuration in one place improves maintainability,
reduces hardcoded values, and simplifies future deployment.
"""

from pathlib import Path

# ==========================================================
# Project Directories
# ==========================================================

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Main directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

SRC_DIR = PROJECT_ROOT / "src"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
DOCS_DIR = PROJECT_ROOT / "docs"
MODELS_DIR = PROJECT_ROOT / "models"
TESTS_DIR = PROJECT_ROOT / "tests"

# ==========================================================
# Dataset Paths
# ==========================================================

RAW_DATASET_PATH = RAW_DATA_DIR / "rainfall2025.nc"

PROCESSED_DATASET_PATH = (
    PROCESSED_DATA_DIR / "rainfall_ai_ready.nc"
)

# ==========================================================
# Coordinate Names
# ==========================================================

LATITUDE_NAMES = [
    "LATITUDE",
    "latitude",
    "lat",
]

LONGITUDE_NAMES = [
    "LONGITUDE",
    "longitude",
    "lon",
]

TIME_NAMES = [
    "TIME",
    "time",
]

# ==========================================================
# Feature Engineering
# ==========================================================

ROLLING_WINDOW_SHORT = 7
ROLLING_WINDOW_LONG = 30
DEFAULT_LAG = 1

# ==========================================================
# Simulation Defaults
# ==========================================================

DEFAULT_INCREASE_PERCENT = 10.0
DEFAULT_DECREASE_PERCENT = 10.0

HEAVY_RAIN_THRESHOLD = 100.0

# ==========================================================
# Application Metadata
# ==========================================================

PROJECT_NAME = "AI Digital Twin"

VERSION = "1.0.0"
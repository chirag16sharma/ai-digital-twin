"""
config/settings.py

Central configuration for the AI Digital Twin project. Every module
that needs a file path, a dataset schema constant (variable/coordinate
names), or a default value should import it from here rather than
hardcoding it locally. This is the single place these values change
when the project's data or environment changes.
"""

import logging
from pathlib import Path

# ------------------------------------------------------------------
# Project directories
# ------------------------------------------------------------------

# Root of the project (parent of the config/ folder this file lives in)
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

# ------------------------------------------------------------------
# Dataset paths
# ------------------------------------------------------------------

# Default raw IMD dataset, as loaded by IMDLoader in the pipeline.
# NOTE: adjust this filename to match your actual raw file.
RAW_DATASET_PATH: Path = RAW_DATA_DIR / "imd_rainfall.nc"

# Default processed, ML-ready dataset produced by DataPipeline and
# consumed by DigitalTwin. This replaces the temporary DATASET_PATH
# constant that previously lived directly in main.py.
PROCESSED_DATASET_PATH: Path = PROCESSED_DATA_DIR / "rainfall_ai_ready.nc"

# ------------------------------------------------------------------
# Dataset schema — variable and coordinate naming
# ------------------------------------------------------------------

# The name of the rainfall data variable in the dataset. Currently
# assumed identical across all IMD files; if that ever changes, this
# is the one line to edit.
RAINFALL_VARIABLE_NAME: str = "RAINFALL"

# Possible coordinate name variants, in priority order, used by
# find_coordinate() (src/utils/coordinates.py). IMD NetCDF files are
# inconsistent about naming/casing across dataset versions.
LATITUDE_ALIASES: list[str] = ["LATITUDE", "latitude", "lat"]
LONGITUDE_ALIASES: list[str] = ["LONGITUDE", "longitude", "lon"]
TIME_ALIASES: list[str] = ["TIME", "time"]

# ------------------------------------------------------------------
# Feature engineering defaults
# ------------------------------------------------------------------

ROLLING_AVERAGE_SHORT_WINDOW: int = 7   # days, used by add_7day_average()
ROLLING_AVERAGE_LONG_WINDOW: int = 30   # days, used by add_30day_average()
LAG_FEATURE_DAYS: int = 1               # days, used by add_lag_feature()

# ------------------------------------------------------------------
# Simulation defaults / validation bounds
# ------------------------------------------------------------------

# Used by SimulationEngine._validate_percentage() — a "decrease"
# percentage above this value would produce negative rainfall.
MAX_RAINFALL_DECREASE_PERCENTAGE: float = 100.0

# ------------------------------------------------------------------
# Logging configuration
# ------------------------------------------------------------------

LOG_LEVEL: int = logging.INFO
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
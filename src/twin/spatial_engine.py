"""
twin/spatial_engine.py

Responsible for spatial reasoning over the rainfall dataset: detecting
coordinate naming conventions, finding the nearest grid point to a
requested (latitude, longitude), and retrieving rainfall data for a
specific location. This is the first of the six Digital Twin
components (Spatial, Temporal, State, Query, Simulation, DigitalTwin)
and the only one that understands the dataset's spatial structure.
"""

from typing import Tuple

import numpy as np
import xarray as xr

from src.exceptions import DatasetSchemaError, InvalidCoordinateError
from src.utils.coordinates import find_coordinate
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SpatialEngine:
    """
    Handles spatial queries over an IMD rainfall dataset.

    Single Responsibility:
        Spatial reasoning only — coordinate detection, nearest-grid
        lookup, and location-based rainfall retrieval. Time-based
        reasoning belongs to TemporalEngine; this class only touches
        the TIME dimension incidentally, in rainfall_on_date(), to
        select a date after the spatial lookup is done.

    Attributes:
        dataset (xr.Dataset): The AI-ready rainfall dataset.
        lat_name (str): The actual latitude coordinate name found in
            this dataset (e.g. "LATITUDE" or "lat" — IMD datasets are
            inconsistent about casing).
        lon_name (str): The actual longitude coordinate name found in
            this dataset.
        time_name (str): The actual time coordinate name found in
            this dataset.
        latitudes (np.ndarray): All latitude grid values in the
            dataset, cached at construction time.
        longitudes (np.ndarray): All longitude grid values in the
            dataset, cached at construction time.
    """

    def __init__(self, dataset: xr.Dataset) -> None:
        """
        Initialize the Spatial Engine and auto-detect coordinate names.

        Args:
            dataset: AI-ready rainfall dataset, expected to contain
                latitude, longitude, and time coordinates under one
                of several possible naming conventions.

        Raises:
            CoordinateNotFoundError: If no recognized latitude,
                longitude, or time coordinate name is found in the
                dataset.
        """
        self.dataset: xr.Dataset = dataset

        logger.info("Initializing SpatialEngine")

        # Automatically detect coordinate names (shared implementation
        # in src/utils/coordinates.py — also used by TemporalEngine)
        self.lat_name: str = find_coordinate(
            self.dataset, ["LATITUDE", "latitude", "lat"]
        )

        self.lon_name: str = find_coordinate(
            self.dataset, ["LONGITUDE", "longitude", "lon"]
        )

        self.time_name: str = find_coordinate(
            self.dataset, ["TIME", "time"]
        )

        # Store coordinate arrays
        self.latitudes: np.ndarray = self.dataset[self.lat_name].values
        self.longitudes: np.ndarray = self.dataset[self.lon_name].values

        logger.info(
            f"SpatialEngine ready. lat_name={self.lat_name!r}, "
            f"lon_name={self.lon_name!r}, time_name={self.time_name!r}, "
            f"lat range=[{self.latitudes.min():.2f}, {self.latitudes.max():.2f}], "
            f"lon range=[{self.longitudes.min():.2f}, {self.longitudes.max():.2f}]"
        )

    # _find_coordinate() removed — now imported as find_coordinate()
    # from src.utils.coordinates

    # ... available_coordinates(), _validate_latitude(),
    # _validate_longitude(), nearest_latitude(), nearest_longitude(),
    # nearest_grid(), rainfall_at(), rainfall_on_date() all stay
    # exactly as in the previous message — no changes below this point.
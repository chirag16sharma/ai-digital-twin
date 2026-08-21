"""
twin/spatial_engine.py

Responsible for spatial reasoning over the rainfall dataset: detecting
coordinate naming conventions, finding the nearest grid point to a
requested (latitude, longitude), and retrieving rainfall data for a
specific location. This is the first of the six Digital Twin
components (Spatial, Temporal, State, Query, Simulation, DigitalTwin)
and the only one that understands the dataset's spatial structure.
"""

import numpy as np
import xarray as xr

from config.settings import LATITUDE_ALIASES, LONGITUDE_ALIASES, RAINFALL_VARIABLE_NAME, TIME_ALIASES
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
                of several possible naming conventions (see
                config.settings.LATITUDE_ALIASES / LONGITUDE_ALIASES
                / TIME_ALIASES).

        Raises:
            CoordinateNotFoundError: If no recognized latitude,
                longitude, or time coordinate name is found in the
                dataset.
        """
        self.dataset: xr.Dataset = dataset

        logger.info("Initializing SpatialEngine")

        # Automatically detect coordinate names, using the alias
        # lists defined centrally in config/settings.py rather than
        # literal lists here.
        self.lat_name: str = find_coordinate(self.dataset, LATITUDE_ALIASES)
        self.lon_name: str = find_coordinate(self.dataset, LONGITUDE_ALIASES)
        self.time_name: str = find_coordinate(self.dataset, TIME_ALIASES)

        # Store coordinate arrays
        self.latitudes: np.ndarray = self.dataset[self.lat_name].values
        self.longitudes: np.ndarray = self.dataset[self.lon_name].values

        logger.info(
            f"SpatialEngine ready. lat_name={self.lat_name!r}, "
            f"lon_name={self.lon_name!r}, time_name={self.time_name!r}, "
            f"lat range=[{self.latitudes.min():.2f}, {self.latitudes.max():.2f}], "
            f"lon range=[{self.longitudes.min():.2f}, {self.longitudes.max():.2f}]"
        )

    def available_coordinates(self) -> None:
        """
        Print the available latitude and longitude ranges for this
        dataset, for quick diagnostic/exploration purposes.

        Returns:
            None. Console output only.
        """
        print(f"Latitude ({self.lat_name})")
        print(f"Minimum : {self.latitudes.min()}")
        print(f"Maximum : {self.latitudes.max()}")

        print()

        print(f"Longitude ({self.lon_name})")
        print(f"Minimum : {self.longitudes.min()}")
        print(f"Maximum : {self.longitudes.max()}")

    def _validate_latitude(self, latitude: float) -> None:
        """
        Check that a requested latitude falls within the dataset's
        actual coverage range.

        Args:
            latitude: The requested latitude.

        Raises:
            InvalidCoordinateError: If latitude is outside
                [self.latitudes.min(), self.latitudes.max()].
        """
        lat_min, lat_max = self.latitudes.min(), self.latitudes.max()

        if not (lat_min <= latitude <= lat_max):
            logger.error(
                f"Requested latitude {latitude} is outside dataset "
                f"coverage [{lat_min}, {lat_max}]"
            )
            raise InvalidCoordinateError(
                f"Latitude {latitude} is outside the dataset's "
                f"coverage range [{lat_min}, {lat_max}]."
            )

    def _validate_longitude(self, longitude: float) -> None:
        """
        Check that a requested longitude falls within the dataset's
        actual coverage range.

        Args:
            longitude: The requested longitude.

        Raises:
            InvalidCoordinateError: If longitude is outside
                [self.longitudes.min(), self.longitudes.max()].
        """
        lon_min, lon_max = self.longitudes.min(), self.longitudes.max()

        if not (lon_min <= longitude <= lon_max):
            logger.error(
                f"Requested longitude {longitude} is outside dataset "
                f"coverage [{lon_min}, {lon_max}]"
            )
            raise InvalidCoordinateError(
                f"Longitude {longitude} is outside the dataset's "
                f"coverage range [{lon_min}, {lon_max}]."
            )

    def nearest_latitude(self, latitude: float) -> float:
        """
        Find the latitude grid value closest to a requested latitude.

        Args:
            latitude: The requested latitude, which may not exactly
                match a grid point (IMD data is on a fixed grid,
                e.g. 0.25-degree resolution).

        Returns:
            float: The nearest actual latitude value present in the
                dataset's grid.

        Raises:
            InvalidCoordinateError: If latitude is outside the
                dataset's coverage range.
        """
        self._validate_latitude(latitude)

        idx = np.abs(self.latitudes - latitude).argmin()

        return self.latitudes[idx]

    def nearest_longitude(self, longitude: float) -> float:
        """
        Find the longitude grid value closest to a requested longitude.

        Args:
            longitude: The requested longitude, which may not exactly
                match a grid point.

        Returns:
            float: The nearest actual longitude value present in the
                dataset's grid.

        Raises:
            InvalidCoordinateError: If longitude is outside the
                dataset's coverage range.
        """
        self._validate_longitude(longitude)

        idx = np.abs(self.longitudes - longitude).argmin()

        return self.longitudes[idx]

    def nearest_grid(
        self, latitude: float, longitude: float
    ) -> tuple[float, float]:
        """
        Find the nearest (latitude, longitude) grid point to a
        requested location.

        Args:
            latitude: Requested latitude.
            longitude: Requested longitude.

        Returns:
            tuple[float, float]: The nearest (latitude, longitude)
                pair actually present in the dataset's grid.

        Raises:
            InvalidCoordinateError: If latitude or longitude is
                outside the dataset's coverage range.
        """
        lat = self.nearest_latitude(latitude)
        lon = self.nearest_longitude(longitude)

        logger.debug(
            f"Nearest grid to ({latitude}, {longitude}) "
            f"is ({lat}, {lon})"
        )

        return lat, lon

    def rainfall_at(self, latitude: float, longitude: float) -> xr.DataArray:
        """
        Return the full rainfall time series for the nearest grid
        point to a requested location.

        Args:
            latitude: Requested latitude.
            longitude: Requested longitude.

        Returns:
            xr.DataArray: Rainfall values over all time steps at the
                nearest grid point, indexed along the TIME dimension.

        Raises:
            InvalidCoordinateError: If latitude or longitude is
                outside the dataset's coverage range.
            DatasetSchemaError: If the rainfall variable is not
                present in the dataset.
        """
        lat, lon = self.nearest_grid(latitude, longitude)

        try:
            rainfall: xr.DataArray = self.dataset[RAINFALL_VARIABLE_NAME].sel(
                {
                    self.lat_name: lat,
                    self.lon_name: lon
                }
            )
        except KeyError as exc:
            logger.error(
                f"Dataset is missing the {RAINFALL_VARIABLE_NAME!r} variable"
            )
            raise DatasetSchemaError(
                f"Dataset is missing the required "
                f"{RAINFALL_VARIABLE_NAME!r} variable."
            ) from exc

        return rainfall

    def rainfall_on_date(
        self, latitude: float, longitude: float, date: str
    ) -> xr.DataArray:
        """
        Return rainfall at a given location on a specific date.

        Args:
            latitude: Requested latitude.
            longitude: Requested longitude.
            date: Date string in a format xarray's .sel() can parse
                against the TIME coordinate, e.g. "2025-07-15".

        Returns:
            xr.DataArray: A single rainfall value (as a 0-dimensional
                DataArray) for the nearest grid point on the given
                date.

        Raises:
            InvalidCoordinateError: If latitude or longitude is
                outside the dataset's coverage range.
            DatasetSchemaError: If the rainfall variable is missing,
                or if `date` does not exist in the dataset's TIME
                coordinate.
        """
        rainfall = self.rainfall_at(latitude, longitude)

        try:
            return rainfall.sel(
                {
                    self.time_name: date
                }
            )
        except KeyError as exc:
            logger.error(f"Date {date!r} not found in dataset TIME coordinate")
            raise DatasetSchemaError(
                f"Date {date!r} not found in dataset."
            ) from exc
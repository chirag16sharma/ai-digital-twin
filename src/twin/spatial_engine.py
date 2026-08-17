"""
twin/spatial_engine.py

Responsible for spatial reasoning over the rainfall dataset: detecting
coordinate naming conventions, finding the nearest grid point to a
requested (latitude, longitude), and retrieving rainfall data for a
specific location. This is the first of the six Digital Twin
components (Spatial, Temporal, State, Query, Simulation, DigitalTwin)
and the only one that understands the dataset's spatial structure.
"""

from typing import List, Tuple

import numpy as np
import xarray as xr


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
                _find_coordinate).

        Raises:
            ValueError: If no recognized latitude, longitude, or time
                coordinate name is found in the dataset.
        """
        self.dataset: xr.Dataset = dataset

        # Automatically detect coordinate names
        self.lat_name: str = self._find_coordinate(
            ["LATITUDE", "latitude", "lat"]
        )

        self.lon_name: str = self._find_coordinate(
            ["LONGITUDE", "longitude", "lon"]
        )

        self.time_name: str = self._find_coordinate(
            ["TIME", "time"]
        )

        # Store coordinate arrays
        self.latitudes: np.ndarray = self.dataset[self.lat_name].values
        self.longitudes: np.ndarray = self.dataset[self.lon_name].values

    def _find_coordinate(self, possible_names: List[str]) -> str:
        """
        Find which of several possible coordinate names is actually
        present in the dataset.

        This exists because IMD NetCDF files are inconsistent about
        coordinate naming/casing across dataset versions (e.g.
        "LATITUDE" vs "latitude" vs "lat"), so a single hardcoded
        name would break silently on a differently-formatted file.

        Args:
            possible_names: Candidate coordinate names to check, in
                priority order. The first match found is returned.

        Returns:
            str: The matching coordinate name, exactly as it appears
                in self.dataset.coords.

        Raises:
            ValueError: If none of possible_names exist in the
                dataset's coordinates.
        """
        for name in possible_names:
            if name in self.dataset.coords:
                return name

        raise ValueError(
            f"None of the coordinate names {possible_names} found."
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
        """
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
        """
        idx = np.abs(self.longitudes - longitude).argmin()

        return self.longitudes[idx]

    def nearest_grid(
        self, latitude: float, longitude: float
    ) -> Tuple[float, float]:
        """
        Find the nearest (latitude, longitude) grid point to a
        requested location.

        Args:
            latitude: Requested latitude.
            longitude: Requested longitude.

        Returns:
            Tuple[float, float]: The nearest (latitude, longitude)
                pair actually present in the dataset's grid.
        """
        lat = self.nearest_latitude(latitude)
        lon = self.nearest_longitude(longitude)

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
            KeyError: If "RAINFALL" is not present in the dataset.
        """
        lat, lon = self.nearest_grid(latitude, longitude)

        rainfall: xr.DataArray = self.dataset["RAINFALL"].sel(
            {
                self.lat_name: lat,
                self.lon_name: lon
            }
        )

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
            KeyError: If "RAINFALL" is missing, or if `date` does not
                exist in the dataset's TIME coordinate.
        """
        rainfall = self.rainfall_at(latitude, longitude)

        return rainfall.sel(
            {
                self.time_name: date
            }
        )
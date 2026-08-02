import numpy as np


class SpatialEngine:
    """
    Spatial Engine for the AI Digital Twin.

    Responsibilities:
    - Detect latitude, longitude, and time coordinates.
    - Find the nearest grid point.
    - Return rainfall data for any location.
    """

    def __init__(self, dataset):
        """
        Initialize the Spatial Engine.

        Parameters
        ----------
        dataset : xarray.Dataset
            AI-ready rainfall dataset.
        """

        self.dataset = dataset

        # Automatically detect coordinate names
        self.lat_name = self._find_coordinate(
            ["LATITUDE", "latitude", "lat"]
        )

        self.lon_name = self._find_coordinate(
            ["LONGITUDE", "longitude", "lon"]
        )

        self.time_name = self._find_coordinate(
            ["TIME", "time"]
        )

        # Store coordinate arrays
        self.latitudes = self.dataset[self.lat_name].values
        self.longitudes = self.dataset[self.lon_name].values

    def _find_coordinate(self, possible_names):
        """
        Find the first matching coordinate name.

        Parameters
        ----------
        possible_names : list
            Possible coordinate names.

        Returns
        -------
        str
            Matching coordinate name.
        """

        for name in possible_names:
            if name in self.dataset.coords:
                return name

        raise ValueError(
            f"None of the coordinate names {possible_names} found."
        )

    def available_coordinates(self):
        """
        Display available latitude and longitude ranges.
        """

        print(f"Latitude ({self.lat_name})")
        print(f"Minimum : {self.latitudes.min()}")
        print(f"Maximum : {self.latitudes.max()}")

        print()

        print(f"Longitude ({self.lon_name})")
        print(f"Minimum : {self.longitudes.min()}")
        print(f"Maximum : {self.longitudes.max()}")

    def nearest_latitude(self, latitude):
        """
        Find the nearest latitude.

        Parameters
        ----------
        latitude : float

        Returns
        -------
        float
        """

        idx = np.abs(self.latitudes - latitude).argmin()

        return self.latitudes[idx]

    def nearest_longitude(self, longitude):
        """
        Find the nearest longitude.

        Parameters
        ----------
        longitude : float

        Returns
        -------
        float
        """

        idx = np.abs(self.longitudes - longitude).argmin()

        return self.longitudes[idx]

    def nearest_grid(self, latitude, longitude):
        """
        Find the nearest grid point.

        Parameters
        ----------
        latitude : float
        longitude : float

        Returns
        -------
        tuple
            (latitude, longitude)
        """

        lat = self.nearest_latitude(latitude)
        lon = self.nearest_longitude(longitude)

        return lat, lon

    def rainfall_at(self, latitude, longitude):
        """
        Return rainfall time series for the nearest grid point.

        Parameters
        ----------
        latitude : float
        longitude : float

        Returns
        -------
        xarray.DataArray
        """

        lat, lon = self.nearest_grid(latitude, longitude)

        rainfall = self.dataset["RAINFALL"].sel(
            {
                self.lat_name: lat,
                self.lon_name: lon
            }
        )

        return rainfall

    def rainfall_on_date(self, latitude, longitude, date):
        """
        Return rainfall at a given location on a specific date.

        Parameters
        ----------
        latitude : float
        longitude : float
        date : str
            Example: "2025-07-15"

        Returns
        -------
        xarray.DataArray
        """

        rainfall = self.rainfall_at(latitude, longitude)

        return rainfall.sel(
            {
                self.time_name: date
            }
        )
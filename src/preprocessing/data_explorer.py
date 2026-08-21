"""
preprocessing/data_explorer.py

Responsible for producing a human-readable summary of a loaded IMD
rainfall dataset — dimensions, date range, and basic rainfall
statistics. This is a read-only diagnostic tool: it does not modify
the dataset in any way, which is why every method here only reads
from self.ds and never writes to it.
"""

import xarray as xr

from config.settings import (
    LATITUDE_ALIASES,
    LONGITUDE_ALIASES,
    RAINFALL_VARIABLE_NAME,
    TIME_ALIASES,
)
from src.exceptions import DatasetSchemaError
from src.utils.coordinates import find_coordinate
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataExplorer:
    """
    Provides exploratory summaries of an IMD rainfall xarray Dataset.

    Single Responsibility:
        Describe the dataset to a human (via console output). This
        class does not clean, validate, or transform data.

    Attributes:
        ds (xr.Dataset): The rainfall dataset being explored.
    """

    def __init__(self, dataset: xr.Dataset) -> None:
        """
        Initialize the explorer with a dataset to inspect.

        Args:
            dataset: An xarray Dataset containing the rainfall data
                variable with latitude, longitude, and time
                coordinates (naming auto-detected).
        """
        self.ds: xr.Dataset = dataset

    def summary(self) -> None:
        """
        Print a summary of the dataset to the console.

        Reports:
            - Number of time steps, latitudes, and longitudes
            - The full date range covered by the dataset
            - Minimum, maximum, and average rainfall (in mm)

        Returns:
            None. Console output only.

        Raises:
            DatasetSchemaError: If the rainfall variable, or a
                recognizable latitude/longitude/time coordinate, is
                not present in the dataset.
        """
        logger.info("Generating dataset summary")

        try:
            rainfall: xr.DataArray = self.ds[RAINFALL_VARIABLE_NAME]
        except KeyError as exc:
            logger.error(
                f"Dataset is missing the {RAINFALL_VARIABLE_NAME!r} variable"
            )
            raise DatasetSchemaError(
                f"Dataset is missing the required "
                f"{RAINFALL_VARIABLE_NAME!r} variable."
            ) from exc

        # find_coordinate() already raises CoordinateNotFoundError
        # (a DatasetSchemaError subclass) on its own — no need to
        # wrap it again here.
        time_name = find_coordinate(self.ds, TIME_ALIASES)
        lat_name = find_coordinate(self.ds, LATITUDE_ALIASES)
        lon_name = find_coordinate(self.ds, LONGITUDE_ALIASES)

        time_steps = rainfall.sizes[time_name]
        lat_steps = rainfall.sizes[lat_name]
        lon_steps = rainfall.sizes[lon_name]

        print("=" * 50)
        print("IMD DATASET SUMMARY")
        print("=" * 50)

        print(f"Time Steps : {time_steps}")
        print(f"Latitudes  : {lat_steps}")
        print(f"Longitudes : {lon_steps}")

        print()

        print("Date Range")
        print(self.ds[time_name].values[0], "to", self.ds[time_name].values[-1])

        print()

        print("Rainfall Statistics")

        rainfall_min = float(rainfall.min())
        rainfall_max = float(rainfall.max())
        rainfall_mean = float(rainfall.mean())

        print(f"Minimum : {rainfall_min:.2f} mm")
        print(f"Maximum : {rainfall_max:.2f} mm")
        print(f"Average : {rainfall_mean:.2f} mm")

        logger.info(
            f"Summary generated. Time steps: {time_steps}, "
            f"Lat: {lat_steps}, Lon: {lon_steps}, "
            f"Rainfall range: [{rainfall_min:.2f}, {rainfall_max:.2f}] mm"
        )
"""
preprocessing/data_explorer.py

Responsible for producing a human-readable summary of a loaded IMD
rainfall dataset — dimensions, date range, and basic rainfall
statistics. This is a read-only diagnostic tool: it does not modify
the dataset in any way, which is why every method here only reads
from self.ds and never writes to it.
"""

import xarray as xr

from src.exceptions import DatasetSchemaError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataExplorer:
    """
    Provides exploratory summaries of an IMD rainfall xarray Dataset.

    Single Responsibility:
        Describe the dataset to a human (via console output). This
        class does not clean, validate, or transform data — that's
        DataCleaner's job. DataExplorer exists purely so a developer
        can quickly sanity-check a dataset after loading it.

    Attributes:
        ds (xr.Dataset): The rainfall dataset being explored.
    """

    def __init__(self, dataset: xr.Dataset) -> None:
        """
        Initialize the explorer with a dataset to inspect.

        Args:
            dataset: An xarray Dataset containing at minimum a
                "RAINFALL" data variable with TIME, LATITUDE, and
                LONGITUDE dimensions.
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
            None. This method is for console output / diagnostics
            only — it does not return the computed statistics. If a
            caller needs the actual numbers programmatically, this
            method should be split into a version that returns a
            dict/dataclass in a future refactor.

        Raises:
            DatasetSchemaError: If "RAINFALL", "TIME", "LATITUDE", or
                "LONGITUDE" are not present in the dataset.
        """
        logger.info("Generating dataset summary")

        try:
            rainfall: xr.DataArray = self.ds["RAINFALL"]
        except KeyError as exc:
            logger.error("Dataset is missing the 'RAINFALL' variable")
            raise DatasetSchemaError(
                "Dataset is missing the required 'RAINFALL' variable."
            ) from exc

        try:
            time_steps = rainfall.sizes["TIME"]
            lat_steps = rainfall.sizes["LATITUDE"]
            lon_steps = rainfall.sizes["LONGITUDE"]
        except KeyError as exc:
            logger.error(
                f"Dataset is missing an expected dimension: {exc}"
            )
            raise DatasetSchemaError(
                f"Dataset is missing an expected dimension: {exc}"
            ) from exc

        print("=" * 50)
        print("IMD DATASET SUMMARY")
        print("=" * 50)

        print(f"Time Steps : {time_steps}")
        print(f"Latitudes  : {lat_steps}")
        print(f"Longitudes : {lon_steps}")

        print()

        print("Date Range")
        print(self.ds.TIME.values[0], "to", self.ds.TIME.values[-1])

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
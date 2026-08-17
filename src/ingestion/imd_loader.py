"""
ingestion/imd_loader.py

Responsible for loading raw IMD rainfall datasets from disk into memory
as xarray Datasets. This is the entry point of the data pipeline — every
other component (DataExplorer, DataCleaner, FeatureEngineer, ...) operates
on the Dataset this module produces.
"""

from pathlib import Path

import xarray as xr


class IMDLoader:
    """
    Loads IMD (India Meteorological Department) rainfall data stored in
    NetCDF (.nc) format into an xarray Dataset.

    Single Responsibility:
        This class does exactly one job — read a NetCDF file from disk
        and return it as an xarray Dataset. It does not clean, validate,
        or transform the data; that is the job of DataExplorer,
        DataCleaner, and FeatureEngineer downstream.

    Attributes:
        file_path (Path): Path to the NetCDF file to be loaded.
    """

    def __init__(self, file_path: str | Path) -> None:
        """
        Initialize the loader with a path to a NetCDF dataset.

        Args:
            file_path: Path to the .nc file, as a string or Path object.
                The path is not validated here — validation happens at
                load time, so the loader can be constructed even before
                the file exists (e.g. before a download step completes).
        """
        self.file_path: Path = Path(file_path)

    def load(self) -> xr.Dataset:
        """
        Load the NetCDF dataset from disk.

        Returns:
            xr.Dataset: The raw rainfall dataset, unmodified.

        Raises:
            FileNotFoundError: If no file exists at `self.file_path`.
        """
        if not self.file_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.file_path}"
            )

        dataset: xr.Dataset = xr.open_dataset(self.file_path)

        return dataset
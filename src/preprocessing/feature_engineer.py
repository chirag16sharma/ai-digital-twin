"""
preprocessing/feature_engineer.py

Responsible for deriving new rainfall-related features from the
cleaned dataset — cumulative rainfall, rolling averages, and lag
features. These engineered features are what the ML models (Week 4)
will eventually train on, so correctness here matters more than in
earlier pipeline stages.
"""

from pathlib import Path

import xarray as xr

from config.settings import (
    LAG_FEATURE_DAYS,
    RAINFALL_VARIABLE_NAME,
    ROLLING_AVERAGE_LONG_WINDOW,
    ROLLING_AVERAGE_SHORT_WINDOW,
)
from src.exceptions import DatasetSaveError, DatasetSchemaError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureEngineer:
    """
    Adds derived rainfall features to a cleaned IMD rainfall dataset.

    Single Responsibility:
        Feature derivation — computing new variables from existing
        ones (e.g. rolling averages of RAINFALL). This class assumes
        the incoming dataset has already been cleaned (by
        DataCleaner); it does not handle missing values or negative
        readings itself.

    Attributes:
        ds (xr.Dataset): A working copy of the dataset, to which
            engineered features are added as new data variables.
    """

    def __init__(self, dataset: xr.Dataset) -> None:
        """
        Initialize the engineer with a (cleaned) dataset.

        Args:
            dataset: An xarray Dataset containing the rainfall data
                variable indexed along a "TIME" dimension. A copy is
                made immediately so the caller's original dataset is
                left untouched.
        """
        self.ds: xr.Dataset = dataset.copy()

    def _get_rainfall(self) -> xr.DataArray:
        """
        Retrieve the rainfall variable, raising a domain-specific
        exception if it's missing.

        Shared by every add_*() method below, so the same
        try/except KeyError block doesn't need to be repeated four
        times.

        Returns:
            xr.DataArray: The rainfall data variable.

        Raises:
            DatasetSchemaError: If the rainfall variable is not
                present.
        """
        try:
            return self.ds[RAINFALL_VARIABLE_NAME]
        except KeyError as exc:
            logger.error(
                f"Dataset is missing the {RAINFALL_VARIABLE_NAME!r} variable"
            )
            raise DatasetSchemaError(
                f"Dataset is missing the required "
                f"{RAINFALL_VARIABLE_NAME!r} variable."
            ) from exc

    def add_cumulative_rainfall(self) -> None:
        """
        Add a "CUMULATIVE_RAINFALL" variable: the running total of
        rainfall over time, computed via cumulative sum along TIME.

        Returns:
            None. Mutates self.ds in place by adding the new
            variable; use get_dataset() to retrieve the result.

        Raises:
            DatasetSchemaError: If the rainfall variable or the
                "TIME" dimension is not present in the dataset.
        """
        logger.info("Adding feature: CUMULATIVE_RAINFALL")

        rainfall = self._get_rainfall()

        try:
            self.ds["CUMULATIVE_RAINFALL"] = rainfall.cumsum(dim="TIME")
        except ValueError as exc:
            logger.error(f"Failed to compute cumulative rainfall: {exc}")
            raise DatasetSchemaError(
                f"Dataset is missing the 'TIME' dimension required "
                f"for cumulative rainfall: {exc}"
            ) from exc

        logger.info("CUMULATIVE_RAINFALL added successfully")

    def add_short_window_average(self) -> None:
        """
        Add a rolling mean of rainfall along TIME, using a window of
        config.settings.ROLLING_AVERAGE_SHORT_WINDOW days (7 by
        default). The resulting variable is named
        "RAINFALL_{window}DAY_AVG", e.g. "RAINFALL_7DAY_AVG".

        min_periods=1 means the average is computed even near the
        start of the series, using however many days are available,
        rather than producing NaN until a full window exists.

        Returns:
            None. Mutates self.ds in place.

        Raises:
            DatasetSchemaError: If the rainfall variable or the
                "TIME" dimension is not present in the dataset.
        """
        window = ROLLING_AVERAGE_SHORT_WINDOW
        variable_name = f"RAINFALL_{window}DAY_AVG"

        logger.info(f"Adding feature: {variable_name}")

        rainfall = self._get_rainfall()

        try:
            self.ds[variable_name] = (
                rainfall.rolling(TIME=window, min_periods=1).mean()
            )
        except ValueError as exc:
            logger.error(f"Failed to compute {window}-day average: {exc}")
            raise DatasetSchemaError(
                f"Dataset is missing the 'TIME' dimension required "
                f"for the {window}-day rolling average: {exc}"
            ) from exc

        logger.info(f"{variable_name} added successfully")

    def add_long_window_average(self) -> None:
        """
        Add a rolling mean of rainfall along TIME, using a window of
        config.settings.ROLLING_AVERAGE_LONG_WINDOW days (30 by
        default). The resulting variable is named
        "RAINFALL_{window}DAY_AVG", e.g. "RAINFALL_30DAY_AVG".

        See add_short_window_average() for the meaning of
        min_periods=1.

        Returns:
            None. Mutates self.ds in place.

        Raises:
            DatasetSchemaError: If the rainfall variable or the
                "TIME" dimension is not present in the dataset.
        """
        window = ROLLING_AVERAGE_LONG_WINDOW
        variable_name = f"RAINFALL_{window}DAY_AVG"

        logger.info(f"Adding feature: {variable_name}")

        rainfall = self._get_rainfall()

        try:
            self.ds[variable_name] = (
                rainfall.rolling(TIME=window, min_periods=1).mean()
            )
        except ValueError as exc:
            logger.error(f"Failed to compute {window}-day average: {exc}")
            raise DatasetSchemaError(
                f"Dataset is missing the 'TIME' dimension required "
                f"for the {window}-day rolling average: {exc}"
            ) from exc

        logger.info(f"{variable_name} added successfully")

    def add_lag_feature(self) -> None:
        """
        Add a "PREVIOUS_DAY_RAINFALL" variable: each time step's
        rainfall value shifted forward by config.settings.
        LAG_FEATURE_DAYS days (1 by default), so that at time t this
        variable holds the rainfall value from time t-1.

        The first LAG_FEATURE_DAYS time step(s) will have a missing
        (NaN) value here, since there is no prior data to reference.

        Returns:
            None. Mutates self.ds in place.

        Raises:
            DatasetSchemaError: If the rainfall variable or the
                "TIME" dimension is not present in the dataset.
        """
        logger.info("Adding feature: PREVIOUS_DAY_RAINFALL")

        rainfall = self._get_rainfall()

        try:
            self.ds["PREVIOUS_DAY_RAINFALL"] = rainfall.shift(TIME=LAG_FEATURE_DAYS)
        except ValueError as exc:
            logger.error(f"Failed to compute lag feature: {exc}")
            raise DatasetSchemaError(
                f"Dataset is missing the 'TIME' dimension required "
                f"for the lag feature: {exc}"
            ) from exc

        logger.info("PREVIOUS_DAY_RAINFALL added successfully")

    def get_dataset(self) -> xr.Dataset:
        """
        Retrieve the dataset with all engineered features added so far.

        Returns:
            xr.Dataset: self.ds, including any features added by the
                add_*() methods called before this.
        """
        return self.ds

    def save(self, output_path: str | Path) -> None:
        """
        Save the feature-engineered dataset to disk as a NetCDF file.

        Args:
            output_path: Destination path for the .nc file, as a
                string or Path object. Parent directories are not
                created automatically — they must already exist.

        Returns:
            None.

        Raises:
            DatasetSaveError: If the file cannot be written — e.g.
                the parent directory of output_path does not exist.
        """
        output_path = Path(output_path)

        logger.info(f"Saving feature dataset to: {output_path}")

        try:
            self.ds.to_netcdf(output_path)
        except FileNotFoundError as exc:
            logger.error(f"Failed to save dataset to {output_path}: {exc}")
            raise DatasetSaveError(
                f"Could not save dataset to {output_path} — "
                f"check that the parent directory exists."
            ) from exc

        print(f"Feature dataset saved to: {output_path}")
        logger.info(f"Feature dataset saved successfully to: {output_path}")
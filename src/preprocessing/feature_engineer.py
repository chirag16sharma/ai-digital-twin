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
            dataset: An xarray Dataset containing a "RAINFALL" data
                variable indexed along a "TIME" dimension. A copy is
                made immediately so the caller's original dataset is
                left untouched.
        """
        self.ds: xr.Dataset = dataset.copy()

    def add_cumulative_rainfall(self) -> None:
        """
        Add a "CUMULATIVE_RAINFALL" variable: the running total of
        rainfall over time, computed via cumulative sum along TIME.

        Returns:
            None. Mutates self.ds in place by adding the new
            variable; use get_dataset() to retrieve the result.

        Raises:
            KeyError: If "RAINFALL" or the "TIME" dimension is not
                present in the dataset.
        """
        self.ds["CUMULATIVE_RAINFALL"] = (
            self.ds["RAINFALL"].cumsum(dim="TIME")
        )

    def add_7day_average(self) -> None:
        """
        Add a "RAINFALL_7DAY_AVG" variable: a 7-day rolling mean of
        rainfall along TIME.

        min_periods=1 means the average is computed even near the
        start of the series, using however many days are available
        (e.g. the average for day 1 is just day 1's value, day 2 is
        the mean of days 1-2, etc.) rather than producing NaN until
        a full 7-day window exists.

        Returns:
            None. Mutates self.ds in place.

        Raises:
            KeyError: If "RAINFALL" or the "TIME" dimension is not
                present in the dataset.
        """
        self.ds["RAINFALL_7DAY_AVG"] = (
            self.ds["RAINFALL"]
            .rolling(TIME=7, min_periods=1)
            .mean()
        )

    def add_30day_average(self) -> None:
        """
        Add a "RAINFALL_30DAY_AVG" variable: a 30-day rolling mean of
        rainfall along TIME. See add_7day_average() for the meaning
        of min_periods=1.

        Returns:
            None. Mutates self.ds in place.

        Raises:
            KeyError: If "RAINFALL" or the "TIME" dimension is not
                present in the dataset.
        """
        self.ds["RAINFALL_30DAY_AVG"] = (
            self.ds["RAINFALL"]
            .rolling(TIME=30, min_periods=1)
            .mean()
        )

    def add_lag_feature(self) -> None:
        """
        Add a "PREVIOUS_DAY_RAINFALL" variable: each time step's
        rainfall value shifted forward by one day, so that at time t
        this variable holds the rainfall value from time t-1.

        The first time step will have a missing (NaN) value here,
        since there is no prior day to reference.

        Returns:
            None. Mutates self.ds in place.

        Raises:
            KeyError: If "RAINFALL" or the "TIME" dimension is not
                present in the dataset.
        """
        self.ds["PREVIOUS_DAY_RAINFALL"] = (
            self.ds["RAINFALL"].shift(TIME=1)
        )

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
            FileNotFoundError: If the parent directory of
                output_path does not exist.
        """
        self.ds.to_netcdf(output_path)

        print(f"Feature dataset saved to: {output_path}")
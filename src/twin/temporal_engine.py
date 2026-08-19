"""
twin/temporal_engine.py

Responsible for temporal reasoning over the rainfall dataset: detecting
the time coordinate, reporting the available date range, and
retrieving rainfall data for a specific date or date range. This is
the temporal counterpart to SpatialEngine — where SpatialEngine
answers "where," TemporalEngine answers "when."
"""

import numpy as np
import xarray as xr

from src.exceptions import DatasetSchemaError, InvalidDateError
from src.utils.coordinates import find_coordinate
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TemporalEngine:
    """
    Handles temporal queries over an IMD rainfall dataset.

    Single Responsibility:
        Time-based reasoning only — time coordinate detection, date
        range reporting, and date-based rainfall retrieval. Location
        is out of scope here; this class always queries rainfall
        across the *entire spatial grid* for a given date, not for a
        specific point (that combination is QueryEngine's job).

    Attributes:
        dataset (xr.Dataset): The AI-ready rainfall dataset.
        time_name (str): The actual time coordinate name found in
            this dataset (e.g. "TIME" or "time").
        times (np.ndarray): All timestamp values in the dataset,
            cached at construction time.
    """

    def __init__(self, dataset: xr.Dataset) -> None:
        """
        Initialize the Temporal Engine and auto-detect the time
        coordinate.

        Args:
            dataset: AI-ready rainfall dataset, expected to contain a
                time coordinate under one of several possible naming
                conventions.

        Raises:
            CoordinateNotFoundError: If no recognized time coordinate
                name is found in the dataset.
        """
        self.dataset: xr.Dataset = dataset

        logger.info("Initializing TemporalEngine")

        # Automatically detect the time coordinate (shared
        # implementation in src/utils/coordinates.py — also used by
        # SpatialEngine)
        self.time_name: str = find_coordinate(
            self.dataset, ["TIME", "time"]
        )

        # Store all timestamps
        self.times: np.ndarray = self.dataset[self.time_name].values

        logger.info(
            f"TemporalEngine ready. time_name={self.time_name!r}, "
            f"date range=[{self.times[0]}, {self.times[-1]}], "
            f"total days={len(self.times)}"
        )

    # _find_coordinate() removed — now imported as find_coordinate()
    # from src.utils.coordinates

    # ... available_dates(), first_date(), last_date(),
    # number_of_days(), get_date(), get_date_range() all stay exactly
    # as in the previous message — no changes below this point.
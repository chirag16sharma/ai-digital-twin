"""
twin/simulation_engine.py

Responsible for running "what-if" rainfall simulations — increasing,
decreasing, or overriding rainfall values across a date range —
without ever mutating the original dataset. All simulation methods
operate on a deep-copied working dataset, so the Digital Twin can run
an unlimited number of scenarios and always fall back to real data
via reset().
"""

import xarray as xr

from config.settings import MAX_RAINFALL_DECREASE_PERCENTAGE, RAINFALL_VARIABLE_NAME, TIME_ALIASES
from src.exceptions import DatasetSchemaError, SimulationError
from src.utils.coordinates import find_coordinate
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SimulationEngine:
    """
    Runs rainfall "what-if" simulations on a deep copy of the dataset.

    Single Responsibility:
        Simulation only — applying hypothetical rainfall adjustments
        (increase, decrease, dry spell, heavy rainfall) to a working
        copy of the data. This class has no awareness of coordinates
        beyond the TIME dimension, and does no spatial reasoning
        (that's SpatialEngine's job) or state tracking (StateManager's
        job).

    Attributes:
        original_dataset (xr.Dataset): The untouched source dataset,
            kept as a reference so simulations can always be reset
            back to real data.
        simulated_dataset (xr.Dataset): A deep copy of the dataset
            that simulation methods mutate. This is the dataset
            returned by every simulate_* method and get_dataset().
        time_name (str): The actual time coordinate name used by
            dry_spell() and heavy_rainfall() for date-range slicing.
    """

    def __init__(self, dataset: xr.Dataset, time_name: str | None = None) -> None:
        """
        Initialize the Simulation Engine.

        Args:
            dataset: The source rainfall dataset to simulate against.
                A deep copy is taken immediately, so the caller's
                original dataset is never mutated by any simulation.
            time_name: The actual time coordinate name to use for
                date-range operations (dry_spell, heavy_rainfall).
                If None (default), it is auto-detected from the
                dataset using config.settings.TIME_ALIASES — the
                same detection SpatialEngine and TemporalEngine use.
                Callers that already have a TemporalEngine instance
                should pass its time_name explicitly (e.g.
                temporal_engine.time_name) to avoid re-detecting it
                and to guarantee all three engines agree on the same
                coordinate name.

        Raises:
            CoordinateNotFoundError: If time_name is None and no
                recognized time coordinate name is found in the
                dataset.

        Note:
            This replaces the previous behavior where "TIME" was
            hardcoded directly inside dry_spell() and
            heavy_rainfall(). Flagged since the Day 1 code review —
            resolved today by accepting time_name at construction
            instead. self.original_dataset still stores a reference
            rather than a defensive copy — that item remains open.
        """
        self.original_dataset: xr.Dataset = dataset
        self.simulated_dataset: xr.Dataset = dataset.copy(deep=True)

        if time_name is None:
            time_name = find_coordinate(dataset, TIME_ALIASES)

        self.time_name: str = time_name

        logger.info(f"SimulationEngine initialized. time_name={self.time_name!r}")

    def reset(self) -> None:
        """
        Discard all simulated changes and restore simulated_dataset
        to a fresh deep copy of the original data.

        Returns:
            None.
        """
        self.simulated_dataset = self.original_dataset.copy(deep=True)
        logger.info("Simulation reset — simulated_dataset restored to original")

    def _validate_percentage(self, percentage: float, operation: str) -> None:
        """
        Validate a percentage argument for rainfall_increase() or
        rainfall_decrease().

        Args:
            percentage: The percentage value to validate.
            operation: "increase" or "decrease", used only to produce
                a clear error message naming which operation failed.

        Raises:
            SimulationError: If percentage is negative, or if a
                "decrease" percentage exceeds
                config.settings.MAX_RAINFALL_DECREASE_PERCENTAGE
                (which would flip rainfall negative — physically
                invalid).
        """
        if percentage < 0:
            logger.error(
                f"Invalid {operation} percentage: {percentage} "
                f"(must be non-negative)"
            )
            raise SimulationError(
                f"Rainfall {operation} percentage must be non-negative, "
                f"got {percentage}."
            )

        if operation == "decrease" and percentage > MAX_RAINFALL_DECREASE_PERCENTAGE:
            logger.error(
                f"Invalid decrease percentage: {percentage} "
                f"(exceeds {MAX_RAINFALL_DECREASE_PERCENTAGE}, "
                f"would produce negative rainfall)"
            )
            raise SimulationError(
                f"Rainfall decrease percentage cannot exceed "
                f"{MAX_RAINFALL_DECREASE_PERCENTAGE} (would produce "
                f"negative rainfall), got {percentage}."
            )

    def rainfall_increase(self, percentage: float) -> xr.Dataset:
        """
        Increase rainfall across the entire simulated dataset by a
        given percentage.

        Args:
            percentage: The percentage increase to apply, e.g. 20 for
                a 20% increase (factor = 1.20). Must be non-negative.

        Returns:
            xr.Dataset: The updated simulated_dataset (same object,
                mutated in place — not a new copy).

        Raises:
            SimulationError: If percentage is negative.
        """
        self._validate_percentage(percentage, operation="increase")

        factor = 1 + (percentage / 100)

        self.simulated_dataset[RAINFALL_VARIABLE_NAME] = (
            self.simulated_dataset[RAINFALL_VARIABLE_NAME] * factor
        )

        logger.info(f"Simulated rainfall increase: {percentage}% (factor={factor})")

        return self.simulated_dataset

    def rainfall_decrease(self, percentage: float) -> xr.Dataset:
        """
        Decrease rainfall across the entire simulated dataset by a
        given percentage.

        Args:
            percentage: The percentage decrease to apply, e.g. 20 for
                a 20% decrease (factor = 0.80). Must be in
                [0, MAX_RAINFALL_DECREASE_PERCENTAGE].

        Returns:
            xr.Dataset: The updated simulated_dataset (same object,
                mutated in place).

        Raises:
            SimulationError: If percentage is negative or exceeds
                MAX_RAINFALL_DECREASE_PERCENTAGE.
        """
        self._validate_percentage(percentage, operation="decrease")

        factor = 1 - (percentage / 100)

        self.simulated_dataset[RAINFALL_VARIABLE_NAME] = (
            self.simulated_dataset[RAINFALL_VARIABLE_NAME] * factor
        )

        logger.info(f"Simulated rainfall decrease: {percentage}% (factor={factor})")

        return self.simulated_dataset

    def dry_spell(self, start_date: str, end_date: str) -> xr.Dataset:
        """
        Simulate a dry spell: force rainfall to 0 for every day in
        the given date range (inclusive).

        Args:
            start_date: Start of the dry spell, e.g. "2025-07-01".
            end_date: End of the dry spell, e.g. "2025-07-10".

        Returns:
            xr.Dataset: The updated simulated_dataset.

        Raises:
            DatasetSchemaError: If the rainfall variable or the time
                dimension is not present in the dataset.
        """
        try:
            self.simulated_dataset[RAINFALL_VARIABLE_NAME].loc[
                {self.time_name: slice(start_date, end_date)}
            ] = 0
        except KeyError as exc:
            logger.error(
                f"Failed to apply dry spell [{start_date}, {end_date}]: {exc}"
            )
            raise DatasetSchemaError(
                f"Could not apply dry spell — dataset may be missing "
                f"{RAINFALL_VARIABLE_NAME!r} or the {self.time_name!r} "
                f"dimension: {exc}"
            ) from exc

        logger.info(f"Simulated dry spell: [{start_date}, {end_date}] set to 0mm")

        return self.simulated_dataset

    def heavy_rainfall(
        self, start_date: str, end_date: str, multiplier: float
    ) -> xr.Dataset:
        """
        Simulate heavy rainfall: multiply rainfall by a given factor
        for every day in the given date range (inclusive).

        Args:
            start_date: Start of the heavy rainfall period.
            end_date: End of the heavy rainfall period.
            multiplier: Factor to multiply rainfall by, e.g. 2.0 to
                double rainfall. Must be non-negative.

        Returns:
            xr.Dataset: The updated simulated_dataset.

        Raises:
            SimulationError: If multiplier is negative.
            DatasetSchemaError: If the rainfall variable or the time
                dimension is not present in the dataset.
        """
        if multiplier < 0:
            logger.error(f"Invalid heavy_rainfall multiplier: {multiplier}")
            raise SimulationError(
                f"heavy_rainfall multiplier must be non-negative, "
                f"got {multiplier}."
            )

        try:
            self.simulated_dataset[RAINFALL_VARIABLE_NAME].loc[
                {self.time_name: slice(start_date, end_date)}
            ] *= multiplier
        except KeyError as exc:
            logger.error(
                f"Failed to apply heavy rainfall [{start_date}, {end_date}]: {exc}"
            )
            raise DatasetSchemaError(
                f"Could not apply heavy rainfall — dataset may be missing "
                f"{RAINFALL_VARIABLE_NAME!r} or the {self.time_name!r} "
                f"dimension: {exc}"
            ) from exc

        logger.info(
            f"Simulated heavy rainfall: [{start_date}, {end_date}] "
            f"multiplied by {multiplier}"
        )

        return self.simulated_dataset

    def get_dataset(self) -> xr.Dataset:
        """
        Return the current simulated dataset, reflecting all
        simulations applied so far.

        Returns:
            xr.Dataset: simulated_dataset as it currently stands.
        """
        return self.simulated_dataset
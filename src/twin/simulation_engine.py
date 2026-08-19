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

from src.exceptions import DatasetSchemaError, SimulationError
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
    """

    def __init__(self, dataset: xr.Dataset) -> None:
        """
        Initialize the Simulation Engine.

        Args:
            dataset: The source rainfall dataset to simulate against.
                A deep copy is taken immediately, so the caller's
                original dataset is never mutated by any simulation.

        Note:
            self.original_dataset stores a reference to the caller's
            dataset object, not a copy of it. See the Day 2 note on
            this — still an open item, not addressed today since
            Day 3 is scoped to logging/exceptions, not defensive
            copying.
        """
        self.original_dataset: xr.Dataset = dataset
        self.simulated_dataset: xr.Dataset = dataset.copy(deep=True)

        logger.info("SimulationEngine initialized")

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
                "decrease" percentage exceeds 100 (which would flip
                rainfall negative — physically invalid).
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

        if operation == "decrease" and percentage > 100:
            logger.error(
                f"Invalid decrease percentage: {percentage} "
                f"(exceeds 100, would produce negative rainfall)"
            )
            raise SimulationError(
                f"Rainfall decrease percentage cannot exceed 100 "
                f"(would produce negative rainfall), got {percentage}."
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

        self.simulated_dataset["RAINFALL"] = (
            self.simulated_dataset["RAINFALL"] * factor
        )

        logger.info(f"Simulated rainfall increase: {percentage}% (factor={factor})")

        return self.simulated_dataset

    def rainfall_decrease(self, percentage: float) -> xr.Dataset:
        """
        Decrease rainfall across the entire simulated dataset by a
        given percentage.

        Args:
            percentage: The percentage decrease to apply, e.g. 20 for
                a 20% decrease (factor = 0.80). Must be in [0, 100].

        Returns:
            xr.Dataset: The updated simulated_dataset (same object,
                mutated in place).

        Raises:
            SimulationError: If percentage is negative or exceeds
                100.
        """
        self._validate_percentage(percentage, operation="decrease")

        factor = 1 - (percentage / 100)

        self.simulated_dataset["RAINFALL"] = (
            self.simulated_dataset["RAINFALL"] * factor
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
            DatasetSchemaError: If "RAINFALL" or the "TIME" dimension
                is not present in the dataset.

        Note:
            Uses the hardcoded dimension name "TIME" rather than an
            auto-detected time coordinate name. Still flagged from
            Day 1/2 — the proper fix is for SimulationEngine to
            accept a time_name parameter at construction (mirroring
            how QueryEngine receives its dependencies), rather than
            assuming "TIME". Not addressed today, since it requires
            a constructor signature change, which is out of scope
            for a logging/exceptions pass.
        """
        try:
            self.simulated_dataset["RAINFALL"].loc[
                dict(TIME=slice(start_date, end_date))
            ] = 0
        except KeyError as exc:
            logger.error(
                f"Failed to apply dry spell [{start_date}, {end_date}]: {exc}"
            )
            raise DatasetSchemaError(
                f"Could not apply dry spell — dataset may be missing "
                f"'RAINFALL' or a 'TIME' dimension: {exc}"
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
            DatasetSchemaError: If "RAINFALL" or the "TIME" dimension
                is not present in the dataset.

        Note:
            Same hardcoded "TIME" issue as dry_spell() — see that
            method's docstring for details.
        """
        if multiplier < 0:
            logger.error(f"Invalid heavy_rainfall multiplier: {multiplier}")
            raise SimulationError(
                f"heavy_rainfall multiplier must be non-negative, "
                f"got {multiplier}."
            )

        try:
            self.simulated_dataset["RAINFALL"].loc[
                dict(TIME=slice(start_date, end_date))
            ] *= multiplier
        except KeyError as exc:
            logger.error(
                f"Failed to apply heavy rainfall [{start_date}, {end_date}]: {exc}"
            )
            raise DatasetSchemaError(
                f"Could not apply heavy rainfall — dataset may be missing "
                f"'RAINFALL' or a 'TIME' dimension: {exc}"
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
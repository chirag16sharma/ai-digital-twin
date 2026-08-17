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
            dataset object, not a copy of it. If the caller mutates
            their own dataset elsewhere after passing it in here,
            that change would be reflected in original_dataset too.
            Only simulated_dataset is protected via deep copy. Worth
            confirming whether original_dataset should also be
            defensively copied at construction time.
        """
        self.original_dataset: xr.Dataset = dataset
        self.simulated_dataset: xr.Dataset = dataset.copy(deep=True)

    def reset(self) -> None:
        """
        Discard all simulated changes and restore simulated_dataset
        to a fresh deep copy of the original data.

        Returns:
            None.
        """
        self.simulated_dataset = self.original_dataset.copy(deep=True)

    def rainfall_increase(self, percentage: float) -> xr.Dataset:
        """
        Increase rainfall across the entire simulated dataset by a
        given percentage.

        Args:
            percentage: The percentage increase to apply, e.g. 20 for
                a 20% increase (factor = 1.20). No validation is
                currently applied — a negative percentage would
                decrease rainfall instead, and a percentage below
                -100 would produce negative rainfall values, which
                are physically invalid. See open note below.

        Returns:
            xr.Dataset: The updated simulated_dataset (same object,
                mutated in place — not a new copy).
        """
        factor = 1 + (percentage / 100)

        self.simulated_dataset["RAINFALL"] = (
            self.simulated_dataset["RAINFALL"] * factor
        )

        return self.simulated_dataset

    def rainfall_decrease(self, percentage: float) -> xr.Dataset:
        """
        Decrease rainfall across the entire simulated dataset by a
        given percentage.

        Args:
            percentage: The percentage decrease to apply, e.g. 20 for
                a 20% decrease (factor = 0.80). No validation is
                currently applied — a percentage above 100 produces a
                negative factor, which would flip rainfall sign
                rather than floor it at 0. See open note below.

        Returns:
            xr.Dataset: The updated simulated_dataset (same object,
                mutated in place).
        """
        factor = 1 - (percentage / 100)

        self.simulated_dataset["RAINFALL"] = (
            self.simulated_dataset["RAINFALL"] * factor
        )

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

        Note:
            Uses the hardcoded dimension name "TIME" rather than an
            auto-detected time coordinate name. This is inconsistent
            with SpatialEngine and TemporalEngine, both of which
            detect their coordinate names dynamically (see
            SpatialEngine._find_coordinate) specifically because IMD
            datasets vary in whether this dimension is called "TIME"
            or "time". Flagged in the Day 1 code review as
            "hardcoded coordinate names in SimulationEngine" — this
            method (and heavy_rainfall(), below) are exactly that
            issue. Planned fix: accept a time_name (e.g. from
            TemporalEngine) at construction time instead of assuming
            "TIME".
        """
        self.simulated_dataset["RAINFALL"].loc[
            dict(TIME=slice(start_date, end_date))
        ] = 0

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
                double rainfall. No validation is applied — a
                negative multiplier would produce negative rainfall.

        Returns:
            xr.Dataset: The updated simulated_dataset.

        Note:
            Same hardcoded "TIME" issue as dry_spell() — see that
            method's docstring for details.
        """
        self.simulated_dataset["RAINFALL"].loc[
            dict(TIME=slice(start_date, end_date))
        ] *= multiplier

        return self.simulated_dataset

    def get_dataset(self) -> xr.Dataset:
        """
        Return the current simulated dataset, reflecting all
        simulations applied so far.

        Returns:
            xr.Dataset: simulated_dataset as it currently stands.
        """
        return self.simulated_dataset
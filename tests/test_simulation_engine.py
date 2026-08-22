"""
tests/test_simulation_engine.py

Tests for twin/simulation_engine.py — rainfall "what-if" simulations,
input validation (SimulationError), and the deep-copy isolation
guarantee that keeps original_dataset untouched by any simulation.
"""

import numpy as np
import pytest

from src.exceptions import DatasetSchemaError, SimulationError
from src.twin.simulation_engine import SimulationEngine


class TestSimulationEngineConstruction:
    """Tests for __init__, including the Day 4 time_name parameter."""

    def test_auto_detects_time_name_when_not_provided(self, synthetic_rainfall_dataset):
        """
        If time_name isn't passed, SimulationEngine should
        auto-detect it via find_coordinate(), same as SpatialEngine
        and TemporalEngine.
        """
        engine = SimulationEngine(synthetic_rainfall_dataset)

        assert engine.time_name == "TIME"

    def test_auto_detects_lowercase_time_name(self, synthetic_dataset_lowercase_coords):
        """Auto-detection should also work on the lowercase-named fixture."""
        engine = SimulationEngine(synthetic_dataset_lowercase_coords)

        assert engine.time_name == "time"

    def test_uses_explicitly_provided_time_name(self, synthetic_rainfall_dataset):
        """
        When time_name is explicitly provided (as DigitalTwin now
        does, passing TemporalEngine's detected name), it should be
        used directly rather than re-detected.
        """
        engine = SimulationEngine(synthetic_rainfall_dataset, time_name="TIME")

        assert engine.time_name == "TIME"

    def test_simulated_dataset_is_separate_object_from_original(
        self, synthetic_rainfall_dataset
    ):
        """
        simulated_dataset should be a deep copy, not the same object
        as original_dataset — this is the foundation of the whole
        class's "never touch real data" guarantee.
        """
        engine = SimulationEngine(synthetic_rainfall_dataset)

        assert engine.simulated_dataset is not engine.original_dataset


class TestDeepCopyIsolation:
    """
    Tests that simulations never mutate the caller's original
    dataset — the core guarantee this class exists to provide.
    """

    def test_rainfall_increase_does_not_mutate_original_dataset(
        self, synthetic_rainfall_dataset
    ):
        """
        After simulate_increase(), the ORIGINAL dataset object passed
        in at construction should still have its untouched values —
        only simulated_dataset should reflect the change.
        """
        original_values_before = synthetic_rainfall_dataset["RAINFALL"].values.copy()

        engine = SimulationEngine(synthetic_rainfall_dataset)
        engine.rainfall_increase(50)

        np.testing.assert_array_equal(
            synthetic_rainfall_dataset["RAINFALL"].values, original_values_before
        )

    def test_dry_spell_does_not_mutate_original_dataset(self, synthetic_rainfall_dataset):
        """Same guarantee, for dry_spell()."""
        original_values_before = synthetic_rainfall_dataset["RAINFALL"].values.copy()

        engine = SimulationEngine(synthetic_rainfall_dataset)
        engine.dry_spell("2025-07-01", "2025-07-03")

        np.testing.assert_array_equal(
            synthetic_rainfall_dataset["RAINFALL"].values, original_values_before
        )


class TestReset:
    """Tests for reset()."""

    def test_reset_discards_simulated_changes(self, synthetic_rainfall_dataset):
        """
        After a simulation and then reset(), simulated_dataset should
        match the original values again.
        """
        original_values = synthetic_rainfall_dataset["RAINFALL"].values.copy()

        engine = SimulationEngine(synthetic_rainfall_dataset)
        engine.rainfall_increase(50)
        engine.reset()

        np.testing.assert_array_almost_equal(
            engine.simulated_dataset["RAINFALL"].values, original_values
        )


class TestRainfallIncrease:
    """
    Tests for rainfall_increase(), using the fixture's deterministic
    formula: rainfall[0, 0, 0] = 0.0 (a degenerate case), so
    rainfall[1, 0, 0] = 10.0 is used instead as a non-zero base value.
    """

    def test_increase_applies_correct_factor(self, synthetic_rainfall_dataset):
        """
        A 50% increase on a base value of 10.0 (day_idx=1, lat_idx=0,
        lon_idx=0) should produce 15.0 (factor = 1.5).
        """
        engine = SimulationEngine(synthetic_rainfall_dataset)

        result = engine.rainfall_increase(50)

        assert float(result["RAINFALL"].values[1, 0, 0]) == pytest.approx(15.0)

    def test_zero_percent_increase_leaves_values_unchanged(
        self, synthetic_rainfall_dataset
    ):
        """A 0% increase should leave rainfall values exactly as they were."""
        original_values = synthetic_rainfall_dataset["RAINFALL"].values.copy()

        engine = SimulationEngine(synthetic_rainfall_dataset)
        result = engine.rainfall_increase(0)

        np.testing.assert_array_almost_equal(result["RAINFALL"].values, original_values)

    def test_raises_simulation_error_for_negative_percentage(
        self, synthetic_rainfall_dataset
    ):
        """A negative increase percentage should raise SimulationError."""
        engine = SimulationEngine(synthetic_rainfall_dataset)

        with pytest.raises(SimulationError):
            engine.rainfall_increase(-10)


class TestRainfallDecrease:
    """Tests for rainfall_decrease()."""

    def test_decrease_applies_correct_factor(self, synthetic_rainfall_dataset):
        """
        A 30% decrease on a base value of 10.0 (day_idx=1, lat_idx=0,
        lon_idx=0) should produce 7.0 (factor = 0.7).
        """
        engine = SimulationEngine(synthetic_rainfall_dataset)

        result = engine.rainfall_decrease(30)

        assert float(result["RAINFALL"].values[1, 0, 0]) == pytest.approx(7.0)

    def test_100_percent_decrease_is_allowed_and_zeros_rainfall(
        self, synthetic_rainfall_dataset
    ):
        """
        Exactly 100% should be accepted (boundary value) and produce
        zero rainfall everywhere — this is the maximum valid decrease.
        """
        engine = SimulationEngine(synthetic_rainfall_dataset)

        result = engine.rainfall_decrease(100)

        assert float(result["RAINFALL"].values[1, 0, 0]) == pytest.approx(0.0)

    def test_raises_simulation_error_for_negative_percentage(
        self, synthetic_rainfall_dataset
    ):
        """A negative decrease percentage should raise SimulationError."""
        engine = SimulationEngine(synthetic_rainfall_dataset)

        with pytest.raises(SimulationError):
            engine.rainfall_decrease(-10)

    def test_raises_simulation_error_when_decrease_exceeds_100(
        self, synthetic_rainfall_dataset
    ):
        """
        A decrease above 100% would produce negative rainfall —
        physically invalid — and should raise SimulationError.
        This is the exact validation gap flagged since Day 2.
        """
        engine = SimulationEngine(synthetic_rainfall_dataset)

        with pytest.raises(SimulationError):
            engine.rainfall_decrease(150)


class TestDrySpell:
    """Tests for dry_spell()."""

    def test_sets_rainfall_to_zero_within_range(self, synthetic_rainfall_dataset):
        """
        A dry spell over 2025-07-01 to 2025-07-02 (day_idx 0-1)
        should zero out rainfall for those days at every grid point.
        """
        engine = SimulationEngine(synthetic_rainfall_dataset)

        result = engine.dry_spell("2025-07-01", "2025-07-02")

        affected = result["RAINFALL"].sel(TIME=slice("2025-07-01", "2025-07-02"))
        assert float(affected.values.max()) == pytest.approx(0.0)

    def test_leaves_rainfall_outside_range_unchanged(self, synthetic_rainfall_dataset):
        """
        Days outside the dry spell range should retain their original
        values — e.g. day_idx=4 (2025-07-05) at lat_idx=0, lon_idx=0
        should still be 40.0.
        """
        engine = SimulationEngine(synthetic_rainfall_dataset)

        result = engine.dry_spell("2025-07-01", "2025-07-02")

        untouched_value = float(result["RAINFALL"].values[4, 0, 0])
        assert untouched_value == pytest.approx(40.0)

    def test_raises_dataset_schema_error_when_rainfall_missing(self):
        """
        A dataset without a RAINFALL variable should raise
        DatasetSchemaError, not a raw KeyError, when dry_spell() is
        attempted.
        """
        import xarray as xr

        dataset = xr.Dataset(
            coords={"TIME": np.array(["2025-07-01"], dtype="datetime64[ns]")}
        )
        engine = SimulationEngine(dataset, time_name="TIME")

        with pytest.raises(DatasetSchemaError):
            engine.dry_spell("2025-07-01", "2025-07-01")


class TestHeavyRainfall:
    """Tests for heavy_rainfall()."""

    def test_applies_multiplier_within_range(self, synthetic_rainfall_dataset):
        """
        Doubling rainfall (multiplier=2.0) over 2025-07-02 (day_idx=1)
        at lat_idx=0, lon_idx=0 should turn 10.0 into 20.0.
        """
        engine = SimulationEngine(synthetic_rainfall_dataset)

        result = engine.heavy_rainfall("2025-07-02", "2025-07-02", multiplier=2.0)

        assert float(result["RAINFALL"].values[1, 0, 0]) == pytest.approx(20.0)

    def test_leaves_rainfall_outside_range_unchanged(self, synthetic_rainfall_dataset):
        """Days outside the range should be unaffected by the multiplier."""
        engine = SimulationEngine(synthetic_rainfall_dataset)

        result = engine.heavy_rainfall("2025-07-02", "2025-07-02", multiplier=2.0)

        untouched_value = float(result["RAINFALL"].values[0, 0, 0])
        assert untouched_value == pytest.approx(0.0)

    def test_raises_simulation_error_for_negative_multiplier(
        self, synthetic_rainfall_dataset
    ):
        """A negative multiplier should raise SimulationError."""
        engine = SimulationEngine(synthetic_rainfall_dataset)

        with pytest.raises(SimulationError):
            engine.heavy_rainfall("2025-07-01", "2025-07-02", multiplier=-1.0)


class TestGetDataset:
    """Tests for get_dataset()."""

    def test_reflects_applied_simulations(self, synthetic_rainfall_dataset):
        """get_dataset() should return the same object simulate methods mutated."""
        engine = SimulationEngine(synthetic_rainfall_dataset)
        engine.rainfall_increase(50)

        result = engine.get_dataset()

        assert result is engine.simulated_dataset
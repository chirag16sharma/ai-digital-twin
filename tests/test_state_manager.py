"""
tests/test_state_manager.py

Tests for twin/state_manager.py — in-memory state storage, including
the specific behavioral quirks documented since Day 2: update_state()
treats None as "leave unchanged" (not "clear this field"), and
clear_state() does NOT stamp last_updated (unlike every other
mutating method).
"""

from datetime import datetime

from src.twin.state_manager import StateManager


class TestInitialState:
    """Tests for StateManager.__init__."""

    def test_all_fields_start_as_none(self):
        """A freshly constructed StateManager should have every field set to None."""
        manager = StateManager()

        assert manager.state == {
            "latitude": None,
            "longitude": None,
            "date": None,
            "rainfall": None,
            "last_updated": None,
        }


class TestIndividualSetters:
    """Tests for set_location(), set_date(), set_rainfall()."""

    def test_set_location_updates_both_fields(self):
        """set_location() should update latitude and longitude together."""
        manager = StateManager()

        manager.set_location(19.076, 72.8777)

        assert manager.state["latitude"] == 19.076
        assert manager.state["longitude"] == 72.8777

    def test_set_location_stamps_last_updated(self):
        """set_location() should update last_updated as a side effect."""
        manager = StateManager()

        assert manager.state["last_updated"] is None

        manager.set_location(19.076, 72.8777)

        assert isinstance(manager.state["last_updated"], datetime)

    def test_set_date_updates_date_field(self):
        """set_date() should update only the date field."""
        manager = StateManager()

        manager.set_date("2025-07-15")

        assert manager.state["date"] == "2025-07-15"

    def test_set_rainfall_accepts_plain_float(self):
        """set_rainfall() should accept and store a plain float."""
        manager = StateManager()

        manager.set_rainfall(42.5)

        assert manager.state["rainfall"] == 42.5


class TestUpdateState:
    """
    Tests for update_state(), including its documented "None means
    leave unchanged" behavior.
    """

    def test_updates_only_provided_fields(self):
        """
        Calling update_state() with only latitude/longitude provided
        should leave date and rainfall untouched (still None).
        """
        manager = StateManager()

        manager.update_state(latitude=10.0, longitude=75.0)

        assert manager.state["latitude"] == 10.0
        assert manager.state["longitude"] == 75.0
        assert manager.state["date"] is None
        assert manager.state["rainfall"] is None

    def test_updates_all_fields_when_all_provided(self):
        """Calling update_state() with all four values should set all four."""
        manager = StateManager()

        manager.update_state(
            latitude=10.0, longitude=75.0, date="2025-07-15", rainfall=20.0
        )

        assert manager.state["latitude"] == 10.0
        assert manager.state["longitude"] == 75.0
        assert manager.state["date"] == "2025-07-15"
        assert manager.state["rainfall"] == 20.0

    def test_none_argument_leaves_existing_value_unchanged(self):
        """
        This is the key documented quirk: calling update_state() a
        second time with date=None should NOT clear a previously-set
        date — None means "don't touch this field," not "clear it."
        """
        manager = StateManager()
        manager.update_state(date="2025-07-15")

        manager.update_state(latitude=10.0, date=None)

        assert manager.state["date"] == "2025-07-15"
        assert manager.state["latitude"] == 10.0

    def test_update_state_stamps_last_updated(self):
        """update_state() should update last_updated as a side effect."""
        manager = StateManager()

        manager.update_state(latitude=10.0)

        assert isinstance(manager.state["last_updated"], datetime)


class TestGetState:
    """Tests for get_state()."""

    def test_returns_current_state_dict(self):
        """get_state() should return a dict matching the current state."""
        manager = StateManager()
        manager.update_state(latitude=10.0, longitude=75.0)

        result = manager.get_state()

        assert result["latitude"] == 10.0
        assert result["longitude"] == 75.0

    def test_returns_live_reference_not_a_copy(self):
        """
        Documented behavior: get_state() returns the SAME dict object
        as self.state, not a copy. This test locks that behavior in
        explicitly, since it's easy to accidentally change (e.g. by
        adding a defensive .copy()) without realizing it's a
        behavioral change some caller might depend on.
        """
        manager = StateManager()

        result = manager.get_state()

        assert result is manager.state

class TestClearState:
    """Tests for clear_state()."""

    def test_resets_data_fields_to_none(self):
        """clear_state() should reset latitude/longitude/date/rainfall to None."""
        manager = StateManager()
        manager.update_state(
            latitude=10.0, longitude=75.0, date="2025-07-15", rainfall=20.0
        )

        manager.clear_state()

        assert manager.state["latitude"] is None
        assert manager.state["longitude"] is None
        assert manager.state["date"] is None
        assert manager.state["rainfall"] is None

    def test_clear_state_stamps_last_updated(self):
        """
        Changed on Day 5: clear_state() now stamps last_updated,
        consistent with every other mutating method — clearing the
        state is itself a modification worth timestamping.
        """
        manager = StateManager()
        manager.update_state(latitude=10.0)

        manager.clear_state()

        assert isinstance(manager.state["last_updated"], datetime)

class TestDisplayState:
    """Tests for display_state() — console output only, minimal testing needed."""

    def test_does_not_raise_on_empty_state(self, capsys):
        """
        display_state() should run without error even on a freshly
        constructed (all-None) state, and should print something.
        """
        manager = StateManager()

        manager.display_state()

        captured = capsys.readouterr()
        assert "CURRENT DIGITAL TWIN STATE" in captured.out
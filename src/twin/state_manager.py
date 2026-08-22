"""
twin/state_manager.py

Responsible for holding the Digital Twin's current in-memory state:
the last-queried location, date, and rainfall value, plus a timestamp
of when the state was last modified. This is intentionally the
simplest component in the twin/ package — pure data storage with no
domain logic (no coordinate math, no dataset access).
"""

from datetime import datetime
from typing import Any, Optional, TypedDict

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DigitalTwinState(TypedDict):
    """
    Shape of the Digital Twin's state dictionary.

    Defined as a TypedDict (rather than a plain Dict[str, Any]) so
    that static analysis and IDE autocomplete know exactly which keys
    exist and what type each one holds — a plain Dict[str, Any] would
    type-check `state["latitude"]` and `state["banana"]` identically,
    which defeats the purpose of typing a fixed-shape structure like
    this one.

    Attributes:
        latitude: The last-queried latitude, or None if unset.
        longitude: The last-queried longitude, or None if unset.
        date: The last-queried date, or None if unset.
        rainfall: The last-retrieved rainfall value, or None if
            unset. Typed as Any because callers may pass a raw float,
            a numpy scalar, or an xr.DataArray depending on where the
            value came from (e.g. SpatialEngine.rainfall_at() returns
            a DataArray) — see the note on set_rainfall() below.
        last_updated: Timestamp of the most recent state change, or
            None if the state has never been updated.
    """
    latitude: Optional[float]
    longitude: Optional[float]
    date: Optional[str]
    rainfall: Optional[Any]
    last_updated: Optional[datetime]


class StateManager:
    """
    Stores and manages the current state of the AI Digital Twin.

    Single Responsibility:
        In-memory state storage only. This class has no awareness of
        datasets, coordinates, or rainfall calculations — it just
        remembers the most recent values it was told about. All
        domain logic (finding rainfall, resolving coordinates) lives
        upstream in SpatialEngine/TemporalEngine/QueryEngine.

    Attributes:
        state (DigitalTwinState): The current state dictionary.
    """

    def __init__(self) -> None:
        """
        Initialize an empty state, with every field set to None.
        """
        self.state: DigitalTwinState = {
            "latitude": None,
            "longitude": None,
            "date": None,
            "rainfall": None,
            "last_updated": None
        }

        logger.info("StateManager initialized with empty state")

    def set_location(self, latitude: float, longitude: float) -> None:
        """
        Update the current location in the state.

        Args:
            latitude: The latitude to store.
            longitude: The longitude to store.

        Returns:
            None. Also updates last_updated as a side effect.
        """
        self.state["latitude"] = latitude
        self.state["longitude"] = longitude

        self._update_timestamp()

        logger.debug(f"State location updated: ({latitude}, {longitude})")

    def set_date(self, date: str) -> None:
        """
        Update the current date in the state.

        Args:
            date: The date to store, e.g. "2025-07-15".

        Returns:
            None. Also updates last_updated as a side effect.
        """
        self.state["date"] = date

        self._update_timestamp()

        logger.debug(f"State date updated: {date}")

    def set_rainfall(self, rainfall: Any) -> None:
        """
        Store the most recently retrieved rainfall value.

        Args:
            rainfall: The rainfall value to store. Accepts Any
                because callers pass different types depending on
                source — a raw float, a numpy scalar, or an
                xr.DataArray (e.g. straight from
                SpatialEngine.rainfall_at()). No conversion is
                applied here.

        Returns:
            None. Also updates last_updated as a side effect.
        """
        self.state["rainfall"] = rainfall

        self._update_timestamp()

        logger.debug(f"State rainfall updated: {rainfall}")

    def update_state(
        self,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        date: Optional[str] = None,
        rainfall: Optional[Any] = None
    ) -> None:
        """
        Update one or more state fields at once.

        Only fields explicitly passed (i.e. not None) are updated;
        omitted fields keep their current value. Note this means
        there is no way to explicitly reset a single field back to
        None via this method — use clear_state() to reset everything.

        Args:
            latitude: New latitude, or None to leave unchanged.
            longitude: New longitude, or None to leave unchanged.
            date: New date, or None to leave unchanged.
            rainfall: New rainfall value, or None to leave unchanged.

        Returns:
            None. Also updates last_updated as a side effect.
        """
        updated_fields = []

        if latitude is not None:
            self.state["latitude"] = latitude
            updated_fields.append("latitude")

        if longitude is not None:
            self.state["longitude"] = longitude
            updated_fields.append("longitude")

        if date is not None:
            self.state["date"] = date
            updated_fields.append("date")

        if rainfall is not None:
            self.state["rainfall"] = rainfall
            updated_fields.append("rainfall")

        self._update_timestamp()

        logger.info(f"State updated: fields changed = {updated_fields}")

    def get_state(self) -> DigitalTwinState:
        """
        Return the current state.

        Returns:
            DigitalTwinState: The full state dictionary. Note this
                returns a reference to the internal dict, not a copy
                — callers should treat it as read-only, since
                mutating the returned dict directly would bypass
                _update_timestamp() and silently desync last_updated.
        """
        return self.state

    def clear_state(self) -> None:
        """
        Reset the Digital Twin's state back to its initial (all None)
        values, except last_updated, which is still stamped with the
        current time — clearing the state IS a modification, and
        this keeps last_updated consistent with every other mutating
        method (previously it was reset to None as well; changed on
        Day 5 for consistency).

        Returns:
            None.
        """
        self.state = {
            "latitude": None,
            "longitude": None,
            "date": None,
            "rainfall": None,
            "last_updated": None
        }

        self._update_timestamp()

        logger.info("State cleared — reset to initial (all None) values)")

    def display_state(self) -> None:
        """
        Print the current state to the console.

        Returns:
            None. Console output only.
        """
        print("=" * 50)
        print("CURRENT DIGITAL TWIN STATE")
        print("=" * 50)

        for key, value in self.state.items():
            print(f"{key:<15}: {value}")

    def _update_timestamp(self) -> None:
        """
        Stamp last_updated with the current time.

        Returns:
            None.
        """
        self.state["last_updated"] = datetime.now()
from datetime import datetime


class StateManager:
    """
    State Manager for the AI Digital Twin.

    Responsibilities
    ----------------
    - Store the current state of the Digital Twin.
    - Update the state.
    - Retrieve the state.
    - Reset the state.
    """

    def __init__(self):
        """
        Initialize an empty state.
        """

        self.state = {
            "latitude": None,
            "longitude": None,
            "date": None,
            "rainfall": None,
            "last_updated": None
        }

    def set_location(self, latitude, longitude):
        """
        Update the current location.
        """

        self.state["latitude"] = latitude
        self.state["longitude"] = longitude

        self._update_timestamp()

    def set_date(self, date):
        """
        Update the current date.
        """

        self.state["date"] = date

        self._update_timestamp()

    def set_rainfall(self, rainfall):
        """
        Store rainfall information.
        """

        self.state["rainfall"] = rainfall

        self._update_timestamp()

    def update_state(
        self,
        latitude=None,
        longitude=None,
        date=None,
        rainfall=None
    ):
        """
        Update multiple values at once.
        """

        if latitude is not None:
            self.state["latitude"] = latitude

        if longitude is not None:
            self.state["longitude"] = longitude

        if date is not None:
            self.state["date"] = date

        if rainfall is not None:
            self.state["rainfall"] = rainfall

        self._update_timestamp()

    def get_state(self):
        """
        Return the current state.
        """

        return self.state

    def clear_state(self):
        """
        Reset the Digital Twin.
        """

        self.state = {
            "latitude": None,
            "longitude": None,
            "date": None,
            "rainfall": None,
            "last_updated": None
        }

    def display_state(self):
        """
        Display the current state.
        """

        print("=" * 50)
        print("CURRENT DIGITAL TWIN STATE")
        print("=" * 50)

        for key, value in self.state.items():
            print(f"{key:<15}: {value}")

    def _update_timestamp(self):
        """
        Update the last modification time.
        """

        self.state["last_updated"] = datetime.now()
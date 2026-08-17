"""
main.py

Entry point for the AI Digital Twin project. Constructs the
DigitalTwin, runs a small demonstration of its core capabilities
(querying rainfall, checking state, running a simulation), and
serves as a manual smoke test that the whole system wires together
correctly end to end.
"""

from pathlib import Path

from src.twin.digital_twin import DigitalTwin


# TEMPORARY: hardcoded dataset path. This will move into
# config/settings.py on Day 4 (Configuration System), at which point
# this constant disappears and DigitalTwin will source its default
# path from config instead of requiring the caller to know it.
DATASET_PATH: Path = Path("data/processed/rainfall_ai_ready.nc")


def main() -> None:
    """
    Run a demonstration of the AI Digital Twin's core capabilities.

    This function is intentionally simple: it exists to prove the
    system works end to end (load -> query -> simulate -> reset), not
    to be a full CLI. A proper CLI/API interface is planned for a
    later week (FastAPI, per the project roadmap).

    Raises:
        FileNotFoundError: If DATASET_PATH does not point to an
            existing file (propagated from DigitalTwin -> IMDLoader).
        ValueError: If the dataset is missing a recognizable
            latitude, longitude, or time coordinate (propagated from
            DigitalTwin -> SpatialEngine/TemporalEngine).
    """
    print("=" * 60)
    print("AI DIGITAL TWIN PROJECT STARTED")
    print("=" * 60)

    twin = DigitalTwin(DATASET_PATH)

    print(f"\nDataset loaded: {DATASET_PATH}")
    print(f"Available coordinate ranges:")
    twin.spatial.available_coordinates()

    # --- Example query ---
    print("\n" + "-" * 60)
    print("Example rainfall query")
    print("-" * 60)

    result = twin.rainfall(
        latitude=19.0760,
        longitude=72.8777,
        date=str(twin.temporal.first_date())[:10]
    )
    print(result)

    print("\nCurrent Digital Twin state:")
    twin.state.display_state()

    # --- Example simulation ---
    print("\n" + "-" * 60)
    print("Example simulation: 20% rainfall increase")
    print("-" * 60)

    twin.simulate_increase(percentage=20)
    print("Simulation applied to simulated_dataset.")

    twin.reset_simulation()
    print("Simulation reset — simulated_dataset restored to original.")

    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
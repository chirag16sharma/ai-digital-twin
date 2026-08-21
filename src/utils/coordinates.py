"""
src/utils/coordinates.py

Shared coordinate-detection logic for the AI Digital Twin. IMD
NetCDF datasets are inconsistent about coordinate naming/casing
across versions (e.g. "LATITUDE" vs "latitude" vs "lat"), so both
SpatialEngine and TemporalEngine need to auto-detect which naming
convention a given dataset actually uses. This was previously
duplicated as a private method in both classes (flagged in the Day 1
code review); it now lives here as the single implementation both
classes call.
"""

import xarray as xr

from src.exceptions import CoordinateNotFoundError
from src.utils.logger import get_logger

logger = get_logger(__name__)


def find_coordinate(dataset: xr.Dataset, possible_names: list[str]) -> str:
    """
    Find which of several possible coordinate names is actually
    present in a dataset.

    Args:
        dataset: The xarray Dataset to search.
        possible_names: Candidate coordinate names to check, in
            priority order. The first match found is returned.
            Callers typically pass one of the *_ALIASES constants
            from config/settings.py (e.g. LATITUDE_ALIASES) rather
            than writing out a literal list.

    Returns:
        str: The matching coordinate name, exactly as it appears in
            dataset.coords.

    Raises:
        CoordinateNotFoundError: If none of possible_names exist in
            the dataset's coordinates.
    """
    for name in possible_names:
        if name in dataset.coords:
            logger.debug(f"Coordinate match found: {name!r}")
            return name

    logger.error(
        f"None of the coordinate names {possible_names} found "
        f"in dataset coordinates: {list(dataset.coords)}"
    )
    raise CoordinateNotFoundError(
        f"None of the coordinate names {possible_names} found."
    )
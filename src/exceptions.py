"""
src/exceptions.py

Defines the custom exception hierarchy for the AI Digital Twin
project. Every domain-specific error raised anywhere in src/ should
be one of these, rather than a bare built-in exception (ValueError,
KeyError, FileNotFoundError). This lets calling code choose between
catching narrowly (e.g. except InvalidDateError) or broadly
(except DigitalTwinError) depending on what it needs to handle.
"""


class DigitalTwinError(Exception):
    """
    Base exception for all errors raised by the AI Digital Twin
    system.

    Catching this catches any domain-specific error from this
    project, without also catching unrelated bugs that happen to
    raise a built-in exception like KeyError or ValueError for
    completely different reasons.
    """


class DatasetNotFoundError(DigitalTwinError):
    """
    Raised when a dataset file does not exist at the expected path.

    Replaces the generic FileNotFoundError previously raised by
    IMDLoader.load().
    """
class DatasetSaveError(DigitalTwinError):
    """
    Raised when a dataset fails to save to disk — e.g. the parent
    directory of the target path does not exist.

    Distinct from DatasetNotFoundError, which is specifically about
    loading a dataset that doesn't exist yet, not writing one out.
    """

class DatasetSchemaError(DigitalTwinError):
    """
    Raised when a dataset is missing an expected variable or
    dimension (e.g. no "RAINFALL" data variable).

    Replaces the generic KeyError previously raised implicitly by
    dataset["RAINFALL"] lookups across DataExplorer, DataCleaner,
    FeatureEngineer, SpatialEngine, TemporalEngine, and QueryEngine.
    """


class CoordinateNotFoundError(DatasetSchemaError):
    """
    Raised when none of the expected latitude, longitude, or time
    coordinate name variants (e.g. "LATITUDE" / "latitude" / "lat")
    are present in a dataset.

    Replaces the generic ValueError previously raised by
    SpatialEngine._find_coordinate() and
    TemporalEngine._find_coordinate().

    This is a subclass of DatasetSchemaError (not a sibling) because
    it IS a schema problem — a missing coordinate is a specific case
    of "the dataset doesn't have what we expected."
    """


class InvalidCoordinateError(DigitalTwinError):
    """
    Raised when a requested latitude or longitude is outside the
    dataset's actual coverage area.

    Distinct from CoordinateNotFoundError: this means the dataset's
    schema is fine (it has a valid latitude/longitude coordinate),
    but the specific value requested by the caller falls outside the
    range that coordinate actually covers.
    """


class InvalidDateError(DigitalTwinError):
    """
    Raised when a requested date is outside the dataset's available
    date range, or is not a validly formatted date string.
    """


class PipelineError(DigitalTwinError):
    """
    Raised when a stage of DataPipeline fails in a way that should
    halt the pipeline, wrapping the underlying cause with context
    about which stage failed.
    """


class SimulationError(DigitalTwinError):
    """
    Raised when simulation parameters are invalid — e.g. a rainfall
    decrease percentage above 100 (which would produce negative
    rainfall), or a negative multiplier passed to heavy_rainfall().
    """
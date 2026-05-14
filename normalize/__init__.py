from .core import ingest


class NormalizeError(Exception):
    pass


__all__ = ["ingest", "NormalizeError"]

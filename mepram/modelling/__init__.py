"""Three-level modelling package."""

__all__ = ["main"]


def main(*args, **kwargs):
    from .cli import main as _main
    return _main(*args, **kwargs)

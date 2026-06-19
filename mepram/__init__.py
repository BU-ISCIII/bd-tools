"""Top-level mepram package."""

__all__ = ["modelling_main"]


def modelling_main(*args, **kwargs):
    from .modelling import main as _modelling_main
    return _modelling_main(*args, **kwargs)

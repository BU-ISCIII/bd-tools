#!/usr/bin/env python3
"""Generic staged modelling entrypoint.

Examples
--------
python modelling.py --project sepsis_three_level ...
python modelling.py --project bacthecom_mortality ...
"""

from staged.cli import main


if __name__ == "__main__":
    main()

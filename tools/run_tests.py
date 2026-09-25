"""Run the offline pressure regression suite after installation."""

from pathlib import Path
import sys
import unittest

from pyib.cli import _prepare_imports


if __name__ == "__main__":
    _prepare_imports()
    suite = unittest.defaultTestLoader.discover(
        str(Path(__file__).resolve().parents[1] / "tests"),
        pattern="test_ib_*.py",
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)

"""Console wrappers for preserved modules using legacy local imports.

Each console command starts a separate process. The numerical backend is
selected once per process; this module does not support switching devices
after importing a solver or embedding multiple backend instances.
"""

from importlib import import_module
from pathlib import Path
import sys


def _prepare_imports():
    package = Path(__file__).resolve().parent
    for directory in (package, package / "solver"):
        location = str(directory)
        if location not in sys.path:
            sys.path.insert(0, location)


def run():
    _prepare_imports()
    return import_module("pyib.solver.cfdagent_ib").main()


def postprocess():
    _prepare_imports()
    return import_module("pyib.solver.ib_postprocess").main()


def validate():
    _prepare_imports()
    return import_module("pyib.solver.validate_ib_result").main()

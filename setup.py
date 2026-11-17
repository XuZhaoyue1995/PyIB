"""Opt-in OpenMP build; ordinary installs remain pure Python.

Use PYIB_BUILD_OPENMP=1 with a Linux GCC-compatible C compiler. Compilation is
an installation step only; the solver never compiles or downloads code.
"""
import os
import sys
from setuptools import Extension, setup

enabled = os.environ.get("PYIB_BUILD_OPENMP", "0")
if enabled not in {"0", "1"}:
    raise RuntimeError("PYIB_BUILD_OPENMP must be 0 or 1")
extensions = []
if enabled == "1":
    if not sys.platform.startswith("linux"):
        raise RuntimeError("The optional OpenMP build currently supports Linux GCC-compatible compilers only")
    flags = ["-O2", "-std=c11", "-fopenmp", "-fno-fast-math",
             "-fno-associative-math", "-ffp-contract=off", "-fno-tree-vectorize",
             "-frounding-math"]
    extensions.append(Extension(
        "pyib.solver._csr_openmp", ["src/pyib/solver/_csr_openmp.c"],
        extra_compile_args=flags, extra_link_args=["-fopenmp"], libraries=["dl", "m"],
        define_macros=[("PYIB_BUILD_FLAGS", '"' + " ".join(flags) + '"')],
    ))
setup(ext_modules=extensions)

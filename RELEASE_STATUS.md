# PyIB research prerelease: 0.1.0rc1

PyIB `0.1.0rc1` is a research prerelease. This version identifies the packaged
numerical snapshot and its documented checks. It is not a package-registry
release or a statement of completed scientific acceptance.

The project repository is
`https://github.com/XuZhaoyue1995/gpu-ib-code-paper-2026-08-snapshot`.
The existing `paper/IB_GPU_Solver_Paper_CN_JCP.pdf` manuscript is retained
unchanged when replacing the repository's code with this PyIB prerelease.

**Sole software author: Zhaoyue Xu.** `AUTHORS.md`, `CITATION.cff` and package
metadata record that author only. Existing papers and retained manuscript
drafts keep their original authorship; software metadata do not rewrite them.

The author selected **PyIB — A Python Immersed Boundary Solver for CPUs and
GPUs**. The independent solver can be called by CFDAgent. Its distribution and
namespace are `pyib`, with `pyib-run`, `pyib-postprocess` and `pyib-validate`
console commands. The earlier `cfdagent-ib` name was internal and temporary;
historical checks retain their actual tested command names. The preserved
internal `solver/cfdagent_ib.py` filename and its legacy error-message prefix
remain unchanged for source provenance.

## Licensing and ongoing work

The software license is pending. No open-source license is granted by this
prerelease.

Scientific validation continues, including review of the ongoing D32/D40
calculations. Manufactured-field tests and coarse smoke runs demonstrate the
limited properties described in the test reports; they are not a replacement
for physical accuracy and convergence evidence.

The inherited local paths listed in `docs/DEPENDENCIES.md` are in inactive
diagnostic blocks; standard packaged entry points do not use them. Numerical
files remain byte-for-byte identical to the original snapshot. Any subsequent
cleanup should be reviewed separately and update the provenance manifest.

The package does not contain cluster credentials, host addresses, Fortran
source/binaries/golden outputs, WUR datasets, production results, or historical
logs. It includes only the explicit source allowlist, existing sphere JSON
examples, the offline manufactured-field tests, and new packaging material.

The bundled JSON examples and recorded packaging checks use sphere cases.
Their scope does not imply that the immersed-boundary method is limited to
closed spheres. The author describes a thin-wall treatment using a one-sided,
locally infinite-thickness interpretation; this packaging update did not
verify a thin-wall adapter. CPU and CUDA backends are retained, while GPU
cross-validation and distributed execution have their own evidence needs.

Old generated build files are retained in the ignored `.packaging-history/`
directory. They are outside the source/build paths used for PyIB and are
excluded from release archives.

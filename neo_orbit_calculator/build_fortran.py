"""Build the real128 Fortran backend against SpiceyPy's CSPICE library."""

from __future__ import annotations

import subprocess
from pathlib import Path

import spiceypy


def build() -> Path:
    package = Path(__file__).resolve().parent
    source = package / "fortran" / "neo_integrator.f90"
    output = package / "fortran" / "libneo_integrator.so"
    cspice = Path(spiceypy.__file__).resolve().parent / "utils" / "libcspice.so"
    command = [
        "gfortran",
        "-O3",
        "-fPIC",
        "-shared",
        "-ffree-line-length-none",
        str(source),
        str(cspice),
        f"-Wl,-rpath,{cspice.parent}",
        "-o",
        str(output),
    ]
    subprocess.run(command, check=True)
    return output


if __name__ == "__main__":
    print(build())

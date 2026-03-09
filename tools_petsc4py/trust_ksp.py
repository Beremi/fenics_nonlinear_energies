import ctypes
import os
from typing import Optional


_LIBPETSC: Optional[ctypes.CDLL] = None
_KSPCG_SET_RADIUS = None


def _get_libpetsc() -> ctypes.CDLL:
    global _LIBPETSC
    if _LIBPETSC is not None:
        return _LIBPETSC

    petsc_dir = os.environ.get("PETSC_DIR", "/usr/local/petsc")
    petsc_arch = os.environ.get("PETSC_ARCH", "")
    search_paths = [
        os.path.join(petsc_dir, petsc_arch, "lib", "libpetsc.so"),
        os.path.join(petsc_dir, "lib", "libpetsc.so"),
        "libpetsc.so",
    ]
    for path in search_paths:
        try:
            _LIBPETSC = ctypes.CDLL(path)
            return _LIBPETSC
        except OSError:
            continue
    raise RuntimeError(
        f"Could not load libpetsc.so. Searched: {search_paths}. "
        "Set PETSC_DIR / PETSC_ARCH environment variables."
    )


def _get_kspcg_set_radius():
    global _KSPCG_SET_RADIUS
    if _KSPCG_SET_RADIUS is not None:
        return _KSPCG_SET_RADIUS

    fn = _get_libpetsc().KSPCGSetRadius
    fn.argtypes = [ctypes.c_void_p, ctypes.c_double]
    fn.restype = ctypes.c_int
    _KSPCG_SET_RADIUS = fn
    return fn


def ksp_cg_set_radius(ksp, radius: float) -> None:
    err = _get_kspcg_set_radius()(
        ctypes.c_void_p(int(ksp.handle)),
        ctypes.c_double(float(radius)),
    )
    if err != 0:
        raise RuntimeError(f"KSPCGSetRadius failed with PETSc error code {err}")

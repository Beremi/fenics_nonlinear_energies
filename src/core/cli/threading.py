"""Shared CPU-thread configuration helpers for JAX/PETSc CLIs."""

from __future__ import annotations

import os


def configure_jax_cpu_threading(nproc: int) -> int:
    """Apply the repo's standard single-device CPU-threading environment."""
    threads = max(1, int(nproc))
    os.environ["XLA_FLAGS"] = (
        "--xla_cpu_multi_thread_eigen=false "
        "--xla_force_host_platform_device_count=1"
    )
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    return threads

#!/usr/bin/env python3
"""
Run GL benchmarks: SNES and Custom Newton, serial + parallel.
Writes results to tmp_work/gl_benchmark_*.txt
"""
import subprocess
import sys
import os

DOCKER_BASE = [
    "docker", "run", "--rm", "-e", "PYTHONUNBUFFERED=1",
    "-v", os.getcwd() + ":/work", "-w", "/work", "fenics_test"
]

runs = [
    # (name, entrypoint_args, script_args)
    ("snes_np1", ["--entrypoint", "python3"], ["/work/GinzburgLandau2D_fenics/solve_GL_snes_newton.py", "--levels", "5", "6", "7", "8", "9", "--json", "/work/tmp_work/gl_snes_np1.json"]),
    ("custom_np1", ["--entrypoint", "python3"], ["/work/GinzburgLandau2D_fenics/solve_GL_custom_jaxversion.py", "--levels", "5", "6", "7", "8", "9", "--quiet", "--json", "/work/tmp_work/gl_custom_np1.json"]),
    ("snes_np4", ["--entrypoint", "mpirun"], ["-n", "4", "python3", "/work/GinzburgLandau2D_fenics/solve_GL_snes_newton.py", "--levels", "5", "6", "7", "8", "9", "--json", "/work/tmp_work/gl_snes_np4.json"]),
    ("custom_np4", ["--entrypoint", "mpirun"], ["-n", "4", "python3", "/work/GinzburgLandau2D_fenics/solve_GL_custom_jaxversion.py", "--levels", "5", "6", "7", "8", "9", "--quiet", "--json", "/work/tmp_work/gl_custom_np4.json"]),
    ("snes_np8", ["--entrypoint", "mpirun"], ["-n", "8", "python3", "/work/GinzburgLandau2D_fenics/solve_GL_snes_newton.py", "--levels", "5", "6", "7", "8", "9", "--json", "/work/tmp_work/gl_snes_np8.json"]),
    ("custom_np8", ["--entrypoint", "mpirun"], ["-n", "8", "python3", "/work/GinzburgLandau2D_fenics/solve_GL_custom_jaxversion.py", "--levels", "5", "6", "7", "8", "9", "--quiet", "--json", "/work/tmp_work/gl_custom_np8.json"]),
    ("snes_np16", ["--entrypoint", "mpirun"], ["-n", "16", "python3", "/work/GinzburgLandau2D_fenics/solve_GL_snes_newton.py", "--levels", "5", "6", "7", "8", "9", "--json", "/work/tmp_work/gl_snes_np16.json"]),
    ("custom_np16", ["--entrypoint", "mpirun"], ["-n", "16", "python3", "/work/GinzburgLandau2D_fenics/solve_GL_custom_jaxversion.py", "--levels", "5", "6", "7", "8", "9", "--quiet", "--json", "/work/tmp_work/gl_custom_np16.json"]),
]

for name, entry_args, script_args in runs:
    cmd = DOCKER_BASE[:2] + entry_args + DOCKER_BASE[2:] + script_args
    out_file = "tmp_work/gl_bench_{}.txt".format(name)
    sys.stdout.write("Running {}...\n".format(name))
    sys.stdout.flush()
    with open(out_file, "w") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=1200)
    sys.stdout.write("  exit code: {}\n".format(result.returncode))
    sys.stdout.flush()

sys.stdout.write("All benchmarks done.\n")

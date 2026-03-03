# Local Build Guide — FEniCSx + PETSc (matching devcontainer)

This guide builds **everything locally** into `local_env/` inside this repository,
matching the `dolfinx/lab:stable` container (DOLFINx 0.10.0, PETSc 3.24.2, Python 3.12).

**Host system**: Arch Linux x86_64, GCC 15, OpenMPI 5.0.10, CMake 4.2

---

## Directory Layout

```
fenics_nonlinear_energies/
└── local_env/
    ├── src/        # Downloaded source tarballs / repos
    ├── prefix/     # --prefix install target for all C/C++ libs
    ├── python/     # Python 3.12 built from source
    └── venv/       # Python venv (activated for daily use)
```

---

## Phase 0 — System Prerequisites (Arch Linux)

These are build-time tools that must be installed system-wide.  
They do NOT leak into `local_env/` — they are just compilers/tools.

```bash
sudo pacman -S --needed \
    base-devel cmake ninja git wget curl pkg-config \
    gcc-fortran openmpi openblas boost pugixml \
    openssl zlib xz tk libffi bzip2 readline ncurses sqlite \
    spdlog fmt flex bison libxt
```

> **Why each group?**
> - `base-devel cmake ninja git wget curl pkg-config` — generic build tools
> - `gcc-fortran` — Fortran compiler (PETSc, MUMPS, ScaLAPACK)
> - `openmpi` — MPI (the container uses MPICH; OpenMPI works equally well)
> - `openblas boost pugixml spdlog fmt` — C/C++ deps used by DOLFINx
> - `openssl zlib xz tk libffi bzip2 readline ncurses sqlite` — needed to compile Python 3.12
> - `flex bison` — needed by ptscotch
> - `libxt` — may be needed by some PETSc configure tests

Optional but recommended:

```bash
sudo pacman -S --needed hdf5-openmpi   # system parallel HDF5 2.0 (saves build time)
sudo pacman -S --needed eigen           # Eigen3 headers (used by Basix & DOLFINx)
```

If `eigen` is not available from pacman, we build it from source below (Phase 3).

---

## Phase 1 — Set Up Shell Environment

```bash
# All paths relative to the repo root
export REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# OR just set it manually:
# export REPO_ROOT=/home/michal/repos/fenics_nonlinear_energies

export LOCAL_ENV="${REPO_ROOT}/local_env"
export LOCAL_SRC="${LOCAL_ENV}/src"
export LOCAL_PREFIX="${LOCAL_ENV}/prefix"
export LOCAL_PYTHON_PREFIX="${LOCAL_ENV}/python"

mkdir -p "${LOCAL_SRC}" "${LOCAL_PREFIX}" "${LOCAL_PYTHON_PREFIX}"
```

---

## Phase 2 — Build Python 3.12 from Source

The system Python is 3.14 which is too new for the FEniCSx 0.10.x ecosystem.
We build Python 3.12 into a local prefix.

```bash
cd "${LOCAL_SRC}"
PYTHON_VER=3.12.10
wget -nc "https://www.python.org/ftp/python/${PYTHON_VER}/Python-${PYTHON_VER}.tar.xz"
tar xf "Python-${PYTHON_VER}.tar.xz"
cd "Python-${PYTHON_VER}"

./configure \
    --prefix="${LOCAL_PYTHON_PREFIX}" \
    --enable-optimizations \
    --enable-shared \
    --with-lto \
    --with-system-ffi \
    --with-ensurepip=install \
    LDFLAGS="-Wl,-rpath,${LOCAL_PYTHON_PREFIX}/lib"

make -j$(nproc)
make install
```

Verify:

```bash
"${LOCAL_PYTHON_PREFIX}/bin/python3.12" --version
# → Python 3.12.10
```

---

## Phase 3 — Create the Virtual Environment

```bash
"${LOCAL_PYTHON_PREFIX}/bin/python3.12" -m venv "${LOCAL_ENV}/venv"
source "${LOCAL_ENV}/venv/bin/activate"

pip install --upgrade pip setuptools wheel
pip install numpy cython mpi4py       # needed before PETSc
pip install nanobind scikit-build-core cmake     # needed for Basix/DOLFINx Python
```

> From this point on, every command assumes the venv is activated.

Set convenience variables used by every subsequent build:

```bash
export CC=mpicc
export CXX=mpicxx
export FC=mpifort

export PREFIX="${LOCAL_PREFIX}"
export PATH="${PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${PREFIX}/lib:${PREFIX}/lib64:${LD_LIBRARY_PATH:-}"
export PKG_CONFIG_PATH="${PREFIX}/lib/pkgconfig:${PREFIX}/lib64/pkgconfig:${PKG_CONFIG_PATH:-}"
export CMAKE_PREFIX_PATH="${PREFIX}:${CMAKE_PREFIX_PATH:-}"
export PYTHONPATH="${PREFIX}/lib/python3.12/site-packages:${PYTHONPATH:-}"
```

### 3a — Eigen3 (if not installed via pacman)

Check first: `pkg-config --modversion eigen3 2>/dev/null`

If it prints nothing:

```bash
cd "${LOCAL_SRC}"
wget -nc https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.bz2
tar xf eigen-3.4.0.tar.bz2
cd eigen-3.4.0
cmake -B build -G Ninja \
    -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build
cmake --install build
```

---

## Phase 4 — Build PETSc 3.24.2 (+ Hypre + METIS + MUMPS + ptscotch + petsc4py)

This is the single biggest step. PETSc's `configure` downloads and builds
Hypre, METIS, MUMPS, ptscotch, ScaLAPACK, SuperLU, SuperLU_dist, and SPAI for you.

```bash
cd "${LOCAL_SRC}"
PETSC_VER=3.24.2
wget -nc "https://web.cels.anl.gov/projects/petsc/download/release-snapshots/petsc-${PETSC_VER}.tar.gz"
tar xf "petsc-${PETSC_VER}.tar.gz"
cd "petsc-${PETSC_VER}"

export PETSC_DIR="${PWD}"
export PETSC_ARCH=linux-gnu-real64-32
```

### 4a — Configure

```bash
python3 ./configure \
    PETSC_ARCH=${PETSC_ARCH} \
    --prefix="${PREFIX}" \
    --COPTFLAGS="-O2" \
    --CXXOPTFLAGS="-O2" \
    --FOPTFLAGS="-O2" \
    --with-64-bit-indices=no \
    --with-debugging=no \
    --with-fortran-bindings=no \
    --with-shared-libraries=1 \
    --with-scalar-type=real \
    --with-precision=double \
    --download-hypre \
    --download-metis \
    --download-parmetis \
    --download-mumps \
    --download-ptscotch \
    --download-scalapack \
    --download-suitesparse \
    --download-superlu \
    --download-superlu_dist \
    --download-spai \
    --with-petsc4py=1
```

> **Notes:**
> - `--with-petsc4py=1` builds petsc4py into `${PETSC_DIR}/${PETSC_ARCH}` and later
>   installs it into `${PREFIX}` via `make install`.
> - PETSc will auto-detect your system MPI (OpenMPI) via `mpicc`.
> - If configure complains about Fortran, ensure `gcc-fortran` is installed.
> - If you want MUMPS to avoid MPI_IN_PLACE issues (like the container), you can add:
>   `--download-mumps-avoid-mpi-in-place` (this is a known PETSc configure flag).

### 4b — Build & Install

```bash
make PETSC_DIR="${PETSC_DIR}" PETSC_ARCH=${PETSC_ARCH} all -j$(nproc)
make PETSC_DIR="${PETSC_DIR}" PETSC_ARCH=${PETSC_ARCH} install
```

After install, update the environment:

```bash
# After install, PETSC_DIR points to the prefix, PETSC_ARCH is empty
export PETSC_DIR="${PREFIX}"
unset PETSC_ARCH
```

Verify:

```bash
python3 -c "import petsc4py; from petsc4py import PETSc; print('PETSc', PETSc.Sys.getVersion())"
```

### 4c — (Optional) Build SLEPc 3.24.1

If you need eigenvalue solvers (DOLFINx can optionally use SLEPc):

```bash
cd "${LOCAL_SRC}"
SLEPC_VER=3.24.1
wget -nc "https://slepc.upv.es/download/distrib/slepc-${SLEPC_VER}.tar.gz"
tar xf "slepc-${SLEPC_VER}.tar.gz"
cd "slepc-${SLEPC_VER}"

export SLEPC_DIR="${PWD}"
python3 ./configure --prefix="${PREFIX}"
make SLEPC_DIR="${PWD}" -j$(nproc)
make SLEPC_DIR="${PWD}" install
export SLEPC_DIR="${PREFIX}"
```

---

## Phase 5 — Build Parallel HDF5 (if not using system package)

If you have `hdf5-openmpi` installed via pacman (check: `h5pcc --version`), **skip this step**.
The system package at `/usr/lib/openmpi/` should work.

Otherwise:

```bash
cd "${LOCAL_SRC}"
HDF5_VER=2.0.0
# Use the 1.14.x series URL naming or the HDF5 2.0.0 URL:
wget -nc "https://github.com/HDFGroup/hdf5/releases/download/hdf5_2.0.0/hdf5-2.0.0.tar.gz"
tar xf hdf5-2.0.0.tar.gz
cd hdf5-2.0.0

CC=mpicc ./configure \
    --prefix="${PREFIX}" \
    --enable-parallel \
    --enable-shared \
    --disable-static
make -j$(nproc)
make install
```

---

## Phase 6 — Build ADIOS2 2.11.0 (optional, for checkpoint I/O)

DOLFINx can use ADIOS2 for file I/O. It's optional but the container has it.

```bash
cd "${LOCAL_SRC}"
wget -nc https://github.com/ornladios/ADIOS2/archive/refs/tags/v2.11.0.tar.gz -O adios2-2.11.0.tar.gz
tar xf adios2-2.11.0.tar.gz
cd ADIOS2-2.11.0

cmake -B build -G Ninja \
    -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DADIOS2_USE_MPI=ON \
    -DADIOS2_USE_Python=ON \
    -DADIOS2_USE_Fortran=OFF \
    -DADIOS2_USE_HDF5=ON \
    -DADIOS2_BUILD_EXAMPLES=OFF \
    -DADIOS2_BUILD_TESTING=OFF
cmake --build build -j$(nproc)
cmake --install build
```

---

## Phase 7 — Build FEniCSx Components

All four components (Basix, UFL, FFCx, DOLFINx) must be version-matched.

### 7a — Basix 0.10.0

```bash
cd "${LOCAL_SRC}"
git clone --branch v0.10.0 --depth 1 https://github.com/FEniCS/basix.git
cd basix

# Build C++ library
cmake -B build-cpp -G Ninja \
    -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpp -j$(nproc)
cmake --install build-cpp

# Build Python bindings
cd python
pip install .
```

### 7b — UFL 2025.0.0

```bash
cd "${LOCAL_SRC}"
git clone --branch 2025.0.0 --depth 1 https://github.com/FEniCS/ufl.git
cd ufl
pip install .
```

### 7c — FFCx 0.10.1

```bash
cd "${LOCAL_SRC}"
git clone --branch v0.10.1 --depth 1 https://github.com/FEniCS/ffcx.git
cd ffcx
pip install .
```

### 7d — DOLFINx 0.10.0

```bash
cd "${LOCAL_SRC}"
git clone --branch v0.10.0.0 --depth 1 https://github.com/FEniCS/dolfinx.git
cd dolfinx
```

#### Build C++ library

```bash
cmake -B build-cpp -S cpp -G Ninja \
    -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DDOLFINX_ENABLE_PETSC=ON \
    -DDOLFINX_ENABLE_SLEPC=ON \
    -DDOLFINX_ENABLE_SCOTCH=ON \
    -DDOLFINX_ENABLE_ADIOS2=ON \
    -DDOLFINX_ENABLE_KAHIP=OFF
cmake --build build-cpp -j$(nproc)
cmake --install build-cpp
```

> Set `-DDOLFINX_ENABLE_SLEPC=OFF` if you skipped SLEPc.  
> Set `-DDOLFINX_ENABLE_ADIOS2=OFF` if you skipped ADIOS2.

#### Build Python bindings

```bash
cd python
pip install -v --no-build-isolation .
```

> `--no-build-isolation` is needed so the build finds your local `basix`, `ffcx` and the
> C++ `dolfinx` library in `${PREFIX}`.

Verify:

```bash
python3 -c "import dolfinx; print('DOLFINx', dolfinx.__version__)"
# → DOLFINx 0.10.0.0
python3 -c "from dolfinx import mesh, fem; print('DOLFINx imports OK')"
```

---

## Phase 8 — Install Python Packages (matching devcontainer)

```bash
pip install \
    ipympl \
    ipywidgets \
    opencv-python \
    "jax[cpu]" \
    pyamg \
    h5py \
    igraph \
    scalene \
    Jinja2 \
    meshio \
    ipykernel \
    jupyter \
    jupyterlab \
    pandas \
    tabulate \
    pyvista \
    matplotlib \
    scipy \
    numba \
    pytest \
    gmsh
```

> **h5py with parallel HDF5**: If you need parallel h5py (MPI-IO), install it from source:
> ```bash
> HDF5_MPI="ON" CC=mpicc pip install --no-binary=h5py h5py
> ```

---

## Phase 9 — Daily Usage: Activation Script

Create `local_env/activate.sh` (done once):

```bash
cat > "${LOCAL_ENV}/activate.sh" << 'ACTIVATE_EOF'
#!/bin/bash
# Source this file:  source local_env/activate.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
LOCAL_ENV="${SCRIPT_DIR}"
PREFIX="${LOCAL_ENV}/prefix"

# Activate the Python venv
source "${LOCAL_ENV}/venv/bin/activate"

# PETSc
export PETSC_DIR="${PREFIX}"

# Paths
export PATH="${PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${PREFIX}/lib:${PREFIX}/lib64:${LD_LIBRARY_PATH:-}"
export PKG_CONFIG_PATH="${PREFIX}/lib/pkgconfig:${PREFIX}/lib64/pkgconfig:${PKG_CONFIG_PATH:-}"
export CMAKE_PREFIX_PATH="${PREFIX}:${CMAKE_PREFIX_PATH:-}"
export PYTHONPATH="${PREFIX}/lib/python3.12/site-packages:${PYTHONPATH:-}"

echo "FEniCSx local environment activated."
echo "  Python:  $(python3 --version)"
echo "  PETSc:   $(python3 -c 'from petsc4py import PETSc; print(PETSc.Sys.getVersion())' 2>/dev/null || echo 'not found')"
echo "  DOLFINx: $(python3 -c 'import dolfinx; print(dolfinx.__version__)' 2>/dev/null || echo 'not found')"
ACTIVATE_EOF

chmod +x "${LOCAL_ENV}/activate.sh"
```

Then daily:

```bash
source local_env/activate.sh
# Now run notebooks, scripts, etc.
```

---

## Phase 10 — Register Jupyter Kernel

So Jupyter notebooks in VS Code (or JupyterLab) pick up this environment:

```bash
source local_env/activate.sh
python3 -m ipykernel install --user --name fenics-local --display-name "FEniCSx (local)"
```

---

## Quick Reference: Version Table

| Component | Version        | Source                                                |
| --------- | -------------- | ----------------------------------------------------- |
| Python    | 3.12.10        | built from source                                     |
| PETSc     | 3.24.2         | built from source (downloads Hypre, METIS, MUMPS ...) |
| petsc4py  | 3.24.2         | built with PETSc (`--with-petsc4py=1`)                |
| SLEPc     | 3.24.1         | built from source (optional)                          |
| Basix     | 0.10.0         | git tag `v0.10.0`                                     |
| UFL       | 2025.0.0       | git tag `2025.0.0`                                    |
| FFCx      | 0.10.1         | git tag `v0.10.1`                                     |
| DOLFINx   | 0.10.0         | git tag `v0.10.0.0`                                   |
| HDF5      | 2.0.0          | system `hdf5-openmpi` or built from source            |
| MPI       | OpenMPI 5.0.10 | system                                                |
| ADIOS2    | 2.11.0         | built from source (optional)                          |

---

## Troubleshooting

### PETSc configure fails to find MPI

Ensure `mpicc` is on PATH and works: `mpicc --version`. On Arch with OpenMPI,
you may need: `export OMPI_CC=gcc OMPI_CXX=g++ OMPI_FC=gfortran`.

### DOLFINx CMake can't find PETSc

Make sure `PETSC_DIR` is set to `${PREFIX}` (not the source dir) **after** `make install`.
`PETSC_ARCH` must be **unset** after install.

### Python import errors / `libdolfinx.so not found`

Check `LD_LIBRARY_PATH` includes `${PREFIX}/lib`. The activation script handles this.

### nanobind / scikit-build-core version issues

```bash
pip install --upgrade "nanobind>=2.0" "scikit-build-core>=0.10"
```

### MUMPS build fails with "MPI_IN_PLACE" errors

Add `--download-mumps-avoid-mpi-in-place` to the PETSc configure line (already included above in notes).

### `pkg-config` can't find `dolfinx`

Ensure `PKG_CONFIG_PATH` includes `${PREFIX}/lib/pkgconfig`. Run:
```bash
pkg-config --modversion dolfinx
```

### GCC 15 compatibility issues

If any component fails with GCC 15, you can try adding these flags:
```bash
export CFLAGS="-Wno-error=incompatible-pointer-types -Wno-error=implicit-function-declaration"
export CXXFLAGS="-Wno-error=template-id-cdtor"
```
Or install GCC 14 from AUR (`gcc14`) and point `CC/CXX/FC` to it.

---

## Disk Space Estimate

| Component        | Source | Build artifacts | Installed   |
| ---------------- | ------ | --------------- | ----------- |
| Python 3.12      | ~30 MB | ~300 MB         | ~120 MB     |
| PETSc + deps     | ~60 MB | ~2 GB           | ~500 MB     |
| SLEPc            | ~15 MB | ~200 MB         | ~50 MB      |
| FEniCSx (all)    | ~50 MB | ~500 MB         | ~100 MB     |
| ADIOS2           | ~30 MB | ~300 MB         | ~80 MB      |
| Python venv+pkgs | —      | —               | ~2 GB       |
| **Total**        |        |                 | **~3-4 GB** |

After building you can delete `local_env/src/` to reclaim ~3 GB of build artifacts,
keeping only `local_env/prefix/`, `local_env/python/`, and `local_env/venv/`.

---

## Full Build Script

See **`local_env_build.sh`** for an automated version of all the above steps.

#!/bin/bash
###############################################################################
# local_env_build.sh
#
# Automated build of FEniCSx + PETSc + all dependencies into local_env/
# Matching dolfinx/lab:stable container (DOLFINx 0.10.0, PETSc 3.24.2)
#
# Usage:
#   chmod +x local_env_build.sh
#   ./local_env_build.sh [--skip-python] [--skip-petsc] [--skip-slepc]
#                        [--skip-adios2] [--skip-fenics] [--skip-pip]
#                        [--jobs N]
#
# Prerequisites (Arch Linux):
#   sudo pacman -S --needed base-devel cmake ninja git wget curl pkg-config \
#       gcc-fortran openmpi openblas boost pugixml spdlog fmt flex bison libxt \
#       openssl zlib xz tk libffi bzip2 readline ncurses sqlite hdf5-openmpi eigen
###############################################################################

set -euo pipefail

# ─── Parse arguments ────────────────────────────────────────────────────────
SKIP_PYTHON=0
SKIP_PETSC=0
SKIP_SLEPC=0
SKIP_ADIOS2=0
SKIP_FENICS=0
SKIP_PIP=0
JOBS=$(nproc)

for arg in "$@"; do
    case "$arg" in
        --skip-python) SKIP_PYTHON=1 ;;
        --skip-petsc)  SKIP_PETSC=1 ;;
        --skip-slepc)  SKIP_SLEPC=1 ;;
        --skip-adios2) SKIP_ADIOS2=1 ;;
        --skip-fenics) SKIP_FENICS=1 ;;
        --skip-pip)    SKIP_PIP=1 ;;
        --jobs=*)      JOBS="${arg#*=}" ;;
        --jobs)        shift; JOBS="${1:-$(nproc)}" ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

# ─── Paths ──────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
LOCAL_ENV="${REPO_ROOT}/local_env"
LOCAL_SRC="${LOCAL_ENV}/src"
PREFIX="${LOCAL_ENV}/prefix"
PYTHON_PREFIX="${LOCAL_ENV}/python"
VENV="${REPO_ROOT}/.venv"

mkdir -p "${LOCAL_SRC}" "${PREFIX}" "${PYTHON_PREFIX}"

# ─── Version pins (match dolfinx/lab:stable) ───────────────────────────────
PYTHON_VER=3.12.10
PETSC_VER=3.24.2
SLEPC_VER=3.24.1
ADIOS2_VER=2.11.0
BASIX_TAG=v0.10.0
UFL_TAG=2025.1.0
FFCX_TAG=v0.10.1
DOLFINX_TAG=v0.10.0.post5

PYTHON_BIN="${PYTHON_PREFIX}/bin/python3.12"

log() { echo -e "\n\033[1;34m>>>\033[0m \033[1m$*\033[0m\n"; }

###############################################################################
# Phase 1 — Python 3.12
###############################################################################
if [[ $SKIP_PYTHON -eq 0 ]]; then
    log "Phase 1: Building Python ${PYTHON_VER}"
    cd "${LOCAL_SRC}"
    if [[ ! -f "Python-${PYTHON_VER}.tar.xz" ]]; then
        wget "https://www.python.org/ftp/python/${PYTHON_VER}/Python-${PYTHON_VER}.tar.xz"
    fi
    rm -rf "Python-${PYTHON_VER}"
    tar xf "Python-${PYTHON_VER}.tar.xz"
    cd "Python-${PYTHON_VER}"

    ./configure \
        --prefix="${PYTHON_PREFIX}" \
        --enable-optimizations \
        --enable-shared \
        --with-lto \
        --with-system-ffi \
        --with-ensurepip=install \
        LDFLAGS="-Wl,-rpath,${PYTHON_PREFIX}/lib"

    make -j"${JOBS}"
    make install

    log "Python ${PYTHON_VER} installed → ${PYTHON_PREFIX}"
    "${PYTHON_BIN}" --version
else
    log "Phase 1: SKIPPED (--skip-python)"
fi

###############################################################################
# Phase 2 — Virtual environment
###############################################################################
log "Phase 2: Creating virtual environment"
if [[ ! -d "${VENV}" ]]; then
    "${PYTHON_BIN}" -m venv "${VENV}"
fi

# Activate
source "${VENV}/bin/activate"

# Set compiler wrappers
export CC=mpicc
export CXX=mpicxx
export FC=mpifort

# Prefix paths
export PATH="${PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${PREFIX}/lib:${PREFIX}/lib64:${LD_LIBRARY_PATH:-}"
export PKG_CONFIG_PATH="${PREFIX}/lib/pkgconfig:${PREFIX}/lib64/pkgconfig:${PKG_CONFIG_PATH:-}"
export CMAKE_PREFIX_PATH="${PREFIX}:${CMAKE_PREFIX_PATH:-}"
export PYTHONPATH="${PREFIX}/lib/python3.12/site-packages:${PYTHONPATH:-}"

pip install --upgrade pip "setuptools<80" wheel
pip install numpy cython mpi4py
pip install "nanobind>=2.0" "scikit-build-core>=0.10"

###############################################################################
# Phase 2a — Eigen3 (if not found)
###############################################################################
if ! pkg-config --exists eigen3 2>/dev/null; then
    log "Phase 2a: Building Eigen3 3.4.0"
    cd "${LOCAL_SRC}"
    if [[ ! -f "eigen-3.4.0.tar.bz2" ]]; then
        wget "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.bz2"
    fi
    rm -rf eigen-3.4.0
    tar xf eigen-3.4.0.tar.bz2
    cd eigen-3.4.0
    cmake -B build -G Ninja \
        -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
        -DCMAKE_BUILD_TYPE=Release
    cmake --build build
    cmake --install build
    log "Eigen3 installed"
else
    log "Phase 2a: Eigen3 found (system), skipping"
fi

###############################################################################
# Phase 3 — PETSc
###############################################################################
if [[ $SKIP_PETSC -eq 0 ]]; then
    log "Phase 3: Building PETSc ${PETSC_VER}"
    cd "${LOCAL_SRC}"
    if [[ ! -f "petsc-${PETSC_VER}.tar.gz" ]]; then
        wget "https://web.cels.anl.gov/projects/petsc/download/release-snapshots/petsc-${PETSC_VER}.tar.gz"
    fi
    rm -rf "petsc-${PETSC_VER}"
    tar xf "petsc-${PETSC_VER}.tar.gz"
    cd "petsc-${PETSC_VER}"

    export PETSC_DIR="${PWD}"
    export PETSC_ARCH=linux-gnu-real64-32

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
        --download-mumps-avoid-mpi-in-place \
        --download-ptscotch \
        --download-scalapack \
        --download-suitesparse \
        --download-superlu \
        --download-superlu_dist

    make PETSC_DIR="${PETSC_DIR}" PETSC_ARCH=${PETSC_ARCH} all -j"${JOBS}"
    make PETSC_DIR="${PETSC_DIR}" PETSC_ARCH=${PETSC_ARCH} install

    # Install petsc4py separately (PETSc's built-in install breaks with modern setuptools)
    log "Phase 3b: Installing petsc4py"
    cd "${LOCAL_SRC}/petsc-${PETSC_VER}/src/binding/petsc4py"
    pip install --no-build-isolation .

    # Post-install: point to prefix
    export PETSC_DIR="${PREFIX}"
    unset PETSC_ARCH

    log "PETSc ${PETSC_VER} + petsc4py installed → ${PREFIX}"
    python3 -c "import petsc4py; from petsc4py import PETSc; print('petsc4py OK, version', PETSc.Sys.getVersion())"
else
    log "Phase 3: SKIPPED (--skip-petsc)"
    export PETSC_DIR="${PREFIX}"
    unset PETSC_ARCH
fi

###############################################################################
# Phase 4 — SLEPc (optional)
###############################################################################
if [[ $SKIP_SLEPC -eq 0 ]]; then
    log "Phase 4: Building SLEPc ${SLEPC_VER}"
    cd "${LOCAL_SRC}"
    if [[ ! -f "slepc-${SLEPC_VER}.tar.gz" ]]; then
        wget "https://slepc.upv.es/download/distrib/slepc-${SLEPC_VER}.tar.gz"
    fi
    rm -rf "slepc-${SLEPC_VER}"
    tar xf "slepc-${SLEPC_VER}.tar.gz"
    cd "slepc-${SLEPC_VER}"

    export SLEPC_DIR="${PWD}"
    python3 ./configure --prefix="${PREFIX}"
    make SLEPC_DIR="${PWD}" -j"${JOBS}"
    make SLEPC_DIR="${PWD}" install
    export SLEPC_DIR="${PREFIX}"

    log "SLEPc ${SLEPC_VER} installed → ${PREFIX}"
else
    log "Phase 4: SKIPPED (--skip-slepc)"
fi

###############################################################################
# Phase 5 — ADIOS2 (optional)
###############################################################################
if [[ $SKIP_ADIOS2 -eq 0 ]]; then
    log "Phase 5: Building ADIOS2 ${ADIOS2_VER}"
    cd "${LOCAL_SRC}"
    if [[ ! -f "adios2-${ADIOS2_VER}.tar.gz" ]]; then
        wget "https://github.com/ornladios/ADIOS2/archive/refs/tags/v${ADIOS2_VER}.tar.gz" \
            -O "adios2-${ADIOS2_VER}.tar.gz"
    fi
    rm -rf "ADIOS2-${ADIOS2_VER}"
    tar xf "adios2-${ADIOS2_VER}.tar.gz"
    cd "ADIOS2-${ADIOS2_VER}"

    cmake -B build -G Ninja \
        -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DADIOS2_USE_MPI=ON \
        -DADIOS2_USE_Python=ON \
        -DADIOS2_USE_Fortran=OFF \
        -DADIOS2_USE_HDF5=ON \
        -DADIOS2_BUILD_EXAMPLES=OFF \
        -DADIOS2_BUILD_TESTING=OFF
    cmake --build build -j"${JOBS}"
    cmake --install build

    log "ADIOS2 ${ADIOS2_VER} installed → ${PREFIX}"
else
    log "Phase 5: SKIPPED (--skip-adios2)"
fi

###############################################################################
# Phase 6 — FEniCSx (Basix → UFL → FFCx → DOLFINx)
###############################################################################
if [[ $SKIP_FENICS -eq 0 ]]; then

    # ── Basix ───────────────────────────────────────────────────────────────
    log "Phase 6a: Building Basix (${BASIX_TAG})"
    cd "${LOCAL_SRC}"
    if [[ ! -d basix ]]; then
        git clone --branch "${BASIX_TAG}" --depth 1 https://github.com/FEniCS/basix.git
    fi
    cd basix
    cmake -B build-cpp -G Ninja \
        -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
        -DCMAKE_BUILD_TYPE=Release
    cmake --build build-cpp -j"${JOBS}"
    cmake --install build-cpp
    cd python && pip install . && cd ..
    log "Basix installed"

    # ── UFL ─────────────────────────────────────────────────────────────────
    log "Phase 6b: Installing UFL (${UFL_TAG})"
    cd "${LOCAL_SRC}"
    if [[ ! -d ufl ]]; then
        git clone --branch "${UFL_TAG}" --depth 1 https://github.com/FEniCS/ufl.git
    fi
    cd ufl && pip install . && cd ..
    log "UFL installed"

    # ── FFCx ────────────────────────────────────────────────────────────────
    log "Phase 6c: Installing FFCx (${FFCX_TAG})"
    cd "${LOCAL_SRC}"
    if [[ ! -d ffcx ]]; then
        git clone --branch "${FFCX_TAG}" --depth 1 https://github.com/FEniCS/ffcx.git
    fi
    cd ffcx && pip install . && cd ..
    log "FFCx installed"

    # ── DOLFINx ─────────────────────────────────────────────────────────────
    log "Phase 6d: Building DOLFINx (${DOLFINX_TAG})"
    cd "${LOCAL_SRC}"
    if [[ ! -d dolfinx ]]; then
        git clone --branch "${DOLFINX_TAG}" --depth 1 https://github.com/FEniCS/dolfinx.git
    fi
    cd dolfinx

    # Determine optional enables based on what was built
    SLEPC_FLAG="-DDOLFINX_ENABLE_SLEPC=ON"
    ADIOS2_FLAG="-DDOLFINX_ENABLE_ADIOS2=ON"
    [[ $SKIP_SLEPC -eq 1 ]]  && SLEPC_FLAG="-DDOLFINX_ENABLE_SLEPC=OFF"
    [[ $SKIP_ADIOS2 -eq 1 ]] && ADIOS2_FLAG="-DDOLFINX_ENABLE_ADIOS2=OFF"

    cmake -B build-cpp -S cpp -G Ninja \
        -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DHDF5_NO_FIND_PACKAGE_CONFIG_FILE=ON \
        -DDOLFINX_ENABLE_PETSC=ON \
        "${SLEPC_FLAG}" \
        -DDOLFINX_ENABLE_SCOTCH=ON \
        "${ADIOS2_FLAG}" \
        -DDOLFINX_ENABLE_KAHIP=OFF
    cmake --build build-cpp -j"${JOBS}"
    cmake --install build-cpp

    # Patch installed DOLFINXConfig.cmake for HDF5 2.0 compatibility
    # (HDF5 2.0 sets HDF5_PROVIDES_PARALLEL instead of HDF5_IS_PARALLEL)
    sed -i 's/if(HDF5_FOUND AND NOT HDF5_IS_PARALLEL)/if(HDF5_FOUND AND NOT HDF5_IS_PARALLEL AND NOT HDF5_PROVIDES_PARALLEL)/' \
        "${PREFIX}/lib/cmake/dolfinx/DOLFINXConfig.cmake"
    log "Patched DOLFINXConfig.cmake for HDF5 2.0 parallel detection"

    # Python bindings (petsc4py must already be installed)
    cd python
    pip install -v --no-build-isolation .
    cd ..

    log "DOLFINx installed → ${PREFIX}"
    python3 -c "import dolfinx; print('DOLFINx', dolfinx.__version__)"
else
    log "Phase 6: SKIPPED (--skip-fenics)"
fi

###############################################################################
# Phase 7 — Python packages (matching devcontainer)
###############################################################################
if [[ $SKIP_PIP -eq 0 ]]; then
    log "Phase 7: Installing Python packages"
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
        pytest \
        gmsh

    # Optional: parallel h5py (uncomment if needed)
    # HDF5_MPI="ON" CC=mpicc pip install --no-binary=h5py --force-reinstall h5py

    log "Python packages installed"
else
    log "Phase 7: SKIPPED (--skip-pip)"
fi

###############################################################################
# Phase 8 — Activation script + Jupyter kernel
###############################################################################
log "Phase 8: Writing activation script"

cat > "${LOCAL_ENV}/activate.sh" << 'ACTIVATE_EOF'
#!/bin/bash
# Source this file:  source local_env/activate.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
LOCAL_ENV="${SCRIPT_DIR}"
PREFIX="${LOCAL_ENV}/prefix"

# Activate the Python venv
source "${REPO_ROOT}/.venv/bin/activate"

# PETSc / SLEPc
export PETSC_DIR="${PREFIX}"
if [[ -f "${PREFIX}/lib/libslepc.so" ]]; then
    export SLEPC_DIR="${PREFIX}"
fi

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

# Register Jupyter kernel
python3 -m ipykernel install --user --name fenics-local --display-name "FEniCSx (local)"

###############################################################################
log "BUILD COMPLETE"
log "Activate with:  source local_env/activate.sh  (venv is at .venv/)"
###############################################################################

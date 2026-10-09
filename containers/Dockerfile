# ==============================================================================
# Dockerfile — Reproducible Datacenter Build & Benchmark Container
#
# Target Environments:
#   - NVIDIA Datacenter GPUs (H100, A100, L40S, RTX 6000 Ada, V100, etc.)
#   - HPC Clusters & Multi-GPU compute nodes
#
# Toolchains:
#   - CUDA Toolkit (nvcc) with multi-architecture support
#   - GNU Fortran 15 (gfortran-15) for pinned ELMM codebase
#   - GCC 14 / G++ 14 (host compiler for nvcc compatibility)
#   - FFTW3 (single/double/OpenMP), OpenBLAS, LAPACK
#   - Python 3 with fparser, islpy, matplotlib, TeX live
# ==============================================================================
ARG BASE_IMAGE=nvidia/cuda:12.6.2-devel-ubuntu24.04
FROM ${BASE_IMAGE}

LABEL maintainer="Research Team" \
      description="Fortran/CUDA compiler and ELMM reproducible benchmark pipeline" \
      version="2.0"

# Avoid interactive prompts during apt package installation
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib

# ------------------------------------------------------------------------------
# 1. System Dependencies & Compiler Toolchains
# ------------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        wget \
        gnupg \
        software-properties-common \
        git \
        build-essential \
        make \
        cmake \
        ninja-build \
        scons \
        pkg-config \
        libtool \
        autotools-dev \
        # Native math libraries for ELMM and stencils
        libfftw3-dev \
        libfftw3-double3 \
        libfftw3-single3 \
        libfftw3-omp-dev \
        libopenblas-dev \
        liblapack-dev \
        # Python 3
        python3 \
        python3-pip \
        python3-venv \
        python3-dev \
        # LaTeX packages for publication-grade matplotlib figures
        texlive-latex-base \
        texlive-latex-extra \
        texlive-fonts-recommended \
        cm-super \
        dvipng \
        # Diagnostic & system utilities
        time \
        procps \
        rsync \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------------------------
# 2. GNU Toolchain Setup (GFortran 15 + GCC 14)
# ------------------------------------------------------------------------------
# Add ubuntu-toolchain-r/test PPA for gfortran-15
RUN add-apt-repository -y ppa:ubuntu-toolchain-r/test && \
    apt-get update && apt-get install -y --no-install-recommends \
        gfortran-15 \
        gcc-14 \
        g++-14 \
    && rm -rf /var/lib/apt/lists/*

# Configure compiler symlinks and default alternatives
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-14 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-14 100 \
    && update-alternatives --install /usr/bin/gfortran gfortran /usr/bin/gfortran-15 100 \
    && ln -sf /usr/bin/gfortran-15 /usr/bin/fc \
    && ln -sf /usr/bin/g++-14 /usr/local/cuda/bin/gcc-host 2>/dev/null || true

# ------------------------------------------------------------------------------
# 3. Python Virtual Environment & Packages
# ------------------------------------------------------------------------------
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv ${VIRTUAL_ENV}
ENV PATH="${VIRTUAL_ENV}/bin:/usr/local/cuda/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}" \
    CUDA_HOME="/usr/local/cuda" \
    FC="gfortran-15" \
    CC="gcc-14" \
    CXX="g++-14" \
    CUDA_HOST_CXX="/usr/bin/g++-14"

# Upgrade pip & install core Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install required pipeline packages
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    pytest \
    numpy==2.5.3 \
    scipy \
    ruff \
    tabulate

# ------------------------------------------------------------------------------
# 4. Workspace & Mount Points Setup
# ------------------------------------------------------------------------------
WORKDIR /workspace

# Create output volume mount point with universal write permissions
RUN mkdir -p /output && chmod 777 /output
RUN mkdir -p /tmp/matplotlib && chmod 777 /tmp/matplotlib

# Copy entire repository into container image
COPY . /workspace

# Set execution permissions on helper scripts
RUN chmod +x /workspace/containers/entrypoint.sh \
             /workspace/containers/reproduce_plots.sh

ENV PYTHONPATH="/workspace:/workspace/elmm-pipeline"

# ------------------------------------------------------------------------------
# 5. Default Entrypoint & Command
# ------------------------------------------------------------------------------
ENTRYPOINT ["/workspace/containers/entrypoint.sh"]
CMD ["reproduce-plots"]


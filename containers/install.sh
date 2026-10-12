#!/usr/bin/env bash
# Shared Docker/Apptainer toolchain installation; run during image construction.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends ca-certificates software-properties-common \
    git build-essential make cmake python3 python3-venv python3-dev \
    libfftw3-dev libopenblas-dev liblapack-dev time procps
add-apt-repository -y ppa:ubuntu-toolchain-r/test
apt-get update
apt-get install -y --no-install-recommends gfortran-15 gcc-14 g++-14
update-alternatives --install /usr/bin/gfortran gfortran /usr/bin/gfortran-15 100
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-14 100
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-14 100
python3 -m venv /opt/venv
/opt/venv/bin/python -m pip install --no-cache-dir -r /workspace/requirements.txt \
    numpy==2.5.3 scons==4.11.1 pytest
rm -rf /var/lib/apt/lists/*

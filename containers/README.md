# Reproducible Datacenter Container Pipeline

Production-ready Docker and Apptainer (Singularity) container environments to reproducibly compile, execute, benchmark, and generate publication-quality figures on datacenter-grade GPU systems (e.g., NVIDIA H100, A100, L40S, RTX 6000 Ada, V100).

---

## 1. Architecture & Toolchain Overview

The containers provide a fully isolated and reproducible environment conforming to all compiler constraints:
- **Base Image:** Ubuntu 24.04 LTS (`nvidia/cuda:12.6.2-devel-ubuntu24.04`)
- **Fortran Compiler:** GNU Fortran 15 (`gfortran-15`) from `ppa:ubuntu-toolchain-r/test` (strictly required; GCC 16 rejects an upstream pure geometry constructor in ELMM)
- **C/C++ Host Compiler:** GCC 14 / G++ 14 (`gcc-14` / `g++-14`, required for host compatibility with `nvcc`)
- **CUDA Toolkit:** NVIDIA CUDA 12.6 (`nvcc`, runtime libraries, CUBLAS, Nsight Systems CLI `nsys`)
- **Native Math Libraries:** FFTW3 (single-precision, double-precision, and OpenMP), OpenBLAS, LAPACK
- **Python Environment:** Python 3 with `fparser==0.2.5`, `matplotlib==3.11.2`, `islpy==2026.2.2`, `numpy==2.5.3`, `scipy`, `pytest`, `ruff`
- **Publication Graphics:** TeX Live (`pdflatex`, `cm-super`, `dvipng`) for Computer Modern typography and publication styling in Matplotlib

## 2. Supported Datacenter GPU Architectures

| GPU Model | Microarchitecture | Compute Capability | CUDA Target (`CUDA_ARCH`) | nvfortran flag |
|---|---|---|---|---|
| **Native (Auto-Detect GPU)** | Host Machine GPU | Auto-detected | `native` | auto |
| **NVIDIA B200 / GB200** | Blackwell | 10.0 | `sm_100` | `-gpu=cc100` |
| **NVIDIA H100 (SXM / PCIe)** | Hopper | 9.0 | `sm_90` | `-gpu=cc90` |
| **NVIDIA L40 / L40S** | Ada Lovelace | 8.9 | `sm_89` | `-gpu=cc89` |
| **NVIDIA RTX 6000 Ada** | Ada Lovelace | 8.9 | `sm_89` | `-gpu=cc89` |
| **NVIDIA A100 (SXM / PCIe)** | Ampere | 8.0 | `sm_80` | `-gpu=cc80` |
| **NVIDIA A10 / A30 / RTX 3090** | Ampere | 8.6 | `sm_86` | `-gpu=cc86` |
| **NVIDIA V100 (SXM / PCIe)** | Volta | 7.0 | `sm_70` | `-gpu=cc70` |

> [!TIP]
> **Native Compilation (`native`):**
> - **GPU (CUDA):** Setting `CUDA_ARCH=native` or `--arch native` instructs `nvcc -arch=native` (analogous to CMake's `CMAKE_CUDA_ARCHITECTURES=native`) to query the installed GPU and compile for its exact compute capability.
> - **CPU (Fortran / C++):** The benchmark build flags in [`benchmarks/common/defaults.mk`](file:///home/jirka/research/fortran-abomination/benchmarks/common/defaults.mk) already default to `-march=native -flto` for both `FC` and `CXX`, automatically generating optimized code with the host CPU's instruction set (AVX-512, AVX2, FMA, etc.).

---

## 3. Quickstart: Reproducing All Plots

All publication figures can be reproduced immediately into the `./out` directory without re-executing long GPU runs by extracting from archived campaign data, or generated from fresh GPU runs.

### With Docker:
```bash
# 1. Build the Docker image
./containers/build-docker.sh

# 2. Reproduce all publication plots
./containers/run-docker.sh reproduce-plots
```

### With Apptainer (Singularity on HPC):
```bash
# 1. Build the SIF image
./containers/build-apptainer.sh

# 2. Reproduce all publication plots
./containers/run-apptainer.sh reproduce-plots
```

### Generated Artifacts in `./out/`:
- `grid_512x512x512_niter100.{png,pdf}` — Microbenchmark execution time (memory transfer vs kernel time)
- `complete-wall-relative.{png,pdf,svg}` — Six-policy complete application wall-clock ratio vs native CPU
- `entry-time.{png,pdf,svg}` — Execution time per entry call across grid sizes
- `transfer-volume.{png,pdf,svg}` — Host/device data transfer volume per entry
- `scoped-wall-relative.{png,pdf,svg}` — Scoped region application comparison across momentum/SGS closures
- `scoped-transfers-wall-relative.{png,pdf,svg}` — Scoped transfer controls (pinned, direct, pipelined)
- `scoped-application-report.md` — Verified timing summary table and provenance receipts
- `manifest.json` — Cryptographic SHA-256 digests and file metadata for every generated figure

---

## 4. Running Fresh Benchmarks on Datacenter GPUs

When running on an active GPU node, pass `--run-benchmarks` and specify the target architecture:

### NVIDIA A100 (Compute 8.0):
```bash
# Docker
./containers/run-docker.sh run-benchmarks --arch sm_80

# Apptainer
./containers/run-apptainer.sh run-benchmarks --arch sm_80
```

### NVIDIA H100 (Compute 9.0):
```bash
# Docker
./containers/run-docker.sh run-benchmarks --arch sm_90

# Apptainer
./containers/run-apptainer.sh run-benchmarks --arch sm_90
```

### NVIDIA L40S / RTX 6000 Ada (Compute 8.9):
```bash
# Docker
./containers/run-docker.sh run-benchmarks --arch sm_89

# Apptainer
./containers/run-apptainer.sh run-benchmarks --arch sm_89
```

The runner automatically:
1. Rebuilds cases using `FC=gfortran-15`, `CXX=g++-14`, and `NVCC=nvcc -arch=<CUDA_ARCH>`.
2. Validates deterministic correctness against serial Fortran (`python3 -m benchmarks.harness.check`).
3. Executes multi-grid timing sweeps (`python3 -m benchmarks.harness.run`).
4. Generates publication-ready figures directly into the mounted output volume (`/output`).

---

## 5. HPC / SLURM Batch Execution

On supercomputers managed by SLURM, use Apptainer with `--nv` for GPU passthrough. Below is a production SLURM submission script:

```bash
#!/bin/bash
#SBATCH --job-name=fortran-pipeline
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00

module load apptainer cuda/12.6

# Set output directory
OUTPUT_DIR="${SLURM_SUBMIT_DIR}/out-${SLURM_JOB_ID}"
mkdir -p "${OUTPUT_DIR}"

# Execute benchmark pipeline in container
apptainer run \
    --nv \
    --bind "${OUTPUT_DIR}:/output" \
    "${SLURM_SUBMIT_DIR}/fortran-abomination.sif" \
    reproduce-plots

echo "Generated artifacts saved in: ${OUTPUT_DIR}"
```

Submit with:
```bash
sbatch slurm_job.sh
```

---

## 6. Interactive Debugging & Development

To drop into a bash shell inside the container with all tools and compilers pre-configured:

### Docker:
```bash
./containers/run-docker.sh bash
```

### Apptainer:
```bash
./containers/run-apptainer.sh bash
```

### Running the Test Suites:
```bash
# Run all compiler and harness unit tests
./containers/run-docker.sh test
```

---

## 7. Configuration Reference

### Environment Variables
| Variable | Default Value | Description |
|---|---|---|
| `OUTPUT_DIR` | `./out` | Host directory mounted into container at `/output` |
| `IMAGE_TAG` | `fortran-abomination:latest` | Docker image tag |
| `SIF_FILE` | `./fortran-abomination.sif` | Path to Apptainer SIF image |
| `CUDA_ARCH` | Auto-detected | Target compute architecture (e.g. `sm_80`, `sm_90`) |
| `MPLCONFIGDIR` | `/tmp/matplotlib` | Writable Matplotlib cache directory |
| `PYTHONPATH` | `/workspace:/workspace/elmm-pipeline` | Python module import path |


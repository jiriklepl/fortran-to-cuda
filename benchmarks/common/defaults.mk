# ============================================================
# defaults.mk — shared default values for all benchmarks
# Included by the top-level Makefile; override any variable
# on the command line:  make CASE=CVD NX=256 FC=ifort
# ============================================================

# ---- Benchmark grid / iteration parameters -----------------
NX      ?= 64
NY      ?= 64
NZ      ?= 64
NITER   ?= 100
NWARMUP ?= 5

# ---- Fortran compiler & flags ------------------------------
# Guard against make's built-in FC=f77 default
ifeq ($(origin FC),default)
  FC = gfortran
else
  FC ?= gfortran
endif
FFLAGS       ?= -O3 -march=native -flto
EXTRA_FFLAGS ?=

# ---- C++ compiler & flags ---------------------------------
# Select compatible gfortran/g++ installations through PATH or command-line overrides.
# Guard against make's built-in CXX=g++ default (same pattern as FC above).
ifeq ($(origin CXX),default)
  CXX = g++
else
  CXX ?= g++
endif
CXXFLAGS     ?= -O3 -march=native -flto
EXTRA_CXXFLAGS ?=

# ---- CUDA compiler & flags ---------------------------------
CUDA_HOME  ?= /usr/local/cuda
NVCC       ?= $(CUDA_HOME)/bin/nvcc
NVCCFLAGS  ?= -O3
CUDA_ARCH  ?=

# ---- OpenACC Fortran compiler & flags ----------------------
# nvfortran (NVIDIA HPC SDK) is required for GPU offloading.
# Set FC_ACC=gfortran and ACC_FLAGS=-fopenacc for a CPU-only build.
FC_ACC         ?= nvfortran
ACC_FLAGS      ?= -acc=gpu
# Optionally specify compute capability, e.g. ACC_GPU_CC=89 for Ada Lovelace,
# ACC_GPU_CC=100 for Blackwell.  Leave empty to let nvfortran auto-detect.
ACC_GPU_CC     ?=
# Module output directory flag: gfortran uses -J, nvfortran uses -module
FC_MODULE_FLAG ?= -J

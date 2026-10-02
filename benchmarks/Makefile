# ============================================================
# Benchmarks — unified top-level Makefile
#
# Usage:
#   make CASE=CDV VARIANT=Fortran
#   make CASE=CDV VARIANT=CUDA NX=128 NY=128 NZ=128 NITER=200 NWARMUP=10
#   make CASE=CDU VARIANT=Fortran-OMP NX=256
#   make CASE=CDU VARIANT=CPP
#   make CASE=CDU VARIANT=CPP-OMP NX=256
#   make clean CASE=CDV VARIANT=Fortran
#   make clean-all                          # wipe the entire bin/ tree
#
# Required:
#   CASE    — benchmark kernel (CDU | CDW | CDV)
#
# Optional (all have defaults in common/defaults.mk):
#   VARIANT       — Fortran | Fortran-OMP | Fortran-ACC | CUDA | CUDA-pinned | CPP | CPP-OMP
#   NX NY NZ      — interior grid dimensions
#   NITER         — timed iterations
#   NWARMUP       — warm-up iterations
#   FC FFLAGS     — Fortran compiler and base flags
#   EXTRA_FFLAGS  — appended to FFLAGS (handy for one-off flags)
#   CXX CXXFLAGS  — C++ compiler and base flags (CPP / CPP-OMP variants)
#   EXTRA_CXXFLAGS — appended to CXXFLAGS
#   CUDA_HOME NVCC NVCCFLAGS CUDA_ARCH
# ============================================================

CASE    ?= CDV
VARIANT ?= Fortran
MAIN_SRC ?= main.f90
PYTHON ?= python3

SOURCE_CDU = cdu.f90
SOURCE_CDV = cvd.f90
SOURCE_CDW = cdw.f90
CASE_SRC = $(SOURCE_$(CASE))
ifeq ($(CASE_SRC),)
  $(error Unknown CASE '$(CASE)'; choose CDU, CDV, or CDW)
endif
ifeq ($(filter $(VARIANT),Fortran Fortran-OMP Fortran-ACC CUDA CUDA-pinned CPP CPP-OMP),)
  $(error Unknown VARIANT '$(VARIANT)')
endif

include common/defaults.mk
include common/variants.mk

# ---- Derived variables -------------------------------------
DEFINES      = -DVAR_NX=$(NX) -DVAR_NY=$(NY) -DVAR_NZ=$(NZ) \
               -DVAR_NITER=$(NITER) -DVAR_NWARMUP=$(NWARMUP)

# CUDA-pinned selects its allocation mode without runner-specific flags.
USE_PINNED_MEMORY ?= $(if $(filter CUDA-pinned,$(VARIANT)),1,0)
ifeq ($(USE_PINNED_MEMORY),1)
  PINNED_FLAG = -DUSE_PINNED_MEMORY
endif

ALL_FFLAGS   = $(FFLAGS) $(EXTRA_FFLAGS) $(VARIANT_FFLAGS) -cpp
ALL_CXXFLAGS = $(CXXFLAGS) $(EXTRA_CXXFLAGS) $(VARIANT_CXXFLAGS) $(PINNED_FLAG)

BUILD_ID    = $(CASE)_$(VARIANT)_NX$(NX)_NY$(NY)_NZ$(NZ)_NITER$(NITER)_NWARMUP$(NWARMUP)
ifneq ($(MAIN_SRC),main.f90)
  BUILD_ID := $(BUILD_ID)_$(basename $(notdir $(MAIN_SRC)))
endif
BINDIR_ROOT = $(CURDIR)/bin
BINDIR      = $(BINDIR_ROOT)/$(BUILD_ID)
BENCH       = $(BINDIR)/benchmark

# Generated variants share the same output filenames.
CASE_OBJ    = $(BINDIR)/case_src.o
MAIN_OBJ    = $(BINDIR)/main.o

ifneq ($(filter $(VARIANT),CUDA CUDA-pinned),)
  ALL_OBJS = $(CASE_OBJ) $(BINDIR)/generated_code.o
else ifneq ($(filter $(VARIANT),CPP CPP-OMP),)
  ALL_OBJS = $(CASE_OBJ) $(BINDIR)/generated_cpp_impl.o
else
  ALL_OBJS = $(CASE_OBJ)
endif

# ---- Source directory mapping -------------------------------
SRC_VARIANT = $(if $(filter CUDA-pinned,$(VARIANT)),CUDA,$(VARIANT))
SRC_ROOT = $(if $(filter CPP CPP-OMP CUDA,$(SRC_VARIANT)),generated,cases)
SRC_DIR = $(SRC_ROOT)/$(CASE)/$(SRC_VARIANT)

# ---- Targets -----------------------------------------------
.PHONY: all clean clean-all test

all: $(BENCH)

$(BINDIR):
	mkdir -p $@

# Compile the variant's Fortran source (using SRC_VARIANT)
$(CASE_OBJ): $(SRC_DIR)/$(CASE_SRC) $(MAKEFILE_LIST) | $(BINDIR)
	$(FC) $(ALL_FFLAGS) $(DEFINES) $(FC_MODULE_FLAG) $(BINDIR) -I$(BINDIR) -c $< -o $@

# Compile any extra CUDA sources (using SRC_VARIANT)
$(BINDIR)/%.o: $(SRC_DIR)/%.cu common/common_functions.cuh $(MAKEFILE_LIST) | $(BINDIR)
	$(NVCC) $(NVCCFLAGS) $(PINNED_FLAG) -I$(CURDIR)/common -c $< -o $@

# Compile any extra C++ sources (using SRC_VARIANT)
$(BINDIR)/%.o: $(SRC_DIR)/%.cpp common/common_functions.cuh $(MAKEFILE_LIST) | $(BINDIR)
	$(CXX) $(ALL_CXXFLAGS) -I$(CURDIR)/common -c $< -o $@

# Compile the benchmark driver (main.f90 in the CASE directory)
$(MAIN_OBJ): cases/$(CASE)/$(MAIN_SRC) $(CASE_OBJ)
	$(FC) $(ALL_FFLAGS) $(DEFINES) -I$(BINDIR) -c $< -o $@

# Link everything
$(BENCH): $(MAIN_OBJ) $(ALL_OBJS)
	$(FC) $(FFLAGS) $(EXTRA_FFLAGS) -o $@ $^ $(VARIANT_LDFLAGS)

# Run correctness tests for all (or a specific) case
test:
	@cd .. && $(PYTHON) -m benchmarks.harness.check $(CASE)

# Remove build artefacts for the current CASE/VARIANT/params combination
clean:
	rm -rf $(BINDIR)

# Wipe the entire bin/ tree (all cases, all variants)
clean-all:
	rm -rf $(BINDIR_ROOT)

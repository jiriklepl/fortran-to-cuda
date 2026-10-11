# ============================================================
# variants.mk — per-variant compiler flags and linker flags
# Included by the top-level Makefile after defaults.mk.
# Defines: VARIANT_FFLAGS, VARIANT_LDFLAGS
# ============================================================

ifeq ($(VARIANT),Fortran-ACC)
  FC               = $(FC_ACC)
  # Override FFLAGS: nvfortran doesn't support -march=native or -flto;
  # gfortran -fopenacc + -flto also has LTO/module symbol issues.
  FFLAGS           = -O3
  FC_MODULE_FLAG   = -module
  VARIANT_FFLAGS   = $(ACC_FLAGS)$(if $(ACC_GPU_CC), -gpu=cc$(ACC_GPU_CC),)
  VARIANT_LDFLAGS  = $(ACC_FLAGS)$(if $(ACC_GPU_CC), -gpu=cc$(ACC_GPU_CC),)
  VARIANT_CXXFLAGS =

else ifeq ($(VARIANT),Fortran-OMP)
  VARIANT_FFLAGS   = -fopenmp
  VARIANT_LDFLAGS  = -fopenmp
  VARIANT_CXXFLAGS =

else ifneq ($(filter $(VARIANT),CUDA CUDA-pinned),)
  VARIANT_FFLAGS   =
  VARIANT_LDFLAGS  = -L$(CUDA_HOME)/lib64 -lcudart -lstdc++
  VARIANT_CXXFLAGS =
  ifneq ($(CUDA_ARCH),)
    NVCCFLAGS += -arch=$(CUDA_ARCH)
  endif

else ifeq ($(VARIANT),CPP)
  VARIANT_FFLAGS   =
  VARIANT_LDFLAGS  = -lstdc++
  VARIANT_CXXFLAGS =

else ifeq ($(VARIANT),CPP-OMP)
  VARIANT_FFLAGS   = -fopenmp
  VARIANT_LDFLAGS  = -lstdc++ -fopenmp
  VARIANT_CXXFLAGS = -fopenmp

else
  # Plain Fortran (no extra flags needed)
  VARIANT_FFLAGS   =
  VARIANT_LDFLAGS  =
  VARIANT_CXXFLAGS =
endif

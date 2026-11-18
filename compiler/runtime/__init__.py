"""Reusable generated-code runtime resources."""

# The runtime translation units that a GPU-resident application compiles and
# links, relative to this package. The headers beside them go on the include path.
RESIDENT_SOURCES = ("scoped_memory.f90", "resident_domain.f90", "reductions.f90",
                    "scoped_runtime.cu", "reductions.cu", "snapshots.f90", "snapshots.cu")

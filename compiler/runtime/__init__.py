"""Reusable generated-code runtime resources."""

# The runtime translation units that a GPU-resident application compiles and
# links, relative to this package. The headers beside them go on the include path.
RESIDENT_SOURCES = ("scoped_memory.f90", "resident_domain.f90", "scoped_runtime.cu", "snapshots.f90", "snapshots.cu")
# The domain module of an application that brings none of its own; compiled after RESIDENT_SOURCES.
DEFAULT_DOMAIN_SOURCE = "resident_application.f90"

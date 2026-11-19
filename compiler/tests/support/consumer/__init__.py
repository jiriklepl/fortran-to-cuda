"""Consumer-side checks of what the compiler issues.

A second, independent reading of the scope manifest, its build roles and the
runtime's trace lines. The test suite uses it as an oracle: a manifest the
compiler writes must pass a reader that shares no code with the writer.
"""

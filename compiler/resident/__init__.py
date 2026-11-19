"""Build an application whose selected numerical code stays on the device.

The pieces here work on whole application sources: preprocessing them,
rewriting loop forms the compiler can then outline, and locating places in
them. Outlining, compilation and splicing are in ``compiler.scopes``.
"""

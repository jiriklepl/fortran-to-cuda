# One explicit local Poisson provider, 2026-11-10

The resident application build and optional prepared-pressure API now share one local source snapshot implementation. Both use the caller's explicit `poisson-solver` checkout, including local edits, with the pinned upstream revision and VkFFT dependency. Neither selects a bundled/default solver. This removes a source of drift between the two build paths while preserving the existing numerical implementation and coherence engine.

`prepare_pressure_backend` requires `solver_source`. Its manifest is schema v2 and binds the complete source snapshot. A v1 manifest fails with an explicit rebuild instruction. Snapshot validation checks file contents and inventory; resuming after source changes requires a fresh build. Tests cover missing/unbound source inputs, mutation, additional files, symlinks, compiler identity and the removed manifest format.

The 95-file duplicate `elmm-pipeline/vendor/poisson-solver` implementation is removed. Its exact historical bytes are preserved in the verified `/tmp/fortran-local-provider-migration-20261110-01/retired-vendor-source.tar.gz` archive, with per-file identities in `retired-vendor-source.json`. Historical timing results and controls are not rewritten. There is no new public numerical mode, residency policy or runtime ABI change.

## Integration milestone

- 106 focused pressure preparation/cache/interface checks pass. Six checks initially require explicit pinned native/solver/FFTW paths; those six subsequently pass, for 112 passing checks across the selected set.
- The migrated public preparation API freshly configures, compiles and installs the local provider and adapter, then validates its v2 manifest with the exact selected Fortran compiler.
- Relinking the exact retained application objects against that provider passes simple2, channel2, natural simple109 and channel100. All 20 fields/bounds/halos match the retained GPU control exactly, and every output passes the original native tolerances. This is a provider rebuild and integration check, not a complete application regeneration.
- Short traces establish all 31 expected simple numerical GPU procedures and all 28 channel procedures, with zero CPU FFTW plans. All application processes exit normally and leave no process groups.

The first focused run exposed six test-fixture migration failures: cache fixtures dropped newly required source bindings, and a lazy fixture referenced the retired manifest field. Their corrections retain the stronger production checks. A subsequent attempt to supply FFTW paths failed because the system has versioned libraries without unversioned development links. The final six checks use private aliases to the exact installed libraries. Both failures are retained.

No new performance comparison is claimed. The selected warm-cache result remains [3.60×/3.45× native](../gpu-local-solver-cache-20261110/README.md). The [operator comparison and proposed long-run norm criterion](../gpu-operator-parity-20261110/README.md) are separate work: the original strict natural2458 gate remains failed, and no criterion change is adopted without the user's decision. Broader generic compiler migration, remaining policy cleanup, full release qualification and the research target remain open.

The [machine-readable summary](summary.json) binds build, test, source archive and application receipts. The preparation/relink/qualification scripts are retained in `/tmp/fortran-local-provider-migration-20261110-01`.

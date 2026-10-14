# Intermediate correction review

These checks establish three narrow correctness fixes. **They do not validate the final corrected release, application generation, calibration measurements or performance.** Later captured-lowering changes followed the 90-test run; final full host and CUDA/native validation is recorded separately.

| Check | Before | Corrected behavior |
|---|---|---|
| Independent Python processes, serial and original-team fixtures | PRIVATE inventories changed order with the hash seed, changing execution and contract fingerprints. | Inventories have deterministic ordering; both fingerprints agree under seeds `1` and `91237`. |
| Unknown frontend-shaped metadata | Merely providing `validate()` and `public()` bypassed freezing. | Unknown dataclasses freeze, detach retained aliases and bind their semantic fields into the fingerprint. |
| Copied preparation proof | A copied proof could enter the borrowed-authority path. | Existing proof classes require their exact registered object identity. |

The [baseline log](baseline/pytest.log) records **3 expected failures**. The corrected original-source [focused log](focused/pytest.log) records **90 passes**, including the three regressions, checked-plan coverage, exact-window coverage and unchanged native-preflight tests. A separate [overlay consumer run](overlay/pytest.log) records **120 passes** across original teams, reached owners, native environment operations and immutable inputs. These are test counts; elapsed times in the receipts are pytest durations, not application timings.

The [focused receipt](focused/receipt.json) binds all 168 compiler Python implementation files and the three invoked test modules to unchanged before/after hashes. Its exact intermediate snapshot is reconstructable from `c16d01a0e2028c0f610f1291277f37df413344c9` plus [intermediate-snapshot.patch](focused/intermediate-snapshot.patch). The [revision binding](revision-binding.json) names the five later source changes present in `ff9565f8ac941f9d9fad19e20027840e83f953b5`; this evidence is not relabelled as testing those changes.

The baseline receipt covers four repository files. Its external regression script is archived with a curation-time hash; no pre/post script hash was recorded. The overlay receipt covers two repository implementation files, six test modules and the three replacement modules. Those narrower inventories do not establish that every compiler file was unchanged. The archived `.py.txt` files preserve the temporary module/script bytes without adding discoverable repository tests.

[cross-process-before.json](cross-process-before.json) preserves a compact extraction of the original hash-seed mismatch and the differing PRIVATE inventories. It also records hashes of the larger temporary diagnostic records, which are not duplicated here. The after evidence is the subprocess regression in the source-bound 90-test run. Synthetic profitable observations elsewhere in those unit tests are test inputs, never measured calibration.

Fingerprint portability is limited to the admitted complete-original-window class. Outside that class, rejected speculative work can change sharing among equal internal containers and therefore change graph fingerprints while leaving public semantics and emitted artifacts identical. The reviewed fallback had ownership boundaries, opaque calls and allocation-changing calls, so it could not enter exact-window calibration. This review does not generalize portability to that path.

Immutability follows ordinary supported APIs: owned execution attributes and containers reject mutation; explicitly borrowed registered proofs retain their existing frozen-dataclass, read-only-container and validation contracts. Mutable numerical extraction payloads remain checked before emission/publication. Reflective bypasses through `vars()` or `object.__setattr__()` are outside this claim.

[evidence.json](evidence.json) records the scope and provenance. [artifact-inventory.json](artifact-inventory.json) binds every archived file by SHA256. No compiler sources were changed while curating this directory.

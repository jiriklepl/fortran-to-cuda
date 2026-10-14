# Original advection definitions: the two primary branches

**The six historical declaration edits mix three partial-output routines with three true undefined-entry reads. They apply to the active `channel-simple` path; they are not required by the active fourth-order `channel` path.** These findings describe the original source. No application declaration was modified for this audit or either bounded generation attempt.

All paths below refer to the frozen files under [evidence/application](evidence/application); their SHA-256 values are in [application-source-inventory.json](evidence/application-source-inventory.json). The original `advection.f90` hash is `6437b7e3a1e748eb39f4fcb8c865f23aafccfbfa0842838bd16c0c54b95c10ca`.

## `channel-simple`: method 2, uniform grid

The maintained `containers/cases/channel-simple/main.conf:3` selects method 2. All three grid-file flags in `grid.conf` are false; `initial.f90:807–809` selects `GRID_UNIFORM`. `dynamics.f90:358–366` therefore calls `CDU`, `CDV`, and `CDW`. This is the actual maintained second-order branch, not merely inactive method 6.

| Procedure | Original declaration | Definition and subsequent use |
|---|---|---|
| `CDUdiv` | `advection.f90:16`, `U2` OUT | Interior overwrite at line 39; loops cover `1:Unx,1:Uny,1:Unz`. No read of incoming `U2`. Other sections become undefined on entry and are not redefined. |
| `CDVdiv` | `advection.f90:60`, `V2` OUT | Interior overwrite at line 83; corresponding V extents. Same partial-output distinction. |
| `CDWdiv` | `advection.f90:103`, `W2` OUT | Interior overwrite at line 126; corresponding W extents. Same partial-output distinction. |
| `CDUadv` | `advection.f90:152`, `U2` OUT | Line 177 immediately reads and updates `U2(i,j,k)` without a local prior definition. |
| `CDVadv` | `advection.f90:195`, `V2` OUT | Line 220 immediately reads and updates `V2(i,j,k)` without a local prior definition. |
| `CDWadv` | `advection.f90:238`, `W2` OUT | Line 263 immediately reads and updates `W2(i,j,k)` without a local prior definition. |

`CDU` lines 294–297, `CDV` lines 309–312, and `CDW` lines 323–326 each execute whole-array `set → div → adv → multiply`. The div call invalidates the earlier whole-array definition under its OUT contract and redefines only the interior. The later adv call invalidates even that newly defined interior before reading it. A precise section proof can represent the div behavior; it cannot make the adv entry read defined under the original OUT contract. The final whole-array multiply also reads halo sections that have not been redefined by those calls.

`ArrayUtilities.set_int_3D` at `arrayutilities.f90:410–420` writes every element through `to(:,:,k)` for every third-axis element. `multiply_3D` at lines 457–468 reads and writes every element, including halos. The initial integer zero in the second-order wrappers selects `set_int_3D`.

Method 5 directly calls the same div routines at `dynamics.f90:373–375` and does not add an immediate interior entry-read problem. Method 6 calls the adv routines after whole-array zeroing at lines 377–382 and has the same OUT-entry-read issue. Those methods are not selected by either primary configuration.

## `channel`: method 4, uniform grid

The upstream `elmm/examples/channel/main.conf:3` selects method 4. Its grid-file flags are also false. `dynamics.f90:368–371` calls `CD4divU`, `CD4divV`, and `CD4divW`. There are no `CD4adv` routines in this source file, and this branch does not call any of the six historically edited routines.

| Procedure | OUT declaration | Whole-array initialization | First interior read/update | Whole-array final scaling |
|---|---:|---:|---:|---:|
| `CD4divU` | `advection.f90:336` | 353 | 385 | 489 |
| `CD4divV` | `advection.f90:505` | 522 | 554 | 658 |
| `CD4divW` | `advection.f90:679` | 696 | 728 | 832 |

Each procedure calls real-valued whole-array `set` after entry, before any RMW. `set_3D` at `arrayutilities.f90:363–373` defines the full passed descriptor, including halos. Interior directional updates then read locally defined values; cells outside the updated sections retain the defined zero. The final multiply reads locally defined interior and halo elements. No intervening OUT call kills that definition. The fourth-order branch therefore does not inherit the second-order undefined-entry blocker; scratch arrays, filtering, effects, lowering, and automatic estimates require their own separate proofs.

`initial.f90:130–132` reads the configured advection method and forces method 4 if global discretization order is 4. This reinforces that branch selection must use authenticated runtime/configuration facts for the entire optimized interval; a source-only static branch assumption is insufficient.

## Unapplied declaration diagnostic

A separate private build compared the six historically proposed `OUT → INOUT` declaration changes under the recorded native GFortran 14 flags. [The diagnostic receipt](evidence/advection-intent-diagnostic/report.json) and [exact patch](evidence/advection-intent-diagnostic/source.patch) are retained as a reviewable option. The original application sources, objects, and module files were not changed.

The compiled advection object's executable `.text`, all seven allocated sections, both non-debug relocation sections, and normalized instruction/code-relocation disassembly match. Full object files differ in debug metadata and the generated module interface differs as expected. Callers were not rebuilt and no application was run for this diagnostic. This evidence does not establish full-build numerical equivalence, automatic offloading eligibility, or performance, and it does not convert the patched source into an unchanged-source control. Applying any source correction would require an explicit, authenticated source transition and fresh controls.

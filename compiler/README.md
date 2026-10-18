# Fortran-to-CUDA/C++ compiler

The compiler lowers selected Fortran procedures into immutable computation IR, validates
types and definite definitions, proves loop independence with `islpy`, and emits C++, CUDA,
and a Fortran bridge from checked execution, scheduling, addressing, and memory plans.

Generated numerical code publishes a versioned `numerical_contract` in the public
JSON `build_sources`, library generation metadata and scoped runtime descriptors.
The `separate-arithmetic-v1` contract requires CUDA `--fmad=false --ftz=false
--prec-div=true --prec-sqrt=true` and host C++ `-ffp-contract=off`; NVCC receives
the host requirement through `-Xcompiler`. Its SHA-256 covers the canonical
contract, and generated source/runtime hashes retain that identity. Independent
consumers must apply these options and record the actual command: NVCC does not
expose a verified predefined macro for every option. Original native Fortran
flags remain unchanged. This contract does not promise bitwise agreement for
all arithmetic, authorize IEEE flag observers, or change numerical tolerances.

Offline CUDA and generated-CPU calibration uses the same requirements and binds
them into its identities. Nonempty `NVCC_PREPEND_FLAGS` or `NVCC_APPEND_FLAGS`
is rejected before calibration work because hidden options can override the
recorded command. Historical profiles without the contract remain readable as
evidence; they cannot price the changed backend, and automatic placement uses
native execution until compatible calibration is supplied.

```mermaid
flowchart LR
    F[Fortran procedures] --> P[fparser frontend]
    P --> I[Typed computation IR]
    I --> A[Scalar lifetime and ISL conflicts]
    A --> T[Checked scalar motion and fusion]
    T --> A
    T --> S[Explicit target schedules]
    S --> D[Proved subscript widths]
    D --> E[Checked execution plan]
    E --> M[Memory operations]
    M --> C[C++ / OpenMP and owned CPU sessions]
    M --> G[CUDA and owned GPU sessions]
    E --> B[Fortran bridge / shared ABI]
```

Proved scalar motion and loop fusion are enabled by default. Equal rectangular
loop domains fuse only when fresh dependence analysis proves the combined region
independent; inconclusive candidates retain their original passes. A rectangular
perfect prefix maps to parallel iterations; its remaining ordered body can contain
assignments, conditionals, and sequential inner loops. `--opt-level 0` retains the
source regions, source indexing, and source axis order unless overridden by
`--indexing` or `--schedule`. Automatic
axis ordering favors Fortran locality; spatial tiling is explicit through
`--tile-sizes`. Strict parallelization is the default; `--fallback host` executes
valid regions whose independence is unproved sequentially on the host. Explicit
sessions retain owned storage across calls, using the same Fortran interface for
CPU and CUDA backends.
See [CODE_MAP.md](CODE_MAP.md) for the implementation and extension points.

## Implementation layout

The only Python modules at the package root are the package marker and
`python -m compiler` entry point. Implementation code is grouped by pipeline stage:

```text
compiler/
├── driver/                  CLI, pipeline orchestration, output publication
├── frontend/                Fortran parsing and lowering
├── ir/                      Immutable computation nodes and execution plans
├── analysis/                Semantics, effects, dependence proofs, execution policy
├── transforms/              Checked scalar setup motion and greedy loop fusion
├── scheduling/              Locality ordering and explicit spatial tile plans
├── addressing/              Static integer ranges and proved subscript widths
├── memory/                  Explicit acquisition, coherence, and lifecycle plans
├── scopes/                  Bounded source scopes, native hooks, original-module clones
├── runtime/                 Numeric helpers, profiling, owned buffers, token registry
├── emission/
│   ├── driver.py            Generate all sources from one shared ABI
│   ├── common/              ABI, expressions, loops, session glue, runtime header
│   ├── c/                   C declarations and serial/OpenMP C++ generation
│   ├── cuda/                Device kernels, launches, memory-plan execution
│   └── fortran/             C bindings, public bridge, workspace procedures
├── tests/                   Python and native acceptance tests
└── debugging/               Standalone fparser tree viewer
```

The frontend and analyzer depend on the IR. Emitters consume the IR and checked
plans without importing fparser or ISL. `emission/driver.py` derives the memory plan
after scheduling and addressing, and shared session glue renders buffer ownership through the
runtime. Shared emission helpers do not import backends; each backend owns its language-specific rendering. The runtime
header combines the template under `emission/common/templates/` with numeric,
profiling, and storage units in `runtime/`, preserving a single distributable
support header and `--common-header` behavior.

Reached scope generation uses one bounded consumption pass in
`scopes/reached_consumption.py`. It authenticates the original source, admitted
facts and compiler configuration before generation, then indexes exact prepared
numerical regions together with their physical effects and mapped child calls.
Only explicitly enrolled child builders can use that pass. It expires on exit;
source authority is checked again before returning generated work and adopting
or publishing it. This avoids reconstructing the complete reached plan for each
lookup without retaining proof state across compiler invocations. Replanning
replaces the view and rebinds staged children; the existing generation
transaction owns rollback.

Ordinary lexical generation also reuses exact prepared/effect pairs within one
read-only planning and emission pass. Newly issued pairs retain their fresh
issuance checks; private lookups check the registered owner, tokens, facts and
policy, then retained pairs are revalidated before the pass returns. Failed
passes roll back candidate artifacts and fence publication. Public resolution,
proof validation and later publication remain fresh demands.

Inline source selection uses exact per-procedure node membership recorded in
the immutable compilation-session graph. It does not repeatedly format or walk
the original procedure. Generated projections still require their registered
identity, text and source span; copied or foreign nodes cannot borrow original
authority. Final publication continues to check the live source files.

Package entry points remain `compiler.frontend.lower_file`,
`compiler.analysis.build_execution_plan`, and `compiler.emission.generate_sources`.
`compiler.driver.pipeline.prepare_function(function, *, options=CompilerOptions())`
returns the optimized function and freshly checked plan with schedules and addressing decisions. The immutable
`CompilerOptions` in `compiler.driver.options` is shared by this pipeline and the
CLI; direct calls to `build_execution_plan` keep their existing analysis behavior.
Emitters accept these unscheduled plans using source axis order and source indexing. Schedule and addressing records
remain independent of fparser and ISL.

## Quick start

Run from the repository root with Python 3.10 or newer:

```bash
python -m pip install -r requirements.txt
python -m compiler --input fortran-stencils/elmm_cdu.f90 --kernel CDU --output-dir out --verbose
```

`! kernels` and `! kernel` markers are optional. Select a module subroutine with
`--kernel NAME`, or `--kernel MODULE::NAME` when several modules contain the same
name. Reachable helpers in the same module are discovered and inlined without
markers; unrelated unsupported routines remain untouched. The input must still
parse as Fortran 2008. Use `--require-markers` for the previous strict selection
contract, including markers on every reachable helper.

Inspect candidates without generating files:

```bash
python -m compiler --input ordinary.f90 --list-candidates --json
```

The list reports every module subroutine, its marker status, and either successful
generation eligibility or a rejection reason with source location. It runs the
same semantic, dependence, scheduling, and emission checks as normal compilation,
under the selected options. A rejected procedure is not a promise of automatic
host fallback; the application integration must retain its native implementation.
The Python `discover_file(path)` API reports the narrower frontend result in
`ProcedureCandidate.lowerable`; callers must still run `prepare_function` and
`generate_sources`. `lower_file(path, entry, require_markers=False)` preserves
explicit entry selection and supports the same module qualification.

Normal generation also accepts `--json`:

```bash
python -m compiler --input ordinary.f90 --kernel advance --fallback error \
  --output-dir out --json
```

Its stdout is one object with `kernel`, `supported`, `reason`, and `outputs`.
Successful generation additionally reports `parallel_regions` (including kernels
in conditional branches), `execution_plan`, and the ordinary pooled-call
`memory_plan`. The last two fields reuse the existing human-readable plan reports;
they are explanatory text, not serialized compiler IR. `outputs` lists the files
published in the output directory. Compilation rejection returns
`supported: false`, a diagnostic `reason`, an empty output list, and a nonzero exit
status; existing output files remain unchanged. With both `--verbose` and `--json`,
verbose diagnostics go to stderr so stdout remains valid JSON. Candidate-list
JSON retains its existing array format.

`supported` describes successful generation under the requested fallback policy.
Integrations that require GPU work should select `--fallback error` and require
`parallel_regions` greater than zero. The compiler proves each parallel region
separately and preserves dependence order between kernels; callers should not
require independent accesses across the entire entry.

Runtime dependencies include `fparser` and **`islpy==2026.2.2`**. Generated C++ uses
C++17; add `-fopenmp` to enable its verified OpenMP loops. Compile without that
flag for serial execution of the same generated implementation. CUDA uses one
kernel per accepted region and 256 threads per block. Axis order comes from the
schedule; bounded grids use grid-stride iteration to cover larger domains. Empty
domains skip the launch. CDU, CDV, and CDW each produce one CUDA kernel by default
and four with `--opt-level 0`.

## CLI and outputs

```bash
python -m compiler --input FILE --kernel NAME [options]
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--input`, `-i` | required | Fortran source file |
| `--kernel`, `-k` | required except when listing | Entry subroutine or `module::name`, matched case-insensitively |
| `--require-markers` | off | Require legacy file and routine markers |
| `--list-candidates` | off | Report each module procedure's generation eligibility; write no files |
| `--json` | off | Machine-readable candidate list or generation result; verbose diagnostics use stderr |
| `--output-dir`, `-o` | current directory | Destination directory |
| `--cuda-output` | `generated_code.cu` | CUDA kernels and host C wrapper |
| `--cpp-output` | `generated_cpp_impl.cpp` | C++ implementation with OpenMP annotations |
| `--fortran-output` | `generated_interface.f90` | Original module/procedure interface using `iso_c_binding` |
| `--common-header` | `common_functions.cuh` | Shared indexing, numeric, storage, and timing header |
| `--no-common-header` | off | Use a separately supplied shared header |
| `--opt-level {0,1}` | `1` | Enable proved scalar motion, fusion, and wide addressing; `0` retains source passes |
| `--schedule {source,auto}` | `auto` at level 1, `source` at level 0 | Select axis ordering independently of fusion |
| `--indexing {source,auto}` | `auto` at level 1, `source` at level 0 | Select source INTEGER addressing or statically proved wide subscripts independently of fusion and scheduling |
| `--tile-sizes N[,N...]` | untiled | Positive tile sizes in scheduled fastest-to-slowest order; omitted axes use 1 |
| `--fallback {error,host}` | `error` | Reject unproved regions or run them sequentially on the host |
| `--gpu-policy {always,sections,auto,chunked,hybrid}` | `always` | Select the ordinary-call transfer and execution prototype |
| `--calibration-profile FILE` | none | Explicit reusable hardware calibration for automatic decisions |
| `--memory-model {call,scoped}` | `call` | Shared-buffer numerical entry prototype with `scoped`; requires sections/auto |
| `--scope-transfers {direct,pinned,pipelined,auto,coalesced}` | `direct` | Scoped transfer choice, separate from CPU/GPU placement |
| `--scope-execution {bounded,reached}` | `bounded` | Opt-in reached ownership in original lexical procedures; requires `--form-scopes` |
| `--analyze-effects` | off | Bounded native source effects without requiring GPU lowering; writes no artifacts |
| `--form-scopes` | off | Emit bounded serial source scopes, original-module helpers, and public source/build manifests |
| `--scope-preflight` | off | With `--form-scopes`, report independent source/cost blockers without emitting scopes or artifacts |
| `--scope-facts FILE` | none | Source-hash-bound capture, initialization, and caller facts; requires `--form-scopes` |
| `--scope-cost FILE` | none | Source-bound scope cost request JSON; serial `--form-scopes` with `auto` or `sections`, separate from diagnostic measurement |
| `--source-file FILE` | none | Additional effect-analysis source; repeat for separate modules |
| `--effect-contracts FILE` | none | Versioned explicit contracts for opaque native calls; requires effect analysis |
| `--summary-cache DIRECTORY` | memory only | Optional source-summary disk cache for effect/scope analysis |
| `--host-threads N` | `4` | Total host thread budget, including hybrid GPU coordination |
| `--gpu-collective` | off | Require every thread of one existing OpenMP team to call the entry |
| `--verbose`, `-v` | off | Normalized IR, applied/skipped transformations, schedules, indexing decisions and reasons, region boundaries, scalar privacy, ISL relations, legality results, and memory operations |

### Opt-in CPU/GPU policies

`always` keeps the existing whole-array implementation. The alternatives apply
to ordinary calls; explicit-session coherence is unchanged. `sections` copies
physical rectangular read/write footprints synchronously, preserving separate
opposite faces and values in partially written outputs. Unknown sections use
whole-array transfers. Device storage retains the original logical layout.
Contiguous sections use flat copies; pitched rectangles use 2D or 3D copies.
An interior 3D rectangle takes one runtime copy call instead of one per plane;
higher ranks use one 3D copy per remaining coordinate. Transfer estimates count
these same calls. Disjoint faces stay separate and copyback never fills gaps.
Overlapping sections are reduced to an exact disjoint union when worthwhile,
with at most 32 rectangles and 1,024 intersection checks per array/direction.
The same rectangles feed execution, cost estimates, and transfer traces.
Without calibration, deduplication must reduce bytes without adding copy calls.
With calibration, extra calls are allowed only when saved transfer time exceeds
their latency cost. Ties, uncertain arithmetic, and excessive fragmentation
retain the original rectangles; no bounding volume replaces holes or faces.

`auto` compares native units with GPU intervals of up to four adjacent units
and the complete legal group. It retains units before optional fusion, respects
source control flow and synchronization boundaries, and requires a modeled 20% advantage
for GPU work. Dependent intervals execute in order, with host-visible outputs
at interval boundaries. Missing, uncertain, overflowing, or incompatible cost
estimates select native execution.
Conditional work whose count is only an upper bound currently selects native
execution as well; the compiler does not assume that every branch runs.
Entries mixing empty and active units also retain the original native block:
an empty unit can protect an undefined inner bound from the numerical value ABI.
Scalar inputs used only inside conditional branches or potentially empty retained
loops also keep the entry native, as do entirely unused scalar parameters. This
conservative rule applies to the forced `sections` and `chunked` controls too:
their numerical value ABI would otherwise read inputs that native execution can
leave untouched. Conditional holes using constants, indices, or scalar inputs
already required unconditionally remain supported.
For `sections` and `auto`, entries with host preparation additionally admit
local scalar assignments and host branches. Preparation may read scalar inputs,
descriptors, and arrays proved unchanged throughout the entry. Selected GPU
intervals retain their storage across these approved host nodes: inputs transfer
once for the union of active physical sections, and written sections return
before the following CPU interval. Host statements and branches still execute
in source order; each worker receives its current local scalar values.

The public query evaluates only the checked INTEGER/LOGICAL slice needed for
predicates, bounds, and footprints. It checks integer intermediates and immutable
array indices, preserves nested guards, and never computes numerical REAL setup.
Definition-only queries with exact bounded effects omit unused placement-cost
footprints while retaining checked domains, exact effects and ordered definition
validation. The planning manifest reports `cost_footprints_prepared` per unit.
REAL-dependent decision expressions, host array writes, preparation reading
arrays written by the entry, and sequential fallback regions remain native.
LOGICAL query conversion is permitted only for inputs unconditionally readable
at entry. Structured queries prove that every live by-value scalar is consumed
on their selected source path before accepting execution; unknown read coverage
retains the original native call. Completely unused scalars receive typed zeros
in the generated numerical bridge, preserving its public signature without
reading the original argument. This structured proof can admit inactive branches
alongside active kernels without exposing protected bounds. The original flat
policy guard rules and chunked/hybrid eligibility remain unchanged.
Public `offload.preparation` reports eligibility and reasons; decision traces
include `FORT_OFFLOAD_ACTIVE` with the ordered original region identities.
GPU intervals execute on the device checked during selection, restoring the
executing host thread's previous device afterward.

These prototypes currently retain unfused source units. The default whole-array
path retains normal compiler fusion, so comparisons can differ in kernel count
as well as transferred bytes.

`chunked` provides an asynchronous GPU-only control. `hybrid` additionally
selects a CPU/GPU split from fixed candidates using the same calibration. Only
entries with proven independent slabs across all grouped kernels qualify;
immutable input halos are allowed. Two pinned/device slots and non-default
streams keep upload, computation, and download ordered per slot. The initial
total pinned-memory budget is 64 MiB. Hybrid reserves one host thread to manage
GPU work, while remaining workers process disjoint CPU slabs. All work and
copyback complete before return. Cross-entry residency and dependent chunk
pipelines remain outside these prototypes.
The runtime retains one completed pair of slots, streams, and events between
calls, reusing sufficient capacity on the same device. Inputs are packed and
uploaded afresh on every call, including after host edits, shape changes, or
reallocation. This is scratch reuse, not cross-entry data residency. Concurrent
calls lease separate pairs; active and idle pinned capacity together remain
within the process-wide 64 MiB budget shared with scoped pinned transfers. In-progress
allocations also reserve their capacity. An incompatible idle pair is evicted before allocating or waiting
for budget. Allocation failure before execution selects native work once.
`FORT_RUNTIME_TRACE=1` reports `scratch_reuse` and its retained capacity.
Compact slabs retain
their complete nonslab dimensions; uploads preserve untouched halo elements
that share those slabs. Synchronous section transfers can omit uploads of
rectangles proved to be completely overwritten.

Create calibration independently of any application:

```bash
python -m compiler.offload.calibrate --output hardware.json \
  --threads 4 --precision 64 --cuda-host-cxx g++-14
python -m compiler --input ordinary.f90 --kernel advance --output-dir out \
  --gpu-policy auto --calibration-profile hardware.json --host-threads 4 --json
```

Calibration records launch, transfer, and packing costs, CPU/GPU compute and
memory rates, and CPU worker rates with the coordination thread reserved.
Runtime compatibility requires the calibrated CPU model, GPU UUID/compute
capability, GCC and NVCC version triples, CUDA runtime/driver versions, precision,
and thread budget. Unrecognized toolchain identity disables automatic selection.
No application profile or online timing tunes decisions.
Throughput is an approximation measured with the recorded calibration build
flags (`-O3` and host `-fopenmp`, matching the supplied benchmark workers).
Different application flags and instruction mixes can change realized
rates; retain the recorded flags and validate complete application time before
promoting a policy.

Non-default JSON includes an `offload` contract with analysis availability,
physical footprints, work and launch estimates, eligible intervals/strategies,
and a `native_fallback_query` name. Integrations import that generated Fortran
logical function and pass the original arguments only after allocation checks
and collective synchronization. A false result executes the original native
block and is a successful policy decision. The compiler owns placement and
scheduling; callers need not inspect compiler IR or CUDA text. Query scalar
arguments use references so guarded inner bounds are read only when needed.
Callers with protected inputs must honor a false query and execute their original
native block rather than call the numerical entry directly.
`FORT_OFFLOAD_TRACE=1` records placement decisions separately from CUDA activity;
disable tracing and phase profiling for timing and overlap measurements.

REAL precision is resolved from declarations rather than the spelling of a kind
name. The target kind model supports `REAL(4)`, `REAL(8)`, default `REAL`,
`DOUBLE PRECISION`, INTEGER parameter aliases and arithmetic, and `KIND` of
numeric or logical literals.
Intrinsic `iso_fortran_env` (`real32`, `real64`) and `iso_c_binding` (`c_float`,
`c_double`) kind imports are recognized, including `ONLY` renames. Unknown or
unsupported kinds are rejected. Local INTEGER kind parameters are allowed, but
general constant folding and external module resolution remain unsupported.

The ABI preserves the original dummy argument order, public names, and
`cpp_<procedure>` C symbol. Array extents follow their array pointer in dimension
order. The generated bridge keeps `knd` mapped to binary64 and retains
`start_hot`/`finish_hot` timing hooks. The C timing exports are
`cpp_<procedure>_start_hot` and
`cpp_<procedure>_finish_hot`; the public Fortran hook names remain unchanged.
Numerical CUDA kernels have translation-unit-local linkage, so independently
generated entries link without adapter-side symbol rewriting.
Profiling starts disabled: `start_hot` resets and enables collection, and
`finish_hot` reports and disables it. Ordinary
execution creates no profiling events or phase-by-phase profiling barriers.
Profiled public CUDA calls serialize through a shared profiling lock. Unprofiled
calls do not take that lock. The timing hooks delimit quiescent batches: start
profiling before starting concurrent callers and join them before finishing it.
`calls` counts entry runs, including empty and host-only runs; `kernel_launches`
counts actual launches. Kernel time excludes host execution and transfers.
These three public names are reserved for that generated API. All sources are generated and validated before destination
files are written; an unsupported or unsafe input leaves existing outputs intact.

The ordinary CUDA wrapper creates private owned state, uploads `intent(in)` and
`intent(inout)` data, executes the memory plan, downloads outputs, and releases the
state on each call. Ordinary calls do not use the session token registry, so
independent concurrent callers have independent storage. Ordinary calls remain
synchronous and host-visible. On supported CUDA devices, a private memory pool
reuses device allocation backing across calls; each call still transfers fresh
inputs and outputs. `USE_PINNED_MEMORY`
enables optional pinned-memory support. Host array reads and writes between launches synchronize affected arrays:
a read sees previous device updates, and a partial host write preserves other
device-produced cells before uploading the changed array. Array-valued launch
bounds and strides also see previous device writes. These extra transfers are
included in timing hooks. Unwritten `intent(inout)` cells are preserved. Unwritten portions of
`intent(out)` arrays have no promised values.

## Memory planning and runtime

Shared ownership extends call-local transfers to shared buffers and section
coherence across GPU entries and native CPU operations. The common runtime and
bounded source scopes are opt-in features. Their public interfaces and supported
boundaries are described below; eligibility alone does not establish a GPU
speedup in a complete application.

`compiler.memory.plan_memory(plan, parameters, *, acquisition_policy=None)` derives immutable acquisition,
caller uploads/downloads, host/device access, execution, write-invalidation, synchronization, and release
operations from an execution plan. It preserves conditional branches, counts
predicate and array-valued launch-bound reads, and requests current contents before
partial writes. `format_memory` describes these operations independently of
emission. Host/device access operations represent conditional transfers: a storage
runtime decides whether its current ownership state requires a copy.
Explicit uploads/downloads always transfer the selected caller arrays. Before
emission, `validate_memory` checks operation payloads, phase placement, array
ownership, and complete acquisition, input upload, output retrieval, and release.
Unsupported operations fail instead of being silently skipped.
Acquisition policy is `dedicated` or `pooled`; omitted metadata preserves dedicated
allocation. Default generation supplies dedicated session plans and pooled ordinary
CUDA plans. Explicit `generate_cuda(..., memory=...)` plans remain authoritative
for both APIs unless `ordinary_memory=` supplies a separate ordinary lifecycle.
Helpers are shared only when their operation sequences match.

`runtime/storage.hpp` provides owned fixed-shape CPU/CUDA buffers, lazy CUDA host
mirrors, selective updates, checked extent/byte products, scoped pinned-memory
registration, and a registry that diagnoses invalid or stale tokens. Buffers retain
no caller pointers. `runtime/allocation.hpp` owns dedicated and pooled CUDA
allocations separately from coherence state; `runtime/timing.hpp` contains opt-in
profiling. These units are assembled into the existing support header and tested
with instrumented CUDA calls.

Emission renders every lifecycle operation, including synchronization without an
array payload. State starts with empty buffer slots and fixed extent metadata;
planned acquisition constructs buffers and planned release destroys them, with
RAII as a cleanup backstop. Shared state helpers implement ordinary CUDA calls
and explicit sessions; the registry only owns states and validates tokens.
Host statements, branches, fallback regions, and array-valued launch bounds share
the coherence path. CPU sessions consume the same memory operations with both
address spaces mapped to owned host buffers; ordinary C++ calls use direct lowering.
Verbose output displays the
memory operations for both lifecycles; `FORT_RUNTIME_TRACE=1` logs successful GPU
allocation requests, transfers, kernel launches, and releases, with additional
events identifying pool operations. Allocation-request counts do not measure
physical backing allocations: native pool requests still occur on warm calls.

### Shared memory runtime foundation

Export the independently compiled runtime and its versioned manifest:

```bash
python -m compiler --emit-scoped-runtime --json --output-dir out/scoped
```

The output contains `scoped_runtime.h`, `scoped_entry.hpp`, `section_copy.hpp`, `staging.hpp`,
`scoped_regions.hpp`, `scoped_planning.hpp`, `scoped_cuda_resources.hpp`,
`scoped_runtime.cu`, `fort_scoped_memory.f90`, and `scoped-runtime.json`. Compile and link the CUDA
runtime once per executable with host OpenMP support (`-Xcompiler=-fopenmp`) and compile the common Fortran interface before its
users. The manifest identifies the ABI, source hashes, source languages, and
build ordering. No input procedure is required for this operation.

Completed GPU scopes can reuse one idle stream and CUDA memory pool per device.
Idle pools retain at most 64 MiB; the runtime explicitly trims and checks the
reserved capacity, discarding resources if the driver cannot meet that limit.
Only physical execution resources survive: buffer registrations, definitions
and freshness are rebuilt for every invocation. Poisoned or incomplete scopes
never enter this cache. `FORT_RUNTIME_TRACE=1` reports `initialize_reuse` with the
retained byte capacity; legacy allocation counters still count logical field
allocations. The cache is lazy, so native-only execution does not query CUDA.
At scope close, stream-ordered pools queue final field downloads and field/scratch
retirement before one completion wait. Native access hooks and unregister still
publish synchronously; devices without memory-pool support retain the original
synchronous allocator path.

The public C/Fortran API provides explicit contexts, canonical buffer handles,
host/device access begin/end, completion, and close. Buffers borrow stable
contiguous host allocations; host storage must remain valid until unregister or
close. Registrations use explicit identities and allocation generations, validate
layouts, and reject independently registered overlapping storage. Sections use
zero-based physical coordinates with exclusive upper bounds. Access descriptors
separate reads, possible writes, and proven overwrites.
After bounded union and subtraction, adjacent rectangles merge only when their
union is itself a rectangle. This reduces copy submissions without adding cells
or filling holes. Execution and planning use the same normalized coverage; the
32-rectangle and 1,024-intersection limits remain in force.

`fort_scope_register_sections` accepts partially initialized host coverage.
`fort_scope_forget_definition` discards old defined/current coverage while
retaining the device allocation, preserving procedure-entry `INTENT(OUT)` events.
Neither re-registering a pointer nor keeping its allocation defines its contents.
`fort_scope_set_device_budget` limits live full-layout fields and cached scratch
payload bytes; set
it before CUDA initialization. Resource exhaustion before an operation starts
permits native continuation, including after earlier completed operations.

CPU reads retain device validity. CPU writes invalidate only affected sections;
GPU writes similarly invalidate host sections. On excess fragmentation, the
runtime reconciles current pieces before coarsening. Excess fragmented partial
initialization requests a boundary before execution. Contexts initialize CUDA
lazily, use one non-default stream and private pool where supported, and wait at
host boundaries. All-native and empty accesses initialize no CUDA resources.
Execution failures poison the context, preventing unsafe replay; abandoning a
failed context releases resources without claiming valid host results.

The additive [scratch ABI](runtime/SCOPED_SCRATCH_ABI.md) supplies one reusable,
context-owned device arena. Leases carry unique tokens, share the context stream,
and cannot publish host data. Zero-byte leases initialize no CUDA resources;
cached capacity counts against the device budget. Separate statistics expose
allocation reuse and combined field/scratch peaks. This storage interface does
not itself implement runtime snapshots or establish automatic placement costs.

Scoped transfers default to `direct`. The opt-in `pinned` control packs exact
sections into reusable pinned staging, copies against the existing full-layout
device buffers, waits for completion, and unpacks only downloaded sections.
Logical indices and array pitches are unchanged. This control is synchronous;
it does not claim transfer/kernel overlap. Two staging slots, streams and events
share the ordinary chunked/hybrid pool's 64 MiB pinned budget, including active,
allocating and cached storage. A scoped copy acquires a short lease without
waiting for another caller to release budget, and releases it before native
procedure execution. A pre-copy resource failure uses direct transfers with an
explicit reason; a failure after enqueue poisons the context and cannot replay
numerical work.

The experimental `coalesced` control can replace multiple physical upload
copies with one dense physical span when it adds at most 12.5% bytes and the
whole canonical source resource is defined and host current. It uses no pinned staging and
keeps original layouts and coordinates. Ineligible uploads use exact direct
copies with `coalesced_upload_ineligible`; downloaded outputs remain exact.
Actual upload counters include the extra bytes, while `upload_requested` and
`upload_coalesced` trace events distinguish required and copied bytes.
Cost-selected schedules retain exact direct transfers until coalesced costs
are calibrated. This control does not change default or automatic transfer
selection.

`--scope-transfers` requires scoped memory for nondefault modes and remains
independent of `--gpu-policy`. Forced `sections` can use `pinned` without a
calibration profile. Automatic GPU placement with explicit `pinned` requires
the additional scoped transfer calibration; pinned bandwidth alone omits
preparation, row packing and event costs. Older profiles remain valid for direct
execution and cannot supply missing transfer costs. Default direct placement
keeps its existing costs and behavior.

`pipelined` admits complete GPU subchains with a common, source-proven independent
slab partition. Two slots each enqueue upload, all subchain kernels, required
download and a completion event in that order on non-default streams. Reusing
a slot waits for its event and unpacks only its exact written output sections.
The executor retains full device-array layouts, original logical bounds and
pitches. Ordered definitions determine incoming reads: a GPU-produced
intermediate is never uploaded from undefined or stale host storage. Immutable
inputs with overlapping chunk reads have their exact required union established
on the device before any batch kernel; subsequent slot uploads exclude them.

Batch sizes come from the existing 256 KiB, 1 MiB, 4 MiB and 16 MiB payload
candidates using offline costs, including packing row overhead, resource
preparation, launches, completion events and pipeline fill/drain. Transfer
`auto` requires a modeled 20% advantage over the corresponding synchronous
transfer execution. Placement remains a separate decision; pipelining cannot
create a GPU placement that the conservative placement planner rejected.
Missing or incompatible calibration and unsupported slab proofs retain direct
execution with a public reason. Resources are acquired before numerical work;
an unapplied batch leaves the ordinary schedule available, and an error after
work starts forbids replay. One batch can be active per context, and both slots
complete before returning to a native boundary.

The additive batch interface publishes eligibility, selected payload and chunk
count, preparation work, modeled execution and terminal-cost change, and actual
completed transfers and launches. A complete-owner estimate becomes unavailable
when changed batch publication has not been repriced; direct estimates are not
reported as if they described pipelined execution. Initial source integration
uses direct numerical leaves in one reached straight-line segment. Native
operations, mutable control boundaries, unsupported wrappers, uncertain aliases
and cross-chunk dependencies retain synchronous execution. This eligibility is
not overlap or speed evidence: use separate Nsight diagnostics with the
synchronizing phase timer disabled and measure complete application wall time
before promoting a policy.

The additive `fort_scope_set_transfers` metadata operation configures a context
before registration/planning and initializes no CUDA resources. Generated
interfaces publish a `configure(context)` helper, and nondefault source owners
call it once before planning. Standalone callers use the public helper before
adding query operations. Complete pinned costs are installed by the additive
`fort_scope_set_transfer_costs_v1` after host/toolchain compatibility checks;
GPU identity is still checked before a GPU selection executes. The versioned
`fort_scope_batch_execute_v1` provides a metadata-only preview and a
synchronous-on-return pipelined executor. Its callbacks receive original
full-array views and the slot stream, and must not call context APIs while the
executor owns that context. `fort_scope_transfer_stats_get_v1` reports requested
and effective modes, fallback reasons, exact packing/transfer counts, staging
reuse, capacity and event statistics without changing the original stats ABI.
The runtime manifest describes the shared budget and supported modes. Common
and scoped runtime units must link into the same program with the published
default-linkage support header to share one pool; isolated loader namespaces
are not covered by that contract.

Emit a numerical entry that borrows common buffer handles:

```bash
python -m compiler --input input.f90 --kernel advance --gpu-policy sections \
  --memory-model scoped --json --output-dir out/scoped-entry
```

The additional shared CUDA source and Fortran interface are identified by public
JSON, including argument order, array types, execution modes, and runtime build
artifacts. Independently generated entries link one common runtime and borrow
handles registered by their caller. Existing ordinary and owned-session outputs
are preserved. Shared entries require read-only scalar parameters and default to
a serial coordinator. Native, forced GPU, and calibrated automatic execution are
available. Planning ABI version 1 exports side-effect-free `plan` queries and a
`choose` selector. Queries record physical effects and checked control values;
mode 2 consumes the resulting worker decisions in source order. Unknown work,
unsafe preparation, or absent/incompatible calibration selects native execution.
Effect-query availability is published separately from cost-estimate availability.
The common runtime's `fort_scope_plan_validate` checks ordered initialization and
preservation requirements without calibration, CUDA initialization, or changes to
live coverage. Source owners invoke it before numerical work for both `sections`
and `auto`; an unavailable or failed preflight retains the original native span.
Successful definition proofs and recorded query snapshots are reused only within
the same context's query and data-state generations. Query changes, registrations,
coherence/definition changes and execution invalidate them; read-only metadata
probes preserve them. Preview choices also check current driver state and exact
calibration costs. No application decision is cached by allocation address.
For a fresh host-current scope, a proven GPU startup lower bound can select native
execution after definition preflight without constructing a coherence simulation.
With `FORT_RUNTIME_TRACE=1`, planning diagnostics report construction, validation
and selection wall intervals and cache hits. The intervals are inclusive and can
nest; they must not be added together as independent costs. Startup-shortcut
evidence is explicitly aggregate-only. Tracing adds no CUDA synchronization.
Query inputs must remain safe and unchanged throughout their recorded planning
segment. The legacy `fort_scope_plan_reset` still plans a complete scope. The
additive `fort_scope_plan_reset_mode` accepts `FORT_SCOPE_PLAN_CONTINUE` to plan a
reached segment without closing its owner. Complete earlier work with
`fort_scope_wait` before resetting a consumed schedule. Successful continuation
selection installs CPU choices even when GPU estimates are unavailable; those
workers execute with access hooks and preserve earlier device results. A failed
query may run only the current segment through safe native hooks. Execution
failure never permits replay of an earlier segment or the owning region.

`fort_scope_plan_report_v2` separates incremental execution cost from hypothetical
publication and teardown. A continuation ranks execution plus the change in
terminal liability against native continuation from the same incoming state;
the ranking can be negative when native writes retire dirty device values.
Projected terminal operations do not execute between segments and do not count
as their transferred bytes. Owner estimates sum reached execution and one final
terminal cost. Creation and registrations are charged when incurred, and retained
allocations are charged only when newly allocated. Unknown or failed execution
invalidates the complete-owner estimate. Existing planning costs and decision
structures remain ABI version 1.
Shared numerical entry ABI version 2 accepts scalar pointers through its C
interface, with matching Fortran reference arguments. Binding those references
does not read their values. A scalar used only behind a conditional or possibly
empty retained loop keeps that worker native, preserving the original protection
before CUDA would capture its value. Other safe workers in the same entry can
still execute on the GPU. Public `region_execution` metadata explains each such
choice; the common memory runtime remains ABI version 1.
Protected CPU reads retain their original guards under host optimization. If a
physical section offset also uses a protected scalar, its native access uses
conservative whole-resource effects without evaluating that offset early.

Adding `--gpu-collective` emits an additive `run_team` interface in the same
Fortran module and an additional C entry identified by public `scoped.team` JSON.
Every member of the fixed host-budget team at OpenMP level one must call it with
the same context, buffer handles, mode, and immutable scalar bindings. This is an
explicit caller contract, not permission to rewrite arbitrary call sites. The
master performs preparation, access hooks and GPU work once; CPU workers use
the existing thread IDs and team size. Shared preparation and nested branch
choices are published at matching barriers. The entry returns a uniform status
after all workers finish; failures after work starts prohibit replay. Existing
serial entry names and behavior remain unchanged. Automatic team execution
requires the separately measured collective protocol described below; serial
calibration alone retains native execution with an explicit reason.

Explicit entries and the initial source scopes below remain opt-in. Complete
application evaluation and remaining lifetime/effect integration are tracked in
[the memory-model plan](MEMORY_MODEL_PLAN.md).

### Compiler-owned source scopes

```bash
python -m compiler --input application.f90 --kernel application::advance \
  --source-file helpers.f90 --form-scopes --scope-facts captures.json \
  --memory-model scoped --gpu-policy sections --json --output-dir out/scopes
```

The capture document has `schema_version: 1`, `participation: "serial"`, and
`sources` mapping every supplied absolute source path to its SHA-256 hash.
`captures` maps canonical identities such as `argument::a` or `module::field`
to `storage: "stable"`, `escapes: false`, `allocation_changes: false`, and
`initialized: "whole"`, `"none"`, or `"sections"`. Partial initialization adds
`sections`, a bounded list of `lower`/exclusive `upper` physical coordinates.
These are storage and source-definition assertions; they do not specify GPU
leaves or transfer recipes. An optional `device_budget_bytes` defaults to 256 MiB.
An optional capture `lifetime: "selected_window"` limits those assertions to a
registered reached window. Such a capture cannot authorize invocation-wide
module stability or be registered before that window's fresh allocation,
original-bound, layout and alias guards. The default remains `"invocation"`.
Neither lifetime choice derives initialized payload from allocation presence.

With `--gpu-policy auto`, absence of both a compatible calibration profile and
an explicit complete-window evidence request selects unchanged native source
before window discovery or numerical lowering. Source/configuration hashes,
entry and package authentication, collective caller assertions, and final live
input checks still apply. The result contains a reasoned boundary, zero scopes,
no source edits, and no generated runtime or build sources. Its native-effect
report contains local structure without materializing the transitive closure.
A supplied profile or independent complete-window request follows its existing
pricing and validation path; one unpriced leaf cannot suppress other estimable
alternatives. This early native decision is not evidence of automatic offloading.

An optional capture `expected_descriptor` has exactly `lower_bounds` and
`extents`: rank-matched lists of signed 64-bit integer lower bounds and
nonnegative signed 64-bit extents. For example,
`{"lower_bounds": [-3], "extents": [12]}` constrains the original reached
descriptor. Original lexical owners check it after separate allocation,
contiguity and integer ABI checks, before registration or numerical queries.
Borrowed module calls check the original actual before dummy association;
rebased dummy bounds do not replace this check. A mismatch publishes and closes
the current owner and executes the original suffix once, without replay or
reopening. Expectations do not establish initialized values or replace
`initialized`/`sections` evidence. Synthetic owners, qualified collective owners
and opaque native metadata currently reject this capture field. The additive public
`capture_descriptor_expectations` version 1 capability and per-resource guard
records distinguish enforcing compilers from older readers that ignore unknown
fact keys; clients using this field must require that capability for generated
owners.

Inside an original joined team, a numerical DO and its exact native alternative
validate their input definitions when selected. A conservative whole-array native
fallback does not impose that requirement before an exact GPU query. Selecting
the native alternative still validates its original read and preservation
requirements before host access; failure publishes and closes ownership before
the original DO and remaining suffix run once. This deferral adds no definitions
and does not apply to other native operations or boundary inspectors.

The compiler selects bounded source spans, generates source helpers in their
original modules, and passes explicit context/root handles down call-only helper
paths. Original native procedures remain available. Native operations execute
behind compiler-generated access hooks; numerical leaves retain physical section
transfers and borrow the same registered buffers. CPU reads retain a current GPU
mirror. Unknown calls, recursion, lifetime changes, uncertain effects, writable
aliases, and unsupported mappings are boundaries. Structured owners preserve
conditions and mutable scalar reads at their original execution points and plan
only reached straight-line segments after their inputs become available. Original
caller guards remain outside their replacements. Active OpenMP teams select the original native span
before descriptors are inspected; a runtime without OpenMP support also selects
native execution because caller participation is unknown.

Straight-line native assignment leaves can publish checked rectangular reads,
writes, and complete overwrites, including separate opposite faces. Queries and
execution hooks use the same physical mapping from the original dummy lower
bounds to full registered storage; staging layouts never change source indices.
The bounded subset admits whole axes, literal points and unit-stride ranges, and
checked same-array dimension inquiries. Dynamic specifications, uncertain
inquiry/function effects, unsupported bounds and aliased mappings retain
conservative whole-resource effects. A write-only partial native `INTENT(OUT)`
can discard old definitions and define its exact written sections; native reads
of that output or nested definition changes remain boundaries. The complete
ordered preflight rejects any undefined read or preservation requirement before
numerical execution, rather than discovering it after a preceding GPU write.
Native OpenMP helpers, including hidden callees, require proved completion.
Serial scopes admit bounded, matched plain `do` or `parallel do` directives,
whose work completes before the original helper returns. Other directives remain
boundaries; memory effects alone cannot show that asynchronous work is finished.
Proven joined inline native OpenMP operations may also remain in a structured
owner; they finish before its next segment. Array conditions receive their
required host coverage through hooks before evaluation, while unrelated device
copies remain valid. The original allocation bounds reach structured owner
dummies before assumed-shape association can rebase them. Unsupported exits,
allocation changes, unknown effects and unproved completion end ownership.
An original `ALLOCATED` predicate on a captured stable allocation may become a
known true value only after the original caller's allocation check. Unallocated
callers execute the unchanged source. This proof does not authorize allocation
changes or substitute ordinary view descriptors for allocatable arguments.

The public `scopes` JSON and saved `scope-manifest.json` identify approved source
replacements, original hashes, artifact hashes, runtime identity, and build roles.
Each scope publishes its synthetic owner parameters with canonical resources and
original actual names. Structured owners additionally publish their reached-segment
tree, native operations, retained resources and explicit ownership boundaries.
Capture dummies use private generated names so original
USE associations and re-exported module fields retain their source bindings.
Hidden mutable module scalars retain their original host or USE bindings rather
than acquiring a second dummy association. When an original native helper
accesses a captured module array through such a binding, the original array
must satisfy the `TARGET` aliasing proof; otherwise ownership ends at that call.
After the caller and contiguity checks, contiguous pointer views bind the original
storage. Context-aware helper calls use those views so `CONTIGUOUS` dummies do not
introduce array temporaries that diverge from the registered host buffers. Native
whole-span fallback retains the original arguments and their normal semantics.
For numerical leaves with writable hidden module arrays that lack `TARGET`, a
pre-execution fallback returns to the original caller before running that span.
Once a segment has started, native continuation uses its coherent workers and
cannot replay the owner.
Stable allocatable owning arrays can be borrowed through ordinary assumed-shape
helpers. A guard at the original caller checks serial participation, then
`ALLOCATED`, before associating synthetic owner arguments or querying their
descriptors. Unallocated and collective paths retain the exact original span;
original outer guards stay in place. Empty allocated arrays remain valid, and
the owning context closes before return or later reallocation. The guard forces
the intrinsic in its own block; conflicting captured/call names remain boundaries.
Context-aware clone array dummies also declare `TARGET`, making their association
with runtime access through the registered address explicit.
Native fallbacks after this storage proof also use the contiguous views, avoiding
whole-array temporaries when the original helper or wrapper declares `CONTIGUOUS`.
Stable module allocatable arrays can also participate in original native
operations behind conservative whole-resource hooks. Their exact canonical
roots need source-hash-bound stable, non-escaping capture facts; the compiler
checks these before effect analysis consumes them. Allocation agreement at the
original caller still precedes descriptor queries and owner association. The
unallocated path executes the original source, preserving its guards. Element
and explicit-section writes retain the original allocation and logical bounds.
Whole-variable allocatable assignment, explicit allocation changes, writable allocatable
formals, uncertain pointer effects, and unknown callees remain boundaries even
when stable storage is asserted. Standalone effect analysis without capture
authority remains conservative. Normalized numerical packages may borrow these
hidden allocations when they supply a unique `lower_bound_dimension` parameter
for every dimension of each dynamic resource. Runtime origins come from the
original allocation descriptor, after allocation agreement and before synthetic
assumed-shape arguments can rebase it. Collective callers use their agreed
original descriptors for the same mapping. The dispatch checks 64-bit `LBOUND`,
`UBOUND` and dimension `SIZE` against the existing 32-bit INTEGER ABI before
conversion; unsupported ranges retain original native execution. Resources used
only for descriptor inquiries still require capture and allocation guards.
The mapping preserves original logical indices, bounds and index-as-data
expressions while numerical buffers retain their full physical layout. Missing
origin mappings remain conservative boundaries; lifetime authority does not
establish an allocation's numerical bounds.
An adapter verifies these artifacts, applies replacements to an application copy,
and links the common runtime once. It does not inspect compiler IR or CUDA text.
No accepted scopes is a successful unchanged/native result.
The independent pipeline exposes this path through opt-in
`--memory-model scoped`, an explicit scope entry/source selection, and a capture
facts file. It exports configured and normalized source packages and consumes
these public build roles; it does not infer initialization from argument intents.
See [the pipeline interface](../elmm-pipeline/README.md) for its options.

### Qualified source callers on existing teams

Source scopes with `--gpu-collective` accept a separate capture schema version 2.
The `sources` and stable initialization facts retain their original meaning.
Array captures additionally assert `association: "shared_whole_storage"` and
`descriptor_uniform: true`; immutable scalar captures assert
`association: "shared_immutable_control"`. The compiler verifies source-backed
caller and worksharing roles independently of these assertions.

`participation` is an object with `kind: "omp_full_team"`,
`dispatch: "qualified_companion"`, the qualified `entry`,
`expected_omp_level: 1`, the fixed `host_threads`, and `call_sites`.
Each site supplies its original `source`, `first_line`, `last_line`, the exact
statement `span_sha256`, enclosing `team_first_line`/`team_last_line`, and
`uniform_guard: "unconditional"`. Optional source/team hashes and a qualified
`caller` assertion must agree with the original AST. The compiler resolves
renames and generic calls, requires an unconditional call in a plain lexical
parallel region, proves whole shared captures and immutable controls, and rejects
private/threadprivate captures, writable aliases, nested or partial participation,
and unsupported clauses. Original configured-source spans remain authoritative.

Initially the owning entry must contain only direct leaf calls and ordinary
formals; owning allocatable formals remain native so entry-time deallocation and
descriptor semantics cannot be bypassed, including for unused outputs. Each original
leaf must contain matched clause-free orphaned `do`/`end do` worksharing, with
no computation outside those loops, nested worksharing, calls, persistent state,
or external scalar writes. Activation-local scalar assignments may be admitted;
the numerical frontend independently proves private definitions and rejects
state carried between iterations. An optional `native_participation` map can assert an
original qualified leaf's source/hash, `kind: "existing_team_worksharing"` and
`completion: "all_participants_before_effect_commit"`; an assertion cannot create
a role the compiler cannot prove. This proof uses generic source structure.

Only the individually proved caller sites enter an additive owning companion.
The original entry and every other caller remain unchanged. Uniform level and
thread-budget checks precede new barriers. Allocation agreement at the original
caller precedes owner argument association, bounds, addresses and query inputs.
The companion compares all participants' extents, original lower bounds,
contiguity and empty-aware addresses, then creates one context and records and
validates the complete ordered definition query before computation. Descriptor
metadata has a one-MiB bound; controller failure or failed preflight before work
retains original execution. Required original native worksharing executes on the
whole team between coordinator access hooks and a completion barrier. The context
closes and all participants finish before capture reuse or return.

This source prototype supports forced `sections`. Automatic execution requires
the optional offline `scoped.collective` calibration for the actual persistent
team, emitted coordination protocol, hardware and toolchain. Without it, `auto`
produces a successful unchanged native result. The public
manifest publishes versioned `participation` proof even when no scopes qualify,
plus original-source role reasons. The independent adapter transports these
facts, applies public source edits and build roles, and performs no role inference
or kernel-text inspection.
Authority facts whose call or effect closure cannot be verified are rejected;
stable module allocation captures can participate in original native leaves
under the lifetime and allocation checks above. Normalized numerical packages
can also supply their original runtime origins under the same descriptor checks.

Add `--collective-costs --fortran gfortran-15` to `--scoped-costs` calibration
to measure original native throughput, generated persistent-team CPU throughput,
and exclusive owner, descriptor, entry, CPU/GPU-worker and native-call costs.
The fixture compiles the public source-scope artifacts, keeps the original team
alive during each measurement, checks complete numerical fields and validates
exclusive protocol costs using fixed 4K and 16K training shapes and an independent
8K warm validation shape. It also checks that public
automatic-mode decisions agree with execution; an all-native decision is valid.
These protocol checks exclude numerical computation and runtime API work, whose
costs are measured separately. Complete-application performance remains a separate
gate. Calibration does not time an application or tune live calls. Production
observation is disabled. The profile records the protocol source hashes and original Fortran
compiler version and semantic options; dispatch checks those identities and the
actual team level/width before reading descriptors. A cached preview cannot bypass
the actual-team check. A whole-native decision closes its metadata-only context
and executes the untouched original calls, using their separately measured rates.
In addition to each interval's margin, a fresh collective owner's complete GPU
or mixed schedule must beat the original whole-native estimate by at least 20%.
Unknown common native computation remains excluded from both estimates.
Repeated `--fortran-flag=FLAG` options replace the default
`-std=f2018 -O3 -fopenmp` list and must include `-fopenmp`. Use the application's
semantic flags in their original order. Only include/module/output location flags
are excluded from the compatibility comparison.

Independent source extractors may add `--numerical-sources package.json` to offer
normalized numerical leaves without selecting scopes or execution policies. The
compiler validates the package against the original sources and lowers each
offered leaf through its ordinary numerical frontend. This allows capture-safe
OpenMP removal and coordinate normalization performed by an independent adapter.
The original procedure remains the native fallback, and its `INTENT(OUT)`
definition events remain at the original call position.

The package contains `schema_version: 1`, `source_inputs` (the same original
path/hash mapping as the capture document), and an `entries` list. Each entry
provides the original qualified `procedure` and `source_sha256`, the absolute
normalized source `path`, its `sha256` and qualified `entry`, and these facts:
`normalization: "whole_storage_rebased_v1"`,
`participation: "serial_coordinator"`, `capture_safe: true`, and
`preserves_source_order: true`. These are extractor assertions of equivalence and
capture safety; hashes bind them to source contents, rather than proving a
transformation equivalent by themselves.

Each entry's `parameters` maps normalized arguments in signature order to original
canonical resources. For example:

```json
[
  {"name": "a", "resource": "argument::a", "physical_origin": [0]},
  {"name": "weights", "resource": "settings::weights", "physical_origin": [0]},
  {"name": "coefficient", "resource": "settings::coefficient"},
  {"name": "a_lower", "resource": "argument::a", "lower_bound_dimension": 1}
]
```

Array arguments use the complete physical storage with normalized lower bound 1;
their origins contain one zero per dimension. Types, kinds, ranks, resource
identity, write access intents and parameter order are checked. Arrays use access
intent `in` or `inout`; external scalars are read-only. Lower-bound parameters are
INTEGER(4), with original declared bounds initially required to be statically
representable in that ABI. Their values are queried inside the original helper,
so an empty assumed-shape dimension correctly has `LBOUND=1` even when its
declared bound is negative. The public manifest records the normalized source
identities/hashes, which are checked again before artifacts are published.

Automatic scope planning recognizes leaf-local scalar default `INTEGER PARAMETER`
inputs in these packages after resolving their original lexical binding and
initializer with checked signed 32-bit arithmetic. Their declarations and scalar
actuals stay in the leaf query and execution clones; the outer owner does not
need to capture them. Unresolved or unsupported constants retain the whole native
span. A same-name mutable resource in another leaf still requires its own planning
proof.

Configured builds may also supply `--analysis-sources configured.json` with
`--analyze-effects` or `--form-scopes`. It keeps preprocessed analysis text separate
from original edit targets. The document has `schema_version: 1`, matching
`source_inputs`, `preserves_source_order: true`, a `configuration` object recording
the preprocessing/build configuration, and `dependencies` mapping absolute include
paths to hashes. Every original input has one `entries` record containing `source`
(the original absolute path), `path` (prepared absolute path), `sha256`, and
`line_map`. The map has one entry per prepared line: an original 1-based line
number or `null` for included/unmapped text. The extractor asserts preprocessing
equivalence; the application must be built with that recorded configuration.

The compiler checks source/include/prepared hashes and line-map order, analyzes
the configured declarations and active code, and emits edits against original
lines. Inactive branches and includes stay in the original source. Unmapped
statements and preprocessor control between sibling calls are edit boundaries.
The manifest publishes these facts for independent adapters to validate again
before applying changes. Public precision constants can resolve through multiple
module reexports; private constants remain unavailable outside their module.

Native effect summaries also report `definition_diagnostics`. Specification expressions are analyzed at procedure entry: descriptor inquiries
need no payload transfer, array-element bounds require native coherence, and
unknown specification functions are source boundaries. An array payload
read before any source write in an `INTENT(OUT)` leaf prevents captured scope
execution, including through its caller closure. Original native calls remain
available. Ordered physical-section preflight separately validates read and
preservation requirements for the complete supported span.
Native helpers with `INTENT(OUT)` arrays require an exact mapped discard and
ordered definition proof. Unsupported preservation of undefined holes or
unmapped nested definition changes remains a boundary. Numerical workers retain
their separate physical access and definition handling.

Serial source scopes support whole-array and proved rectangular bindings,
numerical leaves, mode-bearing call-only wrappers, registered hidden module
arrays (including visible reexports), exact native view sections and conservative
whole-resource effects where they remain safe. Read-only allocatable and optional
arguments can pass through supported native calls with their original descriptors
and presence. Writable allocatable formals, scalar array-element actuals,
unproved explicit dummy shapes and unknown library effects remain boundaries.
Collective call graphs are currently limited to proved direct worksharing leaves.

The compiler can also offer bounded original inline loops to its ordinary
numerical frontend. Original saved allocations, allocation guards, counters and
initialization stay in their original procedure. A worker borrows the registered
storage after fresh allocation and original-bound checks; it does not clone
persistent state. Private scalar definitions must be local to the region, and
values live across a cut remain native. One mode-bearing dispatcher per procedure
reuses approved numerical entries by region ID. Reached branch segments evaluate
their own query after the original conditions and preceding scalar updates;
native partial writes use the same coherence hooks as other source operations.
Discovery accounts for rejected attempts too, with limits of 32 regions and
256 original operations. Unknown
effects, allocation changes, unsupported synchronization and scalar state keep
their original native boundaries. The `inline_numerical_regions` manifest records
source identities, captures, completion proofs, shared computation and rejection
reasons. These candidates use source properties, without application names or
per-kernel transfer recipes.

Automatic serial source scopes require an explicit profile with costs calibrated
for the exact common runtime:

```bash
python -m compiler.offload.calibrate --output hardware.json --threads 4 \
  --precision 64 --cuda-host-cxx g++-14 --scoped-costs
python -m compiler --input application.f90 --kernel application::advance \
  --form-scopes --scope-facts captures.json --memory-model scoped --gpu-policy auto \
  --calibration-profile hardware.json --host-threads 4 --json --output-dir out/scopes
```

The optional calibration block preserves ordinary profiles and measures cold
startup separately from the warm GPU context lifecycle (setup and teardown), allocation/release, access hooks, launch
queueing, waits, and bounded planner work. It covers a recorded allocation-size
range and rejects larger GPU alternatives. Queries preserve source definition
positions and require immutable control scalars/payload arrays across each recorded
segment. Dynamic specification expressions retain original native execution.
Legacy complete-scope owners select the original span before creating a context
when calibration or queries are unavailable; a valid zero-GPU decision closes
untouched metadata before native work. Structured continuation owners instead
retain their context and execute the reached native segment with coherence hooks.
Host CPU, compiler, precision, and thread-budget compatibility are checked without
CUDA initialization; GPU identity is checked only for a candidate requiring it.

Fresh `--scoped-costs` calibration also records the optional
`scoped.transfers` extension for the exact published runtime identity. It
measures cold/cached two-slot preparation for each finite payload, physical
contiguous and one-byte strided row packing/unpacking, event record/wait/readiness
operations, and actual batch-preview work. A conservative byte bandwidth plus
per-row cost prevents thin faces from inheriting bulk-copy packing estimates.
Raw observations remain in the profile. Direct-only profiles remain valid;
neither base pinned bandwidth nor live application timings fill missing batch
costs.

When only the scoped runtime changes,
`--scoped-costs --refresh-scoped previous-hardware.json` refreshes these costs while retaining the exact base
rates and raw observations. It verifies CPU, toolchain, precision and thread
budget before measuring, then verifies the current GPU/runtime/driver through
the new scoped observations. The refreshed profile records the input's hash.
This permits ordinary timing reuse only after the emitted artifacts themselves
are also proved equivalent.

The compiler considers native workers, GPU intervals of up to four adjacent units,
and complete legal worker blocks. A bounded 16-state frontier retains distinct
coherence states; at most 128 complete alternatives are compared. The same region
algebra and physical copy plan drive execution and simulation, including retained
full-array allocation peaks, CPU mirror reads/writes, waits, and final publication.
Each GPU interval requires an estimated 20% advantage. A proven startup lower bound
avoids searching GPU schedules when even their unavoidable fixed cost exceeds
all modeled native work. No online timing or application-specific recipes are used.

`FORT_RUNTIME_TRACE=1` publishes decisions, modeled bytes, launches, waits, and
planning work. Its versioned `FORT_SCOPED evidence` JSON records show current and
required physical rectangles, preservation reads, planned copies, final exports,
and the selected intervals' cost gates. Continuation decisions add execution,
terminal projection, signed ranking and owner totals to the existing decision
record. Hypothetical terminal bytes remain separate from observed traffic. The public scope manifest maps registration
identities to source resources. Evidence is bounded to 4,096 detail records, with
explicit truncation; tracing adds no CUDA work or synchronization. Keep tracing
off during performance measurements and compare modeled copies with actual events.

Actual scoped events also append provenance version 1: a process-unique context
handle, a process-unique buffer handle where applicable, registration identity
and generation, and full SHA256 owner, procedure, segment, operation,
implementation and boundary identities. The public `runtime_provenance.records`
mapping resolves those identities to original source and generated workers.
Unbound fields say `unknown`; standalone numerical entries do not invent a
source owner. Borrowed calls restore their caller's attribution. Collective
entries update tags only in their existing coordinator. Batch callbacks remain
uninstrumented; individual batch-worker implementation attribution is unavailable.
Initial context creation precedes owner binding and reports unknown source IDs.
Execution errors retain their existing reporting and have no new attributed event.

These events describe API submissions and completed coherence commits, rather
than GPU completion timestamps. A `device_commit` can complete a read-only access
and does not by itself prove a write. The additive `fort_scope_trace_set_v1` and
`fort_scope_trace_restore_v1` APIs use fixed caller-owned state and an explicit
field mask; masked empty fields clear attribution, and unmasked fields remain
unchanged. They must not be called inside a batch-execute callback. Invalid or
stale diagnostic handles do not alter numerical errors, coherence or planning
caches. With tracing disabled, tags return before context lookup or allocation;
they still perform the environment check. No timers or synchronization are added.
The changed runtime identity requires refreshed scoped calibration before AUTO
can use its cost estimates; historical application timings do not price this change.

Fixed native helper computation is an equally omitted common term,
so estimates are not absolute application wall time. Zero-GPU estimates
conservatively include coherent CPU-worker hooks, whereas source fallback runs
original native procedures. This prototype has correctness evidence, but has not
established complete ELMM speedups; planner overhead remains part of evaluation.
Indexed actuals remain source boundaries because their array coherence and private
capture mapping must be established before evaluation. Completed GPU scopes restore
host-visible values before these original calls. Whole scalar variables and literal
actuals remain supported.

### Native source effects

Analyze native routines that do not satisfy numerical GPU lowering:

```bash
python -m compiler --input input.f90 --kernel application::advance \
  --source-file helpers.f90 --analyze-effects --json
```

Provide source after preprocessing with the application's actual configuration.
Effect budgets apply independently to each requested proof; unrelated candidate
procedures and rejected branches do not consume a later proof's budget. A formed
source span additionally checks the combined distinct call closure against the
same limits and publishes its procedure, operation, and depth counts. Cached
effects cannot bypass a deeper call path's depth limit. Effect reports contain
only the requested closure.

The compiler resolves bounded direct module calls and unambiguous generic
overloads by type/kind/rank, including reordered keywords. Call summaries retain
omitted and forwarded optional arguments, read-only allocatable descriptor
requirements, and rank-preserving unit-stride rectangular actuals. Bounds stay at
their original call under the original guards. Original native children may keep
omitted/present optional arguments and read-only allocatable formals inside
residency when their complete effects are known. Allocatable forwarding uses the
original allocation descriptor; a synthetic pointer view cannot stand in for it.
Numerical optional/allocatable formals and optional generated call wrappers
remain conservative boundaries. Source-backed free
subroutines with ordinary implicit-interface signatures can be analyzed through
direct calls; assumed-shape, optional or allocation descriptors and keyword calls
need a proven explicit external interface. Free entries remain execution
boundaries, and dynamic targets remain analysis boundaries.

The public report contains source hashes, formal/root
mappings, original guards, descriptor reads, memory reads/writes, procedure-entry
definition changes, persistent state, OpenMP directives, and boundary reasons.
Array effects are currently conservative whole-resource effects, with a separate
must-write proof for whole-array assignments and complete unit-stride sweeps.
Conditional holes, strided writes, uncertain bounds, and early exits supply no
whole-array overwrite proof. Retained source
subscripts are evidence, not physical transfer coordinates. Unknown effects,
storage lifetime, recursion, or analysis-budget exhaustion make the corresponding
summary incomplete. Mutable saved state and OpenMP directives prevent cloning.

Each procedure publishes a summary identity and the report publishes the resolved
call graph. `ordered_effects` composes repeated and nested calls onto canonical
resources while retaining original guard frames and logical view chains. A whole
formal overwrite or `INTENT(OUT)` event through a rectangular actual covers that
view, rather than the entire caller allocation. These are source facts; physical
transfer sections still require runtime descriptor checks. Composition expansion
is bounded by the source operation limit.

Summary version 9 retains a reusable local source skeleton: ordered
sequences, guarded branches, counted loops, direct call references, procedure
entry events and explicit native boundaries. Callees and unselected branches
are not eagerly expanded. Reached planning segments materialize their effects
under the existing limits; an unknown alternate arm or an oversized complete
call closure therefore need not reject a separately proved reached segment.
The complete legacy summary remains strict. Segment records identify their
original source nodes, enclosing guards, definition events and demand identity.
Copied or fabricated syntax cannot establish this authority; source text,
configuration, contract, capture, descriptor and original span changes invalidate
the corresponding proof. The graph itself does not grant GPU eligibility.

Reached serial owners also traverse original counted coordinator loops when
their headers have complete scalar/descriptor-only effects. Headers and exits
remain in the original body. Each reached child retains the existing analysis
and generation limits; a coordinator shell is not charged as one eagerly
expanded numerical candidate. The owner spans one invocation, without resetting
handles, assuming fresh host values, or reopening at a backedge. A reached
unsupported operation publishes and disables it before the original continuation
runs once. Private values consumed by a later iteration remain native, including
paths where `CYCLE` bypasses a subsequent definition. This is not cross-timestep
ownership or proof of independent iterations.

Authenticated intrinsic `IEEE_GET_HALTING_MODE`/`IEEE_SET_HALTING_MODE` calls
can remain original native operations inside reached ownership. Complete joined
teams retain every original call, caller and join. Unsaved local default
`LOGICAL` storage used exclusively by these calls remains in its original owner,
with its original representation and shared/private arrangement. It is not a
managed buffer or numerical input. Device completion precedes an environment
operation without publishing unrelated arrays. Effects retain ordered environment
reads/writes and typed native-state accesses; unknown exports, escapes, other
state uses and unproved completion remain boundaries. This grants no GPU IEEE
legality, worksharing cuts or new per-thread saved-state semantics. AUTO retains
native execution until this coordination is priced. Source summaries use
version 16 and original native completion proofs use version 6.

Source-resolved halting-mode calls reachable from the original configured entry
also require numerical-environment checks for ordinary reached arithmetic,
including parent regions outlined before a borrowed child is discovered.
A restored host trap selects the coherent original
native operation at that point. Joined numerical teams check every original
participant before making one uniform decision; the master's environment alone
is insufficient. Unsupported environments retain the original worksharing body.
This conservative restriction does not establish GPU legality or price
the additional checks for automatic placement.

Structured effect schema 6 keeps OpenMP directives inside counted loops under
the loop's original guard. Attached joins outside the loop keep their enclosing
guard; cached schema 5 structures cannot authorize these source selections.

Reached manifests use `SourceEffects.report(entry, materialize=False)` for native
effect diagnostics. Report schema 2 publishes the authenticated local skeleton,
already materialized complete source-leaf and reached-segment proofs, and observed
rejections in `materialized_rejections`; it requests no transitive closure or
additional reduction analysis. Its `complete: false`,
`closure_complete_available: false` and `closure_materialization: not_requested`
describe the unrequested whole-entry closure, not reached-scope eligibility.
Each proof retains its own identity and authority; the diagnostic call graph
covers those records only. Reduction records appear only after an explicit proof
request. The default `report(entry)` and explicit `materialize=True` retain the
legacy schema 1 complete-closure behavior and cache accounting.

The default `bounded` coordinator path requires complete bounded legacy closures
for called procedures. The experimental `--scope-execution reached` path uses
reusable child requirements and local source graphs instead. Each original
procedure supplies its own numerical regions, descriptor bounds and segment
dispatcher; repeated calls do not duplicate the transitive effect list or consume
one wrapper variant per source span. The four-variant per-procedure and
128-variant compilation limits remain unchanged.
For compatibility, rejected initial inline extraction attempts remain in the
top-level `boundaries` list with their original reasons, labeled
`kind: candidate_rejection` and `phase: inline_numerical_extraction`. A later
reached owner can still cover that source; its own `boundaries` describe actual
ownership breaks.
Structured representation version 4 accounts for source operations separately
from sequence, branch and loop containers. It retains the 256-operation limit
and a derived structural-node bound; container bookkeeping cannot exhaust the
work budget by itself. Changed representation identities invalidate dependent
cached proofs.

Reached ownership starts lazily inside the original serial invocation. Each
reached operation checks its descriptors before registration or bound queries.
Conditions and scalar assignments retain their original positions. A supported
no-argument internal child keeps its original `CONTAINS` scope, declarations and
saved state, sharing the context and resource handles by host association.
Internal children with formals can also retain their original descriptor-only
guards through a conservative native continuation. This path accepts whole
assumed-shape numeric arguments, read-only allocation descriptors and direct
source procedure arguments; it creates no numerical child worker. Reaching
payload work publishes and closes the context first. Repeated calls can use
different actual arrays because their original associations remain unchanged.
Serial counted loops with proved scalar/descriptor-only headers keep their
original bounds, strides and iterator updates. A zero-trip loop does not close
ownership; publication precedes the first reached payload operation. Bounds
that read array values or call unproved helpers still require an earlier close.
Copy-producing, optional or `INTENT(OUT)` associations and unproved entry-time
specification expressions remain boundaries. Child-local arrays that would
outlive their procedure also remain boundaries.
Original complete joined OpenMP operations retain their teams and joins.
In opt-in reached mode, supported separate worksharing DOs can use collective
numerical workers inside that same original team. One master coordinates runtime
queries and coherence, all members call the fixed-budget worker, and native
corrections retain their original worksharing and synchronization. Uniform
branches remain at their original execution points. A different actual team size
or level publishes and closes the owner before the original work proceeds.
Fortran's optional combined `END PARALLEL DO` and worksharing `END DO` retain
their implicit joins at the end of the associated original loop. A following
statement cannot become part of that team through extraction.
An unchanged joined native operation whose expanded effects exceed the budget
can compose separately bounded original worksharing units. Their conservative
access union is bounded too; calls, allocation/definition events and inferred
whole overwrites are excluded. The original team still executes once, and this
proof does not authorize an ordered child-call summary or GPU scatter writes.
If the original flat skeleton exhausts its raw operation budget, reached mode
can instead retain a complete native joined group as an explicit boundary with
separately bounded condition and worksharing units. Generic descriptor,
reduction and numerical consumers still treat that boundary conservatively.
Only an exact selection of the whole original group and its registered native
completion proof can demand those units. No member loop, opening or join grants
the same authority. Selecting an enclosing sequence or branch cannot bypass
the deferred native boundary. Units, original control depth, effects and rectangle unions
retain their existing limits; unknown calls, allocation/definition changes and
unsupported exits remain boundaries. The unchanged team executes as one native
operation, including original `NOWAIT` clauses when its final join proves
completion; no internal coherence cuts are introduced.
Plain original `SECTIONS` can remain in such a whole native joined operation
when every explicit `SECTION` contains one complete counted DO body. Each loop
iterator must be explicitly PRIVATE in the original PARALLEL, including nested
counted loops. Matching `END SECTIONS`, its optional `NOWAIT`, and the final
`END PARALLEL` remain unchanged. Deferred proofs bound each section separately
and retain original uniform branch guards. No worksharing extraction or internal
coherence cut is authorized anywhere in a team containing SECTIONS. Nested
OpenMP constructs, implicit first sections, unsupported section clauses and
incomplete bodies remain boundaries. This proves native completion and effects
only; it establishes neither cross-section independence nor GPU legality.
Exact section refinement deduplicates repeated physical rectangles and preserves
opposite faces, without inferring whole overwrites. Guarded units with scalar
bound dependencies retain whole-resource hooks because the scalar may be
undefined in an inactive branch. Private or internally changed bounds also
decline refinement. Constant bounds and supported same-resource descriptor
inquiries can refine sections; the current access emitter cannot prepare a
resource's section using another array's descriptor. Such cases also retain
conservative whole-resource communication.
For a hash-bound configured source package, a complete native joined group can
retain its original includes and preprocessor directives. The lexical owner
inserts coherence hooks around that unchanged group, which appears once in the
generated source. Unmapped included statements contribute to effects and scalar
liveness but cannot supply edit coordinates. These groups initially use
conservative whole-resource hooks and require complete definition coverage.

Original native operations can access fixed scalar numeric fields in arrays of
derived objects. Those objects use opaque host-only range registrations for
alias detection; they are never numerical device captures, and no definition
coverage is claimed for their bytes or padding. Their original allocation
guards, field expressions and indexed updates stay in place. Overlap with a
managed numerical resource closes ownership before the native operation.
In reached lexical owners, source-complete native reads of module REAL/INTEGER
allocatable arrays can use the same opaque range reservation without GPU
capture facts. The original allocation guards and native expressions remain in
place; SHAPE, LBOUND and C_LOC for the reservation execute only when allocated.
These bytes acquire no initialized or managed definition coverage. Original
read effects remain in the manifest, alongside `host_only_native_reads` proof
identities and separate `managed_resources`. Existing capture facts retain
managed coherence. Writes, lifetime changes, calls with uncertain effects and
later conflicting managed uses publish and close before the reached operation;
they cannot promote an opaque reservation or replay earlier GPU work. Automatic
placement stays native until reservation preparation has compatible offline
cost estimates.
Retained native worksharing corrections can inspect stable INTEGER metadata
fields to prepare exact read/write sections at their original reached boundary.
The inspector uses original Fortran field selectors and full-layout coordinates;
it does not assume a C layout or establish independent GPU scatter writes. It
checks coordinates and conversions, preserves holes and opposite faces, and
coalesces up to 32 rectangles per access direction. At most 1,048,576 counted
iterations or unlooped accesses are inspected per resource preparation. An
invalid coordinate or exhausted budget publishes and closes ownership before
the original native operation. Unproved footprints retain conservative whole
effects and their definition requirements. Pointer fields, dynamic fields and
indirect GPU scatter remain unsupported. This analysis has a separate summary identity
from numerical eligibility, including when it is used only to check scalar
liveness after a candidate loop.

An incompletely supported module child can use one additive original-body
`ENTRY` companion. It shares the context control and canonical resource handles;
normal callers retain the native ABI and never read absent control arguments.
The original declarations, body, module state and internal procedures remain
in their original owner. This path initially requires ordinary assumed-shape
numeric formals, no saved local storage and one canonical actual mapping per
owning invocation. Different mappings and uncertain hidden aliases close
ownership. Admitted child operations determine the managed resource subset;
unused input formals require no payload facts or device registration. Original
allocation, contiguity and alias checks still cover every array association.
An unsupported operation closes ownership when its original branch is reached.
`INTENT(OUT)` actuals are registered before the original call, even when the
child never uses them, so the unconditional entry event invalidates coverage
and later registration cannot restore old caller definitions. Nested companions
forward handles for their hidden resources without copying persistent objects.
Read-only optional numeric scalar formals retain their original `OPTIONAL`
association and `PRESENT` guards. Omitted arguments and original keyword order
are preserved; compiler controls are passed by keyword. Proved scalar-only
operations remain in the original body without capturing optional values.
Optional arrays, writable optionals, and allocation- or association-dependent
optional actuals remain boundaries, as do optional values needed by an unproved
numerical capture or planning query.
Caller contiguity guards run before a callee can create an array temporary.
Original child returns retain the outer context; only an owning return closes
it. Descriptor-only guards such as `ALLOCATED` remain at their original source
points and require no payload capture or device transfer.

Reusable child requirements follow definitions in source order. Complete
original assignments can satisfy a later conservative native read; branch joins
retain only definitions guaranteed on every path, and nested `INTENT(OUT)`
events invalidate earlier coverage. Before entering a reached child, its owner
validates remaining whole-array obligations against current runtime coverage.
That check does not download data or treat registration facts as current
freshness. Failed entry validation closes before executing the original child
once. Public records include the ordered requirements and return guarantees.
Single-line guards are evaluated once before their action's descriptor queries.
Array payloads in an original IF header are published immediately before that
header evaluates its expression once. An array-valued ELSEIF remains a boundary
until conditional publication under all preceding guards is supported.
Supported `ASSOCIATE` bodies retain their original selectors and use separate
reached operations, so a later fallback cannot repeat an earlier GPU prefix.

Full-section initialization and pointwise updates inside independent loops
(for example `a(:,:,k) = value`) use original logical bounds and checked
conformance. Shifted or nonpointwise reads that require a snapshot remain native.

A reached unknown call, unsupported operation or return publishes and closes
the context before the original continuation runs once. It cannot replay an
earlier GPU prefix; later operations remain native for that invocation. An
original procedure with statement labels remains native in reached mode;
unproved branch targets cannot bypass owner setup or enter a generated block. An
inactive, safely evaluated scalar branch does not close ownership. Array-valued
conditions require coherent host reads; unproved effects still close ownership.
Read-only imported `PROTECTED` arrays are borrowed directly without an illegal
pointer association. Public scope records expose the lexical owner, internal and
original-body module coordinators, resource mappings,
planning segments, retained state and precise closure reasons. Reached plans
carry a separate schema version. Canonical registration identities in each
owner's versioned `resource_bindings` record map runtime buffer counters to
source resources without inspecting generated workers. This experimental
path does not by itself establish whole-stage residency or application speedup.

The separate `lexical_owner_attempt` manifest record retains bounded original
member boundaries even when lexical planning finds no numerical leaves. It
distinguishes a generated source owner, a successful native AUTO decision and
an error, with original source spans and preparation counters. Legacy fallback
reasons remain separate. These diagnostics describe source planning and do not
establish application or GPU execution.
When whole-team numerical lowering is unsupported, the compiler also tries the
independently proved mixed-team implementation before the complete native
operation. Each alternative retains its own rejection reason.

Source imports resolve `USE` renames with and without `ONLY`, including public
re-exports and ambiguity checks. Imported objects retain the declaring scope of
their types even when only the object is imported or its type name is shadowed.
Private component access still uses the consumer's scope. Original leading executable OpenMP directives
are associated with their execution tree before source proofs are issued;
detached child loops still cannot borrow a complete team's authority.

Native section schema 2 refines original assignment and independent Cartesian
counted-loop accesses into at most 32 physical rectangles. Opposite faces stay
separate; small constant strides use exact unions, including their holes, and
never acquire a volume-sized overwrite proof. Runtime affine INTEGER bounds use
safely reached canonical scalar mappings and checked wider arithmetic at every
intermediate operation. Empty loops retain their original nested evaluation
guards. Diagonal/moving sections, mutable bounds and unbounded strided unions
keep conservative effects. Stable allocatable storage obtains lower bounds
from its guarded original allocation descriptor. Reached parent operations use
the original dummy descriptor for dynamic specification bounds, rather than
reevaluating a scalar that may already have changed. These proofs neither assert
allocation/presence nor define unwritten array sections.

A complete original native OpenMP region can supply a compiler-owned completion
token. The token retains the original team, worksharing directives, private
storage and final join; a serialized completion claim cannot authorize native
section hooks. Uniform branches may contribute bounded unions of possible
reads and writes, while conditional writes receive no overwrite claim. Original
private scalars and arrays stay private native state and are excluded from
owner captures. Detached work, unresolved calls, nonuniform conditions and
unsupported directives remain boundaries. A token for an entire joined group
cannot be reused for one of its unfinished child loops.

Native completion schema 7 admits original unsaved explicit-shape automatic
arrays as private native storage, including runtime extents. An outlined owner
receives their already allocated original descriptor bounds and recreates the
same private shape; it never evaluates the original specification expressions
again. Changes to their scalar inputs after procedure entry therefore cannot
change the private shape. Empty arrays retain their zero extent. This removes
the numerical scalarization limit from native storage without granting GPU
array lowering, registration, or payload capture. Pointer, allocatable, saved,
target, optional and volatile storage remain unsupported by this private proof.
Source summary version 21 invalidates earlier cached effects.

Native completion schema 4 also proves uniform reads of shared rank-one REAL
or INTEGER array elements with literal or source-backed INTEGER PARAMETER
subscripts. Fixed explicit bounds must prove the point is in range in the
original declaring scope. Original assumed-shape dummies require checked
descriptor coordinates and remain unsupported in nested, ELSEIF or compound
logical conditions. Fixed, in-range storage can appear in those conditions.
No condition value is evaluated during analysis: coherence hooks publish its
required host point before the unchanged complete original team, and normal
registration, alias and defined-coverage checks still apply. The public proof
records the canonical resource, constant index, storage and runtime
requirements. Private, THREADPRIVATE, written, allocatable, pointer, optional,
TARGET, volatile and asynchronous storage remains unsupported, as do indirect
indices, slices and uncertain storage association. This proof grants no GPU
scatter independence and does not authorize reading an inactive allocation.

`SourceEffects.worksharing_completion` supplies a separate versioned proof for
one original worksharing DO inside a complete joined team. It requires uniform
participation and completion at every DO; a `NOWAIT` anywhere in that team
prevents this split. Numerical extraction can consume the registered proof
without pretending the loop was serial. Thread-private inputs, changed source,
copied proofs and proofs for a sibling loop are rejected. Numerical legality is
proved separately. This token does not authorize native coherence hooks or
standalone execution. `worksharing_native_completion` supplies distinct native
authority for contiguous synchronized original units or complete uniform
branches. Native exact-section bounds cannot depend on another thread's private
state. Host begin precedes a team barrier; the unchanged original units finish
before host end and its publication barrier.

Worksharing proof schema 2 binds each cut to one registered ordered inventory
of its original team. Discovery and admission reuse the complete candidate
inventory under the same numerical policy, including declined candidates.
Reusing it authenticates the source, completion tokens, policy and prepared
payloads; it does not repeat whole-team completion analysis for every DO.
Private native analysis contexts issue their own inventories because their
lifetime assumptions may differ. The existing analysis limits and completion
requirements remain in force.

The reached mixed-team coordinator shares invocation-local control through
always-present local mirrors. Native entries never reference absent borrowed
control arguments, including in OpenMP SHARED clauses. Failed preflight closes
and executes only the current original operation; failures after numerical work
cannot replay it. The public manifest identifies the original team, numerical
participation proofs, native completion proofs and any runtime inspector.
The existing source and generation limits still apply. Expanded native effects
are bounded per reached operation rather than summed across the whole owner;
the manifest reports both the reached-proof bound and the expanded total.
Automatic continuation through these teams remains native without compatible
calibration for the complete original-team coordination protocol.

An unsaved reusable source helper whose entire body is one supported complete
numerical OpenMP group can retain that team when the original environment needs
participant checks. Each call keeps its original arguments and borrowed handles,
so the same helper can serve different arrays and dummy bounds. All original
workers vote before the master prepares or queries numerical work. A native
choice uses an independent complete source-effects plan: validation and host
begin precede the unchanged body, and host end follows the team join. This worker
never closes its borrowed context; failed native validation is a checked hard
error. Original INTENT(OUT) definition loss still occurs at child entry. Whole
return coverage requires the same exact group's independent source must-write
proof and validated numerical equivalence; partial or conditional writes gain no
whole-array credit.

Numerical outlining uses a distinct registered completion proof when the
original joined loop calls bounded source-backed PURE helpers. The complete
transitive helper closure must finish synchronously, without unknown calls,
I/O, OpenMP work or allocation changes; output arguments must refer to original
private scalar or fixed-array storage. This proof grants neither native memory
hook authority nor GPU legality. Numerical lowering still proves definitions,
aliases, supported computation and numerical semantics before generation.

Reduction source proofs are separate from generated execution support. They
retain the original intrinsic form, argument order, MASK/DIM, empty identity,
exact input sections and a single scalar publication at the original statement.
The first exact parallel subset is integer MINVAL/MAXVAL and default-logical
ALL/ANY. An IEEE_IS_NAN predicate additionally requires the actual intrinsic
module identity and a verified native classification/exception policy. Scalar
MIN/MAX assignments are not mistaken for array reductions. Real extrema need
their own NaN/signed-zero/native exception contract; a guarded implementation
must publish the exact input and run the original reduction statement once on
exceptional data, without replaying its owner. SUM remains native without an
authorized original numerical reassociation contract. A source proof alone
neither emits a GPU reduction nor supplies calibration for automatic placement.
An authenticated original OpenMP `reduction(+:x)` is a separate source-backed
contract: OpenMP leaves the combination order unspecified, so it can permit a
parallel real sum without granting that permission to a serial recurrence or
an intrinsic SUM statement. Its proof must retain the original value of `x`,
private initialization, contribution computations and the original completion
point. Numerical field tolerances and supported IEEE behavior still apply;
task/inscan variants and unproved intermediate accumulator consumers remain
boundaries. The lazy `SourceEffects.openmp_reduction(procedure, original_nodes)`
API proves one explicitly requested complete original joined group with a
single ordinary local REAL(4/8) `+` accumulator and a bounded rectangular loop
nest. Per-item private scalar definitions, contribution math and complete source
effects are checked; uncertain aliases, array writes, calls and exits reject
the proof. INTEGER(4) sums additionally need a proof that every partial sum is
representable. Public reports include only requested proof records and report
`execution_supported: false`: generated OpenMP reduction execution and its
calibration remain unavailable. Ordinary native completion tokens still reject
reduction clauses; this separate source contract does not grant them authority.
See [OpenMP reduction scoping](https://www.openmp.org/spec-html/5.2/openmpsu50.html)
and [the reduction clause](https://www.openmp.org/spec-html/5.2/openmpsu52.html).
`SourceEffects.reduction_candidates` lazily scans only an authenticated reached
selection or one bounded local source skeleton, without expanding callees. Its
public records distinguish `source_analysis_available` from
`execution_supported`; generated reduction execution is currently unavailable.
The existing numerical runtime supports INTEGER(4), while INTEGER(8) and
default LOGICAL(4) require their own representation-preserving execution ABI.
The runtime's one-byte logical buffers cannot stand in for original default
Fortran logical storage. Integer extrema additionally retain a nonempty tag:
the empty MAXVAL result is `-HUGE`, which is not a neutral accumulator for the
minimum representable signed integer. A worker must initialize from the first
selected value and apply the empty result only when no element was selected.

Successfully completed source closures and local skeletons are reused through a bounded memory cache.
`--summary-cache DIRECTORY` also enables immutable keyed disk records. Keys include
source and include hashes, prepared source/configuration/line-map identity, explicit
contracts, capture authorizations, analysis version and budgets. Changed facts
invalidate affected proofs; private structured fragments cannot borrow complete
procedure summaries. Imports recheck graph identities, reachability, depth and
combined operation/procedure budgets. Cached skeletons are rebound to original
parser nodes and compared against current source authority before use. Public records are copied values, and cache
corruption or write failure causes a miss. The cache retains at most 32 records in
memory, each capped at 8 MiB; disk-directory cleanup belongs to its caller.
`summary_cache` reports construction-local cache statistics. No application data,
runtime placement or allocation addresses are cached by this mechanism.

Original syntax is authenticated against a bounded structural snapshot rather
than repeatedly formatting every configured module. An active `ScopeBuilder`
then seals the original syntax and source spans for its compilation session.
Original parser attributes, child lists, statement labels and source spans reject
writes; pre-existing dictionary/list aliases and serialization state cannot
modify the originals. Repeated demands authenticate original node identities
without rescanning immutable syntax or its source-location inventory. Copies
create mutable private projections and cannot acquire original-source authority.
Files, bindings, namespaces, contracts and budgets are still checked separately.
Standalone mutable `SourceEffects` analyses retain fresh structural checks, with
comparisons shared only within one immediate demand. Unknown or cyclic parser
state fails closed. This seals syntax/provenance; it does not yet make the whole
namespace and binding graph immutable.

A synchronous recursive summary shares syntax comparisons and one frozen
canonical authority only while its original parser roots remain intact. Its
recursive stamping, native-section queries and cache access reuse that immutable
authority within the demand. Fresh file, syntax and complete-authority checks
precede closure storage or adoption; failed checks discard candidate
section-cache changes. Local cache
hits still validate current authority on each call. Copied source-function
projections use their own checks. The bounded invocation-environment traversal
similarly shares one local checkpoint, then rechecks the original syntax and
every visited procedure before retaining its Boolean result. Neither operation
retains a successful syntax checkpoint for later calls or other builders.

Numerical procedure classification reuses its locally issued source summary
for capture origins, native effects and team participation. The summary stays
inside that synchronous classification; standalone queries and recursive child
demands still validate independently. Fresh original-source, complete-authority
and compiler-configuration checks precede adoption of generated artifacts.
These checks are hard failures rather than numerical-support declines, and no
successful validation is retained for the next classification.

Generated procedure variants have a separate bounded registry. The public
`implementation_variants` manifest retains the original native entry and records summary
identities, interfaces, requirements and shared numerical artifacts. Workers
carry runtime placement modes; shapes and CPU/GPU partitions do not create new
procedure copies. Generation is capped at four variants per procedure and 128
per compilation. Repeated leaves reuse their artifacts, and a rejected source
candidate restores its generation budget before scanning later candidates.
Budget exhaustion produces an explicit native boundary.

The additive `fort_scope_view_v1` interface borrows a rectangular view of an
existing canonical resource. It contains its registration generation, physical
origin and extent, and logical dummy lower bounds. `fort_scope_view_get_v1`
checks the descriptor without initializing CUDA or allocating another buffer.
Numerical workers retain full-root pitches; normalized numerical dummy bounds
remain one, with original logical origins supplied separately where required.
Read-only aliases merge exact access unions, while writable formal views must
be disjoint. Partial `INTENT(OUT)` events discard only the mapped view, preserving
unrelated initialized or device-current sections. Invalid preflight can retain
native execution; failures after numerical work starts poison the context and
cannot replay the source span. View subchain batching remains unavailable until
its complete window mapping is proved. Source rectangular actuals currently
require a straight-line owner; reached structured owners retain a native boundary
until per-segment view descriptor preflight is supported.

The additive `fort_scope_view_v2` interface also represents rank-reduced
rectangles such as `field(:,plane,:)`. Its origins retain the canonical root
rank, while a checked axis map connects the child's logical extents and lower
bounds to the original root pitches. Source-generated borrowed workers use this
interface; the version 1 runtime and companion interface remain available.
Nested supported call-only wrappers compose axis maps without creating packed
allocations. Fixed coordinates and every retained extent are checked at the
original call, before numerical work. Native effects and partial `INTENT(OUT)`
events project to the exact root plane. Vector subscripts and nonunit-stride
actual arguments remain boundaries; they require an additional physical mapping
proof.

Original conformant array assignments are also numerical candidates in scoped
source generation. Broadcasts, scaling, accumulation and boundary planes are
normalized to ordinal loops with separate original logical lower bounds.
Constant positive and negative section strides are supported inside these
operations; strided actual arguments through a call remain unsupported.
Checked allocation, bound arithmetic, containment and shape-conformance guards
run at the original reached operation, in order. A failed guard executes the
original native assignment with its proved coherence effects; an unproved native
fallback retains an ownership boundary. Ordinary dependence
proofs reject shifted or reversed self-assignments needing a cross-iteration RHS
snapshot; those assignments retain native Fortran snapshot semantics.

Fixed numeric leaves of scalar, nonpolymorphic derived objects can be captured
without copying their owning objects. Supported lexical `ASSOCIATE` aliases
resolve to the same original resources. Unsupported dynamic siblings do not
invalidate a fixed field, while pointers, optional object associations and
array-valued object selectors retain explicit boundaries. Public inline records
identify `operation_kind`, canonical fields, original source selections, semantic
guards and native fallback requirements. New real array math preserves the
reached floating-point environment check and rejects unproved observable
exception behavior or real multi-operand extrema ordering.

Source-call effect mapping also accepts original whole scalar derived objects
with `INTENT(IN)` or `INTENT(INOUT)`, when both declarations resolve to the same
original declaring type. Renamed imports and repeated reexports preserve that
identity; identical type names in different modules do not. The bounded proof
checks the entire type, including fixed numeric fields, fixed arrays and nested
scalar types. Polymorphic, extended, parameterized, finalizable, type-bound or
descriptor-bearing types and uncertain object associations remain boundaries.
Original calls, keyword order, objects and declarations stay unchanged. Public
`source_object` records authorize only source effect mapping of component
suffixes; they do not authorize object copies, a numerical ABI, capture, cloning
or native continuation. Whole-object assignments and reads still require their
own assignment/operator proof, and object mapping proofs are rebuilt rather
than imported from cached summaries.

Exact intrinsic `IEEE_ARITHMETIC::IEEE_IS_NAN` imports, including source-proved
renames and reexports, can describe original native predicate reads. Public
`native_predicate_requirements` retain the original expression, actual ordering
and surrounding guards; ordinary read effects still require their own storage
proof. This metadata does not evaluate the predicate, authorize numerical
cloning or GPU execution, or admit exception observers. These source tokens
are reissued from original syntax rather than cached as authority.
Configured intrinsic exception-flag observers also prevent ordinary reached
arithmetic loops from being outlined: closing a device context cannot recreate
CPU exception flags. Same-spelling user procedures do not establish intrinsic
observer authority. Complete native OpenMP operations retain every unit's
predicate requirements without introducing a generated predicate operation.

Source-backed scalar functions have a separate native read-effect proof. A
complete bounded closure must be explicitly `PURE` or implicitly pure
`ELEMENTAL`, with required fixed numeric or logical `INTENT(IN)` arguments,
a scalar result and read-only transitive effects. Version 2 also accepts bounded
explicit-shape array arguments and private arrays. Fixed bounds and extents
must fit the existing INTEGER ABI and analysis budget. Whole fixed actuals
must conform in every extent; their original lower bounds remain distinct.
Supported constant in-bounds rectangular sections retain the caller's root
rank and mapping. Uncertain bounds, array-expression actuals and element sequence
association remain unsupported, as do descriptor-bearing or uncertain array
associations. Private scratch reads stay private when helper effects are mapped
back to their caller. Exact intrinsic `DOT_PRODUCT` expressions contribute
ordinary read effects; shadowed or unresolved names retain source boundaries.
Imports, renames, kind-specific generics and internal lexical captures resolve
against original source. Purity does not authorize unknown calls, hidden storage
lifetimes, OpenMP or opaque library effects. Public
`native_function_requirements` preserve original expressions, keywords, source
identities and guards. Functions stay outside the public subroutine/structured
entry registry; an isolated private view supplies effects without a synthetic
call or generated reevaluation. These records deny cloning, GPU lowering and
atomic native callbacks, and are rebuilt rather than restored from persistent
summary caches. Changed source, resolution maps or kind state invalidate them.

Source summary version 20 carries child `capture_lifetime_requirements` through
original actual mappings and guards, even when the child remains incomplete.
Distinct view chains retain their obligations; forwarding does not upgrade an
authorization or turn partial raw operations into complete ordered effects.
Selected descriptor capabilities also participate in summary/cache identity;
an admitted private child cannot populate an ordinary summary cache entry.
The source-function and summary versions also invalidate older cached function
mapping semantics. Array-function read effects do not authorize numerical
lowering, reassociation, IEEE behavior or initialized payloads.

The source-only `prove_descriptor_window` API authenticates descriptor/no-escape
effects within an exact original interval, separately from entry definitions and
excluded prefix/tail operations. Numeric allocatable formals and counted headers
are supported. Unknown calls, allocation replacement and uncertain associations
inside the interval reject. `segment_summary(..., descriptor_window=proof)` may
consume that registered token in an isolated projection; the ordinary
whole-procedure descriptor proof remains unchanged.
Version 2 also checks original fixed numeric object schemas and exact source-call
object mappings. Existing registered whole-group completion tokens may
authenticate original OpenMP directives; the complete descriptor/escape scan
still checks every group statement. Neither object mappings nor completion
tokens establish GPU legality or initialized payloads.
Version 5 authenticates resolution at original semantic units: each assignment,
condition, counted-loop header and call retains the existing lookup and depth
limits. All operands of one unit share its lookup budget. Declaration
dependencies and requested module-storage visibility retain original namespace
authority. Syntax and lookup work also spend the existing aggregate source-scan
budget; public diagnostics report both counts and the configured limits.
One interval proof reuses successful authentication of each exact consulted
namespace across its semantic units. It still charges every syntax visit and
lookup transition. After the last charged operation, fresh namespace, source,
routine and complete authority checks gate publication of the proof.
Whole-procedure descriptor closures similarly share one source checkpoint and
canonical authority across their bounded recursive graph walk. New graphs and
cache writes remain local until fresh exit validation succeeds. Each subsequent
query authenticates again; private projections retain their separate authority.
Every reached native child still receives a complete descriptor/escape scan,
and registered readonly functions retain their complete body proof. Unrelated
internal siblings and declaration/import spellings are not executable lookup
units. These checks do not establish reached-path effects or entry definitions.
Version 4 accepts explicit `module_roots` for original numeric module
allocations. Their authority stays inside the exact private projection and its
checked call closure, with separate cache identities; ordinary summaries and
global module authorizations remain unchanged. Public or renamed module storage
still requires original visibility and fresh descriptor guards. TARGET,
THREADPRIVATE, uncertain associations and selected allocation changes reject.
Depth limits count source controls and semantic expressions rather than synthetic
sequence and argument-list containers. Exact original joined groups may use
registered native or numerical completion proofs for descriptor/noescape checks;
the checker retains every original statement and keeps its proof registries
private. Original IEEE environment calls require their own registered source
proof. These descriptor checks grant no GPU execution or numerical semantics.

The separate `prove_native_prefix` API can establish complete effects of an
unchanged native call before any context exists. A detached, bounded analysis
may resolve missing lifetime obligations for original read-only numeric module
allocations. It neither grants device captures nor changes the ordinary source
summary or its cache. A distinct completion contract admits complete original
joined `PARALLEL DO` groups with supported scalar private/reduction clauses;
the native compiler executes every original reduction. No reduction lowering,
reassociation or active-owner coherence is authorized.

`prove_reached_window` version 16 transfers separately supplied entry facts through
these prefix and descriptor proofs to an exact contiguous window, at top level
or in the first traversal of one top-level serial counted loop. Branches meet
conservatively and definitions retain original ordering. Hidden read-only native
prefix allocations require no GPU capture; array dependencies in the selected
window still do. Its conditional lifecycle is dormant → active → spent, with a
mandatory close before the excluded tail or any enclosing backedge. Entry facts,
prior nonescape, allocation presence, layout and alias guards require independent
admission. Within the selected window, exact original assignments and
single-assignment affine loops can establish bounded partial definitions. Later
reads must be covered, conditional paths meet conservatively, and scalar changes
invalidate rectangles that used the old bound. These return rectangles are
conditional diagnostics; they never seed registration or whole-return facts.
Runtime workers still validate actual physical coverage before numerical work.
These APIs do not create an owner or authorize execution. For the ordinary
complete-window proof, partial definitions before activation, deeper or
conditional activation, direct OpenMP cuts, unknown effects, local allocation
ownership and prefix allocation replacement remain limitations. No address/shape
freshness or owner reopening is inferred.

The compiler's `plan_reached_window` uses the same transfer to identify exact
close-before-original operations within a selected window. The ordinary
complete-window proof API retains its complete prefix requirement. Planned
windows additionally track dormant prefix eligibility at original source points.
Plain numeric whole-name copies and single-object explicit-shape `ALLOCATE` or
plain `DEALLOCATE` may precede activation only when their destination capture
explicitly has `lifetime: "selected_window"` and `initialized: "none"`.
The original statements execute once; fresh guards acquire the actual reached
descriptor. A conforming copy can retain destination bounds, so RHS bounds are
never substituted for them. Allocation alone defines no payload. Whole copies
require current whole RHS coverage from invocation-entry assertions or earlier
source definitions of a selected-window-none capture. A selected-window-whole
assertion cannot authorize these earlier prefix reads. Supplied facts and their
hashes remain unchanged; no activation assertion is unioned over a definition
kill. Allocation options, pointer/target associations and unsupported shapes
remain native. Prefix support declines disable eligibility before the complete
original operation. A supported original single-line `IF` condition retains its
evaluation once: a registered block rewrite disables eligibility inside its
true branch before the original unsupported action executes once. The proof
binds the original parent span, condition and action identities and hashes.
Unsupported conditions, labels, preprocessor boundaries and potentially shared
semicolon spans retain whole-conditional disabling. Disabled eligibility is
absorbing for that invocation, including `CYCLE`, `EXIT`, `RETURN` and later loop traversals. Only
eligible branches contribute activation coverage. Source/token and budget
failures remain hard; incomplete generic native-prefix proofs are not converted
into support declines.

Active planned owners can likewise retain residency across an original
single-line `IF` whose unsupported action needs closure. The compiler authenticates
the complete parent span, condition and sole action. For a supported condition
with no managed payload reads, it evaluates the condition once and publishes and
closes only in the true branch, immediately before the original action. The
false path retains its device state and definitions. Closed paths remain native
for the rest of the owner and never reopen or replay earlier work. Conditions
that read managed payloads, unsafe shared source spans and forced whole-parent
cuts retain conservative closure before the complete original `IF`.

Before expanding a dormant native call's effects, a bounded original-source
support check identifies native I/O and OpenMP team shapes outside the supported
before-owner grammar. It resolves and authenticates the original call chain;
it does not prove effects or grant capture authority. Reaching such a call
disables eligibility before the unchanged call, while an inactive caller branch
does not reject the later owner. Source, binding, resolution and budget errors
remain hard. Active owners retain their ordinary coherence and closure checks.

Before a root owner exists, a complete original condition may read uncaptured
fixed numeric or logical metadata on the host. The source proof authenticates
its original storage and permits only read effects; uncertain associations,
allocation lifetimes and side effects remain boundaries. The condition and its
mutable inputs execute at their original point on every invocation. This grants
no capture, initialized payload, descriptor stability or device freshness.
Managed inputs still need their ordered definition coverage. Active owners and
mapped child conditions retain their existing coherence requirements.

Planned windows can also transfer exact interior definitions through a complete
original joined OpenMP prefix. Native-definition version 3 authenticates the
whole team and its unconditional single-assignment perfect nests. Every unit's
reads require coverage at group entry, including with `NOWAIT`; writes become
available only after the final join. May-write unions are never definitions.
Original worksharing and synchronization remain unchanged. Invocation-entry
facts and source-established selected-window-none coverage may justify these
prefix reads; a selected-window-whole assertion cannot do so.

`prove_native_definition_state` exposes a separate native-only logical transfer
for exact requested output bindings. It shares the original access walker but
does not require physical GPU layouts for native inputs such as explicit-shape
coefficient arrays. It retains all array, scalar, index and descriptor reads;
uncovered array reads withhold output-definition credit. An authenticated
PARAMETER call association can establish coefficient value-definedness, while
the validity of the original dynamic index remains a separate precondition.
Joined units use the same entry snapshot, and branch joins retain only common
definitions. This token grants no physical mapping, allocation lifetime,
activation, or execution authority. Native coherence continues to require its
existing physical-section proofs.

`prove_native_preservation` filters existing definition coverage through an
unchanged native prefix. Complete ordinary reads and payload writes preserve
already defined points; they grant no new coverage. Original OUT associations,
possible hazardous aliases and scalar-bound changes remove affected facts.
Descriptor changes and unknown effects clear coverage. Scalar and descriptor
dependencies retain their exact original bindings. This dormant-state proof
does not authorize active host/device coherence or allocation lifetime.

Partial activation uses registered typed source geometry, distinct from external
literal physical `sections` facts. First registration maps that geometry through
the guarded original descriptor before a runtime handle exists, retaining
checked coordinates, empty-domain guards and original lower bounds. Scalars
used by this geometry must remain unchanged along active selected paths because
registration is lazy. The original caller registers required partial resources
before borrowed dummy association; borrowed workers require the existing handle
and never reinterpret root geometry. `INTENT(OUT)` still kills prior definitions.
Coverage comparisons recognize bounded constant translations of logical bounds
and equivalent unit-stride empty-domain conditions across mapped calls. This
proof does not rewrite runtime indices or relax descriptor and overflow checks;
other symbolic expressions retain conservative coverage requirements.
Unsupported nonlexical or collective registration paths retain explicit native
boundaries. No runtime ABI or memory budget changes are required.

Supported branch paths may keep ownership while another path publishes and closes before
an unsupported call or allocation change. Closure is absorbing: that path
executes its original native suffix once, without descriptor registration,
planning queries or reopening. Return coverage after a possible close is
explicitly unknown; separately reported active-path coverage is conditional and
cannot establish an unconditional child definition summary. Failed speculative
units discard their demands while preserving consumed analysis work. Source
authority, IEEE checks and aggregate analysis budgets remain independent gates.
Joined close and prefix-disable points identify the entire ordered original
selection, including its opening directive and final join. Emission cannot cut
or validate only the first node of a team. Discovery version 6 retains these
selections and completion identities in its decline diagnostics.
An explicitly supplied `NumericalPreparationProvider` can extend a planned
mapped child with exact physical effects from its original numerical group.
The same route supports original serial pointwise counted loops containing
array-element assignments with iterator or literal indices, without OpenMP
directives. Calls, indirect indexing and scalar writes retain the ordinary
native path. Exact registered numerical effects avoid demanding
undefined output values solely to prepare a conditional native fallback. A
failed numerical-environment or geometry guard still publishes and closes
before the unchanged original loop executes once; numerical effects never
authorize native memory hooks.
Preparation is source-only: it neither admits captures nor publishes generated
artifacts. The separate registered effect proof maps actual numerical indices
back to original logical coordinates; preservation uploads are not reads or
definitions. Every unit in a joined group checks reads against the same entry
snapshot. Only exact unconditional writes contribute definitions after the join.
Descriptor, lifetime, allocation and alias checks remain independent. The
ordinary native child summary and complete-window proof remain unchanged.
Conditional tile geometry is checked after original allocation guards and before
the numerical unit's registration, definition events or queries. An outer owner
may already have registered independently proved original descriptors; that
registration establishes no tile-derived definitions. A failed guard publishes and closes
before the complete original group, which executes once; the owner cannot reopen.
Public records identify the preparation, physical effects and typed preconditions;
these conditional return effects never become unconditional native contracts.
For an exact prepared numerical group, both ordinary lexical owners and reached
mapped children can retain the original team and its complete join. Ordinary
owners issue their preparation/effect pair from the original source, without
requiring a reached-window plan. The bounded local registry revalidates the exact
selection, source, bindings and compiler policy at use and publication, and rolls
back with failed generation candidates. It grants no storage or definition facts;
a bound reached plan remains authoritative for its own conditional regions.
Preparation fingerprints each configured line map as bounded provenance data,
retaining its exact length and contents without charging every line number as
an AST proof operation. Each demand recomputes the fingerprint; changed maps,
configuration, source paths or dependencies still invalidate the prepared proof.
Source, scope and numerical validation retain their existing work limits.
Invocation-wide IEEE-state queries visit original lexical statements and
single-line IF actions to find calls. Expression payloads retain their separate
source and numerical checks without adding work to this call-only search.
Each query authenticates configured files, source spans and module syntax once
at entry, then checks each visited routine against that immediate checkpoint.
Before every Boolean result, it revalidates consulted namespaces and takes a
fresh global checkpoint to check all visited routines again. This avoids
repeating whole-input authentication for every callee; public source demands
remain fresh, and no successful authentication is cached across queries.
Original liveness contexts reuse prepared-span aggregates only within one query.
The index retains expression descendants, unmapped include coordinates and
missing-span rejection, while avoiding repeated subtree walks for enclosing
branches and loop backedges. Cached lookups still spend the existing local and
parent work budgets. Every new demand rebuilds the index; source authority,
capture definitions and execution eligibility remain separate checks.
Numerical preparation uses that fresh structured-source authority to check its
exact original parser selection, rather than repeatedly scanning unrelated
owner expressions for membership. String graph IDs, copied parser nodes and
foreign selections do not grant numerical preparation authority. Selected
expressions, helpers, complete source provenance and final publication checks
still spend their existing budgets and revalidate every demand.
Indexed fields of arrays of derived objects remain outside numerical lowering.
Their typed support decline reauthenticates the selected source, consulted
namespaces and registry roles after budget callbacks, then tries the original
native proof. This can preserve a reached path with a conservative native
boundary; it grants no GPU metadata capture. Source, token, budget and other
component failures remain fatal. Explicit native metadata proofs are unchanged.

Supported groups are a combined `PARALLEL DO`, or one flat `PARALLEL` containing
up to 32 complete numerical worksharing `DO` units. The latter preserves every
original worksharing directive, barrier and `END DO NOWAIT` in the native
alternative. Generated units execute in their proved order. Whole-group read
coverage still uses the original entry snapshot; a final join cannot justify
reading an undefined intermediate or authorize an individual NOWAIT cut.
Private state carried between units, native siblings and nested teams retain
their existing conservative boundaries. The emitter preserves supported
`PRIVATE`, `SHARED` and `DEFAULT` clauses on the team, and `COLLAPSE` and
`SCHEDULE` clauses on the original worksharing loop. Every actual participant
checks its numerical environment before the master evaluates guarded bounds,
registers this unit or constructs its query. No separate sampling team is used.
The native ABI borrows no absent companion controls; always-present local
mirrors carry coordination state inside the original team.
An unsupported team, numerical environment, geometry or native placement
publishes and closes before that original loop executes once. After successful
preflight, the existing collective worker may handle pre-execution resource
failure with its checked generated CPU implementation and coherent commit.
Execution or completion failure cannot replay work. This consumer grants no
native-effect hooks from numerical footprints. Its versioned
version 2 `combined_numerical_team` manifest records group kind and source-unit
count, completion, source partition,
preparation and physical-effect identities, participant requirements and failure
policies. Missing applicable collective calibration still selects native AUTO.
An original direct child call can carry its own registered conditional plan.
The separate `prove_native_call_coordinates` version 3 API authenticates whole
ordinary assumed-shape associations, original caller lower origins, read-only
scalar mappings and every `INTENT(OUT)` entry event, including unused formals.
After those entry events, an existing unconditional whole definition subsumes
redundant partial definitions before mapping coverage into a child. This does
not supply missing scalar mappings or widen partial-only coverage.
Read-only real/logical scalar fields require their original fixed-component
proof; integer coordinate fields and descriptor-bearing fields remain native.
Field selectors use original type authority rather than the caller's `USE`
namespace, while original syntax and aggregate work remain charged.
Shared constant declaration bounds are authenticated once per derivation, with
charged reuse under the existing work limits. Runtime descriptor and scalar
bounds retain their original dependencies and are not cached this way.
Source authentication and rectangle transfer each retain their own bounded
work phase, and both charge the same parent aggregate work limit.
Ordinary coordinate mappings do not inspect the child's executable closure or
grant storage, initialized payload, completion or GPU authority. The reached transfer visits the child at
that original call, maps ordered partial definitions and shares aggregate work
limits. Conditional active return coverage remains conditional; a child close
disables the caller's owner before either original native suffix continues.
Unsupported associations still close at the original call. Allocation-changing
callee formals, recursion, uncertain aliases and unproved saved storage remain
boundaries; immutable array arguments require their existing independent proof.

Version 3 also retains an original scalar-element association such as
`call child(a, weights(selector))`, without replacing it with an early scalar
snapshot. Initially this requires a rank-one, fixed, declaration-initialized
module REAL table, an INTEGER name or literal index, and a required same-kind
scalar `INTENT(IN)` formal. Pointer, TARGET, optional, VALUE, volatile and
asynchronous associations remain unsupported. A distinct `scalar_element` role
stays outside array-resource mappings, scalar renaming and coordinate bounds.
The selected element remains associated with the original child dummy even if
the child subsequently changes the selector.

A separate bounded source witness checks the complete transitive child closure:
no other reference to the backing table and only unchanged scalar IN forwarding.
It supplies no effects, completion, initialization or GPU authority. Unknown
calls, backing writes, write-then-restore, writable forwarding and unproved
escapes close ownership before the original call. At that original guarded
point, an undefined BYTE reservation checks the full table address range against
managed captures. The reservation reads no payload and allocates no device
buffer. Overlap publishes and closes before the unchanged call; distinct source
root names do not establish physical disjointness. Checked original bounds guard
the index before borrowed associations. Neither check moves above the original
condition. Public `original_scalar_elements`, `scalar_element_closure` and the
opaque alias-reservation role distinguish these obligations from managed data.
Source summary version 20, structured-effect version 7 and reached-window version
14 invalidate previous association representations and dependent cache entries.

Guarded alternatives require a typed source property and proof that the emitted
guard establishes it for the entire optimized interval. An entry value or version
check alone cannot prove future immutability; an exit equality check misses a
write followed by restoration. A missing proof may require complete mutation
tracking or an ownership close at the unknown effect. Decline text is diagnostic
input, not executable proof. Keep guard reads at their original safe evaluation
points, including allocation, optional-argument and collective synchronization
requirements. Before numerical execution, failure may select the original native
path. After a device prefix, failure must publish and continue the original
suffix once; it must never replay the prefix. Legal guarded execution still
requires an independently valid cost estimate for automatic placement.

Available original scalar `ASSOCIATE` scopes participate in the same lexical
resolution authentication as their enclosing procedures. Numerical preparation
can retain those aliases without bypassing original binding, parent or source
checks. Unsupported selectors remain local boundaries.

The internal `ReachedWindowOwner` emitter consumes an exact registered window
proof together with the same independently admitted capture document. It keeps
the original native prefix and excluded tail in place, declares control at
procedure entry, and activates once at the selected source point. The attempt
is spent before prefix eligibility, allocation, layout, alias or placement guards.
An invocation-local eligibility flag gates all later associations, registration
and queries; the serial-caller guard cannot overwrite a prefix disable. It closes before
the excluded tail or an enclosing backedge, including resource failures; later
traversals remain native. Source-position definition coverage is private to the
window-bound builder and does not rewrite entry facts or whole-procedure
descriptor proofs. Per-unit native summaries reissue descriptor tokens in their
own analysis, and managed captures require exact original bindings. Numerical
lowering and existing native coherence/failure checks remain separate gates.
Reached-window proof version 16 also accepts bounded literal physical entry
`sections` when an `expected_descriptor` supplies the original lower bounds and
extents. Physical coordinates remain zero-based and half-open; the proof converts
each rectangle separately to logical inclusive coordinates. Empty rectangles
define nothing, and an uncovered halo or gap does not become a whole definition.
The original descriptor expectation is checked after allocation/layout guards
and before registration. These are independently admitted entry definitions,
not source-proven overwrites; the manifest records that distinction in
`entry_section_facts`. Original `INTENT(OUT)` events still kill them, and
selected-position assertions cannot authorize earlier native-prefix reads.
Partial caller-to-child mapping with unproved descriptor origins remains a
conservative boundary. This extension does not authorize repeated activation.
Reached child calls preserve every original actual association, while only already
managed IN/INOUT actuals and visible managed module resources inherit definition
and coherence state. Unused array actuals require no new capture facts or handles.
The first reached payload or descriptor effect on an unproved array publishes
managed outputs and closes before the unchanged original operation. An unmanaged
INTENT(OUT) actual closes at the caller because its definition event occurs on
entry; an empty managed projection also closes there. Managed nested OUT events,
caller allocation/bounds/contiguity guards, original alias checks and original
procedure storage remain authoritative. Public child plans identify managed and
unmanaged associations; reached proof version 16 invalidates earlier authority.
Supported child coordinators reuse the same context, handles and enabled flag.
Caller descriptors are registered under their original allocation/layout guards
before child argument association can rebase bounds. The emitted child retains
its original logical coordinates and declarations. Whole joined native groups
require their separate original completion proof and remain indivisible; a
coordinate proof alone cannot authorize a team or a cut inside one.
Rolling back a rejected scope also discards newly cached successful child
coordinators with their numerical proof pairs. Retained coordinators keep their
original registry identities; rebuilding a discarded child does not refund the
procedure, inline-attempt or operation search limits.
In opt-in serial `--scope-execution reached` mode, incomplete whole-entry
descriptor stability for original allocatable formals, or explicit selected-window
module captures, triggers compiler-owned window discovery before inline
preparation. It considers at most four adjacent
original units at entry or inside one top-level counted loop, with at most 16
attempts. A selected window must admit two numerical invocations. Discovery
version 4 also consumes planned closures, so an unsupported branch can finish
ownership without rejecting a supported sibling. Staged emission must match the
exact original close points before adoption. Failed attempts run in isolated generation
transactions: source objects and registered proof identities remain original,
while rejected artifacts, variants, coordinators, edits and analysis caches do
not leak into existing fallback generation. The manifest's additive
`reached_window_discovery` record reports original selections, limits, proof
identities and rejection reasons. Rejected windows without enough numerical
invocations retain bounded exact original prefix-disable and active-close reasons
in `source_declines`, including source kinds, node identities and spans. AUTO with unavailable estimates retains the
original native path. Module captures consume separate descriptor-window proofs
at every reached consumer, without globally blessing their allocations.
Discovery neither invents capture facts nor proves whole-stage residency.
Original descriptor and partial-definition proofs remain separate gates.

Reached owners now hand one checked ordered execution plan from admission to
placement, emission and discovery. Its owned operation graph is immutable,
including native effects, source edits, control guards, resource associations
and child owners. Builders, proof registries and emission results are separate
state. Discovery version 7 counts numerical invocations from this same plan;
it no longer reconstructs an independent call inventory. The additive
`checked_execution_plan` schema-version-1 record binds source/configuration and
capture identities, ordered operations and canonical resources. Its JSON is a
diagnostic view, not importable proof authority. Missing prices still select
native; a checked plan alone grants no automatic GPU placement or transfer saving.
The complete diagnostic `identity` includes the issuing compilation's proof
tokens. A separate `execution_identity` fingerprints the admitted executable
semantics for calibration reuse across independent compilations. It binds the
same source, configuration, captures and immutable operation graph without
mistaking process-local token identities for changes in execution. Registered
proofs retain their exact in-process validation in both paths.
Calibration reuse remains limited to the complete-original-window admission
class below; the fingerprint does not expand that class to reached suffixes.

The checked plan also retains one `IntervalResourceDemand`, available through
`plan.resource_demand()`. It covers all admitted members, canonical child
associations, later numerical uses, section dependencies, and original controls,
including guards that close ownership without containing a numerical unit.
`resource_demand_from_reached_window` can issue a prospective inventory from
registered interval states before scanning. Every such inventory must cover
the final plan before emission; a prefix-only inventory is insufficient.
These inventories establish neither native effects nor storage separation or
cost coverage. They do not change placement or public build manifests.

`plan.interval_applicability()` projects the same checked operations and demand
into an optional scope-level `interval_applicability` schema-version-1 record.
It lists descriptor/control key candidates, original controls and native graphs,
and independent missing measurement, workload-stability, build and execution-outcome
evidence. Candidates grant no permission to evaluate source expressions. This
diagnostic retains the exact configuration identity and existing exact-window
decision; it neither attaches calibration nor supplies an automatic selector.
The pipeline validates its linkage before edits and preserves it through
composition. Missing interval estimates continue to select native execution.

The compilation session shares one authenticated immutable original source
graph across private lowering and generation transactions. Original syntax,
declarations, namespaces, source text and configured-source identities are
sealed together; private projections and analysis caches remain explicit.
Queries authenticate session membership, while publication rechecks live source
and configuration inputs. Standalone mutable analyses retain fresh checks.
Parsing and later numerical lowering consume the captured original or prepared
text, including the qualified collective lowering path. Normalized numerical
packages retain the exact text checked against their
package hash; temporary live-file edits cannot enter lowering and then escape a
publication hash check by being restored. Unexpanded Fortran
INCLUDE readers outside that snapshot block session sealing; the configured
source-export path supplies preexpanded text and authenticated dependencies.
Reusing a frozen object outside its issuing source session grants no source
authority, and failed speculative generation cannot mutate the original graph.
Only explicitly registered proof classes and exact registered objects may be
borrowed by the execution plan. Unknown metadata is frozen as owned data or
rejected. Calibration portability is checked across independent Python processes
and hash seeds; PRIVATE inventories have deterministic order.

For a bounded diagnostic pass before source generation, add `--scope-preflight`
to a serial `--form-scopes` command. Its schema-version-1 `preflight` report
inspects original procedures independently, preserving source order, both sides
of branches, and complete joined OpenMP groups. An unsupported early operation
does not hide later numerical, native-effect, capture, or source-cost diagnostics.
The pass follows resolved calls breadth-first, inspects each local body once,
and never expands a whole-entry effects closure for reporting. Calls retain
their original argument maps; callee formal lifetimes remain explicitly
unproved until an owner checks the mapped source interval.

`compiler.scopes.preflight.preflight_source_scopes(...)` exposes the same pass as
an immutable snapshot with detached `public()` JSON. Per-operation statuses
distinguish `available`, `unavailable`, `not_requested`, and `not_applicable`;
an unavailable proof is not a claim that the original program is invalid.
Transfer bytes remain unknown until a checked owner and runtime descriptors
establish section coherence. Numerical prices do not imply that a complete
resident-window estimate or execution plan exists. The report shares the
existing operation, procedure, and depth limits, marks incomplete traversal as
`truncated`, authenticates source/configuration/capture identities before
publication, and grants no execution or storage authority. It evaluates no
branch predicates and emits no source edits, workers, or output directory.
An optional API `observer` receives detached `analysis_started`,
`procedure_started`, `operation_started`, and `operation_complete` events so
long inspections can retain progress. Events remain explicitly provisional;
only a returned snapshot has passed final source and policy reauthentication.

`preflight_scope(builder, resource_demands={procedure: bindings})` additionally
inspects native calls against explicit tuples of original `Binding` objects.
The bounded `prove_resource_reachability` provider visits original selections
and resolved callees without first materializing complete effect summaries.
Its parameterized native-effect graph scans each selected root or complete
callee body once, then instantiates the original call associations separately.
Repeated helper calls retain distinct actual resources, controls and barriers.
The existing 256-operation limit now bounds unique source statements and call
instances separately; bounded source visits, substitution work, depth and
procedure count remain enforced. This changes accounting, not the limit values.
Reusing a template does not discharge pointer, completion, alias or section
requirements, and the public graph cannot authorize execution.
It records may-accesses, source-call associations, unsupported operations and
storage-separation obligations. Selected allocatable object fields can be
identified without granting their numerical layout or capture authority.
Distinct names do not prove nonaliasing; unresolved calls, pointer association,
implicit calls and completion remain barriers. Demanded effects are incomplete
when any barrier remains. Source, namespace, binding and declaration authority
is rechecked before publication; returned JSON cannot authorize execution.

These explicit preflight demands are procedure-local observations, not a
discovered complete interval inventory. Resource-budget exhaustion marks that
observation `not_checked` and keeps later independent diagnostics; an invalid
source or foreign binding aborts publication. Existing numerical/cost gates
remain independent. The reachability provider alone supplies neither executable
coherence nor automatic application placement or measured transfer saving.

Reached lexical owners can retain one unchanged original native `CALL` when
the separate `prove_native_demand` consumer establishes its effects relative
to the entire prospective resource inventory. This path is attempted only when
ordinary complete-call admission declines. It uses the existing lexical owner,
host-access hooks, completion waits and native suffix continuation. It does not
create another scheduler or mark an incomplete procedure summary complete.
Original argument expressions, branches and the call body retain their native
evaluation positions; an indexed scalar actual is not evaluated for planning.

`resolve_original_call` authenticates source targets and actual/formal type
associations independently of generated execution mappings. The stronger
`resolve_source_call` interface keeps its existing mapping requirements.
Reachability records alone cannot authorize the new path: its registered
`SourceBoundNativeDemandEffects` token must discharge every storage-separation
obligation and reject unknown calls, unsupported completion, allocation changes,
pointer association and implicit execution. The initial separation proof uses
exact distinct owned module or non-dummy local declarations, including selected
native allocatable component payloads. Unknown dummy aliases and storage
association remain boundaries. A component that stays native gains no GPU
capture or device layout.

The native call uses conservative whole-resource read/write accesses for
demanded arrays, requiring whole entry definitions and the ordinary capture and
descriptor checks. It grants no new definitions or overwrite guarantees.
Demanded `INTENT(OUT)` entry changes and writes to demanded planning scalars
remain unsupported. Current host reads use the existing coherence rules and
retain valid device copies; native writes invalidate the affected managed
resource. Precise native sections are not inferred by this new route.

`resource_demand_from_inline_owner` inventories all prepared root numerical
regions and original scalar, descriptor and control dependencies before native
admission. It is prospective, not a whole-call effects closure. The final checked
plan must validate every member and later use against that exact inventory;
an uncovered child resource rejects the candidate before publication instead
of authorizing a prefix-only demand. Source/configuration identities and exact
registered proof objects are rechecked before emission. This bounded route
does not yet discover a complete prospective inventory for arbitrary borrowed
children.

Automatic placement additionally requires source-bound native-interval cost
observations. Hardware or leaf-kernel profiles do not cover this interruption;
the checked plan records the missing requirement and selects unchanged native
execution. `sections` execution is a numerical/coherence diagnostic. This
capability alone establishes neither an automatic application candidate nor
either channel performance target.

One-shot module owners close before excluded allocation
swaps and do not retain state across backedges or invocations.
Numerical extraction checks liveness against the complete original lexical
body even when preparing only a window. Current native suffixes and possible
next-iteration prefixes are checked as separate paths, so one path's definitions
cannot hide another path's scalar or fixed private-array reads. Original DO
final values remain native when observable outside their numerical region.

The `prove_native_definitions` source API retains ordered original
assignments, IF alternatives and single-assignment affine DO nests. Its transfer
reports exact symbolic defined rectangles and guarded unmet read obligations;
only must-definitions from reached leaves contribute, and branch guarantees
meet all alternatives, including a missing ELSE. Opposite faces stay separate,
empty-loop requirements remain attached, and scalar changes invalidate dependent
rectangles. Supplied entry rectangles require independent admission. Calls,
OpenMP, allocation changes and complex loop bodies remain unsupported. This
proof does not register storage or execute numerical work. Version 2 accepts an
exact descriptor-window token and reissues exact per-leaf tokens. Reached-window
owners consume this proof for selected assignments and affine loops; calls and
unsupported footprints keep their conservative whole-definition requirements.

Optional `--effect-contracts` input has `schema_version: 1` and a `procedures`
object keyed by a qualified imported call name. Each contract explicitly declares
an `identity`, `complete: true`, `lifetime: "stable"`, `escapes: false`,
`descriptor_changes: false`, and `ordering: "serial"`. Its `effects` list contains
`kind` (`read`, `write`, or `overwrite`) and `section: "whole"`, plus either a
zero-based `argument` position or a source-available hidden `resource` identity.
The compiler records a contract hash and rejects malformed contracts. A contract
is an explicit assertion about the complete native call, not inferred from INTENT.

Effect completeness is separate from scope legality and GPU eligibility. The
analysis does not rewrite calls or evaluate guards/bounds. Compiler source scope
generation consumes these facts; adapters must not turn source strings into
coherence hooks or placement decisions themselves.

### Ordinary CUDA allocation reuse

Ordinary CUDA calls acquire exclusive allocations from a lazily created private
pool for the calling thread's current device. Generated entries share the pool,
but each call constructs fresh buffer state, shapes, and coherence flags. Host
mirrors and caller pointers are never cached. Explicit workspaces retain dedicated
allocations. Neither the application's current/default pool nor CPU allocation
behavior is changed.

`FORT_CUDA_POOL_BYTES` sets the pool release threshold in unsigned decimal bytes;
the default is `268435456` (256 MiB) per device. The setting is read once on first
pool use. `0` disables reuse and restores ordinary `cudaMalloc`/`cudaFree` behavior,
allowing comparisons with the same generated binary. Invalid or overflowing values
are diagnosed. The threshold is a retention target, not a hard memory limit;
newly freed memory can remain above it until another synchronization. Pooling is
independent of optimization level, schedules, and indexing mode. Toolkits before
CUDA 11.2, incompatible drivers, and devices without pool support use dedicated
allocation; other CUDA failures are not hidden by fallback.

Allocation uses `cudaMallocFromPoolAsync` on the existing default stream. Planned
destruction synchronizes before `cudaFree`, so completed storage can be reused
without adding a final asynchronous-free barrier. Exceptional pooled-buffer
cleanup synchronizes before freeing. Zero-sized buffers do not initialize a pool.

Each generated module also exports `<entry>_trim_cache()` (with the same collision
and length handling as workspace names). After joining ordinary callers, call it
to destroy this runtime's private pool and release any idle chunked/hybrid
scratch slots on the current device. The next ordinary call recreates resources
as needed. It diagnoses outstanding pooled leases and is a no-op
for CPU implementations and absent pools. Sessions remain valid and no outputs
are retrieved. Release the cache before device reset, external context teardown,
or unloading generated code; the runtime performs no CUDA work in static
destructors. Pools otherwise remain available until process termination.

Profiling includes allocation/release requests and lazy pool setup under the
existing timing labels. `start_hot` and `finish_hot` do not clear the cache.
Measure cold and warm calls separately; use pool backing statistics or an
instrumented allocator to verify reuse, and wall-clock timings to measure its
effect. Unprofiled calls create no profiling events or profiling barriers.

## Persistent workspaces

Each entry gains an additive workspace API. For example, the CDU module exports:

```fortran
type(CDU_workspace) :: work
call CDU_create(work, U2, U, V, W)
do iteration = 1, niter
  call CDU_run(work, dxmin, dymin, dzmin, NX, NY, NZ)
end do
call CDU_update_host(work, U2=U2)
call CDU_destroy(work)
```

`create` takes all original arrays in dummy order, owns fixed-shape storage,
and copies only `in`/`inout` data. `run` takes the original scalar arguments in
dummy order. Neither retains caller addresses. `update_device(work, U=U)`
publishes a complete replacement for a selected slot; `update_host` retrieves
selected slots. Both accept optional array keywords, require matching shapes,
and complete borrowed-pointer transfers before returning. Ordinary host writes
are not automatically visible to resident execution.

CUDA runs may enqueue work on the default stream. Updates, destruction, and
profiling boundaries synchronize. Host blocks and fallback use lazy owned host
mirrors and the same coherence plan. C++ implements the same API using owned
host buffers, allowing the Fortran module to link with either backend.

Workspaces use validated opaque tokens. Assignment aliases a session; destroying
one alias invalidates the others. Creating into a live workspace and using stale
tokens are errors. Destroying an empty workspace is a no-op and does not retrieve
outputs. There is no automatic finalizer. Sessions are per entry, fixed-shape,
and exclude concurrent use or cross-entry buffer sharing. Long/colliding API names
receive deterministic shortened names in the generated interface.

Profiling starts disabled. `start_hot` synchronizes, clears measurements, and
enables recording; `finish_hot` synchronizes, prints the existing timing summary,
and disables recording. `FORT_RUNTIME_TRACE=1` separately logs successful CUDA
allocations, transfers, launches, and releases to stderr for validation.

## Supported Fortran subset

Module subroutines are selectable without annotations. Legacy markers remain
accepted, and `--require-markers` makes them mandatory:

```fortran
! kernels
module example
  implicit none
  integer, parameter :: knd = kind(1.0d0)
contains
  ! kernel
  subroutine scale(a, factor, n)
    real(knd), intent(inout) :: a(:)
    real(knd), intent(in) :: factor
    integer, intent(in) :: n
    integer :: i
    real(knd) :: tmp
    do i = 1, n
      tmp = a(i) * factor
      a(i) = tmp
    end do
  end subroutine scale
end module example
```

- Default `integer` (signed 32-bit), default `real` (binary32), and resolved REAL kinds 4 and 8
  scalars and assumed-shape dummy arrays. Array rank and loop nesting depth
  follow native Fortran language/compiler limits.
  Integer literal tokens must fit before applying unary signs: `-2147483648`
  rejects, while `(-2147483647-1)` is valid. Constant integer operations check
  every intermediate result, including division by zero and `ABS(INT_MIN)`.
  Unknown runtime values retain the existing arithmetic and ABI.
- Explicit `intent(in/out/inout)` arrays. Omitted array intent is conservatively
  treated as `inout`; omitted scalar intent is treated as read-only input. Scalar
  entry argument writes remain unsupported. Pure numerical helpers can write
  private scalar actuals with explicit `INTENT(OUT/INOUT)`. `contiguous` and `target` are accepted;
  pointers remain unsupported.
- Counted `do` loops with positive or negative integer strides, including
  invariant runtime stride expressions. Constant zero strides reject; a runtime
  zero stride diagnoses at loop entry. Empty domains do not launch.
- Perfect and imperfect nests. The analyzer maps a rectangular perfect prefix;
  assignments and remaining inner loops execute in their original order within
  each parallel iteration. Bounds may depend on outer indices in retained loops.
  Fusion retains the ordered body of each original mapped iteration.
- Host scalar and array element assignments before, between, or after nests.
  Private temporaries must be defined before each read in the mapped iteration
  and unused after the region. Retained sequential loops may carry a scalar
  initialized within that iteration; their final induction values are available
  within the iteration too.
- Grouped arithmetic `+`, `-`, `*`, `/`, unary signs, integer/real literals,
  `SIZE(array, dimension)`, and `SIZE(array)` total element counts.
  Default real literals retain single precision; `D` exponent literals use
  binary64. Explicit kind suffixes use their declared numeric precision.
  Integer constant powers from zero through sixteen lower to ordered products.
- Affine access relations and constant-stride congruences are modeled exactly.
  Invariant non-affine bounds are captured as symbolic parameters. Unknown
  subscript coordinates are conservatively unconstrained, allowing read-only
  indirect gathers and writes whose other coordinates already distinguish
  parallel iterations. Identical squared affine subscripts, such as `a(i*i)`,
  receive a simple injectivity proof when every access shares that expression
  and its operand is proven nonnegative or nonpositive throughout the region.
- Positional whole-variable calls to same-module helpers. Inlining creates fresh
  locals per call and preserves actual storage identities and case-insensitive
  Fortran name resolution.
- Source-backed pure module/internal functions and subroutines, including lexical
  captures and scalar function results. Pure scalar input actuals may be
  expressions; pure function arguments require `INTENT(IN)`. Helper work remains
  at its original expression evaluation point, including guarded `ELSEIF` paths.
  Recursion rejects; numerical inlining is bounded to depth eight and 128 calls.
- Fixed-size private numerical arrays with constant explicit bounds, including
  negative bounds, scalarize to independently defined scalar storage. Each array
  has at most 256 elements. Constant indexing loops are unrolled with a shared
  256-iteration budget; unknown private subscripts remain boundaries. Array
  arguments retain Fortran element-order association and dummy-bound rebasing.
  Explicit-shape helper arrays require private fixed array actuals or read-only
  fixed `PARAMETER` vectors.
- Fixed-cardinality sections with one varying axis support elementwise
  expressions, private array assignments, short serial `SUM` and `DOT_PRODUCT`.
  Captured-array section origins can depend on the mapped coordinates when the
  endpoint difference is provably constant. Each vector has at most 256 elements,
  with at most 4096 expanded elements per numerical closure. Nonzero constant
  strides and empty vectors retain Fortran order; omitted endpoints use declared
  lower/upper bounds regardless of stride. Right-hand-side values are captured
  before overlapping private-array writes, and scalar broadcasts evaluate once.
  Sums/products accumulate in logical element order from a typed zero under the
  supported native numerical contract; this grants no parallel reduction contract.
  Private `SIZE`, `LBOUND` and `UBOUND` inquiries
  preserve original bounds and empty-array semantics. Scalar real `PARAMETER`
  declarations and rank-one numeric `PARAMETER` vectors preserve typed initializer
  expressions. Outlined source regions retain a bounded closure of imported and
  lexical constants, original bounds and dependencies, without registering them
  as mutable captures or private arrays. Private `INTENT(OUT)`
  outputs require complete ordered definitions on every reached helper path.

An original rank-one numeric `PARAMETER` can also be forwarded unchanged to a
required fixed-shape `INTENT(IN)` module companion. This association keeps the
original argument and native ABI; it adds no `TARGET`, temporary, address or
device registration for the coefficient array. A serial reached perfect loop
nest may capture an invariant element as a separate typed scalar operand.
Ordered nonempty checks precede queries and operand preparation, and the value
is read from the original formal only after its original guards and a valid
index check. Empty nests execute their original source without querying or
reading dormant inner controls. Joined worksharing, uncertain indices or
activation, and unsupported associations stay native. Original array read
effects remain public, with an explicit projection for managed coherence.
Automatic placement remains unavailable until this preparation is calibrated.
When managed and immutable entities share a declaration, only the managed
entities receive `TARGET`; the original entity order and attributes are retained.

An immutable scalar `PARAMETER` whose initializer is outside the outlined
constant language can supply its original Fortran value as a read-only scalar
argument. Its canonical binding must be visible from the original numerical
owner, including through imports and renames. The compiler does not recompute
that initializer on the GPU or fold it with Python. Inaccessible helper-local
constants remain native, and array `PARAMETER` initializers still require their
complete compile-time dependency proof. Source and initializer changes invalidate
the outlined artifact identity.

Real fixed-vector work, including short `SUM`/`DOT_PRODUCT`, carries a
numerical-environment requirement through lowering, preparation and emitted entries.
Source-integrated execution checks
the original thread environment and retains the unchanged native span when
rounding is not nearest or host traps are enabled. Configured source that
observes IEEE exception flags remains native. Forced standalone CUDA entries
and session runs have no original Fortran body: they diagnose unsupported
environments before numerical work. Their caller must not observe newly raised
host floating-point flags; a runtime check cannot prove future observations.
Runtime setup and profiling preserve pre-existing flags, rounding and trap masks.
Short serial reduction semantics were checked against the supported native
gfortran 15 backend and semantic flags in both precisions; a different native
backend needs independent conformance evidence, not an assumed parallel contract.
Automatic estimates for these guarded numerical entries remain unavailable until
offline calibration measures their extra entry/session checks and collective
coordination. Forced controls remain available for correctness and measurement;
an unchanged profile hash alone cannot price the added protocol.

Logical arrays, writable scalar entry arguments, recursion, runtime-cardinality
slices and snapshots,
non-pure expression call arguments, other intrinsics/operators, explicit lower
dummy array bounds, and
unsupported specification statements reject with a source location.
Parallel reductions remain unsupported by this ordinary numerical frontend.
Scoped source generation has the additional proven array-operation normalization
described above; it does not make arbitrary numerical slices legal. Initialized scalar recurrences, valid
scalar live-outs, final induction values, and otherwise unproved regions can
execute sequentially with `--fallback host`. General nonlinear or indirect writes
are parallelized only when conservative relations prove independence; no runtime
alias or index-uniqueness checks are introduced.

Transcendental helper tests cover both precisions, zero gradients, repeated
singular values and signed zero against native Fortran. Ill-conditioned spectral
intermediates can amplify small scalar/vector math-library rounding differences:
an independent rank-deficient diagnostic observed approximately `7.5e-9` in a
nominally zero binary64 singular value with gfortran 16.2 at `-O3` and contraction
disabled. This does not establish strict agreement for those intermediates.
Complete application field tolerances remain the acceptance criterion and are
not relaxed to admit a helper closure.

### Conditions and scalar intrinsics

Scalar `LOGICAL` values, logical literals/operators, numeric comparisons, block
and single-line `IF`, `ELSEIF`, and `ELSE` are supported. Definitions after a branch
must be valid on every path. Host branches use recursive execution plans, and
device predicates participate in conservative dependence analysis. CUDA preserves
host/device coherence when entering and leaving host branches. Logical scalar
arguments are explicitly converted to `logical(c_bool)` by the Fortran bridge.

The following intrinsics support scalar expressions in both C++ and CUDA:

- Numeric functions: `ABS`, `MIN`, `MAX`, `SQRT`, `EXP`, `LOG`, `LOG10`, `SIN`,
  `COS`, `TAN`, `ASIN`, `ACOS`, `ATAN`, `ATAN2`, `SINH`, `COSH`, `TANH`, `MOD`,
  `MODULO`, `SIGN`, and `DIM`.
- Conversions and rounding: `REAL`, `INT`, `NINT`, `FLOOR`, `CEILING`, and `DBLE`.
  `REAL` defaults to binary32 and accepts constant `KIND=4` or `KIND=8`; `DBLE`
  returns binary64. Integer results use the default signed 32-bit kind, including
  explicit `KIND=4`. `INT` truncates toward zero; `NINT` rounds ties away from zero.
- Type/model inquiries: `KIND`, `EPSILON`, `TINY`, and `HUGE`. These use the
  declared type of a scalar or whole array without reading its value. `HUGE`
  supports INTEGER and REAL; `EPSILON` and `TINY` require REAL.
- Array inquiries: `SIZE`, `LBOUND`, and `UBOUND` on whole assumed-shape dummy
  arrays, with constant or runtime INTEGER `DIM` and optional `KIND=4`. `SIZE`
  without `DIM` returns the total element count. `LBOUND` and `UBOUND` require
  `DIM` because array-valued results are unsupported. Dummy lower bounds are 1,
  and upper bounds are their extents, including 0 for an empty dimension. Runtime
  dimensions must lie within the array rank.
- Selection: `MERGE` with matching numeric or logical scalar sources and a
  logical mask.

Positional and standard keyword arguments are accepted. Numeric functions preserve
supported argument kinds and arithmetic grouping; `MIN`/`MAX` require at least
two arguments, and multi-argument numeric functions require matching types and
kinds. Transcendental functions require REAL arguments. `MOD` follows the dividend's
sign and `MODULO` follows the divisor's sign. Numeric runtime helpers evaluate each
argument once. Logical arrays, array-valued elemental calls, other result kinds,
and unlisted intrinsics remain unsupported.

Semantic validation and definite definitions live in `analysis/semantics.py`;
structural effects live in `analysis/effects.py`; `analysis/planning.py` builds
execution plans according to the selected fallback policy. Intrinsic signatures
live in `ir/intrinsics.py`; `runtime/numeric.hpp` is assembled into the existing
shared support header.

The optimized GNU Fortran numerical contract is checked against `gfortran-15
-O3` for both REAL kinds. Two-argument runtime `MIN`/`MAX` preserve ordered
operand selection, including signed zero. A finite leading literal uses the
separately checked constant-first form, including degenerate numerical clamps.
Unordered multi-argument forms are not covered by this contract and new helper
closures retain native execution for them. IEEE exception guards remain at
their original source locations. Scoped runtime setup and planning save and
restore the caller thread's rounding mode, exception flags and trap mask;
numerical computation is not moved into setup or query evaluation.

`--numerical-costs` extends offline hardware calibration with generic private
array and transcendental workloads. Numerical calibration v2 measures original
Fortran serial and fork/join implementations separately from generated CPU and
GPU workers. Seven predefined interleaved batches run for at least 200 ms each,
with fixed CPU placement, independent sizes and mixed-expression holdouts, and
a 25% validation ceiling. Rejected observations remain in the profile; each
backend and workload class is accepted independently. Compiler options, device,
precision, CPU placement and runtime size ranges must match before AUTO can
use a model. Existing-team execution requires its own coordination calibration.
Runtime real division has a separately validated primitive family; division by
a compile-time constant remains in ordinary arithmetic. Mixed holdouts include
data-dependent division and validate the exact fixed terms emitted at runtime.
Memory calibration uses eight predefined working-set sizes and independent
intermediate holdouts. A bounded bandwidth table follows cache-size changes;
it never extrapolates. Runtime pricing separates read/write traffic from the
unique physical union of accessed sections. Opposite faces remain separate,
read/write overlaps count once in the working set, and unknown footprints,
repeated touched root aliases or checked-arithmetic failures leave AUTO native.
CPU execution startup is removed from the memory measurements and charged
once outside the computation/memory maximum; validation uses the same formula.
If a fitted startup exceeds a measured memory invocation, that family is
rejected rather than clamped. Successful primitive fits do not override failed
mixed-expression validation or establish costs for a different dependency shape.

Original `SCHEDULE(runtime)` work needs the separate top-level
`source_schedule_validation` profile section. Its fixed paired protocol compares
runtime-static execution against the frozen compute model without fitting new
coefficients, retains every observation, and intersects only structurally valid
working-set ranges. Missing or failed evidence keeps AUTO native. An admitted
entry checks the current OpenMP schedule before context creation and in every
public query; static scheduling with default contiguous chunks is required.
Changing the schedule or selecting cyclic chunks invalidates the estimate.
The standalone `python -m compiler.offload.schedule_calibrate` producer takes
`--profile`, a new `--output`, a new `--build-dir`, and repeated
`--fortran-flag=...` arguments matching the numerical profile. It checks both
compiled Fortran identities and CPU placement before timing. A profile with no
accepted native workload class is rejected before building the validation job.

Two optional CPU calibration sections separate measured execution startup from
the numerical-v2 regression intercept. First,
`python -m compiler.offload.cpu_protocol_calibration` adds
`cpu_execution_protocol`: zero/one/eight-item startup controls, an untimed
OpenMP team-entry proof over the same object files, and fresh three-array memory
observations. Generated controls use the production cyclic worker renderer,
while original Fortran controls retain their original static worksharing.
The timed and proof executables share their numerical objects;
the proof instrumentation is absent from timing. Seven interleaved batches per
size retain raw observations, actual team size, thread limits, CPU placement,
Fortran options and `OMP_WAIT_POLICY`/`GOMP_SPINCOUNT`. Archived numerical-v2
memory observations remain diagnostics and cannot establish this new protocol.
Startup is charged once outside the compute/memory maximum. Failed startup or
memory validation makes the affected backend unavailable.

The explicit `--numerical-costs --numerical-version 3` calibration path records
the repaired CPU protocol in a separate `numerical_execution` section. It uses
the original Fortran serial/static-team bodies and production generated cyclic
workers with four host threads, independent zero-item startup controls, and
seven complete global rounds of at least 200 ms per cell. Size, expression,
helper and domain holdouts never fit coefficients. Fresh memory observations remove the measured startup;
the model adds that startup once outside the compute/memory maximum. The raw
batch file is created exclusively and flushed as each batch completes, so an
interrupted attempt remains evidence rather than being overwritten.
This initial section is CPU-only: it does not supply GPU costs or authorize
automatic SGS placement. If supplied for source placement, incomplete or
rejected new evidence keeps execution native; it cannot borrow older numerical
coefficients. Versions 1 and 2 retain their existing defaults and interpretation.
The standalone `python -m compiler.offload.numerical_execution_calibration`
adds this evidence to a preserved compatible base profile using `--profile`,
a new `--output`, a new `--build-dir`, and explicit original
`--fortran-flag=...` arguments. Its `--preflight-only` mode builds and checks
the workers without sampling or adding cost evidence.

The separate `python -m compiler.offload.numerical_gpu_execution_calibration`
producer records optional `numerical_execution_gpu` observations using those
same seventeen recipes and their production CUDA kernels and launch code.
It accepts `--profile`, new `--output` and `--build-dir` paths, and explicit
`--nvcc`, `--cuda-host-cxx`, `--arch` and `--device` selections. Fixed enqueue
CPU placement comes from `--cpu-affinity` or the existing CPU evidence. Inputs
remain resident during measurement. Complete outputs, unchanged inputs and sentinel
halos are checked before seven global rounds over the predefined sizes;
interrupted attempts retain their flushed raw observations. CUDA-event,
host-enqueue and wall durations and actual launch counts remain separate.
Zero-item controls issue no CUDA launch and cannot establish launch overhead.
This initial GPU section supplies evidence only: it does not fit transferable
startup costs, combine CPU/GPU cost models or authorize automatic placement.
The base profile and existing CPU evidence are preserved. Profile loading
checks the observations without lowering source; using an archived backend
requires a separate check against current renderer and numerical identities.

`python -m compiler.offload.numerical_gpu_launch_calibration` adds the separate
`numerical_execution_gpu_costs` section to that evidence. It takes `--profile`,
fresh `--output` and `--build-dir` paths, and optional `--nvcc` and
`--cuda-host-cxx`. The producer authenticates and reuses the parent worker object,
header and driver, then builds only a launch-control driver. Zero/one/eight/seventeen
item preflights check outputs and sentinels; seven global rounds of at least
200 ms measure the three nonempty sizes. One item fits a launch-inclusive fixed
interval; eight and seventeen items validate it independently. The original
production observations supply compute costs and independent expression,
helper, private-array and domain holdouts, with the same 25% error ceiling.
Cost protocol v2 interpolates total GPU memory-launch intervals without subtracting
the separate fixed estimate, then uses `max(fixed + compute, total_memory)`;
only the tested pointwise traffic-equals-working-set class is priced. Historical
v1 profiles retain their original reconstruction and cannot supply v2 queries.
The cost query requires current source identities and accepted families, classes,
domains, features and size ranges. Resident-event intervals already include
enqueue starvation, so their separately recorded enqueue time is not added again.
The measurement retains the original sixteen-launch/join cadence; applying its
costs to a different enqueue/wait protocol requires separate evidence.
Setup, allocation, transfers, coordination and source guard costs remain separate.
This interface alone does not establish source applicability or AUTO placement.

The other existing producer, `python -m compiler.offload.cpu_dependency_calibration`, adds
`cpu_dependency`, without changing the numerical-v2 section. Its 26 predefined
generic recipes distinguish ordinary operations, constant/dynamic division and
selected transcendental families. One-chain and four-chain bases identify a
bounded work/span hypothesis; independent sizes, inline/separate helper forms,
diamonds, private arrays and argument-domain holdouts validate it. Every timed
batch lasts at least 200 ms; all seven observations and rejected families are
retained. Memory-limited or unidentifiable bases reject instead of changing the
fitting equation. The same three-array access contract and proven execution
identity apply to the new compute observations. Both producers take
`--profile`, a new `--output`, a new `--build-dir`, optional compiler paths and
repeated `--fortran-flag=...` arguments matching the original Fortran build.
Run them sequentially under the campaign memory guard, before application
measurements; neither producer reads or fits application timings.

When the input contains current `numerical_execution` evidence, the dependency
producer uses `source-work-span-current-execution-cpu-v2`: 49 fixed recipes and
735 cells, sampled in seven global rounds. Only the twelve one/four-chain bases
fit coefficients. Two-chain controls and the seventeen unchanged original
execution recipes are independent holdouts. Current startup and memory evidence
can be accepted separately from rejected arithmetic families; rejected arithmetic
is never borrowed. Queries explicitly select `direct_loop`,
`value_function_same_unit` or `value_function_separate`, with independent class,
domain and private-feature validation for each form. Direct loops retain the
measured 32-element/two-array private limit; value-function forms can admit
64 elements/four arrays only when their own controls accept. These names describe
compiled source forms, without asserting vectorization. Source placement still
requires proofs that the original and generated implementations match the
measured forms, memory access, domains and execution protocol. The existing
26-recipe protocol remains available with its original foundation.

Public unit diagnostics expose the bounded source dependency identity and
physical memory access class separately from legality. The first source
attachment admits contiguous one-dimensional pointwise work with two distinct
immutable inputs and one write-only output. Mixed floating-point conversions,
unpriced integer/logical numerical work, unproved primitive input domains
(including dynamic division), uniform numerical folding/hoisting, borrowed-view
addressing, unconsumed private work, other memory patterns and existing-team
costs remain unavailable.
Runtime checks
enforce original root strides, CPU placement, team budget and calibrated wait
environment. The wait/spin environment must remain fixed from process launch:
live checks detect mismatches but cannot read libgomp's private effective wait
state or establish that changing `getenv` values reconfigured an initialized
runtime. An optional rejected dependency section keeps AUTO native rather
than borrowing an accepted older expression class. Numerical-v2-only profiles
retain their existing path; these additions do not establish automatic SGS
placement or an application speedup.

A profile requesting current `cpu_dependency` or GPU execution costs takes the
current source-cost path before the CPU-only placeholder. Original inline-region
preparation can supply `source_compute_applicability_v1` metadata for a single
helper-free direct loop with serial or fixed fork/join participation. The record
is issued before artifact rewriting and carries original preparation and source
identities; it grants no execution, capture or definition authority. Each region
keeps its own provenance when identical work shares a worker, and the manifest
identifies the shared artifact's issuing region separately. Reuse also requires
the same compiled-form, participation and access contract. Whole-procedure
lowering without an original-region preparation does not invent that evidence.

This attachment still requires the exact one-dimensional pointwise memory class,
observable dependency work, supported primitive domains and independently
accepted costs for each implementation that a candidate uses. Source model schema
2 records unavailable native Fortran, generated CPU or GPU estimates as `null`
with separate rejection reasons. It intersects known estimates' tested item
ranges and requires matching CPU placement and native Fortran identities.
Missing current evidence never borrows legacy costs. Unpriced
numerical-environment guards remain unavailable. Optional source classification
does not run for ordinary generation without a current cost bundle.
Preparation schema 2 fingerprints the full raw calibration profile separately
from executable syntax, with a 32 MiB JSON limit and charged hashing work.
Every validation rehashes it; changing any retained observation invalidates
preparation. Large profiles do not consume the statement budget as if their
measurement fields were source operations.

One synchronous source-cost attachment shares raw reconstruction across its
units and their native/generated implementation queries. Evidence is copied
lazily after the source eligibility guards into an owned JSON snapshot; small
validated cost sections are retained as immutable serialized data. Each query
still checks its compiled form, workload, domains, features and access class.
Returned dictionaries own their values. No snapshot survives the operation or
supplies allocation freshness at application runtime.

Before publishing any estimates, attachment checks the complete original profile
and current reader/backend identities again. A change discards all estimates
from that operation, including estimates for earlier units, and keeps AUTO
native. Public cost queries continue to reconstruct evidence afresh. Rejected
observations and acceptance thresholds are unchanged; reader changes still
invalidate their calibration identities and do not relabel archived profiles.

Source planning supplies complete compute seconds for known implementations
through `fort_scope_plan_add_compute_costs_v3`. Transfers, launches, allocation
and coordination are charged separately. The unchanged compute-cost payload
layout accepts versions 1 and 2. Version 2 can mark a known GPU total with
`FORT_SCOPE_COMPUTE_GPU_LAUNCH_INCLUDED`; the planner then omits only that
operation's separate enqueue charge. Legacy neighbors retain their own launch
costs. The cheap native-rejection bound also respects this coverage, while all
setup, transfer and synchronization charges remain. A native operation inside an active
owner uses its actual coordinated host cost, including candidate-only inspectors;
that differs from the whole-original native fallback. A fresh owner can compare
the original native body with an all-GPU alternative without pricing unused
generated CPU workers. Mixed candidates require each coordinated CPU cost they
use; missing preparation costs reject those candidates. A missing original
native estimate keeps AUTO unavailable. Continuations compare against coherent host execution
and cannot replay an earlier GPU prefix. The existing v2 additive-cost ABI and
legacy calibration reader remain available (`--numerical-version 1`), but C++
measurements cannot establish original Fortran costs.

When every numerical alternative statically lacks an applicable estimate,
AUTO preserves the original caller source without descriptor or IEEE guards,
runtime registration or queries. This is a successful native decision. A valid
alternative retains reached runtime planning and its guarded evaluations.
For a fresh, unconditional single numerical unit, `native_preflight_v1` can also
prove native execution from an empty domain or an item count outside the model's
validated range. It reads only guarded descriptor metadata and read-only INTEGER
inputs, before context creation or registration. Unknown or overflowing inputs
continue ordinary planning. Reached lexical units with definition changes retain
registration and ordered validation; an existing owner always uses coherent
continuation. This preflight does not yet prove native execution from GPU startup
costs; no unproved native memory-cost upper bound is assumed.

## Legality and caller contract

For each region the analyzer constructs statement iteration domains,
multidimensional read/write access maps, and the original lexical schedule. It
composes access maps to find ordered read-after-write (RAW), write-after-read
(WAR), and write-after-write (WAW) conflicts. Every conflicting pair must have
identical **mapped iteration coordinates**, including every dimension assigned
to CUDA threads. Ordered same-cell updates and recurrences inside retained
sequential loops are accepted within one mapped iteration. Conflicts between
mapped iterations reject by default or select sequential host fallback when
requested. A perfect rectangular nest maps all its dimensions; an inner
recurrence in such a mapped nest requires host fallback.

A dependence error includes source/call provenance, affected accesses, the
symbolic conflict relation, and a conflicting-iteration witness. Embedded
single-line actions inherit their enclosing source span while retaining inline
call stacks. If unknown indices, variable-stride congruences, or branch effects
without predicate constraints widen the model, successful reports and conflict
diagnostics label the dependence model conservative. A conservative conflict
witness shows why independence could not be proved, rather than promising a
collision for every caller's runtime data.
Branch-free affine models remain exact; proofs do not split paths by predicates.
`--verbose` also displays exact/conservative modeling and retained loop counts.
Unfused regions execute in original order, honoring cross-region dependencies.

### Checked fusion

The pipeline first proves source regions independently. Greedy fusion then matches
equal rectangular domains with constant nonzero strides and scalar/extent-only
bounds, substituting iterator identities without regrouping expressions. Same-point
operations retain their order. Fresh analysis must prove that combining the bodies
introduces no conflicts between parallel iterations; shifted RAW, WAR, or WAW
conflicts keep the passes separate. Conservative non-affine models may prevent
fusion even when each original region is independently legal.

Intervening scalar setup can move before the first pass only when its inputs are
available and unchanged and its destinations are unused by the crossed loop. Array
accesses never move. Scalar lifetimes, bound snapshots, symbol identities, and
source provenance are preserved. Fusion stays inside each host branch. Invalid
source remains a compilation error; an inconclusive optimization keeps its original
form. Verbose reports explain each applied or skipped candidate.

### Scheduling and explicit tiles

Scheduling follows legality analysis and fusion. `RegionSchedule` records axis
order, optional tile sizes, and CUDA thread count; emitters render those choices.
Automatic ordering scores unit-stride accesses in Fortran storage order, weighting
writes twice as heavily as reads. Unknown accesses receive no locality credit,
and ties keep original innermost-first order. `--schedule source` preserves that
order; either explicit schedule choice overrides the optimization-level default.

For example, `--schedule auto --tile-sizes 32,4` sets sizes for the two fastest
scheduled axes; remaining axes use size 1. The compiler rejects specifications
longer than the maximum mapped rank. Omitting `--tile-sizes` keeps execution
untiled; no default tile size is selected.

Bounds are captured in source order, with inner bounds suppressed when an outer
loop is empty. The schedule reorders ordinal coordinates and reconstructs signed
Fortran indices. CPU emission parallelizes a flattened tile loop and executes
points sequentially inside each tile. CUDA maps tiles to blocks and distributes
points across threads, including tiles larger than a block, with guards for tile
tails. Flattened tile coordinates support arbitrary rank. Iteration products are
checked for overflow before launch, and zero factors suppress spurious overflow
errors for empty domains.

### Checked wide addressing

After scheduling, the addressing planner selects subscript arithmetic that can
use signed 64-bit loop coordinates directly. Static interval proofs must show
that every original intermediate result fits default INTEGER, including division
and integer `ABS`, `MIN`, and `MAX`. Unknown or unsafe arithmetic keeps its
existing lowering; expressions are never reassociated. This is separate from
dependence analysis and does not add runtime guards or additional kernels.

Only mapped iterator ranges are refined, using bounds and signed strides;
constant loops use their exact last executed index. Other integer values retain
the full signed 32-bit range. There is no branch-sensitive or assignment-based
refinement. Bound arithmetic without a safe range proof contributes an unknown
INTEGER snapshot. Neither array extents nor valid accesses imply tighter scalar
bounds. For example, a loop beginning at 2 with an unknown upper INTEGER bound
can use wide `i` and `i-1` subscripts, while `i+1` retains source arithmetic.

Each complete subscript is selected independently, including nested array
accesses. Shared C++/CUDA rendering consumes the plan and preserves scalar
INTEGER arithmetic, the ABI, `SIZE` conversions, and integer array-load values.
Loop-bound snapshots, retained-loop headers, host statements, and sequential
fallback remain unchanged. Source-order bound evaluation and empty-loop
suppression are preserved. No array bounds checks are introduced.

Use `--indexing source` and `--indexing auto` with the same optimization level,
schedule, and tiles to measure addressing independently. Both choices override
the optimization-level default. Plans constructed without addressing metadata
use source indexing. Verbose reports count promoted and retained subscript
occurrences and explain decisions; accesses in unchanged loop headers are excluded.

### Sequential host fallback

`--fallback error` keeps strict parallelization. `--fallback host` permits otherwise
valid regions with unproved independence, initialized recurrences, or scalar
live-outs to run in source order. Proved neighboring regions stay parallel, and
fusion never crosses a fallback region. Verbose output identifies each sequential
region and its proof-failure reason.

The execution plan represents these loops as `SequentialRegion` nodes. Shared
loop lowering preserves signed and runtime strides, empty domains, final induction
values, and scalar state. Analysis discards obsolete scalar substitutions before
proving later regions. CUDA synchronizes affected arrays before host execution and
uploads host writes before later device work; partial writes preserve other cells.

Fallback is an execution policy after semantic validation. Undefined scalar reads,
invalid types, unsupported constructs, zero-stride errors, and internal failures
remain errors. It does not implement parallel reductions.

Distinct array arguments must not overlap whenever either is written. The
compiler does not perform runtime overlap checks. Repeated actual arguments in
an inlined helper retain shared identity and are analyzed as aliases. Callers
must provide valid extents and sufficient storage for every accessed index;
general bounds checking and broader Fortran semantics are outside this subset.

### Experimental complete-window observations

`--scope-cost FILE` transports a JSON object into `OffloadConfig.exact_window`.
The existing complete-owner request requires `--scope-execution reached`.
A request with `"coverage": "original_selection"` uses bounded execution and
the checked plan for a complete original numerical OpenMP group. The original
scalar preparation and enclosing branch remain in place. The compiler closes
the group's local context before any whole-group native fallback and never
replays executed work. This discovery path uses the same numerical, definition,
completion and resource proofs as reached owners; it does not supply effects
for unsupported native operations or extend ownership across them.

Cost requests are independent of per-kernel hardware profiles. Supplying a
request does not establish that its evidence is available, compatible or
profitable. The automatic policy retains native execution when those checks
fail. Diagnostic `--scope-measurement` and `--scope-cost` are mutually exclusive.

The checked-interval request uses `"coverage": "checked_interval"`; original
numerical-group requests use `"coverage": "original_selection"` through the same
checked-plan attachment. Both use
`"calibration": "native"` or `"resident"` for separate observation artifacts,
an executed application `build_contract`, and explicit `workloads`. Production
omits `calibration` and supplies `evidence`. All three variants use the same
automatic numerical options. The `checked-interval-process-v4` protocol
compares complete application processes, including startup, every invocation,
output and teardown. It checks seven rotated pairs and two holdouts, full-field
agreement, actual completed GPU launches, reached descriptor/control keys and
build identities. Successful interval cleanup and a sticky zero-fallback count
are required, with actual GPU work in each resident process. Planned native
interruptions are part of the measured alternative; a GPU prefix followed by a
native suffix cannot qualify as complete resident evidence. Warm group timing
alone cannot establish an application gain.

The selection contract (schema 2) binds a timing acceptance criterion before
calibration. Omitting `acceptance` retains `{"criterion": "strict_advantage"}`:
every resident sample must be more than 5% faster than every native sample.
An explicit `"acceptance": {"criterion": "paired_non_regression"}` instead
requires the resident process to take no longer than its corresponding native
process in each of the seven measured rounds and both holdouts. Equality is
allowed; one slower pair rejects residency even if its median improves. This
choice changes only the timing criterion. Numerical validation, build identity,
completed GPU work and zero-fallback requirements remain unchanged. Changing the
criterion or supplying earlier protocol evidence requires fresh calibration.

Checked-interval discovery uses the same original-position block control as
closed numerical groups. It considers contiguous serial source selections,
keeps complete OpenMP groups indivisible, and prioritizes possible array reuse
between bulk producers and consumers that write separate outputs. These hints
use original assignments, loops and call argument intents; they are neither
effect summaries nor cost estimates. Tied reuse counts prefer fewer complete
source units. Candidates without visible reuse retain their original order,
as does the entire search if its existing source-scan budget is exhausted.
The candidate and admission budgets are unchanged. The manifest's
`original_interval_discovery` records hints, attempts and the selected span;
the selected interval still requires independent legality and placement proofs.
Discovery opens a fresh context each time
the selected site executes. The context publishes and closes before an excluded
suffix or backedge. This path does not retry a declined selection through the
older first-traversal or whole-entry owner. A proved local scalar prefix retains
the existing entry contract. For a general native prefix, registered selection
activation instead starts every managed resource with unknown definition state.
The original prefix and conditions execute once, before any context opens; no
complete prefix effect closure or invocation-entry payload fact is reused.
Canonical roots register before borrowed dummy association rebases coordinates.

At each reached numerical operation, a separate source-bound proof establishes
the exact values required by well-defined original execution. It retains ordered
unconditional reads before any write to that resource, exact affine sections,
typed tiled-domain guards and independent joined-unit entry requirements.
Conditional operands, whole-array or synthetic preservation uploads, shifted
self updates and conflicting joined units decline this proof. Every upload must
be covered by the exact original reads. After safe descriptor and original guard
checks, `fort_scope_admit_source_reads_v1` adds only previously untracked sections
to host definition coverage. Already current device values remain current;
unread halos acquire no authority. This precedes the local planning query, whose
cost and transfer simulation therefore sees the same coverage as execution.

Unconditional native assignments can use the same reached-read admission before
their native planning query. The existing ordered native-effect proof authenticates
the original assignment, and exact read sections are rederived from that source.
This covers native scalar setup from array elements and point corrections, through
both precise native effects and demand-relative interval effects. Conditional
expressions, masked operations, function calls and loop unions acquire no new read
authority. Reads retain current device copies; subsequent native writes still
invalidate only their proved sections. Entry state and unread sections remain
unknown, and a rejected preflight continues only the unexecuted native suffix.

These read preconditions grant no storage lifetime, alias freedom, native-effect
or numerical-execution authority. Opaque native operations inside ownership still
require the complete existing effects and coherence checks. General-prefix
activation does not yet support repeated selections; the scalar-prefix repeated
path retains its restricted suffix checks. Every admitted interval still needs
at least two numerical invocations and checked resource and completion proofs.
Unsupported consumers may instead have checked close points at their exact
original positions. Reaching one publishes pending device writes, disables
offloading for the remaining activation, and executes the original native
suffix once. An unreached branch grants no effect authority. Close patches are
rendered after placement is attached so process calibration observes every
fallback; qualifying resident measurements still require zero fallbacks.
Line-based replacements also require distinct original statement lines in the
selection and its borrowed procedures. Sharing a line with another executable
statement declines the interval; punctuation inside comments or literals does
not count as another statement.

Discovery inventories bounded source positions before spending admission
attempts. Reached loop bodies precede their enclosing loop; outer sequences
precede nested branches, and maximal selections at distinct positions precede
shorter overlapping cuts. The existing source, adjacency and attempt limits
still apply. An early branch therefore cannot consume the entire attempt budget
by offering many overlapping cuts before later positions are considered.

Entry keys use only the finalized plan's safely readable root descriptors and
proved root controls. Original native branches, child-local tiling and private
work remain at their original evaluation points and are included in measured
process time. Their exact iteration counts need not be predicted. This empirical
cost model grants no alias, lifetime, definition, numerical or completion proof.

Before admitting operations, a source interval records numerical anchors and
prospective resources across original calls. Repeated calls share an inventory
member only when their original procedure, source selection and canonical array
bindings match. Every resolved call edge retains its own checked association;
unresolved calls remain obligations. Sharing analysis does not merge executions
or assume equivalent scalar values.
The draft's version 3 record represents this graph without expanding every call
path. Each exact original call and canonical caller has one source-authenticated
route: prepared numerical child, declined numerical child, or native operation.
A numerical decline still permits one native admission attempt with the existing
effect proofs; it does not imply that a native correction must close residency.
Failed child or native admission closes at the original call without retrying
through another child implementation. Source routing does not
revalidate unrelated numerical payloads; final plan validation still checks
every admitted preparation, native effect, definition and completion proof.

An unconditional checked close ends admission of its exact original sequence.
The remaining native text is authenticated and retained once. A close inside a
branch, child, or loop does not truncate sibling branches, caller continuations,
or later loop paths. Conditional closes retain later eligible operations, and
the runtime's absorbing disable state prevents execution from replaying a GPU
prefix or reactivating after a reached close.

Mixed original teams share one source-only worksharing inventory between
interval discovery and `JoinedTeam` admission. Each supported original DO
retains its registered preparation and team-completion proof; emission consumes
that same preparation. Unsupported corrections remain ordered native units.
The complete team must satisfy every original DO barrier before a subregion
can be prepared. Numerical preparation issues no capture, native-effect,
definition or execution authority, and emits no artifacts.
Exact lexical contexts over a sealed compilation session are reused without
rewalking their immutable syntax. Reuse charges the same logical traversal work
as a fresh query and retains source, configuration and publication checks;
mutable standalone source analyses still derive fresh contexts.

Workload documents bind arguments, relevant environment, input-file hashes and
completion facts. The selector checks actual arguments, environment and bounded
regular input files at the first eligible operation of each interval; these checks also run during
calibration. No cached timestamp or launcher approval replaces the file contents.
This is empirical applicability to observed complete workloads, not a proof of
future invocation counts. Initially production pricing supports one admitted
selection per bundle. Discovery may retain additional declined groups unchanged;
multiple admitted selections support diagnostic capture but select native until
complete bundle pricing is available.

The process request binds its numerical validation policy before capture:
required full-precision fields, storage layouts, ordinary output files,
comparison tolerances and a reviewed diagnostic driver source. The compiler
verifies actual payload hashes and numerical comparisons; a reported pass flag
does not authorize placement. Distinct field hashes are retained when values
agree within that fixed policy. The existing complete-window fixture protocol
keeps its bitwise comparison requirement.

Version 2 policies (`full-float64-comparison-v2`) and numerical receipts
(`checked-selection-numerics-v2`) support either legacy binary VTK or raw
float64 storage as ordinary outputs. Both use `schema_version: 2`.
For raw outputs, declare `outputs: {"format": "raw-float64-storage-v1",
"files": {"state": {"payload": "output/state.bin", "metadata":
"output/state.shape"}}}`. Each receipt's `outputs.state` contains `payload`
and `metadata` file references with absolute `path` and `sha256`; both files
must appear in that run's `output_hashes`. Metadata uses the same
`storage-shape-bounds-v1` format as diagnostic fields: precision `64`, then
shape, lower bounds and upper bounds in dimension order. Payloads use the
policy's `<f8` or `>f8` byte order, finite values and fixed tolerances; storage
shapes and bounds must match exactly. Normal output and diagnostic field paths
must be distinct, and both inventories remain nonempty and mandatory.

For VTK, each declared ordinary output lists required data fields. The verifier
requires matching complete layouts and compares every field present, including
additional fields; other recorded output files need not be selected for
comparison. Existing version 1 VTK policies and receipts retain their checks
and tolerances; raw outputs require version 2. The build attestation binds the
exact selection contract, including this policy and its workload identities.

Each workload needs one full-float64 diagnostic pair from its native and
resident capture builds, including original bounds and halos. Every measured
pair also needs its required ordinary output comparisons. The diagnostic relink
replaces one explicitly identified source/object and preserves the remaining
objects, ordered compiler/link arguments and recorded environment. Its reviewed
driver source is a trusted validation input: authenticating its hash does not
prove that arbitrary instrumentation preserves numerical semantics. Diagnostic
module inputs are checked against their recorded hashes; existing application
receipts do not establish an original compile-time module inventory. Capture
generation and build attestation alone do not establish application AUTO
qualification. Earlier version 1, 2 or 3 process evidence cannot be relabeled as
version 4 interval evidence.

`python -m compiler.offload.selection_cost builds|evaluate --request FILE
--output FILE` provides the independent collector interface. `builds` accepts
hashed contract JSON plus both application build observations and diagnostic
relink observations; it returns the verified binary paths after rechecking all
receipt inputs. `evaluate` accepts hashed contract and complete-process evidence
JSON. It canonicalizes only observed CPU Fortran option paths using the existing
raw-log verifier's normalization, then verifies the unchanged timing, build and
numerical requirements. Its response includes the canonical evidence and either
a resident/native decision or rejection. Neither command issues an execution
plan or placement attachment; production generation revalidates the evidence
against its own checked interval. The pipeline's `selection_campaign` collector
uses these commands without importing compiler implementation modules. Compiler
verification has a 240-second hard limit; application runs retain their separate,
explicitly supplied runtime limit.

`python -m compiler.offload.application_build preflight|attest` verifies the
consumer's build through a versioned JSON request. Preflight binds original
source destinations, final compiler artifacts and selector-specific macros.
Attestation checks executed compiler/link receipts and the final binary before
delivery. Native observations preserve the original group and need no numerical
GPU workers; static native production needs no selector macro. The recorded
local contract preserves ordered options and explicit inputs and libraries. It
does not claim an inventory of implicit system headers or libraries. Consumers
must preserve authenticated original source paths across calibration and
production and apply edits to separate build destinations.

`--scope-measurement native|resident` with `--form-scopes`,
`--gpu-policy sections` and `--scope-execution reached` emits diagnostic timing
for an already admitted complete
lexical owner. Separate variants retain either the original native callee
(including its unchanged declarations) or the existing resident implementation.
Compiler-owned clock blocks surround authenticated standalone direct calls in
registered ordinary subroutine callers. They include argument evaluation,
automatic storage, original OpenMP joins, publication, cleanup and return;
logging occurs afterward. PURE/ELEMENTAL, concurrent and directive-enclosed
caller positions, ambiguous calls, labels and alternate returns are rejected.
Completed original teams before the call remain supported. Main programs,
function callers and indirect calls are outside this observation subset.

The manifest binds both variants to the same admitted plan and call sites.
`automatic_scope_available` retains its existing admission meaning; measurement
records explicitly set placement and estimate authority to false. There is no
runtime workload key: these observations are neither calibration evidence nor
application performance qualification, and they cannot authorize AUTO
placement. Resident guards may select native work; inspect separate runtime
traces before attributing an observation to GPU execution. This measurement
interface does not change the exact-window selector's admission restrictions.

`compiler.offload.exact_window_calibration` is an independent offline consumer
for a narrow alternative to per-kernel cost models. It compiles the untouched
original Fortran and the compiler's resident implementation of a three-dimensional
producer, current-host reader, native correction, diffusion and accumulation.
Seven rotated pairs and two separate input holdouts measure complete **cold
invocations**, including resident guards, context creation, transfers, native
interruptions, publication and close. Full float64 fields, including halos, must
agree. Output serialization and fixture input initialization are outside this
window interval; these are not complete-application timing results.

After selection, another seven rotated native/AUTO pairs and two input holdouts
measure the final production selector without tracing or calibration overrides.
These independent observations report an actual advantage or a model miss; they
never revise the calibration or force a different decision. Performance reports
use these production samples. Separate diagnostic traces require exactly three
worker launches, full-field agreement, and predicted versus observed complete
transfer bytes; transfer trace-event counts are not individual CUDA API counts.
A changed thread setting must preserve native results with zero device launches.

```bash
python -m compiler.offload.exact_window_calibration \
  --output /tmp/exact-window-observations --sizes 16,192 --architecture sm_86
```

Add `--original-teams` to retain three original four-thread joined loops in the
native control. Complete-window protocol 4 binds the checked plan's portable
execution fingerprint and independent source completion
proofs for every original team and measures their startup, worksharing and joins
inside the same complete invocation. It does not borrow serial or per-loop
coefficients. Runtime thread limits, active-level limits and OpenMP placement
environment are part of the identity; incompatible limits select native before
CUDA work. The independent native measurement and resident selector both query
`omp_get_schedule`: the effective kind (including the monotonic modifier bit)
and chunk size must match, even when `OMP_SCHEDULE` is unchanged. Kinds use an
unsigned 32-bit representation; zero is a valid static chunk size. The independent
native sample uses a C-bound getter because GNU's legacy Fortran getter omits
the monotonic modifier. A production
schedule mismatch selects native before CUDA identity queries or ownership.
Protocols 1–3 observations do not authorize this execution identity. The current
complete-owner admission excludes calls that could change the schedule inside
the measured interval; this check does not admit larger intervals.
Original PRIVATE arrays contribute their entry-time sizes and lower bounds to
the exact cost key, once per array even when several teams reuse it. This also
covers unused PRIVATE storage that may incur allocation costs. Descriptor
inquiries never re-evaluate specification expressions or register the payload.

The experimental `OffloadConfig.exact_window` route consumes the measured record
at one original lexical entry. It selects residency only when every measured
resident interval beats every native interval by at least 5%. It never scales a
pointwise coefficient to a stencil or extrapolates to different dimensions.
Source/plan/runtime/helper identities, compiler options, capture facts, exact
descriptor and integer-control values, original Fortran options, CPU model and placement
and GPU identity bind each observation. A mismatch selects the existing native
path before work. The normal numerical, effect, definition, alias and execution
guards still apply, including publication without replay after a GPU prefix.

The compiler retains this decision as an immutable attachment registered to the
exact checked plan and its evidence request. The cost gate, per-unit pricing,
control emission and diagnostics consume that same attachment; a public record,
copied token or boolean marker cannot substitute for it. Changed evidence
invalidates the attachment, while source, configuration and numerical proofs
still undergo their existing checks. Missing or unprofitable complete-window
evidence preserves the original native routine, including its declarations,
before adding ownership instrumentation. Valid evidence that selects native
still reports its estimate as available. The fixture build consumer retains its
existing runtime linkage even for this static native selection.

For the complete-owner protocol above, the **independent producer is currently
the only supported build consumer**. The original-selection application build
protocol described earlier is separate.
It records and validates exact semantic CUDA/Fortran compile options before
defining the contract-specific selector attestation macro. Without that macro,
the emitted selector returns native. Other consumers must reject this manifest
until they can enforce the complete performance build contract; enforcing only
floating-point flags is insufficient. Calibration overrides exist only in
explicit diagnostic artifacts, never in the resulting automatic selector.

This admission class contains a complete branchless original owner with stable
real array formals, integer input controls, affine counted loops and independently
proved top-level joined OpenMP teams. Payload-dependent bounds/indices, hidden
state controlling work, helpers, numerical-domain-sensitive primitives,
allocation changes and child
owners remain unsupported. In particular, this fixture does **not** establish
automatic offloading or performance for either channel application. The
maintained application controls and their broader cost models remain separate.

## Development and validation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade "pip>=25.1"
python -m pip install --group ./compiler/pyproject.toml:dev

python -m pytest compiler/tests
python -m ruff check compiler
python -m ruff format --check compiler
```

Tests cover lowering, source ordering, inline aliases, scalar lifetimes, ISL
relations against brute-force ordered accesses, rejection without publication,
and deterministic generation. Fusion tests cover scalar motion, shifted conflicts,
unequal domains, conservative accesses, branch boundaries, and signed-stride small
domains compared with source execution. Scheduling tests execute serial/OpenMP
and simulated CUDA coordinate maps for signed strides, tile tails, oversized
tiles, and arbitrary rank; CUDA launches are also compiled with nvcc when available.
Native tests exercise both optimization
levels and compile and compare original Fortran,
generated serial C++, OpenMP, and CUDA using deterministic nonuniform inputs and
absolute tolerance `1e-10` for real values and exact comparison for integers.
Coverage includes fill/scale, non-cubic/singleton/empty domains, sentinel halos,
CDU/CDV/CDW, signed/runtime strides, rank-5 integer arrays, rank-15 bridge
compilation, imperfect nests, fresh host/device values, indirect gathers,
squared indices, and default-real arithmetic. Fallback tests cover recurrences,
scalar live-outs, final induction values, neighboring parallel regions, and CUDA
host/device transitions; invalid source is checked under both execution policies.
Runtime tests also check CPU ownership, selective host/device coherence, pinned
registration lifetimes, stale tokens, shape/size errors, and profiling events with
instrumented CUDA calls. Session tests cover temporary inputs, changed scalars,
selective updates, multiple workspaces, aliases, and lifecycle errors. Generated
CUDA sessions are also executed with a simulated runtime to check exact transfer
counts without a GPU. Resident fallback and logical arguments are compared with
original Fortran across serial C++, OpenMP, and simulated CUDA.
Pool tests distinguish allocation requests from backing reuse, exercise concurrent
callers on multiple simulated devices, and check fallback, cleanup failures,
retention settings, and unchanged transfers. Native pool tests compile both CUDA
default-stream modes; when hardware is available they check retained backing and
cache release/reset, and record cold/warm timings without performance thresholds.
Native compiler absence skips the corresponding capability;
CUDA compilation requires `nvcc`, and execution also requires a usable device. Once
those capabilities are available, build or runtime failures fail the tests. All builds and outputs use temporary directories.

```bash
# Parser/analysis checks without native builds:
python -m pytest compiler/tests -m "not native and not cuda"

# Reached-generation authority and numerical source integration:
python -m pytest compiler/tests/test_reached_consumption.py \
  compiler/tests/test_reached_numerical_effects.py \
  compiler/tests/test_combined_numerical_team.py
```

Shared numerical fixtures live in `tests/support/`. Read-only generated-source
checks may reuse immutable source text and serialized manifests. Mutation and
authority tests construct fresh analysis, builders and proofs; they must not
borrow shared mutable frontend objects. CUDA fixtures reuse the original driver
and strict arithmetic checks through `tests/support/numerical_cuda.py`.

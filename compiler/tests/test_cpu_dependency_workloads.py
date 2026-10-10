"""Predeclared generic fixtures keep exact graphs, domains and source ABIs."""

import ctypes
import hashlib
import json
import math
import shutil
import subprocess
from collections import Counter
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from compiler.frontend import lower_source
from compiler.ir import ArrayAccess, Loop
from compiler.ir.nodes import walk_expr
from compiler.offload.analysis import _workload_features
from compiler.offload.compute_dependencies import analyze_compute_dependencies
from compiler.offload.cpu_dependency_model import coefficient_family
from compiler.offload.cpu_dependency_workloads import (
    BASIS_FAMILIES,
    BASIS_STEPS,
    DOMAIN_FAMILIES,
    PROTOCOL_REQUIREMENTS,
    dependency_recipes,
    native_identity_source,
    recipes_for_precision,
    registry_source,
    write_dependency_registry,
)


@pytest.mark.parametrize("precision", [32, 64])
def test_fixed_recipe_registry_has_stable_predeclared_order_and_independent_roles(precision):
    recipes = recipes_for_precision(precision)
    assert len(recipes) == 26
    assert Counter(recipe.role for recipe in recipes) == {"basis": 12, "structural_holdout": 8, "domain_holdout": 6}
    assert [(recipe.coefficient_family, recipe.width) for recipe in recipes[:12]] == [
        (family, width) for family in BASIS_FAMILIES for width in (1, 4)
    ]
    assert all(recipe.steps == BASIS_STEPS and recipe.helper_form == "inline" for recipe in recipes[:12])
    assert all(recipe.memory_arrays == 3 for recipe in recipes)
    assert all(recipe.graph.available and len(recipe.identity) == 64 for recipe in recipes)
    assert len({recipe.name for recipe in recipes}) == len({recipe.identity for recipe in recipes}) == 26
    assert dependency_recipes(precision) == recipes
    with pytest.raises(FrozenInstanceError):
        recipes[0].name = "changed"


@pytest.mark.parametrize("precision", [32, 64])
def test_basis_each_uses_only_ordinary_and_its_own_primitive(precision):
    for recipe in dependency_recipes(precision)[:12]:
        families = {coefficient_family(operation.family) for operation in recipe.graph.operations}
        assert families <= {"ordinary", recipe.coefficient_family}
        assert recipe.coefficient_family in families
        assert not any(
            operation.constant for operation in recipe.graph.operations if operation.family == recipe.coefficient_family
        )
        if recipe.coefficient_family == "divide_constant":
            divisions = [operation for operation in recipe.graph.operations if operation.family == "divide_constant"]
            assert len(divisions) == recipe.width * BASIS_STEPS
            assert all(float(operation.literal_operands[1][1]) == 3.0 for operation in divisions)
        assert "if(" not in recipe.source_body
        assert "select case" not in recipe.source_body


@pytest.mark.parametrize("precision", [32, 64])
def test_inline_and_separate_helpers_have_same_numerical_graph_but_real_separate_files(precision):
    recipes = dependency_recipes(precision)
    for inline, separate in zip(recipes[12::2], recipes[13::2], strict=True):
        assert inline.helper_form == "inline"
        assert separate.helper_form == "separate"
        assert inline.graph.operations == separate.graph.operations
        assert inline.graph.outputs == separate.graph.outputs
        assert len(inline.fortran_sources) == len(inline.cpp_sources) == 1
        assert len(separate.fortran_sources) == 2
        assert len(separate.cpp_sources) == 1
        helper_file, caller_file = separate.fortran_sources
        assert "pure function" in helper_file[1]
        assert "pure function" not in caller_file[1]
        assert "use h_" in caller_file[1]
        assert "no LTO" in " ".join(PROTOCOL_REQUIREMENTS).replace("LTO prohibited", "no LTO")
        assert "_cpu_worker(" in separate.cpp_sources[0][1]
        assert "extern" not in separate.cpp_sources[0][1].split("static void", 1)[1].split('extern "C"', 1)[0]
        # The compiler-owned graph equivalent normalizes module placement only;
        # actual native source still retains its external module boundary.
        function = lower_source(separate.analysis_source, "evaluate", source_name="independent-check.f90")
        (loop,) = function.body.statements
        assert isinstance(loop, Loop)
        from dataclasses import replace

        transitive = analyze_compute_dependencies(
            replace(
                separate.region,
                body=loop.body,
                private_symbols=tuple(
                    symbol
                    for symbol in function.symbols
                    if symbol not in function.parameters and symbol != loop.iterator
                ),
                captured_symbols=function.parameters,
                loops=(loop,),
            )
        )
        assert transitive.available
        assert transitive.operations == inline.graph.operations
        assert transitive.outputs == inline.graph.outputs
        assert transitive.floating_dtypes == inline.graph.floating_dtypes
        assert transitive.unpriced_numerical_operations == inline.graph.unpriced_numerical_operations


@pytest.mark.parametrize("precision", [32, 64])
def test_generated_workers_use_actual_production_cyclic_dispatch_and_bound_its_identity(precision):
    from compiler.emission.c.declarations import cpp_declaration
    from compiler.emission.common.abi import abi_arguments
    from compiler.emission.common.c_family import cpp_type
    from compiler.emission.cuda.offload import _cpu_worker
    from compiler.offload.analysis import Unit

    for recipe in dependency_recipes(precision):
        function = lower_source(recipe.analysis_source, "evaluate", source_name="worker-authority.f90")
        arguments = abi_arguments(function.parameters)
        signature = ", ".join(cpp_declaration(argument) if argument.symbol.rank else
            f"const {cpp_type(argument.symbol)} &{argument.name}" for argument in arguments)
        # Separate helpers have synthetic inlined scalar names; the recipe's
        # proved inline region remains the exact generated-worker authority.
        expected = "\n".join(_cpu_worker(Unit(0, recipe.region, (), None), signature,
                                       recipe.name + "_cpu_worker"))
        source, = (text for _, text in recipe.cpp_sources)
        assert expected in source
        assert "for (std::size_t flat = tid; flat < total; flat += team)" in source
        assert "omp_get_thread_num(), omp_get_num_threads()" in source
        assert "parallel for" not in source
        assert "schedule(static)" not in source
        assert recipe.generated_dispatch_identity == hashlib.sha256(source.encode()).hexdigest()
        assert recipe.to_dict()["generated_dispatch_identity"] == recipe.generated_dispatch_identity
        assert recipe.to_dict()["generated_helper_form"] == "source_inlined_proven_closure"


def test_ordinary_and_private_holdouts_cover_the_declared_generic_features():
    recipes = dependency_recipes(64)
    ordinary = next(recipe for recipe in recipes if "ordinary_guarded" in recipe.name)
    assert ordinary.workload_class == "ordinary_expression_v2"
    assert {"abs", "min", "max", "negate", "divide_dynamic", "divide_constant"} <= {
        operation.family for operation in ordinary.graph.operations
    }
    private = next(recipe for recipe in recipes if "private_arrays" in recipe.name)
    feature = _workload_features(private.region.private_symbols)
    assert private.workload_class == "fixed_private_array_v2"
    assert feature.classification_complete
    assert feature.private_array_groups == 2
    assert feature.private_array_elements == feature.referenced_private_array_elements == 32
    assert feature.max_private_array_elements == 16
    assert feature.max_private_array_rank == 2
    assert all(len(group.element_offsets) == 16 for group in feature.private_arrays)


def test_every_body_proves_two_input_reads_and_one_output_write_without_cse():
    for recipe in dependency_recipes(64):
        reads = Counter(
            node.symbol.name
            for statement in recipe.region.body.statements
            for node in walk_expr(statement.value)
            if isinstance(node, ArrayAccess)
        )
        writes = Counter(
            statement.target.symbol.name
            for statement in recipe.region.body.statements
            if isinstance(statement.target, ArrayAccess)
        )
        assert reads == {"a": 1, "d": 1}
        assert writes == {"output": 1}
        assert recipe.source_body.count("a(i)") == recipe.source_body.count("d(i)") == 1
        assert recipe.memory_arrays == 3


@pytest.mark.parametrize("precision", [32, 64])
def test_all_recipe_lattices_are_finite_and_reference_results_agree_across_helper_forms(precision):
    for recipe in dependency_recipes(precision):
        for split in ("training", "holdout"):
            inputs = recipe.inputs(55, split=split)
            assert len(inputs.a) == len(inputs.d) == inputs.n == 55
            assert all(math.isfinite(value) for value in (*inputs.a, *inputs.d))
            assert min(inputs.d) >= 2
            assert max(inputs.d) <= 4
            assert all(math.isfinite(value) for value in recipe.reference(inputs))
    for inline, separate in zip(
        dependency_recipes(precision)[12::2], dependency_recipes(precision)[13::2], strict=True
    ):
        assert inline.reference(inline.inputs(55)) == separate.reference(separate.inputs(55))


@pytest.mark.parametrize("precision", [32, 64])
def test_domain_holdouts_actually_reach_declared_lattices_without_contraction(precision):
    for recipe in dependency_recipes(precision):
        if recipe.role != "domain_holdout":
            continue
        family = recipe.coefficient_family
        assert family in DOMAIN_FAMILIES
        training = recipe.inputs(55, split="training").a
        holdout = recipe.inputs(55).a
        assert set(training) != set(holdout)
        if family == "acos":
            assert min(holdout) == -1
            assert max(holdout) == 1
            assert 0 in holdout
            assert dict(recipe.domains)[family] == "finite_unit_interval"
        elif family == "sqrt":
            assert min(holdout) == 0
            assert max(holdout) == 2.0**60
            assert 2.0**-60 in holdout
            assert dict(recipe.domains)[family] == "finite_nonnegative_normal_or_zero"
        else:
            assert max(holdout) == pytest.approx(math.pi, rel=1e-7)
            assert min(holdout) == pytest.approx(-math.pi, rel=1e-7)
            assert dict(recipe.domains)[family] == "finite_pi_interval"


def test_public_records_are_detached_and_source_lines_keep_default_fortran_limits():
    recipes = dependency_recipes(64)
    for recipe in recipes:
        public = json.loads(json.dumps(recipe.to_dict()))
        assert public["memory_arrays"] == 3
        assert public["identity"] == recipe.identity
        assert public["domains"] == dict(recipe.domains)
        for _, source in recipe.fortran_sources:
            assert max(map(len, source.splitlines())) <= 132
        assert len(public["fortran_sources"]) == len(recipe.fortran_sources)
        public["compile_requirements"].append("changed")
        assert tuple(public["compile_requirements"]) != PROTOCOL_REQUIREMENTS


@pytest.mark.parametrize("precision", [True, 0, 16, 128, 32.0])
def test_invalid_precision_never_reuses_cached_fixture(precision):
    with pytest.raises(ValueError, match="precision"):
        dependency_recipes(precision)


@pytest.mark.parametrize("n", [True, -1, 1.0, 2**31])
def test_input_item_count_has_checked_integer_abi(n):
    with pytest.raises(ValueError, match="INTEGER"):
        dependency_recipes(64)[0].inputs(n)


def test_empty_and_mismatched_inputs():
    from dataclasses import replace

    recipe = dependency_recipes(64)[0]
    assert recipe.reference(recipe.inputs(0)) == ()
    with pytest.raises(ValueError, match="split"):
        recipe.inputs(1, split="unexpected")
    with pytest.raises(ValueError, match="precision"):
        recipe.reference(replace(recipe.inputs(1), precision_bits=32))


@pytest.mark.parametrize("precision", [32, 64])
def test_registry_header_has_typed_periodic_inputs_references_and_exact_abi(tmp_path, precision):
    recipes = dependency_recipes(precision)
    generator = "1234abcd" * 8
    header = registry_source(recipes, generator)
    destination = tmp_path / "cpu_dependency_recipes.hpp"
    write_dependency_registry(destination, recipes, generator)
    assert destination.read_text() == header
    assert f"using dependency_real={'float' if precision == 32 else 'double'};" in header
    assert (
        "using dependency_worker=void(*)(int,int,const dependency_real*,const dependency_real*,dependency_real*);"
        in header
    )
    assert "training_reference_count" in header
    assert "holdout_reference_count" in header
    assert generator in header
    assert len([line for line in header.splitlines() if line.startswith('{"dep_')]) == 26
    for index, recipe in enumerate(recipes):
        assert recipe.identity in header
        for split in ("training", "holdout"):
            prefix = f"inline constexpr dependency_real dependency_{index}_{split}_reference[]={{"
            text = next(
                line[len(prefix) :].removesuffix("};") for line in header.splitlines() if line.startswith(prefix)
            )
            reference = tuple(
                float.fromhex(token.removesuffix("f")) if precision == 32 else float.fromhex(token)
                for token in text.split(",")
            )
            assert reference == recipe.reference(recipe.inputs(len(reference), split=split))
    cxx = shutil.which("g++")
    if cxx:
        source = tmp_path / "check.cpp"
        source.write_text(
            '#include "cpu_dependency_recipes.hpp"\n'
            "static_assert(sizeof(dependency_recipes)/sizeof(dependency_recipes[0])==26);\n"
            "static_assert(dependency_fit_sizes[0]==65536);\n"
            "static_assert(dependency_holdout_sizes[1]==524288);\n"
        )
        subprocess.run(
            [cxx, "-std=c++17", "-fsyntax-only", str(source)], check=True, capture_output=True, text=True, timeout=15
        )


@pytest.mark.parametrize("identity", [None, "abc", "g" * 64, "A" * 64])
def test_registry_rejects_unbound_generator_identity(identity):
    with pytest.raises(ValueError, match="SHA256"):
        registry_source(dependency_recipes(64), identity)


def test_registry_rejects_mixed_or_duplicate_precision_and_names():
    first = dependency_recipes(64)[0]
    for recipes in ((), (first, first), (first, dependency_recipes(32)[0])):
        with pytest.raises(ValueError, match="bounded"):
            registry_source(recipes, "a" * 64)


@pytest.mark.native
def test_original_fortran_identity_has_checked_null_terminated_native_abi(tmp_path):
    gfortran = shutil.which("gfortran")
    if not gfortran:
        pytest.skip("native Fortran compiler unavailable")
    source = tmp_path / "identity.f90"
    source.write_text(native_identity_source())
    library = tmp_path / "identity.so"
    subprocess.run(
        [gfortran, "-O0", "-fopenmp", "-shared", "-fPIC", str(source), "-o", str(library)],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    function = ctypes.CDLL(str(library)).fort_cpu_dependency_fortran_identity_v1
    function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    function.restype = None
    version, options = ctypes.create_string_buffer(4096), ctypes.create_string_buffer(4096)
    function(version, options, 4096)
    assert "version" in version.value.decode().lower()
    assert "-O0" in options.value.decode()
    assert "-fopenmp" in options.value.decode()
    for capacity in (0, 1):
        version, options = ctypes.create_string_buffer(b"xx"), ctypes.create_string_buffer(b"yy")
        function(version, options, capacity)
        assert version.raw[:2] == (b"xx" if capacity == 0 else b"\0x")
        assert options.raw[:2] == (b"yy" if capacity == 0 else b"\0y")


@pytest.mark.native
@pytest.mark.parametrize(("precision", "selector"), [(32, "private_arrays"), (64, "dependent_mixed")])
def test_actual_fortran_and_cpp_separate_translation_units_match_typed_reference(tmp_path, precision, selector):
    gfortran, cxx = shutil.which("gfortran"), shutil.which("g++")
    if not gfortran or not cxx:
        pytest.skip("native Fortran and C++ compilers unavailable")
    recipe = next(
        recipe
        for recipe in dependency_recipes(precision)
        if selector in recipe.name and recipe.helper_form == "separate"
    )
    objects = []
    for index, (name, source) in enumerate((*recipe.fortran_sources, *recipe.cpp_sources)):
        path = tmp_path / name
        path.write_text(source)
        output = tmp_path / f"part{index}.o"
        compiler = gfortran if path.suffix == ".f90" else cxx
        command = [compiler, "-O0", "-fPIC", "-fopenmp", "-c", str(path), "-o", str(output)]
        if compiler == cxx:
            command += ["-std=c++17", "-I", str(Path(__file__).parents[1] / "runtime")]
        subprocess.run(command, cwd=tmp_path, check=True, capture_output=True, text=True, timeout=30)
        objects.append(str(output))
    library = tmp_path / "fixture.so"
    subprocess.run(
        [gfortran, "-shared", "-fopenmp", *objects, "-lstdc++", "-o", str(library)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    loaded = ctypes.CDLL(str(library))
    scalar = ctypes.c_float if precision == 32 else ctypes.c_double
    inputs = recipe.inputs(55)
    expected = recipe.reference(inputs)
    a, d = (scalar * inputs.n)(*inputs.a), (scalar * inputs.n)(*inputs.d)
    for entry in (recipe.native_serial_entry, recipe.native_fork_join_entry, recipe.generated_entry):
        worker = getattr(loaded, entry)
        worker.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(scalar),
            ctypes.POINTER(scalar),
            ctypes.POINTER(scalar),
        ]
        worker.restype = None
        output = (scalar * inputs.n)()
        worker(inputs.n, 4, a, d, output)
        assert list(output) == pytest.approx(
            expected, rel=2e-6 if precision == 32 else 2e-13, abs=2e-7 if precision == 32 else 2e-14
        )

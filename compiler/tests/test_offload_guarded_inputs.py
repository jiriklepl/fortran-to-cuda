"""Native guards must protect scalar inputs before the numerical value ABI."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from compiler.driver.options import CompilerOptions
from compiler.driver.pipeline import prepare_function
from compiler.emission import generate_sources, read_common_header
from compiler.emission.c.declarations import cpp_declaration
from compiler.emission.common.abi import abi_arguments
from compiler.emission.common.c_family import cpp_type
from compiler.emission.cuda.offload import generate_offload
from compiler.frontend import lower_file
from compiler.offload.config import OffloadConfig


def generate(tmp_path, declaration, body, policy="sections"):
    path = tmp_path / "guarded.f90"
    parameters = "a,n,flag,protected" if declaration else "a,n,flag"
    path.write_text("""module guarded_inputs
contains
subroutine advance(""" + parameters + """)
real(8),intent(inout)::a(:)
integer,intent(in)::n
logical,intent(in)::flag
""" + (declaration or "") + "\ninteger::i,j\nreal(8)::temporary\n" + body + "\nend subroutine\nend module\n")
    function, plan = prepare_function(lower_file(path, "advance"), options=CompilerOptions(gpu_policy=policy))
    return function, generate_offload(function, plan, OffloadConfig(policy))


GUARDED = [
    ("integer,intent(in)::protected", "do i=1,n\nif(flag) a(i+protected)=1.0_8\nend do"),
    ("real(8),intent(in)::protected", "do i=1,n\nif(flag) a(i)=protected\nend do"),
    ("logical,intent(in)::protected", "do i=1,n\nif(flag) then\nif(protected) a(i)=1.0_8\nend if\nend do"),
    ("integer,intent(in)::protected", "do i=1,n\nif(flag) then\ndo j=1,protected\na(i)=a(i)+1.0_8\nend do\nend if\nend do"),
    ("real(8),intent(in)::protected", "do i=1,n\ntemporary=0.0_8\ndo j=1,0\ntemporary=temporary+protected\nend do\na(i)=temporary\nend do"),
]


@pytest.mark.parametrize("policy", ["sections", "chunked"])
@pytest.mark.parametrize(("declaration", "body"), GUARDED)
def test_protected_inputs_make_forced_policies_native_only(tmp_path, policy, declaration, body):
    _, generated = generate(tmp_path, declaration, body, policy)
    assert not generated.report["analysis"]["available"]
    assert "protected" in generated.report["analysis"]["reason"]
    assert "conditional or empty retained-loop" in generated.report["analysis"]["reason"]
    assert generated.report["supported_strategies"] == ["native"]
    assert generated.query_scalars == frozenset()
    assert generated.decision_body[-1] == "return 0;"
    assert not generated.helpers


@pytest.mark.parametrize("policy", ["sections", "chunked"])
@pytest.mark.parametrize("body", [
    "do i=1,n\nif(flag) a(i)=1.0_8\nend do",
    "do i=1,n\nif(flag) a(i)=real(i+n,8)\nend do",
    "do i=1,n\ntemporary=protected\nif(flag) a(i)=temporary\nend do",
])
def test_unconditional_inputs_and_index_only_holes_remain_supported(tmp_path, policy, body):
    declaration = "real(8),intent(in)::protected" if "protected" in body else None
    _, generated = generate(tmp_path, declaration, body, policy)
    assert generated.report["analysis"]["available"], generated.report
    assert policy in generated.report["supported_strategies"]


@pytest.mark.parametrize("policy", ["sections", "chunked"])
def test_unused_scalar_in_branch_free_entry_keeps_original_native_call(tmp_path, policy):
    _, generated = generate(tmp_path, "integer,intent(in)::protected",
                            "do i=1,n\na(i)=real(n,8)\nend do", policy)
    assert generated.report["supported_strategies"] == ["native"]
    assert "unused scalar inputs" in generated.report["analysis"]["reason"]
    assert "protected" in generated.report["analysis"]["reason"]
    assert generated.decision_body[-1] == "return 0;"


@pytest.mark.native
@pytest.mark.parametrize("policy", ["sections", "chunked"])
@pytest.mark.parametrize("kind", ["offset", "rhs", "unused"])
def test_generated_query_preserves_unreadable_conditional_and_unused_inputs(tmp_path, policy, kind):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("C++ compiler required")
    rhs = kind == "rhs"
    declaration, body = GUARDED[int(rhs)]
    if kind == "unused":
        body = "do i=1,n\nif(flag) a(i)=1.0_8\nend do"
    function, generated = generate(tmp_path, declaration, body, policy)
    abi = abi_arguments(function.parameters)
    signature = ", ".join(cpp_declaration(a) if a.symbol.rank else
                          f"const {cpp_type(a.symbol)} &{a.name}" for a in abi)
    runtime = Path(__file__).resolve().parents[1] / "runtime/offload.hpp"
    query = "int query(" + signature + ") {\n" + "\n".join(generated.decision_body) + "\n}\n"
    dtype = "double" if rhs else "int"
    assignment = "a[i-1]=protected_value" if rhs else "a[i+protected_value-1]=1"
    if kind == "unused":
        assignment = "a[i-1]=1"
        assert "unused scalar inputs" in generated.report["analysis"]["reason"]
    source = tmp_path / "query.cpp"
    source.write_text("""#include <sys/mman.h>
#include <cstdlib>
#include <cstdio>
""" + f'#include "{runtime}"\nusing namespace generated_kernels;\n' + generated.helpers + query + f"""
void native(double *a,const int& n,const bool& flag,const {dtype}& protected_value) {{
    for(int i=1;i<=n;++i) if(flag) {assignment};
}}
int main() {{
    void *page=mmap(nullptr,4096,PROT_NONE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    if(page==MAP_FAILED) return 1;
    const auto& protected_value=*static_cast<{dtype}*>(page);
    double a[4]={{7,7,7,7}};
    const int n=1;
    const bool flag=false;
    if(query(a,4,n,flag,protected_value)) return 2;
    native(a,n,flag,protected_value);
    for(double value:a) if(value!=7) return 3;
    munmap(page,4096);
    std::puts("GUARDED_INPUT_QUERY_PASS");
}}
""")
    commands = [[compiler, "-std=c++17", "-O2", "-fopenmp", str(source), "-o", str(tmp_path / "query")],
                [str(tmp_path / "query")]]
    (tmp_path / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    for command in commands:
        result = subprocess.run(command, env=dict(os.environ, FORT_OFFLOAD_TRACE="1"),
                                capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr
    (tmp_path / "query-run.log").write_text(result.stdout + result.stderr)
    assert "GUARDED_INPUT_QUERY_PASS" in result.stdout
    assert "mode=native" in result.stderr


@pytest.mark.native
@pytest.mark.cuda
@pytest.mark.parametrize("policy", ["sections", "chunked"])
@pytest.mark.parametrize("rhs", [False, True])
def test_cuda_collective_query_does_not_read_protected_scalar(tmp_path, policy, rhs):
    nvcc = shutil.which(os.environ.get("NVCC", "/usr/local/cuda/bin/nvcc"))
    host = shutil.which(os.environ.get("CUDAHOSTCXX", "g++-14"))
    if not nvcc or not host:
        pytest.skip("requires CUDA and GNU C++")
    generate(tmp_path, *GUARDED[int(rhs)], policy)
    function, plan = prepare_function(lower_file(tmp_path / "guarded.f90", "advance"),
                                     options=CompilerOptions(gpu_policy=policy))
    generated = generate_sources(function, plan, common_header="runtime.hpp",
                                 offload_config=OffloadConfig(policy, collective=True))
    assert generated.offload["supported_strategies"] == ["native"]
    query = "cpp_" + generated.offload["native_fallback_query"]
    (tmp_path / "runtime.hpp").write_text(read_common_header())
    (tmp_path / "generated.cu").write_text(generated.cuda)
    dtype = "double" if rhs else "int"
    (tmp_path / "driver.cpp").write_text(f"""#include <sys/mman.h>
#include <cstdlib>
#include <cstdio>
#include <cstddef>
extern "C" int {query}(double*,std::size_t,const int&,const bool&,const {dtype}&);
int main() {{
    void *page=mmap(nullptr,4096,PROT_NONE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    if(page==MAP_FAILED) return 1;
    const auto& protected_value=*static_cast<{dtype}*>(page);
    double a[4]={{7,7,7,7}};
    const int n=1;
    const bool flag=false;
    #pragma omp parallel num_threads(4)
    {{ if({query}(a,4,n,flag,protected_value)) std::abort(); }}
    for(double value:a) if(value!=7) return 2;
    munmap(page,4096);
    std::puts("CUDA_GUARDED_QUERY_PASS");
}}
""")
    commands = [
        [nvcc, "-O2", "-std=c++17", "-Xcompiler=-fopenmp", "-arch=sm_86", "-ccbin", host,
         "-c", "generated.cu", "-o", "generated.o"],
        [host, "-O2", "-std=c++17", "-fopenmp", "driver.cpp", "generated.o",
         "-L" + str(Path(nvcc).resolve().parents[1] / "lib64"), "-lcudart", "-o", "query"],
        [str(tmp_path / "query")],
    ]
    (tmp_path / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    environment = dict(os.environ, OMP_DYNAMIC="FALSE", FORT_OFFLOAD_TRACE="1", FORT_RUNTIME_TRACE="1")
    for command in commands:
        result = subprocess.run(command, cwd=tmp_path, env=environment,
                                capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
    (tmp_path / "query-run.log").write_text(result.stdout + result.stderr)
    assert "CUDA_GUARDED_QUERY_PASS" in result.stdout
    assert result.stderr.count("FORT_OFFLOAD entry=") == 1
    assert "mode=native" in result.stderr
    assert "FORT_RUNTIME" not in result.stderr

"""Joined native teams keep proved PRIVATE storage local to their operation."""

from compiler.tests.test_source_scopes import FACT, PROGRAM, generate


def test_fixed_private_native_array_requires_no_capture_fact(tmp_path):
    source = PROGRAM.replace("integer,intent(in)::n\ncall producer(a,b,n)",
                             "integer,intent(in)::n\ninteger::i\nreal(8)::scratch(2)\ncall producer(a,b,n)")
    source = source.replace("call transform(b)\ncall consumer", """!$omp parallel private(i,scratch) shared(a,b,n)
!$omp do
do i=1,n
scratch(1)=a(i)
scratch(2)=2*a(i)
b(i)=b(i)+sum(scratch)
enddo
!$omp end do
!$omp end parallel
call consumer""")
    original, output, manifest = generate(tmp_path, source, facts={
        "schema_version": 1, "participation": "serial",
        "captures": {"argument::a": FACT, "argument::b": FACT, "argument::out": FACT}})
    scope, = manifest["scopes"]
    operation, = [item for item in scope["native_operations"] if item["kind"] == "joined native OpenMP"]
    assert set(operation["private_resources"]) == {"original::step::i", "original::step::scratch"}
    assert set(operation["resources"]) == {"argument::a", "argument::b"}
    assert operation["sections"]["available"]
    assert "original::step::scratch" not in scope["ownership"]["retained_resources"]
    generated = (output / manifest["sources"][str(original)]["replacement"]).read_text()
    owner = generated[generated.index("subroutine " + scope["owner"]):generated.index("end subroutine " + scope["owner"])]
    assert ":: scratch(1:2)" in owner
    assert "private(i,scratch)" in owner.lower()
    assert original.read_text() == source

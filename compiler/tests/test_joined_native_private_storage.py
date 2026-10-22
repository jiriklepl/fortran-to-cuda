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
! Keep this an original native operation: short fixed SUM is now numerical.
b(i)=b(i)+sum(scratch)+sum(a)
enddo
!$omp end do
!$omp end parallel
call consumer""")
    original, output, manifest = generate(tmp_path, source, scope_execution="reached", facts={
        "schema_version": 1, "participation": "serial",
        "captures": {"argument::a": FACT, "argument::b": FACT, "argument::out": FACT}})
    scope, = manifest["scopes"]
    parent, = [item for item in scope["checked_execution_plan"]["procedures"]
               if item["procedure"] == "original::step"]
    operation, = [item for unit in parent["units"] for item in unit["operations"]["native_operations"]
                  if item["kind"] == "joined native OpenMP"]
    assert set(operation["private_resources"]) == {"original::step::i", "original::step::scratch"}
    assert set(operation["resources"]) == {"argument::a", "argument::b"}
    assert operation["sections"]["available"]
    private, = operation["completion"]["original_private_arrays"]
    assert private["resource"] == "original::step::scratch"
    assert private["contents"] == "undefined original OpenMP PRIVATE contents; no capture or registration"
    assert operation["completion"]["retains_original_team_and_directives"]
    assert "original::step::scratch" not in scope["ownership"]["retained_resources"]
    generated = (output / manifest["sources"][str(original)]["replacement"]).read_text()
    start = generated.index("subroutine step(")
    owner = generated[start:generated.index("end subroutine", start)]
    assert "real(8)::scratch(2)" in owner
    assert "private(i,scratch)" in owner.lower()
    assert original.read_text() == source

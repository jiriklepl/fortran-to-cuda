"""Source ownership survives reached guards and coherent native operations."""

from __future__ import annotations

import pytest

from compiler.tests.test_source_scopes import FACT, PROGRAM, assert_uncalibrated_native, generate

TREE = PROGRAM.replace("integer,intent(in)::n\ncall producer(a,b,n)",
                       "integer,intent(inout)::n\nlogical::flag\ncall producer(a,b,n)")
TREE = TREE.replace("subroutine step(a,b,out,n)\nreal(8),intent(in)::a(:)\n"
                    "real(8),intent(out)::b(:),out(:)",
                    "subroutine step(a,b,out,n)\nreal(8),intent(in)::a(-2:)\n"
                    "real(8),intent(inout)::b(-2:),out(-2:)")
TREE = TREE.replace("call transform(b)\ncall consumer(a,b,out,n)\nend subroutine",
                    "flag=sum(b)>0\nif(flag) then\n"
                    "  if(b(-2)>0) b(-2)=b(-2)+7\n"
                    "else if(n<0) then\n  call consumer(a,b,out,n)\n"
                    "else\n  call transform(b)\nendif\nn=n-1\n"
                    "call consumer(a,b,out,n)\nend subroutine")


def generate_tree(directory, source=TREE, *, mode="sections", profile=None):
    facts = {"schema_version": 1, "participation": "serial",
             "captures": {"argument::a": FACT, "argument::b": FACT, "argument::out": FACT}}
    original, output, manifest = generate(directory, source, mode=mode, facts=facts, profile=profile)
    replacement = manifest["sources"].get(str(original), {}).get("replacement")
    text = (output / replacement).read_text() if replacement else original.read_text()
    return original, output, manifest, text


def owner_text(manifest, text):
    owner, = manifest["scopes"]
    name = owner["owner"]
    return owner, text[text.index("subroutine " + name):text.index("end subroutine " + name)]


def test_missing_calibration_selects_original_before_creating_context(tmp_path):
    _, _, manifest, text = generate_tree(tmp_path, mode="auto")
    assert_uncalibrated_native(manifest)
    assert text == TREE


def test_one_available_alternative_preserves_reached_automatic_planning():
    from types import SimpleNamespace

    from compiler.scopes.source import ScopeBuilder

    builder = object.__new__(ScopeBuilder)
    builder.config = SimpleNamespace(policy="auto")
    available = SimpleNamespace(scoped={"planning": {
        "available": True, "profile_available": True, "reason": None, "profile_reason": None}})
    missing = SimpleNamespace(scoped={"planning": {
        "available": False, "profile_available": True, "reason": "numerical calibration missing", "profile_reason": None}})
    builder.numerical = lambda procedure: {"heavy": missing, "simple": available}[procedure]
    assert builder.automatic_native_preflight({"heavy", "simple"}) is None
    assert "numerical calibration missing" in builder.automatic_native_preflight({"heavy"})


@pytest.mark.parametrize("statement", ["deallocate(b)", "return", "exit", "call unknown(b)"])
def test_unknown_or_lifetime_operations_end_structured_ownership(tmp_path, statement):
    source = TREE.replace("n=n-1", statement)
    if statement.startswith("deallocate"):
        source = source.replace("real(8),intent(inout)::b(-2:),out(-2:)",
                                "real(8),allocatable,intent(inout)::b(:)\nreal(8),intent(inout)::out(-2:)")
    _, _, manifest = generate(tmp_path, source, facts={"schema_version": 1, "participation": "serial",
                                                    "captures": {"argument::a": FACT, "argument::b": FACT, "argument::out": FACT}})
    assert not any(scope.get("ownership") and scope["last_line"] > source.splitlines().index(statement) + 1
                   and scope["first_line"] < source.splitlines().index(statement) + 1 for scope in manifest["scopes"])


JOINED = TREE.replace("real(8),intent(out)::b(:)", "real(8),intent(inout)::b(:)").replace("logical::flag", "logical::flag\ninteger::i").replace(
    "n=n-1\ncall consumer", "!$omp parallel private(i) shared(b,n)\n!$omp do\n"
    "do i=-2,n-3\nb(i)=b(i)+1\nenddo\n!$omp end do nowait\n"
    "!$omp end parallel\nn=n-1\ncall consumer")

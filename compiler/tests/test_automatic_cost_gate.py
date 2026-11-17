"""Missing cost evidence retains authenticated source without speculative work."""

from hashlib import sha256


from compiler.driver.options import CompilerOptions
from compiler.offload.config import OffloadConfig
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_numerical_sources import package
from compiler.tests.test_source_scopes import FACT, PROGRAM


def builder(tmp_path, *, execution="bounded", config=None, normalized=False, entry="step", corrupt=False):
    if normalized:
        path, lowered, document = package(tmp_path)
    else:
        path = lowered = tmp_path / "source.f90"
        path.write_text(PROGRAM)
        document = None
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
             "captures": {root: FACT for root in
                          ("argument::a", "argument::b", "argument::out", "settings::weights")}}
    if corrupt:
        facts["sources"][str(path)] = "0" * 64
    value = ScopeBuilder([path], entry, facts=facts, options=CompilerOptions(),
                         config=config or OffloadConfig("auto", scope_execution=execution),
                         numerical_sources=document)
    return path, lowered, value


def assert_native_only(outputs, report):
    assert set(outputs) == {"scope-manifest.json"}
    assert report["scope_count"] == 0
    assert not report["automatic_estimate_available"]
    for key in ("scopes", "source_edits", "sources", "build_sources", "artifacts_sha256", "numerical_decisions"):
        assert not report[key], key
    assert report["runtime"] is None
    assert report["native_effects"]["closure_materialization"] == "not_requested"
    assert "no calibration profile or exact-window evidence" in report["boundaries"][0]["reason"]


def test_incompatible_profile_reason_is_preserved(tmp_path):
    config = OffloadConfig("auto", profile={"schema_version": 0})
    assert config.profile is None
    _, _, candidate = builder(tmp_path, config=config)
    outputs, report = candidate.run()
    assert_native_only(outputs, report)
    assert config.profile_reason in report["boundaries"][0]["reason"]

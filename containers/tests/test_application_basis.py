"""Check clean source preparation with real, small Git repositories."""
import hashlib
import importlib.util
import json
import os
import subprocess
import tomllib
from pathlib import Path

import pytest

CONTAINERS = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("container_application_basis", CONTAINERS / "application_basis.py")
basis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(basis)


@pytest.fixture
def sources(tmp_path):
    environment = {**os.environ, "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null",
                   "GIT_AUTHOR_NAME": "basis test", "GIT_COMMITTER_NAME": "basis test",
                   "GIT_AUTHOR_EMAIL": "basis@example.invalid", "GIT_COMMITTER_EMAIL": "basis@example.invalid"}

    def git(root, *args):
        return subprocess.check_output(["git", "-c", "commit.gpgsign=false", "-C", str(root), *args],
                                       env=environment, text=True, stderr=subprocess.DEVNULL).strip()

    upstream = tmp_path / "upstream"
    for name in ("elmm", "PoisFFT"):
        root = upstream / name
        (root / "src").mkdir(parents=True)
        (root / "src/SConstruct").write_text("# unchanged build\n")
        (root / "README").write_text("tracked provenance input\n")
        if name == "elmm":
            (root / "src/pressure.f90").write_text("pinned pressure sum\n")
        git(root, "init", "-q")
        git(root, "add", ".")
        git(root, "commit", "-qm", "pinned upstream")
    elmm = upstream / "elmm"
    pinned = git(elmm, "rev-parse", "HEAD")
    # The upstream checkout may have moved on; the basis is the pinned revision.
    (elmm / "src/pressure.f90").write_text("later pressure sum\n")
    git(elmm, "commit", "-qam", "later upstream")
    contract = {"schema_version": 2,
                "source_sha256": {"elmm/src/pressure.f90": hashlib.sha256(b"pinned pressure sum\n").hexdigest()},
                "include_sha256": {}, "build_sha256": {
                    name + "/src/SConstruct": basis.digest(upstream / name / "src/SConstruct") for name in ("elmm", "PoisFFT")},
                "repositories": {"elmm": {"revision": pinned},
                                 "PoisFFT": {"revision": git(upstream / "PoisFFT", "rev-parse", "HEAD")}}}
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract))
    return upstream, tmp_path / "prepared", contract_path, git


def test_checks_out_the_pinned_revision_and_reuses_clean_basis(sources):
    upstream, output, contract, git = sources
    later = (upstream / "elmm/src/pressure.f90").read_bytes()
    assert basis.prepare_application_basis(upstream, output, contract_path=contract) == output
    assert git(output / "elmm", "rev-parse", "HEAD") == json.loads(contract.read_text())["repositories"]["elmm"]["revision"]
    assert (output / "elmm/src/pressure.f90").read_text() == "pinned pressure sum\n"
    assert (upstream / "elmm/src/pressure.f90").read_bytes() == later
    assert git(upstream / "elmm", "status", "--porcelain") == ""
    receipt = basis.preparation_receipt(output)
    before = receipt.read_bytes()
    assert json.loads(before)["status"] == "prepared_clean_common_basis"
    assert basis.prepare_application_basis(upstream, output, contract_path=contract) == output
    assert receipt.read_bytes() == before


@pytest.mark.parametrize("relative", ["src/pressure.f90", "README", "untracked-input.f90"])
def test_existing_source_drift_is_rejected_without_repair(sources, relative):
    upstream, output, contract, _git = sources
    basis.prepare_application_basis(upstream, output, contract_path=contract)
    receipt = basis.preparation_receipt(output).read_bytes()
    changed = output / "elmm" / relative
    changed.write_text("unexpected source change\n")
    with pytest.raises(ValueError, match="hash mismatch|changed tracked inputs|untracked input"):
        basis.prepare_application_basis(upstream, output, contract_path=contract)
    assert changed.read_text() == "unexpected source change\n"
    assert basis.preparation_receipt(output).read_bytes() == receipt


def test_existing_unrelated_directory_is_not_reset_or_overwritten(sources):
    upstream, output, contract, _git = sources
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("original")
    with pytest.raises(ValueError, match="app_root must contain"):
        basis.prepare_application_basis(upstream, output, contract_path=contract)
    assert sentinel.read_text() == "original"
    assert not basis.preparation_receipt(output).exists()


def test_sources_that_differ_from_the_contract_fail_and_leave_evidence(sources):
    upstream, output, contract, _git = sources
    changed = json.loads(contract.read_text())
    changed["source_sha256"]["elmm/src/pressure.f90"] = "0" * 64
    contract.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="hash mismatch"):
        basis.prepare_application_basis(upstream, output, contract_path=contract)
    assert json.loads(basis.preparation_receipt(output).read_text())["status"] == "failed"


def test_changed_include_rejects_contract_before_rebinding(tmp_path):
    for name in ("ELMM", "PoisFFT"):
        (tmp_path / name / "src").mkdir(parents=True)
        (tmp_path / name / "src/SConstruct").write_text("pinned build\n")
    source = tmp_path / "ELMM/src/subgrid.f90"
    include = tmp_path / "ELMM/src/kinds.inc"
    source.write_text("pinned source\n")
    include.write_text("pinned include\n")
    contract = {"schema_version": 2, "source_sha256": {"elmm/src/subgrid.f90": basis.digest(source)},
                "include_sha256": {"elmm/src/kinds.inc": basis.digest(include)}, "build_sha256": {}}
    basis.verify_contract(tmp_path, contract)
    include.write_text("changed caller proof\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        basis.verify_contract(tmp_path, contract)
    with pytest.raises(ValueError, match="unsupported or incomplete source basis"):
        basis.verify_contract(tmp_path, {**contract, "schema_version": 1})


def test_contract_cannot_escape_application_copy(tmp_path):
    with pytest.raises(ValueError, match="invalid portable"):
        basis.contract_path({"elmm": tmp_path}, "elmm/../outside.f90")


def test_whole_checkout_rejects_wrong_pin_dirty_or_untracked_inputs(tmp_path):
    (tmp_path / "src").mkdir()
    source = tmp_path / "src/native_only.f90"
    source.write_text("outside the reviewed source closure\n")
    contract = {"repositories": {"elmm": {"url": "example.invalid/elmm", "revision": "pinned"}}}
    status_key = ("status", "--porcelain=v1", "--untracked-files=no", "--ignore-submodules=none")
    answers = {("rev-parse", "--show-toplevel"): str(tmp_path),
               ("rev-parse", "--verify", "HEAD"): "pinned",
               status_key: "",
               ("ls-files", "--others", "-z"): "scratch.o\0",
               ("ls-files", "-z"): "src/native_only.f90\0"}

    def command(argv, **_kwargs):
        return answers[tuple(argv[3:])], 0.

    provenance = basis.repository_provenance({"elmm": tmp_path}, contract, command)
    assert provenance["repositories"]["elmm"]["tracked_sha256"] == {"src/native_only.f90": basis.digest(source)}
    answers[("rev-parse", "--verify", "HEAD")] = "other"
    with pytest.raises(ValueError, match="revision mismatch"):
        basis.repository_provenance({"elmm": tmp_path}, contract, command)
    answers[("rev-parse", "--verify", "HEAD")] = "pinned"
    answers[status_key] = " M src/native_only.f90"
    with pytest.raises(ValueError, match="changed tracked"):
        basis.repository_provenance({"elmm": tmp_path}, contract, command)
    answers[status_key] = ""
    answers[("ls-files", "--others", "-z")] = "scratch.o\0src/ignored.f90\0"
    with pytest.raises(ValueError, match="untracked input"):
        basis.repository_provenance({"elmm": tmp_path}, contract, command)


def test_the_basis_pins_the_revisions_elmm_s_description_builds_from():
    inputs = tomllib.loads((CONTAINERS.parent / "benchmarks/elmm/elmm.toml").read_text())["inputs"]
    contract = json.loads(basis.DEFAULT_CONTRACT.read_text())["repositories"]
    assert contract == {name: {"url": inputs[name]["url"], "revision": inputs[name]["revision"]} for name in contract}

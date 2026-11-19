"""Check clean deterministic source preparation with real, small Git repositories."""
import hashlib
import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest

CONTAINERS = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("container_application_basis", CONTAINERS / "application_basis.py")
basis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(basis)


@pytest.fixture
def sources(tmp_path, monkeypatch):
    environment = {**os.environ, "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null",
                   "GIT_AUTHOR_NAME": basis.BASIS["name"], "GIT_COMMITTER_NAME": basis.BASIS["name"],
                   "GIT_AUTHOR_EMAIL": basis.BASIS["email"], "GIT_COMMITTER_EMAIL": basis.BASIS["email"],
                   "GIT_AUTHOR_DATE": basis.BASIS["date"], "GIT_COMMITTER_DATE": basis.BASIS["date"]}

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
            (root / "src/pressure.f90").write_text("original pressure sum\n")
        git(root, "init", "-q")
        git(root, "add", ".")
        git(root, "commit", "-qm", "pinned upstream")
    elmm = upstream / "elmm"
    base = git(elmm, "rev-parse", "HEAD")
    original = (elmm / "src/pressure.f90").read_bytes()
    (elmm / "src/pressure.f90").write_text("approved fixed order pressure sum\n")
    approved = (elmm / "src/pressure.f90").read_bytes()
    patch = tmp_path / "pressure.patch"
    patch.write_text(subprocess.check_output(["git", "-C", str(elmm), "diff"], text=True))
    git(elmm, "add", ".")
    # Normal git commit independently creates the expected commit identity.
    git(elmm, "commit", "-qm", basis.BASIS["message"])
    revision, tree = git(elmm, "rev-parse", "HEAD"), git(elmm, "rev-parse", "HEAD^{tree}")
    git(elmm, "checkout", "--detach", base)
    identity = {**basis.BASIS, "base": base, "revision": revision, "tree": tree,
                "original_pressure_sha256": hashlib.sha256(original).hexdigest(),
                "pressure_sha256": hashlib.sha256(approved).hexdigest(), "patch_sha256": basis.digest(patch)}
    monkeypatch.setattr(basis, "BASIS", identity)
    monkeypatch.setattr(basis, "PATCH", patch)
    contract = {"schema_version": 2, "source_sha256": {"elmm/src/pressure.f90": identity["pressure_sha256"]},
                "include_sha256": {}, "build_sha256": {
                    name + "/src/SConstruct": basis.digest(upstream / name / "src/SConstruct") for name in ("elmm", "PoisFFT")},
                "repositories": {"elmm": {"revision": revision},
                                 "PoisFFT": {"revision": git(upstream / "PoisFFT", "rev-parse", "HEAD")}}}
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract))
    return upstream, tmp_path / "prepared", contract_path, git


def test_reconstructs_only_approved_file_and_reuses_clean_basis(sources):
    upstream, output, contract, git = sources
    original = (upstream / "elmm/src/pressure.f90").read_bytes()
    assert basis.prepare_application_basis(upstream, output, contract_path=contract) == output
    assert git(output / "elmm", "rev-parse", "HEAD") == basis.BASIS["revision"]
    assert git(output / "elmm", "diff", "--name-only", basis.BASIS["base"], "HEAD") == "src/pressure.f90"
    assert (upstream / "elmm/src/pressure.f90").read_bytes() == original
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


def test_changed_patch_fails_before_creating_output(sources):
    upstream, output, contract, _git = sources
    basis.PATCH.write_text("unexpected patch")
    with pytest.raises(ValueError, match="source patch changed"):
        basis.prepare_application_basis(upstream, output, contract_path=contract)
    assert not output.exists()


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

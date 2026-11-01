"""Real GPU/native/GPU execution observes exact translated metadata each call."""

import json
import re

import pytest

from compiler.tests import test_joined_team_cuda as base


def translated_source():
    source = base.indirect_source("indirect_faces")
    start = source.index("subroutine setup_sites(n,m)")
    end = source.index("end subroutine", start) + len("end subroutine")
    source = source[:start] + """subroutine setup_sites(n,m)
integer,intent(in)::n,m
integer::q
if(allocated(sites))deallocate(sites)
if(n>0)then
allocate(sites(m))
do q=1,m
sites(q)%row=-2
sites(q)%column=q+2
enddo
else
allocate(sites(0))
endif
end subroutine""" + source[end:]
    original = "b(sites(q)%row,sites(q)%column)=b(sites(q)%row,sites(q)%column)+a(sites(q)%row,sites(q)%column)"
    translated = original.replace("%row", "%row+1")
    assert source.count(original) == 1
    return source.replace(original, original + "\n" + translated)


def translated_driver():
    driver = base.driver_case("indirect_faces")
    driver = driver.replace("step,visits,setup_sites", "step,visits,setup_sites,sites")
    driver = driver.replace("flag=repetition==2", "sites%row=-3+repetition\nflag=.true.")
    return driver


@pytest.fixture(scope="module")
def translated_binaries(tmp_path_factory):
    source, driver = translated_source(), translated_driver()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(base, "CASES", ("indirect_faces",))
        patch.setattr(base, "source_case", lambda _label: source)
        patch.setattr(base, "driver_case", lambda _label: driver)
        return base.joined_binaries.__wrapped__(tmp_path_factory)


@pytest.mark.cuda
def test_translated_native_points_preserve_fields_halos_and_exact_copy_ledger(translated_binaries):
    result, manifest = base.execute(translated_binaries, "indirect_faces")
    _, build, _expected, _ = translated_binaries["indirect_faces"]
    output = build.parent / "output"
    emitted = "\n".join((output / item["path"]).read_text()
                        for item in manifest["build_sources"] if item["language"] == "fortran")
    assert "_cache_lows" in emitted
    ledger = []
    for ordinal, n, m, trace in base.calls(result.stderr):
        launches = list(re.finditer(r"^FORT_SCOPED launch ", trace, re.M))
        assert len(launches) == (2 if n else 0)
        corrected = 2 * m * 8 if n else 0
        assert base.transfer_bytes(trace, "upload") == n * m * 8 + corrected
        assert base.transfer_bytes(trace, "download") == 2 * n * m * 8
        if n:
            between = trace[launches[0].end():launches[1].start()]
            assert base.transfer_bytes(between, "download") == corrected
            assert base.transfer_bytes(between, "upload") == corrected
        ledger.append({"call": ordinal, "n": n, "m": m, "metadata_row": -2 if ordinal % 2 else -1,
                       "corrected_bytes_each_direction_between_kernels": corrected,
                       "upload_bytes": base.transfer_bytes(trace, "upload"),
                       "download_bytes": base.transfer_bytes(trace, "download")})
    (build / "translated-copy-ledger.json").write_text(json.dumps(ledger, indent=2) + "\n")

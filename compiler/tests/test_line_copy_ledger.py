"""The geometry oracle rejects redundant copies under the old allocation cap."""

import pytest

from compiler.tests.test_line_source_tasks_cuda import checked_mask_upload_phases


def upload(buffer, size, *, generation=1, handle=4):
    return f"FORT_SCOPED upload buffer={buffer} generation={generation} buffer_handle={handle} bytes={size} context=2"


def fixture():
    return [
        upload(2, 780),
        "FORT_SCOPED external_commit bytes=2",
        "FORT_SCOPED launch bytes=0",
        "FORT_SCOPED external_commit bytes=2",
        upload(2, 252),
        upload(2, 252),
        "FORT_SCOPED external_commit bytes=2",
        upload(2, 420),
        upload(2, 420),
        "FORT_SCOPED external_commit bytes=2",
    ]


def test_new_halo_slabs_extend_the_same_unchanged_mask_replica():
    ledger = checked_mask_upload_phases(fixture(), 2, (7, 5, 3), repeat_first=True)
    assert ledger["physical_mask_upload_bytes"] == [780, 0, 504, 840]
    assert ledger["equal_axis_payload_copy_count"] == 0


@pytest.mark.parametrize("fault", ["interior", "other_field", "download", "generation", "handle", "late", "overlap"])
def test_redundant_or_retired_replica_cannot_hide_under_full_allocation_size(fault):
    rows = fixture()
    if fault == "interior":
        rows.insert(2, upload(2, 420))  # Total remains below the old 5,148-byte cap.
    elif fault == "other_field":
        rows.insert(2, upload(1, 8))
    elif fault == "download":
        rows.insert(2, "FORT_SCOPED download buffer=1 generation=1 bytes=8")
    elif fault == "generation":
        rows[4] = upload(2, 252, generation=2)
    elif fault == "handle":
        rows[4] = upload(2, 252, handle=99)
    elif fault == "late":
        rows.append(upload(2, 4))
    elif fault == "overlap":
        rows.insert(5, upload(2, 420))
    with pytest.raises(AssertionError):
        checked_mask_upload_phases(rows, 2, (7, 5, 3), repeat_first=True)

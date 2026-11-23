"""Read and compare the complete-array dumps of two application runs."""
import math

import numpy as np


def fields(directory, driver):
    """Read explicit native-precision dumps, optionally with original bounds."""
    result = {}
    dtype = np.dtype(driver['dtype'])
    metadata = driver.get('metadata_format')
    if metadata not in (None, 'storage-shape-bounds-v1'):
        raise ValueError('unsupported complete-field metadata format')
    for path in directory.glob('fields-*.bin'):
        values = np.fromfile(path, dtype=dtype)
        if path.stat().st_size != values.size * dtype.itemsize:
            raise ValueError('partial element in complete field')
        layout = None
        if metadata:
            parts = [int(value) for value in path.with_suffix('.shape').read_text().split()]
            rank, remainder = divmod(len(parts)-1, 3)
            if remainder or not 1 <= rank <= 15 or parts[0] != dtype.itemsize * 8:
                raise ValueError('complete field precision or metadata rank mismatch')
            shape, lower, upper = (parts[1+index*rank:1+(index+1)*rank] for index in range(3))
            if (any(extent < 0 or extent != max(0, hi-lo+1)
                    for extent, lo, hi in zip(shape, lower, upper, strict=True))
                    or math.prod(shape) != values.size):
                raise ValueError('complete field extent or byte count mismatch')
            layout = {'bits': parts[0], 'shape': shape, 'lower_bounds': lower, 'upper_bounds': upper}
        result[path.name] = (values, layout)
    if len(result) != driver['field_count']:
        raise ValueError('missing complete fields')
    return result


def compare_fields(actual, reference, driver):
    if any(not math.isfinite(driver[name]) or driver[name] < 0 for name in ('atol', 'rtol')):
        raise ValueError('validation tolerances must be finite and nonnegative')
    if actual.keys() != reference.keys():
        raise ValueError('missing complete fields')
    comparisons = {}
    for name, (values, layout) in actual.items():
        expected, original_layout = reference[name]
        if values.shape != expected.shape or layout != original_layout:
            raise ValueError('complete field shape or original bounds mismatch')
        finite_actual, finite_reference = np.isfinite(values), np.isfinite(expected)
        # Nonfinite values are always failures, including matching infinities.
        # Keep the failure receipt serializable even when subtraction overflows.
        with np.errstate(over='ignore', invalid='ignore'):
            matches = (finite_actual & finite_reference & np.isclose(
                values, expected, rtol=driver['rtol'], atol=driver['atol'], equal_nan=False))
            errors = np.abs(values - expected)
        failed = np.flatnonzero(~matches)
        finite_errors = np.isfinite(errors)
        comparisons[name] = {'values': int(values.size), 'passed': not bool(failed.size),
            'failed_values': int(failed.size),
            'candidate_nonfinite_values': int(np.count_nonzero(~finite_actual)),
            'reference_nonfinite_values': int(np.count_nonzero(~finite_reference)),
            'absolute_error_nonfinite_values': int(np.count_nonzero(~finite_errors)),
            'max_absolute_error': float(np.max(errors, initial=0)) if np.all(finite_errors) else None,
            'failed_flat_indices': failed[:8].tolist(),
            'failed_locations_truncated': bool(failed.size > 8),
            'atol': driver['atol'], 'rtol': driver['rtol']}
        if layout is not None:
            comparisons[name]['layout'] = layout
    return comparisons

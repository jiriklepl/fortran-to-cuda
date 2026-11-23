import math
from pathlib import Path
import re
import shutil


def data_lines(path):
    return [s for line in Path(path).read_text().splitlines()
            if (s := re.split(r"[!#]", line)[0].strip())]


def parse_grid_spec(value):
    """Parse a 3D grid cell count specification like '128x64x64' or '128,64,64'."""
    if isinstance(value, (tuple, list)):
        parts = list(value)
    elif isinstance(value, str):
        parts = [part.strip() for part in re.split(r"[,xX]", value.strip())]
    else:
        raise ValueError(f"Invalid --grid specification: {value!r}")
    if len(parts) != 3:
        raise ValueError(f"--grid must have 3 dimensions (NXxNYxNZ or NX,NY,NZ), got {value!r}")
    try:
        dims = tuple(int(part) for part in parts)
    except (TypeError, ValueError) as error:
        raise ValueError(f"--grid dimensions must be integers, got {value!r}") from error
    if any(isinstance(part, bool) or d <= 0 for part, d in zip(parts, dims)):
        raise ValueError(f"--grid dimensions must be positive integers, got {value!r}")
    return dims


def parse_domain_size_spec(value):
    """Parse a 3D domain length specification like '12x4x2' or '12,4,2'."""
    if isinstance(value, (tuple, list)):
        parts = list(value)
    elif isinstance(value, str):
        parts = [part.strip() for part in re.split(r"[,xX]", value.strip())]
    else:
        raise ValueError(f"Invalid --domain-size specification: {value!r}")
    if len(parts) != 3:
        raise ValueError(f"--domain-size must have 3 lengths (LXxLYxLZ or LX,LY,LZ), got {value!r}")
    try:
        lengths = tuple(float(part) for part in parts)
    except (TypeError, ValueError) as error:
        raise ValueError(f"--domain-size lengths must be numbers, got {value!r}") from error
    if any(isinstance(part, bool) or not math.isfinite(length) or length <= 0
           for part, length in zip(parts, lengths)):
        raise ValueError(f"--domain-size lengths must be positive finite numbers, got {value!r}")
    return lengths


def _positive_float(value, name):
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive finite number") from error
    if isinstance(value, bool) or not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return parsed


def _replace_conf_data_line(path, data_index, value, comment):
    lines = path.read_text().splitlines(keepends=True)
    data_indices = [i for i, s in enumerate(lines) if re.split(r"[!#]", s)[0].strip()]
    if data_index >= len(data_indices):
        raise ValueError(f"Configuration file {path.name} has too few data lines for index {data_index}")
    line_index = data_indices[data_index]
    lines[line_index] = f"{value} ! {comment}\n"
    path.write_text("".join(lines))


def _replace_time_stepping_option(path, key, value):
    text = path.read_text()
    # Preserve offsets while masking comments, including inline and multiline
    # assignments. Never insert a duplicate of an existing active setting.
    active = re.sub(r"[!#][^\n]*", lambda m: " " * len(m[0]), text)
    assignments = list(re.finditer(rf"(?i)\b{re.escape(key)}\s*=", active))
    if len(assignments) > 1:
        raise ValueError(f"Duplicate active {key} assignments")
    if assignments:
        match = re.match(r"\s*[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?", active[assignments[0].end():])
        if not match:
            raise ValueError(f"Cannot parse existing {key} value")
        start = assignments[0].end()
        text = text[:start] + f" {value}" + text[start + match.end():]
    else:
        block = re.search(r"(?i)time_stepping\s*\(", active)
        if not block:
            raise ValueError(f"Cannot locate time_stepping(...) to apply {key}")
        text = text[:block.end()] + f"\n  {key} = {value}," + text[block.end():]
    path.write_text(text)


def prepare_config(source, destination, *, steps=None, advection=None, grid=None,
                   domain_size=None, end_time=None, cfl=None, re_number=None, profile=None):
    source, destination = Path(source).resolve(), Path(destination)
    if source.is_file() and (profile is None or profile.config_mode == "elmm-conf"):
        source = source.parent
    config_root = source.parent if source.is_file() else source
    if destination.resolve() == config_root or config_root in destination.resolve().parents:
        raise ValueError("Configuration snapshot must be outside the input configuration directory")
    if destination.exists():
        raise ValueError(f"Configuration snapshot already exists: {destination}")

    if profile is not None and profile.config_mode == "generic":
        if any(v is not None for v in (advection, grid, domain_size, end_time, cfl, re_number)):
            raise ValueError("ELMM configuration overrides are not supported for generic config_mode")
        if source.is_dir():
            shutil.copytree(source, destination, ignore=shutil.ignore_patterns(".git", "__pycache__"))
        elif source.is_file():
            destination.mkdir(parents=True, exist_ok=False)
            shutil.copy2(source, destination / source.name)
        else:
            raise ValueError(f"Configuration path does not exist: {source}")
        overrides = {}
        if steps is not None:
            if steps <= 0:
                raise ValueError("--steps must be positive")
            overrides["steps"] = steps
        return {
            "source": str(source),
            "overrides": overrides,
            "expected_max_steps": steps,
            "runtime_arguments": list(profile.runtime_arguments),
        }

    if not (source / "main.conf").is_file():
        raise ValueError(f"Configuration needs main.conf: {source}")
    # Copy every asset, including input/ and relative profiles. Dereference symlinks
    # so the snapshot can be moved independently of the source configuration.
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    original_advection = int(data_lines(destination / "main.conf")[0])
    overrides = {}
    if advection is not None:
        _replace_conf_data_line(destination / "main.conf", 0, advection, "explicit pipeline --advection override")
        overrides["advection_method"] = {"original": original_advection, "effective": advection}
    if re_number is not None:
        re_val = _positive_float(re_number, "--re")
        _replace_conf_data_line(destination / "main.conf", 9, f"{re_val:g}", "explicit pipeline --re override")
        overrides["re"] = re_val
    if domain_size is not None or grid is not None:
        grid_conf = destination / "grid.conf"
        if not grid_conf.is_file():
            raise ValueError(f"Configuration needs grid.conf for --grid/--domain-size overrides: {source}")
        if domain_size is not None:
            lx, ly, lz = parse_domain_size_spec(domain_size)
            for idx, val, axis in ((6, lx, "lx"), (7, ly, "ly"), (8, lz, "lz")):
                _replace_conf_data_line(grid_conf, idx, f"{val:g}", f"explicit pipeline --domain-size {axis} override")
            overrides["domain_size"] = {"lx": lx, "ly": ly, "lz": lz}
        if grid is not None:
            nx, ny, nz = parse_grid_spec(grid)
            for idx, val, axis in ((9, nx, "nx"), (10, ny, "ny"), (11, nz, "nz")):
                _replace_conf_data_line(grid_conf, idx, val, f"explicit pipeline --grid {axis} override")
            overrides["grid"] = {"nx": nx, "ny": ny, "nz": nz}
    if steps is not None:
        if steps <= 0:
            raise ValueError("--steps must be positive")
        _replace_time_stepping_option(destination / "time_stepping.conf", "max_number_of_time_steps", steps)
        overrides["max_number_of_time_steps"] = steps
    if end_time is not None:
        end_val = _positive_float(end_time, "--end-time")
        _replace_time_stepping_option(destination / "time_stepping.conf", "end_time", f"{end_val:g}")
        overrides["end_time"] = end_val
    if cfl is not None:
        cfl_val = _positive_float(cfl, "--cfl")
        _replace_time_stepping_option(destination / "time_stepping.conf", "CFL", f"{cfl_val:g}")
        overrides["cfl"] = cfl_val
    method = int(data_lines(destination / "main.conf")[0])
    cmd_file = destination / "cmd.conf"
    cmd_text = cmd_file.read_text().splitlines()[0] if cmd_file.exists() and cmd_file.read_text().splitlines() else ""
    discretization = re.search(r"(?i)\bdiscretization_order\s*=\s*(\d+)", cmd_text.split("!")[0])
    if discretization and int(discretization[1]) == 4:
        method = 4  # ELMM applies this override after reading main.conf.
    text = (destination / "time_stepping.conf").read_text()
    text = "\n".join(re.split(r"[!#]", s)[0] for s in text.splitlines())
    maximum = re.search(r"(?i)max_number_of_time_steps\s*=\s*(\d+)", text)
    return {"source": str(source), "overrides": overrides, "advection_method": method,
            "expected_max_steps": int(maximum[1]) if maximum else None,
            "runtime_arguments": ["output_dir='output/'"]}

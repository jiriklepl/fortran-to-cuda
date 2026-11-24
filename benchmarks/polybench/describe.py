"""Write the description of every benchmark of PolyBench/Fortran 1.0, from the suite's archive.

    python benchmarks/polybench/describe.py polybench-fortran-1.0.tar.gz DIRECTORY

Each benchmark is one main program whose kernel is a procedure inside it. A
description selects the kernel, keeps its arrays on the device for the call,
and prints a checksum of what the program would print (see benchmark.toml.in).
The names it needs are read from the program: the kernel, the arrays the
program prints, and the array it deallocates first.
"""
import hashlib
import re
import sys
import tarfile
from pathlib import Path, PurePosixPath

URL = "https://www.cs.colostate.edu/~pouchet/software/polybench/download/polybench-fortran-1.0.tar.gz"
SHA256 = "792320c28132c06ac340fa97f1444b39e4182e9fe9c06e8e870857f56de575d4"
TEMPLATE = Path(__file__).with_name("benchmark.toml.in")


def benchmarks(archive):
    """Each benchmark's program, by its name, with the text of its header."""
    found = {}
    with tarfile.open(archive) as members:
        files = {PurePosixPath(*PurePosixPath(member.name).parts[1:]): member
                 for member in members.getmembers() if member.isfile()}
        for path, member in sorted(files.items()):
            if path.suffix == ".F90":
                header = files[path.with_suffix(".h")]
                found[path.stem] = (path, members.extractfile(member).read().decode(),
                                    members.extractfile(header).read().decode())
    return found


def description(path, text, header):
    """The description of the benchmark whose program is ``path``, with ``text``, and whose header is ``header``."""
    def one(pattern):
        matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
        if len(matches) != 1:
            raise ValueError(f"{path}: expected one match of {pattern}, found {len(matches)}")
        return matches[0]

    program = one(r"^[ \t]*program\s+(\w+)")
    kernel = one(r"^[ \t]*subroutine\s+(kernel_\w+)")
    printed, arguments = one(r"^([ \t]*polybench_prevent_dce\(print_array\((.*)\)\);?[ \t]*)$")
    # The arguments that are not sizes, which the header defines, are what the program prints.
    sizes = set(re.findall(r"#\s*define\s+(\w+)", header))
    names = [name.strip() for name in arguments.split(",") if name.strip() not in sizes]
    first = re.search(r"POLYBENCH_DEALLOC_ARRAY\((\w+)\)", text).group(1)
    checksums = "".join(f"      write(*, '(a, es24.16)') 'checksum {name.lower()}: ', sum([real({name}, kind=8)])\n"
                        for name in names)
    return TEMPLATE.read_text().format(
        name=path.stem, path=path.as_posix(), directory=path.parent.as_posix(), program=program.lower(),
        kernel=kernel.lower(), printed=printed.rstrip("\n") + "\n", checksums=checksums, first=first,
        url=URL, sha256=SHA256)


def describe(archive, directory):
    """Write one description per benchmark into ``directory``; return their paths, by benchmark."""
    archive = Path(archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256:
        raise ValueError(f"Not the PolyBench/Fortran 1.0 archive (sha256 {SHA256}): {archive}")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, (path, text, header) in benchmarks(archive).items():
        written[name] = directory / (name + ".toml")
        written[name].write_text(description(path, text, header))
    return written


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__.split("\n\n")[1])
    for name, path in describe(*sys.argv[1:]).items():
        print(name, path)

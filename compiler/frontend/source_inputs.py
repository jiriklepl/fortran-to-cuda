"""The source files of an analysis: their text, read once, and its hash."""

from __future__ import annotations

from hashlib import sha256
from io import StringIO
from pathlib import Path

from fparser.common.readfortran import FortranFileReader
from fparser.two.utils import FortranSyntaxError

from compiler.frontend.syntax import parser
from compiler.ir import CompilationError, SourceLocation


class SourceInputs:
    """Each file is read when the analysis starts; ``sources`` maps its path to the hash of what was read."""

    def __init__(self, paths):
        try:
            self.paths=list(dict.fromkeys(Path(path).resolve(strict=True) for path in paths))
            raw={str(path):path.read_bytes() for path in self.paths}
            self.sources={path:sha256(data).hexdigest() for path,data in raw.items()}
            self._texts={path:data.decode().replace("\r\n", "\n").replace("\r", "\n") for path,data in raw.items()}
        except (OSError, UnicodeError) as error:
            raise CompilationError("original analysis source is unavailable") from error

    def parse(self, path):
        # The filename still determines source form/include lookup and error
        # locations, but the parser consumes the exact bytes already hashed.
        source = StringIO(self._texts[str(path)])
        source.name = str(path)
        try:
            reader = FortranFileReader(source, ignore_comments=False)
            return parser()(reader)
        except FortranSyntaxError as error:
            raise CompilationError(f"invalid Fortran syntax: {error}", SourceLocation(str(path))) from error

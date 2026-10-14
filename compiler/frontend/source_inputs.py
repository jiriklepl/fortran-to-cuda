"""Hash-bound configured source text and provenance for effect/scope analysis."""

from __future__ import annotations

import json
from copy import deepcopy
from hashlib import sha256
from io import StringIO
from pathlib import Path

from fparser.common.readfortran import FortranFileReader
from fparser.two.parser import ParserFactory
from fparser.two.utils import FortranSyntaxError, walk

from compiler.ir import CompilationError, SourceLocation


class SourceInputs:
    """Keep original edit targets separate from configured analysis inputs.

    The extractor asserts preprocessing equivalence for one build configuration.
    Hashes bind that assertion to all sources/includes and prepared text. An
    unmappable statement remains an edit boundary rather than using its prepared
    line number to rewrite the original file.
    """

    def __init__(self, paths, document=None):
        try:
            self.paths=list(dict.fromkeys(Path(path).resolve(strict=True) for path in paths))
            raw={str(path):path.read_bytes() for path in self.paths}
            self.sources={path:sha256(data).hexdigest() for path,data in raw.items()}
            self._texts={path:data.decode().replace("\r\n", "\n").replace("\r", "\n") for path,data in raw.items()}
        except (OSError, UnicodeError) as error:
            raise CompilationError("original analysis source is unavailable") from error
        self._external_readers=set()
        self.entries={}
        self.dependencies={}
        self.configuration=None
        if document is None:
            return
        try:
            json.dumps(document, allow_nan=False)
        except (TypeError, ValueError, RecursionError) as error:
            raise CompilationError("configured analysis sources require finite JSON facts") from error
        # Configuration and provenance belong to this analysis snapshot. A
        # caller changing its dictionaries must not change a cache authority.
        document = deepcopy(document)
        if (not isinstance(document,dict) or document.get("schema_version")!=1
                or document.get("source_inputs")!=self.sources or document.get("preserves_source_order") is not True
                or not isinstance(document.get("entries"),list)
                or not isinstance(document.get("configuration"),dict)
                or not isinstance(document.get("dependencies"),dict)):
            raise CompilationError("configured analysis sources require matching versioned source/configuration facts")
        self.configuration=document["configuration"]
        for name,digest in document["dependencies"].items():
            if not isinstance(name,str) or not Path(name).is_absolute():
                raise CompilationError("configured source dependencies require absolute paths")
            try:
                path=Path(name).resolve(strict=True)
                if sha256(path.read_bytes()).hexdigest()!=digest:
                    raise CompilationError("configured source dependency hash differs: "+name)
            except OSError as error:
                raise CompilationError("configured source dependency is unavailable: "+name) from error
            self.dependencies[str(path)]=digest
        for item in document["entries"]:
            if not isinstance(item,dict):
                raise CompilationError("configured analysis source entries must be objects")
            source=item.get("source")
            if not isinstance(source,str) or source not in self.sources or source in self.entries:
                raise CompilationError("configured analysis source is unavailable or duplicated")
            try:
                if not Path(item["path"]).is_absolute():
                    raise ValueError("absolute prepared source required")
                path=Path(item["path"]).resolve(strict=True)
                data=path.read_bytes()
                text=data.decode().replace("\r\n", "\n").replace("\r", "\n")
            except (KeyError,TypeError,OSError,ValueError) as error:
                raise CompilationError("configured analysis source path is unavailable") from error
            if sha256(data).hexdigest()!=item.get("sha256"):
                raise CompilationError("configured analysis source hash differs")
            mapping=item.get("line_map")
            self._texts[str(path)]=text
            count=len(self._texts[source].splitlines())
            if (not isinstance(mapping,list) or len(mapping)!=len(text.splitlines())
                    or any(value is not None and (type(value) is not int or not 1<=value<=count) for value in mapping)):
                raise CompilationError("configured analysis source has an invalid original line map")
            positions=[value for value in mapping if value is not None]
            if positions!=sorted(positions):
                raise CompilationError("configured analysis source line map changes original source order")
            self.entries[source]={"source":source,"path":str(path),"sha256":item["sha256"],"line_map":mapping}
        if set(self.entries)!=set(self.sources):
            raise CompilationError("configured analysis package must cover every supplied original source")

    def path(self, original):
        item=self.entries.get(str(original))
        return Path(item["path"]) if item else original

    def parse(self, original):
        path = self.path(original)
        # The filename still determines source form/include lookup and error
        # locations, but the parser consumes the exact bytes already hashed.
        source = StringIO(self._texts[str(path)])
        source.name = str(path)
        try:
            reader = FortranFileReader(source, ignore_comments=False)
            tree = ParserFactory().create(std="f2008")(reader)
        except FortranSyntaxError as error:
            raise CompilationError(f"invalid Fortran syntax: {error}", SourceLocation(str(path))) from error
        # fparser expands raw Fortran INCLUDE through an independent live
        # reader. Standalone analysis keeps that behavior, but such bytes are
        # not the captured root text and cannot authorize an immutable session.
        readers = [getattr(getattr(node, "item", None), "reader", None) for node in walk(tree)]
        readers.append(getattr(reader, "reader", None))
        self._external_readers.update(str(getattr(item, "id", "external include"))
                                      for item in readers if item is not None and item is not reader)
        record=self.entries.get(str(original))
        if record:
            for node in walk(tree):
                item=getattr(node,"item",None)
                if item is None:
                    continue
                first,last=item.span
                mapping=record["line_map"][first-1:last]
                item.fort_original_span=((mapping[0],mapping[-1])
                                         if mapping and all(value is not None for value in mapping) else None)
        return tree

    def require_captured_parsing(self):
        if self._external_readers:
            raise CompilationError("immutable source session requires captured prepared text for INCLUDE: "
                                   + ", ".join(sorted(self._external_readers)))

    def source_text(self, path, *, captured=False):
        path = str(Path(path).resolve())
        if captured:
            if path not in self._texts:
                raise CompilationError("source text is outside the compilation session: " + path)
            return self._texts[path]
        return Path(path).read_text()

    def verify(self):
        """Standalone input validation is always fresh, even for frozen copies."""
        self.verify_live()

    def verify_live(self):
        try:
            for path,digest in {**self.sources,**self.dependencies}.items():
                if sha256(Path(path).read_bytes()).hexdigest()!=digest:
                    raise CompilationError("configured source changed while artifacts were being prepared: "+path)
            for item in self.entries.values():
                if sha256(Path(item["path"]).read_bytes()).hexdigest()!=item["sha256"]:
                    raise CompilationError("configured analysis source changed while artifacts were being prepared")
        except OSError as error:
            raise CompilationError("configured source became unavailable while artifacts were being prepared") from error

    def public(self):
        if not self.entries:
            return None
        return deepcopy({"schema_version":1,"source_inputs":self.sources,"configuration":self.configuration,
                "preserves_source_order":True,"dependencies":self.dependencies,
                "entries":[self.entries[name] for name in sorted(self.entries)]})

    def identity(self):
        """Immutable JSON provenance for reusable source proofs, never ASTs."""
        return deepcopy({"schema_version": 1, "sources": self.sources,
                         "dependencies": self.dependencies,
                         "configuration": self.configuration,
                         "entries": [self.entries[name] for name in sorted(self.entries)]})

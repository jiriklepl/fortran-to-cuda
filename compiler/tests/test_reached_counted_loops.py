"""Original serial loops retain one owner and their reached escape semantics."""

from hashlib import sha256

from compiler.driver.options import CompilerOptions
from compiler.frontend.source_effects import _children, _kind
from compiler.offload.config import OffloadConfig
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_lexical_source_owner import SOURCE
from compiler.tests.test_source_scopes import FACT

COUNTED_SOURCE = (SOURCE.replace("integer::i", "integer::i,stage")
    .replace("visits=visits+1", "visits=visits+1\ndo stage=1,3")
    .replace("if(escape) then", "if(escape.and.stage==2) then")
    .replace("end subroutine\nend module", "enddo\nend subroutine\nend module"))


def builder_for(directory, source=COUNTED_SOURCE):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "counted.f90"
    path.write_text(source)
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
             "captures": {"argument::"+name: FACT for name in ("a", "b", "out")}}
    return ScopeBuilder([path], "local_owner::step", facts=facts,
                        options=CompilerOptions(),
                        config=OffloadConfig(policy="sections", scope_execution="reached"))


def original_loop(builder):
    return next(node for node in _children(builder.entry.execution)
                if _kind(node) == "Block_Nonlabel_Do_Construct")

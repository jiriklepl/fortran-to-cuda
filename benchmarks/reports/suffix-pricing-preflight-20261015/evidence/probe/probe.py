"""Exact extended probe stdin, preserved after its one bounded execution.

Invocation: PYTHONDONTWRITEBYTECODE=1 timeout 60 taskset -c 4,5,10,11
            .venv/bin/python -u - < probe.py
The original execution used a shell heredoc, not this subsequently saved file.
"""
import json,time
from pathlib import Path
from compiler.driver.options import CompilerOptions
from compiler.offload.config import OffloadConfig
from compiler.scopes.source import ScopeBuilder
from compiler.frontend.native_group_structure import NativeGroupCandidate,grouped_original_sequence
from compiler.frontend.native_definitions import prove_native_definitions
from compiler.frontend.numerical_preparation import NumericalPreparationProvider
from compiler.frontend.reached_windows import plan_reached_window
start=time.monotonic()
def out(phase,**values): print(json.dumps(dict(phase=phase,seconds=round(time.monotonic()-start,3),**values)),flush=True)
r=json.loads(Path('benchmarks/results/checked-plan-application-20261014/request.json').read_text())
f=json.loads(Path(r['capture_facts']).read_text())
b=ScopeBuilder([r['source'],*r['source_files']],r['entry'],facts=f,options=CompilerOptions(gpu_policy='auto',memory_model='scoped'),config=OffloadConfig(policy='auto',host_threads=4,scope_execution='reached'),analysis_sources=json.loads(Path(r['analysis_sources']).read_text()))
out('source_session',session=bool(b.analysis._source_session))
groups=[n for n in grouped_original_sequence(b.entry.execution.content,{}) if isinstance(n,NativeGroupCandidate)]
matching=[g for g in groups if 'beta(rk_stage)' in ''.join(str(n) for n in g.original_nodes).replace(' ','').lower()]
out('accumulation_selection',matched=len(matching),all_joined_groups=len(groups))
if len(matching)==1:
 selection=matching[0].original_nodes
 try:
  completion=b.analysis.joined_completion(r['entry'],selection)
  out('accumulation_completion',identity=completion.identity,public=completion.public())
  proof=prove_native_definitions(b.analysis,r['entry'],selection,completion=completion)
  out('accumulation_definition',identity=proof.identity,public=proof.public())
 except Exception as exc: out('accumulation_failure',type=type(exc).__name__,reason=str(exc))
selection=tuple(n for n in b.entry.execution.content if type(n).__name__=='If_Construct' and any(type(c).__name__=='If_Then_Stmt' and 'explicit_diffusion' in str(c).lower() for c in n.content))
out('diffusion_selection',matched=len(selection))
if len(selection)==1:
 roots=[]
 for root in sorted(f['captures']):
  try:
   binding=b.visible_binding(b.entry,root)
   if binding.rank:
    b.capture_fact(binding)
    roots.append(root)
  except Exception as exc: out('capture_decline',root=root,reason=str(exc))
 provider=NumericalPreparationProvider(b.analysis,options=b.options,config=b.config,environment_root=b._numerical_environment_root,immutable_inputs=getattr(b,'immutable_inputs',{}))
 out('diffusion_proof_start',roots=roots)
 try:
  proof=plan_reached_window(b.analysis,r['entry'],selection,tuple(roots),f,numerical_provider=provider)
  out('diffusion_proof',identity=proof.identity,public=proof.public())
 except Exception as exc: out('diffusion_failure',type=type(exc).__name__,reason=str(exc))
try:
 b.analysis.verify_publication()
 out('publication_authentication',valid=True)
except Exception as exc: out('publication_authentication',valid=False,type=type(exc).__name__,reason=str(exc))

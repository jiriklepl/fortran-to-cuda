import json
from pathlib import Path
from tempfile import TemporaryDirectory
from compiler.frontend.source_effects import SourceEffects
from compiler.frontend.call_bindings import resolve_source_call, _actual
from compiler.frontend.resource_reachability import prove_resource_reachability
from compiler.ir import CompilationError

TEMPLATE='''module shape_probe
implicit none
interface add
module procedure add2,add3
end interface
contains
subroutine owner(old,a,n,dt,flux,dzPr,spacing)
real(8),intent(in)::old(:,:,:),dt,flux,dzPr(:),spacing
real(8),intent(inout)::a(:,:,:)
integer,intent(in)::n
real(8)::value
BODY
end subroutine
subroutine add2(to,from)
real(8),intent(inout)::to(:,:)
real(8),intent(in)::from
to=to+from
end subroutine
subroutine add3(to,from)
real(8),intent(inout)::to(:,:,:)
real(8),intent(in)::from
to=to+from
end subroutine
end module
'''
variants={
 'rank_reduced_literal':'call add(a(:,:,n), 1.0_8)',
 'rank_reduced_scalar_arithmetic':'call add(a(:,:,n), -dt*flux/spacing)',
 'rank_reduced_indexed_arithmetic':'call add(a(:,:,n), -dt*flux/dzPr(n))',
 'direct_candidate_indexed_arithmetic':'call add2(a(:,:,n), -dt*flux/dzPr(n))',
 'original_indexed_read_then_scalar_actual':'value=dzPr(n)\ncall add(a(:,:,n), -dt*flux/value)',
}
results=[]
with TemporaryDirectory(prefix='resource-add-shape-') as directory:
 for label,body in variants.items():
  path=Path(directory)/(label+'.f90');path.write_text(TEMPLATE.replace('BODY',body))
  analysis=SourceEffects([path]);routine=analysis.routines['shape_probe::owner']
  selection=tuple(routine.execution.content)
  call=next(node for node in selection if type(node).__name__=='Call_Stmt')
  row={'variant':label,'source':body}
  try:
   resolved=resolve_source_call(analysis,routine.scope,call)
   row['resolved']=resolved.procedure
   row['first_actual_source_rank']=resolved.mappings[0].binding.rank
   row['first_actual_logical_rank']=resolved.mappings[0].section.logical_rank
  except CompilationError as error: row['resolver_error']=str(error)
  try: row['second_actual_signature']=_actual(analysis,routine.scope,call.items[1].items[1])[0]
  except CompilationError as error:row['second_actual_error']=str(error)
  proof=prove_resource_reachability(analysis,routine.qualified,selection,(routine.scope.bindings['old'],))
  proof.validate(analysis,routine.qualified,selection,(routine.scope.bindings['old'],))
  row['selection_covered']=proof.public()['selection_covered']
  row['barriers']=proof.public()['barriers']
  row['indexed_array_reads']=[a.source for a in proof.accesses if a.storage.resource=='argument::dzpr' and a.kind=='read']
  results.append(row)
Path('/tmp/resource-add-repro.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))

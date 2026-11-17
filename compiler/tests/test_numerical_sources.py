"""Normalized source contracts preserve original ownership and coordinate events."""

from __future__ import annotations

import copy
from hashlib import sha256

import pytest

from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError
from compiler.scopes.numerical import load_numerical_sources

ORIGINAL = """module settings
implicit none
real(8) :: gain=2
real(8) :: weights(32)
end module
module relay
use settings
end module
module original
use relay
implicit none
contains
subroutine producer(a,b,n)
real(8),intent(in)::a(-2:)
real(8),intent(inout)::b(-2:)
integer,intent(in)::n
integer::i
!$omp parallel do private(i)
do i=-1,n-4
b(i)=gain*a(i)+weights(i+3)+real(i,8)
enddo
!$omp end parallel do
end subroutine
subroutine transform(b)
real(8),intent(inout)::b(:)
b=3*b
end subroutine
subroutine consumer(b,out,n)
real(8),intent(in)::b(:)
real(8),intent(inout)::out(:)
integer,intent(in)::n
integer::i
do i=2,n-1
out(i)=b(i)
enddo
end subroutine
subroutine step(a,b,out,n)
use relay
real(8),intent(in)::a(:)
real(8),intent(inout)::b(:),out(:)
integer,intent(in)::n
call producer(a,b,n)
call transform(b)
call consumer(b,out,n)
end subroutine
end module
"""

NORMALIZED = """module extracted
implicit none
contains
subroutine producer(a,b,n,gain,weights,alb,blb,wlb)
real(8),intent(in)::a(:),weights(:),gain
real(8),intent(inout)::b(:)
integer,intent(in)::n,alb,blb,wlb
integer::i
do i=-1,n-4
b(i-blb+1)=gain*a(i-alb+1)+weights(i+3-wlb+1)+real(i,8)
enddo
end subroutine
end module
"""

DRIVER = """program caller
use original
implicit none
real(8),allocatable::a(:),b(:),output(:)
integer::n,i,shape,repeat
weights=[(real(i,8)*0.1d0,i=1,32)]
do shape=1,2
 n=8+shape*8
 allocate(a(-2:n-3),b(-2:n-3),output(-2:n-3))
 do repeat=1,2
  gain=2+repeat
  a=[(real(i,8)*0.25d0,i=-2,n-3)]
  b=-99
  output=-99
  call step(a,b,output,n)
  if (b(-2)/=-297 .or. b(n-3)/=-297) error stop 'intermediate halo'
  if (output(-2)/=-99 .or. output(n-3)/=-99) error stop 'output halo'
  do i=-1,n-4
   if (b(i)/=3*(gain*a(i)+weights(i+3)+real(i,8))) error stop 'source coordinates'
   if (output(i)/=b(i)) error stop 'consumer disagreement'
  enddo
 enddo
 deallocate(a,b,output)
enddo
print *, 'FIELDS_OK'
end program
"""


def package(directory, *, original=ORIGINAL, normalized=NORMALIZED):
    directory.mkdir(parents=True,exist_ok=True)
    source=directory/"original.f90"
    source.write_text(original)
    numerical=directory/"normalized.f90"
    numerical.write_text(normalized)
    hashes={str(source):sha256(source.read_bytes()).hexdigest()}
    parameters=[{"name":name,"resource":resource,**({"physical_origin":[0]} if rank else {})}
                for name,resource,rank in [("a","argument::a",1),("b","argument::b",1),
                                           ("n","argument::n",0),("gain","settings::gain",0),
                                           ("weights","settings::weights",1)]]
    parameters += [{"name":name,"resource":root,"lower_bound_dimension":1}
                   for name,root in [("alb","argument::a"),("blb","argument::b"),("wlb","settings::weights")]]
    document={"schema_version":1,"source_inputs":hashes,"entries":[{
        "procedure":"original::producer","source_sha256":hashes[str(source)],
        "path":str(numerical),"sha256":sha256(numerical.read_bytes()).hexdigest(),
        "entry":"extracted::producer","normalization":"whole_storage_rebased_v1",
        "participation":"serial_coordinator","capture_safe":True,"preserves_source_order":True,
        "parameters":parameters}]}
    return source,numerical,document


@pytest.mark.parametrize(("field","value","reason"),[
    ("sha256","stale","hash differs"), ("normalization","compact","coordinate facts"),
    ("capture_safe",False,"capture/coordinate facts"),
    ("participation","collective","capture/coordinate facts"),
    ("source_sha256","stale","source/capture"),
])
def test_package_rejects_stale_or_unsupported_contracts(tmp_path,field,value,reason):
    source,_normalized,document=package(tmp_path)
    document["entries"][0][field]=value
    with pytest.raises(CompilationError,match=reason):
        load_numerical_sources(document,SourceEffects([source]))


def test_package_rejects_wrong_array_mapping_and_missing_write_intent(tmp_path):
    source,_normalized,document=package(tmp_path)
    altered=copy.deepcopy(document)
    altered["entries"][0]["parameters"][1]["physical_origin"]=[1]
    with pytest.raises(CompilationError,match="zero-origin"):
        load_numerical_sources(altered,SourceEffects([source]))
    source,_normalized,document=package(tmp_path,normalized=NORMALIZED.replace("intent(inout)::b", "intent(in)::b"))
    with pytest.raises(CompilationError,match="written resource"):
        load_numerical_sources(document,SourceEffects([source]))


def test_effects_keep_do_header_after_leading_openmp_comment(tmp_path):
    source,_normalized,_document=package(tmp_path,original=ORIGINAL.replace("integer::i\n!$omp parallel", "integer::i\nb=0\n!$omp parallel"))
    summary=SourceEffects([source]).summarize("original::producer")
    assert summary["complete"],summary["reasons"]
    assert not summary["cloneable"]
    assert any(operation.get("resource")=="settings::weights" and operation["guard"]
               for operation in summary["operations"])

"""Generic host preparation, guarded arguments, and shared GPU intervals."""

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from compiler.driver.options import CompilerOptions
from compiler.driver.pipeline import prepare_function
from compiler.emission import read_common_header
from compiler.frontend import lower_file
from compiler.offload.config import OffloadConfig
from compiler.offload.profile import load_profile

SOURCE = """module prepared_case
contains
subroutine advance(a,b,configuration,n,scale,unused)
real(8),intent(inout)::a(:)
real(8),intent(in)::b(:)
integer,intent(in)::configuration(:),n,unused
real(8),intent(in)::scale
integer::i,limit
real(8)::coefficient
limit=configuration(1)
coefficient=scale/2.0_8
if(limit>0) then
 do i=1,n
  a(i+1)=b(i+1)*coefficient+real(i,8)
 end do
end if
limit=configuration(2)
coefficient=scale/4.0_8
if(limit>0) then
 do i=1,n
  a(i+1)=a(i+1)+b(i+1)*coefficient
 end do
end if
end subroutine
end module
"""


def run(argv, directory, environment=None):
    result = subprocess.run(argv, cwd=directory, env=environment, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout + result.stderr

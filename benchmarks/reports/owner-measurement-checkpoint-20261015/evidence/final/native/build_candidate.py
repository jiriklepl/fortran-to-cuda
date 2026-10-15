"""Compose compiler-issued packages and build immutable diagnostic candidates."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path('/home/jirka/research/fortran-abomination')
HERE = Path(__file__).resolve().parent
SEED = ROOT / 'benchmarks/results/checked-plan-application-20261014/seed'
NATIVE = Path('/tmp/fortran-residency-e2e-20261014/repaired-native-checkpoint-gfortran14')
MODE_DIRS = {'owner-measurement':'candidate'}

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def write(path,value):
    Path(path).write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')

def inventory(paths):
    return {str(path):digest(path) for path in paths}

def original_key(source,root):
    return str(Path(source).relative_to(root))

def normalized_argv(unit, root, build):
    return [arg.replace(str(build),'<BUILD>').replace(str(root),'<SOURCE>')
            for arg in unit['execution']['argv'][1:]]

def worker(mode):
    sys.dont_write_bytecode=True
    sys.path.insert(0,str(HERE/'pipeline'))
    from elmm_pipeline.local_adapter import compose_scope_packages
    from elmm_pipeline.build import build, preprocessing_defines
    from elmm_pipeline.build_receipts import require_build_receipts
    out=HERE/MODE_DIRS[mode]
    receipt_path=out/'build-generation.json'
    if receipt_path.exists() or (out/'application').exists() or (out/'build').exists():
        raise ValueError('build attempt is immutable')
    out.mkdir(exist_ok=True)
    parent=read(HERE/'request.json')
    assert read(HERE/'completion.json')['source_unchanged']
    packages=[HERE/'artifacts/scope-transformation.json']
    metadata=[read(p) for p in packages]
    seed_hashes=inventory(p for p in SEED.rglob('*') if p.is_file())
    input_paths=[*map(Path,seed_hashes),*packages,HERE/'request.json',
                 NATIVE/'build/build-receipts.json',NATIVE/'build/build-command.json']
    for package in metadata:
        root=Path(package['artifact_root'])
        input_paths.extend(root/name for name in package['scope_manifest']['artifacts_sha256'])
        input_paths.append(root/'scope-manifest.json')
    for name in ('compiler','pipeline'):
        assert digest(HERE/(name+'.tar'))==parent['archives'][name]['sha256']
        input_paths.extend(p for p in (HERE/name).rglob('*') if p.is_file())
    before=inventory(input_paths)
    write(out/'build-source-before.json',before)
    native=require_build_receipts(NATIVE/'build/build-receipts.json')
    assert len(native['units'])==72
    native_units={original_key(u['source'],NATIVE):u for u in native['units']}
    assert len(native_units)==72
    for key,unit in native_units.items():
        assert digest(SEED/key)==digest(unit['source']),key
    # Match the preserved native toolchain and numerical flags. Generated CUDA
    # retains its compiler-issued strict options through the public build API.
    for key in list(os.environ):
        if key.startswith(('OMP_','GOMP_','CUDA_','FORT_','ELMM_','NVCOMPILER_','PGI_')):
            os.environ.pop(key)
    for key in ('LD_PRELOAD','LD_AUDIT','CPATH','C_INCLUDE_PATH','CPLUS_INCLUDE_PATH',
                'GCC_EXEC_PREFIX','COMPILER_PATH','NVCC_PREPEND_FLAGS','NVCC_APPEND_FLAGS',
                'FFLAGS','FCFLAGS','F90FLAGS','CFLAGS','CXXFLAGS','LDFLAGS'):
        os.environ.pop(key,None)
    environment=native['units'][0]['execution']['environment']
    for key in ('LIBRARY_PATH','LD_LIBRARY_PATH'):
        # build() prepends its FFTW path, so retain only the recorded suffix.
        value=environment[key].split(os.pathsep)
        assert value[0]==str(ROOT/'elmm-pipeline/.cache/native/lib')
        os.environ[key]=os.pathsep.join(value[1:])
    os.environ.update(FC='/usr/bin/gfortran-14',CC='/usr/bin/gcc-16',CXX='/usr/bin/g++-14',
        PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',OMP_DYNAMIC='FALSE',
        OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    assert preprocessing_defines([])==['DPREC']
    receipt={'schema_version':1,'status':'applying_packages','mode':mode,'commits':parent['commits'],
        'selection':'compiler-issued native call observation; no automatic placement or qualification claim',
        'packages':[{'path':str(p),'sha256':digest(p)} for p in packages],
        'native_control':{'binary':native['binary'],'binary_sha256':digest(native['binary']),
                          'receipt_identity':native['identity'],'original_compilation_units':72},
        'configurations':parent['configurations'],'source_seed':str(SEED),
        'source_seed_sha256':seed_hashes,'cpu_affinity':sorted(os.sched_getaffinity(0)),
        'jobs':4,'runtime_threads':4,'application_executed':False,'performance_measured':False,
        'runner_sha256':digest(__file__)}
    started=time.monotonic()
    write(receipt_path,receipt)
    try:
        shutil.copytree(SEED,out/'application')
        destinations={str(p):out/'application'/p.relative_to(SEED) for p in SEED.rglob('*') if p.is_file()}
        composition=compose_scope_packages(metadata,destinations)
        write(out/'composition.json',composition)
        changed={original_key(p,out/'application') for p in composition['patched_sources']}
        receipt.update(status='building',runtime=composition['runtime'],
            source_replacements=sorted(changed),cuda_units=len(composition['gpu_sources']),
            generated_fortran_units=len(composition['extra_sources']))
        write(receipt_path,receipt)
        binary=build(out/'application/ELMM',out/'application/PoisFFT',out/'build',
            ROOT/'elmm-pipeline/.cache/envs/build/bin/python',ROOT/'elmm-pipeline/.cache/native/lib',
            jobs=4,threads=4,extra_args=[],extra_sources=[Path(p) for p in composition['extra_sources']],
            patched_sources=[Path(p) for p in composition['patched_sources']],
            gpu_sources=composition['gpu_sources'],gpu_options={'architecture':'86',
                'nvcc':'/usr/local/cuda-13.4/bin/nvcc','cuda_host_cxx':'/usr/bin/g++-14',
                'cuda_library_dir':'/usr/local/cuda-13.4/targets/x86_64-linux/lib'})
        actual=require_build_receipts(out/'build/build-receipts.json')
        actual_units={original_key(u['source'],out/'application'):u for u in actual['units']
                      if Path(u['source']).is_relative_to(out/'application')}
        assert set(actual_units)==set(native_units)
        comparisons=[]
        for key,old in native_units.items():
            new=actual_units[key]
            compiler_equal=old['execution']['compiler']==new['execution']['compiler']
            old_argv=normalized_argv(old,NATIVE,NATIVE/'build')
            new_argv=normalized_argv(new,out/'application',out/'build')
            row={'source':key,'compiler_equal':compiler_equal,'normalized_arguments_equal':old_argv==new_argv,
                 'native_argv':old_argv,'candidate_argv':new_argv,'replacement_authorized':key in changed,
                 'source_equal':digest(old['source'])==digest(new['source'])}
            comparisons.append(row)
        write(out/'native-command-comparison.json',comparisons)
        assert all(r['compiler_equal'] and r['normalized_arguments_equal'] for r in comparisons)
        assert all(r['source_equal'] or r['replacement_authorized'] for r in comparisons)
        after=inventory(input_paths);write(out/'build-source-after.json',after)
        assert before==after,'immutable generation/seed/build-control inputs changed'
        receipt.update(status='built_not_executed',binary={'path':str(binary),'sha256':digest(binary)},
            build_receipts={'path':str(out/'build/build-receipts.json'),'identity':actual['identity']},
            actual_compilation_units=len(actual['units']),original_compilation_flags_match_native=True,
            source_unchanged=True)
    except BaseException as exc:
        receipt.update(status='failed',error={'type':type(exc).__name__,'reason':str(exc)})
        raise
    finally:
        receipt['elapsed_seconds']=time.monotonic()-started
        write(receipt_path,receipt)
    print(json.dumps({key:receipt.get(key) for key in ('status','mode','binary','elapsed_seconds','cuda_units')}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=MODE_DIRS,required=True)
    args=parser.parse_args();worker(args.mode)

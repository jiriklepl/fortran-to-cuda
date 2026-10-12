"""Exercise wrapper command boundaries without Docker, SLURM, or a GPU."""
from __future__ import annotations

import json
import os
from pathlib import Path
import pty
import shlex
import shutil
import subprocess
import sys

import pytest

CONTAINERS = Path(__file__).resolve().parents[1]


@pytest.fixture
def sandbox(tmp_path):
    commands = tmp_path / "commands"
    commands.mkdir()
    # Deliberately omit real container and scheduler executables from PATH.
    for name in ("bash", "basename", "cat", "chmod", "date", "dirname", "env", "id", "mkdir", "realpath"):
        (commands / name).symlink_to(shutil.which(name))
    (commands / "python3.12").symlink_to(sys.executable)
    capture = tmp_path / "capture.json"
    env = {key: value for key, value in os.environ.items()
           if key not in {"WORKSPACE_ROOT", "PIPELINE_ROOT", "COMPILER_ROOT", "OUTPUT_DIR", "WORK_DIR",
                          "SIF_FILE", "IMAGE_TAG", "GPU_DEVICES", "CUDA_VISIBLE_DEVICES"}
           and not key.startswith(("APPTAINER", "SINGULARITY", "SLURM_"))}
    env.update(PATH=str(commands), CAPTURE=str(capture), OUTPUT_DIR=str(tmp_path / "output"))

    def stub(name, *, directory=commands):
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "if sys.argv[1:] == ['--version']:\n"
            "    print(os.environ.get('STUB_VERSION', 'Docker version 28.0.0'))\n"
            "    sys.exit(0)\n"
            "Path(os.environ['CAPTURE']).write_text(json.dumps({'executable':sys.argv[0], 'argv':sys.argv[1:], 'env':dict(os.environ)}))\n"
            "print(os.environ.get('STUB_STDOUT', ''))\n"
            "sys.exit(int(os.environ.get('STUB_EXIT', '0')))\n"
        )
        path.chmod(0o755)
        return path

    def invoke(script, *args, overrides=None):
        selected = {**env, **(overrides or {})}
        return subprocess.run(["/bin/bash", str(CONTAINERS / script), *map(str, args)],
                              cwd=tmp_path, env=selected, text=True, capture_output=True, timeout=10)

    return tmp_path, env, capture, stub, invoke


def values(argv, flag):
    return [argv[index + 1] for index, value in enumerate(argv[:-1]) if value == flag]


def fake_workspace(sandbox):
    tmp, env, _, stub, _ = sandbox
    workspace = tmp / "workspace & literal$"
    stub("entrypoint.sh", directory=workspace / "containers")
    stub("run-apptainer.sh", directory=workspace / "containers")
    env["WORKSPACE_ROOT"] = str(workspace)
    return workspace


def execute_payload(sandbox, node="hopper01"):
    _, env, _, _, _ = sandbox
    script = Path(env["OUTPUT_DIR"]) / f"submit-{node}.sh"
    return subprocess.run(["/bin/bash", str(script)], env=env, text=True,
                          capture_output=True, timeout=10)


def batch_request(sandbox, node="hopper01"):
    script = Path(sandbox[1]["OUTPUT_DIR"]) / f"submit-{node}.sh"
    return dict(shlex.split(line.removeprefix("#SBATCH "))[0].removeprefix("--").split("=", 1)
                for line in script.read_text().splitlines() if line.startswith("#SBATCH "))


def test_node_default_submission_and_full_case(sandbox):
    _, env, capture, stub, invoke = sandbox
    fake_workspace(sandbox)
    stub("sbatch")
    result = invoke("run_node.sh", "hopper01")
    assert result.returncode == 0, result.stderr
    submission = json.loads(capture.read_text())["argv"]
    assert "--parsable" in submission and "--wait" in submission
    request = batch_request(sandbox)
    assert request["partition"] == "gpu-ffa-gpulab"
    assert request["cpus-per-task"] == "8"
    assert request["mem"] == "32G" and request["gpus"] == "1"
    assert request["time"] == "01:30:00"
    assert not any(arg.startswith(("--partition", "--cpus-per-task", "--mem", "--gpus", "--time")) for arg in submission)
    assert execute_payload(sandbox).returncode == 0
    argv = json.loads(capture.read_text())["argv"]
    assert argv[0] == "run-elmm"
    assert values(argv, "--case") == ["channel-simple"]
    assert values(argv, "--threads") == ["4"]
    assert values(argv, "--jobs") == ["8"]
    assert values(argv, "--repeats") == ["7"]
    assert values(argv, "--warmups") == ["1"]
    assert values(argv, "--arch") == ["auto"]
    assert "--steps" not in argv
    assert values(argv, "--output-dir") == [env["OUTPUT_DIR"]]


def test_node_dry_run_quotes_and_forwards_every_option(sandbox):
    tmp, env, capture, _, invoke = sandbox
    fake_workspace(sandbox)
    config = tmp / "config & spaces"
    config.mkdir()
    marker = tmp / "must-not-exist"
    literal = f"$(touch {marker})"
    result = invoke("run_node.sh", "bw01", "--dry-run", "--runtime", "native",
                    "--case", "channel", "--config", config, "--steps", "50",
                    "--threads", "6", "--jobs", "10", "--cpus", "10",
                    "--repeats", "5", "--warmups", "0", "--arch", "sm_120",
                    "--", "--target-reduction", "0.20", "--label", literal)
    assert result.returncode == 0, result.stderr
    assert not capture.exists() and not marker.exists()
    assert execute_payload(sandbox, "bw01").returncode == 0
    argv = json.loads(capture.read_text())["argv"]
    for option, expected in {"--case": "channel", "--config": str(config), "--steps": "50",
                             "--threads": "6", "--jobs": "10", "--repeats": "5",
                             "--warmups": "0", "--arch": "sm_120", "--label": literal}.items():
        assert values(argv, option) == [expected]
    assert not marker.exists()


def test_node_missing_scheduler_never_runs_locally(sandbox):
    _, _, capture, _, invoke = sandbox
    fake_workspace(sandbox)
    result = invoke("run_node.sh", "hopper01")
    assert result.returncode != 0 and "sbatch is unavailable" in result.stderr
    assert not capture.exists()


def test_node_job_script_is_immutable(sandbox):
    _, env, _, _, invoke = sandbox
    fake_workspace(sandbox)
    assert invoke("run_node.sh", "hopper01", "--dry-run").returncode == 0
    script = Path(env["OUTPUT_DIR"]) / "submit-hopper01.sh"
    before = script.read_bytes()
    result = invoke("run_node.sh", "hopper01", "--dry-run", "--case", "channel")
    assert result.returncode != 0 and "immutable" in result.stderr
    assert script.read_bytes() == before


@pytest.mark.parametrize("args", [
    ["--cpus", "3"], ["--threads", "9"], ["--jobs", "9"], ["--warmups", "-1"],
    ["--repeats", "0"], ["--case", "wrong"], ["--runtime", "wrong"], ["--steps"],
])
def test_node_rejects_invalid_resources_before_dispatch(sandbox, args):
    _, _, capture, _, invoke = sandbox
    fake_workspace(sandbox)
    result = invoke("run_node.sh", "hopper01", *args)
    assert result.returncode != 0 and not capture.exists()


@pytest.mark.parametrize("args", [
    ["--threads", "99"], ["--jobs=99"], ["--config", "/host/config"],
    ["--output-dir=/host/results"], ["--work-dir", "/host/work"],
    ["--arch", "100"], ["--case", "channel"], ["--steps", "10"],
])
def test_node_extra_arguments_cannot_override_checked_options(sandbox, args):
    _, _, capture, _, invoke = sandbox
    fake_workspace(sandbox)
    result = invoke("run_node.sh", "hopper01", "--dry-run", "--", *args)
    assert result.returncode != 0 and "before --" in result.stderr
    assert not capture.exists()


def test_node_scheduler_status_and_no_wait(sandbox):
    _, _, capture, stub, invoke = sandbox
    fake_workspace(sandbox)
    stub("sbatch")
    result = invoke("run_node.sh", "bw01", "--no-wait", "--account", "account & literal",
                    overrides={"STUB_EXIT": "37"})
    assert result.returncode == 37
    argv = json.loads(capture.read_text())["argv"]
    assert "--parsable" in argv and "--wait" not in argv
    assert batch_request(sandbox, "bw01")["account"] == "account & literal"


def test_original_top_level_command_preserves_site_defaults_and_prints_paths(sandbox):
    _, env, capture, stub, invoke = sandbox
    fake_workspace(sandbox)
    stub("sbatch")
    result = invoke("../run_on_node.sh", "-p", "gpu-short", "-A", "kdss", "bw01", overrides={
        "STUB_STDOUT": "12345;cluster", "SBATCH_MEM_PER_NODE": "16000", "SBATCH_CPUS_PER_TASK": "4",
    })
    assert result.returncode == 0, result.stderr
    request = batch_request(sandbox, "bw01")
    assert {key:request[key] for key in ("partition", "account", "nodelist")} == {
        "partition":"gpu-short", "account":"kdss", "nodelist":"bw01"}
    record = json.loads(capture.read_text())
    # Slurm retains its original environment-over-directive precedence.
    assert record["env"]["SBATCH_MEM_PER_NODE"] == "16000"
    assert record["env"]["SBATCH_CPUS_PER_TASK"] == "4"
    assert record["argv"] == ["--parsable", "--wait", str(Path(env["OUTPUT_DIR"]) / "submit-bw01.sh")]
    for expected in (env["OUTPUT_DIR"], "experiment/report.json", "experiment/figures",
                     "Job ID: 12345", "slurm-12345.out", "slurm-12345.err", "squeue -j 12345", "scancel 12345"):
        assert expected in result.stdout


def test_reduced_cpu_request_defaults_build_jobs_to_that_budget(sandbox):
    _, _, capture, _, invoke = sandbox
    fake_workspace(sandbox)
    result = invoke("run_node.sh", "bw01", "--dry-run", "-c", "4")
    assert result.returncode == 0, result.stderr
    assert batch_request(sandbox, "bw01")["cpus-per-task"] == "4"
    assert execute_payload(sandbox, "bw01").returncode == 0
    assert values(json.loads(capture.read_text())["argv"], "--jobs") == ["4"]


def test_site_reduced_cpu_allocation_caps_default_build_jobs(sandbox):
    _, env, capture, _, invoke = sandbox
    fake_workspace(sandbox)
    env["SLURM_CPUS_PER_TASK"] = "4"
    assert invoke("run_node.sh", "bw01", "--dry-run").returncode == 0
    result = execute_payload(sandbox, "bw01")
    assert result.returncode == 0, result.stderr
    argv = json.loads(capture.read_text())["argv"]
    assert values(argv, "--jobs") == ["4"]
    assert values(argv, "--threads") == ["4"]


def test_modules_are_loaded_before_automatic_apptainer_selection(sandbox):
    tmp, env, capture, stub, invoke = sandbox
    fake_workspace(sandbox)
    module_bin = tmp / "module-tools"
    stub("apptainer", directory=module_bin)
    sif = tmp / "image.sif"
    sif.touch()
    module_log = tmp / "modules.log"
    env.update(SIF_FILE=str(sif), MODULE_BIN=str(module_bin), MODULE_LOG=str(module_log))
    env["BASH_FUNC_module%%"] = '''() { printf '%s\\n' "$*" >> "$MODULE_LOG";
if [[ "$*" == "load apptainer" ]]; then export PATH="$MODULE_BIN:$PATH"; fi
}'''
    result = invoke("run_node.sh", "bw01", "--dry-run")
    assert result.returncode == 0, result.stderr
    result = execute_payload(sandbox, "bw01")
    assert result.returncode == 0, result.stderr
    assert module_log.read_text().splitlines() == ["load apptainer", "load cuda", "load gcc", "load python3.12"]
    record = json.loads(capture.read_text())
    assert Path(record["executable"]).name == "run-apptainer.sh"
    assert "Results and logs:" in result.stdout


def test_worker_reports_failure_and_retains_status(sandbox):
    _, env, _, _, invoke = sandbox
    fake_workspace(sandbox)
    env["STUB_EXIT"] = "37"
    assert invoke("run_node.sh", "bw01", "--dry-run").returncode == 0
    result = execute_payload(sandbox, "bw01")
    assert result.returncode == 37
    assert "Job exit status: 37" in result.stdout
    assert env["OUTPUT_DIR"] in result.stdout


def test_native_uses_newer_python_without_replacing_plot_python(sandbox):
    _, env, capture, _, invoke = sandbox
    fake_workspace(sandbox)
    env["PLOT_PYTHON"] = "/system/python3.9-with-plots"
    assert invoke("run_node.sh", "bw01", "--dry-run", "--runtime", "native").returncode == 0
    result = execute_payload(sandbox, "bw01")
    assert result.returncode == 0, result.stderr
    selected = json.loads(capture.read_text())["env"]
    assert selected["PYTHON"] == "python3.12"
    assert selected["PLOT_PYTHON"] == env["PLOT_PYTHON"]
    assert "compiler/pipeline Python: python3.12" in result.stdout


def test_original_interactive_launch_preserves_account_and_resources(sandbox):
    _, _, capture, stub, invoke = sandbox
    fake_workspace(sandbox)
    stub("srun")
    result = invoke("run_node.sh", "-p", "gpu-short", "-A", "kdss", "bw01", "--interactive")
    assert result.returncode == 0, result.stderr
    record = json.loads(capture.read_text())
    assert Path(record["executable"]).name == "srun"
    assert "--partition=gpu-short" in record["argv"]
    assert "--account=kdss" in record["argv"]
    assert "--cpus-per-task=8" in record["argv"]
    assert record["argv"][-2] == "bash"


def test_node_container_maps_external_config_and_scratch(sandbox):
    tmp, env, capture, stub, invoke = sandbox
    fake_workspace(sandbox)
    stub("apptainer")
    sif = tmp / "image with spaces.sif"
    sif.touch()
    env.update(SIF_FILE=str(sif), CUDA_VISIBLE_DEVICES="GPU-selected")
    config = tmp / "outside config"
    config.mkdir()
    scratch = tmp / "outside scratch"
    result = invoke("run_node.sh", "bw01", "--dry-run", "--config", config,
                    "--work-dir", scratch)
    assert result.returncode == 0, result.stderr
    assert execute_payload(sandbox, "bw01").returncode == 0
    record = json.loads(capture.read_text())
    assert values(record["argv"], "--output-dir") == ["/output"]
    assert values(record["argv"], "--work-dir") == ["/work"]
    assert values(record["argv"], "--config") == ["/container-config"]
    assert f"{scratch}:/work" in record["env"]["APPTAINER_BINDPATH"]
    assert f"{config}:/container-config:ro" in record["env"]["APPTAINER_BINDPATH"]
    assert record["env"]["CUDA_VISIBLE_DEVICES"] == "GPU-selected"


def test_node_payload_reaches_actual_apptainer_wrapper(sandbox):
    tmp, env, capture, stub, invoke = sandbox
    workspace = fake_workspace(sandbox)
    shutil.copy2(CONTAINERS / "run-apptainer.sh", workspace / "containers" / "run-apptainer.sh")
    stub("apptainer")
    sif = tmp / "image.sif"
    sif.touch()
    env["SIF_FILE"] = str(sif)
    config = tmp / "external config"
    config.mkdir()
    result = invoke("run_node.sh", "hopper01", "--dry-run", "--config", config)
    assert result.returncode == 0, result.stderr
    result = execute_payload(sandbox)
    assert result.returncode == 0, result.stderr
    record = json.loads(capture.read_text())
    argv = record["argv"]
    command = argv[argv.index(str(sif)) + 1:]
    assert values(command, "--output-dir") == ["/output"]
    assert values(command, "--work-dir") == ["/output/work"]
    assert values(command, "--config") == ["/container-config"]
    assert f"{env['OUTPUT_DIR']}:/output" in values(argv, "--bind")
    assert f"{workspace}:/workspace:ro" in values(argv, "--bind")
    assert f"{config}:/container-config:ro" in record["env"]["APPTAINER_BINDPATH"]


def test_docker_cpu_command_has_writable_output_without_tty(sandbox):
    _, env, capture, stub, invoke = sandbox
    stub("docker")
    result = invoke("run-docker.sh", "doctor")
    assert result.returncode == 0, result.stderr
    argv = json.loads(capture.read_text())["argv"]
    assert not any(flag in argv for flag in ("-it", "-t", "--tty", "--gpus"))
    assert values(argv, "--volume") == [f"{env['OUTPUT_DIR']}:/output"]
    assert "WORK_DIR=/output/work" in values(argv, "--env")
    assert "NVIDIA_VISIBLE_DEVICES=void" in values(argv, "--env")
    assert argv[-1] == "doctor"


def test_docker_sources_are_explicit_read_only_overrides(sandbox):
    tmp, _, capture, stub, invoke = sandbox
    stub("docker")
    overrides = {}
    for name in ("WORKSPACE_ROOT", "PIPELINE_ROOT", "COMPILER_ROOT"):
        path = tmp / f"{name} & spaces"
        path.mkdir()
        overrides[name] = str(path)
    result = invoke("run-docker.sh", "prepare", "--label", "literal & value", overrides=overrides)
    assert result.returncode == 0, result.stderr
    argv = json.loads(capture.read_text())["argv"]
    mounts = values(argv, "--volume")
    for name, target in {"WORKSPACE_ROOT": "/workspace", "PIPELINE_ROOT": "/workspace/elmm-pipeline",
                         "COMPILER_ROOT": "/compiler-source"}.items():
        assert f"{overrides[name]}:{target}:ro" in mounts
    assert "--gpus" not in argv
    assert argv[-3:] == ["prepare", "--label", "literal & value"]


@pytest.mark.parametrize("selection,expected", [("2,4", '"device=2,4"'), ("GPU-selected", '"device=GPU-selected"')])
def test_docker_preserves_scheduler_gpu_selection(sandbox, selection, expected):
    _, _, capture, stub, invoke = sandbox
    stub("docker")
    result = invoke("run-docker.sh", "run-elmm", overrides={"CUDA_VISIBLE_DEVICES": selection})
    assert result.returncode == 0, result.stderr
    argv = json.loads(capture.read_text())["argv"]
    assert values(argv, "--gpus") == [expected]
    assert f"NVIDIA_VISIBLE_DEVICES={selection}" in values(argv, "--env")
    assert not any(value.startswith("CUDA_VISIBLE_DEVICES=") for value in values(argv, "--env"))


def test_docker_explicit_gpu_override_and_disable(sandbox):
    _, _, capture, stub, invoke = sandbox
    stub("docker")
    overrides = {"CUDA_VISIBLE_DEVICES": "1", "GPU_DEVICES": "GPU-selected"}
    assert invoke("run-docker.sh", "run-elmm", overrides=overrides).returncode == 0
    assert values(json.loads(capture.read_text())["argv"], "--gpus") == ['"device=GPU-selected"']
    overrides["GPU_DEVICES"] = ""
    assert invoke("run-docker.sh", "run-elmm", overrides=overrides).returncode == 0
    argv = json.loads(capture.read_text())["argv"]
    assert "--gpus" not in argv and "CUDA_VISIBLE_DEVICES=" in values(argv, "--env")


def test_podman_uses_cdi_selected_devices(sandbox):
    _, _, capture, stub, invoke = sandbox
    stub("podman")
    result = invoke("run-docker.sh", "run-elmm", overrides={"GPU_DEVICES": "0,2"})
    assert result.returncode == 0, result.stderr
    argv = json.loads(capture.read_text())["argv"]
    assert values(argv, "--device") == ["nvidia.com/gpu=0", "nvidia.com/gpu=2"]
    assert values(argv, "--unsetenv") == ["NVIDIA_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"]
    assert values(argv, "--userns") == (["keep-id"] if os.getuid() else [])
    assert values(argv, "--user") == [f"{os.getuid()}:{os.getgid()}"]


def test_podman_cpu_command_disables_image_default_gpu_visibility(sandbox):
    _, _, capture, stub, invoke = sandbox
    stub("podman")
    result = invoke("run-docker.sh", "doctor")
    assert result.returncode == 0, result.stderr
    argv = json.loads(capture.read_text())["argv"]
    assert "--device" not in argv
    assert "NVIDIA_VISIBLE_DEVICES=void" in values(argv, "--env")


def test_docker_compatibility_command_uses_podman_flags(sandbox):
    _, _, capture, stub, invoke = sandbox
    stub("docker")
    stub("podman")
    result = invoke("run-docker.sh", "run-elmm", overrides={
        "STUB_VERSION": "podman version 5.4.0", "CUDA_VISIBLE_DEVICES": "GPU-one,GPU-two", "SLURM_JOB_ID": "123",
    })
    assert result.returncode == 0, result.stderr
    record = json.loads(capture.read_text())
    assert Path(record["executable"]).name == "docker"
    assert "--gpus" not in record["argv"]
    assert values(record["argv"], "--device") == ["nvidia.com/gpu=GPU-one", "nvidia.com/gpu=GPU-two"]
    assert values(record["argv"], "--unsetenv") == ["NVIDIA_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"]
    assert values(record["argv"], "--userns") == (["keep-id"] if os.getuid() else [])


@pytest.mark.parametrize("engine,version", [
    ("docker", "Docker version 28.0.0"), ("podman", "podman version 5.4.0"),
    ("docker", "podman version 5.4.0"),
])
@pytest.mark.parametrize("stdin_tty,stdout_tty", [(True, True), (True, False), (False, True), (False, False)])
def test_interactive_flags_require_both_input_and_output_terminals(sandbox, engine, version, stdin_tty, stdout_tty):
    tmp, env, capture, stub, _ = sandbox
    stub(engine)
    master, slave = pty.openpty()
    try:
        result = subprocess.run(
            ["/bin/bash", str(CONTAINERS / "run-docker.sh"), "bash"], cwd=tmp,
            env={**env, "STUB_VERSION": version},
            stdin=slave if stdin_tty else subprocess.DEVNULL,
            stdout=slave if stdout_tty else subprocess.PIPE,
            stderr=subprocess.PIPE, timeout=10,
        )
    finally:
        os.close(slave)
        os.close(master)
    assert result.returncode == 0, result.stderr
    argv = json.loads(capture.read_text())["argv"]
    assert ("-it" in argv) == (stdin_tty and stdout_tty)
    assert (values(argv, "--userns") == ["keep-id"]) == ("podman" in version and os.getuid() != 0)


def test_rootful_podman_keeps_existing_user_namespace_behavior(sandbox):
    tmp, _, capture, stub, invoke = sandbox
    stub("podman")
    identity = tmp / "commands" / "id"
    identity.unlink()
    identity.write_text("#!/bin/bash\nprintf '0\\n'\n")
    identity.chmod(0o755)
    result = invoke("run-docker.sh", "doctor")
    assert result.returncode == 0, result.stderr
    argv = json.loads(capture.read_text())["argv"]
    assert values(argv, "--user") == ["0:0"]
    assert "--userns" not in argv


@pytest.mark.parametrize("engine,version", [
    ("docker", "Docker version 28.0.0"), ("podman", "podman version 5.4.0"),
    ("docker", "podman version 5.4.0"),
])
@pytest.mark.parametrize("job_variable,selection", [
    ("SLURM_JOB_ID", "0"), ("SLURM_JOBID", "0,1"), ("SLURM_JOB_ID", "GPU-selected,1"),
])
def test_container_engine_rejects_ambiguous_slurm_gpu_ordinals(sandbox, engine, version, job_variable, selection):
    _, _, capture, stub, invoke = sandbox
    stub(engine)
    result = invoke("run-docker.sh", "run-elmm", overrides={
        job_variable: "123", "CUDA_VISIBLE_DEVICES": selection, "STUB_VERSION": version,
    })
    assert result.returncode != 0 and "GPU_DEVICES=GPU-UUID" in result.stderr
    assert not capture.exists()


@pytest.mark.parametrize("engine,version", [
    ("docker", "Docker version 28.0.0"), ("podman", "podman version 5.4.0"),
    ("docker", "podman version 5.4.0"),
])
def test_container_engine_slurm_accepts_explicit_gpu_uuid(sandbox, engine, version):
    _, _, capture, stub, invoke = sandbox
    stub(engine)
    result = invoke("run-docker.sh", "run-elmm", overrides={
        "SLURM_JOB_ID": "123", "CUDA_VISIBLE_DEVICES": "0", "GPU_DEVICES": "GPU-selected",
        "STUB_VERSION": version,
    })
    assert result.returncode == 0, result.stderr
    argv = json.loads(capture.read_text())["argv"]
    if "podman" in version:
        assert values(argv, "--device") == ["nvidia.com/gpu=GPU-selected"]
        assert "--gpus" not in argv
    else:
        assert values(argv, "--gpus") == ['"device=GPU-selected"']


def test_build_preserves_docker_compatibility_launcher(sandbox):
    _, _, capture, stub, invoke = sandbox
    stub("docker")
    stub("podman")
    result = invoke("build-docker.sh", "--tag", "local-image:test", "--no-cache", overrides={
        "STUB_VERSION": "podman version 5.4.0",
    })
    assert result.returncode == 0, result.stderr
    assert "Engine:     docker (podman)" in result.stdout
    record = json.loads(capture.read_text())
    assert Path(record["executable"]).name == "docker"
    assert record["argv"][0] == "build"
    assert values(record["argv"], "-t") == ["local-image:test"]
    assert "--no-cache" in record["argv"]


@pytest.fixture
def sif_env(sandbox):
    tmp, env, _, _, _ = sandbox
    path = tmp / "image with spaces.sif"
    path.touch()
    env["SIF_FILE"] = str(path)
    return path


def test_apptainer_cpu_preparation_and_environment_isolation(sandbox, sif_env):
    _, env, capture, stub, invoke = sandbox
    stub("apptainer")
    result = invoke("run-apptainer.sh", "prepare", overrides={"APPTAINERENV_LD_PRELOAD": "/bad",
                                                              "SINGULARITYENV_WORK_DIR": "/bad"})
    assert result.returncode == 0, result.stderr
    record = json.loads(capture.read_text())
    argv = record["argv"]
    assert all(flag in argv for flag in ("--cleanenv", "--no-home")) and "--nv" not in argv
    assert f"{env['OUTPUT_DIR']}:/output" in values(argv, "--bind")
    assert argv[-2:] == [str(sif_env), "prepare"]
    assert not any(name.startswith(("APPTAINERENV_", "SINGULARITYENV_")) for name in record["env"])


def test_apptainer_gpu_visibility_and_existing_bind_plan(sandbox, sif_env):
    _, _, capture, stub, invoke = sandbox
    stub("apptainer")
    result = invoke("run-apptainer.sh", "run-elmm", "--arch", "120",
                    overrides={"CUDA_VISIBLE_DEVICES": "GPU-selected", "APPTAINER_BINDPATH": "/scratch:/work"})
    assert result.returncode == 0, result.stderr
    record = json.loads(capture.read_text())
    assert "--nv" in record["argv"]
    assert "CUDA_VISIBLE_DEVICES=GPU-selected" in values(record["argv"], "--env")
    assert record["env"]["APPTAINER_BINDPATH"] == "/scratch:/work"
    assert record["argv"][-3:] == ["run-elmm", "--arch", "120"]


@pytest.mark.parametrize("script,engine", [("run-docker.sh", "docker"), ("run-apptainer.sh", "apptainer")])
def test_container_failure_status_is_preserved(sandbox, sif_env, script, engine):
    _, _, _, stub, invoke = sandbox
    stub(engine)
    result = invoke(script, "run-elmm", overrides={"STUB_EXIT": "41"})
    assert result.returncode == 41

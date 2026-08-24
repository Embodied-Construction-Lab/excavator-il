import subprocess

import pytest

from excavator_il.remote_runtime import SshRuntimeHost


def test_ssh_runtime_host_runs_one_validated_remote_command():
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, stdout="ready\n", stderr="")

    host = SshRuntimeHost("jetson16@192.168.50.2", run_command=run)

    assert host.run("printf ready") == "ready\n"
    assert calls == [
        (
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=5",
                "jetson16@192.168.50.2",
                "printf ready",
            ],
            {"capture_output": True, "text": True, "timeout": 30},
        )
    ]


def test_ssh_runtime_host_rejects_remote_failure_without_losing_detail():
    def run(argv, **_kwargs):
        return subprocess.CompletedProcess(argv, 16, stdout="", stderr="serial busy\n")

    host = SshRuntimeHost("jetson16@192.168.50.2", run_command=run)

    with pytest.raises(RuntimeError, match="serial busy"):
        host.run("check serial")


def test_stop_owned_process_checks_identity_serial_release_and_cleanup():
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, stdout="released\n", stderr="")

    host = SshRuntimeHost("jetson16@192.168.50.2", run_command=run)

    host.stop_owned_process(
        pid=4242,
        identity_ere=r"[o]rin_state_sender\.py",
        serial_path="/dev/ttyTHS1",
        timeout_s=8,
        cleanup_paths=("/tmp/hybrid.start",),
    )

    remote_command = calls[0][0][-1]
    assert "pid=4242" in remote_command
    assert "grep -Eq" in remote_command
    assert "kill -TERM" in remote_command
    assert "fuser -s /dev/ttyTHS1" in remote_command
    assert "rm -f -- /tmp/hybrid.start" in remote_command


@pytest.mark.parametrize("pid", [0, -1, True])
def test_stop_owned_process_rejects_invalid_pid(pid):
    host = SshRuntimeHost("jetson16@192.168.50.2")

    with pytest.raises(ValueError, match="pid"):
        host.stop_owned_process(
            pid=pid,
            identity_ere="runtime",
            serial_path="/dev/ttyTHS1",
            timeout_s=8,
        )


def test_reclaim_serial_owner_uses_exact_known_argv_and_confirms_release():
    commands = []

    host = SshRuntimeHost("jetson16@192.168.50.2")
    result = host.reclaim_serial_owner(
        serial_path="/dev/ttyTHS1",
        known_argv_suffixes=(
            (
                "/opt/excavator/bin/excavator-il",
                "collect",
                "--config",
                "config/collection.orin.json",
            ),
            (
                "/opt/excavator-orin/bin/python",
                "-u",
                "orin_state_sender.py",
                "--serial-port",
                "/dev/ttyTHS1",
            ),
        ),
        timeout_s=8,
        execute=lambda command: commands.append(command) or "reclaimed\n",
    )

    assert result == "reclaimed"
    assert len(commands) == 1
    command = commands[0]
    assert "fuser" in command
    assert "/proc/{pid}/cmdline" in command
    assert "argv[-len(expected):] == expected" in command
    assert "os.kill(pid, signal.SIGTERM)" in command
    assert "serial owner is not reclaimable" in command


def test_reclaim_serial_owner_propagates_unknown_owner_failure():
    host = SshRuntimeHost("jetson16@192.168.50.2")

    def reject(_command):
        raise RuntimeError(
            "remote command failed: serial owner is not reclaimable: "
            "pid=77 command=/usr/bin/python unrelated.py"
        )

    with pytest.raises(RuntimeError, match="pid=77.*unrelated.py"):
        host.reclaim_serial_owner(
            serial_path="/dev/ttyTHS1",
            known_argv_suffixes=(("/opt/bin/excavator-il", "collect"),),
            timeout_s=8,
            execute=reject,
        )


def test_reclaim_hardware_gated_runtime_matches_gate_and_refuses_device_owner():
    commands = []
    host = SshRuntimeHost("jetson16@192.168.50.2")

    result = host.reclaim_hardware_gated_runtime(
        process_marker="orin_state_sender.py",
        gate_prefix="/tmp/excavator-rl-control/hybrid_",
        protected_devices=("/dev/ttyTHS1",),
        timeout_s=8,
        execute=lambda command: commands.append(command) or "reclaimed\n",
    )

    assert result == "reclaimed"
    assert len(commands) == 1
    command = commands[0]
    assert "orin_state_sender.py" in command
    assert "/tmp/excavator-rl-control/hybrid_" in command
    assert "--hardware-start-gate" in command
    assert "/dev/ttyTHS1" in command
    assert "refusing stale reclaim" in command
    assert "os.kill(pid, signal.SIGTERM)" in command


@pytest.mark.parametrize(
    ("process_marker", "gate_prefix", "protected_devices"),
    [
        ("", "/tmp/control/hybrid_", ("/dev/ttyTHS1",)),
        ("runtime", "relative/hybrid_", ("/dev/ttyTHS1",)),
        ("runtime", "/tmp/control/hybrid_", ()),
        ("runtime", "/tmp/control/hybrid_", ("relative",)),
    ],
)
def test_reclaim_hardware_gated_runtime_rejects_unsafe_scope(
    process_marker, gate_prefix, protected_devices
):
    host = SshRuntimeHost("jetson16@192.168.50.2")

    with pytest.raises(ValueError):
        host.reclaim_hardware_gated_runtime(
            process_marker=process_marker,
            gate_prefix=gate_prefix,
            protected_devices=protected_devices,
            timeout_s=8,
        )

@pytest.mark.parametrize(
    "known",
    [(), ((),), (("", "collect"),), (("bad\x00path",),)],
)
def test_reclaim_serial_owner_rejects_invalid_known_argv(known):
    host = SshRuntimeHost("jetson16@192.168.50.2")

    with pytest.raises(ValueError, match="known_argv_suffixes"):
        host.reclaim_serial_owner(
            serial_path="/dev/ttyTHS1",
            known_argv_suffixes=known,
            timeout_s=8,
        )

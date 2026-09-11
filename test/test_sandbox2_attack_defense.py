#!/usr/bin/env python3
#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.
#
"""Manual integration test: Sandbox2 attack-defense end-to-end smoke test.

Verifies that Sandbox2 defends against a traced PyTorch model that attempts to
write a file outside its allowed scope, using the real
`$TMPDIR/ml-child-ipc/<child-id>` per-child IPC layout
(`include/sandbox/CPytorchInferenceSandboxPolicy.h`'s `SChildIpcLaunchSpec`
contract, validated by `validateChildIpcLaunchSpec()`) rather than a synthetic
flat directory.

Not run in CI; use after local Sandbox2 or policy changes. CI coverage for the
*pre-execution* graph-validator layer is provided by CModelGraphValidatorTest
and test_pytorch_inference_evil_models.py; CI coverage for the syscall
inventory is CSandboxedProcessSpawnerTest_Linux. This harness is the only
proof that the *runtime* Sandbox2 filesystem/syscall boundary - not the
static graph validator - stops a malicious model that already got past model
load.

Every malicious model is launched with `--skipModelValidation`. Without that
flag, `CModelGraphValidator` rejects these particular models (they use
`aten::as_strided` with an out-of-bounds offset) before `forward()` ever
runs - so a run without the flag would report "target file not created" for
a reason that has nothing to do with Sandbox2, which is exactly the kind of
crashed-before-reaching-the-boundary false positive the "reached marker"
requirement below exists to rule out (a crash inside `getpgid` before
reaching the boundary previously produced exactly this false positive in
`testPolicyViolationDifferential`).

Each case in this harness satisfies a five-part evidence requirement:
1. Positive control: the same model is also run through the controller's
   `--disableSandbox` legacy route (Sandbox2 structurally absent) and must
   demonstrate the payload actually works there.
2. Reached marker: a `model loaded` line observed on the model's own
   `--logPipe` proves it survived `--skipModelValidation` load, and either a
   `request_id`-correlated response on the output FIFO, or a confirmed
   process death occurring only after that log line, proves `forward()` was
   entered. Absent both, the case is an inconclusive FAIL, never a silent
   PASS.
3. Negative assertion: under Sandbox2, the protected target file must not be
   created.
4. Mechanism assertion: the controller's `start`/`kill` JSON responses and
   the discovered child PID's `/proc` liveness.
5. Cleanup assertion: a `kill <pid>` command against the controller must
   report failure once the case is done, proving no live child, and hence no
   lingering FIFO listener, survives into the next case.

Usage:
    ./dev-tools/run_sandbox2_attack_defense.sh
    python3 test/test_sandbox2_attack_defense.py [--test {1,2,all}]

    1 = benign model (functional positive control)
    2 = exploit model (heap-address leak used to build a ROP chain that
        attempts an out-of-sandbox file write)

Requires: Linux, python3, torch, user namespaces (or root), and built
controller and pytorch_inference binaries under
build/distribution/platform/linux-*/bin/.
"""

import argparse
import fcntl
import json
import os
import re
import select
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

TARGET_FILE = '/usr/share/elasticsearch/config/jvm.options.d/gc.options'

# Bounded waits. Generous because Sandbox2 setup (userns, seccomp filter
# install) and libtorch model load are both slow relative to plain process
# start.
MODEL_LOAD_TIMEOUT = 20
FORWARD_PASS_TIMEOUT = 15
PID_DISCOVERY_TIMEOUT = 5
CONTROLLER_RESPONSE_TIMEOUT = 5

HEAP_ADDRESS_PATTERN = re.compile(r'0x[0-9a-fA-F]{8,}')


class PipeReaderThread(threading.Thread):
    """Thread that reads from a named pipe and writes to a file.

    One instance is scoped to exactly one FIFO for exactly one test case
    (see run_pytorch_case()) - it is always .stop()/.join()'d before its
    FIFO is removed and a same-named FIFO is recreated for the next case.
    Reusing an instance, or leaving an old one running, across cases lets a
    reader from a stale case win the open() race on the recreated FIFO and
    silently steal/split a later case's bytes (the "single shared un-drained
    FIFO reader" defect this harness fixes).
    """

    def __init__(self, pipe_path, output_file):
        self.pipe_path = pipe_path
        self.output_file = output_file
        self.fd = None
        self.running = True
        self.error = None
        super().__init__(daemon=True)

    def run(self):
        # Opened O_NONBLOCK so this never blocks waiting for a writer to
        # show up (a plain O_RDONLY open() would) - self.fd is populated
        # almost immediately either way, which is what lets stop() actually
        # interrupt this thread instead of racing a still-None self.fd
        # against a blocking open() that may never return (e.g. when
        # run_pytorch_case() bails out early because pytorch_inference never
        # opened the other end of this FIFO for writing).
        try:
            self.fd = os.open(self.pipe_path, os.O_RDONLY | os.O_NONBLOCK)
        except OSError as e:
            self.error = str(e)
            return
        try:
            with open(self.output_file, 'w') as f:
                while self.running:
                    try:
                        ready, _, _ = select.select([self.fd], [], [], 0.2)
                    except (OSError, ValueError):
                        break
                    if not ready:
                        continue
                    try:
                        data = os.read(self.fd, 4096)
                    except BlockingIOError:
                        continue
                    except OSError as e:
                        if self.running:
                            self.error = str(e)
                        break
                    if not data:
                        break
                    f.write(data.decode('utf-8', errors='replace'))
                    f.flush()
        except Exception as e:
            self.error = str(e)
        finally:
            if self.fd is not None:
                try:
                    os.close(self.fd)
                except OSError:
                    pass

    def stop(self):
        self.running = False
        if self.fd is not None:
            try:
                os.close(self.fd)
            except OSError:
                pass


class StdinKeeperThread(threading.Thread):
    """Thread that keeps stdin pipe open for controller by writing to it."""

    def __init__(self, stdin_pipe_path):
        self.stdin_pipe_path = stdin_pipe_path
        self.fd = None
        self.running = True
        super().__init__(daemon=True)

    def run(self):
        try:
            self.fd = os.open(self.stdin_pipe_path, os.O_WRONLY | os.O_NONBLOCK)
            flags = fcntl.fcntl(self.fd, fcntl.F_GETFL)
            fcntl.fcntl(self.fd, fcntl.F_SETFL, flags & ~os.O_NONBLOCK)
            while self.running:
                try:
                    os.write(self.fd, b'\n')
                    time.sleep(0.5)
                except (OSError, BrokenPipeError):
                    break
        except Exception:
            pass
        finally:
            if self.fd is not None:
                try:
                    os.close(self.fd)
                except OSError:
                    pass

    def stop(self):
        self.running = False
        if self.fd is not None:
            try:
                os.close(self.fd)
            except OSError:
                pass


def _read_new_content(path, since_offset):
    """Read only the bytes appended to path since since_offset.

    Used to scope every wait_for_*_response() call to exactly the command it
    is waiting for, instead of re-parsing the whole (ever-growing, never
    closed until process exit) JSON-array output file on every poll - the
    latter is how a response to an earlier command could leak into a later
    command's parsing.
    """
    path = Path(path)
    if not path.exists():
        return ''
    size = path.stat().st_size
    if size <= since_offset:
        return ''
    with open(path, 'r') as f:
        f.seek(since_offset)
        return f.read()


def _parse_json_objects(new_content):
    """Parse a slice of a live, never-closed JSON array (`[{...}\n,{...}`)
    into a list of dicts. Tolerates a leading comma (the slice starts mid
    array) and a missing trailing bracket (the array is still open)."""
    content = new_content.strip()
    if not content:
        return []
    if content.startswith(','):
        content = content[1:].strip()
    if not content:
        return []
    if not content.startswith('['):
        content = '[' + content
    if not content.endswith(']'):
        content = content + ']'
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return []
    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list):
        return parsed
    return []


def pid_alive(pid):
    """Best-effort liveness check via /proc. Works for the Sandbox2 sandboxee
    too: it runs in its own PID namespace but is still visible under its real
    host PID in the host's own /proc, which is the PID the controller logs and
    the PID the controller's own registry keys kill/reap on."""
    return os.path.exists(f'/proc/{pid}')


#! Both spawner backends log the child's host PID on a successful spawn, and
#! both lines are captured on the controller's log pipe:
#!   lib/sandbox/CSandboxedProcessSpawner_Linux.cc
#!     LOG_INFO(<< "Spawned sandboxed process " << processPath << " with PID " << sandboxPid)
#!   lib/core/CDetachedProcessSpawner.cc
#!     LOG_DEBUG(<< "Spawned '" << processPath << "' with PID " << childPid)
SPAWNED_PID_RE = re.compile(
    r"Spawned (?:sandboxed process )?'?(?P<path>[^'\s]+)'? with PID (?P<pid>\d+)")


def find_child_pid(controller, process_path, since_offset, timeout=PID_DISCOVERY_TIMEOUT):
    """Discover the child's host PID by parsing the controller's own log
    output, scoped to the bytes appended since since_offset (the offset taken
    immediately before the 'start' command was sent).

    Why not /proc PPid filtering: the Sandbox2 sandboxee is *not* a direct
    child of the controller process - it is forked by the Sandbox2 forkserver
    (see lib/sandbox/CSandboxedProcessSpawner_Linux.cc), so a
    `PPid == controller.process.pid` filter never matches on the sandboxed
    route and every sandboxed case would fail at PID discovery. Only the
    unsandboxed control (a real CDetachedProcessSpawner posix_spawn child)
    would ever pass such a filter.

    The controller's 'start' response carries no PID (see
    bin/controller/CCommandProcessor.cc handleStart()), so the log line each
    spawner already emits is the discovery channel - the same one an operator
    debugging a stuck deployment reads. Deliberately uniform across both
    routes: one mechanism, exercised by every case including the control.
    """
    log_path = controller.control_dir / 'controller_log_output.txt'
    deadline = time.time() + timeout
    while True:
        pid = None
        for match in SPAWNED_PID_RE.finditer(_read_new_content(log_path, since_offset)):
            if match.group('path') == process_path:
                # Last match wins: within one case only one start command is
                # issued, but a retry would append a newer line.
                pid = int(match.group('pid'))
        if pid is not None:
            return pid
        if time.time() >= deadline:
            return None
        time.sleep(0.1)


#! The controller's sandbox2_launch structured once-per-launch signal,
#! emitted by bin/controller/CProcessSpawnerRouter.cc emitLaunchSignal()
#! over the same log pipe. Boost.Log escapes the embedded quotes, so the raw
#! capture is unescaped before matching.
LAUNCH_SIGNAL_ROUTE_RE = re.compile(r'"event":"sandbox2_launch".*?"route":"(?P<route>[a-z0-9_]+)"')


def find_launch_route(controller, since_offset, timeout=PID_DISCOVERY_TIMEOUT):
    """Return the route ("sandbox2" / "legacy") the controller's own
    sandbox2_launch signal reports for the launch issued after since_offset,
    or None if no such signal appeared within timeout.

    This is the harness's guard against silently invalidating the security
    proof: a "sandboxed" case that actually routed to the legacy path would
    still produce "no target file" for entirely the wrong reason (see
    run_pytorch_case()).
    """
    log_path = controller.control_dir / 'controller_log_output.txt'
    deadline = time.time() + timeout
    while True:
        raw = _read_new_content(log_path, since_offset).replace('\\"', '"')
        route = None
        for match in LAUNCH_SIGNAL_ROUTE_RE.finditer(raw):
            # Last match wins, consistent with find_child_pid().
            route = match.group('route')
        if route is not None:
            return route
        if time.time() >= deadline:
            return None
        time.sleep(0.1)


def tail_contains(path, needle, deadline):
    """Poll path until it contains needle or deadline (a time.time() value)
    passes."""
    while time.time() < deadline:
        try:
            with open(path, 'r') as f:
                if needle in f.read():
                    return True
        except OSError:
            pass
        time.sleep(0.2)
    try:
        with open(path, 'r') as f:
            return needle in f.read()
    except OSError:
        return False


class ControllerProcess:
    """Manages the controller process and its own command/output/log/stdin
    pipes, kept in control_dir - deliberately separate from any child's
    `$TMPDIR/ml-child-ipc/<child-id>` directory, so a sandboxed child's mount
    policy for its own IPC root can never be confused with, or accidentally
    widened to include, the controller's own command channel.
    """

    def __init__(self, binary_path, control_dir, controller_dir, child_tmp_base):
        self.binary_path = binary_path
        self.control_dir = Path(control_dir)
        self.controller_dir = controller_dir
        self.process = None
        self.log_reader = None
        self.output_reader = None
        self.stdin_keeper = None
        self.cmd_pipe_fd = None
        self._output_path = self.control_dir / 'controller_output.txt'

        self.pipes = {
            'cmd': str(self.control_dir / 'controller_cmd'),
            'out': str(self.control_dir / 'controller_out'),
            'log': str(self.control_dir / 'controller_log'),
            'stdin': str(self.control_dir / 'controller_stdin'),
        }

        try:
            for pipe_path in self.pipes.values():
                if os.path.exists(pipe_path):
                    os.remove(pipe_path)
                os.mkfifo(pipe_path, stat.S_IRUSR | stat.S_IWUSR)

            script_dir = Path(__file__).parent
            source_config = script_dir / 'boost.log.ini'
            test_config = self.control_dir / 'boost.log.ini'
            if source_config.exists():
                shutil.copy(source_config, test_config)
            else:
                with open(test_config, 'w') as f:
                    f.write('[Core]\n')
                    f.write('Filter="%Severity% >= TRACE"\n')
                    f.write('\n')
                    f.write('[Sinks.Stderr]\n')
                    f.write('Destination=Console\n')

            log_file = str(self.control_dir / 'controller_log_output.txt')
            self.log_reader = PipeReaderThread(self.pipes['log'], log_file)
            self.output_reader = PipeReaderThread(self.pipes['out'], str(self._output_path))
            self.log_reader.start()
            self.output_reader.start()
            time.sleep(0.2)

            print("Pipe readers started (will connect when controller opens pipes)")
            sys.stdout.flush()
            print("Starting controller process...")
            sys.stdout.flush()

            stdin_opened = threading.Event()
            stdin_fd_holder = {'fd': None}

            def open_stdin_for_controller():
                stdin_fd_holder['fd'] = os.open(self.pipes['stdin'], os.O_RDONLY)
                stdin_opened.set()

            stdin_opener_thread = threading.Thread(target=open_stdin_for_controller, daemon=True)
            stdin_opener_thread.start()

            self.stdin_keeper = StdinKeeperThread(self.pipes['stdin'])
            self.stdin_keeper.start()

            if not stdin_opened.wait(timeout=3.0):
                raise RuntimeError("Failed to open stdin pipe - stdin_keeper did not connect")

            stdin_fd = stdin_fd_holder['fd']
            if stdin_fd is None:
                raise RuntimeError("stdin_fd is None after opening")

            print(f"stdin opened: fd={stdin_fd}, stdin_keeper: fd={self.stdin_keeper.fd}")
            sys.stdout.flush()

            # trustedTmpDir for validateChildIpcLaunchSpec() is derived by the
            # controller itself from its own TMPDIR env var
            # (CSandboxedProcessSpawner_Linux.cc / CProcessSpawnerRouter.cc both
            # read getenv("TMPDIR"), defaulting to "/tmp"). child_tmp_base must
            # therefore be passed as this process's TMPDIR, not merely used
            # locally to build pipe paths, or every child spawn will be rejected
            # for living outside the "trusted" base the controller believes in.
            env = dict(os.environ)
            env['TMPDIR'] = str(child_tmp_base)

            self._start_controller_with_stdin(stdin_fd, env)

            time.sleep(0.3)
            print(f"Controller started (PID: {self.process.pid})")
            time.sleep(1.0)

            print("Opening command pipe...")
            sys.stdout.flush()
            cmd_pipe_opened = threading.Event()
            cmd_pipe_fd_holder = {}

            def open_cmd_pipe():
                try:
                    cmd_pipe_fd_holder['fd'] = os.open(self.pipes['cmd'], os.O_WRONLY)
                except Exception as e:
                    cmd_pipe_fd_holder['error'] = e
                finally:
                    cmd_pipe_opened.set()

            cmd_pipe_thread = threading.Thread(target=open_cmd_pipe, daemon=True)
            cmd_pipe_thread.start()

            if not cmd_pipe_opened.wait(timeout=5.0):
                raise RuntimeError("Timeout waiting for controller to open command pipe")
            if 'error' in cmd_pipe_fd_holder:
                raise RuntimeError(f"Failed to open command pipe: {cmd_pipe_fd_holder['error']}")

            self.cmd_pipe_fd = cmd_pipe_fd_holder.get('fd')
            if self.cmd_pipe_fd is None:
                raise RuntimeError("cmd_pipe_fd is None after opening")

            print(f"Command pipe opened: fd={self.cmd_pipe_fd}")
            sys.stdout.flush()
        except Exception:
            # Best-effort teardown of whatever was already started
            # (subprocess, reader threads, pipes) before re-raising. main()
            # only assigns its `controller` variable after __init__ returns,
            # so if construction fails partway through, this is the only
            # place that can reap the already-spawned controller binary and
            # its reader/stdin-keeper threads - main()'s
            # `finally: if controller is not None: controller.cleanup()`
            # never runs for a partially-constructed instance.
            self.cleanup()
            raise

    def _start_controller_with_stdin(self, stdin_fd, env):
        try:
            cmd_args = [
                self.binary_path,
                '--logPipe=' + self.pipes['log'],
                '--commandPipe=' + self.pipes['cmd'],
                '--outputPipe=' + self.pipes['out'],
            ]
            self.process = subprocess.Popen(
                cmd_args,
                stdin=stdin_fd,
                stdout=open(self.control_dir / 'controller_stdout.log', 'w'),
                stderr=open(self.control_dir / 'controller_stderr.log', 'w'),
                cwd=self.controller_dir,
                env=env,
            )
            for i in range(5):
                time.sleep(0.2)
                if self.process.poll() is not None:
                    break

            if self.process.poll() is not None:
                stderr_file = self.control_dir / 'controller_stderr.log'
                stderr_msg = stderr_file.read_text() if stderr_file.exists() else ''
                raise RuntimeError(
                    f"Controller exited immediately with code {self.process.returncode}\n"
                    f"Stderr: {stderr_msg}")

            if self.log_reader.error:
                raise RuntimeError(f"Log pipe reader error: {self.log_reader.error}")
            if self.output_reader.error:
                raise RuntimeError(f"Output pipe reader error: {self.output_reader.error}")
        except Exception:
            if stdin_fd is not None:
                try:
                    os.close(stdin_fd)
                except OSError:
                    pass
            raise

    def send_command(self, command_id, verb, args):
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError(
                f"Controller process is not running "
                f"(exit code: {self.process.returncode if self.process else 'N/A'})")
        if self.cmd_pipe_fd is None:
            raise RuntimeError("Command pipe is not open")
        cmd_line = f"{command_id}\t{verb}\t" + "\t".join(args) + "\n"
        try:
            os.write(self.cmd_pipe_fd, cmd_line.encode('utf-8'))
        except Exception as e:
            raise RuntimeError(f"Failed to send command: {e}")

    def send_command_and_wait(self, command_id, verb, args, timeout=CONTROLLER_RESPONSE_TIMEOUT):
        """Send a command and wait only for bytes appended after this call -
        the per-command drain that replaces re-parsing the whole shared
        output file (see _read_new_content())."""
        since_offset = self._output_path.stat().st_size if self._output_path.exists() else 0
        self.send_command(command_id, verb, args)
        deadline = time.time() + timeout
        while time.time() < deadline:
            for obj in _parse_json_objects(_read_new_content(self._output_path, since_offset)):
                if isinstance(obj, dict) and obj.get('id') == command_id:
                    return obj
            time.sleep(0.1)
        return None

    def kill_pid(self, command_id, pid, timeout=CONTROLLER_RESPONSE_TIMEOUT):
        """Issue a controller 'kill <pid>' command. Returns the response
        dict, or None on timeout. response['success'] is False both when
        the PID was never one of the controller's live children and when it
        already exited - exactly the registry-poll cleanup mechanism the
        cleanup assertion below needs (see
        bin/controller/CCommandProcessor.cc handleKill() ->
        CSandboxedProcessSpawner::terminateChild())."""
        return self.send_command_and_wait(command_id, 'kill', [str(pid)], timeout=timeout)

    def log_offset(self):
        """Current size of the captured controller log, for scoping a later
        find_child_pid() scan to one command's own output."""
        log_file = self.control_dir / 'controller_log_output.txt'
        return log_file.stat().st_size if log_file.exists() else 0

    def check_controller_logs(self, max_lines=50):
        log_file = self.control_dir / 'controller_log_output.txt'
        if not log_file.exists():
            return
        try:
            lines = log_file.read_text().splitlines()[-max_lines:]
        except OSError:
            return
        interesting = [ln for ln in lines if
                       '"level":"ERROR"' in ln or '"level":"WARN"' in ln or
                       'sandbox' in ln.lower()]
        if interesting:
            print("--- Controller log (errors/warnings/sandbox) ---")
            for ln in interesting[-15:]:
                print(f"  {ln}")
            print("--- end ---")
            sys.stdout.flush()

    def cleanup(self):
        if self.cmd_pipe_fd is not None:
            try:
                os.close(self.cmd_pipe_fd)
            except OSError:
                pass
            self.cmd_pipe_fd = None

        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            except Exception:
                pass

        for keeper in (self.stdin_keeper, self.log_reader, self.output_reader):
            if keeper:
                keeper.stop()
                keeper.join(timeout=1)

        for pipe_path in self.pipes.values():
            try:
                if os.path.exists(pipe_path):
                    os.remove(pipe_path)
            except OSError:
                pass


def find_binaries():
    """Find controller and pytorch_inference binaries."""
    import platform

    script_dir = Path(__file__).parent
    project_root = script_dir.parent.absolute()

    machine = platform.machine()
    if machine in ('aarch64', 'arm64'):
        arch = 'linux-aarch64'
    elif machine in ('x86_64', 'amd64'):
        arch = 'linux-x86_64'
    else:
        arch = f'linux-{machine}'

    for candidate_arch in (arch, 'linux-x86_64'):
        dist_path = project_root / 'build' / 'distribution' / 'platform' / candidate_arch / 'bin'
        controller_path = dist_path / 'controller'
        pytorch_path = dist_path / 'pytorch_inference'
        if controller_path.exists():
            return str(controller_path.absolute()), str(pytorch_path.absolute())

    build_path = project_root / 'build' / 'bin'
    controller_path = build_path / 'controller' / 'controller'
    pytorch_path = build_path / 'pytorch_inference' / 'pytorch_inference'
    if controller_path.exists():
        return str(controller_path.absolute()), str(pytorch_path.absolute())

    controller_bin = os.environ.get('CONTROLLER_BIN')
    pytorch_bin = os.environ.get('PYTORCH_BIN')
    if controller_bin and pytorch_bin:
        return os.path.abspath(controller_bin), os.path.abspath(pytorch_bin)

    raise RuntimeError("Could not find controller or pytorch_inference binaries")


def send_inference_request_with_timeout(input_pipe_path, request, timeout=5):
    """Write request to input_pipe_path (blocks until pytorch_inference
    opens it for reading), bounded by timeout."""
    import queue

    result_queue = queue.Queue()

    def open_and_write():
        try:
            with open(input_pipe_path, 'w') as f:
                json.dump(request, f)
                f.flush()
            result_queue.put(True)
        except Exception as e:
            result_queue.put(e)

    writer_thread = threading.Thread(target=open_and_write, daemon=True)
    writer_thread.start()
    writer_thread.join(timeout=timeout)

    if writer_thread.is_alive():
        print(f"Warning: Timeout ({timeout}s) waiting to open pytorch_inference input pipe")
        return False
    try:
        result = result_queue.get_nowait()
    except queue.Empty:
        print("Warning: No result from inference request writer thread")
        return False
    if isinstance(result, Exception):
        print(f"Warning: Could not send inference request: {result}")
        return False
    return True


def generate_models(output_dir):
    """Generate test models using the ported generator script."""
    script_dir = Path(__file__).parent
    generator_script = script_dir / 'evil_model_generator.py'
    project_root = script_dir.parent

    if not generator_script.exists():
        raise RuntimeError(f"Model generator not found: {generator_script}")

    venv_python = project_root / 'test_venv' / 'bin' / 'python3'
    python_exec = str(venv_python) if venv_python.exists() else sys.executable

    result = subprocess.run(
        [python_exec, str(generator_script), str(output_dir)],
        capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Model generation failed: {result.stderr}")

    for model in ('model_benign.pt', 'model_exploit.pt', 'model_leak.pt'):
        if not (Path(output_dir) / model).exists():
            raise RuntimeError(f"Model {model} was not generated")


def prepare_restore_file(model_path, restore_path):
    """Wrap a .pt file with the 4-byte big-endian size header that
    CBufferedIStreamAdapter expects (matching how Elasticsearch sends
    models)."""
    model_bytes = Path(model_path).read_bytes()
    with open(restore_path, 'wb') as restore_file:
        restore_file.write(struct.pack('!I', len(model_bytes)))
        restore_file.write(model_bytes)


def make_child_ipc_root(tmp_base, child_id):
    """Create $TMPDIR/ml-child-ipc/<child-id> (mode 0700), matching the
    layout the real controller creates before policy construction per
    include/sandbox/CPytorchInferenceSandboxPolicy.h's SChildIpcLaunchSpec
    doc comment (and the pattern every C++ unit test for this contract
    already uses, e.g. CPytorchInferenceSandboxPolicyTest.cc,
    CSandboxedProcessSpawnerLifecycleTest_Linux.cc). This harness plays the
    role production code doesn't yet implement (no ml-cpp binary creates
    this directory today - see bin/controller/*.cc), the same role
    Elasticsearch's ES-side launch code will eventually play."""
    tmp_base = Path(tmp_base)
    ml_child_ipc = tmp_base / 'ml-child-ipc'
    ml_child_ipc.mkdir(mode=0o700, exist_ok=True)
    child_root = ml_child_ipc / child_id
    if child_root.exists():
        shutil.rmtree(child_root)
    child_root.mkdir(mode=0o700)
    return child_root


class CaseResult:
    def __init__(self, label):
        self.label = label
        self.ok = True
        self.notes = []

    def fail(self, message):
        self.ok = False
        self.notes.append(f"FAIL: {message}")
        print(f"FAIL: {message}")
        sys.stdout.flush()

    def info(self, message):
        self.notes.append(message)
        print(message)
        sys.stdout.flush()


def run_pytorch_case(controller, pytorch_bin, model_path, tmp_base, command_id, label,
                      unsandboxed, request_id):
    """Launch pytorch_inference against model_path through the controller,
    either sandboxed (default) or unsandboxed (--disableSandbox, the
    positive control), using the real per-child ml-child-ipc/<child-id>
    layout, and return (CaseResult, reached: bool, target_file_created: bool,
    response_or_none: dict|None, leaked_address_seen: bool).

    Every FIFO reader started here is stopped before this function returns,
    on every exit path, so no reader survives into the next case.
    """
    result = CaseResult(f"{label} ({'unsandboxed' if unsandboxed else 'sandboxed'})")
    child_id = f"{label}-{uuid.uuid4().hex[:8]}"
    child_root = make_child_ipc_root(tmp_base, child_id)

    pytorch_name = Path(pytorch_bin).name
    controller_dir = Path(controller.binary_path).parent
    pytorch_in_controller_dir = controller_dir / pytorch_name
    if pytorch_in_controller_dir.exists() or pytorch_in_controller_dir.is_symlink():
        pytorch_in_controller_dir.unlink()
    os.symlink(pytorch_bin, pytorch_in_controller_dir)

    pipes = {
        'input': str(child_root / 'input'),
        'output': str(child_root / 'output'),
        'log': str(child_root / 'log'),
    }
    for pipe_path in pipes.values():
        os.mkfifo(pipe_path, stat.S_IRUSR | stat.S_IWUSR)

    restore_path = child_root / f'{model_path.stem}_restore.bin'
    prepare_restore_file(model_path, restore_path)

    output_file = str(child_root / 'output_captured.txt')
    log_file = str(child_root / 'log_captured.txt')
    output_reader = PipeReaderThread(pipes['output'], output_file)
    log_reader = PipeReaderThread(pipes['log'], log_file)
    output_reader.start()
    log_reader.start()

    reached = False
    target_file_created = False
    response = None
    leaked_address_seen = False
    pid = None

    try:
        # Taken before the start command so find_child_pid() only ever sees
        # this case's own "Spawned ... with PID" line, never a previous
        # case's.
        log_offset = controller.log_offset()
        cmd_args = [
            f'./{pytorch_name}',
            f'--restore={restore_path}',
            f'--input={pipes["input"]}',
            '--inputIsPipe',
            f'--output={pipes["output"]}',
            '--outputIsPipe',
            f'--logPipe={pipes["log"]}',
            '--validElasticLicenseKeyConfirmed=true',
            '--skipModelValidation',
            f'--modelid={label}',
        ]
        # Explicit intent instead of a global-default side channel: every
        # "sandboxed" case sends --requireSandbox rather than relying on a
        # no-token default, so the routing decision here is the same one
        # Elasticsearch is expected to make per-launch (see
        # bin/controller/CCommandProcessor.cc). Without this, a "sandboxed"
        # case landing on the legacy path would make the harness's negative
        # assertion ("the malicious model's target file must not exist")
        # meaningless - checked against a child that was never sandboxed at
        # all.
        cmd_args.append('--disableSandbox' if unsandboxed else '--requireSandbox')

        result.info(f"Sending start command (id={command_id}) for {label}...")
        response = controller.send_command_and_wait(command_id, 'start', cmd_args)
        if response is None:
            result.fail("No response from controller to 'start' command")
            controller.check_controller_logs()
            return result, reached, target_file_created, None, leaked_address_seen, pid
        if response.get('success') is not True:
            result.fail(f"Controller rejected start: {response.get('reason')}")
            controller.check_controller_logs()
            return result, reached, target_file_created, response, leaked_address_seen, pid
        result.info(f"Controller accepted start: {response.get('reason')}")

        # Routing assertion, BEFORE any boundary assertion: the case is only
        # evidence about Sandbox2 if the controller actually routed this
        # launch the way the case intends. A sandboxed case that silently
        # landed on the legacy path (e.g. --requireSandbox not
        # reaching the controller, or a route-decision regression) would
        # still show "no target file" - for the wrong reason. Fail loudly
        # here instead.
        expected_route = 'legacy' if unsandboxed else 'sandbox2'
        actual_route = find_launch_route(controller, log_offset)
        if actual_route is None:
            result.fail(
                "No sandbox2_launch signal observed on the controller log within "
                f"{PID_DISCOVERY_TIMEOUT}s of a successful start response - cannot confirm "
                f"this launch took the '{expected_route}' route; not asserting on target file")
            controller.check_controller_logs()
            return result, reached, target_file_created, response, leaked_address_seen, pid
        if actual_route != expected_route:
            result.fail(
                f"Routing regression: controller's sandbox2_launch signal reports "
                f"\"route\":\"{actual_route}\" but this case requires "
                f"\"{expected_route}\". The child was not sandboxed as intended, so any "
                f"target-file assertion below would prove nothing about Sandbox2; "
                f"not asserting on target file")
            controller.check_controller_logs()
            return result, reached, target_file_created, response, leaked_address_seen, pid
        result.info(f"sandbox2_launch signal confirms route: {actual_route}")

        pid = find_child_pid(controller, f'./{pytorch_name}', log_offset)
        if pid is None:
            result.fail(
                "Could not discover pytorch_inference child PID from the controller's "
                f"'Spawned ... with PID' log line within {PID_DISCOVERY_TIMEOUT}s of a "
                "successful start response")
        else:
            result.info(f"Discovered child PID: {pid}")

        # Reached-marker step 1: the model survived --skipModelValidation
        # load and reached ioLoop. Without this, "no target file" is
        # indistinguishable from "crashed during model load", which is
        # exactly defect 1's false-positive pattern.
        model_loaded = tail_contains(log_file, 'model loaded',
                                     time.time() + MODEL_LOAD_TIMEOUT)
        if not model_loaded:
            result.fail(
                f"'model loaded' never observed on --logPipe within "
                f"{MODEL_LOAD_TIMEOUT}s - cannot distinguish a Sandbox2 block "
                f"from a load-time crash; not asserting on target file")
            return result, reached, target_file_created, response, leaked_address_seen, pid
        result.info("Reached marker (1/2): 'model loaded' observed on --logPipe")

        request = {
            'request_id': request_id,
            'tokens': [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]],
            'arg_1': [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]],
            'arg_2': [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]],
            'arg_3': [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]],
        }
        if not send_inference_request_with_timeout(pipes['input'], request, timeout=5):
            result.fail("Failed to write inference request to input pipe")
            return result, reached, target_file_created, response, leaked_address_seen, pid
        result.info("Inference request written")

        # Reached-marker step 2: either a response correlated to our
        # request_id (forward() ran to completion or raised a caught
        # exception), or the child dying only after "model loaded" was
        # already observed (forward() was interrupted mid-flight by
        # Sandbox2 - a crash here is a legitimate block outcome, a crash
        # before model load is not).
        forward_response = None
        deadline = time.time() + FORWARD_PASS_TIMEOUT
        while time.time() < deadline:
            for obj in _parse_json_objects(Path(output_file).read_text()
                                           if Path(output_file).exists() else ''):
                if isinstance(obj, dict) and obj.get('request_id') == request_id:
                    forward_response = obj
                    break
            if forward_response is not None:
                break
            if pid is not None and not pid_alive(pid):
                break
            time.sleep(0.2)

        if forward_response is not None:
            reached = True
            result.info(f"Reached marker (2/2): correlated output response: {forward_response}")
            error_obj = forward_response.get('error') if isinstance(forward_response, dict) else None
            if isinstance(error_obj, dict):
                message = error_obj.get('error', '')
                if HEAP_ADDRESS_PATTERN.search(str(message)):
                    leaked_address_seen = True
        elif pid is not None and not pid_alive(pid):
            reached = True
            result.info(
                "Reached marker (2/2): child PID exited after 'model loaded' was "
                "observed and the request was written - treated as forward() "
                "having been interrupted mid-flight")
        else:
            result.fail(
                f"Neither a correlated output response nor child death observed "
                f"within {FORWARD_PASS_TIMEOUT}s after sending the request - "
                f"inconclusive, not asserting on target file")
            return result, reached, target_file_created, response, leaked_address_seen, pid

        target_file_created = os.path.exists(TARGET_FILE)

    finally:
        output_reader.stop()
        log_reader.stop()
        output_reader.join(timeout=1)
        log_reader.join(timeout=1)
        for pipe_path in pipes.values():
            try:
                if os.path.exists(pipe_path):
                    os.remove(pipe_path)
            except OSError:
                pass

    return result, reached, target_file_created, response, leaked_address_seen, pid


def cleanup_and_verify_reaped(controller, result, pid, base_command_id):
    """Cleanup assertion (the fifth part of the evidence requirement): issue
    kill(pid) via the controller until it reports failure (registry has no
    such live child),
    proving the case's child is fully reaped before the next case starts.
    If the child is still alive, the first kill() should succeed (True) and
    terminate it; the follow-up kill() must then report failure."""
    if pid is None:
        result.fail("No PID discovered - cannot assert per-case cleanup/reap")
        return

    first = controller.kill_pid(base_command_id, pid)
    if first is not None and first.get('success') is True:
        result.info(f"kill({pid}) succeeded - child was still live, now terminated")
    elif first is not None and first.get('success') is False:
        result.info(f"kill({pid}) already failed - child was already reaped (e.g. Sandbox2 killed it)")
    else:
        result.fail(f"No response to first kill({pid}) command")
        return

    # Give the registry/process a moment to settle, then confirm reaped.
    time.sleep(0.3)
    second = controller.kill_pid(base_command_id + 1, pid)
    if second is None:
        result.fail(f"No response to confirmation kill({pid}) command")
        return
    if second.get('success') is not False:
        result.fail(
            f"Confirmation kill({pid}) reported success={second.get('success')!r}; "
            f"expected failure (no live child) - child may still be running/leaked")
        return
    if pid_alive(pid):
        result.fail(f"/proc/{pid} still exists after controller reported it reaped")
        return
    result.info(f"Cleanup assertion passed: pid {pid} confirmed reaped")


def test_benign_model(controller, pytorch_bin, model_path, tmp_base, command_id):
    """Functional positive control: a model using only allowlisted ops must
    run to completion under Sandbox2 and must not have its target write path
    touched (it never attempts one)."""
    print("\n" + "=" * 40)
    print("Test 1: Benign model (Sandbox2 does not break legitimate use)")
    print("=" * 40)
    sys.stdout.flush()

    result, reached, target_file_created, response, _, pid = run_pytorch_case(
        controller, pytorch_bin, model_path, tmp_base, command_id,
        'benign', unsandboxed=False, request_id='test_benign')

    if not result.ok:
        return False
    if not reached:
        result.fail("Benign model never reached a response - infrastructure problem, not a security result")
        return False
    if target_file_created:
        result.fail(f"Target file unexpectedly created by benign model: {TARGET_FILE}")
        return False

    cleanup_and_verify_reaped(controller, result, pid, command_id + 10)

    if result.ok:
        print("Benign model test passed")
    return result.ok


def test_exploit_model(controller, pytorch_bin, model_path, tmp_base, command_id):
    """Attack case: the model uses a heap-address leak (an intra-process
    memory read Sandbox2 does not, and is not meant to, block - it is not a
    syscall or filesystem boundary) to build a ROP chain that attempts to
    write a file outside the sandboxed child's allowed scope. Sandbox2's
    proof obligation is the write attempt, not the memory read; the
    positive control below demonstrates the read+write chain actually
    works when Sandbox2 is structurally absent, and the leak-address
    pattern check documents (without asserting on) the memory-disclosure
    half of the technique so the docstring stays honest about what is and
    is not defended here.

    This folds the frozen script's separate 'leak model' case in here: that
    case ran the identical target_file check as this one and asserted
    nothing about address leakage, so it tested nothing this case doesn't
    already test (see task-6 defect 3).
    """
    print("\n" + "=" * 40)
    print("Test 2: Exploit model (heap leak -> ROP chain -> file write)")
    print("=" * 40)
    sys.stdout.flush()

    if os.path.exists(TARGET_FILE):
        os.remove(TARGET_FILE)
    try:
        os.makedirs(os.path.dirname(TARGET_FILE), exist_ok=True)
    except PermissionError:
        pass

    # Positive control: same model, same request, Sandbox2 structurally
    # absent via the controller's own --disableSandbox kill switch. Without
    # this, "target file absent" only proves the mitigated run behaved
    # differently from nothing - it does not prove the mitigation stopped a
    # payload that would otherwise have succeeded.
    control_result, control_reached, control_target_created, _, control_leak_seen, control_pid = run_pytorch_case(
        controller, pytorch_bin, model_path, tmp_base, command_id,
        'exploit', unsandboxed=True, request_id='test_exploit_control')
    cleanup_and_verify_reaped(controller, control_result, control_pid, command_id + 20)

    if not control_result.ok or not control_reached:
        control_result.fail(
            "Positive control did not reach a verdict - cannot claim Sandbox2 "
            "defended against anything this run")
        return False
    if not control_target_created:
        control_result.fail(
            f"Positive control did NOT create {TARGET_FILE} - the exploit "
            f"technique itself is not demonstrated to work in this "
            f"environment (stale ROP offsets, ASLR, or a libtorch version "
            f"mismatch), so a subsequent sandboxed PASS would be meaningless")
        return False
    print(f"Positive control: exploit succeeded unsandboxed (target file created); "
          f"leaked-address pattern observed: {control_leak_seen}")
    if os.path.exists(TARGET_FILE):
        os.remove(TARGET_FILE)

    # Mitigated run: same model, same request, through Sandbox2.
    result, reached, target_file_created, _, _, pid = run_pytorch_case(
        controller, pytorch_bin, model_path, tmp_base, command_id + 1,
        'exploit', unsandboxed=False, request_id='test_exploit')
    cleanup_and_verify_reaped(controller, result, pid, command_id + 30)

    if not result.ok:
        return False
    if not reached:
        result.fail("Sandboxed run never reached a verdict - inconclusive, not a pass")
        return False
    if target_file_created:
        result.fail(f"FAIL: Target file was created under Sandbox2: {TARGET_FILE}")
        return False

    print("Exploit model test passed (file write prevented under Sandbox2, "
          "proven effective by the unsandboxed positive control)")
    return True


def main():
    parser = argparse.ArgumentParser(description='Sandbox2 Attack Defense Test')
    parser.add_argument('--test', choices=['1', '2', 'all'], default='all',
                       help='Which test to run: 1=benign, 2=exploit, all=all tests (default: all)')
    args = parser.parse_args()

    print("=" * 40)
    print("Sandbox2 Attack Defense Test")
    print("=" * 40)
    print()

    try:
        controller_bin, pytorch_bin = find_binaries()
        print(f"Using controller: {controller_bin}")
        print(f"Using pytorch_inference: {pytorch_bin}")
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    harness_root = Path(tempfile.mkdtemp(prefix='sandbox2_test_'))
    # Separate controller/child roots: the controller's own command/output/
    # log/stdin FIFOs live in control_dir; every sandboxed child's IPC
    # directory lives under child_tmp_base/ml-child-ipc/<child-id>. Passing
    # child_tmp_base as the controller's own TMPDIR is what makes
    # validateChildIpcLaunchSpec() (and CProcessSpawnerRouter's
    # emitLaunchSignal()) treat those per-child directories as trusted.
    control_dir = harness_root / 'controller_control'
    child_tmp_base = harness_root / 'child_tmp'
    models_dir = harness_root / 'models'
    control_dir.mkdir()
    child_tmp_base.mkdir()
    models_dir.mkdir()
    # Canonicalize now: validateChildIpcLaunchSpec() compares canonical
    # forms, and tempfile.mkdtemp() output can traverse a symlink (macOS
    # /tmp -> /private/tmp; some Linux distros similarly alias /tmp).
    child_tmp_base = Path(os.path.realpath(child_tmp_base))

    print(f"Harness root: {harness_root}")
    print(f"Child IPC TMPDIR: {child_tmp_base}")

    failed = False
    controller = None
    try:
        print("\nGenerating models...")
        generate_models(models_dir)
        print("Models generated successfully")

        controller_dir = Path(controller_bin).parent
        controller = ControllerProcess(controller_bin, control_dir, controller_dir, child_tmp_base)
        print(f"Controller started (PID: {controller.process.pid})")

        if args.test in ('1', 'all'):
            model_path = models_dir / 'model_benign.pt'
            if not test_benign_model(controller, pytorch_bin, model_path, child_tmp_base, 1):
                failed = True

        if args.test in ('2', 'all'):
            model_path = models_dir / 'model_exploit.pt'
            if not test_exploit_model(controller, pytorch_bin, model_path, child_tmp_base, 100):
                failed = True

        print("\n" + "=" * 40)
        if failed:
            print("Some tests FAILED")
        else:
            print("All tests PASSED")

    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        failed = True
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        failed = True
    finally:
        if controller is not None:
            controller.cleanup()
        try:
            shutil.rmtree(harness_root)
        except OSError:
            pass

    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()

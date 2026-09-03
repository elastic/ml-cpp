# Sandbox2 production failure modes

Operational watchlist for the Linux Sandbox2 integration that spawns
`pytorch_inference` from the ML controller. Use these signals when triaging
model deployment failures after PR #2873.

## Highest priority

### Sandbox cannot start (no fallback)

**Symptom:** Model deployment fails; Elasticsearch reports "Unexpected end of file"
or the inference process never appears.

**Log to search:** `Sandbox2 failed to start pytorch_inference`, `Sandbox2 environment self-check`

**Common causes:**
- Unprivileged user namespaces disabled (`kernel.unprivileged_userns_clone=0`,
  AppArmor/SELinux restrictions)
- A container runtime whose seccomp profile denies `unshare(CLONE_NEWUSER)`
- Something mounted over part of `/proc`, which makes the kernel refuse the
  fresh procfs Sandbox2 mounts (see CI hardening below for how to confirm)

**Action:** Check the controller boot-time `Sandbox2 environment self-check` INFO line
for userns and `/tmp` status. On spawn failure, the controller response and ERROR log
include the `sandbox2::Result` status. Verify user namespaces on the host. Check
`/tmp` permissions when pipe or temp-file creation fails inside the sandbox.

### FIFO / mount-namespace path mismatch

**Symptom:** Pipe connection timeouts; FIFO never visible to Elasticsearch.

**Log to search:** `Sandbox2 pytorch_inference spawn context`, `rejectedPipeArgs`,
`Failed to open` pipe messages from `CIoManager`, or ES-side
`Failed to connect to pytorch process` with `logPipe=` / `namedPipeConnectTimeout=`.

**Common causes:**
- Pipe paths passed in a form the spawner's `argDirs` extractor does not
  recognize (not `--key=/absolute/path`, relative paths, symlinked components
  not double-mounted)

**Action:** Inspect the per-spawn `Sandbox2 pytorch_inference spawn context` INFO line:
confirm `pipeDirs` contains the ES temp directory, `rejectedPipeArgs` is empty, and
`pipeDirAliases` shows any symlink double-mounts. Compare literal vs canonical pipe
directory paths.

### Seccomp policy violation (SIGSYS)

**Symptom:** `pytorch_inference` dies shortly after start or under load.

**Log to search:** `terminated abnormally` with `VIOLATION` status,
`seccomp violation: syscall=`, or signal logs from the sandbox waiter thread.

**Common causes:**
- libtorch/glibc upgrade introducing a syscall not in the Sandbox2 policy
- Missing futex operations under sustained concurrent inference

**Action:** Use the `seccomp violation: syscall=` ERROR line to identify the missing
syscall. Compare syscall allowlist in `buildPytorchInferencePolicy()` with
`CSystemCallFilter_Linux.cc` and `CPytorchInferenceSyscallAllowlist.h`.

## Medium priority

### Resource limits inside the sandbox

**Symptom:** `EMFILE` or intermittent I/O failures under heavy concurrency.

**Notes:** `rlimit_nofile` is set to 65536. Wall-time and CPU limits are
intentionally disabled for the long-lived daemon. The spawn context INFO line
records `rlimit_nofile=65536`.

**Action:** Monitor open file descriptors on busy clusters. Rely on ES-level
job limits for runaway processes.

### Filesystem view gaps

**Symptom:** ENOENT or open failures inside the sandbox only.

**Notes:** Only paths listed in the spawn context `fixedMounts` and `pipeDirs`
are visible inside the sandbox (`binDir`, `libDir`, standard system lib paths,
`/etc`, `/proc`, `/sys`, `/tmp`, and resolved pipe directories). Runtime dependencies
outside these paths (CA certs, timezone data, NSS, libtorch data files) will fail.

## Lower priority

### Lifecycle edges

**Symptom:** Orphaned sandboxed children after controller crash; rare PID reuse
window.

**Log to search:** `terminated by signal 9 (SIGKILL)` (OOM killer)

**Action:** Check host memory pressure and OOM killer logs.

## CI hardening

Set `ML_REQUIRE_SANDBOX2=1` on Linux CI runners that support user namespaces so
the attack-defense harness (`dev-tools/run_sandbox2_attack_defense.sh`) fails
instead of silently skipping when the environment is misconfigured.

### Unit-test coverage modes

`CSandboxedProcessSpawnerTest` covers two mutually exclusive modes, selected from
a probe of what the runner's kernel actually permits:

| mode | requires | covers |
|---|---|---|
| `enforced` | user namespaces available | real sandboxed spawn/terminate, filesystem-policy differential |
| `fail_closed` | user namespaces unavailable | spawn refusal, kill-switch hint in the failure reason |

The allowlist drift check is mode independent and always runs.

Set `ML_SANDBOX2_REQUIRE=enforced` (or `fail_closed`) on a runner whose namespace
support is established: the suite then fails rather than quietly covering the
other mode. Leave it unset elsewhere — developer machines and agents we have not
characterised each land wherever their kernel puts them.

**Where each mode is covered on `core-almalinux-8-aarch64` (kernel 4.18),** as
measured by `diagnose_userns.sh`:

| runner | probe | covers |
|---|---|---|
| host | all stages OK | `enforced` — **pinned** in `run_tests.sh` |
| docker, default | denied at `unshare(CLONE_NEWUSER)` | `fail_closed` |
| docker + `seccomp=unconfined` | denied at `mount(proc)`, masked `/proc` paths | — |
| docker + `seccomp` + `systempaths=unconfined` | all stages OK | — |
| docker `--privileged` | all stages OK | — |

Both modes are therefore covered on one agent, and the enforced half needs no
privilege escalation: the host runner suffices. The container run is left
unpinned on purpose — it supplies the fail-closed half, and if a Docker upgrade
ever permits `CLONE_NEWUSER` it will simply cover enforced instead, while the
host pin still guarantees enforced coverage exists.

**Diagnosing a runner.** The suite logs, once per run:

```
Sandbox2 environment self-check: uid=.. euid=.. user.max_user_namespaces=..
  kernel.unprivileged_userns_clone=.. selinux.enforce=.. mountsCoveringProc=..
Sandbox2 user-namespace probe: unavailable: mount(proc, /proc, proc) failed with errno 1 (...)
```

The probe reports the first stage the kernel denied out of
`unshare(CLONE_NEWUSER)`, the uid/gid mapping writes,
`unshare(CLONE_NEWNS|CLONE_NEWPID)`, the fork into the new PID namespace,
`mount(/, MS_REC|MS_PRIVATE)` and `mount(proc, /proc, proc)`.

Two causes of an `EPERM` from the `mount(proc)` stage are worth knowing:

- **No new PID namespace.** The kernel refuses a procfs instance for a PID
  namespace that already has one mounted, so `mount(proc)` inside a user
  namespace needs `CLONE_NEWPID` as well — and, because `CLONE_NEWPID` only
  affects children, the mount has to happen in a fork that is PID 1 of that
  namespace. Verified both ways on a 7.0 kernel:
  `unshare -U -m --map-root-user sh -c 'mount -t proc proc /proc'` fails, while
  adding `--pid --fork` succeeds. Sandbox2 unshares `CLONE_NEWPID`, so a probe
  that omits it under-reports availability.
- **A mount covering `/proc`.** The kernel refuses a fresh procfs while any
  mount covers part of `/proc`; `mountsCoveringProc` lists them. On EC2
  AlmaLinux that is usually `/proc/sys/fs/binfmt_misc`, and inside a default
  Docker container it is the runtime's masked paths.

Docker's default seccomp profile denies `unshare(CLONE_NEWUSER)` outright, so a
container fails at the first stage rather than the last unless it is privileged
or run with `--security-opt seccomp=unconfined`.

`.buildkite/scripts/steps/diagnose_userns.sh` reports the same stages across the
host and several container configurations without needing a build;
`userns_probe.sh` is the per-runner body it feeds to each.

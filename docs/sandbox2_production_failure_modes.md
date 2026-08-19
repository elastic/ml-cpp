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
sandbox integration tests fail instead of silently skipping when the environment
is misconfigured.

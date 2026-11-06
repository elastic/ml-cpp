# Tracing pytorch_inference filesystem access

Two tools for deriving and verifying the Landlock filesystem ruleset the ML
controller applies to `pytorch_inference` when a host cannot run full Sandbox2
isolation:

| File | Purpose |
|---|---|
| `trace_pytorch_inference_fs.bt` | bpftrace script: logs every Landlock-mediated filesystem syscall of any `pytorch_inference` process, with its result. |
| `analyze_pytorch_inference_fs_trace.py` | Post-processes that log into the exact set of paths accessed after the ruleset takes effect, and the denials (which must be empty). |

Neither is run in CI. They are for the developer who changes
`lib/seccomp/CLandlockFilesystemPolicy_Linux.cc` or needs to confirm the
ruleset still fits a new model, native library, or CPU.

## Why trace instead of reading the code

A Landlock ruleset that is *too tight* does not necessarily make inference
fail. Intel oneMKL selects a CPU-specific compute kernel at runtime by
`dlopen()`ing it from the bundled library directory and by reading
`/proc/cpuinfo`. If the ruleset denies either, MKL silently falls back to a
slower or lower-precision path: inference still returns a result, just a
subtly different (and slower) one, with no error in the log. The only reliable
way to know the true set of paths that must be granted is to watch a real
inference run. This tooling is what produced the current ruleset; re-run it
whenever that ruleset might need to change.

## Requirements

- Linux with `bpftrace` installed, and `CAP_BPF`/`CAP_PERFMON` (in practice,
  run the tracer under `sudo`). Kernel tracepoints for the syscalls must be
  available (any modern distribution kernel).
- Landlock enabled (Linux 5.13+ with `landlock` in the kernel `lsm=` list) if
  you want a *confined* trace; otherwise you get a baseline (unconfined) trace,
  which is still useful for diffing.
- A way to drive inference through a `pytorch_inference` process: a local
  Elasticsearch node with the ML native binaries, or the controller-driven
  harness in `dev-tools/run_sandbox2_attack_defense.sh`.

The tracer matches the process by name. The kernel truncates `comm` to 15
characters, so the match is `pytorch_inferen` (no trailing `ce`) - already
handled in the script.

## Usage

Start the tracer (it attaches to *any* `pytorch_inference` that execs while it
runs), then start a trained model deployment and send it a few inference
requests. A real model such as the packaged ELSER or multilingual-e5-small is
strongly preferred over a tiny test model: the small models barely exercise
MKL, so they will not reveal the `/proc/cpuinfo` and `libmkl_*` dependencies.

```bash
# 1. Capture. Ctrl-C after inference has run a few times.
sudo bpftrace dev-tools/trace_pytorch_inference_fs.bt | tee /tmp/pt_fs.log

# 2. In another shell: deploy a real model and infer against it a few times,
#    then wait past one periodic memory report (~10s) so the RSS reader's
#    /proc/self/statm read is captured too.

# 3. Summarize.
dev-tools/analyze_pytorch_inference_fs_trace.py /tmp/pt_fs.log
```

To compare the confined ruleset against the full access set, capture one trace
with `xpack.ml.trained_models.sandbox_enabled=true` on a host where the
Landlock fallback runs (its trace has a `LANDLOCK_RESTRICT_SELF` marker) and
one with the setting `false` (a baseline, no marker), then diff the two
summaries.

## Reading the output

Raw trace lines (see the script header for the full grammar):

```
<nsecs> <pid> <tid> <op> flags=0x<hex> ret=<n> <path>
```

`ret >= 0` is success; `ret < 0` is `-errno`, so `ret=-13` is `EACCES`, the
code Landlock returns for a denied access. `LANDLOCK_RESTRICT_SELF` marks the
instant the ruleset takes effect - only accesses *after* it are subject to it.

The analyzer prints, per process, the deduplicated set of paths accessed after
that marker and, separately, the denials. **On a correctly derived ruleset the
denial list is empty.** A denial there means either a path that must be added
to `pytorchInferenceLandlockPaths()`, or - the dangerous case - a path whose
denial is silently tolerated at a correctness or performance cost. When adding
a grant, prefer an exact file over a directory for anything under `/proc`,
`/sys` or `/etc`, and confirm inference output is unchanged (byte-for-byte on
a fixed input) against an unconfined baseline.

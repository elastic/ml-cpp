# Sandbox2 production failure modes

This document tracks the operational log vocabulary the controller and
`pytorch_inference` emit around the Sandbox2 rollout. Per
`docs/projects/mlcpp-sandbox2-pr2873/design.md` §Failure behavior and
observability, this schema is itself an API: field names and types must not
change without updating both this document and any downstream consumer
(notably PR F's ES-side observability work).

This file currently documents the log line introduced by PR E Task 4 (the
H4 structured once-per-launch enforced-mode signal) and, below, the V14
attack-defense evidence source added by PR E Task 6.

## Log vocabulary

### `sandbox2_launch`

Emitted exactly once per `CProcessSpawnerRouter::spawn()` call, for
processes eligible for sandboxing only (i.e. `processPath` is one of the
controller's configured `sandboxedProcessPaths` - never for unrelated
permitted processes such as `autodetect`). Fires on every dispatch outcome,
including a failed spawn, so it is never gated behind the controller's own
success handling.

Logged via `LOG_INFO` over the controller's existing log pipe (the same
channel/style MG8's `degradedModeAttestationMarker()` marker uses), as a
single-line JSON object.

| Field                   | Type    | Meaning |
|-------------------------|---------|---------|
| `event`                 | string  | Always `"sandbox2_launch"`. |
| `deployment_id`         | string  | `SChildIpcLaunchSpec::s_ChildId`, from a single `sandbox::validateChildIpcLaunchSpec()` call made **once per `spawn()`, before dispatch**, so the value cannot disagree with the state the dispatch decision was taken against and is populated on the `degraded`/`fail_closed` modes too. Empty string (`""`, explicit, never omitted) only when no path-bearing launch option (`input`/`output`/`restore`/`logPipe`) was present at all. Control characters, quotes and backslashes are JSON-escaped so the line stays single-line JSON. |
| `model_id`              | string  | Scanned from a `--modelid=<value>` launch argument, using the same linear string-prefix scan style as the controller's `--disableSandbox` token scan. Empty string if absent. Escaped as for `deployment_id`. |
| `route`                 | string  | `"sandbox2"` when `CProcessSpawnerRouter::ERoute::E_Sandbox2` was in effect, `"legacy"` when the controller selected `E_Legacy` - either via the operator kill-switch (`--disableSandbox`) or via the dormant no-token default (see "Dormant no-token default" below). |
| `legacy_reason`         | string  | **Only present when `route == "legacy"`** (equivalently, `mode == "degraded"`); **omitted entirely** - never `""`, never `null` - on `route == "sandbox2"`, i.e. on both `enforced` and `fail_closed`. `"kill_switch"` when a validated `--disableSandbox` token selected the legacy route, `"dormant_default"` when no token was needed and `ML_SANDBOX2_DEFAULT_ENFORCED` simply is not enabled. Provenance is passed in by `CCommandProcessor` (the only place it is known); the router never derives it from `args`. |
| `sandbox2_established`  | boolean | JSON boolean (`true`/`false`, never the string `"y"`/`"n"`). `true` iff `mode == "enforced"`, else `false`. |
| `mode`                  | string  | One of `"enforced"`, `"fail_closed"`, `"degraded"` - see mapping below. |

`legacy_reason` exists because `mode == "degraded"` alone conflates a
deliberate operator kill-switch launch with the dormant default that is in
effect for the entire rollout window - during that window every ordinary
launch is `degraded`, so the mode carries no diagnostic information on its
own. It is additive: `event`/`deployment_id`/`model_id`/`route`/
`sandbox2_established`/`mode` and their semantics are unchanged.

**`mode` mapping** (binding, PR E Task 4 controller ruling):

- `enforced` - `route == "sandbox2"` (no operator kill-switch token) and the
  Sandbox2 spawn returned `true`.
- `fail_closed` - `route == "sandbox2"` and the spawn returned `false`
  (includes the build/deployment contradiction case where `processPath` is
  configured as sandboxed but this build has no Sandbox2 support).
- `degraded` - `route == "legacy"` (operator kill-switch token present and
  validated, or the dormant no-token default in effect), regardless of
  whether the legacy spawn itself succeeded or failed. `legacy_reason` names
  which of the two it was, and is emitted only on this mode.

### Dormant no-token default

A `start` command with **no** `--disableSandbox` token for a configured
sandboxed process path selects the **legacy** route unless the internal
controller option `ML_SANDBOX2_DEFAULT_ENFORCED` is set to exactly `1`.
Anything else (unset, `""`, `0`, `true`) leaves it off. Off is the shipped
default, so this rollout starts dormant: a plain `pytorch_inference` launch
behaves exactly as it did before typed routing existed, on every platform,
including builds without Sandbox2 support. With the option on, the same
command requires Sandbox2 and never falls back (V2).

`ML_SANDBOX2_DEFAULT_ENFORCED` is an internal seam, not an operator setting;
the change that turns it on is the Elasticsearch-side default-false feature
flag, not ml-cpp.

Provenance lines (`LOG_INFO`/`LOG_DEBUG`, `bin/controller/CCommandProcessor.cc`)
name which of the two decided a legacy route - the router itself only ever
sees an already-decided route and never claims a kill switch that was not
present.

### In-process seccomp is legacy-route only

`pytorch_inference` installs its own in-process seccomp filter - and emits
`{"ml_sandbox2_route":"legacy","event":"seccomp_installed"}` - only when
`ML_SANDBOXED` is **not** exactly `1`. On a Sandbox2-launched child
(`ML_SANDBOXED=1`, set by `CSandboxedProcessSpawner`), the installation, the
hard-termination decision and the attestation marker are all skipped
entirely: the executor's own policy is the security boundary, an install
attempt from inside the sandbox could fail and terminate an otherwise-healthy
enforced launch, and emitting the marker would attest a legacy-route filter
on a launch `sandbox2_launch` reports as `"route":"sandbox2"`. So a
`"route":"sandbox2"` launch never carries a `seccomp_installed` marker, and
that absence is expected, not a missing signal.

`ML_SANDBOXED` is a fail-open marker, so it is stripped from the environment
of every child the legacy spawner launches
(`lib/core/CDetachedProcessSpawner.cc`, `detail::buildChildEnvironment()`) -
an inherited or externally injected `ML_SANDBOXED=1` in the controller's own
environment can therefore never suppress a legacy-route child's mandatory
in-process filter. Only `CSandboxedProcessSpawner` sets it, and only on real
sandboxees.

Hard termination on a failed in-process seccomp installation
(`TERMINATE_ON_DEGRADED_SECCOMP_FAILURE` in
`bin/pytorch_inference/Main.cc`) is deliberately **off** while the legacy
route is still the production default: during the dormant window every
ordinary launch is a degraded-route launch, so terminating would fail every
launch on a host without usable seccomp BPF. It becomes safe to activate at
the same time the default stops being legacy.

Example:

```json
{"event":"sandbox2_launch","deployment_id":"a1b2c3","model_id":"my-model","route":"sandbox2","sandbox2_established":true,"mode":"enforced"}
{"event":"sandbox2_launch","deployment_id":"a1b2c3","model_id":"my-model","route":"legacy","legacy_reason":"dormant_default","sandbox2_established":false,"mode":"degraded"}
```

Emission site: `bin/controller/CProcessSpawnerRouter.cc`,
`CProcessSpawnerRouter::spawn()` (via the private `emitLaunchSignal()`
helper) - chosen because this class owns both the already-decided route
parameter and the actual spawn-outcome boolean the `mode` field depends on.

## V14 evidence source: attack-defense harness

Per `docs/projects/mlcpp-sandbox2-pr2873/design.md`'s Required proof matrix,
V14 ("Attack-defense harness blocks maintained malicious models on PR tip
after proving each model reached execution") closes only with a dated
`attack-defense-<ml-cpp-head-sha>.md` record in this directory. This section
names the harness that produces that evidence and the exact command; it does
not itself constitute a V14 closure record (no run has been recorded against
a head SHA yet - the harness requires production-like Linux with Sandbox2,
so it is Buildkite/manual-devbox-deferred, per design.md: "Permanent CI is
optional; final-tip evidence is not").

**Harness:** `test/test_sandbox2_attack_defense.py`, invoked via
`dev-tools/run_sandbox2_attack_defense.sh`. It drives the real controller /
`pytorch_inference` binaries through the actual
`$TMPDIR/ml-child-ipc/<child-id>` per-child IPC layout (see
`include/sandbox/CPytorchInferenceSandboxPolicy.h`'s `SChildIpcLaunchSpec`),
and satisfies the Verification contract's Oracle rule for every case: an
unsandboxed positive control (`--disableSandbox`), a reached marker (a
`model loaded` line on the model's own `--logPipe`, plus either a
`request_id`-correlated output-pipe response or a confirmed post-load
process death), a negative assertion (protected file absent under
Sandbox2), a mechanism assertion (controller `start`/`kill` JSON responses
and `/proc` PID liveness for the PID parsed out of the controller's own
`Spawned ... with PID <n>` log line - the sandboxee is a child of the
Sandbox2 forkserver, not of the controller, so `/proc` `PPid` filtering
cannot find it), and a per-case cleanup assertion (`kill <pid>`
against the controller reports failure once the case ends, proving the
child was reaped).

Because the shipped no-token default is the legacy route, the harness starts
the controller with `ML_SANDBOX2_DEFAULT_ENFORCED=1` in its environment, and
each case asserts the route reported by that launch's own `sandbox2_launch`
signal (`sandbox2` for the sandboxed cases, `legacy` for the
`--disableSandbox` control) **before** any target-file assertion. Without
both, a sandboxed case could route to the legacy path and still show "no
target file" for entirely the wrong reason - a false pass on the security
proof.

**Command:**

```bash
./dev-tools/run_sandbox2_attack_defense.sh
# or directly:
python3 test/test_sandbox2_attack_defense.py --test all
```

**Models exercised:** `model_benign.pt` (functional positive control -
Sandbox2 must not break a legitimate model) and `model_exploit.pt` (a
heap-address leak used to build a ROP chain that attempts to write
`/usr/share/elasticsearch/config/jvm.options.d/gc.options` outside the
sandboxed child's allowed scope). `model_leak.pt` is generated by
`test/evil_model_generator.py` but not asserted on separately - see that
harness's `test_exploit_model` docstring for why a standalone leak
assertion tested nothing beyond the exploit case.

**A closing V14 record must additionally capture:** host/kernel (e.g.
`uname -a`), date, pass/fail per model exercised, the cleanup result (each
case's kill/reap confirmation), and a CI/build link when available, named
`attack-defense-<ml-cpp-head-sha>.md` in this directory.

# Sandbox2 production failure modes

This document tracks the operational log vocabulary the controller and
`pytorch_inference` emit around the Sandbox2 rollout. This schema is itself
an API: field names and types must not change without updating both this
document and any downstream consumer (notably a future change's ES-side
observability work).

This file currently documents the `sandbox2_launch` structured
once-per-launch enforced-mode signal and, below, the attack-defense evidence
source used to validate the Sandbox2 security boundary.

## Log vocabulary

### `sandbox2_launch`

Emitted exactly once per `CProcessSpawnerRouter::spawn()` call, for
processes eligible for sandboxing only (i.e. `processPath` is one of the
controller's configured `sandboxedProcessPaths` - never for unrelated
permitted processes such as `autodetect`). Fires on every dispatch outcome,
including a failed spawn, so it is never gated behind the controller's own
success handling.

Logged via `LOG_INFO` over the controller's existing log pipe (the same
channel/style `degradedModeAttestationMarker()` marker uses), as a
single-line JSON object.

| Field                   | Type    | Meaning |
|-------------------------|---------|---------|
| `event`                 | string  | Always `"sandbox2_launch"`. |
| `deployment_id`         | string  | `SChildIpcLaunchSpec::s_ChildId`, from a single `sandbox::validateChildIpcLaunchSpec()` call made **once per `spawn()`, before dispatch**, so the value cannot disagree with the state the dispatch decision was taken against and is populated on the `degraded`/`fail_closed` modes too. Empty string (`""`, explicit, never omitted) only when no path-bearing launch option (`input`/`output`/`restore`/`logPipe`) was present at all. Control characters, quotes and backslashes are JSON-escaped so the line stays single-line JSON. |
| `model_id`              | string  | Scanned from a `--modelid=<value>` launch argument, using the same linear string-prefix scan style as the controller's `--disableSandbox` token scan. Empty string if absent. Escaped as for `deployment_id`. |
| `route`                 | string  | `"sandbox2"` when `CProcessSpawnerRouter::ERoute::E_Sandbox2` was in effect, `"legacy"` when the controller selected `E_Legacy` - either via the operator kill-switch (`--disableSandbox`), the operator opt-in (`--requireSandbox`) selecting Sandbox2 instead, or the no-token default (see "No-token default" below). |
| `legacy_reason`         | string  | **Only present when `route == "legacy"`** (equivalently, `mode == "degraded"`); **omitted entirely** - never `""`, never `null` - on `route == "sandbox2"`, i.e. on both `enforced` and `fail_closed`. `"kill_switch"` when a validated `--disableSandbox` token selected the legacy route, `"no_token_default"` when neither routing token was present. Provenance is passed in by `CCommandProcessor` (the only place it is known); the router never derives it from `args`. |
| `sandbox2_established`  | boolean | JSON boolean (`true`/`false`, never the string `"y"`/`"n"`). `true` iff `mode == "enforced"`, else `false`. |
| `mode`                  | string  | One of `"enforced"`, `"fail_closed"`, `"degraded"` - see mapping below. |
| `sandbox2_compiled_in`  | boolean | JSON boolean. Sourced from `sandbox::CMlSandboxAvailability::isCompiledIn()`, computed once (a build-time-constant fact, not per-launch state) and included on **every** emitted line, unlike `legacy_reason` which is conditional on route. Lets a consumer distinguish "Sandbox2 supported but no routing token sent" (`route == "legacy"`, `legacy_reason == "no_token_default"`, `sandbox2_compiled_in == true`) from "built without Sandbox2 support at all" (`sandbox2_compiled_in == false`) - both otherwise emit identical `legacy`/`no_token_default`/`degraded` signals for every plain launch. |

`legacy_reason` exists because `mode == "degraded"` alone conflates a
deliberate operator kill-switch launch with the permanent no-token
default - a caller that never sends either routing token always produces
`degraded`, so the mode carries no diagnostic information on its own. It is
additive: `event`/`deployment_id`/`model_id`/`route`/
`sandbox2_established`/`mode` and their semantics are unchanged.

**`mode` mapping** (binding rule):

- `enforced` - `route == "sandbox2"` (a validated `--requireSandbox` token,
  or - historically, before that token existed - the no-token default with
  the now-removed internal enforcement seam) and the Sandbox2 spawn returned
  `true`.
- `fail_closed` - `route == "sandbox2"` and the spawn returned `false`
  (includes the build/deployment contradiction case where `processPath` is
  configured as sandboxed but this build has no Sandbox2 support).
- `degraded` - `route == "legacy"` (operator kill-switch token present and
  validated, or the no-token default in effect), regardless of whether the
  legacy spawn itself succeeded or failed. `legacy_reason` names which of
  the two it was, and is emitted only on this mode.

### No-token default

The command wire format defines exactly two routing tokens:
`--disableSandbox` (operator kill-switch, forces the legacy route) and
`--requireSandbox` (operator opt-in, forces the Sandbox2 route - no
automatic legacy fallback). They are mutually exclusive; a `start` command
naming both is rejected outright rather than resolved by precedence, and
each is separately rejected if repeated.

A `start` command with **neither** token for a configured sandboxed process
path always selects the **legacy** route. This is the permanent behaviour
for any caller that sends no routing token - not a temporary rollout
seam - so a plain `pytorch_inference` launch behaves exactly as it did
before typed routing existed, on every platform, including builds without
Sandbox2 support. Elasticsearch is expected to always send exactly one of
the two tokens, chosen from the live value of its own operator setting at
launch time, so this branch exists for non-ES callers (support/debug
scripts, direct controller invocation) and the test harness.

Provenance lines (`LOG_INFO`/`LOG_DEBUG`, `bin/controller/CCommandProcessor.cc`)
name which token (if any) decided the route - the router itself only ever
sees an already-decided route and never claims a token that was not
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
`bin/pytorch_inference/Main.cc`) is deliberately **off**: an ordinary launch
with no explicit routing token is a degraded-route launch, so terminating
would fail every launch on a host without usable seccomp BPF. It becomes
safe to activate once every caller that matters always sends an explicit
`--disableSandbox` or `--requireSandbox` token per launch.

Example:

```json
{"event":"sandbox2_launch","deployment_id":"a1b2c3","model_id":"my-model","route":"sandbox2","sandbox2_established":true,"mode":"enforced","sandbox2_compiled_in":true}
{"event":"sandbox2_launch","deployment_id":"a1b2c3","model_id":"my-model","route":"legacy","legacy_reason":"no_token_default","sandbox2_established":false,"mode":"degraded","sandbox2_compiled_in":true}
```

Emission site: `bin/controller/CProcessSpawnerRouter.cc`,
`CProcessSpawnerRouter::spawn()` (via the private `emitLaunchSignal()`
helper) - chosen because this class owns both the already-decided route
parameter and the actual spawn-outcome boolean the `mode` field depends on.

## Attack-defense harness evidence

The required proof for the Sandbox2 security boundary is that the
attack-defense harness blocks maintained malicious models on the ml-cpp PR
tip, after proving each model actually reached execution (not merely that it
crashed before getting there). This closes only with a dated
`attack-defense-<ml-cpp-head-sha>.md` record in this directory. This section
names the harness that produces that evidence and the exact command; it does
not itself constitute a closure record (no run has been recorded against a
head SHA yet - the harness requires production-like Linux with Sandbox2, so
it runs on Buildkite or a manual devbox rather than as a permanent CI gate;
permanent CI coverage is optional, but final-tip evidence before a release is
not).

**Harness:** `test/test_sandbox2_attack_defense.py`, invoked via
`dev-tools/run_sandbox2_attack_defense.sh`. It drives the real controller /
`pytorch_inference` binaries through the actual
`$TMPDIR/ml-child-ipc/<child-id>` per-child IPC layout (see
`include/sandbox/CPytorchInferenceSandboxPolicy.h`'s `SChildIpcLaunchSpec`),
and satisfies, for every case, the five-part evidence requirement (a
positive control, a reached marker, a negative assertion, a mechanism
assertion, and a cleanup assertion): an unsandboxed positive control
(`--disableSandbox`), a reached marker (a
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

Because the no-token default is always the legacy route, the harness sends
an explicit `--requireSandbox` token on every sandboxed case's `start`
command (and `--disableSandbox` on the positive-control case), and each case
asserts the route reported by that launch's own `sandbox2_launch` signal
(`sandbox2` for the sandboxed cases, `legacy` for the `--disableSandbox`
control) **before** any target-file assertion. Without both, a sandboxed
case could route to the legacy path and still show "no target file" for
entirely the wrong reason - a false pass on the security proof.

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

**A closing record must additionally capture:** host/kernel (e.g.
`uname -a`), date, pass/fail per model exercised, the cleanup result (each
case's kill/reap confirmation), and a CI/build link when available, named
`attack-defense-<ml-cpp-head-sha>.md` in this directory.

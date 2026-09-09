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
| `deployment_id`         | string  | `SChildIpcLaunchSpec::s_ChildId`, derived by re-running `sandbox::validateChildIpcLaunchSpec()` against the launch args. Empty string (`""`, explicit, never omitted) when no path-bearing launch option (`input`/`output`/`restore`/`logPipe`) was present. |
| `model_id`              | string  | Scanned from a `--modelid=<value>` launch argument, using the same linear string-prefix scan style as the controller's `--disableSandbox` token scan. Empty string if absent. |
| `route`                 | string  | `"sandbox2"` when `CProcessSpawnerRouter::ERoute::E_Sandbox2` was in effect, `"legacy"` when the operator kill-switch (`--disableSandbox`) routed to `E_Legacy`. |
| `sandbox2_established`  | boolean | JSON boolean (`true`/`false`, never the string `"y"`/`"n"`). `true` iff `mode == "enforced"`, else `false`. |
| `mode`                  | string  | One of `"enforced"`, `"fail_closed"`, `"degraded"` - see mapping below. |

**`mode` mapping** (binding, PR E Task 4 controller ruling):

- `enforced` - `route == "sandbox2"` (no operator kill-switch token) and the
  Sandbox2 spawn returned `true`.
- `fail_closed` - `route == "sandbox2"` and the spawn returned `false`
  (includes the build/deployment contradiction case where `processPath` is
  configured as sandboxed but this build has no Sandbox2 support).
- `degraded` - `route == "legacy"` (operator kill-switch token present and
  validated), regardless of whether the legacy spawn itself succeeded or
  failed.

Example:

```json
{"event":"sandbox2_launch","deployment_id":"a1b2c3","model_id":"my-model","route":"sandbox2","sandbox2_established":true,"mode":"enforced"}
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
and `/proc` PID liveness), and a per-case cleanup assertion (`kill <pid>`
against the controller reports failure once the case ends, proving the
child was reaped).

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

# Sandbox2 production failure modes

This document tracks the operational log vocabulary the controller and
`pytorch_inference` emit around the Sandbox2 rollout. Per
`docs/projects/mlcpp-sandbox2-pr2873/design.md` §Failure behavior and
observability, this schema is itself an API: field names and types must not
change without updating both this document and any downstream consumer
(notably PR F's ES-side observability work).

This file currently documents only the log line introduced by PR E Task 4
(the H4 structured once-per-launch enforced-mode signal). A later task (PR E
Task 6) extends it with the remaining production failure-mode vocabulary.

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

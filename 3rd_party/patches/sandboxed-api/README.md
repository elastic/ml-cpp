# Sandboxed API source patches

These patches are applied by [`3rd_party/CMakeLists.txt`](../CMakeLists.txt)
to the vendored `sandboxed-api` checkout (pinned via `FetchContent` to
`GIT_TAG` below) before it is added as a build subdirectory. They replace an
earlier approach that rewrote these files with inline `string(REGEX REPLACE
...)`/`file(WRITE ...)` calls at configure time — fragile because a silent
non-match left the intended change unapplied instead of failing the build.

Applying via `git apply` instead means a patch that no longer matches the
pinned tag's content **fails the configure step loudly** (`FATAL_ERROR`)
rather than degrading into an unpatched build.

Pinned tag: `v20241008` at commit `9e07542a03fefa2cf982ba093b099805362df05d`
(see `ML_SANDBOXED_API_TAG` / `ML_SANDBOXED_API_GIT_SHA` in
`3rd_party/CMakeLists.txt`).

## Patches

| File | Target | Why |
|---|---|---|
| `0001-abseil-cpp-disable-gtest.patch` | `cmake/abseil-cpp.cmake` | The vendored Abseil `FetchContent` override otherwise builds gtest, which ml-cpp does not vendor and does not need. |
| `0002-no-fno-exceptions-propagation.patch` | `CMakeLists.txt` | `sapi_base` exports `-fno-exceptions` as `PUBLIC`; linking against it would propagate that flag into ml-cpp targets, which use exceptions. |
| `0003-python3-optional.patch` | `cmake/SapiDeps.cmake` | `find_package(Python3 ... REQUIRED)` is only needed for `add_sapi_library()` protobuf code generation, which `MlSandbox` does not use; a missing interpreter should not fail configuration. |
| `0004-forkserver-zlib-static-libstdcxx.patch` | `sandboxed_api/sandbox2/CMakeLists.txt` | `sandbox2::unwind` (`libunwind_ptrace`) calls `uncompress()` from libz, which `--as-needed` can drop without an explicit link; the embedded forkserver binary also needs static `libstdc++`/`libgcc` so it does not depend on the host's runtime GLIBCXX version at exec time. |
| `0005-forkserver-initial-namespaces-no-hang.patch` | `sandboxed_api/sandbox2/forkserver.cc` | `CreateInitialNamespaces()` waits for its helper child with a blocking `read()` on an eventfd, which never returns if the helper dies first - e.g. when the host lets the user namespace be created but denies writing `uid_map`. The fork server then never answers, so `Sandbox2::RunAsync()` never returns and the single-threaded ML controller stops answering Elasticsearch altogether. The patch waits in bounded steps and aborts the fork server once the helper is gone, so the launch fails instead. Upstream later replaced this handshake with a `Comms` socket, which has the same effect. |

## Bumping the pinned tag

1. Update `ML_SANDBOXED_API_TAG`, resolve its commit SHA into
   `ML_SANDBOXED_API_GIT_SHA`, and update `sandbox2-INFO.csv` `revision`
   in `3rd_party/CMakeLists.txt`.
2. Re-run configure. A patch that no longer applies fails with
   `FATAL_ERROR: sandboxed-api patch <name> failed to apply` — this is the
   version-drift signal.
3. For each failing patch, regenerate it against the new tag's real file
   content (clone the tag, make the same edit, `git diff`) rather than
   hand-editing the `.patch` file — hand-edited patches drift from what the
   new tag's file actually contains.
4. Reconfigure again to confirm every patch now applies cleanly, then
   rebuild `lib/sandbox` (`ml_test_sandbox`) to confirm the resulting
   Sandbox2 build still passes.

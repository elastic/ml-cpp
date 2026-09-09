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

Pinned tag: `v20241008` (see `3rd_party/CMakeLists.txt`).

## Patches

| File | Target | Why |
|---|---|---|
| `0001-abseil-cpp-disable-gtest.patch` | `cmake/abseil-cpp.cmake` | The vendored Abseil `FetchContent` override otherwise builds gtest, which ml-cpp does not vendor and does not need. |
| `0002-no-fno-exceptions-propagation.patch` | `CMakeLists.txt` | `sapi_base` exports `-fno-exceptions` as `PUBLIC`; linking against it would propagate that flag into ml-cpp targets, which use exceptions. |
| `0003-python3-optional.patch` | `cmake/SapiDeps.cmake` | `find_package(Python3 ... REQUIRED)` is only needed for `add_sapi_library()` protobuf code generation, which `MlSandbox` does not use; a missing interpreter should not fail configuration. |
| `0004-forkserver-zlib-static-libstdcxx.patch` | `sandboxed_api/sandbox2/CMakeLists.txt` | `sandbox2::unwind` (`libunwind_ptrace`) calls `uncompress()` from libz, which `--as-needed` can drop without an explicit link; the embedded forkserver binary also needs static `libstdc++`/`libgcc` so it does not depend on the host's runtime GLIBCXX version at exec time. |

## Bumping the pinned tag

1. Update `GIT_TAG` in `3rd_party/CMakeLists.txt`.
2. Re-run configure. A patch that no longer applies fails with
   `FATAL_ERROR: sandboxed-api patch <name> failed to apply` — this is the
   version-drift signal.
3. For each failing patch, regenerate it against the new tag's real file
   content (clone the tag, make the same edit, `git diff`) rather than
   hand-editing the `.patch` file — hand-edited patches drift from what the
   new tag's file actually contains.
4. Re-run `lib/sandbox/unittest` (`ml_test_sandbox`), which re-applies every
   patch against a fresh shallow clone of the pinned tag as part of the
   patch-drift check, independent of the FetchContent build.

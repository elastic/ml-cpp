# ml-cpp AI Context

This file provides domain knowledge for AI coding assistants working on the ml-cpp repository. It consolidates the detailed rules in `.cursor/rules/` into a single reference.

For full details, see the individual files in `.cursor/rules/`:
- `ml-cpp-build-system.mdc` — CMake, Gradle, Docker, build acceleration
- `ml-cpp-buildkite-ci.mdc` — CI pipelines, API access, known failures
- `ml-cpp-coding-conventions.mdc` — Naming, cross-platform, testing patterns
- `ml-cpp-ci-analytics.mdc` — Elasticsearch, anomaly detection, AI analysis

---

## Build System

- **CMake** is the primary build system. Toolchain files in `cmake/` per platform.
- **Gradle** (`build.gradle`) orchestrates macOS and Windows CI builds, invoking CMake.
- **Docker** is used for Linux builds (`dev-tools/docker/docker_entrypoint.sh`).
- `include(CTest)` reserves the `test` target name — our monolithic test target is `ml_test`.
- `test_individually` runs tests via CTest with parallel execution.

### Build Acceleration Options
- `-DCMAKE_UNITY_BUILD=ON` — combines source files (not effective on all libraries)
- `-DML_PCH=ON` — precompiled headers for STL/Boost
- sccache with GCS backend for persistent compiler caching
- MSVC uses `/Z7` (not `/Zi`) to avoid PDB serialisation bottleneck

### Test Parallelism
- Formula: `numCpus <= 4 ? 2 : ceil(numCpus / 2)`
- Never use wall-clock time for performance assertions — use `std::clock()` (CPU time)
- Use process ID for unique temp file names in tests, not random numbers

## Coding Conventions

- Classes: `CUpperCamelCase`, Methods: `lowerCamelCase`, Members: `m_Name`
- Types: `TUpperCamelCase`, Test files: `CClassNameTest.cc`
- Commit messages: `[ML] Short description`
- Use `peek() == std::char_traits<char>::eof()` for portable end-of-stream detection
- Avoid anonymous-namespace constants with common names in libraries using unity builds

## CI (Buildkite)

- PR pipeline: `ml-cpp-pr-builds` (`.buildkite/pipeline.json.py`)
- Nightly: `ml-cpp-snapshot-builds` (`.buildkite/branch.json.py`)
- Debug: `ml-cpp-debug-build` (`.buildkite/job-build-test-all-debug.json.py`)
- Platforms: Linux x86_64 (6 vCPU), Linux aarch64 (8 vCPU), macOS aarch64 (4 core), Windows x86_64 (16 vCPU)
- Vault secrets via `.buildkite/hooks/post-checkout`
- Diagnostic steps use: `if: "build.state == 'failed'"` + `soft_fail: true` + `allow_dependency_failure: true`

## CI Analytics

- Build timings indexed into Elasticsearch Serverless (`buildkite-build-timings`)
- ML anomaly detection job: `build-timing-regressions` (high_mean by step_key)
- PR regression check: compares against 30-day baseline (mean + 2σ)
- AI failure analysis: Claude diagnoses failures, posts Buildkite annotations
- Kibana dashboard: "ML-CPP Build Timing Overview"
# Test

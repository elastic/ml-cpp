# dev-tools

Developer and CI utility scripts for the ml-cpp project.

## CI Build Analytics

### `ingest_build_timings.py`

Fetches step-level timing data from Buildkite and indexes it into the
`buildkite-build-timings` Elasticsearch index for anomaly detection and
trend analysis.

```bash
# Backfill last 30 builds from all nightly pipelines
python3 dev-tools/ingest_build_timings.py --backfill 30

# Ingest a specific build
python3 dev-tools/ingest_build_timings.py --pipeline ml-cpp-snapshot-builds --build 5819

# Dry run (print without indexing)
python3 dev-tools/ingest_build_timings.py --pipeline ml-cpp-snapshot-builds --latest --dry-run
```

Each document records: pipeline, build number, branch, step key, platform,
step type, duration, agent wait time, and state.

**Environment variables:**
- `BUILDKITE_TOKEN` or `BUILDKITE_API_READ_TOKEN` — Buildkite API read token
- `ES_ENDPOINT` — Elasticsearch endpoint (or `~/.elastic/serverless-endpoint`)
- `ES_API_KEY` — Elasticsearch API key (or `~/.elastic/serverless-api-key`)

In CI, credentials are injected from Vault by the `post-checkout` hook.

### `check_build_regression.py`

Compares step durations from a PR build against a 30-day rolling baseline
from nightly builds. Flags steps whose duration exceeds the baseline
mean + 2σ.

```bash
# Check a specific build
python3 dev-tools/check_build_regression.py --pipeline ml-cpp-pr-builds --build 1234

# In CI (uses BUILDKITE_PIPELINE_SLUG and BUILDKITE_BUILD_NUMBER)
python3 dev-tools/check_build_regression.py --annotate
```

Exits with code 1 if regressions are detected. With `--annotate`, posts
results as a Buildkite annotation.

**Environment variables:** same as `ingest_build_timings.py`.

## CI Build Infrastructure

### `run_es_tests.sh`

Runs Elasticsearch Java integration tests against locally-built ml-cpp
artifacts. Used by the CI pipeline's Java IT steps.

Arguments: `$1` = parent directory for ES clone, `$2` = path to local Ivy repo.

Includes Gradle build cache restore/upload via GCS when
`GRADLE_BUILD_CACHE_GCS_BUCKET` is set.

### `setup_sccache.sh` / `setup_sccache.ps1`

Downloads and configures sccache with a GCS backend for CI builds.
See [SCCACHE_SETUP.md](SCCACHE_SETUP.md) for full setup details.

### `local_sccache_setup.sh`

Configures sccache for local development (without GCS).

### `gradle-build-cache-init.gradle`

Gradle init script that enables the local build cache for Java integration
test builds. Injected via `--init-script` in `run_es_tests.sh`.

### `docker_build.sh` / `docker_test.sh` / `docker_check_style.sh`

Docker wrappers for Linux builds, tests, and clang-format style checking.

## Build Utilities

### `clang-format.sh` / `check-style.sh`

Run clang-format and style validation.

### `strip_binaries.sh`

Strips debug symbols from built binaries.

### `benchmark_build.sh` / `benchmark_windows_build.ps1`

Measure build times with various configurations.

### `download_windows_deps.ps1` / `build_windows_third_party_deps.ps1`

Windows-specific dependency management.

### `mlcpp-release-notes.pl`

Generates release notes from git history.

### `vault-cp.sh`

Copies secrets between Vault paths.

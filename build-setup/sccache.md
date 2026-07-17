# sccache Setup Guide

sccache is a compiler cache that dramatically speeds up C++ rebuilds by
caching compiled object files. When the same source + flags combination is
compiled again, sccache returns the cached result in milliseconds instead
of recompiling.

## Quick Start (Local Development)

### 1. Install sccache

```bash
# macOS
brew install sccache

# Linux (Debian/Ubuntu)
sudo apt install sccache

# Linux (Fedora/RHEL)
sudo dnf install sccache

# Any platform with Rust
cargo install sccache

# Windows (Scoop)
scoop install sccache
```

### 2. Build as Usual

That's it. `CMakeLists.txt` auto-detects sccache (preferred) or ccache and
sets `CMAKE_CXX_COMPILER_LAUNCHER` automatically:

```bash
cmake -B cmake-build-relwithdebinfo
cmake --build cmake-build-relwithdebinfo -j$(nproc)
```

You'll see in the CMake output:
```
-- sccache found: /opt/homebrew/bin/sccache
```

### 3. Verify It's Working

After a build:

```bash
sccache --show-stats
```

You should see compile requests and cache hits/misses. A second build of the
same code should show nearly all cache hits.

### Helper Script (Optional)

A convenience script is provided that installs sccache if missing and
configures cache settings:

```bash
source dev-tools/local_sccache_setup.sh
```

## Shared GCS Cache (Team Cache)

The CI pipeline populates a GCS-backed sccache bucket with compiled objects
for every nightly build. Developers can tap into this cache for read-only
access, meaning your first build after `git pull` gets cache hits for any
files that CI has already compiled.

### Prerequisites

- A GCP **service account** JSON key with `roles/storage.objectViewer` on the
  `elastic-ml-cpp-sccache` bucket

> **Important:** sccache requires a `service_account` type JSON key. The
> `authorized_user` credentials produced by `gcloud auth application-default
> login` are **not supported** by sccache's GCS backend. Ask your team lead
> for a service account key, or create one with:
> ```bash
> gcloud iam service-accounts keys create ~/sccache-key.json \
>   --iam-account=ml-cpp-sccache@elastic-ml.iam.gserviceaccount.com
> ```

### Setup

```bash
# Point sccache at the service account key
export SCCACHE_GCS_KEY_PATH=~/sccache-key.json

# Start sccache with GCS backend
source dev-tools/local_sccache_setup.sh --gcs
```

This configures:
- **Bucket**: `gs://elastic-ml-cpp-sccache`
- **Prefix**: auto-detected from your platform (e.g. `darwin-aarch64`)
- **Mode**: read-only (your local builds read from CI cache but don't write)

Then build normally — cache hits come from both your local cache and the
shared GCS cache.

### GCS performance characteristics

- Average GCS cache read hit: ~1.4s per file
- Average compilation (no cache): ~3.2s per file
- GCS is ~2x faster than compiling, but much slower than local disk cache
- sccache checks local cache first, then GCS — so after one build, GCS
  latency is not a factor

### How It Works

```
Your build
    |
    v
sccache server
    |
    +-- Check local disk cache (~/.cache/sccache)
    |       Hit? -> return cached .o
    |
    +-- Check GCS cache (gs://elastic-ml-cpp-sccache/<platform>/)
    |       Hit? -> return cached .o, store locally
    |
    +-- Miss -> compile, store in local cache
```

### Cache Key

Each cached entry is keyed on:
- Source file content (hash)
- Compiler flags
- Compiler version
- Preprocessed output (includes all headers)

Different branches, build types, and configurations coexist safely.

## Performance

Benchmarked on macOS aarch64 (M-series, 14 logical CPUs):

| Scenario | Build Time | Cache Hits | vs Baseline |
|---|---|---|---|
| No cache (baseline) | **2m 10s** | — | — |
| sccache cold (local, empty cache) | 2m 41s | 0/345 (0%) | +24% |
| sccache warm (local cache) | **13s** | 345/345 (100%) | **-90%** |
| GCS read-only (no local cache) | **1m 14s** | 345/345 (100%) | **-43%** |

On Linux CI agents (16 vCPU):

| Scenario | Typical Build Time |
|---|---|
| No cache (clean build) | 7–13 min |
| Cold local cache | Same as no cache (populates cache) |
| Warm local cache | 18–30 seconds |
| Warm GCS, fresh checkout | 1–3 min (downloads from GCS) |

**Key insight:** local warm cache is the primary win (90% faster). GCS provides
a useful fallback when the local cache is cold (fresh checkout, branch switch,
cleaned build dir), delivering 43% faster builds by downloading pre-compiled
objects from the CI-populated cache.

## Configuration

sccache settings can be customised via environment variables:

| Variable | Default | Description |
|---|---|---|
| `SCCACHE_DIR` | `~/.cache/sccache` | Local cache directory |
| `SCCACHE_CACHE_SIZE` | `10G` | Maximum local cache size |
| `SCCACHE_GCS_BUCKET` | `elastic-ml-cpp-sccache` | GCS bucket name |
| `SCCACHE_GCS_KEY_PREFIX` | `<os>-<arch>` | Per-platform prefix in bucket |
| `SCCACHE_GCS_RW_MODE` | `READ_ONLY` | `READ_ONLY` or `READ_WRITE` |
| `SCCACHE_GCS_KEY_PATH` | (auto-detected) | Path to GCS service account key |

## Shell integration

**Do not** add `source dev-tools/local_sccache_setup.sh` to your
`.bashrc`/`.zshrc`. The script installs packages and starts a daemon, which
is inappropriate for every new shell session.

For always-on sccache, just install it once (`brew install sccache`). CMake
auto-detects it, and the sccache server auto-starts on first compilation.
No shell configuration changes are needed.

If you want GCS always configured, add just the environment variables:

```bash
# In .bashrc / .zshrc
export SCCACHE_GCS_BUCKET="elastic-ml-cpp-sccache"
export SCCACHE_GCS_KEY_PATH="$HOME/sccache-key.json"
export SCCACHE_GCS_RW_MODE="READ_ONLY"
```

## Troubleshooting

### "sccache: error: couldn't connect to server"

The sccache server isn't running:
```bash
sccache --start-server
```

### All cache misses after compiler or flag changes

This is expected. The cache key includes compiler version and flags, so
upgrading the compiler or changing `CMAKE_BUILD_TYPE` invalidates the cache.

### GCS: "loading credential to sign http request" / metadata.google.internal error

You are using `authorized_user` credentials (from `gcloud auth
application-default login`). sccache requires a `service_account` JSON key.
See the GCS setup section above for how to obtain one.

### "Permission denied" on GCS

Your credentials don't have access to the bucket. Verify:
```bash
gsutil ls gs://elastic-ml-cpp-sccache/
```

### Cache is using too much disk

Reduce the cache size:
```bash
export SCCACHE_CACHE_SIZE=5G
sccache --stop-server
sccache --start-server
```

Or clear it entirely:
```bash
sccache --stop-server
rm -rf ~/.cache/sccache
sccache --start-server
```

### Mixing sccache and ccache

If both are installed, CMake prefers sccache. To force ccache:
```bash
cmake -B build -DCMAKE_CXX_COMPILER_LAUNCHER=$(which ccache) ...
```

## Windows

Install sccache via Scoop or Cargo (see above), then build normally.
CMake auto-detects sccache on Windows too.

For CI-specific Windows setup, a PowerShell script is also available:
```powershell
. dev-tools\setup_sccache.ps1
```

## CI Setup

For CI infrastructure setup (GCS bucket creation, Vault secrets, Buildkite
integration), see the comments in `dev-tools/setup_sccache.sh` and
`.buildkite/hooks/post-checkout`.

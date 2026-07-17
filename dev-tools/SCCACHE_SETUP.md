# sccache CI Setup Guide

This document describes the manual steps required to enable sccache with a GCS
backend for ml-cpp CI builds. Once complete, all `build_test_*` Buildkite steps
will automatically use sccache to cache compilation results across builds.

## Overview

- **sccache** is downloaded at build time (no Docker image changes needed)
- **GCS bucket** stores the cache — entries are per-platform and per-compiler-flag combination
- **Vault** provides the GCS service account credentials to Buildkite agents
- **Opt-in**: if Vault credentials are missing, builds proceed normally without caching

## Prerequisites

- Access to GCP project for creating a storage bucket and service account
- Write access to Vault at `secret/ci/elastic-ml-cpp/`
- Buildkite admin access (for verifying hook changes)

## Step 1: Create the GCS Bucket

```bash
# Choose a GCP project (e.g. elastic-ml-dev or similar)
gcloud config set project <PROJECT_ID>

# Create the bucket in a region close to CI agents
gcloud storage buckets create gs://elastic-ml-cpp-sccache \
  --location=us-central1 \
  --uniform-bucket-level-access

# Set lifecycle policy: delete objects older than 30 days to control costs
cat > /tmp/lifecycle.json << 'EOF'
{
  "rule": [
    {
      "action": {"type": "Delete"},
      "condition": {"age": 30}
    }
  ]
}
EOF
gcloud storage buckets update gs://elastic-ml-cpp-sccache \
  --lifecycle-file=/tmp/lifecycle.json
```

## Step 2: Create a GCP Service Account

```bash
# Create service account
gcloud iam service-accounts create ml-cpp-sccache \
  --display-name="ml-cpp sccache CI" \
  --description="Service account for sccache GCS backend in ml-cpp CI"

# Grant it read/write access to the bucket
gcloud storage buckets add-iam-policy-binding gs://elastic-ml-cpp-sccache \
  --member="serviceAccount:ml-cpp-sccache@<PROJECT_ID>.iam.gserviceaccount.com" \
  --role="roles/storage.objectUser"

# Create and download a JSON key
gcloud iam service-accounts keys create /tmp/ml-cpp-sccache-key.json \
  --iam-account=ml-cpp-sccache@<PROJECT_ID>.iam.gserviceaccount.com
```

## Step 3: Store the Key in Vault

```bash
# Login to Vault
export VAULT_ADDR=https://vault-ci-prod.elastic.dev
vault login -method=github token=$GITHUB_TOKEN

# Store the service account key
vault write secret/ci/elastic-ml-cpp/sccache/gcs_service_account \
  key=@/tmp/ml-cpp-sccache-key.json

# Verify
vault read secret/ci/elastic-ml-cpp/sccache/gcs_service_account

# Clean up the local key file
rm /tmp/ml-cpp-sccache-key.json
```

## Step 4: Verify the Integration

The code changes are already in place:

1. **`.buildkite/hooks/post-checkout`** reads the key from Vault for all
   `build_test_*` steps and exports `SCCACHE_GCS_BUCKET`, `SCCACHE_GCS_KEY_FILE`,
   and `GOOGLE_APPLICATION_CREDENTIALS`.

2. **`dev-tools/setup_sccache.sh`** (Linux/macOS) and **`dev-tools/setup_sccache.ps1`**
   (Windows) download sccache, configure the GCS backend, and append
   `-DCMAKE_CXX_COMPILER_LAUNCHER=sccache -DCMAKE_C_COMPILER_LAUNCHER=sccache`
   to `CMAKE_FLAGS`.

3. Build scripts (`docker_entrypoint.sh`, `build_and_test.sh`, `build_and_test.ps1`)
   source the setup script when `SCCACHE_GCS_BUCKET` is set, and print
   cache statistics at the end.

4. For aarch64 Docker builds, the GCS key is passed as a BuildKit secret
   (`--secret id=gcs_key`) so it never appears in Docker image layers.

To test, push a branch and check the build logs for:
```
sccache setup complete
sccache GCS config: bucket=elastic-ml-cpp-sccache prefix=linux-x86_64
```

The first build on each platform will be a cold run (all cache misses).
Subsequent builds will show cache hits for unchanged source files.

## Cache Structure

The GCS bucket is organised by platform prefix:
```
gs://elastic-ml-cpp-sccache/
  linux-x86_64/     # Linux x86_64 builds
  linux-aarch64/    # Linux aarch64 builds
  darwin-aarch64/   # macOS ARM builds
  windows-x86_64/   # Windows builds
```

Each entry is keyed by a hash of: source content + compiler flags + compiler
version. Different branches and build configurations coexist safely.

## Expected Impact

Based on benchmarks with ccache (which has similar caching characteristics):

| Scenario | Build Time |
|----------|-----------|
| No cache (current) | 11-13 min |
| Cold cache (first build) | 8-13 min (slight overhead from writing) |
| Warm cache (unchanged files) | 18-30 seconds |
| Typical PR (few files changed) | 1-3 min (estimated) |

## Troubleshooting

- **"sccache setup complete" not in logs**: Vault secret is missing or
  the step key doesn't match `build_test_*`
- **All cache misses on warm build**: Compiler flags changed, compiler
  version updated, or lifecycle policy deleted old entries
- **Permission denied on GCS**: Service account key expired or bucket
  IAM policy changed
- **sccache download fails**: GitHub rate limiting; consider hosting the
  binary in the build Docker images or a private artifact store

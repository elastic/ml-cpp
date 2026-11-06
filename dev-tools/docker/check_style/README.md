# ml-check-style Docker images (clang-format 5.0.1)

These images run **clang-format 5.0.1** over `ml-cpp` sources—the same version enforced by CI and
`cmake/clang-format.cmake`. Use the image that matches your **Linux container architecture** (not your host OS when
cross-building).

| Architecture | Registry image | Dockerfile directory | Publish script |
| --- | --- | --- | --- |
| x86_64 / amd64 | `docker.elastic.co/ml-dev/ml-check-style:2` | `../check_style_image/` | `../build_check_style_image.sh` |
| aarch64 / arm64 | `docker.elastic.co/ml-dev/ml-check-style-aarch64:1` | `../check_style_image_aarch64/` | `../build_check_style_image_aarch64.sh` |

Inventory and version bumps are summarized in [../README.md](../README.md).

## Day-to-day formatting

From the repository root, set `CPP_SRC_HOME` to your checkout and run:

```bash
CPP_SRC_HOME=/path/to/ml-cpp dev-tools/docker/run_docker_clang_format.sh
```

The script picks `ml-check-style:2` or `ml-check-style-aarch64:1` from `uname -m`. VS Code integration is described in
[build-setup/vscode/using_vscode.md](../../../build-setup/vscode/using_vscode.md).

## Building: x86_64 / amd64 (`ml-check-style`)

The amd64 image installs `clang-format` from Alpine packages (see `../check_style_image/Dockerfile`).

**Local test build:**

```bash
cd dev-tools/docker
docker build --no-cache -t ml-check-style:local check_style_image
```

**Publish to Elastic registry** (infrequent; requires `docker.elastic.co` login):

```bash
cd dev-tools/docker
./build_check_style_image.sh
```

When changing the clang-format version: increment the image tag in the build script and [../README.md](../README.md),
update the Dockerfile, then rebuild and push.

## Building: aarch64 / arm64 (`ml-check-style-aarch64`)

clang-format 5.0.1 has no suitable aarch64 binary for Alpine 3.8, so this image **builds clang-format from LLVM source**
(see `../check_style_image_aarch64/Dockerfile`). Expect **30–60 minutes** per build.

Always pass **`--platform linux/arm64`** when building on an x86_64 host (plain `docker build` without a platform
produces an amd64 image mis-tagged as aarch64).

**Local test build (native arm64 or explicit platform):**

```bash
cd dev-tools/docker
docker build --platform linux/arm64 --no-cache -t ml-check-style-aarch64:local check_style_image_aarch64
```

**Cross-build with buildx** (x86_64 host):

```bash
docker buildx create --name multiarch --use 2>/dev/null || docker buildx use multiarch
docker buildx inspect --bootstrap

cd dev-tools/docker
docker buildx build --platform linux/arm64 --no-cache \
  -t ml-check-style-aarch64:local \
  -f check_style_image_aarch64/Dockerfile check_style_image_aarch64 --load
```

**Publish to Elastic registry:**

```bash
cd dev-tools/docker
./build_check_style_image_aarch64.sh
```

Keep the clang-format version in step with `ml-check-style`; bump both image tags and Dockerfiles together.

## Testing an image

Replace `IMAGE` with your local tag (`ml-check-style:local` or `ml-check-style-aarch64:local`) or a registry reference.

### Verify clang-format version

```bash
docker run --rm IMAGE clang-format --version
```

Expected:

```text
clang-format version 5.0.1 (tags/RELEASE_501/final)
```

### Format a single file

```bash
cat > /tmp/test_format.cc << 'EOF'
int main(){return 0;}
EOF

docker run --rm -v /tmp:/tmp IMAGE clang-format -i /tmp/test_format.cc
cat /tmp/test_format.cc
```

### Format files changed in recent commits

Run `git` on the **host** (the aarch64 runtime image does not include git). Example for the last 20 commits:

```bash
cd /path/to/ml-cpp
IMAGE=ml-check-style-aarch64:local   # or ml-check-style:local / registry tag

git diff --name-only --diff-filter=ACMRT HEAD~20 HEAD \
  | grep -E '\.(cc|h)$' | grep -v '^3rd_party' | grep -v '^build-setup' \
  | xargs -r docker run --rm -v "$(pwd):/ml-cpp" -u "$(id -u):$(id -g)" -w /ml-cpp IMAGE \
      clang-format -i
```

Dry run (list paths only):

```bash
git diff --name-only --diff-filter=ACMRT HEAD~20 HEAD \
  | grep -E '\.(cc|h)$' | grep -v '^3rd_party' | grep -v '^build-setup'
```

To format the whole tree via CMake (same as CI-style local check), use `run_docker_clang_format.sh` above.

## Troubleshooting (aarch64 builds)

| Symptom | What to try |
| --- | --- |
| `clang-format version mismatch` during `docker build` | Rebuild with `--no-cache`; inspect LLVM build logs. |
| Permission errors on formatted files | Pass `-u $(id -u):$(id -g)` on `docker run`. |
| LLVM download failures | Confirm URLs under `https://releases.llvm.org/5.0.1/`; SHA256 pins are in the Dockerfile. |
| Very long build | Expected; only `clang-format` is built, not full LLVM. |

After a local aarch64 build:

```bash
docker images ml-check-style-aarch64:local
docker run --rm ml-check-style-aarch64:local ls -la /usr/local/bin/clang-format
```

# Building and Testing the clang-format 5.0.1 Docker Container for aarch64

This directory contains the Dockerfile and build script for creating a Docker container with clang-format 5.0.1 for Linux aarch64/ARM64 architecture.

## Prerequisites

- Docker installed and running
- For aarch64 builds: Docker running on an aarch64 host, or Docker buildx configured for cross-platform builds
- Git repository with at least 20 commits (for testing the format task)

## Building the Container

### Option 1: Build Locally (Recommended for Testing)

If you're on an aarch64 machine or have buildx configured:

```bash
cd dev-tools/docker
docker build --no-cache -t ml-check-style-aarch64:local check_style_image_aarch64
```

This will:
- Build the image with tag `ml-check-style-aarch64:local`
- Take approximately 30-60 minutes (building LLVM from source)
- Create an image with clang-format 5.0.1 installed

### Option 2: Using the Build Script

The build script is designed for pushing to the Elastic registry:

```bash
cd dev-tools/docker
./build_check_style_image_aarch64.sh
```

**Note**: This requires authentication to `docker.elastic.co`. For local testing, use Option 1.

### Option 3: Cross-Platform Build (if not on aarch64)

If you're not on an aarch64 machine, you can use Docker buildx:

```bash
# Enable buildx (if not already enabled)
docker buildx create --name multiarch --use
docker buildx inspect --bootstrap

# Build for aarch64
cd dev-tools/docker
docker buildx build --platform linux/arm64 --no-cache -t ml-check-style-aarch64:local -f check_style_image_aarch64/Dockerfile check_style_image_aarch64 --load
```

## Testing the Container

### 1. Verify clang-format Version

First, verify that clang-format 5.0.1 is correctly installed:

```bash
docker run --rm ml-check-style-aarch64:local clang-format --version
```

Expected output:
```
clang-format version 5.0.1 (tags/RELEASE_501/final)
```

### 2. Test Formatting a Single File

Test that clang-format works on a sample file:

```bash
# Create a test file with formatting issues
cat > /tmp/test_format.cc << 'EOF'
int main(){return 0;}
EOF

# Format it
docker run --rm -v /tmp:/tmp ml-check-style-aarch64:local clang-format -i /tmp/test_format.cc

# Check the result
cat /tmp/test_format.cc
```

### 3. Test Formatting Files from Recent Commits

Test the complete workflow that will be used in the VS Code task:

```bash
# From the repository root
cd /path/to/ml-cpp

# Get files changed in last 20 commits and format them
docker run --rm \
  -v $(pwd):/ml-cpp \
  -u $(id -u):$(id -g) \
  ml-check-style-aarch64:local \
  bash -c 'cd /ml-cpp && git diff --name-only --diff-filter=ACMRT HEAD~20 HEAD | grep -E "\.(cc|h)$" | grep -v "^3rd_party" | grep -v "^build-setup" | xargs -r clang-format -i'

# Check git status to see what was changed
git status
```

### 4. Test with a Dry Run (Preview Changes)

To see what files would be formatted without actually changing them:

```bash
cd /path/to/ml-cpp

# Get list of files that would be formatted
docker run --rm \
  -v $(pwd):/ml-cpp \
  ml-check-style-aarch64:local \
  bash -c 'cd /ml-cpp && git diff --name-only --diff-filter=ACMRT HEAD~20 HEAD | grep -E "\.(cc|h)$" | grep -v "^3rd_party" | grep -v "^build-setup"'
```

### 5. Test the VS Code Task Command

You can test the exact command that will be used in the VS Code task:

```bash
cd /path/to/ml-cpp

docker run --rm \
  -v $(pwd):/ml-cpp \
  -u $(id -u):$(id -g) \
  ml-check-style-aarch64:local \
  bash -c 'cd /ml-cpp && git diff --name-only --diff-filter=ACMRT HEAD~20 HEAD | grep -E "\.(cc|h)$" | grep -v "^3rd_party" | grep -v "^build-setup" | xargs -r clang-format -i'
```

## Using in VS Code

### Setting Up the Task

1. Create or edit `.vscode/tasks.json` in your workspace root:

```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Format changed files (last 20 commits)",
      "type": "shell",
      "command": "docker run --rm -v ${workspaceFolder}:/ml-cpp -u $(id -u):$(id -g) ml-check-style-aarch64:local bash -c 'cd /ml-cpp && git diff --name-only --diff-filter=ACMRT HEAD~20 HEAD | grep -E \"\\.(cc|h)$\" | grep -v \"^3rd_party\" | grep -v \"^build-setup\" | xargs -r clang-format -i'",
      "problemMatcher": [],
      "presentation": {
        "reveal": "always",
        "panel": "shared"
      }
    }
  ]
}
```

**Note**: Change `ml-check-style-aarch64:local` to `docker.elastic.co/ml-dev/ml-check-style-aarch64:1` if using the official registry image.

### Running the Task

1. Open Command Palette (`Cmd+Shift+P` on macOS, `Ctrl+Shift+P` on Linux/Windows)
2. Type "Tasks: Run Task"
3. Select "Format changed files (last 20 commits)"
4. The task will format all `.cc` and `.h` files changed in the last 20 commits

## Troubleshooting

### Issue: "clang-format version mismatch"
- **Solution**: The build may have failed. Check the Docker build logs for errors.
- Rebuild the container with `--no-cache` flag

### Issue: "Permission denied" when formatting files
- **Solution**: Ensure you're using `-u $(id -u):$(id -g)` in the docker run command
- This ensures files are created with your user ID

### Issue: "No files to format"
- **Solution**: Check that you have at least 20 commits in your repository
- Try reducing the number: change `HEAD~20` to `HEAD~10` or `HEAD~5`

### Issue: Build fails with "No such file or directory" for LLVM
- **Solution**: The LLVM release URLs might have changed. Check `https://releases.llvm.org/5.0.1/` for correct filenames

### Issue: Build takes too long
- **Solution**: This is expected - building LLVM from source takes 30-60 minutes
- The build only compiles clang-format (not full LLVM) to minimize time
- Consider using a pre-built image from the registry if available

## Verifying the Build

After building, you can inspect the image:

```bash
# Check image size
docker images ml-check-style-aarch64:local

# Inspect the image
docker inspect ml-check-style-aarch64:local

# Check what's installed
docker run --rm ml-check-style-aarch64:local ls -la /usr/local/bin/clang-format
```

## Next Steps

Once the container is built and tested:
1. Test formatting on a few files manually
2. Add the VS Code task to your workspace
3. Run the task to format files from recent commits
4. Review the changes with `git diff` before committing


# Elasticsearch Machine Learning C++

## Toolchain
- **Language**: C++20 (`CMAKE_CXX_STANDARD 20`).
- **Build system**: CMake (primary) or Gradle wrapper (`./gradlew`).
- **Compilers**: GCC 13.3 on Linux, Xcode Clang on macOS, MSVC on Windows.
- **Key dependencies**: Boost 1.86.0 (dynamic linking), PyTorch (libtorch), RapidJSON, Eigen, libxml2.
- **Platforms**: Linux x86_64/aarch64, macOS aarch64, Windows x86_64.
- **Toolchain files**: Auto-selected from `cmake/<os>-<arch>.cmake` or set via `CMAKE_TOOLCHAIN_FILE`.

## Build & Run Commands

Configure and build (default `RelWithDebInfo`):
```
cmake -B cmake-build-relwithdebinfo
cmake --build cmake-build-relwithdebinfo -j$(nproc)
```

Or via Gradle:
```
./gradlew :compile
```

Set `ML_DEBUG=1` to switch to a Debug build. Compiler caching (sccache/ccache) is auto-detected.

Refer to `CONTRIBUTING.md` and the `build-setup/` directory for full platform-specific setup instructions.

## Project Structure

```
bin/                    # Application executables
  autodetect/           #   Anomaly detection
  categorize/           #   Log categorization
  controller/           #   Process lifecycle controller
  data_frame_analyzer/  #   Data frame analytics (classification, regression)
  normalize/            #   Anomaly score normalization
  pytorch_inference/    #   PyTorch model inference
lib/                    # Shared libraries
  api/                  #   JSON/REST API layer
  core/                 #   Platform abstractions, I/O, logging, compression
  maths/                #   Mathematical and statistical algorithms
    analytics/          #     Boosted tree, data frame analytics
    common/             #     Bayesian optimisation, distributions, time series
    time_series/        #     Time series decomposition, forecasting
  model/                #   Anomaly detection models
  seccomp/              #   Seccomp/sandbox filters
  test/                 #   Shared test utilities (CBoostTestXmlOutput, etc.)
  ver/                  #   Version information
include/                # Public headers (mirrors lib/ structure)
3rd_party/              # Vendored third-party code (Eigen, etc.)
cmake/                  # CMake toolchain files, helper functions, test runners
build-setup/            # Platform-specific build environment instructions
.buildkite/             # CI pipeline definitions (Buildkite)
.github/workflows/      # GitHub Actions (automatic backport)
dev-tools/              # Developer scripts (clang-format, benchmarks)
```

Libraries must not have circular dependencies. The dependency order is roughly:
`core` -> `maths` -> `model` -> `api` -> `bin/*`.

## Testing

Tests use the **Boost.Test** framework. Each library and application has a `unittest/` subdirectory containing test files and a `Main.cc` entry point.

### Running Tests

Run all tests:
```
cmake --build cmake-build-relwithdebinfo -t test
```

Run tests for a specific library:
```
cmake --build cmake-build-relwithdebinfo -t test_core
cmake --build cmake-build-relwithdebinfo -t test_model
```

Run specific test cases (wildcards supported):
```
TESTS="*/testPersist" cmake --build cmake-build-relwithdebinfo -t test_model
```

Run tests individually in separate processes (better isolation, enables parallelism):
```
cmake --build cmake-build-relwithdebinfo -j8 -t test_individually
cmake --build cmake-build-relwithdebinfo -j8 -t test_api_individually
```

Pass extra flags to the Boost.Test runner:
```
TEST_FLAGS="--random" cmake --build cmake-build-relwithdebinfo -t test
```

### Precommit (format + test)

```
cmake --build cmake-build-relwithdebinfo -j8 -t precommit
```
Or: `./gradlew precommit`

### Writing Tests

- Test files are named `CClassNameTest.cc` and placed in `lib/<module>/unittest/` or `bin/<module>/unittest/`.
- Each test file uses `BOOST_AUTO_TEST_SUITE(CClassNameTest)` / `BOOST_AUTO_TEST_CASE(testMethodName)`.
- Use real classes over mocks wherever possible. Tests should reflect real-world usage.
- Every class should have a corresponding test suite; every public method should have a test.
- Tests must not modify shared resources or leave side effects that affect other tests.

## Formatting & Style

Code is formatted with `clang-format` (LLVM-based style, 4-space indent). Run before committing:
```
cmake --build cmake-build-relwithdebinfo -t format
```
Or: `./gradlew format`

The CI pipeline enforces formatting via the `check-style` step; PRs that fail formatting will not pass CI.

The full coding standard is in `STYLEGUIDE.md`. Key points:

### Naming Conventions
- Classes: `CClassName`, Structs: `SStructName`, Enums: `EEnumName`
- Member variables: `m_ClassMember`, `s_StructMember`
- Static members: `ms_ClassStatic`
- Methods: `methodName` (camelCase)
- Type aliases: `TTypeName` (e.g. `using TDoubleVec = std::vector<double>`)
- Constants: `CONSTANT_NAME`
- Non-boolean accessors: `clientId` (not `getClientId`)
- Boolean accessors: `isComplete` (not `complete`)
- Files: `CClassName.cc` / `CClassName.h`

### Code Conventions
- Use `nullptr`, never `0` or `NULL`.
- No exceptions — use return codes for error handling. Catch third-party exceptions at the smallest scope.
- No `assert()`. No C-style casts. No macros unless unavoidable.
- Prefer smart pointers over raw pointers; prefer references over pointers.
- Scope member function calls with `this->`.
- Use `auto` when the type is obvious; avoid it when the type is unclear.
- Prefer `emplace_back` over `push_back`, range-based for loops, and uniform initializers.
- `override` must be used consistently; `virtual` must not appear alongside `override`.

### File Layout
- Implementation files (`.cc`): own header first, then other ML headers, third-party headers, standard library headers.
- Group includes by library with blank lines between groups (clang-format will sort within groups).
- Use unnamed namespaces in `.cc` files for file-local helpers, not private class members.
- Forward-declare classes in headers rather than including their headers.

### Documentation
- Doxygen comments (exclamation mark style: `//!`) are required for all header files and public/protected methods.
- Implementation files use regular C++ comments, not Doxygen.
- Focus comments on the "why", not the "what".

## License Headers

All source files must include the Elastic License 2.0 header. Copy from `copyright_code_header.txt` or any existing source file.

## CI

CI runs on **Buildkite** (`ml-cpp-pr-builds`). The pipeline builds and tests on all platforms (Linux x86_64, Linux aarch64, macOS aarch64, Windows x86_64) in both `RelWithDebInfo` and Debug configurations. It also runs:
- `clang-format` style validation
- Snyk security/license scanning
- Java integration tests against Elasticsearch

Automatic backporting is handled by a GitHub Action (`.github/workflows/backport.yml`) — add version labels (e.g. `v9.3.0`) to a PR and a backport PR is created automatically when it merges.

## Pull Requests

- Title must be prefixed with `[ML]` (e.g. `[ML] Fix anomaly scoring edge case`).
- Label with `:ml` (mandatory), a type label (`>bug`, `>enhancement`, `>feature`, `>refactoring`, `>test`, `>docs`), and version labels for applicable releases.
- Squash-and-merge is the standard merge strategy; keep commits clean for review but don't squash manually.
- Backports start after merging to `main`. Add version labels to trigger automatic backport PRs.

## Best Practices for Automation Agents

- Always read existing code before editing to understand patterns and conventions.
- Never edit unrelated files; keep diffs tightly scoped.
- Run `clang-format` before presenting any code changes.
- Match the naming conventions exactly — the prefixes (`C`, `m_`, `ms_`, `T`, `E`) are strictly followed throughout the codebase.
- When adding new classes, follow the existing directory and namespace structure. Production code in `lib/foo/` uses namespace `ml::foo`.
- When adding tests, place them in the corresponding `unittest/` directory and register them in the `CMakeLists.txt`.
- Do not introduce new third-party dependencies without discussion.
- Do not add AI attribution trailers (e.g. `Co-Authored-By`) to commit messages.
- Commit messages should follow the `[ML] Summary of change` format.
- If unsure about a convention, check a nearby file for the established pattern — consistency with surrounding code is the highest priority.

Stay aligned with `CONTRIBUTING.md`, `STYLEGUIDE.md`, and the `build-setup/` guides; this AGENTS file summarizes but does not replace those authoritative documents.

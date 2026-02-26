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

## CMake Helper Functions

The build uses custom CMake functions defined in `cmake/functions.cmake`. Use these instead of raw `add_library`/`add_executable` — they handle platform-specific sources, linking, installation, and Windows resource generation automatically.

### Adding a Shared Library

Set `ML_LINK_LIBRARIES` then call `ml_add_library`:

```cmake
project("ML MyLib")

set(ML_LINK_LIBRARIES
  ${Boost_LIBRARIES}
  MlCore
  )

ml_add_library(MlMyLib SHARED
  CMyClass.cc
  CMyOtherClass.cc
  )
```

Libraries are named with the `Ml` prefix (e.g. `MlCore`, `MlModel`). The function handles shared library versioning, RPATH, and installation. Use `SHARED` for distributed libraries or `STATIC` for internal-only ones.

For libraries that should not be installed/distributed (e.g. internal helpers), use `ml_add_non_distributed_library` instead.

### Adding an Executable

Set `ML_LINK_LIBRARIES` then call `ml_add_executable`. A `Main.cc` file is included automatically — do not list it in the sources:

```cmake
project("ML MyApp")

set(ML_LINK_LIBRARIES
  ${Boost_LIBRARIES}
  MlCore
  MlApi
  MlVer
  )

ml_add_executable(myapp
  CCmdLineParser.cc
  )
```

The function creates a companion OBJECT library (`MlMyApp`) from the listed sources, which test executables can link against. The executable itself always builds from `Main.cc` plus those objects.

For executables not intended for distribution (dev tools, benchmarks), use `ml_add_non_distributed_executable`.

### Adding a Test Executable

Test executables live in `unittest/` subdirectories. Set `ML_LINK_LIBRARIES` (including `${Boost_LIBRARIES_WITH_UNIT_TEST}` and `MlTest`), then call `ml_add_test_executable`:

```cmake
project("ML MyLib unit tests")

set(SRCS
  CMyClassTest.cc
  CMyOtherClassTest.cc
  Main.cc
  )

set(ML_LINK_LIBRARIES
  ${Boost_LIBRARIES_WITH_UNIT_TEST}
  MlCore
  MlMyLib
  MlTest
  )

ml_add_test_executable(mylib ${SRCS})
```

The `_target` argument (e.g. `mylib`) is used to derive the test executable name (`ml_test_mylib`) and the CMake targets `test_mylib` and `test_mylib_individually`.

### Registering Tests with the Build

After creating the test executable, register it in `test/CMakeLists.txt` by adding an `ml_add_test` call alongside the existing entries:

```cmake
ml_add_test(lib/core/unittest core)
ml_add_test(lib/maths/common/unittest maths_common)
ml_add_test(lib/maths/time_series/unittest maths_time_series)
ml_add_test(lib/maths/analytics/unittest maths_analytics)
ml_add_test(lib/model/unittest model)
ml_add_test(lib/api/unittest api)
ml_add_test(lib/ver/unittest ver)
ml_add_test(lib/seccomp/unittest seccomp)
ml_add_test(bin/controller/unittest controller)
ml_add_test(bin/pytorch_inference/unittest pytorch_inference)
ml_add_test(lib/mylib/unittest mylib)          # <-- new entry
```

The first argument is the relative path to the unittest directory; the second is the target name matching `ml_add_test_executable`. Note how nested libraries use underscores in the target name (e.g. `lib/maths/common/unittest` -> `maths_common`).

### Platform-Specific Sources

If a source file has a platform-specific variant (e.g. `CMyClass_Linux.cc`, `CMyClass_Darwin.cc`), the `ml_generate_platform_sources` function (called internally by all `ml_add_*` functions) will automatically substitute the platform-specific file at build time. Just list the base filename (`CMyClass.cc`) in your sources.

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

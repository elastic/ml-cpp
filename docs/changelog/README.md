# Changelog entries

Each pull request that changes user-visible behaviour should include a changelog
entry as a YAML file in this directory, named `<PR_NUMBER>.yaml`.

## Schema

The canonical schema is the
[Elasticsearch changelog schema](https://github.com/elastic/elasticsearch/blob/main/build-tools-internal/src/main/resources/changelog-schema.json).
The validator (`dev-tools/validate_changelogs.py`) fetches this schema from
the ES repo at validation time to ensure ml-cpp entries are always compatible
with the ES release notes pipeline. If the fetch fails (e.g. offline
development), it falls back to the local copy in `changelog-schema.json`.

If the local copy has diverged from the ES source, the validator will print a
warning. To update the local copy, download the latest version from the URL
above.

## Format

```yaml
pr: 2914
summary: Split build and test into separate pipeline steps
area: Machine Learning
type: enhancement
issues: []
```

### Required fields

| Field     | Description |
|-----------|-------------|
| `type`    | The type of change (see below). Always required. |
| `summary` | A concise, user-facing description of the change. Always required. |
| `pr`      | The pull request number (integer). Required unless type is `known-issue` or `security`. |
| `area`    | The area of the codebase affected (see below). Required unless type is `known-issue` or `security`. |

### Optional fields

| Field         | Description |
|---------------|-------------|
| `issues`      | List of related GitHub issue numbers (integers). Default: `[]` |
| `highlight`   | Release highlight object (see below). |
| `breaking`    | Breaking change details. **Required** when type is `breaking` or `breaking-java`. |
| `deprecation` | Deprecation details. **Required** when type is `deprecation`. |
| `source_repo` | GitHub repository (`owner/name`) for entries from external repos, e.g. `elastic/ml-cpp`. Used by the ES release notes pipeline to generate correct PR links. Injected automatically by the export script; not normally set manually. |

### Valid areas

Most ml-cpp entries should use **Machine Learning**. Other valid areas from the
ES schema (e.g. **Inference**) may be used when appropriate. The full list of
valid areas is defined in the
[ES changelog schema](https://github.com/elastic/elasticsearch/blob/main/build-tools-internal/src/main/resources/changelog-schema.json).

### Valid types

| Type | Description |
|------|-------------|
| `breaking` | A change that breaks backwards compatibility (requires `breaking` object) |
| `breaking-java` | A breaking change to the Java API (requires `breaking` object) |
| `bug` | A fix for an existing defect |
| `deprecation` | Deprecation of existing functionality (requires `deprecation` object) |
| `enhancement` | An improvement to existing functionality |
| `feature` | A wholly new feature |
| `known-issue` | A known issue (`pr` and `area` not required) |
| `new-aggregation` | A new aggregation type |
| `regression` | A fix for a recently introduced defect |
| `security` | A security fix (`pr` and `area` not required) |
| `upgrade` | An upgrade-related change |

### Highlight object

For changes worthy of a release highlight:

```yaml
highlight:
  notable: true
  title: "Short title for the highlight"
  body: "Longer description in AsciiDoc format (no triple-backtick code blocks)."
```

### Breaking / Deprecation object

Required when `type` is `breaking`, `breaking-java`, or `deprecation`:

```yaml
breaking:
  area: Machine Learning
  title: "Short title describing the breaking change"
  details: "Detailed description of what changed (AsciiDoc, no triple-backticks)."
  impact: "What users need to do to adapt."
  notable: true
```

Valid areas for breaking/deprecation changes are a subset of the main areas,
defined in the ES changelog schema under `compatibilityChangeArea`.

## When is a changelog entry required?

A changelog entry is **required** for any PR that:
- Fixes a bug
- Adds or changes user-visible functionality
- Changes the API or data formats
- Deprecates or removes functionality

A changelog entry is **not required** for:
- Pure refactoring with no behaviour change
- Test-only changes
- CI/build infrastructure changes (unless they affect the shipped artefact)
- Documentation-only changes

PRs that do not require a changelog entry should be labelled with
`>test`, `>refactoring`, `>docs`, `>build`, or `>non-issue` to skip validation.

## Developing changelog tooling

From the repository root, run the Python unit tests for shared validation and the
bundle formatter:

```bash
python3 -m unittest discover -s dev-tools/unittest -p 'test_*.py'
```

### Future improvements

- Run the changelog tooling `unittest` suite in CI (Gradle and/or Buildkite) so
  Python regressions are caught automatically; track wiring work as a separate
  issue.

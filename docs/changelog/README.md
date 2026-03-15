# Changelog entries

Each pull request that changes user-visible behaviour should include a changelog
entry as a YAML file in this directory, named `<PR_NUMBER>.yaml`.

## Format

```yaml
pr: 2914
summary: Split build and test into separate pipeline steps
area: Build
type: enhancement
issues: []
```

### Required fields

| Field     | Description |
|-----------|-------------|
| `pr`      | The pull request number (integer). |
| `summary` | A concise, user-facing description of the change. |
| `area`    | The area of the codebase affected (see below). |
| `type`    | The type of change (see below). |

### Optional fields

| Field    | Description |
|----------|-------------|
| `issues` | List of related GitHub issue numbers (integers). Default: `[]` |

### Valid areas

- **Anomaly Detection** – anomaly detection jobs, modelling, and results
- **Data Frame Analytics** – classification, regression, and outlier detection
- **NLP** – natural language processing and PyTorch inference
- **Core** – core libraries, platform support, and utilities
- **API** – REST API layer and state persistence
- **Build** – build system, CI, packaging, and developer tooling
- **Inference** – inference service integration

### Valid types

- **breaking** – a change that breaks backwards compatibility
- **bug** – a fix for an existing defect
- **deprecation** – deprecation of existing functionality
- **enhancement** – an improvement to existing functionality
- **feature** – a wholly new feature
- **regression** – a fix for a recently introduced defect

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

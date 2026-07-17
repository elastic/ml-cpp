# AI Context Rules for ml-cpp

This directory contains domain-specific knowledge files that help AI coding assistants (Cursor, Claude, Copilot, etc.) understand the ml-cpp codebase, build system, and CI infrastructure.

## File Format

The `.mdc` files use [Cursor Rules](https://docs.cursor.com/context/rules-for-ai) format:

```
---
description: Brief description of the knowledge area
globs: file patterns where this knowledge applies
---

# Markdown content
...
```

The YAML frontmatter (`description`, `globs`) is Cursor-specific — it tells Cursor when to automatically include the file as context. The body is standard Markdown and is readable by any tool.

## Files

| File | Scope | Description |
|---|---|---|
| `ml-cpp-build-system.mdc` | CMake, Gradle, Docker | Build configuration, acceleration options, test parallelism |
| `ml-cpp-buildkite-ci.mdc` | `.buildkite/**` | Pipeline structure, API access, Vault secrets, known failures |
| `ml-cpp-coding-conventions.mdc` | `*.cc`, `*.h` | Naming, cross-platform gotchas, RAII patterns |
| `ml-cpp-ci-analytics.mdc` | `dev-tools/*.py` | Elasticsearch integration, anomaly detection, AI analysis |

## Usage with Different AI Tools

### Cursor
These files are automatically loaded based on the `globs` pattern in the frontmatter. No additional setup needed — just open the project in Cursor.

### Claude Code (claude CLI)
Claude Code reads `CLAUDE.md` from the repository root. A consolidated version is provided at the repo root that references these files:
```bash
claude  # Will automatically pick up CLAUDE.md
```

### Claude Projects / API
Upload the `.mdc` files as project knowledge documents. The YAML frontmatter is ignored but harmless — Claude will read the Markdown content.

### Other AI Tools (Copilot, Windsurf, Aider, etc.)
Point the tool at these files as context. The Markdown content is universally readable. Strip the YAML frontmatter if the tool doesn't handle it:
```bash
# Extract just the markdown content
sed '1,/^---$/d' ml-cpp-build-system.mdc | sed '1,/^$/d'
```

## Maintenance

These files should be updated when:
- Build system configuration changes (new CMake options, Gradle tasks)
- CI pipeline structure changes (new steps, different agents)
- New Vault secrets are added
- Coding conventions evolve
- New known failure patterns are identified

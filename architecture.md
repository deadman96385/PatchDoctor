# architecture.md – Technical Design & Guidance

This document provides a technical overview of **PatchDoctor** for others that may extend, refactor, or maintain the project.

---

## Core Purpose

PatchDoctor is not just a patch parser: it is a **verification and diagnostic system**. Its goal is to bridge the gap between patch files and the actual state of a Git repository, with robust handling of edge cases and developer-friendly reporting.

---

## High-Level Architecture

1. **PatchParser**
   - Reads patch files with encoding detection (`chardet`).
   - Extracts headers (author, subject, date, commit hash).
   - Detects file operations: create, delete, modify, rename, binary.
   - Builds structured objects (`PatchInfo`, `FileOperation`, `DiffHunk`).

2. **RepositoryScanner**
   - Interfaces with the local repository.
   - Retrieves file existence, git status, and commit history.
   - Reads file contents safely with encoding detection.
   - Caches results with `cachetools.LRUCache` to reduce disk I/O.

3. **PatchVerifier**
   - For each file operation, verifies if repository matches patch expectations.
   - Uses fuzzy diff hunk matching (`difflib.SequenceMatcher`).
   - Categorizes results as `OK`, `MISSING`, `MODIFIED`, `ERROR`, etc.
   - Generates actionable `FixSuggestion`s, ranging from `git restore` to manual edit instructions.

4. **ReportGenerator**
   - Produces human-readable console reports (using `rich`).
   - Supports plain-text and JSON outputs for automation.
   - Summarizes status at both patch and file levels.
   - Displays progress bars, truncates long paths, and provides adaptive formatting.

5. **GitRunner**
   - Safe wrapper for executing git commands.
   - Provides caching, timeout handling, and failure resilience.
   - Returns structured `GitResult` objects with stdout, stderr, and return codes.

---

## Key Data Structures

- `PatchInfo`: Metadata about a patch (commit hash, author, subject, files changed).
- `FileOperation`: Represents an action (create, delete, modify, rename, binary).
- `DiffHunk`: Captures old/new line ranges and actual diff content.
- `VerificationResult`: Outcome of applying a patch to a repo.
- `FixSuggestion`: Suggested automated or manual remediation steps.

---

## Error Handling Philosophy

- **Never silently fail**: errors are always surfaced.
- **Graceful fallbacks**: if encoding detection fails, default to UTF-8.
- **Actionable diagnostics**: when a hunk is missing, PatchDoctor provides git commands, mini-patches, or manual edit instructions.

---

## Extension Guidance

When extending PatchDoctor:

1. **Add new verification logic**  
   Example: semantic-aware verification for config files or JSON structures.

2. **Enhance reporting**  
   Extend `ReportGenerator` to output in additional formats (HTML, Markdown).

3. **CI/CD integration**  
   Add new flags for stricter exit codes or pipeline-specific checks.

4. **Performance improvements**  
   Tune caching layers or allow parallel verification of multiple patches.

5. **Testing**  
   Tests are located in `test_patchdoctor.py` and cover:
   - Path truncation and encoding detection
   - Basic patch parsing
   - Binary patch handling
   - Verifier integration
   - JSON report structure

---

## Security Considerations

- Git commands are executed locally with subprocess; arguments are validated.
- No network access is performed.
- Caution: applying suggestions such as `git apply` or `git restore` will modify the repo state—users must review commands before running.

---

## Development Practices

- Code style: enforced with `black`, `ruff`, and `pylint` (see `pyproject.toml`).
- Tests: run with `pytest`, strict markers enabled.
- Python: requires `>=3.10`.

---

## Developer Quick-Start Workflow (uv)

### 1. Prerequisites
- **Python** ≥ 3.10  
- **uv** package manager ([uv installation guide](https://github.com/astral-sh/uv))  
- **git** installed

---

### 2. Clone the Repository
```bash
git clone https://github.com/your-org/patchdoctor.git
cd patchdoctor
```

---

### 3. Environment Setup with uv
```bash
uv venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
```

---

### 4. Install Dependencies
```bash
# Install runtime + dev dependencies defined in pyproject.toml
uv pip install -e ".[dev]"
```

This will install:
- **rich**, **chardet**, **cachetools** (runtime)  
- **black**, **ruff**, **mypy**, **pylint**, **pytest** (dev)

---

### 5. Run Linting & Static Analysis
```bash
uv run black .
uv run ruff check .
uv run pylint patchdoctor
uv run mypy patchdoctor
```

---

### 6. Run Tests
```bash
uv run pytest
```

---

### 7. Try PatchDoctor
```bash
# Analyze patches in the current directory
uv run patchdoctor -v

# Generate JSON report for CI/CD
uv run patchdoctor -j report.json
```

---

### 8. Contribution Workflow
- Follow linting and style checks before commit.  
- Ensure all tests pass locally.  
- Extend tests when adding new features.  

---
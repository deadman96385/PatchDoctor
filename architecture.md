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

## LLM Integration

PatchDoctor supports integration with LLM-based coding assistants for automated patch application and validation workflows.

### Opencode Integration

Opencode can use PatchDoctor via direct import or command execution:

1. **Direct Import**: Import `patchdoctor` and call `run_validation()` or `validate_from_content()` for programmatic validation with full control over options.
2. **Command Execution**: Use bash tools to run `uv run python -m patchdoctor.py -j report.json` and parse the output.
3. **Workflow**: Provide a large patch to the LLM, instruct it to split into small patches, apply each sequentially, then validate with `patchdoctor.run_validation()` to ensure no changes are omitted or skipped. Iterate if discrepancies are found.

### Claude Code Integration

Claude Code can similarly execute PatchDoctor commands:

1. **Command Line**: Run `uv run python -m patchdoctor.py -j report.json` directly in the terminal.
2. **Direct Import**: Import `patchdoctor` for Python-based workflows with `run_validation()` or `validate_from_content()`.
3. **Validation Loop**: In iterative coding sessions, use PatchDoctor functions to verify patch application completeness before proceeding, enabling robust validation in AI-assisted development.

Both assistants can leverage PatchDoctor's JSON reports for structured feedback on patch status.

---

## AI Agent Integration Features

PatchDoctor v2.0 includes comprehensive enhancements specifically designed for AI coding assistants (Claude Code, Cursor, GitHub Copilot, etc.), enabling sophisticated automated workflows while maintaining full CLI backward compatibility.

### Enhanced Error Information System

**Purpose**: Provide structured error information that AI agents can programmatically handle and implement recovery strategies.

**Key Components**:
- `ErrorInfo` dataclass with error codes, recovery suggestions, and context
- Common error codes: `NO_PATCHES_FOUND`, `GIT_TIMEOUT`, `PARSE_ERROR`, etc.
- Enhanced `run_validation()` and `validate_from_content()` with structured returns

**AI Agent Benefits**:
```python
result = run_validation("patch.patch")
if not result["success"]:
    error_info = result["error_info"]
    if error_info["code"] == "GIT_TIMEOUT":
        # Retry with increased timeout
        result = run_validation("patch.patch", timeout=120)
    elif error_info["recoverable"]:
        # Apply suggested recovery action
        print(f"Suggestion: {error_info['suggestion']}")
```

### Safe Patch Application Assistant

**Purpose**: Enable AI agents to automatically apply "safe" fix suggestions with appropriate safeguards and rollback capabilities.

**Key Functions**:
- `apply_safe_fixes()`: Automated fix application with safety controls
- Safety level filtering: `safe`, `review`, `dangerous`
- Dry-run mode for preview without changes
- Built-in rollback mechanisms using GitRunner

**AI Agent Benefits**:
```python
result = run_validation("patch.patch")
if result["success"]:
    fix_result = apply_safe_fixes(
        result["verification_result"],
        confirm=False,
        safety_levels=["safe"],
        dry_run=False
    )
    print(f"Applied: {len(fix_result['applied'])} fixes")
    # Rollback available via fix_result["rollback_info"]
```

### Patch Content Analysis Utilities

**Purpose**: Provide AI agents with structured data for decision-making and patch manipulation.

**Key Functions**:
- `extract_missing_changes()`: Detailed information about missing changes
- `generate_corrective_patch()`: Create targeted patches from missing hunks
- `split_large_patch()`: Break down complex patches into manageable pieces
- `summarize_patch_status()`: Quick status analysis for decision-making

**AI Agent Benefits**:
```python
result = run_validation("large_patch.patch")
if result["verification_result"].overall_status == "PARTIALLY_APPLIED":
    # Extract what's missing
    missing = extract_missing_changes(result["verification_result"])

    # Generate corrective patch for missing parts
    generate_corrective_patch(result["verification_result"], "missing.patch")

    # Or split original patch into smaller pieces
    smaller_patches = split_large_patch(patch_content, strategy="by_file")
```

### Incremental Processing for Large Patches

**Purpose**: Support processing large patch sets with progress monitoring and early termination.

**Key Features**:
- `validate_incremental()`: Process patches with progress callbacks
- Optional parallel processing (configurable max_concurrent)
- Early termination on critical errors
- Real-time progress reporting

**AI Agent Benefits**:
```python
def progress_callback(patch_file: str, result: VerificationResult):
    print(f"Processed: {patch_file} - Status: {result.overall_status}")

result = validate_incremental(
    patch_dir="./patches",
    progress_callback=progress_callback,
    early_stop_on_error=True,
    max_concurrent=4  # Parallel processing
)
```

### Batch Patch Processing Utilities

**Purpose**: Process multiple patches with dependency tracking and rollback capabilities.

**Key Functions**:
- `validate_patch_sequence()`: Ordered patch processing with dependency analysis
- `create_patch_application_plan()`: Generate step-by-step application plans
- Conflict detection and resolution suggestions
- Checkpoint/restore functionality for large batch operations

**AI Agent Benefits**:
```python
patch_files = ["001-setup.patch", "002-feature.patch", "003-tests.patch"]

# Create application plan
plan = create_patch_application_plan(patch_files)
print(f"Application order: {plan['application_order']}")
print(f"Dependencies: {plan['dependencies']}")

# Execute with rollback on failure
result = validate_patch_sequence(
    patch_files,
    dependency_order=True,
    rollback_on_failure=True
)
```

### Workflow Template System

**Purpose**: Provide reusable workflow templates for common AI agent patterns.

**Available Templates**:
- `IterativePatchWorkflow`: Step-by-step patch application with error recovery
- `BatchProcessingWorkflow`: Multi-patch operations with dependency analysis
- `SafeFixApplicationWorkflow`: Automated fix handling with safety checks

**AI Agent Benefits**:
```python
# Use pre-built workflow templates
workflow = IterativePatchWorkflow(config=Config.lenient_mode())
result = workflow.execute(large_patch_content)

# Built-in error recovery and progress monitoring
if result["success"]:
    print(f"Applied {result['steps_completed']} steps successfully")
else:
    print(f"Recovery suggestion: {result['recovery_action']}")
```

### Configuration Profiles for AI Workflows

**Purpose**: Provide predefined configuration profiles optimized for different AI agent scenarios.

**Available Profiles**:
- `Config.strict_mode()`: High-precision validation for critical changes
- `Config.lenient_mode()`: Flexible fuzzy matching for code drift scenarios
- `Config.fast_mode()`: Speed-optimized processing for quick validation

**AI Agent Benefits**:
```python
# Quick setup for different scenarios
strict_config = Config.strict_mode(repo_path="/critical/project")
lenient_config = Config.lenient_mode(similarity_threshold=0.2)
fast_config = Config.fast_mode(timeout=10)

# Use with any PatchDoctor function
result = run_validation("patch.patch", **asdict(lenient_config))
```

### Performance Optimizations

**Purpose**: Enhanced caching and optional parallel processing for better performance in AI workflows.

**Key Improvements**:
- **GitRunner**: Increased cache size (256 entries), performance monitoring, smart invalidation
- **RepositoryScanner**: Enhanced file content caching (64 entries) with hit/miss tracking
- **Performance Metrics**: Timing data, cache hit ratios, performance reports
- **Parallel Processing**: Optional multi-threading in `validate_incremental()`

**AI Agent Benefits**:
```python
# Enable performance monitoring
git_runner = GitRunner(enable_performance_monitoring=True)
print(git_runner.get_performance_report())

# Check cache efficiency
scanner = RepositoryScanner(enable_performance_monitoring=True)
stats = scanner.get_cache_stats()
print(f"File cache hit ratio: {stats['cache_hit_ratio']:.2%}")
```

### OpenAPI Schema Generation

**Purpose**: Enable dynamic API discovery for code generation tools and automated integration.

**Key Features**:
- `generate_api_schema()`: Complete OpenAPI 3.0.0 compatible schema
- Function signatures, parameter types, and return schemas
- Comprehensive error code documentation
- Usage examples and best practices

**AI Agent Benefits**:
```python
# Generate complete API schema
schema = generate_api_schema()

# Use for dynamic tool integration
functions = schema["functions"]
error_codes = schema["error_codes"]
examples = schema["examples"]

# Validate function inputs against schema
# Enable automated code generation
```

### Best Practices for AI Agent Integration

1. **Error Handling**: Always check `result["success"]` and handle `ErrorInfo` for recovery
2. **Configuration**: Use appropriate profiles (`strict_mode`, `lenient_mode`, `fast_mode`)
3. **Safety**: Use `apply_safe_fixes()` with appropriate safety levels and dry-run mode
4. **Progress**: Implement progress callbacks for long-running operations
5. **Performance**: Enable caching and parallel processing for large workloads
6. **Rollback**: Always maintain rollback capabilities for safety

### Integration Examples

The `/examples` directory contains complete working examples:
- `basic_validation_workflow.py`: Simple validation and fix application
- `iterative_patch_refinement.py`: Large patch splitting and incremental application
- `batch_patch_processing.py`: Multi-patch workflows with dependency tracking
- `ci_integration_example.py`: Structured logging and reporting for CI/CD

### Performance Considerations

- **Caching**: File content and git command results are cached with smart invalidation
- **Parallel Processing**: Use `max_concurrent > 1` for CPU-bound operations
- **Memory Usage**: Large files (>50MB default) are automatically excluded from processing
- **Timeouts**: Configurable timeouts prevent hanging on problematic repositories

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
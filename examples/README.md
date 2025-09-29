# PatchDoctor AI Agent Integration Examples

This directory contains practical examples demonstrating how AI agents can integrate with PatchDoctor for automated patch validation and application workflows.

## Available Examples

### 1. Basic Validation Workflow (`basic_validation_workflow.py`)
- **Purpose**: Simple validation → fix application → re-validation pattern
- **Features**: Comprehensive error handling, progress reporting, user feedback patterns
- **Use Case**: Getting started with AI agent integration, basic validation workflows

### 2. Iterative Patch Refinement (`iterative_patch_refinement.py`)
- **Purpose**: Large patch → split → incremental application pattern
- **Features**: Patch splitting, partial failure handling, rollback mechanisms
- **Use Case**: Processing large or complex patches that need to be broken down

### 3. Batch Patch Processing (`batch_patch_processing.py`)
- **Purpose**: Multi-patch workflows with dependency tracking
- **Features**: Dependency analysis, conflict resolution, batch rollback
- **Use Case**: Processing multiple related patches in sequence

### 4. CI Integration (`ci_integration_example.py`)
- **Purpose**: Pipeline integration with structured reporting
- **Features**: JSON output, structured logging, exit codes for CI/CD
- **Use Case**: Automated validation in continuous integration pipelines

## Getting Started

Each example is self-contained and includes:
- Comprehensive error handling using PatchDoctor's structured ErrorInfo system
- Progress monitoring and user feedback
- Best practices for AI agent integration
- Copy-paste starting points for common workflows

## Error Handling

All examples demonstrate proper use of PatchDoctor's structured error handling:

```python
from patchdoctor import run_validation, ErrorInfo, ERROR_NO_PATCHES_FOUND

result = run_validation(patch_dir="./patches", verbose=False)
if not result["success"]:
    error_info = result.get("error_info")
    if error_info and error_info["code"] == ERROR_NO_PATCHES_FOUND:
        # Handle no patches found - create them or check directory
        pass
    else:
        # Handle other errors
        pass
```

## Progress Monitoring

Examples show how to use progress callbacks for long-running operations:

```python
def progress_callback(patch_file, result):
    status = getattr(result, 'overall_status', 'UNKNOWN') if result else 'FAILED'
    print(f"Processed {patch_file}: {status}")

result = validate_incremental(
    patch_dir="./patches",
    progress_callback=progress_callback,
    max_concurrent=1,
    verbose=False
)
```

## Safety and Rollback

Examples demonstrate safe fix application with rollback capabilities and automatic input type detection:

```python
from patchdoctor import apply_safe_fixes, run_validation

# Get validation results as dict
result = run_validation(patch_dir="./patches", verbose=False)

# Apply fixes directly from dict results (automatic detection and conversion)
fix_result = apply_safe_fixes(
    verification_result=result["results"][0],  # First patch result
    safety_levels=["safe"],
    dry_run=False,
    confirm=False  # For AI automation
)

# Check rollback info for safety
rollback_info = fix_result["rollback_info"]
if rollback_info["git_stash_id"]:
    # Rollback available via git stash pop
    pass
```

## Requirements

- Python 3.10+
- PatchDoctor (with AI agent integration features)
- Git repository for testing
- Dependencies: `rich`, `chardet`, `cachetools`

## Installation

```bash
# Install dependencies
pip install rich chardet cachetools

# Or use uv for development
uv pip install -e ".[dev]"
```

## Running Examples

```bash
# Basic validation workflow
python examples/basic_validation_workflow.py --patch-dir ./test-patches

# Iterative refinement of large patch
python examples/iterative_patch_refinement.py --patch-file large-patch.patch

# Batch processing multiple patches
python examples/batch_patch_processing.py --patch-dir ./release-patches

# CI integration
python examples/ci_integration_example.py --json-output results.json
```

## Integration Patterns

### 1. Validation-First Pattern
```python
# Validate first, then decide on fixes
result = run_validation(patch_dir="patches", verbose=False)
if result["success"]:
    # Process successful validation
    print(f"Successfully validated {result.get('total_patches', 0)} patches")
else:
    # Analyze errors and apply fixes
    for error in result.get("errors", []):
        if error.get("recoverable", False):
            # Apply automated recovery
            print(f"Recoverable error: {error.get('message', 'Unknown')}")
```

### 2. Progressive Refinement Pattern
```python
# Split large patches and apply incrementally
patches = split_large_patch(patch_content, strategy="by_file")
for patch in patches:
    result = validate_from_content(patch)
    if not result["success"]:
        # Handle partial failures
        pass
```

### 3. Batch Processing Pattern
```python
# Process multiple patches with dependency tracking
result = validate_incremental(
    patch_dir="patches",
    early_stop_on_error=False,
    max_concurrent=3,
    verbose=False
)
```

## Best Practices

1. **Always check error_info**: Use structured error information for intelligent recovery
2. **Use safety levels**: Apply only "safe" fixes automatically, review others
3. **Monitor progress**: Provide feedback during long operations
4. **Enable rollback**: Always prepare for rollback in automated scenarios
5. **Validate incrementally**: Process large patch sets piece by piece
6. **Handle edge cases**: Check for empty directories, invalid patches, permission issues

## Contributing

When adding new examples:
1. Follow the established error handling patterns
2. Include comprehensive docstrings and comments
3. Demonstrate both success and failure scenarios
4. Add progress monitoring for operations > 5 seconds
5. Include rollback/cleanup mechanisms
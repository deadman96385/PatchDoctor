# PatchDoctor

A Git patch verification and analysis tool that ensures all changes from patch files have been properly applied to your codebase.

## Features

- **Smart File Path Handling**: Support for spaces and special characters in file paths
- **Configurable Similarity Thresholds**: Fine-tune hunk matching sensitivity
- **Intelligent Hunk Matching**: Context-aware positioning with fuzzy matching
- **Comprehensive Caching**: Improved performance with automatic cache management
- **Responsive Terminal Output**: Adaptive width detection with professional formatting
- **Detailed Error Analysis**: Actionable fix suggestions for missing changes
- **Robust Error Handling**: Timeouts and fallbacks for git operations
- **Large File Support**: Configurable size limits with memory-efficient processing

## Installation

```bash
# Create virtual environment
uv venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# Install from source
uv pip install -e .

# Or install dependencies manually
uv pip install rich chardet
```

## Usage

```bash
# Basic verification
uv run patchdoctor.py

# Verbose output with custom similarity threshold
uv run patchdoctor.py -v -s 0.5

# Process large files with extended timeout
uv run patchdoctor.py -M 200 -T 60

# Generate detailed report
uv run patchdoctor.py -r detailed_report.txt

# Generate JSON report for CI/CD
uv run patchdoctor.py -j report.json

# Show all fix suggestions
uv run patchdoctor.py --show-all-fixes
```

## Command Line Options

### Basic Options
- `-d, --patch-dir DIR`: Directory containing patch files (default: current)
- `-R, --repo DIR`: Repository root directory (default: current)
- `-v, --verbose`: Show detailed diff information
- `-r, --report FILE`: Save detailed report to file
- `-j, --json-report FILE`: Save JSON report for CI/CD integration
- `-D, --report-detailed`: Include fix suggestions in file reports
- `--show-all-fixes`: Show all available fix suggestions (default: show first 3 only)
- `-c, --no-color`: Disable colored output

### Advanced Configuration
- `-s, --similarity-threshold FLOAT`: Line similarity threshold (0.0-1.0, default: 0.3)
- `-t, --hunk-tolerance INT`: Search tolerance for hunk matching (default: 5)
- `-T, --timeout INT`: Timeout for git operations in seconds (default: 30)
- `-M, --max-file-size INT`: Maximum file size to process in MB (default: 100)

## Examples

```bash
# Basic verification
uv run patchdoctor.py

# Verbose output with custom settings
uv run patchdoctor.py -v -s 0.5 -t 10

# Process patches in specific directory
uv run patchdoctor.py -d /path/to/patches -v

# Generate report without colors
uv run patchdoctor.py -c -r analysis.txt

# Generate JSON report for automation
uv run patchdoctor.py -j report.json

# Extended timeout for large repositories
uv run patchdoctor.py -T 60 -M 200
```

## Report Output Formats

### Text Report Format
The detailed text report (`-r` option) includes:
- Summary of patch verification results
- Detailed file-by-file analysis
- Fix suggestions for missing changes

Example output showing fix suggestions:
```
PATCH:
  0002-Add-tests.patch
   Commit:    442ecf1c
   Subject:   Add tests
   Status:    [PARTIAL] PARTIAL PARTIALLY_APPLIED
   Progress:  [██████████████░░░░░░] (19/26)


[FILE DETAILS]
--------------------------------------------------
   [+] CREATED  tests/README.md
   [+] CREATED  tests/__init__.py
   [+] CREATED  tests/conftest.py
   [X] MISSING  .../mock_firmware/corrupted_firmware.zip
      |
      [FIX SUGGESTIONS]
      |
      |- [SAFE] Extract missing file from patch commit
      |
      |- [SAFE] Extract binary file (size: ~0 bytes)
      |  Commands:
      |  # For binary files, use git checkout:
      |  git checkout 442ecf1c39f2cec47277dd6fc6336a7b20f115eb -- .../mock_firmware/corrupted_firmware.zip
      |  # Or show file info:
      |  git show --stat 442ecf1c39f2cec47277dd6fc6336a7b20f115eb -- .../mock_firmware/corrupted_firmware.zip
      |
      |- [SAFE] Get information about the missing file
      |  Commands:
      |  # See what this file contains:
      |  git show 442ecf1c39f2cec47277dd6fc6336a7b20f115eb:.../mock_firmware/corrupted_firmware.zip | head -20
      |  # Check file size and type:
      |  git cat-file -s 442ecf1c39f2cec47277dd6fc6336a7b20f115eb:.../mock_firmware/corrupted_firmware.zip
      |  git cat-file -t 442ecf1c39f2cec47277dd6fc6336a7b20f115eb:.../mock_firmware/corrupted_firmware.zip
```

### JSON Report Format
The JSON report (`--json-report` option) provides structured data for CI/CD integration:

```json
[
  {
    "patch_info": {
      "filename": "feature.patch",
      "commit_hash": "a1b2c3d",
      "author": "John Doe",
      "date": "Mon, 01 Oct 2023 12:00:00 +0000",
      "subject": "Add new feature",
      "files_changed": [...],
      "total_insertions": 50,
      "total_deletions": 10
    },
    "file_results": [...],
    "overall_status": "FULLY_APPLIED",
    "success_count": 5,
    "total_count": 5
  }
]
```

## Binary Patch Handling

PatchDoctor supports binary files in patches. Binary patches are detected by the `Binary` keyword in diff stats.

Example binary patch verification:
```bash
# For binary files, PatchDoctor provides git-based restoration commands
patchdoctor -v
```

Output for binary files:
```
[+] image.png
  Expected: modify
  Actual: modified
  Status: OK
  Details: Binary file modified (size: 245KB)

Fix suggestions:
  [SAFE] Extract binary file from patch commit
  Commands:
  $ git show a1b2c3d:image.png > image.png
```

## Exit Codes

- `0`: All patches successfully applied
- `1`: Some patches have issues or errors occurred

## Architecture

PatchDoctor uses a modular architecture with specialized components:

- **PatchParser**: Parses git patch files and extracts file operations
- **RepositoryScanner**: Scans repository state and compares against patches
- **PatchVerifier**: Main verification engine with intelligent matching
- **ReportGenerator**: Generates professional console and file reports
- **GitRunner**: Unified git command execution with caching and error handling

## Dependencies

- **rich**: Professional terminal formatting and UI components
- **chardet**: Automatic encoding detection for file reading
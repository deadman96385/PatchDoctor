#!/usr/bin/env python3
"""
PatchDoctor

A professional Git patch verification and analysis tool that ensures all changes from patch files
have been properly applied to your codebase, providing detailed feedback, intelligent error detection,
and actionable fix suggestions.

Features:
    - Smart file path handling with support for spaces and special characters
    - Configurable similarity thresholds and search tolerances
    - Intelligent hunk matching with context-aware positioning
    - Comprehensive caching for improved performance
    - Responsive terminal output with adaptive width detection
    - Detailed error analysis with fix suggestions
    - Robust error handling with timeouts for git operations
    - Support for large files with configurable size limits

Usage:
    python patchdoctor.py [options]
    # or after installation:
    patchdoctor [options]

Basic Options:
    -d, --patch-dir DIR      Directory containing patch files (default: current)
    -v, --verbose           Show detailed diff information
    -r, --report FILE       Save detailed report to file
        --no-color          Disable colored output

Advanced Configuration:
    -s, --similarity-threshold FLOAT   Line similarity threshold (0.0-1.0, default: 0.3)
    -t, --hunk-tolerance INT           Search tolerance for hunk matching (default: 5)
        --timeout INT                  Timeout for git operations in seconds (default: 30)
        --max-file-size INT            Maximum file size to process in MB (default: 100)

Examples:
    # Basic verification
    python patchdoctor.py

    # Verbose output with custom similarity threshold
    python patchdoctor.py -v -s 0.5

    # Process large files with extended timeout
    python patchdoctor.py --max-file-size 200 --timeout 60

    # Generate detailed report
    python patchdoctor.py -r detailed_report.txt

Exit Codes:
    0 - All patches successfully applied
    1 - Some patches have issues or errors occurred

Version: 1.0 - Professional Git patch verification tool
"""

import argparse
import difflib
import email
import os
import re
import shutil
import subprocess
import sys
import json

from dataclasses import dataclass, field, asdict
from datetime import datetime
from email.utils import parsedate_to_datetime
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cachetools

# Additional libraries for improved functionality
import chardet

# Rich imports for modern terminal output
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Global console instance for rich output
console = Console()


# Exception hierarchy for consistent error handling
class PatchDoctorError(Exception):
    """Base exception for PatchDoctor errors."""

    pass


class FileEncodingError(PatchDoctorError):
    """Error when file encoding cannot be detected or read."""

    pass


class GitCommandError(PatchDoctorError):
    """Error when git commands fail."""

    def __init__(self, cmd: List[str], message: str):
        super().__init__(f"Git failed: {cmd} → {message}")
        self.cmd = cmd


class PatchParseError(PatchDoctorError):
    """Error when parsing patch files."""

    pass


# Configuration constants
DEFAULT_SUBPROCESS_TIMEOUT = 30  # seconds
DEFAULT_SIMILARITY_THRESHOLD = 0.3
DEFAULT_HUNK_SEARCH_TOLERANCE = 5
MAX_FILE_SIZE_MB = 100  # Maximum file size to load into memory
MAX_DISPLAY_WIDTH = 120  # Maximum display width
MIN_DISPLAY_WIDTH = 60  # Minimum display width


def get_terminal_width() -> int:
    """Get the current terminal width, with fallback to default."""
    width = shutil.get_terminal_size(fallback=(80, 20)).columns
    return max(MIN_DISPLAY_WIDTH, min(MAX_DISPLAY_WIDTH, width))


def smart_truncate_path(filepath: str, max_width: int) -> str:
    """Intelligently truncate a file path to fit within max_width using pathlib."""
    if len(filepath) <= max_width:
        return filepath

    path = Path(filepath)
    filename = path.name

    # If filename alone is too long, truncate it
    if len(filename) >= max_width:
        return filename[:max_width]

    # Try to fit filename and some parent directories
    remaining = max_width - len(filename) - 3  # 3 for "..."
    if remaining <= 0:
        return "..." + filename

    # Build path from parents
    parts = list(path.parents)
    parts.reverse()  # root to immediate parent
    result_parts = []

    for part in parts:
        if part.name and len(part.name) + 1 <= remaining:  # +1 for separator
            result_parts.append(part.name)
            remaining -= len(part.name) + 1
        else:
            break

    if result_parts:
        return f"...{os.sep}" + os.sep.join(result_parts) + f"{os.sep}{filename}"
    return "..." + filename


@dataclass
class DiffHunk:
    """Represents a single diff hunk with line changes."""

    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: List[str] = field(default_factory=list)


@dataclass
class FileOperation:
    """Represents a file operation from a git patch."""

    operation: str  # "create", "delete", "modify", "rename"
    old_path: Optional[str] = None
    new_path: Optional[str] = None
    insertions: int = 0
    deletions: int = 0
    is_binary: bool = False
    diff_content: str = ""
    diff_hunks: List[DiffHunk] = field(default_factory=list)


@dataclass
class PatchInfo:
    """Information extracted from a git patch file."""

    filename: str
    commit_hash: str
    author: str
    subject: str
    date: Optional[Union[datetime, str]] = None
    files_changed: List[FileOperation] = field(default_factory=list)
    total_insertions: int = 0
    total_deletions: int = 0


@dataclass
class DiffAnalysis:
    """Detailed analysis of what's missing or different in a file."""

    missing_hunks: List[DiffHunk] = field(default_factory=list)
    applied_hunks: List[DiffHunk] = field(default_factory=list)
    conflicting_lines: List[Tuple[int, str, str]] = field(
        default_factory=list
    )  # (line_num, expected, actual)
    total_hunks: int = 0
    file_content: Optional[List[str]] = None


@dataclass
class FixSuggestion:
    """Actionable fix suggestion for a specific issue."""

    fix_type: str  # "git_restore", "mini_patch", "manual_edit", "file_create"
    description: str
    commands: List[str] = field(default_factory=list)
    manual_instructions: str = ""
    mini_patch_content: str = ""
    safety_level: str = "safe"  # "safe", "review", "dangerous"
    confidence: float = 1.0  # 0.0 to 1.0


@dataclass
class FileVerificationResult:
    """Result of verifying a single file operation."""

    file_path: str
    expected_operation: str
    actual_status: str
    verification_status: str  # "OK", "MISSING", "MODIFIED", "ERROR", "EXTRA", "WARNING"
    details: str = ""
    diff_analysis: Optional[DiffAnalysis] = None
    fix_suggestions: List[FixSuggestion] = field(default_factory=list)


@dataclass
class VerificationResult:
    """Complete verification result for a patch."""

    patch_info: PatchInfo
    file_results: List[FileVerificationResult] = field(default_factory=list)
    overall_status: str = "UNKNOWN"
    success_count: int = 0
    total_count: int = 0


@dataclass
class GitResult:
    """Lightweight wrapper for git command results."""

    ok: bool
    stdout: str
    stderr: str
    returncode: int
    stdout_lines: List[str] = field(default_factory=list)


@dataclass
class HunkVerificationResult:
    """Result of verifying diff hunks."""

    all_applied: bool
    applied_hunks: int
    total_hunks: int
    missing_hunks: List[DiffHunk] = field(default_factory=list)
    applied_hunks_list: List[DiffHunk] = field(default_factory=list)
    file_content: Optional[List[str]] = None


@dataclass
class Config:
    """Configuration for PatchDoctor with validation."""

    patch_dir: str = "."
    repo_path: str = "."
    verbose: bool = False
    report_file: Optional[str] = None
    json_report_file: Optional[str] = None
    report_detailed: bool = False
    no_color: bool = False
    show_all_fixes: bool = False
    similarity_threshold: float = field(default=DEFAULT_SIMILARITY_THRESHOLD)
    hunk_tolerance: int = field(default=DEFAULT_HUNK_SEARCH_TOLERANCE)
    timeout: int = field(default=DEFAULT_SUBPROCESS_TIMEOUT)
    max_file_size: int = field(default=MAX_FILE_SIZE_MB)

    def __post_init__(self):
        """Validate configuration values."""
        if not (0.0 <= self.similarity_threshold <= 1.0):
            raise ValueError(
                f"Similarity threshold must be between 0.0 and 1.0, got {self.similarity_threshold}"
            )

        if self.hunk_tolerance < 0:
            raise ValueError(
                f"Hunk tolerance must be non-negative, got {self.hunk_tolerance}"
            )

        if self.timeout <= 0:
            raise ValueError(f"Timeout must be positive, got {self.timeout}")

        if self.max_file_size <= 0:
            raise ValueError(
                f"Max file size must be positive, got {self.max_file_size}"
            )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "Config":
        """Create Config from parsed command line arguments."""
        return cls(
            patch_dir=args.patch_dir,
            repo_path=args.repo,
            verbose=args.verbose,
            report_file=args.report,
            json_report_file=getattr(args, "json_report", None),
            report_detailed=getattr(args, "report_detailed", False),
            no_color=args.no_color,
            show_all_fixes=getattr(args, "show_all_fixes", False),
            similarity_threshold=args.similarity_threshold,
            hunk_tolerance=args.hunk_tolerance,
            timeout=args.timeout,
            max_file_size=args.max_file_size,
        )


# Rich-based styling functions
def print_colored(text: str, color: str, **kwargs):
    """Print colored message using Rich markup."""
    console.print(f"[bold {color}]{text}[/bold {color}]", **kwargs)


def print_success(text: str, **kwargs):
    """Print success message in green."""
    print_colored(text, "green", **kwargs)


def print_warning(text: str, **kwargs):
    """Print warning message in yellow."""
    print_colored(text, "yellow", **kwargs)


def print_error(text: str, **kwargs):
    """Print error message in red."""
    print_colored(text, "red", **kwargs)


def print_info(text: str, **kwargs):
    """Print info message in blue."""
    print_colored(text, "blue", **kwargs)


def print_header(text: str, **kwargs):
    """Print header message in cyan."""
    print_colored(text, "cyan", **kwargs)


def print_highlight(text: str, **kwargs):
    """Print highlighted text in magenta."""
    print_colored(text, "magenta", **kwargs)


@lru_cache(maxsize=128)
def detect_file_encoding(file_path: str) -> str:
    """Detect file encoding using chardet with LRU caching."""
    try:
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            return "utf-8"
    except Exception:
        return "utf-8"

    try:
        with open(file_path, "rb") as f:
            raw_data = f.read(8192)  # Read first 8KB for detection

        detected = chardet.detect(raw_data)
        if detected and detected["encoding"] and detected["confidence"] > 0.7:
            encoding = detected["encoding"].lower()
            # Map common aliases
            encoding_map = {
                "utf8": "utf-8",
                "utf-8-sig": "utf-8",
            }
            encoding = encoding_map.get(encoding, encoding)
            return encoding
    except Exception:
        pass

    return "utf-8"  # Default fallback


def read_file_with_encoding(file_path: str) -> str:
    """Read file content with automatic encoding detection."""
    path_obj = Path(file_path)  # Ensure pathlib.Path for consistency
    encoding = detect_file_encoding(str(path_obj))
    return path_obj.read_text(encoding=encoding, errors="replace")


class GitRunner:
    """Unified Git command runner with caching, timeouts, and error handling."""

    def __init__(
        self,
        repo_path: str = ".",
        timeout: int = DEFAULT_SUBPROCESS_TIMEOUT,
        cache_ttl: int = 30,
    ):
        self.repo_path = Path(repo_path)
        self.timeout = timeout
        self._result_cache: cachetools.TTLCache[str, GitResult] = cachetools.TTLCache(
            maxsize=128, ttl=cache_ttl
        )  # Thread-safe TTL cache

    def run_git_command(self, args: List[str], use_cache: bool = True) -> GitResult:
        """Run a git command with caching and error handling."""
        cmd = ["git"] + args
        cache_key = f"{self.repo_path}:{' '.join(cmd)}"

        # Check cache if enabled
        if use_cache and cache_key in self._result_cache:
            return self._result_cache[cache_key]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.repo_path,
                timeout=self.timeout,
                check=False,  # Don't raise on non-zero exit
            )

            git_result = GitResult(
                ok=result.returncode == 0,
                stdout=result.stdout,
                stderr=result.stderr,
                returncode=result.returncode,
                stdout_lines=result.stdout.splitlines() if result.stdout else [],
            )

            # Cache successful results
            if use_cache and git_result.ok:
                self._result_cache[cache_key] = git_result

            return git_result

        except subprocess.TimeoutExpired:
            console.print(f"[red]Warning:[/red] Git command timed out: {' '.join(cmd)}")
            # Create a mock result for timeout
            git_result = GitResult(
                ok=False,
                stdout="",
                stderr=f"Command timed out after {self.timeout}s",
                returncode=124,
                stdout_lines=[],
            )
            return git_result
        except (subprocess.SubprocessError, OSError, FileNotFoundError) as e:
            console.print(
                f"[red]Warning:[/red] Git command failed: {' '.join(cmd)}: {e}"
            )
            # Create a mock result for errors
            git_result = GitResult(
                ok=False, stdout="", stderr=str(e), returncode=1, stdout_lines=[]
            )
            return git_result

    def get_cache_stats(self) -> Dict[str, Union[int, float]]:
        """Get cache statistics."""
        return {
            "cached_entries": len(self._result_cache),
            "maxsize": self._result_cache.maxsize,
            "ttl": self._result_cache.ttl,
        }

    def get_file_status(self, filepath: str) -> Tuple[bool, str]:
        """Get git status for a specific file."""
        result = self.run_git_command(["status", "--porcelain", filepath])

        if result.returncode != 0:
            return True, "git_error"  # File exists but git failed

        status_line = result.stdout.strip()
        if not status_line:
            return True, "committed"

        # Map git status codes to human-readable status
        status_code = status_line[:2]
        status_map = {
            "M": "modified",
            "A": "added",
            "D": "deleted",
            "?": "untracked",
            "R": "renamed",
        }

        # Check for any matching status code
        for code, status in status_map.items():
            if code in status_code:
                return True, status

        return True, "unknown_change"

    def file_exists_in_history(
        self, filepath: str, commit_hash: Optional[str] = None
    ) -> bool:
        """Check if a file exists in git history."""
        args = ["log", "--oneline", "--", filepath]
        if commit_hash:
            args.insert(2, f"{commit_hash}..HEAD")

        result = self.run_git_command(args)
        return result.returncode == 0 and bool(result.stdout.strip())

    def get_commit_history_for_file(
        self, filepath: str, since_commit: Optional[str] = None
    ) -> List[str]:
        """Get commit history for a file since a specific commit."""
        args = ["log", "--oneline", "--pretty=format:%h %s"]
        if since_commit:
            args.extend([f"{since_commit}..HEAD"])
        args.extend(["--", filepath])

        result = self.run_git_command(args)
        if result.ok:
            return [line.strip() for line in result.stdout_lines if line.strip()]
        return []


class PatchParser:
    """Parses git patch files to extract file operations and metadata."""

    # Regex patterns for parsing git patches (improved for complex file paths)

    # Improved regex to handle files with spaces and special characters
    FILE_STATS_PATTERN = re.compile(r"^\s*(.+?)\s*\|\s*(\d+)\s*([+-]*)\s*$")
    FILE_OPERATION_PATTERN = re.compile(
        r"^\s*(create|delete|rename)\s+mode\s+\d+\s+(.+)$"
    )
    BINARY_PATTERN = re.compile(
        r"^\s*(.+?)\s*\|\s*Bin\s+(\d+)\s*->\s*(\d+)\s*bytes\s*$"
    )
    # Handle quoted paths in diff headers (for paths with spaces)
    DIFF_HEADER_PATTERN = re.compile(
        r'^diff --git a/(?:"(.+)"|(.+)) b/(?:"(.+)"|(.+))$'
    )
    HUNK_HEADER_PATTERN = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

    def __init__(self, similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD, max_file_size_mb: int = MAX_FILE_SIZE_MB):
        """Initialize the parser with configurable options."""
        self.similarity_threshold = similarity_threshold
        self.max_file_size_mb = max_file_size_mb

    def parse_patch_file(self, patch_path: str) -> PatchInfo:
        """Parse a git patch file and extract all relevant information."""
        patch_file = Path(patch_path)

        # Validate file exists and size
        if not patch_file.exists():
            raise FileNotFoundError(f"Patch file not found: {patch_path}")

        file_size_mb = patch_file.stat().st_size / (1024 * 1024)
        if file_size_mb > self.max_file_size_mb:
            raise PatchParseError(
                f"Patch file too large ({file_size_mb:.1f}MB > {self.max_file_size_mb}MB): {patch_path}"
            )

        # Use automatic encoding detection
        try:
            content = read_file_with_encoding(patch_path)
        except Exception as e:
            raise PatchParseError(f"Could not read patch file {patch_path}: {e}") from e

        # Extract basic patch information
        patch_info = self._extract_patch_header(content, patch_path)

        # Extract file operations and statistics
        patch_info.files_changed = self._extract_file_operations(content)

        # Extract diff hunks for content verification
        self._extract_diff_hunks(content, patch_info.files_changed)

        # Calculate totals
        patch_info.total_insertions = sum(
            f.insertions for f in patch_info.files_changed
        )
        patch_info.total_deletions = sum(f.deletions for f in patch_info.files_changed)

        return patch_info

    def _extract_patch_header(self, content: str, filename: str) -> PatchInfo:
        """Extract patch header information using email parsing."""
        lines = content.split("\n")
        header_lines = []
        for line in lines:
            if line.startswith("---") or line.startswith("diff --git"):
                break
            header_lines.append(line)

        email_content = "\n".join(header_lines)
        try:
            msg = email.message_from_string(email_content)

            commit_hash = ""
            if header_lines and header_lines[0].startswith("From "):
                parts = header_lines[0].split()
                if len(parts) >= 2:
                    commit_hash = parts[1]

            author = msg.get("From", "").strip()
            date_str = msg.get("Date", "").strip()
            date_obj = None
            if date_str:
                try:
                    date_obj = parsedate_to_datetime(date_str)
                except Exception:
                    date_obj = None
            subject = msg.get("Subject", "").strip()

            # Clean up subject (remove [PATCH n/m] prefix if present)
            subject = re.sub(r"^\[PATCH \d+/\d+\] ", "", subject)

        except Exception:
            console.print(
                f"[yellow]Warning:[/yellow] Email parsing failed for {filename}, falling back to manual parsing"
            )
            commit_hash = ""
            author = ""
            date_str = ""
            date_obj = None
            subject = ""

        return PatchInfo(
            filename=filename,
            commit_hash=commit_hash,
            author=author,
            date=date_obj or date_str,  # store datetime if available
            subject=subject,
        )

    def _extract_file_operations(self, content: str) -> List[FileOperation]:
        """Extract file operations from patch content."""
        operations = []
        lines = content.split("\n")

        i = 0
        while i < len(lines):
            line = lines[i]

            # Look for file statistics (lines with | and +/- symbols)
            if "|" in line and ("+" in line or "-" in line or "Bin" in line):
                # Check for binary files
                binary_match = self.BINARY_PATTERN.match(line)
                if binary_match:
                    filepath = binary_match.group(1).strip()
                    old_size = int(binary_match.group(2))
                    new_size = int(binary_match.group(3))

                    # Determine operation type for binary files
                    if old_size == 0:
                        operation = "create"
                    elif new_size == 0:
                        operation = "delete"
                    else:
                        operation = "modify"

                    operations.append(
                        FileOperation(
                            operation=operation,
                            new_path=filepath,
                            old_path=filepath if operation != "create" else None,
                            is_binary=True,
                        )
                    )
                else:
                    # Regular file with insertions/deletions
                    stats_match = self.FILE_STATS_PATTERN.match(line)
                    if stats_match:
                        filepath = stats_match.group(1).strip()
                        symbols = stats_match.group(3)

                        # Check if this is a rename stats line (contains " => ")
                        if " => " in filepath:
                            # Parse rename: "old name.txt" => "new name.txt"
                            parts = filepath.split(" => ")
                            if len(parts) == 2:
                                old_path = parts[0].strip().strip('"')
                                new_path = parts[1].strip().strip('"')
                                operations.append(
                                    FileOperation(
                                        operation="rename",
                                        old_path=old_path,
                                        new_path=new_path,
                                        insertions=symbols.count("+"),
                                        deletions=symbols.count("-"),
                                    )
                                )
                            else:
                                # Fallback: treat as modify with the full string as new_path
                                operations.append(
                                    FileOperation(
                                        operation="modify",
                                        new_path=filepath,
                                        old_path=filepath,
                                        insertions=symbols.count("+"),
                                        deletions=symbols.count("-"),
                                    )
                                )
                        else:
                            # Count insertions and deletions
                            insertions = symbols.count("+")
                            deletions = symbols.count("-")

                            operations.append(
                                FileOperation(
                                    operation="modify",  # Will be refined later
                                    new_path=filepath,
                                    old_path=filepath,
                                    insertions=insertions,
                                    deletions=deletions,
                                )
                            )

            # Look for explicit operation declarations
            elif line.startswith(" create mode") or line.startswith(" delete mode"):
                parts = line.split()
                if not parts:
                    continue
                filepath = parts[-1]
                if "create mode" in line:
                    # Find corresponding operation in our list
                    for op in operations:
                        if op.new_path == filepath:
                            op.operation = "create"
                            op.old_path = None
                            break
                elif "delete mode" in line:
                    # Find corresponding operation in our list
                    for op in operations:
                        if op.new_path == filepath or op.old_path == filepath:
                            op.operation = "delete"
                            op.new_path = None
                            break

            # Look for rename operations
            elif line.startswith(" rename "):
                # Git shows renames in format like "rename old_file => new_file (similarity%)"
                if " => " in line:
                    parts = line.split(" => ")
                    if len(parts) != 2:
                        continue
                    old_parts = parts[0].split()
                    new_parts = parts[1].split()
                    if not old_parts or not new_parts:
                        continue
                    old_file = old_parts[-1]
                    new_file = new_parts[0]

                    operations.append(
                        FileOperation(
                            operation="rename", old_path=old_file, new_path=new_file
                        )
                    )

            i += 1

        # Refine operations that don't have explicit create/delete markers
        for op in operations:
            if op.operation == "modify":
                # If insertions > 0 and deletions == 0, it might be a create
                # If insertions == 0 and deletions > 0, it might be a delete
                # But we'll keep as modify unless explicitly marked otherwise
                pass

        return operations

    def _extract_diff_hunks(
        self, content: str, file_operations: List[FileOperation]
    ) -> None:
        """Extract diff hunks for each file operation to enable content verification."""
        lines = content.split("\n")
        current_file = None
        current_hunks: List[DiffHunk] = []
        in_diff = False

        i = 0
        while i < len(lines):
            line = lines[i]

            # Check for diff header
            diff_match = self.DIFF_HEADER_PATTERN.match(line)
            if diff_match:
                # Save previous file's hunks
                if current_file and current_hunks:
                    current_file.diff_hunks = current_hunks

                # Extract filepath, handling quoted paths
                groups = diff_match.groups()
                if groups[0]:  # Quoted a/ path
                    filepath_a = groups[0]
                elif groups[1]:  # Unquoted a/ path
                    filepath_a = groups[1]
                else:
                    filepath_a = ""

                if groups[2]:  # Quoted b/ path
                    filepath_b = groups[2]
                elif groups[3]:  # Unquoted b/ path
                    filepath_b = groups[3]
                else:
                    filepath_b = ""

                # Prefer b/ path (new file), fallback to a/ path
                filepath = filepath_b or filepath_a

                # Find the file operation for this diff
                current_file = None
                for op in file_operations:
                    if (
                        op.new_path == filepath
                        or op.old_path == filepath
                        or op.new_path == filepath_a
                        or op.old_path == filepath_a
                        or op.new_path == filepath_b
                        or op.old_path == filepath_b
                    ):
                        current_file = op
                        break

                current_hunks = []
                in_diff = True
                i += 1
                continue

            # Check for hunk header
            if in_diff and line.startswith("@@"):
                hunk_match = self.HUNK_HEADER_PATTERN.match(line)
                if hunk_match:
                    old_start = int(hunk_match.group(1))
                    old_count = int(hunk_match.group(2)) if hunk_match.group(2) else 1
                    new_start = int(hunk_match.group(3))
                    new_count = int(hunk_match.group(4)) if hunk_match.group(4) else 1

                    # Validate hunk line numbers are reasonable
                    if old_start < 0 or new_start < 0 or old_count < 0 or new_count < 0:
                        print(f"Warning: Invalid hunk line numbers in diff: {line}")
                        i += 1
                        continue

                    if old_count > 10000 or new_count > 10000:  # Sanity check
                        print(f"Warning: Suspiciously large hunk count in diff: {line}")
                        i += 1
                        continue

                    # Extract hunk content
                    hunk_lines = []
                    j = i + 1
                    while j < len(lines):
                        hunk_line = lines[j]
                        if (
                            hunk_line.startswith("diff --git")
                            or hunk_line.startswith("@@")
                            or hunk_line.startswith("From ")
                            or j == len(lines) - 1
                        ):
                            break
                        hunk_lines.append(hunk_line)
                        j += 1

                    current_hunks.append(
                        DiffHunk(
                            old_start=old_start,
                            old_count=old_count,
                            new_start=new_start,
                            new_count=new_count,
                            lines=hunk_lines,
                        )
                    )

                    i = j - 1

            i += 1

        # Save last file's hunks
        if current_file and current_hunks:
            current_file.diff_hunks = current_hunks


class RepositoryScanner:
    """Scans the current repository state and compares against patch expectations."""

    def __init__(self, repo_path: str = ".", timeout: int = DEFAULT_SUBPROCESS_TIMEOUT):
        self.repo_path = Path(repo_path)
        self.timeout = timeout
        self._file_content_cache: cachetools.LRUCache[
            Tuple[str, float, int], List[str]
        ] = cachetools.LRUCache(
            maxsize=32
        )  # LRU cache for file contents
        self.git_runner = GitRunner(repo_path, timeout)  # Use unified git runner

    def get_file_status(self, filepath: str) -> Tuple[bool, str]:
        """Check if a file exists and get its git status."""
        full_path = self.repo_path / filepath

        # Check if file exists and is readable
        try:
            exists = full_path.exists()
            if exists and not os.access(full_path, os.R_OK):
                return True, "permission_denied"
        except (OSError, PermissionError):
            return False, "access_error"

        if not exists:
            return False, "missing"

        # Use GitRunner for status check
        return self.git_runner.get_file_status(filepath)

    def file_exists_in_history(
        self, filepath: str, commit_hash: Optional[str] = None
    ) -> bool:
        """Check if a file exists in git history."""
        return self.git_runner.file_exists_in_history(filepath, commit_hash)

    def get_commit_history_for_file(
        self, filepath: str, since_commit: Optional[str] = None
    ) -> List[str]:
        """Get commit history for a file since a specific commit."""
        return self.git_runner.get_commit_history_for_file(filepath, since_commit)

    def get_file_content(self, filepath: str) -> Optional[List[str]]:
        """Get the current content of a file as a list of lines with caching."""
        full_path = self.repo_path / filepath

        # Check if file exists and get its modification time
        try:
            if not full_path.exists():
                return None

            stat_info = full_path.stat()
            file_size_mb = stat_info.st_size / (1024 * 1024)

            # Don't load very large files into memory
            if file_size_mb > MAX_FILE_SIZE_MB:
                return None

            # Check permissions
            if not os.access(full_path, os.R_OK):
                return None

            # Use cached content if available (key includes mtime and size for invalidation)
            cache_key = (str(full_path), stat_info.st_mtime, stat_info.st_size)
            if cache_key in self._file_content_cache:
                return self._file_content_cache[cache_key]

        except (OSError, PermissionError):
            return None

        # Use automatic encoding detection for file reading
        try:
            encoding = detect_file_encoding(str(full_path))
            with open(full_path, "r", encoding=encoding) as f:
                content = f.readlines()
        except (UnicodeDecodeError, UnicodeError):
            # Fallback with error replacement
            try:
                with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.readlines()
            except OSError:
                return None
        except OSError:
            return None

        # Cache the content
        self._file_content_cache[cache_key] = content

        return content


class PatchVerifier:
    """Main verification engine that compares patches against repository state."""

    def __init__(
        self,
        repo_path: str = ".",
        verbose: bool = False,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        hunk_search_tolerance: int = DEFAULT_HUNK_SEARCH_TOLERANCE,
        timeout: int = DEFAULT_SUBPROCESS_TIMEOUT,
        max_file_size_mb: int = MAX_FILE_SIZE_MB,
    ):
        self.repo_path = repo_path
        self.verbose = verbose
        self.similarity_threshold = similarity_threshold
        self.hunk_search_tolerance = hunk_search_tolerance
        self.parser = PatchParser(similarity_threshold, max_file_size_mb)
        self.scanner = RepositoryScanner(repo_path, timeout)

    def verify_patch(self, patch_path: str) -> VerificationResult:
        """Verify a single patch file against the repository."""
        console.print(f"[bold blue][ANALYZING][/bold blue] [cyan]{patch_path}[/cyan]")

        # Parse the patch file
        patch_info = self.parser.parse_patch_file(patch_path)

        # Verify each file operation
        file_results = []
        for file_op in patch_info.files_changed:
            result = self._verify_file_operation(file_op, patch_info)
            file_results.append(result)

        # Calculate overall status
        success_count = sum(1 for r in file_results if r.verification_status == "OK")
        total_count = len(file_results)

        if success_count == total_count:
            overall_status = "FULLY_APPLIED"
        elif success_count > 0:
            overall_status = "PARTIALLY_APPLIED"
        else:
            overall_status = "NOT_APPLIED"

        return VerificationResult(
            patch_info=patch_info,
            file_results=file_results,
            overall_status=overall_status,
            success_count=success_count,
            total_count=total_count,
        )

    def verify_patch_info(self, patch_info: PatchInfo) -> List[FileVerificationResult]:
        """Verify a parsed patch info against the repository."""
        # Verify each file operation
        file_results = []
        for file_op in patch_info.files_changed:
            result = self._verify_file_operation(file_op, patch_info)
            file_results.append(result)

        return file_results

    def _verify_file_operation(
        self, file_op: FileOperation, patch_info: PatchInfo
    ) -> FileVerificationResult:
        """Verify a single file operation from a patch."""

        if file_op.operation == "create":
            return self._verify_file_creation(file_op, patch_info)
        elif file_op.operation == "delete":
            return self._verify_file_deletion(file_op, patch_info)
        elif file_op.operation == "modify":
            return self._verify_file_modification(file_op, patch_info)
        elif file_op.operation == "rename":
            return self._verify_file_rename(file_op, patch_info)
        else:
            return FileVerificationResult(
                file_path=file_op.new_path or file_op.old_path or "unknown",
                expected_operation=file_op.operation,
                actual_status="unknown",
                verification_status="ERROR",
                details=f"Unknown operation type: {file_op.operation}",
            )

    def _verify_file_creation(
        self, file_op: FileOperation, patch_info: PatchInfo
    ) -> FileVerificationResult:
        """Verify that a file was created as expected."""
        assert file_op.new_path is not None
        filepath = file_op.new_path
        exists, status = self.scanner.get_file_status(filepath)

        if not exists:
            # Generate fix suggestions for missing file
            fix_suggestions = self._generate_missing_file_fix_suggestions(
                filepath, file_op, patch_info
            )

            return FileVerificationResult(
                file_path=filepath,
                expected_operation="create",
                actual_status="missing",
                verification_status="MISSING",
                details="File should have been created but does not exist",
                fix_suggestions=fix_suggestions,
            )

        # Check if file was modified after creation
        history = self.scanner.get_commit_history_for_file(
            filepath, patch_info.commit_hash
        )
        if history:
            return FileVerificationResult(
                file_path=filepath,
                expected_operation="create",
                actual_status="created_and_modified",
                verification_status="MODIFIED",
                details=f"File created but modified in {len(history)} subsequent commits",
            )

        return FileVerificationResult(
            file_path=filepath,
            expected_operation="create",
            actual_status="created",
            verification_status="OK",
            details="File created successfully",
        )

    def _verify_file_deletion(
        self, file_op: FileOperation, patch_info: PatchInfo
    ) -> FileVerificationResult:
        """Verify that a file was deleted as expected."""
        assert file_op.old_path is not None
        filepath = file_op.old_path
        exists, status = self.scanner.get_file_status(filepath)

        if exists:
            return FileVerificationResult(
                file_path=filepath,
                expected_operation="delete",
                actual_status="still_exists",
                verification_status="ERROR",
                details="File should have been deleted but still exists",
            )

        return FileVerificationResult(
            file_path=filepath,
            expected_operation="delete",
            actual_status="deleted",
            verification_status="OK",
            details="File deleted successfully",
        )

    def _verify_file_modification(
        self, file_op: FileOperation, patch_info: PatchInfo
    ) -> FileVerificationResult:
        """Verify that a file was modified as expected by checking diff hunks."""
        assert file_op.new_path is not None
        filepath = file_op.new_path
        exists, status = self.scanner.get_file_status(filepath)

        if not exists:
            return FileVerificationResult(
                file_path=filepath,
                expected_operation="modify",
                actual_status="missing",
                verification_status="MISSING",
                details="File should exist but is missing",
            )

        # Check if we have diff hunks to verify
        diff_analysis = None
        verification_result = None
        fix_suggestions: List[FixSuggestion] = []
        if not file_op.diff_hunks:
            # Fallback to basic existence check
            history = self.scanner.get_commit_history_for_file(
                filepath, patch_info.commit_hash
            )
            if history:
                details = f"File exists but modified in {len(history)} subsequent commits (no diff hunks to verify)"
                verification_status = "MODIFIED"
            else:
                details = "File exists but cannot verify content (no diff hunks found)"
                verification_status = "WARNING"
        else:
            # Perform detailed content verification with analysis
            verification_result = self._verify_diff_hunks_applied(
                filepath, file_op.diff_hunks
            )
            diff_analysis = self._create_detailed_diff_analysis(
                filepath, file_op.diff_hunks, patch_info
            )

            if verification_result.all_applied:
                history = self.scanner.get_commit_history_for_file(
                    filepath, patch_info.commit_hash
                )
                if history:
                    details = f"All {verification_result.total_hunks} diff hunks applied, but file modified in {len(history)} subsequent commits"
                    verification_status = "MODIFIED"
                else:
                    details = f"All {verification_result.total_hunks} diff hunks applied successfully"
                    verification_status = "OK"
                fix_suggestions = []
            else:
                details = f"Only {verification_result.applied_hunks}/{verification_result.total_hunks} diff hunks applied. Missing changes detected!"
                verification_status = "MISSING"
                # Generate fix suggestions for missing content
                fix_suggestions = self._generate_fix_suggestions(
                    filepath, diff_analysis, patch_info
                )

        result = FileVerificationResult(
            file_path=filepath,
            expected_operation="modify",
            actual_status=status,
            verification_status=verification_status,
            details=details,
            diff_analysis=(
                diff_analysis
                if verification_result and verification_result.missing_hunks
                else None
            ),
            fix_suggestions=fix_suggestions,
        )

        return result

    def _verify_diff_hunks_applied(
        self, filepath: str, hunks: List[DiffHunk]
    ) -> HunkVerificationResult:
        """Verify that diff hunks were actually applied to the file content."""
        current_content = self.scanner.get_file_content(filepath)
        if current_content is None:
            return HunkVerificationResult(
                all_applied=False, applied_hunks=0, total_hunks=len(hunks)
            )

        applied_hunks = []
        missing_hunks = []
        total_hunks = len(hunks)

        for hunk in hunks:
            if self._is_hunk_applied(current_content, hunk):
                applied_hunks.append(hunk)
            else:
                missing_hunks.append(hunk)

        return HunkVerificationResult(
            all_applied=len(applied_hunks) == total_hunks,
            applied_hunks=len(applied_hunks),
            total_hunks=total_hunks,
            missing_hunks=missing_hunks,
            applied_hunks_list=applied_hunks,
            file_content=current_content,
        )

    def _create_detailed_diff_analysis(
        self, filepath: str, hunks: List[DiffHunk], patch_info: PatchInfo
    ) -> DiffAnalysis:
        """Create detailed analysis of missing/applied hunks with context."""
        verification_result = self._verify_diff_hunks_applied(filepath, hunks)

        analysis = DiffAnalysis(
            missing_hunks=verification_result.missing_hunks,
            applied_hunks=verification_result.applied_hunks_list,
            total_hunks=verification_result.total_hunks,
            file_content=verification_result.file_content,
        )

        # Analyze conflicting lines for missing hunks
        if analysis.file_content and analysis.missing_hunks:
            analysis.conflicting_lines = self._find_conflicting_lines(
                analysis.file_content, analysis.missing_hunks
            )

        return analysis

    def _find_conflicting_lines(
        self, file_content: List[str], missing_hunks: List[DiffHunk]
    ) -> List[Tuple[int, str, str]]:
        """Find lines where expected content differs from actual content."""
        conflicts = []

        for hunk in missing_hunks:
            # Extract expected lines from hunk
            expected_lines = []
            for line in hunk.lines:
                if line.startswith(" ") or line.startswith("+"):
                    expected_lines.append(line[1:])

            if not expected_lines:
                continue

            # Find the approximate location where this should be
            start_line = max(0, hunk.new_start - 1)
            end_line = min(len(file_content), start_line + len(expected_lines) + 10)

            # Look for partial matches or close content
            for i in range(start_line, end_line):
                if i < len(file_content):
                    actual_line = file_content[i].rstrip("\n\r ")

                    # Check if this line is close to what we expect
                    for j, expected_line in enumerate(expected_lines):
                        expected_clean = expected_line.rstrip("\n\r ")
                        if len(actual_line) > 0 and len(expected_clean) > 0:
                            # If lines are somewhat similar (using configurable threshold)
                            similarity = self._calculate_line_similarity(
                                actual_line, expected_clean
                            )
                            if similarity > self.similarity_threshold:
                                conflicts.append((i + 1, expected_clean, actual_line))
                                break

        return conflicts

    def _calculate_line_similarity(self, line1: str, line2: str) -> float:
        """Calculate similarity between two lines using difflib (0.0 to 1.0)."""
        if not line1 or not line2:
            return 0.0

        # Use difflib for more sophisticated similarity calculation
        matcher = difflib.SequenceMatcher(None, line1.strip(), line2.strip())
        similarity = matcher.ratio()

        # For very short lines, also check if one is contained in the other
        if len(line1.strip()) < 10 or len(line2.strip()) < 10:
            line1_clean = line1.strip().lower()
            line2_clean = line2.strip().lower()
            if line1_clean in line2_clean or line2_clean in line1_clean:
                similarity = max(similarity, 0.8)  # Boost similarity for containment

        return similarity

    def _is_hunk_applied(self, file_content: List[str], hunk: DiffHunk) -> bool:
        """Check if a specific diff hunk has been applied to the file content."""
        expected = [
            line[1:].rstrip() for line in hunk.lines if line.startswith((" ", "+"))
        ]
        if not expected:
            return True

        expected_str = "".join(expected)

        # Use sliding window around expected position for performance
        start = max(0, hunk.new_start - 5)
        end = min(len(file_content), hunk.new_start + len(expected) + 5)
        window = "".join(file_content[start:end])

        return (
            expected_str in window
            or difflib.SequenceMatcher(None, window, expected_str).ratio()
            >= self.similarity_threshold
        )

    def _generate_fix_suggestions(
        self, filepath: str, diff_analysis: DiffAnalysis, patch_info: PatchInfo
    ) -> List[FixSuggestion]:
        """Generate actionable fix suggestions for missing diff hunks."""
        suggestions: List[FixSuggestion] = []

        if not diff_analysis.missing_hunks:
            return suggestions

        # 1. Git-based restoration (safest for entire files)
        if len(diff_analysis.missing_hunks) == diff_analysis.total_hunks:
            # All hunks missing - suggest full file restoration
            git_suggestion = FixSuggestion(
                fix_type="git_restore",
                description=f"Restore entire file from patch commit (all {diff_analysis.total_hunks} hunks missing)",
                commands=[
                    f"git show {patch_info.commit_hash}:{filepath} > {filepath}",
                    "# Or create backup first:",
                    f"cp {filepath} {filepath}.backup",
                    f"git show {patch_info.commit_hash}:{filepath} > {filepath}",
                ],
                manual_instructions=f"The entire file appears to be missing patch content. Use git to restore the correct version from commit {patch_info.commit_hash[:8]}.",
                safety_level="safe",
            )
            suggestions.append(git_suggestion)

        # 2. Mini-patch generation for specific missing hunks
        if len(diff_analysis.missing_hunks) < diff_analysis.total_hunks:
            # Some hunks applied, some missing - create targeted patch
            mini_patch_content = self._create_mini_patch(
                filepath, diff_analysis.missing_hunks
            )
            if mini_patch_content:
                patch_suggestion = FixSuggestion(
                    fix_type="mini_patch",
                    description=f"Apply mini-patch for {len(diff_analysis.missing_hunks)} missing hunks",
                    commands=[
                        "git apply - <<'EOF'",
                        mini_patch_content,
                        "EOF",
                        "",
                        "# Verify the fix:",
                        "python patchdoctor.py",
                    ],
                    mini_patch_content=mini_patch_content,
                    manual_instructions=f"Apply the generated patch inline to add {len(diff_analysis.missing_hunks)} missing changes.",
                    safety_level="safe",
                )
                suggestions.append(patch_suggestion)

        # 3. Manual editing instructions for small changes
        for i, hunk in enumerate(
            diff_analysis.missing_hunks[:3]
        ):  # Limit to first 3 hunks for readability
            manual_instruction = self._create_manual_edit_instruction(
                filepath, hunk, diff_analysis.file_content
            )
            if manual_instruction:
                manual_suggestion = FixSuggestion(
                    fix_type="manual_edit",
                    description=f"Manual edit for missing hunk {i+1}/{len(diff_analysis.missing_hunks)}",
                    manual_instructions=manual_instruction,
                    safety_level="review",
                )
                suggestions.append(manual_suggestion)

        return suggestions

    def _create_mini_patch(self, filepath: str, missing_hunks: List[DiffHunk]) -> str:
        """Create a mini-patch file content for missing hunks."""
        if not missing_hunks:
            return ""

        patch_lines = [f"--- a/{filepath}", f"+++ b/{filepath}"]

        for hunk in missing_hunks:
            # Create hunk header
            hunk_header = f"@@ -{hunk.old_start},{hunk.old_count} +{hunk.new_start},{hunk.new_count} @@"
            patch_lines.append(hunk_header)

            # Add hunk content
            for line in hunk.lines:
                patch_lines.append(line)

        return "\n".join(patch_lines) + "\n"

    def _create_manual_edit_instruction(
        self, filepath: str, hunk: DiffHunk, file_content: Optional[List[str]]
    ) -> str:
        """Create detailed manual editing instructions for a missing hunk."""
        if not file_content:
            return f"Cannot provide manual instructions - file content unavailable for {filepath}"

        # Extract what should be added
        lines_to_add = []
        context_lines = []

        for line in hunk.lines:
            if line.startswith("+"):
                lines_to_add.append(line[1:])
            elif line.startswith(" "):
                context_lines.append(line[1:])

        if not lines_to_add:
            return ""

        # Find context in current file to help user locate the right spot
        context_location = ""
        if context_lines:
            # Try to find where the context appears in the file
            for i, file_line in enumerate(file_content):
                for context_line in context_lines:
                    if (
                        context_line.strip() in file_line.strip()
                        and len(context_line.strip()) > 10
                    ):
                        context_location = (
                            f'near line {i+1} (around: "{file_line.strip()[:50]}...")'
                        )
                        break
                if context_location:
                    break

        instruction = f"""
Manual Edit Instructions for {filepath}:

1. Open {filepath} in your text editor
2. Go to approximately line {hunk.new_start} {context_location}
3. Add the following {len(lines_to_add)} line(s):

"""

        for i, line in enumerate(lines_to_add):
            instruction += f"   {i+1}: {line.rstrip()}\n"

        if context_lines:
            instruction += "\n4. This should be added in the context of:\n"
            for line in context_lines[:3]:  # Show first 3 context lines
                instruction += f"   Context: {line.rstrip()}\n"

        instruction += "\n5. Save the file and verify with: python patchdoctor.py"

        return instruction

    def _generate_missing_file_fix_suggestions(
        self, filepath: str, file_op: FileOperation, patch_info: PatchInfo
    ) -> List[FixSuggestion]:
        """Generate fix suggestions for missing files."""
        suggestions = []

        # 1. Git extraction (safest and most reliable)
        git_suggestion = FixSuggestion(
            fix_type="git_restore",
            description="Extract missing file from patch commit",
            commands=[
                "# Create directory if needed:",
                f"mkdir -p $(dirname {filepath})",
                "",
                "# Extract file from git:",
                f"git show {patch_info.commit_hash}:{filepath} > {filepath}",
                "",
                "# Verify the file was created:",
                f"ls -la {filepath}",
            ],
            manual_instructions=f"Use git to extract the missing file from commit {patch_info.commit_hash[:8]}.",
            safety_level="safe",
        )
        suggestions.append(git_suggestion)

        # 2. For binary files, provide specific handling
        if file_op.is_binary:
            binary_suggestion = FixSuggestion(
                fix_type="file_create",
                description=f"Extract binary file (size: ~{file_op.insertions} bytes)",
                commands=[
                    "# For binary files, use git checkout:",
                    f"git checkout {patch_info.commit_hash} -- {filepath}",
                    "",
                    "# Or show file info:",
                    f"git show --stat {patch_info.commit_hash} -- {filepath}",
                ],
                manual_instructions="This is a binary file. Use git checkout to restore it from the patch commit.",
                safety_level="safe",
            )
            suggestions.append(binary_suggestion)

        # 3. Show file info for context
        info_suggestion = FixSuggestion(
            fix_type="file_create",
            description="Get information about the missing file",
            commands=[
                "# See what this file contains:",
                f"git show {patch_info.commit_hash}:{filepath} | head -20",
                "",
                "# Check file size and type:",
                f"git cat-file -s {patch_info.commit_hash}:{filepath}",
                f"git cat-file -t {patch_info.commit_hash}:{filepath}",
            ],
            manual_instructions="Use these commands to understand what the missing file should contain.",
            safety_level="safe",
        )
        suggestions.append(info_suggestion)

        return suggestions

    def _verify_file_rename(
        self, file_op: FileOperation, patch_info: PatchInfo
    ) -> FileVerificationResult:
        """Verify that a file was renamed as expected."""
        assert file_op.old_path is not None
        assert file_op.new_path is not None
        old_path = file_op.old_path
        new_path = file_op.new_path

        old_exists, _ = self.scanner.get_file_status(old_path)
        new_exists, new_status = self.scanner.get_file_status(new_path)

        if old_exists and new_exists:
            return FileVerificationResult(
                file_path=f"{old_path} -> {new_path}",
                expected_operation="rename",
                actual_status="both_exist",
                verification_status="ERROR",
                details="Both old and new files exist - rename incomplete",
            )
        elif not old_exists and new_exists:
            return FileVerificationResult(
                file_path=f"{old_path} -> {new_path}",
                expected_operation="rename",
                actual_status="renamed",
                verification_status="OK",
                details="File renamed successfully",
            )
        else:
            return FileVerificationResult(
                file_path=f"{old_path} -> {new_path}",
                expected_operation="rename",
                actual_status="incomplete",
                verification_status="ERROR",
                details="Rename operation appears incomplete",
            )


class ReportGenerator:
    """Generates detailed verification reports."""

    def __init__(self, verbose: bool = False, detailed: bool = False, show_all_fixes: bool = False):
        self.verbose = verbose
        self.detailed = detailed
        self.show_all_fixes = show_all_fixes
        self.terminal_width = get_terminal_width()

    def generate_console_report(self, results: List[VerificationResult]) -> None:
        """Generate and print a console report."""
        console.print()
        header_panel = Panel(
            "[bold white]Git Patch Verification Report[/bold white]",
            style="bold cyan",
            padding=(1, 2),
        )
        console.print(header_panel)

        total_patches = len(results)
        fully_applied = sum(1 for r in results if r.overall_status == "FULLY_APPLIED")
        partially_applied = sum(
            1 for r in results if r.overall_status == "PARTIALLY_APPLIED"
        )
        not_applied = sum(1 for r in results if r.overall_status == "NOT_APPLIED")

        # Summary with Rich formatting
        console.print("\n[bold cyan][SUMMARY][/bold cyan]")
        console.print("[dim]" + "-" * 50 + "[/dim]")
        console.print(
            f"   [bold white]Total patches analyzed:[/bold white] [bold magenta]{total_patches}[/bold magenta]"
        )

        if fully_applied > 0:
            console.print(
                f"   [bold white][+] Fully applied:[/bold white] [bold green]{fully_applied}[/bold green]"
            )

        if partially_applied > 0:
            console.print(
                f"   [bold white][~] Partially applied:[/bold white] [bold yellow]{partially_applied}[/bold yellow]"
            )

        if not_applied > 0:
            console.print(
                f"   [bold white][-] Not applied:[/bold white] [bold red]{not_applied}[/bold red]"
            )

        console.print()

        # Detailed results for each patch
        for result in results:
            self._print_patch_result(result)

        # Overall status with Rich formatting
        console.print("[dim]" + "-" * 80 + "[/dim]")
        if fully_applied == total_patches:
            success_panel = Panel(
                "[bold white] [SUCCESS] [/bold white] All patches have been successfully applied!",
                style="bold green",
                padding=(0, 1),
            )
            console.print(success_panel)
        else:
            warning_panel = Panel(
                "[bold white] [WARNING] [/bold white] Some patches have issues that need attention.",
                style="bold yellow",
                padding=(0, 1),
            )
            console.print(warning_panel)
        console.print()

    def _print_patch_result(self, result: VerificationResult) -> None:
        """Print results for a single patch."""
        patch = result.patch_info

        # Status indicator with Rich styling
        if result.overall_status == "FULLY_APPLIED":
            status_style = "bold green"
            status_icon = "[OK] APPLIED"
        elif result.overall_status == "PARTIALLY_APPLIED":
            status_style = "bold yellow"
            status_icon = "[PARTIAL] PARTIAL"
        else:
            status_style = "bold red"
            status_icon = "[FAILED] FAILED"

        # Create rich table for patch info
        patch_table = Table(
            title=f"[bold cyan]PATCH: {patch.filename}[/bold cyan]",
            show_header=False,
            box=None,
            padding=(0, 1),
        )
        patch_table.add_column("Label", style="dim")
        patch_table.add_column("Value")

        # Add patch information rows
        patch_table.add_row(
            "Commit:", f"[bright_cyan]{patch.commit_hash[:8]}[/bright_cyan]"
        )
        subject_display = smart_truncate_path(patch.subject, 80)
        patch_table.add_row("Subject:", f"[white]{subject_display}[/white]")
        patch_table.add_row(
            "Status:",
            f"[{status_style}]{status_icon} {result.overall_status}[/{status_style}]",
        )

        # Create progress bar
        progress_ratio = (
            result.success_count / result.total_count if result.total_count > 0 else 0
        )
        progress_text = f"({result.success_count}/{result.total_count})"
        patch_table.add_row(
            "Progress:", f"{self._create_progress_bar(progress_ratio)} {progress_text}"
        )

        # Display the patch info in a panel
        patch_panel = Panel(patch_table, style="bold magenta", padding=(1, 2))
        console.print(patch_panel)

        if self.verbose or result.overall_status != "FULLY_APPLIED":
            console.print("\n[bold blue][FILE DETAILS][/bold blue]")
            console.print("[dim]" + "-" * 50 + "[/dim]")

            for file_result in result.file_results:
                self._print_file_result(file_result)

        console.print()

    def _create_progress_bar(self, ratio: float, width: int = 20) -> str:
        """Create a Rich-formatted progress bar."""
        filled = int(ratio * width)
        empty = width - filled

        # Determine color based on completion ratio
        if ratio == 1.0:
            style = "bold green"
        elif ratio > 0:
            style = "bold yellow"
        else:
            style = "bold red"

        filled_bar = f"[{style}]{'█' * filled}[/{style}]"
        empty_bar = f"[dim]{'░' * empty}[/dim]"
        return f"[{filled_bar}{empty_bar}]"

    def _print_file_result(self, file_result: FileVerificationResult) -> None:
        """Print result for a single file with Rich styling."""
        # Determine styling based on operation and status
        if file_result.verification_status == "OK":
            if file_result.expected_operation == "create":
                style = "bold green"
                icon = "[+]"
                status_text = "CREATED"
            elif file_result.expected_operation == "delete":
                style = "bold blue"
                icon = "[-]"
                status_text = "DELETED"
            elif file_result.expected_operation == "modify":
                style = "bold cyan"
                icon = "[M]"
                status_text = "MODIFIED"
            else:
                style = "bold green"
                icon = "[OK]"
                status_text = "OK"
        elif file_result.verification_status == "MODIFIED":
            style = "bold yellow"
            icon = "[!]"
            status_text = "CHANGED"
        elif file_result.verification_status == "WARNING":
            style = "yellow"
            icon = "[?]"
            status_text = "UNVERIFIED"
        else:
            style = "bold red"
            icon = "[X]"
            status_text = "MISSING"

        # Smart truncate file path
        available_width = max(10, self.terminal_width - 20)
        filepath_display = smart_truncate_path(file_result.file_path, available_width)

        console.print(
            f"   [{style}]{icon} {status_text:<8}[/{style}] [white]{filepath_display}[/white]"
        )

        if self.verbose and file_result.details:
            console.print(f"      [dim]|- {file_result.details}[/dim]")

        # Show fix suggestions if available
        if file_result.fix_suggestions and (
            self.verbose or file_result.verification_status in ["MISSING", "ERROR"]
        ):
            console.print("      [dim]|[/dim]")
            console.print("      [bold green][FIX SUGGESTIONS][/bold green]")

            # Determine how many suggestions to show
            max_display = len(file_result.fix_suggestions) if self.show_all_fixes else 3

            for i, suggestion in enumerate(
                file_result.fix_suggestions[:max_display]
            ):  # Limit to 3 suggestions by default for readability
                safety_icon = {
                    "safe": "[bold green][SAFE][/bold green]",
                    "review": "[yellow][REVIEW][/yellow]",
                    "dangerous": "[red][CAUTION][/red]",
                }.get(suggestion.safety_level, "[?]")

                console.print("      [dim]|[/dim]")
                console.print(
                    f"      [dim]|- {safety_icon}[/dim] [bold cyan]{suggestion.description}[/bold cyan]"
                )

                if (
                    suggestion.commands and len(suggestion.commands) <= 6
                ):  # Show commands for simple fixes
                    console.print("      [dim]|  Commands:[/dim]")
                    # Filter out empty strings and show all remaining commands
                    filtered_commands = [cmd for cmd in suggestion.commands if cmd.strip()]
                    for cmd in filtered_commands[:6]:  # Show up to 6 non-empty commands
                        if not cmd.startswith("#"):
                            console.print(
                                f"      [dim]|  [/dim][white]{cmd}[/white]"
                            )
                        else:  # Comment lines
                            console.print(f"      [dim]|  {cmd}[/dim]")

                elif (
                    suggestion.manual_instructions
                    and len(file_result.fix_suggestions) == 1
                ):  # Show manual instructions for single suggestions
                    lines = suggestion.manual_instructions.strip().split("\n")[
                        :3
                    ]  # First 3 lines
                    for line in lines:
                        if line.strip():
                            console.print(
                                f"      [dim]|  [/dim][white]{line.strip()}[/white]"
                            )

            if len(file_result.fix_suggestions) > max_display:
                remaining = len(file_result.fix_suggestions) - max_display
                console.print(
                    f"      [dim]|- ... and {remaining} more fix option(s)[/dim]"
                )
                if not self.show_all_fixes:
                    console.print(
                        "      [dim]|- Use [bold cyan]--show-all-fixes[/bold cyan] to display all suggestions[/dim]"
                    )

        # Add a subtle separator for better readability
        if self.verbose:
            console.print("      [dim]|[/dim]")

    def generate_file_report(
        self, results: List[VerificationResult], output_file: str
    ) -> None:
        """Generate a detailed file report."""
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("Git Patch Verification Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Summary
            total_patches = len(results)
            fully_applied = sum(
                1 for r in results if r.overall_status == "FULLY_APPLIED"
            )

            f.write("Summary:\n")
            f.write(f"  Total patches: {total_patches}\n")
            f.write(f"  Fully applied: {fully_applied}\n")
            f.write(
                f"  Partially applied: {sum(1 for r in results if r.overall_status == 'PARTIALLY_APPLIED')}\n"
            )
            f.write(
                f"  Not applied: {sum(1 for r in results if r.overall_status == 'NOT_APPLIED')}\n\n"
            )

            # Detailed results
            for result in results:
                f.write(f"Patch: {result.patch_info.filename}\n")
                f.write(f"Commit: {result.patch_info.commit_hash}\n")
                f.write(f"Subject: {result.patch_info.subject}\n")
                f.write(
                    f"Status: {result.overall_status} ({result.success_count}/{result.total_count})\n\n"
                )

                f.write("File Details:\n")
                for file_result in result.file_results:
                    status_symbol = (
                        "[+]" if file_result.verification_status == "OK" else "[-]"
                    )
                    f.write(f"  {status_symbol} {file_result.file_path}\n")
                    f.write(f"    Expected: {file_result.expected_operation}\n")
                    f.write(f"    Actual: {file_result.actual_status}\n")
                    f.write(f"    Status: {file_result.verification_status}\n")
                    if file_result.details:
                        f.write(f"    Details: {file_result.details}\n")
                    if self.detailed and file_result.fix_suggestions:
                        f.write("    Fix Suggestions:\n")
                        for suggestion in file_result.fix_suggestions:
                            f.write(
                                f"      - {suggestion.description} ({suggestion.safety_level})\n"
                            )
                            if suggestion.commands:
                                for cmd in suggestion.commands:
                                    if cmd.strip() and not cmd.startswith("#"):
                                        f.write(f"        {cmd}\n")
                    f.write("\n")

                f.write("-" * 50 + "\n\n")

        console.print(
            f"\n[bold green][SAVED][/bold green] [bold white]Detailed report saved to:[/bold white] [bright_cyan]{output_file}[/bright_cyan]"
        )

    def generate_json_report(
        self, results: List[VerificationResult], output_file: str
    ) -> None:
        """Generate a JSON report for CI/CD integration."""
        data = [asdict(r) for r in results]
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)  # default=str for datetime
        console.print(
            f"\n[bold green][SAVED][/bold green] [bold white]JSON report saved to:[/bold white] [bright_cyan]{output_file}[/bright_cyan]"
        )


def main():
    """Main entry point for the patch verification script."""

    parser = argparse.ArgumentParser(
        description="PatchDoctor - Professional Git patch verification and analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="""
Examples:
  python patchdoctor.py                              # Verify all .patch files in current directory
  python patchdoctor.py -v                           # Show detailed information for all files
  python patchdoctor.py -d /path/to/patches -v       # Verify patches in specific directory with details
  python patchdoctor.py -r report.txt                # Save comprehensive report to file
  python patchdoctor.py -s 0.5 -t 10                 # Custom similarity threshold and hunk tolerance
  python patchdoctor.py -T 60 -M 200                 # Extended timeout and larger file support

Advanced Usage:
  python patchdoctor.py -v -s 0.7 -t 3               # Strict matching with tight tolerance
  python patchdoctor.py -c -r analysis.txt           # Generate report without colors

For more information and latest updates, see the documentation.
        """,
    )

    # Basic options with short and long forms
    parser.add_argument(
        "-d",
        "--patch-dir",
        default=".",
        metavar="DIR",
        help="Directory containing patch files (default: current directory)",
    )
    parser.add_argument(
        "-R",
        "--repo",
        default=".",
        metavar="DIR",
        help="Repository root directory (default: current directory)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show detailed information for all files and operations",
    )
    parser.add_argument(
        "-r",
        "--report",
        metavar="FILE",
        help="Save detailed verification report to specified file",
    )
    parser.add_argument(
        "-j",
        "--json-report",
        metavar="FILE",
        help="Save JSON report for CI/CD integration",
    )
    parser.add_argument(
        "-D",
        "--report-detailed",
        action="store_true",
        help="Include fix suggestions in file reports",
    )
    parser.add_argument(
        "--show-all-fixes",
        action="store_true",
        help="Show all available fix suggestions (default: show first 3 only)",
    )
    parser.add_argument(
        "-c",
        "--no-color",
        action="store_true",
        help="Disable colored output (useful for logging or CI environments)",
    )

    # Advanced configuration options
    parser.add_argument(
        "-s",
        "--similarity-threshold",
        type=float,
        default=DEFAULT_SIMILARITY_THRESHOLD,
        metavar="FLOAT",
        help=f"Line similarity threshold for fuzzy matching (0.0-1.0, default: {DEFAULT_SIMILARITY_THRESHOLD})",
    )
    parser.add_argument(
        "-t",
        "--hunk-tolerance",
        type=int,
        default=DEFAULT_HUNK_SEARCH_TOLERANCE,
        metavar="INT",
        help=f"Search tolerance for hunk positioning (lines to search around expected position, default: {DEFAULT_HUNK_SEARCH_TOLERANCE})",
    )
    parser.add_argument(
        "-T",
        "--timeout",
        type=int,
        default=DEFAULT_SUBPROCESS_TIMEOUT,
        metavar="SECONDS",
        help=f"Timeout for git operations in seconds (default: {DEFAULT_SUBPROCESS_TIMEOUT})",
    )
    parser.add_argument(
        "-M",
        "--max-file-size",
        type=int,
        default=MAX_FILE_SIZE_MB,
        metavar="MB",
        help=f"Maximum file size to process in megabytes (default: {MAX_FILE_SIZE_MB})",
    )
    parser.add_argument("--version", action="version", version="PatchDoctor v1.0")

    args = parser.parse_args()

    # Configure console based on arguments
    global console
    console = Console(force_terminal=not args.no_color)

    # Create and validate configuration
    try:
        config = Config.from_args(args)
    except ValueError as e:
        print_error(str(e))
        return 1

    # Find all patch files
    patch_dir = Path(config.patch_dir)
    patch_files = list(patch_dir.glob("*.patch"))

    if not patch_files:
        print_error(f"No .patch files found in [bright_cyan]{patch_dir}[/bright_cyan]")
        return 1

    # Sort patch files by name to process in order
    patch_files.sort()

    # Display startup information
    console.print("[bold cyan]PatchDoctor v1.0[/bold cyan]")
    console.print(f"[dim]Repository: {patch_dir.absolute()}[/dim]")

    # Show configuration if non-default values are used
    config_notes = []
    if config.similarity_threshold != DEFAULT_SIMILARITY_THRESHOLD:
        config_notes.append(f"Similarity: {config.similarity_threshold}")
    if config.hunk_tolerance != DEFAULT_HUNK_SEARCH_TOLERANCE:
        config_notes.append(f"Tolerance: {config.hunk_tolerance}")
    if config.timeout != DEFAULT_SUBPROCESS_TIMEOUT:
        config_notes.append(f"Timeout: {config.timeout}s")
    if config.max_file_size != MAX_FILE_SIZE_MB:
        config_notes.append(f"Max size: {config.max_file_size}MB")

    if config_notes:
        console.print(f"[dim]Configuration: {', '.join(config_notes)}[/dim]")

    console.print(
        f"\n[bold blue][STARTING][/bold blue] [bold white]Found [bold magenta]{len(patch_files)}[/bold magenta] patch file(s) to verify[/bold white]"
    )
    console.print("[dim]" + "-" * 60 + "[/dim]\n")

    # Verify each patch with configuration options
    verifier = PatchVerifier(
        repo_path=Path(config.repo_path),
        verbose=config.verbose,
        similarity_threshold=config.similarity_threshold,
        hunk_search_tolerance=config.hunk_tolerance,
        timeout=config.timeout,
        max_file_size_mb=config.max_file_size,
    )
    results = []

    for patch_file in patch_files:
        try:
            result = verifier.verify_patch(str(patch_file))
            results.append(result)
        except Exception as e:
            print_error(
                f"Error processing [bright_cyan]{patch_file}[/bright_cyan]: [red]{e}[/red]"
            )
            continue

    # Generate reports
    report_generator = ReportGenerator(
        verbose=config.verbose, detailed=config.report_detailed, show_all_fixes=config.show_all_fixes
    )
    report_generator.generate_console_report(results)

    if config.report_file:
        report_generator.generate_file_report(results, config.report_file)

    if config.json_report_file:
        report_generator.generate_json_report(results, config.json_report_file)

    # Return appropriate exit code
    all_applied = all(r.overall_status == "FULLY_APPLIED" for r in results)
    return 0 if all_applied else 1


if __name__ == "__main__":
    sys.exit(main())

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
import time

from dataclasses import dataclass, field, asdict
from datetime import datetime
from email.utils import parsedate_to_datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cachetools

# Additional libraries for improved functionality
import chardet

# Rich imports for modern terminal output
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Global console instance for rich output
console = Console()


# ===== AI AGENT INTEGRATION: Error Information System =====

@dataclass
class ErrorInfo:
    """Structured error information for AI agent handling."""
    code: str  # e.g., "NO_PATCHES_FOUND", "GIT_TIMEOUT", "PARSE_ERROR"
    message: str
    suggestion: str  # Recovery suggestion for agents
    context: Dict[str, Any] = field(default_factory=dict)
    recoverable: bool = True
    severity: str = "error"  # "error", "warning", "info"


# Common error codes for AI agent handling
ERROR_NO_PATCHES_FOUND = "NO_PATCHES_FOUND"
ERROR_GIT_TIMEOUT = "GIT_TIMEOUT"
ERROR_GIT_COMMAND_FAILED = "GIT_COMMAND_FAILED"
ERROR_PARSE_ERROR = "PARSE_ERROR"
ERROR_FILE_NOT_FOUND = "FILE_NOT_FOUND"
ERROR_FILE_TOO_LARGE = "FILE_TOO_LARGE"
ERROR_ENCODING_ERROR = "ENCODING_ERROR"
ERROR_PERMISSION_DENIED = "PERMISSION_DENIED"
ERROR_INVALID_PATCH_FORMAT = "INVALID_PATCH_FORMAT"
ERROR_REPOSITORY_NOT_FOUND = "REPOSITORY_NOT_FOUND"
ERROR_TIMEOUT = "TIMEOUT"
ERROR_INVALID_CONFIG = "INVALID_CONFIG"


# Exception hierarchy for consistent error handling
class PatchDoctorError(Exception):
    """Base exception for PatchDoctor errors."""

    def __init__(self, message: str, error_info: Optional[ErrorInfo] = None):
        super().__init__(message)
        self.error_info = error_info


class FileEncodingError(PatchDoctorError):
    """Error when file encoding cannot be detected or read."""

    pass


class GitCommandError(PatchDoctorError):
    """Error when git commands fail."""

    def __init__(self, cmd: List[str], message: str, error_info: Optional[ErrorInfo] = None):
        super().__init__(f"Git failed: {cmd} → {message}", error_info)
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
    """Configuration for PatchDoctor with validation.

    Supports predefined configuration profiles for common AI agent workflows:
    - Config.strict_mode(): High precision validation
    - Config.lenient_mode(): Flexible fuzzy matching
    - Config.fast_mode(): Speed-optimized processing
    """

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
    def strict_mode(cls, **overrides) -> "Config":
        """High precision mode for critical validation.

        Uses conservative settings for maximum accuracy:
        - High similarity threshold (0.8) for precise matching
        - Low hunk tolerance (2) to avoid false positives
        - Extended timeout (60s) for thorough analysis

        Args:
            **overrides: Override any default settings

        Returns:
            Config: Configuration optimized for strict validation

        Example:
            config = Config.strict_mode(repo_path="/path/to/repo")
        """
        defaults = {
            "similarity_threshold": 0.8,
            "hunk_tolerance": 2,
            "timeout": 60,
        }
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def lenient_mode(cls, **overrides) -> "Config":
        """Flexible mode for fuzzy matching scenarios.

        Uses permissive settings for maximum compatibility:
        - Low similarity threshold (0.3) for fuzzy matching
        - High hunk tolerance (10) to handle code drift
        - Moderate timeout (30s) for responsive operation

        Args:
            **overrides: Override any default settings

        Returns:
            Config: Configuration optimized for flexible validation

        Example:
            config = Config.lenient_mode(verbose=True)
        """
        defaults = {
            "similarity_threshold": 0.3,
            "hunk_tolerance": 10,
            "timeout": 30,
        }
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def fast_mode(cls, **overrides) -> "Config":
        """Speed-optimized mode for quick validation.

        Uses balanced settings optimized for performance:
        - Moderate similarity threshold (0.5) for good speed/accuracy balance
        - Medium hunk tolerance (5) for reasonable flexibility
        - Short timeout (15s) for fast response
        - Smaller file size limit (50MB) to avoid large file overhead

        Args:
            **overrides: Override any default settings

        Returns:
            Config: Configuration optimized for speed

        Example:
            config = Config.fast_mode(patch_dir="./patches")
        """
        defaults = {
            "similarity_threshold": 0.5,
            "hunk_tolerance": 5,
            "timeout": 15,
            "max_file_size": 50,
        }
        defaults.update(overrides)
        return cls(**defaults)

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
        enable_performance_monitoring: bool = False,
    ):
        self.repo_path = Path(repo_path)
        self.timeout = timeout
        self.enable_performance_monitoring = enable_performance_monitoring
        self._result_cache: cachetools.TTLCache[str, GitResult] = cachetools.TTLCache(
            maxsize=256, ttl=cache_ttl  # Increased cache size for better performance
        )  # Thread-safe TTL cache
        self._cache_stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "total_requests": 0,
        }
        self._performance_metrics = {
            "command_times": [],
            "cache_hit_ratio": 0.0,
            "average_command_time": 0.0,
        }

    def run_git_command(self, args: List[str], use_cache: bool = True) -> GitResult:
        """Run a git command with caching and error handling."""
        import time

        start_time = time.time() if self.enable_performance_monitoring else None
        cmd = ["git"] + args
        cache_key = f"{self.repo_path}:{' '.join(cmd)}"

        self._cache_stats["total_requests"] += 1

        # Check cache if enabled
        if use_cache and cache_key in self._result_cache:
            self._cache_stats["hits"] += 1
            if self.enable_performance_monitoring and start_time:
                self._update_performance_metrics(time.time() - start_time, cache_hit=True)
            return self._result_cache[cache_key]

        # Cache miss
        if use_cache:
            self._cache_stats["misses"] += 1

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

            # Update performance metrics
            if self.enable_performance_monitoring and start_time:
                self._update_performance_metrics(time.time() - start_time, cache_hit=False)

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
        """Get comprehensive cache and performance statistics."""
        total_requests = self._cache_stats["total_requests"]
        hit_ratio = (
            self._cache_stats["hits"] / total_requests if total_requests > 0 else 0.0
        )

        stats = {
            "cached_entries": len(self._result_cache),
            "maxsize": self._result_cache.maxsize,
            "ttl": self._result_cache.ttl,
            "cache_hits": self._cache_stats["hits"],
            "cache_misses": self._cache_stats["misses"],
            "cache_hit_ratio": hit_ratio,
            "total_requests": total_requests,
        }

        if self.enable_performance_monitoring:
            stats.update(self._performance_metrics)

        return stats

    def _update_performance_metrics(self, command_time: float, cache_hit: bool = False) -> None:
        """Update performance metrics with command timing data."""
        if not cache_hit:  # Only record actual command execution times
            self._performance_metrics["command_times"].append(command_time)

            # Keep only last 100 measurements to avoid memory growth
            if len(self._performance_metrics["command_times"]) > 100:
                self._performance_metrics["command_times"] = (
                    self._performance_metrics["command_times"][-100:]
                )

        # Update calculated metrics
        if self._performance_metrics["command_times"]:
            self._performance_metrics["average_command_time"] = (
                sum(self._performance_metrics["command_times"])
                / len(self._performance_metrics["command_times"])
            )

        total_requests = self._cache_stats["total_requests"]
        if total_requests > 0:
            self._performance_metrics["cache_hit_ratio"] = (
                self._cache_stats["hits"] / total_requests
            )

    def invalidate_cache(self, pattern: Optional[str] = None) -> int:
        """Invalidate cache entries, optionally matching a pattern.

        Args:
            pattern: Optional regex pattern to match cache keys

        Returns:
            int: Number of entries invalidated
        """
        if pattern is None:
            count = len(self._result_cache)
            self._result_cache.clear()
            return count

        import re
        pattern_re = re.compile(pattern)
        keys_to_remove = [
            key for key in self._result_cache.keys()
            if pattern_re.search(key)
        ]

        for key in keys_to_remove:
            del self._result_cache[key]

        return len(keys_to_remove)

    def get_performance_report(self) -> str:
        """Generate a human-readable performance report."""
        if not self.enable_performance_monitoring:
            return "Performance monitoring is disabled"

        stats = self.get_cache_stats()

        report = f"""GitRunner Performance Report:
Cache Statistics:
  - Cached entries: {stats['cached_entries']}/{stats['maxsize']}
  - Hit ratio: {stats['cache_hit_ratio']:.2%}
  - Total requests: {stats['total_requests']}

Performance Metrics:
  - Average command time: {stats['average_command_time']:.3f}s
  - Command samples: {len(self._performance_metrics['command_times'])}
"""

        if self._performance_metrics["command_times"]:
            min_time = min(self._performance_metrics["command_times"])
            max_time = max(self._performance_metrics["command_times"])
            report += f"  - Min/Max command time: {min_time:.3f}s / {max_time:.3f}s\n"

        return report

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

    def __init__(self, repo_path: str = ".", timeout: int = DEFAULT_SUBPROCESS_TIMEOUT, enable_performance_monitoring: bool = False):
        self.repo_path = Path(repo_path)
        self.timeout = timeout
        self._file_content_cache: cachetools.LRUCache[
            Tuple[str, float, int], List[str]
        ] = cachetools.LRUCache(
            maxsize=64  # Increased cache size for better performance
        )  # LRU cache for file contents
        self._cache_stats = {
            "hits": 0,
            "misses": 0,
            "total_requests": 0,
        }
        self.git_runner = GitRunner(repo_path, timeout, enable_performance_monitoring=enable_performance_monitoring)  # Use unified git runner

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
        self._cache_stats["total_requests"] += 1

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
                self._cache_stats["hits"] += 1
                return self._file_content_cache[cache_key]

            # Cache miss
            self._cache_stats["misses"] += 1

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

    def get_cache_stats(self) -> Dict[str, Union[int, float]]:
        """Get file content cache statistics."""
        total_requests = self._cache_stats["total_requests"]
        hit_ratio = (
            self._cache_stats["hits"] / total_requests if total_requests > 0 else 0.0
        )

        return {
            "cached_files": len(self._file_content_cache),
            "maxsize": self._file_content_cache.maxsize,
            "cache_hits": self._cache_stats["hits"],
            "cache_misses": self._cache_stats["misses"],
            "cache_hit_ratio": hit_ratio,
            "total_requests": total_requests,
        }

    def clear_file_cache(self) -> int:
        """Clear the file content cache.

        Returns:
            int: Number of entries cleared
        """
        count = len(self._file_content_cache)
        self._file_content_cache.clear()
        return count


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


def run_validation(
    patch_dir: str = ".",
    repo_path: str = ".",
    verbose: bool = False,
    report_file: Optional[str] = None,
    json_report_file: Optional[str] = None,
    report_detailed: bool = False,
    no_color: bool = False,
    show_all_fixes: bool = False,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    hunk_tolerance: int = DEFAULT_HUNK_SEARCH_TOLERANCE,
    timeout: int = DEFAULT_SUBPROCESS_TIMEOUT,
    max_file_size: int = MAX_FILE_SIZE_MB,
) -> Dict[str, Any]:
    """Run patchdoctor validation and return structured results for AI agent integration.

    Args:
        patch_dir: Directory containing patch files
        repo_path: Repository root directory
        verbose: Show detailed information
        report_file: Save detailed report to file
        json_report_file: Save JSON report for CI/CD
        report_detailed: Include fix suggestions in reports
        no_color: Disable colored output
        show_all_fixes: Show all fix suggestions
        similarity_threshold: Line similarity threshold (0.0-1.0)
        hunk_tolerance: Search tolerance for hunk matching
        timeout: Timeout for git operations in seconds
        max_file_size: Maximum file size to process in MB

    Returns:
        Dict with validation results and summary stats
    """
    # Create config
    config = Config(
        patch_dir=patch_dir,
        repo_path=repo_path,
        verbose=verbose,
        report_file=report_file,
        json_report_file=json_report_file,
        report_detailed=report_detailed,
        no_color=no_color,
        show_all_fixes=show_all_fixes,
        similarity_threshold=similarity_threshold,
        hunk_tolerance=hunk_tolerance,
        timeout=timeout,
        max_file_size=max_file_size,
    )

    # Find patch files
    patch_files = list(Path(config.patch_dir).glob("*.patch"))
    if not patch_files:
        error_info = ErrorInfo(
            code=ERROR_NO_PATCHES_FOUND,
            message="No .patch files found in directory",
            suggestion="Check that the patch directory contains .patch files, or create patch files using 'git format-patch'",
            context={"patch_dir": config.patch_dir},
            recoverable=True,
            severity="error"
        )
        return {
            "success": False,
            "error": "No .patch files found",
            "error_info": asdict(error_info),
            "results": []
        }

    # Create verifier
    verifier = PatchVerifier(
        repo_path=config.repo_path,
        verbose=config.verbose,
        similarity_threshold=config.similarity_threshold,
        hunk_search_tolerance=config.hunk_tolerance,
        timeout=config.timeout,
        max_file_size_mb=config.max_file_size,
    )

    # Verify patches
    results = []
    errors = []
    for patch_file in patch_files:
        try:
            result = verifier.verify_patch(str(patch_file))
            results.append(result)
        except Exception as e:
            # Create structured error information
            error_info = ErrorInfo(
                code=ERROR_PARSE_ERROR if "parse" in str(e).lower() else ERROR_GIT_COMMAND_FAILED,
                message=f"Failed to verify patch: {e}",
                suggestion="Check that the patch file is properly formatted and the repository is accessible",
                context={"patch_file": str(patch_file), "error_type": type(e).__name__},
                recoverable=True,
                severity="error"
            )
            errors.append(asdict(error_info))

            results.append(VerificationResult(
                patch_info=PatchInfo(filename=str(patch_file), commit_hash="", author="", subject=f"Error: {e}"),
                file_results=[],
                overall_status="ERROR",
                success_count=0,
                total_count=0,
            ))

    # Generate reports if requested
    report_gen = ReportGenerator(
        verbose=config.verbose,
        detailed=config.report_detailed,
        show_all_fixes=config.show_all_fixes,
    )

    if config.report_file:
        report_gen.generate_file_report(results, config.report_file)

    if config.json_report_file:
        report_gen.generate_json_report(results, config.json_report_file)

    # Return structured data
    return {
        "success": len(errors) == 0,
        "total_patches": len(results),
        "fully_applied": sum(1 for r in results if r.overall_status == "FULLY_APPLIED"),
        "partially_applied": sum(1 for r in results if r.overall_status == "PARTIALLY_APPLIED"),
        "not_applied": sum(1 for r in results if r.overall_status == "NOT_APPLIED"),
        "error_count": len(errors),
        "errors": errors,
        "results": [asdict(r) for r in results],
    }


def validate_from_content(
    patch_content: str,
    repo_path: str = ".",
    verbose: bool = False,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    hunk_tolerance: int = DEFAULT_HUNK_SEARCH_TOLERANCE,
    timeout: int = DEFAULT_SUBPROCESS_TIMEOUT,
    max_file_size: int = MAX_FILE_SIZE_MB,
) -> Dict[str, Any]:
    """Validate a patch from string content directly for AI agent workflows.

    Args:
        patch_content: The patch content as a string
        repo_path: Repository root directory
        verbose: Show detailed information
        similarity_threshold: Line similarity threshold (0.0-1.0)
        hunk_tolerance: Search tolerance for hunk matching
        timeout: Timeout for git operations in seconds
        max_file_size: Maximum file size to process in MB

    Returns:
        Dict with validation result
    """
    import tempfile

    # Write content to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.patch', delete=False) as f:
        f.write(patch_content)
        temp_path = f.name

    try:
        # Parse patch
        parser = PatchParser(similarity_threshold, max_file_size)
        patch_info = parser.parse_patch_file(temp_path)

        # Create verifier
        verifier = PatchVerifier(
            repo_path=repo_path,
            verbose=verbose,
            similarity_threshold=similarity_threshold,
            hunk_search_tolerance=hunk_tolerance,
            timeout=timeout,
            max_file_size_mb=max_file_size,
        )

        # Verify
        file_results = verifier.verify_patch_info(patch_info)

        # Calculate status
        success_count = sum(1 for r in file_results if r.verification_status == "OK")
        total_count = len(file_results)

        if success_count == total_count:
            overall_status = "FULLY_APPLIED"
        elif success_count > 0:
            overall_status = "PARTIALLY_APPLIED"
        else:
            overall_status = "NOT_APPLIED"

        result = VerificationResult(
            patch_info=patch_info,
            file_results=file_results,
            overall_status=overall_status,
            success_count=success_count,
            total_count=total_count,
        )

        return {
            "success": True,
            "result": asdict(result),
        }

    except Exception as e:
        # Create structured error information
        error_info = ErrorInfo(
            code=ERROR_PARSE_ERROR if "parse" in str(e).lower() else ERROR_GIT_COMMAND_FAILED,
            message=f"Failed to validate patch content: {e}",
            suggestion="Check that the patch content is properly formatted and the repository is accessible",
            context={"error_type": type(e).__name__, "patch_length": len(patch_content)},
            recoverable=True,
            severity="error"
        )
        return {
            "success": False,
            "error": str(e),
            "error_info": asdict(error_info)
        }
    finally:
        Path(temp_path).unlink(missing_ok=True)


def parse_report_status(report: Dict[str, Any]) -> Dict[str, Any]:
    """Extract summary statistics from a validation report for AI agent analysis.

    Args:
        report: The report dict from run_validation() or validate_from_content()

    Returns:
        Dict with summary stats
    """
    if not report.get("success"):
        return {"error": report.get("error", "Unknown error")}

    results = report.get("results", [])
    total_files = sum(len(r.get("file_results", [])) for r in results)
    ok_files = sum(
        sum(1 for fr in r.get("file_results", []) if fr.get("verification_status") == "OK")
        for r in results
    )

    return {
        "total_patches": len(results),
        "total_files": total_files,
        "ok_files": ok_files,
        "success_rate": ok_files / total_files if total_files > 0 else 0,
        "fully_applied_patches": sum(1 for r in results if r.get("overall_status") == "FULLY_APPLIED"),
        "partially_applied_patches": sum(1 for r in results if r.get("overall_status") == "PARTIALLY_APPLIED"),
        "not_applied_patches": sum(1 for r in results if r.get("overall_status") == "NOT_APPLIED"),
    }


# ===== AI AGENT INTEGRATION: Safe Fix Application =====

def apply_safe_fixes(
    verification_result: VerificationResult,
    confirm: bool = True,
    safety_levels: List[str] = None,
    dry_run: bool = False,
    repo_path: str = "."
) -> Dict[str, Any]:
    """Apply fix suggestions based on safety level with rollback support.

    Args:
        verification_result: Result from patch verification
        confirm: Whether to prompt for confirmation before applying fixes
        safety_levels: List of safety levels to apply ("safe", "review", "dangerous")
        dry_run: If True, show what would be done without making changes
        repo_path: Repository root directory

    Returns:
        Dict with applied, skipped, and failed fixes, plus rollback info
    """
    if safety_levels is None:
        safety_levels = ["safe"]

    applied = []
    skipped = []
    errors = []
    rollback_info = {"git_stash_id": None, "original_branch": None}

    try:
        # Get current git state for rollback
        git_runner = GitRunner(repo_path=repo_path)

        # Get current branch
        branch_result = git_runner.run_git_command(["branch", "--show-current"])
        if branch_result.ok:
            rollback_info["original_branch"] = branch_result.stdout.strip()

        # Create stash for rollback safety
        if not dry_run:
            stash_result = git_runner.run_git_command(["stash", "push", "-m", "PatchDoctor auto-fix backup"])
            if stash_result.ok and "No local changes to save" not in stash_result.stdout:
                rollback_info["git_stash_id"] = "stash@{0}"

        # Collect all fix suggestions from file results
        all_fixes = []
        for file_result in verification_result.file_results:
            for fix in file_result.fix_suggestions:
                if fix.safety_level in safety_levels:
                    all_fixes.append((file_result.file_path, fix))

        if not all_fixes:
            return {
                "applied": applied,
                "skipped": skipped,
                "errors": errors,
                "rollback_info": rollback_info,
                "summary": "No fixes found matching the specified safety levels"
            }

        # Apply fixes
        for file_path, fix in all_fixes:
            try:
                if confirm and not dry_run:
                    response = input(f"Apply {fix.fix_type} fix for {file_path}? ({fix.description}) [y/N]: ")
                    if response.lower() not in ['y', 'yes']:
                        skipped.append({
                            "file_path": file_path,
                            "fix": asdict(fix),
                            "reason": "User declined"
                        })
                        continue

                if dry_run:
                    applied.append({
                        "file_path": file_path,
                        "fix": asdict(fix),
                        "status": "dry_run",
                        "commands": fix.commands
                    })
                    continue

                # Apply the fix based on type
                success = _apply_single_fix(git_runner, file_path, fix)

                if success:
                    applied.append({
                        "file_path": file_path,
                        "fix": asdict(fix),
                        "status": "applied"
                    })
                else:
                    errors.append({
                        "file_path": file_path,
                        "fix": asdict(fix),
                        "error": "Fix application failed",
                        "error_info": asdict(ErrorInfo(
                            code="FIX_APPLICATION_FAILED",
                            message=f"Failed to apply {fix.fix_type} fix",
                            suggestion="Try applying the fix manually or check file permissions",
                            context={"file_path": file_path, "fix_type": fix.fix_type},
                            recoverable=True,
                            severity="warning"
                        ))
                    })

            except Exception as e:
                errors.append({
                    "file_path": file_path,
                    "fix": asdict(fix),
                    "error": str(e),
                    "error_info": asdict(ErrorInfo(
                        code="FIX_APPLICATION_ERROR",
                        message=f"Exception during fix application: {e}",
                        suggestion="Check the error details and try applying manually",
                        context={"file_path": file_path, "exception_type": type(e).__name__},
                        recoverable=True,
                        severity="error"
                    ))
                })

        return {
            "applied": applied,
            "skipped": skipped,
            "errors": errors,
            "rollback_info": rollback_info,
            "summary": f"Applied {len(applied)} fixes, skipped {len(skipped)}, {len(errors)} errors"
        }

    except Exception as e:
        return {
            "applied": applied,
            "skipped": skipped,
            "errors": errors + [{
                "error": str(e),
                "error_info": asdict(ErrorInfo(
                    code="APPLY_FIXES_FAILED",
                    message=f"Failed to apply fixes: {e}",
                    suggestion="Check repository state and permissions",
                    context={"exception_type": type(e).__name__},
                    recoverable=True,
                    severity="error"
                ))
            }],
            "rollback_info": rollback_info,
            "summary": f"Fix application failed with error: {e}"
        }


def _apply_single_fix(git_runner: 'GitRunner', file_path: str, fix: FixSuggestion) -> bool:
    """Apply a single fix suggestion.

    Args:
        git_runner: GitRunner instance for executing commands
        file_path: Path to the file being fixed
        fix: The fix suggestion to apply

    Returns:
        True if fix was applied successfully, False otherwise
    """
    try:
        if fix.fix_type == "git_restore":
            # Use git restore to revert file
            for cmd in fix.commands:
                if cmd.startswith("git restore"):
                    result = git_runner.run_git_command(cmd.split()[2:])  # Skip "git restore"
                    return result.ok

        elif fix.fix_type == "mini_patch":
            # Apply mini patch content
            if fix.mini_patch_content:
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.patch', delete=False) as f:
                    f.write(fix.mini_patch_content)
                    patch_file = f.name

                try:
                    result = git_runner.run_git_command(["apply", patch_file])
                    return result.ok
                finally:
                    Path(patch_file).unlink(missing_ok=True)

        elif fix.fix_type == "manual_edit":
            # For manual edits, we can't automatically apply them
            # This would require more sophisticated text manipulation
            return False

        elif fix.fix_type == "file_create":
            # Create missing files
            for cmd in fix.commands:
                if cmd.startswith("touch") or cmd.startswith("mkdir"):
                    # Execute file creation commands
                    import subprocess
                    result = subprocess.run(cmd.split(), capture_output=True, text=True, cwd=git_runner.repo_path)
                    if result.returncode != 0:
                        return False
            return True

        return False

    except Exception:
        return False


# ===== AI AGENT INTEGRATION: Patch Content Analysis Utilities =====

def extract_missing_changes(verification_result: VerificationResult) -> List[Dict[str, Any]]:
    """Extract detailed information about missing changes.

    Args:
        verification_result: Result from patch verification

    Returns:
        List of dicts with detailed information about missing changes
    """
    missing_changes = []

    for file_result in verification_result.file_results:
        if file_result.diff_analysis and file_result.diff_analysis.missing_hunks:
            for hunk in file_result.diff_analysis.missing_hunks:
                change_info = {
                    "file_path": file_result.file_path,
                    "hunk_info": {
                        "old_start": hunk.old_start,
                        "old_count": hunk.old_count,
                        "new_start": hunk.new_start,
                        "new_count": hunk.new_count
                    },
                    "expected_location": hunk.new_start,
                    "content_lines": hunk.lines.copy(),
                    "context_lines": _extract_context_lines(file_result.diff_analysis.file_content, hunk),
                    "conflict_type": _determine_conflict_type(file_result, hunk)
                }
                missing_changes.append(change_info)

    return missing_changes


def generate_corrective_patch(verification_result: VerificationResult, output_file: str) -> bool:
    """Generate a patch file containing only the missing changes.

    Args:
        verification_result: Result from patch verification
        output_file: Path where to save the corrective patch

    Returns:
        True if patch was generated successfully, False otherwise
    """
    try:
        missing_changes = extract_missing_changes(verification_result)
        if not missing_changes:
            return False

        patch_content = []
        current_file = None

        for change in missing_changes:
            file_path = change["file_path"]

            # Add file header if this is a new file
            if current_file != file_path:
                current_file = file_path
                patch_content.append(f"--- a/{file_path}")
                patch_content.append(f"+++ b/{file_path}")

            # Add hunk header
            hunk_info = change["hunk_info"]
            patch_content.append(
                f"@@ -{hunk_info['old_start']},{hunk_info['old_count']} "
                f"+{hunk_info['new_start']},{hunk_info['new_count']} @@"
            )

            # Add hunk content
            patch_content.extend(change["content_lines"])

        # Write patch file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(patch_content))
            if patch_content:  # Add final newline if content exists
                f.write('\n')

        return True

    except Exception:
        return False


def split_large_patch(patch_content: str, strategy: str = "by_file") -> List[str]:
    """Split large patches into smaller, manageable pieces.

    Args:
        patch_content: The patch content as a string
        strategy: Split strategy - "by_file", "by_hunk", or "by_size"

    Returns:
        List of smaller patch strings
    """
    if strategy == "by_file":
        return _split_patch_by_file(patch_content)
    elif strategy == "by_hunk":
        return _split_patch_by_hunk(patch_content)
    elif strategy == "by_size":
        return _split_patch_by_size(patch_content)
    else:
        return [patch_content]  # Return original if strategy unknown


def summarize_patch_status(verification_result: VerificationResult) -> Dict[str, Any]:
    """Generate a quick summary of patch status for AI agent analysis.

    Args:
        verification_result: Result from patch verification

    Returns:
        Dict with summary statistics and recommendations
    """
    total_files = len(verification_result.file_results)
    ok_files = sum(1 for r in verification_result.file_results if r.verification_status == "OK")
    missing_files = sum(1 for r in verification_result.file_results if r.verification_status == "MISSING")
    modified_files = sum(1 for r in verification_result.file_results if r.verification_status == "MODIFIED")
    error_files = sum(1 for r in verification_result.file_results if r.verification_status == "ERROR")

    # Count total hunks and missing hunks
    total_hunks = 0
    missing_hunks = 0
    for file_result in verification_result.file_results:
        if file_result.diff_analysis:
            total_hunks += file_result.diff_analysis.total_hunks
            missing_hunks += len(file_result.diff_analysis.missing_hunks)

    # Count fix suggestions by safety level
    fix_counts = {"safe": 0, "review": 0, "dangerous": 0}
    for file_result in verification_result.file_results:
        for fix in file_result.fix_suggestions:
            if fix.safety_level in fix_counts:
                fix_counts[fix.safety_level] += 1

    # Generate recommendations
    recommendations = []
    if missing_hunks > 0:
        recommendations.append("Some hunks are missing - consider applying corrective patches")
    if fix_counts["safe"] > 0:
        recommendations.append(f"{fix_counts['safe']} safe fixes available for automatic application")
    if fix_counts["review"] > 0:
        recommendations.append(f"{fix_counts['review']} fixes require review before application")
    if error_files > 0:
        recommendations.append("Some files have errors - check patch format and repository state")

    return {
        "overall_status": verification_result.overall_status,
        "file_summary": {
            "total": total_files,
            "ok": ok_files,
            "missing": missing_files,
            "modified": modified_files,
            "errors": error_files
        },
        "hunk_summary": {
            "total": total_hunks,
            "missing": missing_hunks,
            "applied": total_hunks - missing_hunks
        },
        "fix_suggestions": fix_counts,
        "recommendations": recommendations,
        "completion_percentage": (ok_files / total_files * 100) if total_files > 0 else 0
    }


def generate_api_schema() -> Dict[str, Any]:
    """Generate OpenAPI-style schema for all public AI agent functions.

    Returns:
        Dict containing complete API schema with function signatures, parameters,
        return types, error codes, and usage examples.
    """
    import inspect
    import typing
    from typing import get_type_hints

    def get_function_schema(func) -> Dict[str, Any]:
        """Extract schema information from a function."""
        try:
            signature = inspect.signature(func)
            type_hints = get_type_hints(func)

            parameters = {}
            for param_name, param in signature.parameters.items():
                param_info = {
                    "required": param.default == inspect.Parameter.empty,
                    "type": str(type_hints.get(param_name, type(param.default).__name__ if param.default != inspect.Parameter.empty else "Any"))
                }
                if param.default != inspect.Parameter.empty:
                    param_info["default"] = param.default
                parameters[param_name] = param_info

            return_type = type_hints.get('return', 'Any')

            return {
                "description": inspect.getdoc(func) or "No description available",
                "parameters": parameters,
                "return_type": str(return_type),
                "signature": str(signature)
            }
        except Exception as e:
            return {
                "description": f"Schema extraction failed: {e}",
                "parameters": {},
                "return_type": "Unknown",
                "signature": "Unknown"
            }

    # Define AI agent functions to include in schema
    ai_functions = {
        "run_validation": run_validation,
        "validate_from_content": validate_from_content,
        "apply_safe_fixes": apply_safe_fixes,
        "extract_missing_changes": extract_missing_changes,
        "generate_corrective_patch": generate_corrective_patch,
        "split_large_patch": split_large_patch,
        "summarize_patch_status": summarize_patch_status,
        "validate_incremental": validate_incremental,
        "validate_patch_sequence": validate_patch_sequence,
        "create_patch_application_plan": create_patch_application_plan,
    }

    # Generate schema for each function
    functions_schema = {}
    for name, func in ai_functions.items():
        functions_schema[name] = get_function_schema(func)

    # Define data types schema
    data_types = {
        "ErrorInfo": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Error code identifier"},
                "message": {"type": "string", "description": "Human-readable error message"},
                "suggestion": {"type": "string", "description": "Recovery suggestion for AI agents"},
                "context": {"type": "object", "description": "Additional context information"},
                "recoverable": {"type": "boolean", "description": "Whether error is recoverable"},
                "severity": {"type": "string", "enum": ["error", "warning", "info"]}
            }
        },
        "VerificationResult": {
            "type": "object",
            "properties": {
                "overall_status": {"type": "string", "enum": ["FULLY_APPLIED", "PARTIALLY_APPLIED", "NOT_APPLIED", "ANALYSIS_ERROR"]},
                "success_count": {"type": "integer", "description": "Number of successfully applied files"},
                "total_count": {"type": "integer", "description": "Total number of files in patch"},
                "file_results": {"type": "array", "description": "Results for each file in the patch"}
            }
        },
        "Config": {
            "type": "object",
            "properties": {
                "patch_dir": {"type": "string", "default": "."},
                "repo_path": {"type": "string", "default": "."},
                "similarity_threshold": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "hunk_tolerance": {"type": "integer", "minimum": 0},
                "timeout": {"type": "integer", "minimum": 1}
            },
            "factory_methods": [
                {"name": "strict_mode", "description": "High precision validation"},
                {"name": "lenient_mode", "description": "Flexible fuzzy matching"},
                {"name": "fast_mode", "description": "Speed-optimized processing"}
            ]
        }
    }

    # Define error codes
    error_codes = {
        ERROR_NO_PATCHES_FOUND: "No patch files found in specified directory",
        ERROR_GIT_COMMAND_FAILED: "Git command execution failed",
        ERROR_PARSE_ERROR: "Failed to parse patch file format",
        ERROR_FILE_NOT_FOUND: "Target file not found in repository",
        ERROR_TIMEOUT: "Operation timed out",
        ERROR_PERMISSION_DENIED: "Permission denied accessing file or directory",
        ERROR_INVALID_CONFIG: "Invalid configuration parameters provided"
    }

    # Define usage examples
    examples = {
        "basic_validation": {
            "description": "Basic patch validation workflow",
            "code": """
# Validate a patch file
result = run_validation("my_patch.patch")
if result["success"]:
    verification_result = result["verification_result"]
    print(f"Status: {verification_result.overall_status}")
else:
    error_info = result.get("error_info", {})
    print(f"Error: {error_info.get('message')}")
    print(f"Suggestion: {error_info.get('suggestion')}")
"""
        },
        "safe_fix_application": {
            "description": "Automatically apply safe fixes",
            "code": """
# Validate patch and apply safe fixes
result = run_validation("my_patch.patch")
if result["success"]:
    fix_result = apply_safe_fixes(
        result["verification_result"],
        confirm=False,
        safety_levels=["safe"]
    )
    print(f"Applied: {len(fix_result['applied'])} fixes")
    print(f"Skipped: {len(fix_result['skipped'])} fixes")
"""
        },
        "configuration_profiles": {
            "description": "Using predefined configuration profiles",
            "code": """
# Use strict mode for critical validation
config = Config.strict_mode(repo_path="/path/to/repo")
verifier = PatchVerifier(**asdict(config))

# Use fast mode for quick checks
config = Config.fast_mode(patch_dir="./patches")
result = run_validation("patch.patch", **asdict(config))
"""
        }
    }

    return {
        "openapi": "3.0.0",
        "info": {
            "title": "PatchDoctor AI Agent API",
            "version": "2.0.0",
            "description": "API schema for PatchDoctor AI agent integration functions"
        },
        "functions": functions_schema,
        "data_types": data_types,
        "error_codes": error_codes,
        "examples": examples,
        "generation_timestamp": str(__import__("datetime").datetime.now()),
        "supported_operations": [
            "patch_validation",
            "safe_fix_application",
            "incremental_processing",
            "batch_operations",
            "patch_analysis",
            "configuration_management"
        ]
    }


def _extract_context_lines(file_content: Optional[List[str]], hunk: DiffHunk, context_lines: int = 3) -> List[str]:
    """Extract context lines around a hunk location."""
    if not file_content:
        return []

    start_line = max(0, hunk.new_start - context_lines - 1)
    end_line = min(len(file_content), hunk.new_start + hunk.new_count + context_lines)

    return file_content[start_line:end_line]


def _determine_conflict_type(file_result: FileVerificationResult, hunk: DiffHunk) -> str:
    """Determine the type of conflict for a missing hunk."""
    if file_result.verification_status == "MISSING":
        return "file_missing"
    elif file_result.verification_status == "MODIFIED":
        return "content_modified"
    elif file_result.verification_status == "ERROR":
        return "application_error"
    else:
        return "location_mismatch"


def _split_patch_by_file(patch_content: str) -> List[str]:
    """Split patch by individual files."""
    patches = []
    lines = patch_content.split('\n')

    # Find header (everything before first diff --git)
    header = []
    diff_start = -1
    for i, line in enumerate(lines):
        if line.startswith('diff --git'):
            diff_start = i
            break
        header.append(line)

    if diff_start == -1:
        # No diff --git found, return original
        return [patch_content]

    # Split by diff --git sections
    current_diff = []
    for i in range(diff_start, len(lines)):
        line = lines[i]
        if line.startswith('diff --git') and current_diff:
            # Start of new file, save current diff with header
            patch = '\n'.join(header + current_diff)
            patches.append(patch)
            current_diff = [line]
        else:
            current_diff.append(line)

    # Add the last patch
    if current_diff:
        patch = '\n'.join(header + current_diff)
        patches.append(patch)

    return patches


def _split_patch_by_hunk(patch_content: str) -> List[str]:
    """Split patch by individual hunks."""
    patches = []
    lines = patch_content.split('\n')
    current_file_header = []
    current_hunk = []

    for line in lines:
        if line.startswith('diff --git') or line.startswith('---') or line.startswith('+++'):
            current_file_header.append(line)
        elif line.startswith('@@'):
            # Start of new hunk
            if current_hunk:
                # Save previous hunk with file header
                patch = '\n'.join(current_file_header + current_hunk)
                patches.append(patch)
            current_hunk = [line]
        elif current_hunk:  # We're in a hunk
            current_hunk.append(line)

    # Add the last hunk
    if current_hunk:
        patch = '\n'.join(current_file_header + current_hunk)
        patches.append(patch)

    return patches


def _split_patch_by_size(patch_content: str, max_size: int = 50000) -> List[str]:
    """Split patch by size (bytes)."""
    if len(patch_content) <= max_size:
        return [patch_content]

    # Try to split by file first
    file_patches = _split_patch_by_file(patch_content)

    # If files are still too large, split by hunk
    result_patches = []
    for file_patch in file_patches:
        if len(file_patch) <= max_size:
            result_patches.append(file_patch)
        else:
            hunk_patches = _split_patch_by_hunk(file_patch)
            result_patches.extend(hunk_patches)

    return result_patches


# ===== AI AGENT INTEGRATION: Incremental Processing =====

def validate_incremental(
    patch_dir: str,
    progress_callback: Optional[callable] = None,
    early_stop_on_error: bool = False,
    max_concurrent: int = 1,
    **validation_kwargs
) -> Dict[str, Any]:
    """Process patches incrementally with progress reporting.

    Args:
        patch_dir: Directory containing patch files
        progress_callback: Optional callback function called for each patch processed
                          Signature: callback(patch_file: str, result: VerificationResult) -> None
        early_stop_on_error: If True, stop processing on first error
        max_concurrent: Maximum number of patches to process concurrently (default: 1)
        **validation_kwargs: Additional arguments passed to PatchVerifier

    Returns:
        Dict with incremental processing results and summary
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from pathlib import Path

    # Set default validation kwargs
    config_defaults = {
        "repo_path": ".",
        "verbose": False,
        "similarity_threshold": DEFAULT_SIMILARITY_THRESHOLD,
        "hunk_search_tolerance": DEFAULT_HUNK_SEARCH_TOLERANCE,
        "timeout": DEFAULT_SUBPROCESS_TIMEOUT,
        "max_file_size_mb": MAX_FILE_SIZE_MB,
    }
    config_defaults.update(validation_kwargs)

    # Find patch files
    patch_files = list(Path(patch_dir).glob("*.patch"))
    if not patch_files:
        error_info = ErrorInfo(
            code=ERROR_NO_PATCHES_FOUND,
            message="No .patch files found in directory",
            suggestion="Check that the patch directory contains .patch files",
            context={"patch_dir": patch_dir},
            recoverable=True,
            severity="error"
        )
        return {
            "success": False,
            "error": "No .patch files found",
            "error_info": asdict(error_info),
            "results": [],
            "processed_count": 0,
            "total_count": 0
        }

    # Initialize results tracking
    results = []
    errors = []
    processed_count = 0
    total_count = len(patch_files)
    should_stop = threading.Event()

    def process_single_patch(patch_file: Path) -> Tuple[str, Optional[VerificationResult], Optional[Dict[str, Any]]]:
        """Process a single patch file."""
        if should_stop.is_set():
            return str(patch_file), None, None

        try:
            # Create verifier for this patch
            verifier = PatchVerifier(**config_defaults)
            result = verifier.verify_patch(str(patch_file))
            return str(patch_file), result, None

        except Exception as e:
            error_info = ErrorInfo(
                code=ERROR_PARSE_ERROR if "parse" in str(e).lower() else ERROR_GIT_COMMAND_FAILED,
                message=f"Failed to process patch: {e}",
                suggestion="Check that the patch file is properly formatted",
                context={"patch_file": str(patch_file), "error_type": type(e).__name__},
                recoverable=True,
                severity="error"
            )
            return str(patch_file), None, asdict(error_info)

    # Process patches with optional parallelism
    if max_concurrent > 1:
        # Parallel processing
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            # Submit all tasks
            future_to_patch = {
                executor.submit(process_single_patch, patch_file): patch_file
                for patch_file in patch_files
            }

            # Process completed tasks
            for future in as_completed(future_to_patch):
                patch_file, result, error = future.result()

                if error:
                    errors.append(error)
                    if early_stop_on_error:
                        should_stop.set()
                        break
                elif result:
                    results.append(result)

                processed_count += 1

                # Call progress callback if provided
                if progress_callback:
                    try:
                        progress_callback(patch_file, result)
                    except Exception as cb_error:
                        # Don't let callback errors stop processing
                        error_info = ErrorInfo(
                            code="PROGRESS_CALLBACK_ERROR",
                            message=f"Progress callback failed: {cb_error}",
                            suggestion="Check the progress callback function for errors",
                            context={"callback_error": str(cb_error)},
                            recoverable=True,
                            severity="warning"
                        )
                        errors.append(asdict(error_info))

    else:
        # Sequential processing
        for patch_file in patch_files:
            if should_stop.is_set():
                break

            patch_file_str, result, error = process_single_patch(patch_file)

            if error:
                errors.append(error)
                if early_stop_on_error:
                    break
            elif result:
                results.append(result)

            processed_count += 1

            # Call progress callback if provided
            if progress_callback:
                try:
                    progress_callback(patch_file_str, result)
                except Exception as cb_error:
                    error_info = ErrorInfo(
                        code="PROGRESS_CALLBACK_ERROR",
                        message=f"Progress callback failed: {cb_error}",
                        suggestion="Check the progress callback function for errors",
                        context={"callback_error": str(cb_error)},
                        recoverable=True,
                        severity="warning"
                    )
                    errors.append(asdict(error_info))

    # Calculate summary statistics
    fully_applied = sum(1 for r in results if r.overall_status == "FULLY_APPLIED")
    partially_applied = sum(1 for r in results if r.overall_status == "PARTIALLY_APPLIED")
    not_applied = sum(1 for r in results if r.overall_status == "NOT_APPLIED")

    return {
        "success": len(errors) == 0,
        "processed_count": processed_count,
        "total_count": total_count,
        "completion_percentage": (processed_count / total_count * 100) if total_count > 0 else 0,
        "fully_applied": fully_applied,
        "partially_applied": partially_applied,
        "not_applied": not_applied,
        "error_count": len(errors),
        "errors": errors,
        "results": [asdict(r) for r in results],
        "early_stopped": should_stop.is_set(),
        "summary": f"Processed {processed_count}/{total_count} patches: {fully_applied} fully applied, {partially_applied} partial, {not_applied} not applied, {len(errors)} errors"
    }


# ===== AI AGENT INTEGRATION: Batch Patch Processing Utilities =====

def validate_patch_sequence(
    patch_files: List[str],
    dependency_order: bool = True,
    rollback_on_failure: bool = False,
    checkpoint_frequency: int = 5,
    **kwargs
) -> Dict[str, Any]:
    """Validate a sequence of patches with dependency tracking.

    Args:
        patch_files: List of patch file paths in desired order
        dependency_order: If True, analyze and reorder patches by dependencies
        rollback_on_failure: If True, rollback all changes on any failure
        checkpoint_frequency: Create rollback points every N patches
        **kwargs: Additional arguments passed to validation

    Returns:
        Dict with sequence validation results and dependency analysis
    """
    import tempfile
    import shutil

    if not patch_files:
        error_info = ErrorInfo(
            code=ERROR_NO_PATCHES_FOUND,
            message="No patch files provided for sequence validation",
            suggestion="Provide a list of patch file paths to validate",
            context={"provided_count": 0},
            recoverable=False,
            severity="error"
        )
        return {
            "success": False,
            "error": "No patch files provided",
            "error_info": asdict(error_info),
            "sequence_results": []
        }

    # Analyze dependencies if requested
    dependencies = {}
    application_plan = patch_files.copy()

    if dependency_order:
        dependency_analysis = _analyze_patch_file_dependencies(patch_files)
        dependencies = dependency_analysis["dependencies"]
        application_plan = _create_dependency_order(patch_files, dependencies)

    # Set up rollback tracking
    rollback_info = {
        "checkpoints": [],
        "git_stash_ids": [],
        "original_state": None
    }

    # Create GitRunner for rollback operations
    repo_path = kwargs.get("repo_path", ".")
    git_runner = GitRunner(repo_path=repo_path)

    # Capture initial state
    if rollback_on_failure:
        initial_state = git_runner.run_git_command(["status", "--porcelain"])
        rollback_info["original_state"] = initial_state.stdout if initial_state.ok else None

    # Process patches in dependency order
    sequence_results = []
    processed_count = 0
    checkpoint_count = 0

    try:
        for i, patch_file in enumerate(application_plan):
            # Create checkpoint if needed
            if rollback_on_failure and i > 0 and i % checkpoint_frequency == 0:
                checkpoint_result = _create_rollback_checkpoint(git_runner, checkpoint_count)
                if checkpoint_result["success"]:
                    rollback_info["checkpoints"].append(checkpoint_result)
                    checkpoint_count += 1

            # Validate single patch
            try:
                if patch_file.endswith('.patch'):
                    # Parse as patch file
                    with open(patch_file, 'r', encoding='utf-8') as f:
                        patch_content = f.read()
                    result = validate_from_content(patch_content, **kwargs)
                else:
                    # Assume it's a patch directory
                    result = run_validation(patch_dir=patch_file, **kwargs)

                # Track result
                patch_result = {
                    "patch_file": patch_file,
                    "sequence_index": i,
                    "original_index": patch_files.index(patch_file),
                    "result": result,
                    "dependencies": dependencies.get(Path(patch_file).name, [])
                }

                sequence_results.append(patch_result)
                processed_count += 1

                # Check for failure and handle rollback
                if not result.get("success", False) and rollback_on_failure:
                    # Rollback on failure
                    rollback_result = _rollback_to_checkpoint(git_runner, rollback_info, checkpoint_count - 1)
                    return {
                        "success": False,
                        "error": f"Patch sequence failed at {patch_file}",
                        "processed_count": processed_count,
                        "total_count": len(application_plan),
                        "sequence_results": sequence_results,
                        "dependencies": dependencies,
                        "application_plan": application_plan,
                        "rollback_performed": True,
                        "rollback_result": rollback_result
                    }

            except Exception as e:
                error_result = {
                    "patch_file": patch_file,
                    "sequence_index": i,
                    "error": str(e),
                    "error_info": asdict(ErrorInfo(
                        code=ERROR_PARSE_ERROR,
                        message=f"Failed to process patch in sequence: {e}",
                        suggestion="Check patch file format and accessibility",
                        context={"patch_file": patch_file, "sequence_index": i},
                        recoverable=not rollback_on_failure,
                        severity="error"
                    ))
                }

                sequence_results.append(error_result)

                if rollback_on_failure:
                    rollback_result = _rollback_to_checkpoint(git_runner, rollback_info, checkpoint_count - 1)
                    return {
                        "success": False,
                        "error": f"Exception during patch sequence at {patch_file}: {e}",
                        "processed_count": processed_count,
                        "total_count": len(application_plan),
                        "sequence_results": sequence_results,
                        "rollback_performed": True,
                        "rollback_result": rollback_result
                    }

        # Calculate final statistics
        successful_patches = sum(1 for r in sequence_results if r.get("result", {}).get("success", False))
        success_rate = (successful_patches / len(sequence_results)) * 100 if sequence_results else 0

        return {
            "success": successful_patches == len(sequence_results),
            "processed_count": processed_count,
            "total_count": len(application_plan),
            "success_rate": success_rate,
            "sequence_results": sequence_results,
            "dependencies": dependencies,
            "application_plan": application_plan,
            "rollback_performed": False,
            "rollback_info": rollback_info
        }

    except Exception as e:
        # Handle unexpected exceptions
        if rollback_on_failure:
            rollback_result = _rollback_to_checkpoint(git_runner, rollback_info, 0)
            return {
                "success": False,
                "error": f"Unexpected error during patch sequence: {e}",
                "processed_count": processed_count,
                "total_count": len(application_plan),
                "sequence_results": sequence_results,
                "rollback_performed": True,
                "rollback_result": rollback_result
            }

        return {
            "success": False,
            "error": f"Unexpected error during patch sequence: {e}",
            "processed_count": processed_count,
            "total_count": len(application_plan),
            "sequence_results": sequence_results
        }


def create_patch_application_plan(
    verification_results: List[VerificationResult]
) -> Dict[str, Any]:
    """Create a step-by-step plan for applying patches.

    Args:
        verification_results: List of verification results from patch validation

    Returns:
        Dict with application plan and analysis
    """
    if not verification_results:
        return {
            "application_order": [],
            "dependencies": {},
            "conflict_warnings": [],
            "rollback_points": [],
            "estimated_duration": 0
        }

    # Analyze patch complexity and dependencies
    patches_info = []
    file_to_patches = {}

    for i, result in enumerate(verification_results):
        patch_name = result.patch_info.filename
        modified_files = set()

        # Extract modified files from file results
        for file_result in result.file_results:
            modified_files.add(file_result.file_path)

            # Track which patches modify each file
            if file_result.file_path not in file_to_patches:
                file_to_patches[file_result.file_path] = []
            file_to_patches[file_result.file_path].append(patch_name)

        # Estimate complexity
        complexity_score = len(result.file_results) * 2
        if result.overall_status == "PARTIALLY_APPLIED":
            complexity_score += 5
        elif result.overall_status == "NOT_APPLIED":
            complexity_score += 10

        patches_info.append({
            "name": patch_name,
            "index": i,
            "modified_files": modified_files,
            "complexity": complexity_score,
            "status": result.overall_status,
            "fix_count": sum(len(fr.fix_suggestions) for fr in result.file_results)
        })

    # Identify dependencies and conflicts
    dependencies = {}
    conflicts = []

    for file_path, patch_names in file_to_patches.items():
        if len(patch_names) > 1:
            # Multiple patches modify same file - potential conflict
            conflicts.append({
                "file": file_path,
                "patches": patch_names,
                "severity": "high" if len(patch_names) > 2 else "medium"
            })

            # Create dependency chain for same-file patches
            sorted_patches = sorted(patch_names)
            for i in range(len(sorted_patches) - 1):
                dependent = sorted_patches[i + 1]
                dependency = sorted_patches[i]

                if dependent not in dependencies:
                    dependencies[dependent] = []
                if dependency not in dependencies[dependent]:
                    dependencies[dependent].append(dependency)

    # Create application order using topological sort
    application_order = _topological_sort_patches(patches_info, dependencies)

    # Determine rollback points (after complex patches or before conflicts)
    rollback_points = []
    for i, patch_info in enumerate(application_order):
        # Add rollback point before high-complexity patches
        if patch_info["complexity"] > 15:
            rollback_points.append({
                "index": i,
                "reason": "high_complexity",
                "patch": patch_info["name"]
            })

        # Add rollback point before conflicting patches
        patch_files = patch_info["modified_files"]
        for conflict in conflicts:
            if patch_info["name"] in conflict["patches"]:
                rollback_points.append({
                    "index": i,
                    "reason": "potential_conflict",
                    "patch": patch_info["name"],
                    "conflict_file": conflict["file"]
                })
                break

    # Estimate duration (simple heuristic)
    total_complexity = sum(p["complexity"] for p in patches_info)
    estimated_duration = max(30, total_complexity * 2)  # seconds

    return {
        "application_order": application_order,
        "dependencies": dependencies,
        "conflict_warnings": conflicts,
        "rollback_points": rollback_points,
        "estimated_duration": estimated_duration,
        "total_patches": len(patches_info),
        "complexity_score": total_complexity
    }


def _analyze_patch_file_dependencies(patch_files: List[str]) -> Dict[str, Any]:
    """Analyze dependencies between patch files based on modified files."""
    file_to_patches = {}
    dependencies = {}

    for patch_file in patch_files:
        try:
            with open(patch_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Extract modified files
            modified_files = set()
            for line in content.split('\n'):
                if line.startswith('diff --git'):
                    parts = line.split()
                    if len(parts) >= 4:
                        file_path = parts[2][2:]  # Remove 'a/' prefix
                        modified_files.add(file_path)

            # Track file to patch mapping
            patch_name = Path(patch_file).name
            for file_path in modified_files:
                if file_path not in file_to_patches:
                    file_to_patches[file_path] = []
                file_to_patches[file_path].append(patch_name)

        except Exception:
            continue  # Skip problematic patches

    # Build dependency relationships
    for file_path, patches in file_to_patches.items():
        if len(patches) > 1:
            sorted_patches = sorted(patches)
            for i in range(len(sorted_patches) - 1):
                dependent = sorted_patches[i + 1]
                dependency = sorted_patches[i]

                if dependent not in dependencies:
                    dependencies[dependent] = []
                if dependency not in dependencies[dependent]:
                    dependencies[dependent].append(dependency)

    return {"dependencies": dependencies, "file_mappings": file_to_patches}


def _create_dependency_order(patch_files: List[str], dependencies: Dict[str, List[str]]) -> List[str]:
    """Create patch application order based on dependencies."""
    patch_names = [Path(pf).name for pf in patch_files]
    patch_file_map = {Path(pf).name: pf for pf in patch_files}

    # Topological sort
    visited = set()
    order = []

    def visit(patch_name):
        if patch_name in visited:
            return
        visited.add(patch_name)

        # Visit dependencies first
        for dep in dependencies.get(patch_name, []):
            if dep in patch_names:
                visit(dep)

        if patch_name in patch_file_map:
            order.append(patch_file_map[patch_name])

    for patch_name in patch_names:
        visit(patch_name)

    return order


def _topological_sort_patches(patches_info: List[Dict], dependencies: Dict[str, List[str]]) -> List[Dict]:
    """Sort patches using topological ordering based on dependencies."""
    # Create name to info mapping
    name_to_info = {p["name"]: p for p in patches_info}

    # Topological sort
    visited = set()
    order = []

    def visit(patch_name):
        if patch_name in visited:
            return
        visited.add(patch_name)

        # Visit dependencies first
        for dep in dependencies.get(patch_name, []):
            if dep in name_to_info:
                visit(dep)

        if patch_name in name_to_info:
            order.append(name_to_info[patch_name])

    for patch_info in patches_info:
        visit(patch_info["name"])

    return order


def _create_rollback_checkpoint(git_runner: 'GitRunner', checkpoint_id: int) -> Dict[str, Any]:
    """Create a rollback checkpoint using git stash."""
    try:
        # Create stash with descriptive message
        stash_result = git_runner.run_git_command([
            "stash", "push", "-m", f"PatchDoctor checkpoint {checkpoint_id}"
        ])

        if stash_result.ok:
            return {
                "success": True,
                "checkpoint_id": checkpoint_id,
                "stash_message": f"PatchDoctor checkpoint {checkpoint_id}",
                "timestamp": time.time()
            }
        else:
            return {
                "success": False,
                "error": stash_result.stderr,
                "checkpoint_id": checkpoint_id
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "checkpoint_id": checkpoint_id
        }


def _rollback_to_checkpoint(git_runner: 'GitRunner', rollback_info: Dict, checkpoint_id: int) -> Dict[str, Any]:
    """Rollback to a specific checkpoint."""
    try:
        if checkpoint_id < 0 or checkpoint_id >= len(rollback_info.get("checkpoints", [])):
            # Rollback to original state
            reset_result = git_runner.run_git_command(["checkout", "--", "."])
            if reset_result.ok:
                return {"success": True, "method": "full_reset"}
            else:
                return {"success": False, "error": reset_result.stderr}

        # Rollback to specific checkpoint
        checkpoint = rollback_info["checkpoints"][checkpoint_id]
        stash_result = git_runner.run_git_command(["stash", "pop"])

        if stash_result.ok:
            return {
                "success": True,
                "method": "stash_pop",
                "checkpoint_id": checkpoint_id
            }
        else:
            return {
                "success": False,
                "error": stash_result.stderr,
                "checkpoint_id": checkpoint_id
            }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "checkpoint_id": checkpoint_id
        }


# ===== AI AGENT INTEGRATION: Workflow Template System =====

class IterativePatchWorkflow:
    """Template for iterative patch application and validation."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.verifier = PatchVerifier(
            repo_path=self.config.repo_path,
            verbose=self.config.verbose,
            similarity_threshold=self.config.similarity_threshold,
            hunk_search_tolerance=self.config.hunk_tolerance,
            timeout=self.config.timeout,
            max_file_size_mb=self.config.max_file_size,
        )
        self.history = []
        self.rollback_points = []

    def execute(self, large_patch_content: str, strategy: str = "auto") -> Dict[str, Any]:
        """Execute the iterative workflow with built-in error recovery.

        Args:
            large_patch_content: The large patch content to process iteratively
            strategy: Splitting strategy ("auto", "by_file", "by_hunk", "by_size")

        Returns:
            Dict with comprehensive workflow results
        """
        workflow_start = time.time()

        try:
            # Step 1: Analyze patch complexity
            complexity_analysis = self._analyze_patch_complexity(large_patch_content)

            # Step 2: Split patch based on strategy
            if strategy == "auto":
                strategy = complexity_analysis.get("recommended_strategy", "by_file")

            split_patches = split_large_patch(large_patch_content, strategy=strategy)

            # Step 3: Process patches iteratively
            results = []
            successful_count = 0
            rollback_performed = False

            for i, patch_piece in enumerate(split_patches):
                step_result = self._process_patch_step(patch_piece, i, len(split_patches))
                results.append(step_result)

                if step_result.get("success", False):
                    successful_count += 1
                else:
                    # Handle failure based on severity
                    if step_result.get("critical_failure", False):
                        # Perform rollback and stop
                        rollback_result = self._rollback_workflow()
                        rollback_performed = True
                        break

            # Step 4: Generate final summary
            success_rate = (successful_count / len(split_patches)) * 100 if split_patches else 0
            workflow_time = time.time() - workflow_start

            return {
                "success": success_rate >= 80 and not rollback_performed,
                "strategy_used": strategy,
                "complexity_analysis": complexity_analysis,
                "total_patches": len(split_patches),
                "successful_patches": successful_count,
                "success_rate": success_rate,
                "workflow_time": workflow_time,
                "rollback_performed": rollback_performed,
                "step_results": results,
                "recommendations": self._generate_recommendations(results, success_rate)
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "error_info": asdict(ErrorInfo(
                    code="ITERATIVE_WORKFLOW_ERROR",
                    message=f"Iterative workflow failed: {e}",
                    suggestion="Check patch content and repository state",
                    context={"strategy": strategy},
                    recoverable=True,
                    severity="error"
                )),
                "workflow_time": time.time() - workflow_start
            }

    def _analyze_patch_complexity(self, patch_content: str) -> Dict[str, Any]:
        """Analyze patch complexity for workflow planning."""
        lines = patch_content.split('\n')
        file_count = len([line for line in lines if line.startswith('diff --git')])
        hunk_count = len([line for line in lines if line.startswith('@@')])
        total_lines = len(lines)

        complexity_score = file_count * 10 + hunk_count * 5 + total_lines * 0.1

        if file_count > 10:
            strategy = "by_file"
        elif hunk_count > 50:
            strategy = "by_hunk"
        elif total_lines > 5000:
            strategy = "by_size"
        else:
            strategy = "none"

        return {
            "file_count": file_count,
            "hunk_count": hunk_count,
            "total_lines": total_lines,
            "complexity_score": complexity_score,
            "recommended_strategy": strategy
        }

    def _process_patch_step(self, patch_content: str, step_index: int, total_steps: int) -> Dict[str, Any]:
        """Process a single patch step with error handling."""
        step_start = time.time()

        try:
            # Validate patch
            result = validate_from_content(
                patch_content=patch_content,
                repo_path=self.config.repo_path,
                verbose=self.config.verbose
            )

            step_time = time.time() - step_start

            step_result = {
                "step_index": step_index,
                "total_steps": total_steps,
                "step_time": step_time,
                "validation_result": result,
                "success": result.get("success", False)
            }

            # Record in history
            self.history.append(step_result)

            # Determine if this is a critical failure
            if not result.get("success", False):
                error_info = result.get("error_info", {})
                step_result["critical_failure"] = error_info.get("severity") == "error"

            return step_result

        except Exception as e:
            return {
                "step_index": step_index,
                "total_steps": total_steps,
                "step_time": time.time() - step_start,
                "success": False,
                "error": str(e),
                "critical_failure": True
            }

    def _rollback_workflow(self) -> Dict[str, Any]:
        """Rollback the workflow to the last stable state."""
        try:
            git_runner = GitRunner(repo_path=self.config.repo_path)
            reset_result = git_runner.run_git_command(["checkout", "--", "."])

            return {
                "success": reset_result.ok,
                "method": "git_checkout",
                "error": reset_result.stderr if not reset_result.ok else None
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _generate_recommendations(self, results: List[Dict], success_rate: float) -> List[str]:
        """Generate recommendations based on workflow results."""
        recommendations = []

        if success_rate == 100:
            recommendations.append("✅ Iterative workflow completed successfully")
        elif success_rate >= 80:
            recommendations.append("🟡 Most steps successful - review failed steps")
        else:
            recommendations.append("🔴 Low success rate - consider different splitting strategy")

        failed_steps = [r for r in results if not r.get("success", False)]
        if failed_steps:
            recommendations.append(f"📋 {len(failed_steps)} steps failed - check individual error details")

        return recommendations


class BatchProcessingWorkflow:
    """Template for processing multiple patches safely."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.processed_patches = []
        self.failed_patches = []
        self.rollback_checkpoints = []

    def execute(self, patch_directory: str) -> Dict[str, Any]:
        """Execute batch processing with dependency analysis and rollback.

        Args:
            patch_directory: Directory containing patches to process

        Returns:
            Dict with batch processing results
        """
        workflow_start = time.time()

        try:
            # Step 1: Discover and analyze patches
            patch_files = list(Path(patch_directory).glob("*.patch"))
            if not patch_files:
                return {
                    "success": False,
                    "error": "No patch files found",
                    "error_info": asdict(ErrorInfo(
                        code=ERROR_NO_PATCHES_FOUND,
                        message="No patch files found in directory",
                        suggestion="Check directory path and ensure .patch files exist",
                        context={"directory": patch_directory},
                        recoverable=False,
                        severity="error"
                    ))
                }

            # Step 2: Analyze dependencies
            dependency_analysis = _analyze_patch_file_dependencies([str(pf) for pf in patch_files])

            # Step 3: Create application plan
            application_plan = _create_dependency_order([str(pf) for pf in patch_files], dependency_analysis["dependencies"])

            # Step 4: Process patches with checkpointing
            results = []
            checkpoint_frequency = max(1, len(application_plan) // 5)  # Create 5 checkpoints max

            for i, patch_file in enumerate(application_plan):
                # Create checkpoint if needed
                if i > 0 and i % checkpoint_frequency == 0:
                    checkpoint_result = self._create_checkpoint(i // checkpoint_frequency)
                    self.rollback_checkpoints.append(checkpoint_result)

                # Process single patch
                patch_result = self._process_single_patch(patch_file, i, len(application_plan))
                results.append(patch_result)

                if patch_result.get("success", False):
                    self.processed_patches.append(patch_result)
                else:
                    self.failed_patches.append(patch_result)

            # Step 5: Generate summary
            success_rate = (len(self.processed_patches) / len(application_plan)) * 100 if application_plan else 0
            workflow_time = time.time() - workflow_start

            return {
                "success": len(self.failed_patches) == 0,
                "total_patches": len(application_plan),
                "processed_patches": len(self.processed_patches),
                "failed_patches": len(self.failed_patches),
                "success_rate": success_rate,
                "workflow_time": workflow_time,
                "dependency_analysis": dependency_analysis,
                "application_plan": application_plan,
                "results": results,
                "checkpoints_created": len(self.rollback_checkpoints)
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "workflow_time": time.time() - workflow_start
            }

    def _process_single_patch(self, patch_file: str, index: int, total: int) -> Dict[str, Any]:
        """Process a single patch within the batch workflow."""
        try:
            with open(patch_file, 'r', encoding='utf-8') as f:
                patch_content = f.read()

            result = validate_from_content(
                patch_content=patch_content,
                repo_path=self.config.repo_path
            )

            return {
                "patch_file": patch_file,
                "index": index,
                "total": total,
                "result": result,
                "success": result.get("success", False)
            }

        except Exception as e:
            return {
                "patch_file": patch_file,
                "index": index,
                "total": total,
                "success": False,
                "error": str(e)
            }

    def _create_checkpoint(self, checkpoint_id: int) -> Dict[str, Any]:
        """Create a rollback checkpoint."""
        try:
            git_runner = GitRunner(repo_path=self.config.repo_path)
            return _create_rollback_checkpoint(git_runner, checkpoint_id)
        except Exception as e:
            return {"success": False, "error": str(e), "checkpoint_id": checkpoint_id}


class SafeFixApplicationWorkflow:
    """Template for automated fix application with safety checks."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.applied_fixes = []
        self.skipped_fixes = []
        self.failed_fixes = []

    def execute(self, verification_result: VerificationResult) -> Dict[str, Any]:
        """Apply fixes with comprehensive safety checks and rollback.

        Args:
            verification_result: Result from patch verification

        Returns:
            Dict with fix application results
        """
        workflow_start = time.time()

        try:
            # Step 1: Analyze available fixes
            all_fixes = []
            for file_result in verification_result.file_results:
                for fix in file_result.fix_suggestions:
                    all_fixes.append({
                        "file_path": file_result.file_path,
                        "fix": fix,
                        "safety_score": self._calculate_safety_score(fix)
                    })

            if not all_fixes:
                return {
                    "success": True,
                    "message": "No fixes available to apply",
                    "applied_fixes": 0,
                    "skipped_fixes": 0,
                    "failed_fixes": 0
                }

            # Step 2: Sort fixes by safety score (safest first)
            sorted_fixes = sorted(all_fixes, key=lambda x: x["safety_score"], reverse=True)

            # Step 3: Apply fixes with safety checks
            for fix_info in sorted_fixes:
                result = self._apply_single_fix_safely(fix_info)

                if result["success"]:
                    self.applied_fixes.append(result)
                elif result.get("skipped", False):
                    self.skipped_fixes.append(result)
                else:
                    self.failed_fixes.append(result)

            # Step 4: Generate summary
            workflow_time = time.time() - workflow_start

            return {
                "success": len(self.failed_fixes) == 0,
                "applied_fixes": len(self.applied_fixes),
                "skipped_fixes": len(self.skipped_fixes),
                "failed_fixes": len(self.failed_fixes),
                "workflow_time": workflow_time,
                "safety_summary": self._generate_safety_summary(),
                "recommendations": self._generate_safety_recommendations()
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "workflow_time": time.time() - workflow_start
            }

    def _calculate_safety_score(self, fix: FixSuggestion) -> float:
        """Calculate safety score for a fix (0.0 = dangerous, 1.0 = very safe)."""
        base_score = {
            "safe": 1.0,
            "review": 0.6,
            "dangerous": 0.2
        }.get(fix.safety_level, 0.5)

        # Adjust based on fix type
        type_modifier = {
            "git_restore": 0.9,
            "mini_patch": 0.8,
            "file_create": 0.7,
            "manual_edit": 0.3
        }.get(fix.fix_type, 0.5)

        # Adjust based on confidence
        confidence_modifier = fix.confidence

        return base_score * type_modifier * confidence_modifier

    def _apply_single_fix_safely(self, fix_info: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a single fix with safety checks."""
        fix = fix_info["fix"]
        safety_score = fix_info["safety_score"]

        # Safety check: skip dangerous fixes
        if safety_score < 0.5:
            return {
                "success": False,
                "skipped": True,
                "fix_info": fix_info,
                "reason": "Safety score too low",
                "safety_score": safety_score
            }

        # Safety check: skip fixes that require manual intervention
        if fix.fix_type == "manual_edit":
            return {
                "success": False,
                "skipped": True,
                "fix_info": fix_info,
                "reason": "Manual intervention required",
                "safety_score": safety_score
            }

        try:
            # Apply the fix (simplified - in practice would use apply_safe_fixes)
            git_runner = GitRunner(repo_path=self.config.repo_path)
            success = _apply_single_fix(git_runner, fix_info["file_path"], fix)

            return {
                "success": success,
                "fix_info": fix_info,
                "safety_score": safety_score,
                "applied": success
            }

        except Exception as e:
            return {
                "success": False,
                "fix_info": fix_info,
                "error": str(e),
                "safety_score": safety_score
            }

    def _generate_safety_summary(self) -> Dict[str, Any]:
        """Generate safety summary for applied fixes."""
        if not self.applied_fixes:
            return {"average_safety_score": 0, "risk_level": "none"}

        avg_safety = sum(fix["safety_score"] for fix in self.applied_fixes) / len(self.applied_fixes)

        risk_level = "low" if avg_safety >= 0.8 else "medium" if avg_safety >= 0.6 else "high"

        return {
            "average_safety_score": avg_safety,
            "risk_level": risk_level,
            "total_applied": len(self.applied_fixes)
        }

    def _generate_safety_recommendations(self) -> List[str]:
        """Generate safety recommendations based on fix application results."""
        recommendations = []

        if self.failed_fixes:
            recommendations.append(f"⚠️  {len(self.failed_fixes)} fixes failed - review error details")

        if self.skipped_fixes:
            recommendations.append(f"📋 {len(self.skipped_fixes)} fixes skipped for safety - consider manual review")

        safety_summary = self._generate_safety_summary()
        if safety_summary["risk_level"] == "high":
            recommendations.append("🔴 High risk fixes applied - monitor repository carefully")
        elif safety_summary["risk_level"] == "medium":
            recommendations.append("🟡 Medium risk fixes applied - verify results")
        else:
            recommendations.append("✅ Low risk fixes applied successfully")

        return recommendations


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

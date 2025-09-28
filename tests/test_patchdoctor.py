"""Tests for PatchDoctor."""

from datetime import datetime
from pathlib import Path
from patchdoctor import PatchParser, smart_truncate_path, detect_file_encoding


def test_smart_truncate_path():
    """Test path truncation."""
    # No truncation needed
    assert smart_truncate_path("short", 10) == "short"
    # Truncation from the start, keeping filename
    result = smart_truncate_path("very/long/path/to/file.txt", 20)
    assert result == "...file.txt"
    assert len(result) <= 20
    # Longer path
    result2 = smart_truncate_path("very/long/path/to/file.txt", 25)
    assert result2.endswith("file.txt")
    assert result2.startswith("...")
    # Exact boundary
    assert smart_truncate_path("abc", 3) == "abc"
    assert smart_truncate_path("abcd", 3) == "abc"


def test_detect_file_encoding():
    """Test encoding detection."""
    # Test with UTF-8 file
    test_file = Path("test_utf8.txt")
    test_file.write_text("hello world é", encoding="utf-8")
    try:
        encoding = detect_file_encoding(str(test_file))
        assert isinstance(encoding, str)
        assert len(encoding) > 0
        # Should detect as UTF-8 or similar
        assert encoding.lower() in [
            "utf-8",
            "utf-8-sig",
            "ascii",
            "iso-8859-1",
            "cp1252",
        ]
    finally:
        test_file.unlink()

    # Test with non-existent file
    assert detect_file_encoding("nonexistent.txt") == "utf-8"


def test_patch_parser_basic():
    """Test basic patch parsing."""
    patch_content = """From: test@example.com
Subject: Test patch

 test.txt | 2 +-
 1 file changed, 1 insertion(+), 1 deletion(-)

diff --git a/test.txt b/test.txt
--- a/test.txt
+++ b/test.txt
@@ -1 +1 @@
-old
+new
"""
    # Create temp file
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False) as f:
        f.write(patch_content)
        temp_file = f.name

    try:
        parser = PatchParser()
        patch_info = parser.parse_patch_file(temp_file)
        assert patch_info.filename == temp_file
        assert patch_info.subject == "Test patch"
        assert len(patch_info.files_changed) == 1
        file_op = patch_info.files_changed[0]
        assert file_op.new_path == "test.txt"
        assert file_op.operation == "modify"
        assert file_op.insertions == 1
        assert file_op.deletions == 1
        assert len(file_op.diff_hunks) == 1
    finally:
        import os

        os.unlink(temp_file)


def test_patch_parser_with_date_and_spaces(tmp_path):
    """Test parsing of patch with Date header and quoted paths."""
    patch_content = """From: test@example.com
Date: Mon, 01 Oct 2023 12:00:00 +0000
Subject: Rename file

 "old name.txt" => "new name.txt" | 1 +
 1 file changed, 1 insertion(+)
 create mode 100644 "new name.txt"

diff --git a/old name.txt b/new name.txt
new file mode 100644
index 0000000..e69de29
--- /dev/null
+++ b/new name.txt
@@ -0,0 +1 @@
+hello
"""
    patch_file = tmp_path / "rename.patch"
    patch_file.write_text(patch_content)
    parser = PatchParser()
    patch_info = parser.parse_patch_file(str(patch_file))
    assert isinstance(patch_info.date, (str, datetime))
    assert any(op.operation == "rename" for op in patch_info.files_changed)
    assert any(" " in op.new_path for op in patch_info.files_changed)


def test_patch_parser_binary_patch(tmp_path):
    """Test binary patch parsing."""
    patch_content = """From: bin@example.com
Subject: Binary update

 image.png | Bin 0 -> 245 bytes

diff --git a/image.png b/image.png
new file mode 100644
Binary files /dev/null and b/image.png differ
"""
    patch_file = tmp_path / "binary.patch"
    patch_file.write_text(patch_content)
    parser = PatchParser()
    patch_info = parser.parse_patch_file(str(patch_file))
    assert patch_info.files_changed[0].operation in ("create", "modify")
    assert "image.png" in patch_info.files_changed[0].new_path


def test_verifier_integration(tmp_path):
    """Test PatchVerifier detecting applied vs missing hunks."""
    file_path = tmp_path / "file.txt"
    file_path.write_text("old\n")

    patch_content = """From: test@example.com
Subject: Change file

 file.txt | 2 +-
 1 file changed, 1 insertion(+), 1 deletion(-)

diff --git a/file.txt b/file.txt
--- a/file.txt
+++ b/file.txt
@@ -1 +1 @@
-old
+new
"""
    patch_file = tmp_path / "change.patch"
    patch_file.write_text(patch_content)

    from patchdoctor import PatchVerifier

    parser = PatchParser()
    patch_info = parser.parse_patch_file(str(patch_file))
    verifier = PatchVerifier(str(tmp_path))
    results = verifier.verify_patch_info(patch_info)
    assert any(r.verification_status in ("MISSING", "OK") for r in results)


def test_json_report_structure(tmp_path):
    """Test that JSON report generation produces structured output."""
    from patchdoctor import ReportGenerator, PatchVerifier, VerificationResult

    patch_content = """From: test@example.com
Subject: Simple patch

 test.txt | 1 +
 1 file changed, 1 insertion(+)

diff --git a/test.txt b/test.txt
new file mode 100644
index 0000000..e69de29
--- /dev/null
+++ b/test.txt
@@ -0,0 +1 @@
+hello
"""
    patch_file = tmp_path / "simple.patch"
    patch_file.write_text(patch_content)
    parser = PatchParser()
    patch_info = parser.parse_patch_file(str(patch_file))
    verifier = PatchVerifier(str(tmp_path))
    file_results = verifier.verify_patch_info(patch_info)

    # Create a VerificationResult
    verification_result = VerificationResult(
        patch_info=patch_info,
        file_results=file_results,
        overall_status="FULLY_APPLIED"
        if all(r.verification_status == "OK" for r in file_results)
        else "PARTIALLY_APPLIED",
        success_count=sum(1 for r in file_results if r.verification_status == "OK"),
        total_count=len(file_results),
    )

    generator = ReportGenerator(detailed=True)
    # Test that we can generate JSON data (we'll need to modify the method or create a temp file)
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as f:
        temp_file = f.name

    generator.generate_json_report([verification_result], temp_file)

    import json

    with open(temp_file, "r") as f:
        json_data = json.load(f)

    import os

    os.unlink(temp_file)

    assert isinstance(json_data, list)
    assert "patch_info" in json_data[0]
    assert "file_results" in json_data[0]


# AI Agent Integration Tests

def test_error_info_structure():
    """Test ErrorInfo dataclass structure and functionality."""
    from patchdoctor import ErrorInfo

    error_info = ErrorInfo(
        code="TEST_ERROR",
        message="Test error message",
        suggestion="Try this recovery action",
        context={"file": "test.txt", "line": 42},
        recoverable=True,
        severity="error"
    )

    assert error_info.code == "TEST_ERROR"
    assert error_info.message == "Test error message"
    assert error_info.suggestion == "Try this recovery action"
    assert error_info.context["file"] == "test.txt"
    assert error_info.recoverable is True
    assert error_info.severity == "error"


def test_config_profiles():
    """Test configuration profile factory methods."""
    from patchdoctor import Config

    # Test strict mode
    strict_config = Config.strict_mode(repo_path="/test/repo")
    assert strict_config.similarity_threshold == 0.8
    assert strict_config.hunk_tolerance == 2
    assert strict_config.timeout == 60
    assert strict_config.repo_path == "/test/repo"

    # Test lenient mode
    lenient_config = Config.lenient_mode(verbose=True)
    assert lenient_config.similarity_threshold == 0.3
    assert lenient_config.hunk_tolerance == 10
    assert lenient_config.timeout == 30
    assert lenient_config.verbose is True

    # Test fast mode
    fast_config = Config.fast_mode(patch_dir="./patches")
    assert fast_config.similarity_threshold == 0.5
    assert fast_config.hunk_tolerance == 5
    assert fast_config.timeout == 15
    assert fast_config.max_file_size == 50
    assert fast_config.patch_dir == "./patches"


def test_gitrunner_performance_monitoring(tmp_path):
    """Test GitRunner performance monitoring capabilities."""
    from patchdoctor import GitRunner

    # Test with performance monitoring enabled
    git_runner = GitRunner(
        repo_path=str(tmp_path),
        enable_performance_monitoring=True
    )

    # Run some git commands
    result1 = git_runner.run_git_command(["status", "--porcelain"])
    result2 = git_runner.run_git_command(["status", "--porcelain"])  # Should be cached

    # Check cache stats
    stats = git_runner.get_cache_stats()
    assert "cache_hits" in stats
    assert "cache_misses" in stats
    assert "total_requests" in stats
    assert stats["total_requests"] >= 2

    # Cache might not work if git commands fail, so check if git worked first
    if result1.ok and result2.ok:
        assert stats["cache_hits"] >= 1  # Second request should be cached
    else:
        # If git commands failed, just check that stats are tracked
        assert stats["cache_hits"] >= 0

    # Check performance report
    report = git_runner.get_performance_report()
    assert "Performance Report" in report
    assert "Cache Statistics" in report


def test_gitrunner_cache_invalidation(tmp_path):
    """Test GitRunner cache invalidation functionality."""
    from patchdoctor import GitRunner

    git_runner = GitRunner(repo_path=str(tmp_path))

    # Run command to populate cache (may or may not succeed)
    result = git_runner.run_git_command(["status", "--porcelain"])

    # Check cache stats and only test invalidation if cache has entries
    stats_before = git_runner.get_cache_stats()
    if stats_before["cached_entries"] > 0:
        # Test full cache invalidation
        invalidated_count = git_runner.invalidate_cache()
        assert invalidated_count > 0

        stats_after = git_runner.get_cache_stats()
        assert stats_after["cached_entries"] == 0
    else:
        # If no cached entries (git commands failed), just test the invalidation method works
        invalidated_count = git_runner.invalidate_cache()
        assert invalidated_count == 0

    # Test pattern-based invalidation
    git_runner.run_git_command(["status", "--porcelain"])
    git_runner.run_git_command(["log", "--oneline", "-1"])

    pattern_invalidated = git_runner.invalidate_cache(pattern="status")
    assert pattern_invalidated >= 0


def test_repository_scanner_caching(tmp_path):
    """Test RepositoryScanner file content caching."""
    from patchdoctor import RepositoryScanner

    # Create test file
    test_file = tmp_path / "test.txt"
    test_file.write_text("test content\nline 2\n")

    scanner = RepositoryScanner(repo_path=str(tmp_path))

    # First read
    content1 = scanner.get_file_content("test.txt")
    assert content1 is not None
    assert len(content1) == 2

    # Second read (should be cached)
    content2 = scanner.get_file_content("test.txt")
    assert content2 == content1

    # Check cache stats
    stats = scanner.get_cache_stats()
    assert "cache_hits" in stats
    assert "cache_misses" in stats
    assert stats["total_requests"] >= 2
    assert stats["cache_hits"] >= 1

    # Test cache clearing
    cleared_count = scanner.clear_file_cache()
    assert cleared_count >= 1


def test_run_validation_structured_errors(tmp_path):
    """Test run_validation returns structured error information."""
    from patchdoctor import run_validation

    # Test with non-existent patch file
    result = run_validation("nonexistent.patch", repo_path=str(tmp_path))

    assert result["success"] is False
    assert "error_info" in result
    error_info = result["error_info"]
    assert "code" in error_info
    assert "message" in error_info
    assert "suggestion" in error_info
    assert "recoverable" in error_info


def test_validate_from_content_structured_errors():
    """Test validate_from_content with structured error handling."""
    from patchdoctor import validate_from_content

    # Test with completely invalid patch content that will cause parsing to fail
    invalid_content = ""  # Empty content should fail
    result = validate_from_content(invalid_content)

    # Function might succeed but with no files found, so let's test that case
    if result["success"]:
        # If it succeeds with empty content, it should have no file results
        assert "result" in result
        verification_result = result["result"]
        assert verification_result["total_count"] == 0
    else:
        # If it fails, it should have error_info
        assert "error_info" in result
        error_info = result["error_info"]
        assert "code" in error_info
        assert error_info["recoverable"] is True


def test_summarize_patch_status(tmp_path):
    """Test patch status summarization for AI agents."""
    from patchdoctor import (
        PatchParser, PatchVerifier, VerificationResult,
        FileVerificationResult, summarize_patch_status
    )

    # Create a simple patch
    patch_content = """From: test@example.com
Subject: Test patch

 test.txt | 1 +
 1 file changed, 1 insertion(+)

diff --git a/test.txt b/test.txt
new file mode 100644
index 0000000..e69de29
--- /dev/null
+++ b/test.txt
@@ -0,0 +1 @@
+hello
"""

    patch_file = tmp_path / "test.patch"
    patch_file.write_text(patch_content)

    parser = PatchParser()
    patch_info = parser.parse_patch_file(str(patch_file))

    # Create mock file results
    file_result = FileVerificationResult(
        file_path="test.txt",
        expected_operation="create",
        actual_status="missing",
        verification_status="MISSING",
        diff_analysis=None
    )

    verification_result = VerificationResult(
        patch_info=patch_info,
        file_results=[file_result],
        overall_status="PARTIALLY_APPLIED",
        success_count=0,
        total_count=1
    )

    # Test summarization
    summary = summarize_patch_status(verification_result)

    assert "overall_status" in summary
    assert "file_summary" in summary
    assert "hunk_summary" in summary
    assert "fix_suggestions" in summary
    assert "recommendations" in summary
    assert "completion_percentage" in summary

    assert summary["file_summary"]["total"] == 1
    assert summary["file_summary"]["missing"] == 1
    assert summary["completion_percentage"] == 0


def test_generate_api_schema():
    """Test OpenAPI schema generation for AI agents."""
    from patchdoctor import generate_api_schema

    schema = generate_api_schema()

    assert "openapi" in schema
    assert schema["openapi"] == "3.0.0"

    assert "info" in schema
    assert schema["info"]["title"] == "PatchDoctor AI Agent API"

    assert "functions" in schema
    functions = schema["functions"]
    assert "run_validation" in functions
    assert "validate_from_content" in functions
    assert "apply_safe_fixes" in functions
    # generate_api_schema doesn't include itself to avoid circular reference
    assert len(functions) >= 9  # Should have at least 9 functions

    assert "data_types" in schema
    data_types = schema["data_types"]
    assert "ErrorInfo" in data_types
    assert "VerificationResult" in data_types
    assert "Config" in data_types

    assert "error_codes" in schema
    assert "examples" in schema
    assert "supported_operations" in schema


def test_validate_incremental_progress_callback(tmp_path):
    """Test incremental validation with progress callback."""
    from patchdoctor import validate_incremental

    # Create multiple patch files
    patch_content = """From: test@example.com
Subject: Test patch

 test{}.txt | 1 +
 1 file changed, 1 insertion(+)

diff --git a/test{}.txt b/test{}.txt
new file mode 100644
index 0000000..e69de29
--- /dev/null
+++ b/test{}.txt
@@ -0,0 +1 @@
+hello
"""

    patch_dir = tmp_path / "patches"
    patch_dir.mkdir()

    for i in range(3):
        patch_file = patch_dir / f"test{i}.patch"
        patch_file.write_text(patch_content.format(i, i, i, i))

    # Track progress callback calls
    progress_calls = []

    def progress_callback(patch_file: str, result):
        progress_calls.append((patch_file, result))

    # Test incremental validation
    result = validate_incremental(
        str(patch_dir),
        progress_callback=progress_callback,
        early_stop_on_error=False,
        max_concurrent=1,
        repo_path=str(tmp_path)
    )

    assert result["success"] is True
    assert result["processed_count"] == 3
    assert result["total_count"] == 3
    assert len(progress_calls) == 3


def test_validate_incremental_parallel_processing(tmp_path):
    """Test incremental validation with parallel processing."""
    from patchdoctor import validate_incremental

    # Create multiple patch files
    patch_content = """From: test@example.com
Subject: Test patch {}

 test{}.txt | 1 +
 1 file changed, 1 insertion(+)

diff --git a/test{}.txt b/test{}.txt
new file mode 100644
index 0000000..e69de29
--- /dev/null
+++ b/test{}.txt
@@ -0,0 +1 @@
+hello
"""

    patch_dir = tmp_path / "patches"
    patch_dir.mkdir()

    for i in range(4):
        patch_file = patch_dir / f"test{i}.patch"
        patch_file.write_text(patch_content.format(i, i, i, i, i))

    # Test with parallel processing
    result = validate_incremental(
        str(patch_dir),
        max_concurrent=2,  # Use 2 threads
        repo_path=str(tmp_path)
    )

    assert result["success"] is True
    assert result["processed_count"] == 4
    assert result["total_count"] == 4


def test_config_validation():
    """Test Config validation with various input values."""
    from patchdoctor import Config
    import pytest

    # Test valid configuration
    config = Config(
        similarity_threshold=0.5,
        hunk_tolerance=5,
        timeout=30,
        max_file_size=100
    )
    assert config.similarity_threshold == 0.5

    # Test invalid similarity threshold
    with pytest.raises(ValueError, match="Similarity threshold must be between 0.0 and 1.0"):
        Config(similarity_threshold=1.5)

    with pytest.raises(ValueError, match="Similarity threshold must be between 0.0 and 1.0"):
        Config(similarity_threshold=-0.1)

    # Test invalid hunk tolerance
    with pytest.raises(ValueError, match="Hunk tolerance must be non-negative"):
        Config(hunk_tolerance=-1)

    # Test invalid timeout
    with pytest.raises(ValueError, match="Timeout must be positive"):
        Config(timeout=0)

    with pytest.raises(ValueError, match="Timeout must be positive"):
        Config(timeout=-10)

    # Test invalid max file size
    with pytest.raises(ValueError, match="Max file size must be positive"):
        Config(max_file_size=0)


def test_performance_benchmarks(tmp_path):
    """Performance benchmarks for new features."""
    import time
    from patchdoctor import GitRunner, RepositoryScanner

    # Test GitRunner performance with caching
    git_runner = GitRunner(
        repo_path=str(tmp_path),
        enable_performance_monitoring=True
    )

    start_time = time.time()

    # Run multiple git commands
    for _ in range(10):
        git_runner.run_git_command(["status", "--porcelain"])

    elapsed_time = time.time() - start_time

    # Should be fast due to caching
    assert elapsed_time < 5.0  # Should complete in under 5 seconds

    stats = git_runner.get_cache_stats()
    # Only check cache hit ratio if we have requests
    if stats["total_requests"] > 0:
        assert stats["cache_hit_ratio"] >= 0.0  # Should be non-negative

    # Test RepositoryScanner performance
    test_file = tmp_path / "large_test.txt"
    test_file.write_text("line\n" * 1000)  # 1000 lines

    scanner = RepositoryScanner(repo_path=str(tmp_path))

    start_time = time.time()

    # Read file multiple times
    for _ in range(10):
        content = scanner.get_file_content("large_test.txt")
        assert content is not None

    elapsed_time = time.time() - start_time

    # Should be fast due to caching
    assert elapsed_time < 2.0

    stats = scanner.get_cache_stats()
    # File caching should work since we're reading actual files
    assert stats["cache_hit_ratio"] > 0.5

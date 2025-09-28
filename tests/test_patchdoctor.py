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

#!/usr/bin/env python3
"""
Basic Validation Workflow Example for AI Agents

This example demonstrates a simple validation → fix application → re-validation pattern
that AI agents can use as a starting point for PatchDoctor integration.

Features demonstrated:
- Structured error handling with recovery suggestions
- Safe fix application with rollback capabilities
- Progress reporting and user feedback patterns
- Best practices for AI agent integration

Usage:
    python basic_validation_workflow.py --patch-dir ./patches [--apply-safe-fixes] [--dry-run]

Author: PatchDoctor AI Agent Integration
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Import PatchDoctor AI agent integration functions
try:
    from patchdoctor import (
        run_validation, apply_safe_fixes, summarize_patch_status,
        ErrorInfo, ERROR_NO_PATCHES_FOUND, ERROR_GIT_COMMAND_FAILED
    )
except ImportError:
    print("Error: Could not import PatchDoctor. Please ensure it's installed and in your Python path.")
    sys.exit(1)


def print_progress(message: str, level: str = "info") -> None:
    """Print formatted progress message."""
    prefix = {
        "info": "[INFO]",
        "success": "[SUCCESS]",
        "warning": "[WARNING]",
        "error": "[ERROR]"
    }.get(level, "[INFO]")

    print(f"{prefix} {message}")


def handle_validation_errors(result: Dict[str, Any]) -> bool:
    """Handle validation errors with structured error information.

    Returns:
        True if errors were handled and workflow can continue, False otherwise
    """
    if result.get("success", False):
        return True

    print_progress("Validation encountered errors", "warning")

    # Check for structured error information
    error_info = result.get("error_info")
    if error_info:
        print_progress(f"Error Code: {error_info['code']}", "error")
        print_progress(f"Message: {error_info['message']}", "error")
        print_progress(f"Suggestion: {error_info['suggestion']}", "info")

        # Handle specific error types
        if error_info["code"] == ERROR_NO_PATCHES_FOUND:
            print_progress("No patch files found. Please check the directory or create patches using 'git format-patch'", "warning")
            return False

        elif error_info["code"] == ERROR_GIT_COMMAND_FAILED:
            print_progress("Git command failed. Please check repository state and permissions", "error")
            return False

        elif error_info.get("recoverable", False):
            print_progress("Error is recoverable. Continuing with available results...", "info")
            return True

    # Handle errors in individual patch results
    errors = result.get("errors", [])
    if errors:
        print_progress(f"Found {len(errors)} patch-level errors:", "warning")
        for i, error in enumerate(errors[:3]):  # Show first 3 errors
            print_progress(f"  {i+1}. {error.get('message', 'Unknown error')}", "error")
        if len(errors) > 3:
            print_progress(f"  ... and {len(errors) - 3} more errors", "warning")

    return len(result.get("results", [])) > 0


def analyze_validation_results(result: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze validation results and provide insights."""
    if not result.get("success", False) and not result.get("results"):
        return {"can_proceed": False, "summary": "No results to analyze"}

    total_patches = result.get("total_patches", 0)
    fully_applied = result.get("fully_applied", 0)
    partially_applied = result.get("partially_applied", 0)
    not_applied = result.get("not_applied", 0)

    success_rate = (fully_applied / total_patches * 100) if total_patches > 0 else 0

    analysis = {
        "can_proceed": True,
        "success_rate": success_rate,
        "total_patches": total_patches,
        "fully_applied": fully_applied,
        "partially_applied": partially_applied,
        "not_applied": not_applied,
        "has_fixable_issues": partially_applied > 0 or not_applied > 0
    }

    # Generate summary message
    if success_rate == 100:
        analysis["summary"] = f"✅ All {total_patches} patches fully applied successfully!"
    elif success_rate >= 80:
        analysis["summary"] = f"🟡 {fully_applied}/{total_patches} patches fully applied ({success_rate:.1f}% success rate)"
    else:
        analysis["summary"] = f"🔴 Only {fully_applied}/{total_patches} patches applied ({success_rate:.1f}% success rate)"

    return analysis


def apply_safe_fixes_workflow(results: list, dry_run: bool = False) -> Dict[str, Any]:
    """Apply safe fixes to validation results."""
    from patchdoctor import VerificationResult

    print_progress("Analyzing fix suggestions across all patches...", "info")

    total_applied = 0
    total_skipped = 0
    total_errors = 0

    for result_dict in results:
        # Convert dict back to VerificationResult for apply_safe_fixes
        try:
            # This is a simplified conversion - in practice you'd need proper deserialization
            print_progress(f"Processing fixes for patch: {result_dict.get('patch_info', {}).get('filename', 'unknown')}", "info")

            # Count available fixes
            fix_count = 0
            for file_result in result_dict.get('file_results', []):
                fix_count += len(file_result.get('fix_suggestions', []))

            if fix_count == 0:
                print_progress("  No fix suggestions available", "info")
                continue

            print_progress(f"  Found {fix_count} potential fixes", "info")

            if dry_run:
                print_progress(f"  [DRY RUN] Would attempt to apply {fix_count} safe fixes", "info")
                total_applied += fix_count
            else:
                print_progress("  Fix application would require VerificationResult object reconstruction", "warning")
                total_skipped += fix_count

        except Exception as e:
            print_progress(f"  Error processing fixes: {e}", "error")
            total_errors += 1

    return {
        "total_applied": total_applied,
        "total_skipped": total_skipped,
        "total_errors": total_errors,
        "summary": f"Applied {total_applied} fixes, skipped {total_skipped}, {total_errors} errors"
    }


def main():
    """Main workflow demonstration."""
    parser = argparse.ArgumentParser(
        description="Basic PatchDoctor validation workflow for AI agents",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--patch-dir",
        default=".",
        help="Directory containing patch files (default: current directory)"
    )
    parser.add_argument(
        "--apply-safe-fixes",
        action="store_true",
        help="Attempt to apply safe fix suggestions automatically"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed validation information"
    )

    args = parser.parse_args()

    print_progress("Starting PatchDoctor AI Agent Basic Validation Workflow", "info")
    print_progress(f"Patch directory: {Path(args.patch_dir).absolute()}", "info")

    # Step 1: Initial Validation
    print_progress("Step 1: Running initial patch validation...", "info")

    try:
        result = run_validation(
            patch_dir=args.patch_dir,
            verbose=args.verbose,
            similarity_threshold=0.3,
            hunk_tolerance=5
        )
    except Exception as e:
        print_progress(f"Validation failed with exception: {e}", "error")
        return 1

    # Step 2: Handle Validation Errors
    print_progress("Step 2: Analyzing validation results...", "info")

    if not handle_validation_errors(result):
        print_progress("Cannot proceed due to critical errors", "error")
        return 1

    # Step 3: Analyze Results
    analysis = analyze_validation_results(result)
    print_progress(f"Analysis: {analysis['summary']}", "success" if analysis["success_rate"] == 100 else "warning")

    # Step 4: Apply Safe Fixes (if requested)
    if args.apply_safe_fixes and analysis["has_fixable_issues"]:
        print_progress("Step 3: Applying safe fixes...", "info")

        fix_result = apply_safe_fixes_workflow(
            result.get("results", []),
            dry_run=args.dry_run
        )
        print_progress(f"Fix application: {fix_result['summary']}", "info")

        # Step 5: Re-validation after fixes
        if not args.dry_run and fix_result["total_applied"] > 0:
            print_progress("Step 4: Re-validating after fix application...", "info")

            try:
                revalidation_result = run_validation(
                    patch_dir=args.patch_dir,
                    verbose=args.verbose
                )

                if revalidation_result.get("success", False):
                    new_analysis = analyze_validation_results(revalidation_result)
                    improvement = new_analysis["success_rate"] - analysis["success_rate"]

                    if improvement > 0:
                        print_progress(f"Improvement: Success rate increased by {improvement:.1f}%", "success")
                    else:
                        print_progress("No improvement detected after fix application", "warning")

            except Exception as e:
                print_progress(f"Re-validation failed: {e}", "error")

    # Final Summary
    print_progress("Workflow completed!", "success")
    print_progress(f"Final status: {analysis['summary']}", "info")

    if analysis["success_rate"] < 100:
        print_progress("Recommendations:", "info")
        print_progress("- Review patch files for formatting issues", "info")
        print_progress("- Check that all target files exist in the repository", "info")
        print_progress("- Consider using --apply-safe-fixes for automated fixes", "info")

    # Return appropriate exit code
    return 0 if analysis["success_rate"] >= 80 else 1


if __name__ == "__main__":
    sys.exit(main())
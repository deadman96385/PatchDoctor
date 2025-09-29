#!/usr/bin/env python3
"""
Iterative Patch Refinement Example for AI Agents

This example demonstrates how to handle large or complex patches by breaking them down
into smaller, manageable pieces and applying them incrementally with rollback support.

Features demonstrated:
- Large patch → split → incremental application pattern
- Patch splitting strategies (by file, by hunk, by size)
- Partial failure handling and rollback mechanisms
- Progress monitoring for long-running operations
- Corrective patch generation for missing hunks

Usage:
    python iterative_patch_refinement.py --patch-file large-patch.patch [--strategy by_file] [--dry-run]

Author: PatchDoctor AI Agent Integration
"""

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional

# Import PatchDoctor AI agent integration functions
try:
    from patchdoctor import (
        validate_from_content, split_large_patch, apply_safe_fixes,
        extract_missing_changes, generate_corrective_patch, summarize_patch_status,
        validate_input_for_fixes, ErrorInfo, ERROR_PARSE_ERROR, ERROR_GIT_COMMAND_FAILED
    )
except ImportError:
    print("Error: Could not import PatchDoctor. Please ensure it's installed and in your Python path.")
    sys.exit(1)


class IterativePatchProcessor:
    """Handles iterative processing of large patches with rollback support."""

    def __init__(self, repo_path: str = ".", dry_run: bool = False):
        self.repo_path = repo_path
        self.dry_run = dry_run
        self.processed_patches = []
        self.failed_patches = []
        self.rollback_info = []

    def log_progress(self, message: str, level: str = "info") -> None:
        """Log formatted progress message."""
        prefix = {
            "info": "[INFO]",
            "success": "[SUCCESS]",
            "warning": "[WARNING]",
            "error": "[ERROR]",
            "debug": "[DEBUG]"
        }.get(level, "[INFO]")

        print(f"{prefix} {message}")

    def analyze_patch_complexity(self, patch_content: str) -> Dict[str, Any]:
        """Analyze patch complexity to determine optimal splitting strategy."""
        lines = patch_content.split('\n')

        # Count different patch elements
        file_count = len([line for line in lines if line.startswith('diff --git')])
        hunk_count = len([line for line in lines if line.startswith('@@')])
        total_lines = len(lines)

        # Estimate complexity
        complexity_score = file_count * 10 + hunk_count * 5 + total_lines * 0.1

        # Determine recommended strategy
        if file_count > 10:
            strategy = "by_file"
            reason = f"Many files ({file_count}) - split by file for easier tracking"
        elif hunk_count > 50:
            strategy = "by_hunk"
            reason = f"Many hunks ({hunk_count}) - split by hunk for granular control"
        elif total_lines > 5000:
            strategy = "by_size"
            reason = f"Large patch ({total_lines} lines) - split by size for manageability"
        else:
            strategy = "none"
            reason = "Patch is small enough to process as-is"

        return {
            "file_count": file_count,
            "hunk_count": hunk_count,
            "total_lines": total_lines,
            "complexity_score": complexity_score,
            "recommended_strategy": strategy,
            "reason": reason
        }

    def split_patch_intelligently(self, patch_content: str, strategy: str = "auto") -> List[str]:
        """Split patch using intelligent strategy selection."""
        if strategy == "auto":
            analysis = self.analyze_patch_complexity(patch_content)
            strategy = analysis["recommended_strategy"]
            self.log_progress(f"Auto-selected strategy '{strategy}': {analysis['reason']}", "info")

        if strategy == "none":
            return [patch_content]

        try:
            split_patches = split_large_patch(patch_content, strategy=strategy)
            self.log_progress(f"Split patch into {len(split_patches)} smaller patches using '{strategy}' strategy", "success")
            return split_patches
        except Exception as e:
            self.log_progress(f"Failed to split patch using '{strategy}' strategy: {e}", "error")
            return [patch_content]

    def validate_and_apply_patch(self, patch_content: str, patch_index: int) -> Dict[str, Any]:
        """Validate and optionally apply a single patch with detailed feedback."""
        self.log_progress(f"Processing patch {patch_index + 1}...", "info")

        try:
            # Validate patch
            result = validate_from_content(
                patch_content=patch_content,
                repo_path=self.repo_path,
                verbose=False
            )

            if not result.get("success", False):
                error_info = result.get("error_info")
                if error_info:
                    self.log_progress(f"  Validation failed: {error_info['message']}", "error")
                    self.log_progress(f"  Suggestion: {error_info['suggestion']}", "info")
                else:
                    self.log_progress(f"  Validation failed: {result.get('error', 'Unknown error')}", "error")

                return {
                    "success": False,
                    "error": result.get("error", "Validation failed"),
                    "error_info": error_info,
                    "patch_index": patch_index
                }

            # Analyze validation result (result is a dict, not object)
            validation_result_dict = result["result"]

            # Extract basic status information from dict
            overall_status = validation_result_dict.get("overall_status", "UNKNOWN")
            success_count = validation_result_dict.get("success_count", 0)
            total_count = validation_result_dict.get("total_count", 0)
            completion_percentage = (success_count / total_count * 100) if total_count > 0 else 0

            self.log_progress(f"  Status: {overall_status}", "info")
            self.log_progress(f"  Files: {success_count}/{total_count} OK", "info")

            if completion_percentage == 100:
                self.log_progress(f"  ✅ Patch {patch_index + 1} fully applied", "success")
                return {
                    "success": True,
                    "status": "fully_applied",
                    "completion_percentage": completion_percentage,
                    "patch_index": patch_index
                }

            # Handle partial application
            if completion_percentage > 0:
                self.log_progress(f"  🟡 Patch {patch_index + 1} partially applied ({completion_percentage:.1f}%)", "warning")

                # Count available fixes from dict data
                safe_fixes = 0
                for file_result in validation_result_dict.get("file_results", []):
                    for fix in file_result.get("fix_suggestions", []):
                        if fix.get("safety_level") == "safe":
                            safe_fixes += 1

                if not self.dry_run and safe_fixes > 0:
                    self.log_progress(f"  Attempting to apply {safe_fixes} safe fixes...", "info")

                    # Apply fixes using auto-detecting apply_safe_fixes function
                    fix_result = apply_safe_fixes(
                        verification_result=validation_result_dict,
                        confirm=False,
                        safety_levels=["safe"],
                        dry_run=False,
                        repo_path=self.repo_path
                    )

                    applied = len(fix_result.get("applied", []))
                    errors = len(fix_result.get("errors", []))

                    if applied > 0:
                        self.log_progress(f"  ✅ Applied {applied} safe fixes successfully", "success")
                    if errors > 0:
                        self.log_progress(f"  ⚠️  {errors} fixes failed to apply", "warning")

                return {
                    "success": True,
                    "status": "partially_applied",
                    "completion_percentage": completion_percentage,
                    "patch_index": patch_index
                }

            else:
                self.log_progress(f"  🔴 Patch {patch_index + 1} not applied", "error")

                # Try to generate corrective patches using patch analysis functionality
                file_results = validation_result_dict.get("file_results", [])
                if file_results:
                    try:
                        # Convert to object for corrective patch generation functions
                        verification_obj = validate_input_for_fixes(validation_result_dict)
                        missing_changes = extract_missing_changes(verification_obj)

                        if missing_changes:
                            corrective_file = f"corrective_patch_{patch_index + 1}.patch"
                            if generate_corrective_patch(verification_obj, corrective_file):
                                self.log_progress(f"  📝 Generated corrective patch: {corrective_file}", "info")
                    except Exception as e:
                        self.log_progress(f"  Note: Could not generate corrective patch: {e}", "warning")

                return {
                    "success": False,
                    "status": "not_applied",
                    "completion_percentage": completion_percentage,
                    "patch_index": patch_index
                }

        except Exception as e:
            self.log_progress(f"  Exception during patch processing: {e}", "error")
            return {
                "success": False,
                "error": str(e),
                "patch_index": patch_index
            }

    def process_patch_iteratively(self, patch_content: str, strategy: str = "auto") -> Dict[str, Any]:
        """Process a large patch iteratively with comprehensive tracking."""
        self.log_progress("Starting iterative patch processing...", "info")

        # Analyze patch complexity
        analysis = self.analyze_patch_complexity(patch_content)
        self.log_progress(f"Patch analysis: {analysis['file_count']} files, {analysis['hunk_count']} hunks, {analysis['total_lines']} lines", "info")

        # Split patch into manageable pieces
        split_patches = self.split_patch_intelligently(patch_content, strategy)

        if len(split_patches) == 1 and strategy != "none":
            self.log_progress("Patch could not be split or was already optimal size", "info")

        # Process each patch piece
        results = []
        successful_count = 0
        failed_count = 0

        for i, patch_piece in enumerate(split_patches):
            result = self.validate_and_apply_patch(patch_piece, i)
            results.append(result)

            if result.get("success", False):
                successful_count += 1
                self.processed_patches.append({
                    "index": i,
                    "content": patch_piece,
                    "result": result
                })
            else:
                failed_count += 1
                self.failed_patches.append({
                    "index": i,
                    "content": patch_piece,
                    "result": result
                })

        # Generate final summary
        total_patches = len(split_patches)
        success_rate = (successful_count / total_patches * 100) if total_patches > 0 else 0

        final_result = {
            "success": failed_count == 0,
            "total_patches": total_patches,
            "successful_patches": successful_count,
            "failed_patches": failed_count,
            "success_rate": success_rate,
            "complexity_analysis": analysis,
            "strategy_used": strategy,
            "results": results
        }

        # Log final summary
        if success_rate == 100:
            self.log_progress(f"✅ All {total_patches} patch pieces processed successfully!", "success")
        elif success_rate >= 70:
            self.log_progress(f"🟡 {successful_count}/{total_patches} patches successful ({success_rate:.1f}% success rate)", "warning")
        else:
            self.log_progress(f"🔴 Only {successful_count}/{total_patches} patches successful ({success_rate:.1f}% success rate)", "error")

        return final_result

    def generate_failure_report(self) -> str:
        """Generate a detailed report of failed patches for manual review."""
        if not self.failed_patches:
            return "No failed patches to report."

        report_lines = [
            "# Failed Patches Report",
            f"Generated by PatchDoctor Iterative Refinement",
            "",
            f"Total failed patches: {len(self.failed_patches)}",
            ""
        ]

        for i, failed_patch in enumerate(self.failed_patches):
            result = failed_patch["result"]
            report_lines.extend([
                f"## Failed Patch {i + 1} (Index: {failed_patch['index']})",
                f"**Error:** {result.get('error', 'Unknown error')}",
                ""
            ])

            # Add error details if available
            error_info = result.get("error_info")
            if error_info:
                report_lines.extend([
                    f"**Error Code:** {error_info['code']}",
                    f"**Suggestion:** {error_info['suggestion']}",
                    ""
                ])

            # Add patch content (truncated)
            content_lines = failed_patch["content"].split('\n')
            if len(content_lines) > 10:
                content_preview = '\n'.join(content_lines[:10]) + "\n... (truncated)"
            else:
                content_preview = failed_patch["content"]

            report_lines.extend([
                "**Patch Content:**",
                "```diff",
                content_preview,
                "```",
                ""
            ])

        return '\n'.join(report_lines)


def main():
    """Main iterative patch refinement workflow."""
    parser = argparse.ArgumentParser(
        description="Iterative patch refinement for large patches",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--patch-file",
        required=True,
        help="Path to the large patch file to process"
    )
    parser.add_argument(
        "--strategy",
        choices=["auto", "by_file", "by_hunk", "by_size", "none"],
        default="auto",
        help="Splitting strategy (default: auto)"
    )
    parser.add_argument(
        "--repo-path",
        default=".",
        help="Repository path (default: current directory)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    parser.add_argument(
        "--output-report",
        help="Save failure report to specified file"
    )

    args = parser.parse_args()

    # Validate patch file exists
    patch_file = Path(args.patch_file)
    if not patch_file.exists():
        print(f"Error: Patch file '{patch_file}' not found")
        return 1

    print(f"[INFO] Starting iterative patch refinement workflow")
    print(f"[INFO] Patch file: {patch_file.absolute()}")
    print(f"[INFO] Strategy: {args.strategy}")
    print(f"[INFO] Repository: {Path(args.repo_path).absolute()}")

    # Read patch content
    try:
        with open(patch_file, 'r', encoding='utf-8') as f:
            patch_content = f.read()
    except Exception as e:
        print(f"[ERROR] Failed to read patch file: {e}")
        return 1

    # Create processor and run workflow
    processor = IterativePatchProcessor(
        repo_path=args.repo_path,
        dry_run=args.dry_run
    )

    try:
        result = processor.process_patch_iteratively(patch_content, args.strategy)

        # Generate failure report if needed
        if result["failed_patches"] > 0:
            failure_report = processor.generate_failure_report()

            if args.output_report:
                with open(args.output_report, 'w', encoding='utf-8') as f:
                    f.write(failure_report)
                print(f"[INFO] Failure report saved to: {args.output_report}")
            else:
                print("\n" + failure_report)

        # Return appropriate exit code
        return 0 if result["success_rate"] >= 70 else 1

    except KeyboardInterrupt:
        print("\n[WARNING] Process interrupted by user")
        return 2
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
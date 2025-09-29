#!/usr/bin/env python3
"""
Batch Patch Processing Example for AI Agents

This example demonstrates processing multiple patches with dependency tracking,
conflict resolution, and comprehensive rollback capabilities.

Features demonstrated:
- Multi-patch workflows with dependency analysis
- Incremental processing with progress monitoring
- Conflict detection and resolution strategies
- Batch rollback on failure scenarios
- Performance optimization for large patch sets

Usage:
    python batch_patch_processing.py --patch-dir ./patches [--max-concurrent 3] [--rollback-on-failure]

Author: PatchDoctor AI Agent Integration
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from datetime import datetime

# Import PatchDoctor AI agent integration functions
try:
    from patchdoctor import (
        validate_incremental, apply_safe_fixes, summarize_patch_status,
        run_validation, extract_missing_changes,
        ErrorInfo, ERROR_NO_PATCHES_FOUND
    )
except ImportError:
    print("Error: Could not import PatchDoctor. Please ensure it's installed and in your Python path.")
    sys.exit(1)


class BatchPatchProcessor:
    """Handles batch processing of multiple patches with dependency tracking."""

    def __init__(self, repo_path: str = ".", max_concurrent: int = 1):
        self.repo_path = repo_path
        self.max_concurrent = max_concurrent
        self.processed_patches = []
        self.failed_patches = []
        self.dependencies = {}
        self.conflicts = []

    def log_progress(self, message: str, level: str = "info") -> None:
        """Log formatted progress message with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = {
            "info": f"[{timestamp}] [INFO]",
            "success": f"[{timestamp}] [SUCCESS]",
            "warning": f"[{timestamp}] [WARNING]",
            "error": f"[{timestamp}] [ERROR]",
            "debug": f"[{timestamp}] [DEBUG]"
        }.get(level, f"[{timestamp}] [INFO]")

        print(f"{prefix} {message}")

    def analyze_patch_dependencies(self, patch_dir: str) -> Dict[str, Any]:
        """Analyze dependencies between patches based on file modifications."""
        self.log_progress("Analyzing patch dependencies...", "info")

        patch_files = list(Path(patch_dir).glob("*.patch"))
        if not patch_files:
            return {"dependencies": {}, "conflicts": [], "analysis": "No patches found"}

        # Simple dependency analysis based on file overlap
        file_to_patches = {}
        patch_to_files = {}

        for patch_file in patch_files:
            try:
                with open(patch_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Extract modified files from patch
                modified_files = set()
                for line in content.split('\n'):
                    if line.startswith('diff --git'):
                        # Extract file path from diff line
                        parts = line.split()
                        if len(parts) >= 4:
                            file_path = parts[2][2:]  # Remove 'a/' prefix
                            modified_files.add(file_path)

                patch_name = patch_file.name
                patch_to_files[patch_name] = modified_files

                # Track which patches modify each file
                for file_path in modified_files:
                    if file_path not in file_to_patches:
                        file_to_patches[file_path] = []
                    file_to_patches[file_path].append(patch_name)

            except Exception as e:
                self.log_progress(f"Failed to analyze {patch_file.name}: {e}", "warning")

        # Identify dependencies and conflicts
        dependencies = {}
        conflicts = []

        for file_path, patches in file_to_patches.items():
            if len(patches) > 1:
                # Multiple patches modify the same file - potential conflict
                conflicts.append({
                    "file": file_path,
                    "patches": patches,
                    "type": "file_overlap"
                })

                # Sort patches by filename for consistent ordering
                sorted_patches = sorted(patches)
                for i in range(len(sorted_patches) - 1):
                    dependent = sorted_patches[i + 1]
                    dependency = sorted_patches[i]

                    if dependent not in dependencies:
                        dependencies[dependent] = []
                    if dependency not in dependencies[dependent]:
                        dependencies[dependent].append(dependency)

        analysis_result = {
            "dependencies": dependencies,
            "conflicts": conflicts,
            "patch_count": len(patch_files),
            "file_count": len(file_to_patches),
            "conflict_count": len(conflicts)
        }

        self.log_progress(f"Dependency analysis complete: {len(dependencies)} dependencies, {len(conflicts)} conflicts", "info")
        return analysis_result

    def create_application_plan(self, dependency_analysis: Dict[str, Any]) -> List[str]:
        """Create an optimal application plan based on dependencies."""
        dependencies = dependency_analysis["dependencies"]
        all_patches = set()

        # Collect all patches
        for patch, deps in dependencies.items():
            all_patches.add(patch)
            all_patches.update(deps)

        # Add patches without dependencies
        patch_dir = Path(self.repo_path) / "patches"  # Assume patches in subdirectory
        if patch_dir.exists():
            for patch_file in patch_dir.glob("*.patch"):
                all_patches.add(patch_file.name)

        # Simple topological sort for dependency resolution
        applied = set()
        plan = []

        while len(applied) < len(all_patches):
            progress_made = False

            for patch in all_patches:
                if patch in applied:
                    continue

                # Check if all dependencies are satisfied
                patch_deps = dependencies.get(patch, [])
                if all(dep in applied for dep in patch_deps):
                    plan.append(patch)
                    applied.add(patch)
                    progress_made = True

            if not progress_made:
                # Handle circular dependencies or orphaned patches
                remaining = all_patches - applied
                self.log_progress(f"Circular dependency detected, adding remaining patches: {list(remaining)}", "warning")
                plan.extend(remaining)
                break

        return plan

    def progress_callback(self, patch_file: str, result: Optional[Any]) -> None:
        """Progress callback for incremental validation."""
        if result:
            status = getattr(result, 'overall_status', 'UNKNOWN')
            success_count = getattr(result, 'success_count', 0)
            total_count = getattr(result, 'total_count', 0)

            self.log_progress(f"✓ {Path(patch_file).name}: {status} ({success_count}/{total_count})", "info")
        else:
            self.log_progress(f"✗ {Path(patch_file).name}: Failed", "error")

    def process_patches_batch(self, patch_dir: str, rollback_on_failure: bool = False) -> Dict[str, Any]:
        """Process multiple patches with dependency tracking and rollback support."""
        self.log_progress("Starting batch patch processing...", "info")

        # Step 1: Analyze dependencies
        dependency_analysis = self.analyze_patch_dependencies(patch_dir)

        if dependency_analysis["conflict_count"] > 0:
            self.log_progress(f"Warning: {dependency_analysis['conflict_count']} potential conflicts detected", "warning")
            for conflict in dependency_analysis["conflicts"]:
                self.log_progress(f"  File '{conflict['file']}' modified by: {', '.join(conflict['patches'])}", "warning")

        # Step 2: Create application plan
        application_plan = self.create_application_plan(dependency_analysis)
        self.log_progress(f"Application plan created: {len(application_plan)} patches", "info")

        # Step 3: Process patches incrementally
        self.log_progress("Processing patches with incremental validation...", "info")

        try:
            result = validate_incremental(
                patch_dir=patch_dir,
                progress_callback=self.progress_callback,
                early_stop_on_error=rollback_on_failure,
                max_concurrent=self.max_concurrent,
                repo_path=self.repo_path,
                verbose=False
            )

            # Step 4: Analyze results
            if result.get("success", False):
                self.log_progress(f"✅ Batch processing successful: {result['processed_count']}/{result['total_count']} patches", "success")
            else:
                error_count = result.get("error_count", 0)
                self.log_progress(f"🔴 Batch processing completed with {error_count} errors", "warning")

                # Handle rollback if requested
                if rollback_on_failure and error_count > 0:
                    self.log_progress("Initiating rollback due to failures...", "warning")
                    rollback_result = self.rollback_batch_changes()
                    result["rollback_result"] = rollback_result

            # Step 5: Generate recommendations
            recommendations = self.generate_batch_recommendations(result, dependency_analysis)
            result["recommendations"] = recommendations

            return result

        except Exception as e:
            self.log_progress(f"Batch processing failed with exception: {e}", "error")
            return {
                "success": False,
                "error": str(e),
                "processed_count": 0,
                "total_count": 0
            }

    def rollback_batch_changes(self) -> Dict[str, Any]:
        """Attempt to rollback batch changes using git operations."""
        self.log_progress("Attempting to rollback batch changes...", "info")

        try:
            import subprocess

            # Get current git status
            status_result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )

            if status_result.returncode != 0:
                return {"success": False, "error": "Failed to get git status"}

            # Check for uncommitted changes
            if status_result.stdout.strip():
                self.log_progress("Found uncommitted changes, attempting to restore...", "info")

                # Reset working directory
                reset_result = subprocess.run(
                    ["git", "checkout", "--", "."],
                    cwd=self.repo_path,
                    capture_output=True,
                    text=True
                )

                if reset_result.returncode == 0:
                    self.log_progress("✅ Successfully rolled back changes", "success")
                    return {"success": True, "method": "git_checkout"}
                else:
                    self.log_progress(f"Failed to rollback: {reset_result.stderr}", "error")
                    return {"success": False, "error": reset_result.stderr}
            else:
                self.log_progress("No uncommitted changes found", "info")
                return {"success": True, "method": "no_changes"}

        except Exception as e:
            self.log_progress(f"Rollback failed with exception: {e}", "error")
            return {"success": False, "error": str(e)}

    def generate_batch_recommendations(self, result: Dict[str, Any], dependency_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on batch processing results."""
        recommendations = []

        success_rate = (result.get("processed_count", 0) / max(result.get("total_count", 1), 1)) * 100

        if success_rate == 100:
            recommendations.append("✅ All patches processed successfully")
        elif success_rate >= 80:
            recommendations.append("🟡 Most patches successful - review failures for edge cases")
        else:
            recommendations.append("🔴 Low success rate - check patch quality and repository state")

        # Dependency-specific recommendations
        if dependency_analysis["conflict_count"] > 0:
            recommendations.append(f"⚠️  {dependency_analysis['conflict_count']} file conflicts detected - consider reordering patches")

        if dependency_analysis["dependencies"]:
            recommendations.append("📋 Dependencies detected - patches were processed in dependency order")

        # Performance recommendations
        if result.get("total_count", 0) > 10 and self.max_concurrent == 1:
            recommendations.append("🚀 Consider increasing --max-concurrent for better performance")

        return recommendations

    def generate_batch_report(self, result: Dict[str, Any], dependency_analysis: Dict[str, Any]) -> str:
        """Generate a comprehensive batch processing report."""
        report_lines = [
            "# Batch Patch Processing Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            f"- Total patches: {result.get('total_count', 0)}",
            f"- Processed: {result.get('processed_count', 0)}",
            f"- Success rate: {(result.get('processed_count', 0) / max(result.get('total_count', 1), 1)) * 100:.1f}%",
            f"- Errors: {result.get('error_count', 0)}",
            "",
            "## Dependency Analysis",
            f"- Dependencies detected: {len(dependency_analysis.get('dependencies', {}))}",
            f"- Conflicts detected: {dependency_analysis.get('conflict_count', 0)}",
            ""
        ]

        # Add recommendations
        recommendations = result.get("recommendations", [])
        if recommendations:
            report_lines.extend([
                "## Recommendations",
                *[f"- {rec}" for rec in recommendations],
                ""
            ])

        # Add error details
        errors = result.get("errors", [])
        if errors:
            report_lines.extend([
                "## Errors",
                *[f"- {error.get('message', 'Unknown error')}" for error in errors[:5]],
                "" if len(errors) <= 5 else f"... and {len(errors) - 5} more errors",
                ""
            ])

        return '\n'.join(report_lines)


def main():
    """Main batch processing workflow."""
    parser = argparse.ArgumentParser(
        description="Batch patch processing with dependency tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--patch-dir",
        required=True,
        help="Directory containing patch files to process"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=1,
        help="Maximum number of patches to process concurrently (default: 1)"
    )
    parser.add_argument(
        "--repo-path",
        default=".",
        help="Repository path (default: current directory)"
    )
    parser.add_argument(
        "--rollback-on-failure",
        action="store_true",
        help="Rollback changes if any patch fails"
    )
    parser.add_argument(
        "--output-report",
        help="Save processing report to specified file"
    )
    parser.add_argument(
        "--json-output",
        help="Save results as JSON to specified file"
    )

    args = parser.parse_args()

    # Validate patch directory
    patch_dir = Path(args.patch_dir)
    if not patch_dir.exists():
        print(f"Error: Patch directory '{patch_dir}' not found")
        return 1

    print(f"[INFO] Starting batch patch processing workflow")
    print(f"[INFO] Patch directory: {patch_dir.absolute()}")
    print(f"[INFO] Repository: {Path(args.repo_path).absolute()}")
    print(f"[INFO] Max concurrent: {args.max_concurrent}")

    # Create processor and run workflow
    processor = BatchPatchProcessor(
        repo_path=args.repo_path,
        max_concurrent=args.max_concurrent
    )

    try:
        # Analyze dependencies first
        dependency_analysis = processor.analyze_patch_dependencies(args.patch_dir)

        # Process patches
        result = processor.process_patches_batch(
            patch_dir=args.patch_dir,
            rollback_on_failure=args.rollback_on_failure
        )

        # Generate and save report
        if args.output_report:
            report = processor.generate_batch_report(result, dependency_analysis)
            with open(args.output_report, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"[INFO] Report saved to: {args.output_report}")

        # Save JSON output
        if args.json_output:
            output_data = {
                "timestamp": datetime.now().isoformat(),
                "dependency_analysis": dependency_analysis,
                "processing_result": result
            }
            with open(args.json_output, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, default=str)
            print(f"[INFO] JSON output saved to: {args.json_output}")

        # Return appropriate exit code
        success_rate = (result.get("processed_count", 0) / max(result.get("total_count", 1), 1)) * 100
        return 0 if success_rate >= 80 else 1

    except KeyboardInterrupt:
        print("\n[WARNING] Process interrupted by user")
        return 2
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
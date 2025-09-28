#!/usr/bin/env python3
"""
CI Integration Example for AI Agents

This example demonstrates how to integrate PatchDoctor with CI/CD pipelines,
providing structured logging, JSON output, and appropriate exit codes.

Features demonstrated:
- Automated validation in CI/CD pipelines
- Structured JSON output for parsing by CI tools
- Appropriate exit codes for pipeline decision making
- Integration with common CI platforms (GitHub Actions, GitLab CI, Jenkins)
- Performance monitoring and timeout handling

Usage:
    python ci_integration_example.py --patch-dir ./patches --json-output results.json [--timeout 300]

Author: PatchDoctor AI Agent Integration
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# Import PatchDoctor AI agent integration functions
try:
    from patchdoctor import (
        run_validation, validate_incremental, summarize_patch_status,
        ErrorInfo, ERROR_NO_PATCHES_FOUND, ERROR_GIT_TIMEOUT
    )
except ImportError:
    print("Error: Could not import PatchDoctor. Please ensure it's installed and in your Python path.")
    sys.exit(1)


class CIPatchValidator:
    """Handles patch validation specifically for CI/CD environments."""

    def __init__(self, timeout: int = 300, verbose: bool = False):
        self.timeout = timeout
        self.verbose = verbose
        self.start_time = time.time()
        self.ci_env = self.detect_ci_environment()

    def detect_ci_environment(self) -> Dict[str, str]:
        """Detect which CI environment we're running in."""
        ci_info = {"platform": "unknown", "job_id": "", "build_url": ""}

        # GitHub Actions
        if os.getenv("GITHUB_ACTIONS"):
            ci_info.update({
                "platform": "github-actions",
                "job_id": os.getenv("GITHUB_RUN_ID", ""),
                "build_url": f"https://github.com/{os.getenv('GITHUB_REPOSITORY', '')}/actions/runs/{os.getenv('GITHUB_RUN_ID', '')}"
            })

        # GitLab CI
        elif os.getenv("GITLAB_CI"):
            ci_info.update({
                "platform": "gitlab-ci",
                "job_id": os.getenv("CI_JOB_ID", ""),
                "build_url": os.getenv("CI_JOB_URL", "")
            })

        # Jenkins
        elif os.getenv("JENKINS_URL"):
            ci_info.update({
                "platform": "jenkins",
                "job_id": os.getenv("BUILD_ID", ""),
                "build_url": os.getenv("BUILD_URL", "")
            })

        # Azure DevOps
        elif os.getenv("AZURE_HTTP_USER_AGENT"):
            ci_info.update({
                "platform": "azure-devops",
                "job_id": os.getenv("BUILD_BUILDID", ""),
                "build_url": f"{os.getenv('SYSTEM_TEAMFOUNDATIONCOLLECTIONURI', '')}{os.getenv('SYSTEM_TEAMPROJECT', '')}/_build/results?buildId={os.getenv('BUILD_BUILDID', '')}"
            })

        return ci_info

    def log_ci_message(self, message: str, level: str = "info") -> None:
        """Log messages in CI-friendly format."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elapsed = time.time() - self.start_time

        # Format for different CI platforms
        if self.ci_env["platform"] == "github-actions":
            if level == "error":
                print(f"::error::{message}")
            elif level == "warning":
                print(f"::warning::{message}")
            else:
                print(f"::notice::{message}")
        else:
            # Generic format for other CI systems
            prefix = {
                "info": "[INFO]",
                "success": "[SUCCESS]",
                "warning": "[WARNING]",
                "error": "[ERROR]"
            }.get(level, "[INFO]")

            print(f"{prefix} [{elapsed:.1f}s] {message}")

        # Always print to stdout for visibility
        if self.verbose:
            print(f"[{timestamp}] {message}", file=sys.stderr)

    def check_timeout(self) -> bool:
        """Check if we're approaching the timeout limit."""
        elapsed = time.time() - self.start_time
        return elapsed > (self.timeout * 0.9)  # 90% of timeout

    def validate_patches_for_ci(self, patch_dir: str, max_concurrent: int = 2) -> Dict[str, Any]:
        """Validate patches with CI-specific optimizations and monitoring."""
        self.log_ci_message("Starting patch validation for CI pipeline", "info")
        self.log_ci_message(f"CI Environment: {self.ci_env['platform']}", "info")

        validation_start = time.time()

        # Use incremental validation for better progress monitoring
        results = []
        processed_count = 0
        total_count = 0

        def progress_callback(patch_file: str, result: Optional[Any]) -> None:
            nonlocal processed_count
            processed_count += 1

            if self.check_timeout():
                self.log_ci_message(f"Approaching timeout limit, processed {processed_count}/{total_count}", "warning")

            if result:
                status = getattr(result, 'overall_status', 'UNKNOWN')
                self.log_ci_message(f"Processed {Path(patch_file).name}: {status}", "info")
            else:
                self.log_ci_message(f"Failed to process {Path(patch_file).name}", "error")

        try:
            # Check if patches exist
            patch_files = list(Path(patch_dir).glob("*.patch"))
            total_count = len(patch_files)

            if total_count == 0:
                self.log_ci_message("No patch files found - CI validation skipped", "warning")
                return {
                    "success": True,
                    "skipped": True,
                    "reason": "no_patches_found",
                    "total_patches": 0,
                    "ci_info": self.ci_env
                }

            self.log_ci_message(f"Found {total_count} patch files to validate", "info")

            # Run validation with timeout protection
            result = validate_incremental(
                patch_dir=patch_dir,
                progress_callback=progress_callback,
                early_stop_on_error=False,  # Continue processing in CI for complete results
                max_concurrent=max_concurrent,
                timeout=min(30, self.timeout // 4)  # Individual patch timeout
            )

            validation_time = time.time() - validation_start

            # Add CI-specific metadata
            result.update({
                "ci_info": self.ci_env,
                "validation_time": validation_time,
                "timeout_used": self.timeout,
                "timestamp": datetime.now().isoformat()
            })

            # Analyze results for CI decision making
            success_rate = (result.get("processed_count", 0) / max(result.get("total_count", 1), 1)) * 100

            if result.get("success", False):
                self.log_ci_message(f"✅ All patches validated successfully in {validation_time:.1f}s", "success")
            elif success_rate >= 80:
                self.log_ci_message(f"🟡 Validation completed with {success_rate:.1f}% success rate", "warning")
            else:
                self.log_ci_message(f"🔴 Validation failed with {success_rate:.1f}% success rate", "error")

            return result

        except Exception as e:
            self.log_ci_message(f"Validation failed with exception: {e}", "error")
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "ci_info": self.ci_env,
                "validation_time": time.time() - validation_start,
                "timestamp": datetime.now().isoformat()
            }

    def generate_ci_artifacts(self, result: Dict[str, Any], output_dir: str) -> Dict[str, str]:
        """Generate CI artifacts including reports and badges."""
        artifacts = {}
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        try:
            # Generate JSON report
            json_report = output_path / "patch_validation_report.json"
            with open(json_report, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, default=str)
            artifacts["json_report"] = str(json_report)

            # Generate JUnit XML for test reporting
            junit_xml = self.generate_junit_xml(result)
            junit_file = output_path / "patch_validation_junit.xml"
            with open(junit_file, 'w', encoding='utf-8') as f:
                f.write(junit_xml)
            artifacts["junit_xml"] = str(junit_file)

            # Generate badge data
            badge_data = self.generate_badge_data(result)
            badge_file = output_path / "patch_validation_badge.json"
            with open(badge_file, 'w', encoding='utf-8') as f:
                json.dump(badge_data, f, indent=2)
            artifacts["badge_data"] = str(badge_file)

            # Generate summary for CI platform display
            summary = self.generate_ci_summary(result)
            summary_file = output_path / "patch_validation_summary.md"
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(summary)
            artifacts["summary"] = str(summary_file)

            self.log_ci_message(f"Generated {len(artifacts)} CI artifacts in {output_dir}", "info")

        except Exception as e:
            self.log_ci_message(f"Failed to generate CI artifacts: {e}", "error")

        return artifacts

    def generate_junit_xml(self, result: Dict[str, Any]) -> str:
        """Generate JUnit XML for CI test reporting."""
        total_count = result.get("total_count", 0)
        error_count = result.get("error_count", 0)
        validation_time = result.get("validation_time", 0)

        xml_lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<testsuite name="patch-validation" tests="{total_count}" failures="{error_count}" time="{validation_time:.2f}">',
        ]

        # Add test cases for each patch
        for i, patch_result in enumerate(result.get("results", [])):
            patch_name = patch_result.get("patch_info", {}).get("filename", f"patch_{i}")
            status = patch_result.get("overall_status", "UNKNOWN")

            if status == "FULLY_APPLIED":
                xml_lines.append(f'  <testcase name="{patch_name}" classname="patch-validation" time="0.1"/>')
            else:
                xml_lines.append(f'  <testcase name="{patch_name}" classname="patch-validation" time="0.1">')
                xml_lines.append(f'    <failure message="Patch not fully applied: {status}"/>')
                xml_lines.append('  </testcase>')

        xml_lines.append('</testsuite>')
        return '\n'.join(xml_lines)

    def generate_badge_data(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate badge data for status displays."""
        success_rate = 0
        if result.get("total_count", 0) > 0:
            success_rate = (result.get("fully_applied", 0) / result["total_count"]) * 100

        if success_rate == 100:
            color = "brightgreen"
            message = "passing"
        elif success_rate >= 80:
            color = "yellow"
            message = f"{success_rate:.0f}% passing"
        else:
            color = "red"
            message = f"{success_rate:.0f}% passing"

        return {
            "schemaVersion": 1,
            "label": "patches",
            "message": message,
            "color": color
        }

    def generate_ci_summary(self, result: Dict[str, Any]) -> str:
        """Generate markdown summary for CI platforms."""
        lines = [
            "# 🔍 Patch Validation Report",
            "",
            f"**Validation Time:** {result.get('validation_time', 0):.1f}s",
            f"**CI Platform:** {result.get('ci_info', {}).get('platform', 'unknown')}",
            f"**Timestamp:** {result.get('timestamp', 'unknown')}",
            "",
            "## 📊 Summary",
            ""
        ]

        total = result.get("total_count", 0)
        if total == 0:
            lines.extend([
                "🟡 **No patches found** - validation skipped",
                "",
                "This may be expected if no patches were generated for this change."
            ])
        else:
            fully_applied = result.get("fully_applied", 0)
            partially_applied = result.get("partially_applied", 0)
            not_applied = result.get("not_applied", 0)
            errors = result.get("error_count", 0)

            lines.extend([
                f"- **Total Patches:** {total}",
                f"- **✅ Fully Applied:** {fully_applied}",
                f"- **🟡 Partially Applied:** {partially_applied}",
                f"- **🔴 Not Applied:** {not_applied}",
                f"- **❌ Errors:** {errors}",
                ""
            ])

            success_rate = (fully_applied / total) * 100
            if success_rate == 100:
                lines.append("🎉 **All patches validated successfully!**")
            elif success_rate >= 80:
                lines.append(f"⚠️ **{success_rate:.1f}% success rate** - some issues detected")
            else:
                lines.append(f"🚨 **{success_rate:.1f}% success rate** - significant issues detected")

        return '\n'.join(lines)

    def determine_exit_code(self, result: Dict[str, Any]) -> int:
        """Determine appropriate exit code for CI pipeline."""
        if result.get("skipped", False):
            return 0  # No patches to validate

        if not result.get("success", False):
            return 1  # Validation failed

        total = result.get("total_count", 0)
        if total == 0:
            return 0  # No patches

        success_rate = (result.get("fully_applied", 0) / total) * 100

        if success_rate == 100:
            return 0  # Perfect success
        elif success_rate >= 80:
            return 0  # Acceptable success rate
        else:
            return 1  # Unacceptable failure rate


def main():
    """Main CI integration workflow."""
    parser = argparse.ArgumentParser(
        description="PatchDoctor CI/CD integration",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--patch-dir",
        default=".",
        help="Directory containing patch files (default: current directory)"
    )
    parser.add_argument(
        "--json-output",
        help="Save JSON results to specified file"
    )
    parser.add_argument(
        "--artifacts-dir",
        default="./ci-artifacts",
        help="Directory to save CI artifacts (default: ./ci-artifacts)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout in seconds (default: 300)"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=2,
        help="Maximum concurrent patch processing (default: 2)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    # Create CI validator
    validator = CIPatchValidator(
        timeout=args.timeout,
        verbose=args.verbose
    )

    try:
        # Run validation
        result = validator.validate_patches_for_ci(
            patch_dir=args.patch_dir,
            max_concurrent=args.max_concurrent
        )

        # Save JSON output if requested
        if args.json_output:
            with open(args.json_output, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, default=str)
            validator.log_ci_message(f"Results saved to {args.json_output}", "info")

        # Generate CI artifacts
        artifacts = validator.generate_ci_artifacts(result, args.artifacts_dir)

        # Set environment variables for downstream CI steps
        if result.get("total_count", 0) > 0:
            success_rate = (result.get("fully_applied", 0) / result["total_count"]) * 100
            print(f"::set-output name=success_rate::{success_rate:.1f}")
            print(f"::set-output name=total_patches::{result.get('total_count', 0)}")
            print(f"::set-output name=validation_passed::{success_rate >= 80}")

        # Return appropriate exit code
        exit_code = validator.determine_exit_code(result)
        validator.log_ci_message(f"CI validation completed with exit code {exit_code}", "info")

        return exit_code

    except KeyboardInterrupt:
        validator.log_ci_message("Validation interrupted by user", "warning")
        return 2
    except Exception as e:
        validator.log_ci_message(f"Unexpected error: {e}", "error")
        return 1


if __name__ == "__main__":
    sys.exit(main())
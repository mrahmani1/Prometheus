#!/usr/bin/env python3
"""
Code quality checks script for Prometheus AI Platform
Runs Black, isort, flake8, pylint, and bandit on Python files
"""

import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


def run_command(cmd: List[str], description: str) -> Tuple[bool, str]:
    """Run a command and return success status and output"""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            cwd=Path(__file__).parent.parent,
        )
        return result.returncode == 0, result.stdout + result.stderr
    except Exception as e:
        return False, str(e)


def find_python_files() -> List[str]:
    """Find all Python files in the project"""
    python_files = []

    # Root level Python files
    for py_file in Path(".").glob("*.py"):
        python_files.append(str(py_file))

    # Service Python files
    services_dir = Path("services")
    if services_dir.exists():
        for py_file in services_dir.rglob("*.py"):
            python_files.append(str(py_file))

    # Scripts Python files
    scripts_dir = Path("scripts")
    if scripts_dir.exists():
        for py_file in scripts_dir.rglob("*.py"):
            python_files.append(str(py_file))

    return python_files


def main():
    """Run all code quality checks"""
    print("🔍 Running Code Quality Checks for Prometheus AI Platform")
    print("=" * 60)

    python_files = find_python_files()
    if not python_files:
        print("❌ No Python files found!")
        return 1

    print(f"📁 Found {len(python_files)} Python files to check")

    checks = [
        {
            "name": "Black (Code Formatting)",
            "cmd": ["black", "--check", "--diff"] + python_files,
            "fix_cmd": ["black"] + python_files,
        },
        {
            "name": "isort (Import Sorting)",
            "cmd": ["isort", "--check-only", "--profile=black"] + python_files,
            "fix_cmd": ["isort", "--profile=black"] + python_files,
        },
        {
            "name": "Flake8 (Style Guide)",
            "cmd": ["flake8", "--max-line-length=88", "--extend-ignore=E203,W503"]
            + python_files,
        },
        {
            "name": "Pylint (Code Quality)",
            "cmd": ["pylint", "--disable=C0114,C0115,C0116,R0903,W1203,W1514"]
            + python_files,
        },
        {
            "name": "Bandit (Security)",
            "cmd": ["bandit", "-r", "--skip=B101,B601"] + python_files,
        },
    ]

    results = []

    for check in checks:
        print(f"\n🧪 Running {check['name']}...")
        success, output = run_command(check["cmd"], check["name"])

        if success:
            print(f"✅ {check['name']}: PASSED")
        else:
            print(f"❌ {check['name']}: FAILED")
            if output.strip():
                print(f"Output:\n{output}")

            # Try to auto-fix if possible
            if "fix_cmd" in check:
                print(f"🔧 Attempting to auto-fix with {check['name']}...")
                fix_success, fix_output = run_command(
                    check["fix_cmd"], f"Fix {check['name']}"
                )
                if fix_success:
                    print(f"✅ Auto-fix successful for {check['name']}")
                    # Re-run the check
                    success, _ = run_command(check["cmd"], check["name"])
                    if success:
                        print(f"✅ {check['name']}: NOW PASSING")
                else:
                    print(f"❌ Auto-fix failed for {check['name']}")

        results.append((check["name"], success))

    # Summary
    print("\n" + "=" * 60)
    print("📊 Code Quality Check Summary:")

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status} {name}")

    print(f"\n🏁 Overall: {passed}/{total} checks passed")

    if passed == total:
        print("🎉 All code quality checks passed!")
        return 0
    else:
        print(f"💥 {total - passed} checks failed. Please review and fix the issues.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

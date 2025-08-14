#!/usr/bin/env python3
"""
Test script to verify the automated QA system implementation
"""

import json
import logging
import subprocess
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_script_execution(script_path: str, timeout: int = 60) -> bool:
    """Test if a script can be executed without errors"""
    try:
        logger.info(f"Testing script: {script_path}")
        result = subprocess.run(
            [sys.executable, script_path, "--help"],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=Path(__file__).parent,
        )

        # Some scripts might not have --help, so we just check if they can be imported
        if result.returncode != 0:
            # Try to import the script as a module
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    f"import sys; sys.path.append('.'); "
                    f"exec(open('{script_path}').read())",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                cwd=Path(__file__).parent,
            )

        status = "✅ PASS" if result.returncode == 0 else "❌ FAIL"
        logger.info(f"Script {script_path}: {status}")
        if result.returncode != 0:
            logger.error(f"Error output: {result.stderr}")

        return result.returncode == 0

    except subprocess.TimeoutExpired:
        logger.warning(f"Script {script_path} timed out")
        return False
    except Exception as e:
        logger.error(f"Error testing {script_path}: {e}")
        return False


def test_qa_orchestrator():
    """Test the QA orchestrator"""
    try:
        logger.info("Testing QA orchestrator import")
        sys.path.append(str(Path(__file__).parent / "services" / "system-monitor"))
        from qa_orchestrator import QAOrchestrator

        orchestrator = QAOrchestrator()
        logger.info("✅ QA orchestrator import successful")

        # Test configuration loading
        if hasattr(orchestrator, "qa_components"):
            logger.info(
                f"✅ QA components configured: {list(orchestrator.qa_components.keys())}"
            )

        return True
    except Exception as e:
        logger.error(f"❌ QA orchestrator test failed: {e}")
        return False


def test_hook_configuration():
    """Test hook configuration files"""
    try:
        hook_files = [
            ".kiro/hooks/automated-qa-hooks.json",
            ".kiro/hooks/qa-automation.md",
        ]

        for hook_file in hook_files:
            if Path(hook_file).exists():
                logger.info(f"✅ Hook file exists: {hook_file}")

                if hook_file.endswith(".json"):
                    with open(hook_file, "r") as f:
                        json.load(f)  # Validate JSON
                    logger.info(f"✅ Valid JSON: {hook_file}")
            else:
                logger.warning(f"⚠️ Hook file missing: {hook_file}")

        return True
    except Exception as e:
        logger.error(f"❌ Hook configuration test failed: {e}")
        return False


def test_directory_structure():
    """Test required directory structure"""
    required_dirs = [
        "scripts",
        "services/system-monitor",
        ".kiro/hooks",
        "services/system-monitor/config",
    ]

    all_exist = True
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            logger.info(f"✅ Directory exists: {dir_path}")
        else:
            logger.error(f"❌ Directory missing: {dir_path}")
            all_exist = False

    return all_exist


def main():
    """Run all QA system tests"""
    logger.info("🔬 Testing Prometheus AI QA System Implementation")
    logger.info("=" * 60)

    tests = [
        ("Directory Structure", test_directory_structure),
        ("Hook Configuration", test_hook_configuration),
        ("QA Orchestrator", test_qa_orchestrator),
    ]

    # Test individual scripts
    scripts_to_test = [
        "scripts/run_automated_tests.py",
        "scripts/validate_models.py",
        "scripts/code_quality_scan.py",
        "scripts/generate_validate_docs.py",
        "scripts/generate_quality_dashboard.py",
    ]

    for script in scripts_to_test:
        if Path(script).exists():
            tests.append(
                (f"Script: {script}", lambda s=script: test_script_execution(s))
            )

    # Run all tests
    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        logger.info(f"\n🧪 Running test: {test_name}")
        try:
            if test_func():
                passed += 1
                logger.info(f"✅ {test_name} PASSED")
            else:
                logger.error(f"❌ {test_name} FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name} FAILED with exception: {e}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info(f"🏁 Test Summary: {passed}/{total} tests passed")

    if passed == total:
        logger.info("🎉 All tests passed! QA system is ready.")
        return 0
    else:
        logger.error(
            f"💥 {total - passed} tests failed. Please review the implementation."
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())

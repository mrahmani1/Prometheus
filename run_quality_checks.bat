@echo off
echo Running Prometheus AI Platform Quality Checks
echo =============================================

echo.
echo 1. Running Black (Code Formatting)...
python -m black demo_qa_system.py test_qa_system.py scripts/

echo.
echo 2. Running isort (Import Sorting)...
python -m isort --profile=black demo_qa_system.py test_qa_system.py scripts/

echo.
echo 3. Running Flake8 (Style Guide)...
python -m flake8 --max-line-length=88 --extend-ignore=E203,W503 demo_qa_system.py test_qa_system.py

echo.
echo 4. Running Security Fixes...
python scripts/fix_security_issues.py

echo.
echo 5. Generating Quality Summary...
python scripts/quality_summary.py

echo.
echo Quality checks completed!
pause
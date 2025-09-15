#!/usr/bin/env python3
"""
Syntax validation script for database performance monitoring system
"""

import ast
import sys
import os
from pathlib import Path

def validate_python_syntax(file_path):
    """Validate Python syntax for a file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        
        # Parse the AST to check for syntax errors
        ast.parse(source, filename=file_path)
        return True, None
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Error: {e}"

def main():
    """Main validation function"""
    files_to_check = [
        'backend/core/database_performance_monitor.py',
        'backend/core/database_optimization_engine.py',
        'backend/core/database_backup_monitor.py',
        'backend/core/database_security_auditor.py',
        'backend/api/database_performance_dashboard.py'
    ]
    
    all_valid = True
    
    print("Validating Python syntax for database performance monitoring system...")
    print("=" * 70)
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            is_valid, error_msg = validate_python_syntax(file_path)
            status = "✓ PASS" if is_valid else "✗ FAIL"
            print(f"{status:<8} {file_path}")
            
            if not is_valid:
                print(f"         Error: {error_msg}")
                all_valid = False
        else:
            print(f"✗ MISSING {file_path}")
            all_valid = False
    
    print("=" * 70)
    
    if all_valid:
        print("✓ All files passed syntax validation!")
        return 0
    else:
        print("✗ Some files failed syntax validation!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
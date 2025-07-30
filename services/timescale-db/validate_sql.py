#!/usr/bin/env python3
"""
Simple SQL syntax validation for TimescaleDB initialization script
"""

import re
import sys

def validate_sql_file(filepath):
    """Basic SQL syntax validation"""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Basic checks
        errors = []
        
        # Check for balanced parentheses
        paren_count = content.count('(') - content.count(')')
        if paren_count != 0:
            errors.append(f"Unbalanced parentheses: {paren_count}")
        
        # Check for basic SQL statement endings
        statements = re.split(r';\s*\n', content)
        for i, stmt in enumerate(statements):
            stmt = stmt.strip()
            if not stmt:
                continue
            
            # Check for common SQL keywords
            if stmt.upper().startswith(('CREATE', 'SELECT', 'INSERT', 'UPDATE', 'DELETE', 'ALTER', 'DROP')):
                # Basic statement validation passed
                continue
            elif stmt.upper().startswith(('--', '/*')):
                # Comment, skip
                continue
            elif stmt.upper().startswith('\\'):
                # PostgreSQL command, skip
                continue
            elif 'COMMIT' in stmt.upper() or 'BEGIN' in stmt.upper():
                # Transaction control, skip
                continue
            else:
                # Check if it's a continuation of previous statement
                if i > 0 and not stmt.upper().startswith(('CREATE', 'SELECT', 'INSERT', 'UPDATE', 'DELETE', 'ALTER', 'DROP')):
                    continue
        
        if errors:
            print(f"SQL validation errors in {filepath}:")
            for error in errors:
                print(f"  - {error}")
            return False
        else:
            print(f"SQL file {filepath} passed basic validation")
            return True
            
    except Exception as e:
        print(f"Error validating {filepath}: {e}")
        return False

if __name__ == "__main__":
    sql_files = [
        "database/timescale/init.sql"
    ]
    
    all_valid = True
    for sql_file in sql_files:
        if not validate_sql_file(sql_file):
            all_valid = False
    
    if all_valid:
        print("All SQL files passed validation")
        sys.exit(0)
    else:
        print("Some SQL files failed validation")
        sys.exit(1)
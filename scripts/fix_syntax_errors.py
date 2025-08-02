#!/usr/bin/env python3
"""
Comprehensive syntax error detection and fixing script for the Prometheus AI platform.
"""

from pathlib import Path
import ast
import os
import re
import sys

def check_syntax_errors(file_path):
    """Check for syntax errors in a Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        ast.parse(content, filename=str(file_path))
        return None
    except SyntaxError as e:
        return e
    except Exception as e:
        return e

def fix_common_syntax_issues(content):
    """Fix common syntax issues in Python code."""
    lines = content.split('\n')
    fixed_lines = []
    
    for i, line in enumerate(lines):
        # Fix common indentation issues
        if line.strip() and not line.startswith(' ') and not line.startswith('\t'):
            # Check if this should be indented based on previous line
            if i > 0 and lines[i-1].strip().endswith(':'):
                line = '    ' + line
        
        # Fix missing commas in dictionaries/lists
        if line.strip().endswith('{') or line.strip().endswith('['):
            fixed_lines.append(line)
            continue
            
        # Fix orphaned dictionary/list items
        if re.match(r'^\s*["\']?\w+["\']?\s*:', line) and i > 0:
            prev_line = lines[i-1].strip()
            if not prev_line.endswith(',') and not prev_line.endswith('{'):
                fixed_lines[-1] = fixed_lines[-1] + ','
        
        fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

def scan_and_fix_directory(directory):
    """Scan directory for Python files and fix syntax errors."""
    issues_found = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = Path(root) / file
                error = check_syntax_errors(file_path)
                
                if error:
                    issues_found.append({
                        'file': str(file_path),
                        'error': str(error),
                        'line': getattr(error, 'lineno', None)
                    })
                    
                    print(f"❌ Syntax error in {file_path}: {error}")
                    
                    # Attempt basic fixes
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        fixed_content = fix_common_syntax_issues(content)
                        
                        # Test if fix worked
                        try:
                            ast.parse(fixed_content)
                            with open(file_path, 'w', encoding='utf-8') as f:
                                f.write(fixed_content)
                            print(f"✅ Fixed syntax error in {file_path}")
                        except:
                            print(f"⚠️  Could not auto-fix {file_path}")
                            
                    except Exception as e:
                        print(f"❌ Error processing {file_path}: {e}")
                else:
                    print(f"✅ {file_path} - OK")
    
    return issues_found

if __name__ == "__main__":
    print("🔍 Scanning for Python syntax errors...")
    
    # Scan services directory
    services_dir = Path("services")
    if services_dir.exists():
        issues = scan_and_fix_directory(services_dir)
        
        if issues:
            print(f"\n📊 Found {len(issues)} files with syntax errors:")
            for issue in issues:
                print(f"  - {issue['file']}: {issue['error']}")
        else:
            print("\n✅ No syntax errors found!")
    else:
        print("❌ Services directory not found!")
#!/usr/bin/env python3
"""
Comprehensive code quality check for Prometheus AI Platform.
"""

import ast
import os
import sys

def comprehensive_quality_check():
    print('=== PROMETHEUS AI PLATFORM - COMPREHENSIVE CODE QUALITY REPORT ===')
    print()
    
    total_files = 0
    syntax_errors = 0
    import_issues = 0
    large_files = 0
    
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                total_files += 1
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Check syntax
                    try:
                        ast.parse(content)
                    except SyntaxError:
                        syntax_errors += 1
                        print(f'   SYNTAX ERROR: {filepath}')
                        continue
                    
                    # Check file size
                    lines = len(content.split('\n'))
                    if lines > 1000:
                        large_files += 1
                    
                    # Check for basic import organization
                    import_lines = []
                    for i, line in enumerate(content.split('\n')):
                        if line.strip().startswith(('import ', 'from ')):
                            import_lines.append((i+1, line.strip()))
                    
                    # Check if imports are grouped properly
                    if len(import_lines) > 1:
                        std_imports = []
                        third_party = []
                        local_imports = []
                        
                        for line_no, line in import_lines:
                            if 'from .' in line or 'from ..' in line:
                                local_imports.append(line_no)
                            elif any(lib in line for lib in ['numpy', 'torch', 'pandas', 'scipy', 'matplotlib', 'sklearn']):
                                third_party.append(line_no)
                            else:
                                std_imports.append(line_no)
                        
                        # Check if imports are not grouped (simplified check)
                        all_import_lines = std_imports + third_party + local_imports
                        if len(all_import_lines) > 3 and all_import_lines != sorted(all_import_lines):
                            import_issues += 1
                
                except Exception as e:
                    print(f'   ERROR processing {filepath}: {e}')
                    continue
    
    print('QUALITY METRICS:')
    print(f'   Total Python files: {total_files}')
    print(f'   Files with syntax errors: {syntax_errors}')
    print(f'   Files with import organization issues: {import_issues}')
    print(f'   Large files (>1000 lines): {large_files}')
    print()
    
    if syntax_errors == 0:
        print('SYNTAX: No syntax errors found - All files compile successfully!')
    else:
        print(f'SYNTAX: {syntax_errors} files have syntax errors')
    
    if import_issues < 10:
        print('IMPORTS: Import organization is good')
    else:
        print(f'IMPORTS: {import_issues} files could benefit from better import organization')
    
    print()
    print('OVERALL CODE QUALITY: ', end='')
    if syntax_errors == 0 and import_issues < 20:
        print('EXCELLENT')
    elif syntax_errors == 0:
        print('GOOD')
    else:
        print('NEEDS IMPROVEMENT')
    
    return syntax_errors, import_issues, total_files

if __name__ == "__main__":
    comprehensive_quality_check()
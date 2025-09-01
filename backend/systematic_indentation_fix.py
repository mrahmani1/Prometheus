#!/usr/bin/env python3
"""Systematic indentation fix for all remaining issues in enkf_knowledge_integration.py"""

import re

def fix_all_indentation_issues(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    # Track fixes
    fixes = []
    
    for i in range(len(lines)):
        line = lines[i].rstrip()
        original_line = line
        
        if not line.strip():
            continue
            
        # Skip if already properly indented (starts with 12+ spaces for functions)
        if re.match(r'^\s{12,}(def|async def) ', line):
            continue
            
        # Fix function definitions that should be at 12 spaces but aren't
        if re.match(r'^\s{4}(def|async def) ', line):
            # Check if this is inside a class method (should be 12 spaces)
            # Look backwards to see if we're in a method context
            in_method_context = False
            for j in range(max(0, i-50), i):
                prev_line = lines[j].strip()
                if prev_line.startswith('def ') or prev_line.startswith('async def '):
                    if lines[j].startswith('    ') and not lines[j].startswith('        '):
                        # This is a class method, so nested functions should be 12 spaces
                        in_method_context = True
                        break
                elif prev_line.startswith('class '):
                    break
            
            if in_method_context:
                lines[i] = '            ' + line.lstrip() + '\n'
                fixes.append(f"Fixed function at line {i+1}: {line.strip()}")
        
        # Fix functions that have wrong indentation (like 8 spaces instead of 12)
        elif re.match(r'^\s{8}(def|async def) ', line) and not re.match(r'^\s{12,}', line):
            # Check context - if inside a method, should be 12 spaces
            in_method_context = False
            for j in range(max(0, i-50), i):
                prev_line = lines[j].strip()
                if prev_line.startswith('def ') or prev_line.startswith('async def '):
                    if lines[j].startswith('    ') and not lines[j].startswith('        '):
                        in_method_context = True
                        break
                elif prev_line.startswith('class '):
                    break
            
            if in_method_context:
                lines[i] = '            ' + line.lstrip() + '\n'
                fixes.append(f"Fixed function at line {i+1}: {line.strip()}")
    
    # Write back
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print(f"Applied {len(fixes)} systematic indentation fixes")
    for fix in fixes[:10]:  # Show first 10
        print(f"  {fix}")
    if len(fixes) > 10:
        print(f"  ... and {len(fixes) - 10} more fixes")

if __name__ == "__main__":
    fix_all_indentation_issues("backend/modules/enkf_knowledge_integration.py")
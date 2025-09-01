#!/usr/bin/env python3
"""Comprehensive indentation fix for remaining issues"""

import ast
import re

def fix_indentation_comprehensive(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    # Start from line 9261 and fix systematically
    start_line = 9260  # 0-indexed
    
    # Check each line and fix common indentation patterns
    for i in range(start_line, len(lines)):
        line = lines[i].rstrip()
        if not line.strip():
            continue
            
        # Method definitions that should be at 12 spaces (class methods)
        if re.match(r'^\s{8,11}(async def|def) \w+.*:', line):
            # Fix to 12 spaces
            lines[i] = '            ' + line.lstrip() + '\n'
            print(f"Fixed method at line {i+1}")
            
        # Method content that should be at 16 spaces
        elif re.match(r'^\s{8,15}(""".*|try:|if |for |while |return |class |await |self\.)', line):
            if not line.startswith('            '):  # Not already at 12+ spaces
                lines[i] = '                ' + line.lstrip() + '\n'  # 16 spaces
                print(f"Fixed content at line {i+1}")
    
    # Write back
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print("Applied comprehensive indentation fix from line 9261 onwards")

if __name__ == "__main__":
    fix_indentation_comprehensive("backend/modules/enkf_knowledge_integration.py")
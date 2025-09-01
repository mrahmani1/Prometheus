#!/usr/bin/env python3
"""Fix remaining method indentation issues starting from line 9292"""

import re

def fix_remaining_methods(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    # Fix method definitions from line 9292 onwards
    start_line = 9291  # 0-indexed
    
    for i in range(start_line, len(lines)):
        line = lines[i].rstrip()
        if not line.strip():
            continue
            
        # Method definitions that should be at 4 spaces (class methods)
        if re.match(r'^\s{8,}(async def|def) \w+.*:', line):
            # Fix to 4 spaces
            lines[i] = '    ' + line.lstrip() + '\n'
            print(f"Fixed method at line {i+1}")
    
    # Write back
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print("Fixed remaining method indentations")

if __name__ == "__main__":
    fix_remaining_methods("backend/modules/enkf_knowledge_integration.py")
#!/usr/bin/env python3
"""Fix method definition indentation errors systematically"""

import re

def fix_method_indentations(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Fix incorrectly indented method definitions at 8 spaces to 4 spaces
    for i, line in enumerate(lines):
        # Match method definitions with 8-space indent that should be 4-space
        if re.match(r'^        (async def|def) \w+', line):
            # Change to 4-space indent
            lines[i] = '    ' + line[8:]
            print(f"Fixed line {i+1}: {line.strip()[:60]}...")
    
    # Write back
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print("Fixed method indentation systematically")

if __name__ == "__main__":
    fix_method_indentations("backend/modules/enkf_knowledge_integration.py")
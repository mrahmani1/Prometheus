#!/usr/bin/env python3
"""Fix indentation issues in enkf_knowledge_integration.py"""

import re

def fix_indentation(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    fixed_lines = []
    in_class = False
    class_indent = 0
    
    for i, line in enumerate(lines):
        # Check if this is a class definition
        if re.match(r'^(\s*)class\s+\w+.*:', line):
            match = re.match(r'^(\s*)class\s+\w+.*:', line)
            class_indent = len(match.group(1))
            in_class = True
            fixed_lines.append(line)
        # Check if this is a method definition that needs fixing
        elif re.match(r'^(\s*)def\s+\w+.*:', line):
            match = re.match(r'^(\s*)def\s+\w+.*:', line)
            current_indent = len(match.group(1))
            
            # Check if previous non-empty line was a class definition
            prev_line_idx = i - 1
            while prev_line_idx >= 0 and lines[prev_line_idx].strip() == '':
                prev_line_idx -= 1
            
            if prev_line_idx >= 0:
                prev_line = lines[prev_line_idx]
                # If previous line was a class, this def should be indented 4 spaces more
                if re.match(r'^(\s*)class\s+\w+.*:', prev_line):
                    prev_match = re.match(r'^(\s*)class\s+\w+.*:', prev_line)
                    expected_indent = len(prev_match.group(1)) + 4
                    if current_indent != expected_indent:
                        # Fix the indentation
                        line = ' ' * expected_indent + line.lstrip()
            
            fixed_lines.append(line)
        else:
            fixed_lines.append(line)
    
    # Write back the fixed file
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)
    
    print(f"Fixed indentation in {filepath}")

if __name__ == "__main__":
    fix_indentation("backend/modules/enkf_knowledge_integration.py")
#!/usr/bin/env python3
"""Fix specific indentation issue at line 6494"""

def fix_specific_line(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Fix lines 6493-6506 to have proper 8-space indentation for method content
    start_line = 6492  # 0-indexed line 6493
    end_line = 6505    # 0-indexed line 6506
    
    for i in range(start_line, min(end_line + 1, len(lines))):
        line = lines[i]
        if line.strip():  # Only process non-empty lines
            # Ensure proper 8-space indentation for method content
            if line.startswith('        ') and not line.startswith('            '):
                # Already properly indented at 8 spaces
                continue
            elif line.startswith('    ') and not line.startswith('        '):
                # Fix from 4 spaces to 8 spaces
                lines[i] = '    ' + line
                print(f"Fixed line {i+1}: {line.strip()[:50]}...")
            elif line.startswith('            '):
                # Keep 12-space indentation for nested content
                continue
            elif not line.startswith(' '):
                # Fix lines with no indentation
                lines[i] = '        ' + line
                print(f"Fixed line {i+1}: {line.strip()[:50]}...")
    
    # Write back
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print(f"Fixed indentation for lines {start_line+1}-{end_line+1}")

if __name__ == "__main__":
    fix_specific_line("backend/modules/enkf_knowledge_integration.py")
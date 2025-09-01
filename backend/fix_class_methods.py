#!/usr/bin/env python3
"""Fix method indentation in specific class range"""

def fix_method_indentation(filepath, start_line, end_line, correct_indent):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for i in range(start_line - 1, min(end_line, len(lines))):
        line = lines[i]
        # Check if this is a method definition with wrong indentation
        if line.startswith('    def ') or line.startswith('    async def '):
            # Replace 4-space indent with correct indent
            if line.startswith('    async def '):
                lines[i] = ' ' * correct_indent + line[4:]
            elif line.startswith('    def '):
                lines[i] = ' ' * correct_indent + line[4:]
    
    # Write back
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print(f"Fixed method indentation from line {start_line} to {end_line}")

if __name__ == "__main__":
    # Fix AdvancedHomomorphicEncryption class methods (lines 3423-4240)
    fix_method_indentation("backend/modules/enkf_knowledge_integration.py", 3423, 4240, 12)
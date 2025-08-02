#!/usr/bin/env python3
"""
Simple import organization fixer for Python files.
"""

import ast
import os
import re

def fix_import_spacing(filepath):
    """Fix spacing issues in imports."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        modified = False
        new_lines = []
        prev_was_import = False
        in_import_section = False
        import_section_ended = False
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            is_import = stripped.startswith(('import ', 'from '))
            
            if is_import:
                in_import_section = True
                import_section_ended = False
            elif in_import_section and not is_import and stripped and not stripped.startswith('#'):
                import_section_ended = True
                in_import_section = False
            
            # Add blank line after imports section ends
            if import_section_ended and prev_was_import and stripped and not stripped.startswith('#'):
                if not (i > 0 and lines[i-1].strip() == ''):
                    new_lines.append('\n')
                    modified = True
                import_section_ended = False
            
            new_lines.append(line)
            prev_was_import = is_import
        
        if modified:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
            return True
            
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False
    
    return False

def sort_imports_in_file(filepath):
    """Sort imports within their groups."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        tree = ast.parse(content)
        
        # Find import blocks
        import_blocks = []
        current_block = []
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(('import ', 'from ')):
                current_block.append((i, line))
            elif current_block:
                # End of import block
                import_blocks.append(current_block)
                current_block = []
        
        if current_block:
            import_blocks.append(current_block)
        
        modified = False
        
        # Sort each block
        for block in import_blocks:
            if len(block) <= 1:
                continue
                
            # Extract imports and sort them
            import_lines = [item[1] for item in block]
            sorted_lines = sorted(import_lines, key=lambda x: x.strip().lower())
            
            if import_lines != sorted_lines:
                # Update the lines
                for i, (line_no, _) in enumerate(block):
                    lines[line_no] = sorted_lines[i]
                modified = True
        
        if modified:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            return True
            
    except Exception as e:
        print(f"Error sorting imports in {filepath}: {e}")
        return False
    
    return False

def main():
    """Main function to fix import organization."""
    files_processed = 0
    files_fixed = 0
    
    print("Fixing import organization issues...")
    
    for root, dirs, files in os.walk('.'):
        # Skip certain directories
        if any(skip in root for skip in ['.git', '__pycache__', '.pytest_cache', 'node_modules']):
            continue
            
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                files_processed += 1
                
                fixed_spacing = fix_import_spacing(filepath)
                fixed_sorting = sort_imports_in_file(filepath)
                
                if fixed_spacing or fixed_sorting:
                    files_fixed += 1
                    fixes = []
                    if fixed_spacing:
                        fixes.append("spacing")
                    if fixed_sorting:
                        fixes.append("sorting")
                    print(f"Fixed {', '.join(fixes)} in: {filepath}")
    
    print(f"\nProcessed {files_processed} files, fixed {files_fixed} files")

if __name__ == "__main__":
    main()
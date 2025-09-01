#!/usr/bin/env python3
"""Fix method indentation issues in enkf_knowledge_integration.py"""

def fix_indentation_comprehensive(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Track current class and its indentation
    current_class_indent = None
    in_class = False
    
    for i in range(len(lines)):
        line = lines[i]
        
        # Check if this is a class definition
        if 'class ' in line and ':' in line and not line.strip().startswith('#'):
            # Get the indentation of this class
            current_class_indent = len(line) - len(line.lstrip())
            in_class = True
            continue
            
        # If we're in a class, check for method definitions
        if in_class and current_class_indent is not None:
            # Check if this line is a method definition
            if line.strip().startswith('def ') or line.strip().startswith('async def '):
                # Calculate what the indentation should be
                expected_indent = current_class_indent + 4
                current_indent = len(line) - len(line.lstrip())
                
                # Fix if wrong
                if current_indent != expected_indent:
                    lines[i] = ' ' * expected_indent + line.lstrip()
                    print(f"Fixed line {i+1}: {line.strip()[:50]}...")
            
            # Check if we've left the class (back to original indent or less)
            if line.strip() and not line[current_class_indent:current_class_indent+1].isspace() and not line.strip().startswith('#'):
                if len(line) - len(line.lstrip()) <= current_class_indent:
                    in_class = False
                    current_class_indent = None
    
    # Write back
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print("Comprehensive indentation fix complete")

if __name__ == "__main__":
    fix_indentation_comprehensive("backend/modules/enkf_knowledge_integration.py")
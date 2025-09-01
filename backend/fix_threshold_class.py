#!/usr/bin/env python3
"""Fix method indentation in ThresholdHomomorphicEncryption class"""

def fix_threshold_class_indentation(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Fix methods in ThresholdHomomorphicEncryption class (lines 4241-5350)
    for i in range(4240, min(5350, len(lines))):
        line = lines[i]
        # Check if this is a method definition with 4-space indent
        if line.startswith('    def ') or line.startswith('    async def '):
            # Change to 12-space indent
            if line.startswith('    async def '):
                lines[i] = '            ' + line[4:]
            else:
                lines[i] = '            ' + line[4:]
            print(f"Fixed line {i+1}")
    
    # Write back
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print("Fixed ThresholdHomomorphicEncryption class indentation")

if __name__ == "__main__":
    fix_threshold_class_indentation("backend/modules/enkf_knowledge_integration.py")
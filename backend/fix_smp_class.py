#!/usr/bin/env python3
"""Fix SecureMultiPartyComputation class methods"""

def fix_smp_methods(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Fix methods in SecureMultiPartyComputation class (lines 5344-5700)
    for i in range(5343, min(5700, len(lines))):
        line = lines[i]
        # Check if this is a method definition with 4-space indent
        if line.startswith('    def ') or line.startswith('    async def '):
            # Change to 12-space indent
            lines[i] = '            ' + line[4:]
            print(f"Fixed line {i+1}: {line.strip()[:50]}...")
    
    # Write back
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print("Fixed SecureMultiPartyComputation methods")

if __name__ == "__main__":
    fix_smp_methods("backend/modules/enkf_knowledge_integration.py")
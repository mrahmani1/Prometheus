#!/usr/bin/env python3
"""
Fix indentation issues in consciousness-inspired volcanic system files
"""

import re
import os

def fix_indentation_issues(file_path):
    """Fix common indentation issues in the file"""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix pattern: "return something async def" -> "return something\n\n    async def"
    content = re.sub(r'(\s+return [^\n]+)\s+async def', r'\1\n\n    async def', content)
    
    # Fix pattern: "return something   async def" -> "return something\n\n    async def"  
    content = re.sub(r'(\s+return [^\n]+)\s+asy\nnc def', r'\1\n\n    async def', content)
    
    # Fix pattern where async got split across lines
    content = re.sub(r'asy\nnc def', 'async def', content)
    
    # Fix indentation for methods that start with wrong indentation
    content = re.sub(r'\n   async def', '\n\n    async def', content)
    content = re.sub(r'\n  def', '\n\n    def', content)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Fixed indentation issues in {file_path}")

def main():
    files_to_fix = [
        'modules/consciousness_simulation_engine.py',
        'modules/emergent_behavior_detection_system.py', 
        'modules/system_memory_learning_framework.py',
        'modules/inter_system_communication_modeling.py'
    ]
    
    for file_path in files_to_fix:
        if os.path.exists(file_path):
            fix_indentation_issues(file_path)
        else:
            print(f"File not found: {file_path}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Comprehensive fix for all indentation issues"""

import re

def comprehensive_fix(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix all incorrectly indented method definitions 
    # Replace 4-space indented methods with 12-space indented ones
    content = re.sub(r'^    ((?:async )?def \w+.*?:)$', r'            \1', content, flags=re.MULTILINE)
    
    # Write back
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("Applied comprehensive indentation fix")

if __name__ == "__main__":
    comprehensive_fix("backend/modules/enkf_knowledge_integration.py")
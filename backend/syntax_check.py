#!/usr/bin/env python3
"""Detailed syntax checker"""

import ast
import sys

def check_syntax(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()
        
        ast.parse(source)
        print("✓ File parses successfully!")
        return True
        
    except SyntaxError as e:
        print(f"Syntax Error at line {e.lineno}: {e.msg}")
        print(f"Text: {e.text}")
        print(f"Position: {' ' * (e.offset-1)}^")
        return False
    except Exception as e:
        print(f"Other error: {e}")
        return False

if __name__ == "__main__":
    check_syntax("backend/modules/enkf_knowledge_integration.py")
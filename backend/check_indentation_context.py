#!/usr/bin/env python3
"""Check indentation context around problematic line"""

def check_indentation_context(filepath, target_line):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    start = max(0, target_line - 10)
    end = min(len(lines), target_line + 5)
    
    for i in range(start, end):
        line = lines[i].rstrip()
        indent_count = len(line) - len(line.lstrip())
        marker = " --> " if i == target_line - 1 else "     "
        print(f"{i+1:4d}{marker}[{indent_count:2d}] {line}")

if __name__ == "__main__":
    check_indentation_context("backend/modules/enkf_knowledge_integration.py", 6494)
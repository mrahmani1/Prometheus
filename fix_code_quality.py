#!/usr/bin/env python3
"""
Script to fix common code quality issues automatically.
"""

from pathlib import Path
import os
import re
import subprocess

def remove_unused_imports():
    """Remove unused imports using autoflake."""
    print("🔧 Removing unused imports...")
    try:
        subprocess.run([
            "autoflake", 
            "--remove-all-unused-imports",
            "--remove-unused-variables",
            "--in-place",
            "--recursive",
            "services"
        ], check=True)
        print("✅ Unused imports removed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ autoflake not available, skipping unused import removal")

def fix_line_lengths():
    """Fix line length issues by running black again with specific settings."""
    print("🔧 Fixing line lengths...")
    try:
        subprocess.run([
            "black", 
            "--line-length=88",
            "services"
        ], check=True)
        print("✅ Line lengths fixed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Failed to fix line lengths")

def fix_import_order():
    """Fix import order using isort."""
    print("🔧 Fixing import order...")
    try:
        subprocess.run([
            "isort", 
            "--profile=black",
            "services"
        ], check=True)
        print("✅ Import order fixed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Failed to fix import order")

def fix_common_issues():
    """Fix common Python code issues."""
    print("🔧 Fixing common code issues...")
    
    issues_fixed = 0
    
    for root, dirs, files in os.walk("services"):
        for file in files:
            if file.endswith('.py'):
                file_path = Path(root) / file
                
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    original_content = content
                    
                    # Fix comparison to True/False
                    content = re.sub(r'== True\b', ' is True', content)
                    content = re.sub(r'== False\b', ' is False', content)
                    content = re.sub(r'!= True\b', ' is not True', content)
                    content = re.sub(r'!= False\b', ' is not False', content)
                    
                    # Fix trailing whitespace
                    lines = content.split('\n')
                    lines = [line.rstrip() for line in lines]
                    content = '\n'.join(lines)
                    
                    # Fix blank lines with whitespace
                    content = re.sub(r'\n\s+\n', '\n\n', content)
                    
                    # Fix f-strings without placeholders
                    content = re.sub(r'f"([^"]*)"', lambda m: f'"{m.group(1)}"' if '{' not in m.group(1) else m.group(0), content)
                    content = re.sub(r"f'([^']*)'", lambda m: f"'{m.group(1)}'" if '{' not in m.group(1) else m.group(0), content)
                    
                    # Fix bare except clauses
                    content = re.sub(r'except:\s*$', 'except Exception:', content, flags=re.MULTILINE)
                    
                    # Fix multiple spaces before keyword
                    content = re.sub(r'(\w+)\s{2,}(=)', r'\1 \2', content)
                    
                    if content != original_content:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                        issues_fixed += 1
                        
                except Exception as e:
                    print(f"❌ Error processing {file_path}: {e}")
    
    print(f"✅ Fixed common issues in {issues_fixed} files")

def fix_missing_imports():
    """Fix missing imports for common undefined names."""
    print("🔧 Fixing missing imports...")
    
    # Common missing imports
    import_fixes = {
        'torch': 'import torch',
        'nn': 'import torch.nn as nn',
        'F': 'import torch.nn.functional as F',
        'np': 'import numpy as np',
        'pd': 'import pandas as pd',
        'Any': 'from typing import Any',
        'Tuple': 'from typing import Tuple',
        'signal': 'import signal',
        'uuid': 'import uuid',
    }
    
    issues_fixed = 0
    
    for root, dirs, files in os.walk("services"):
        for file in files:
            if file.endswith('.py'):
                file_path = Path(root) / file
                
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    original_content = content
                    lines = content.split('\n')
                    
                    # Find undefined names in this file
                    undefined_names = set()
                    for name in import_fixes.keys():
                        if name in content and f'import {name}' not in content and f'from typing import {name}' not in content:
                            undefined_names.add(name)
                    
                    if undefined_names:
                        # Find where to insert imports (after existing imports)
                        import_end_idx = 0
                        for i, line in enumerate(lines):
                            if line.strip().startswith(('import ', 'from ')) and not line.strip().startswith('#'):
                                import_end_idx = i + 1
                        
                        # Insert missing imports
                        new_imports = []
                        for name in sorted(undefined_names):
                            new_imports.append(import_fixes[name])
                        
                        if new_imports:
                            lines = lines[:import_end_idx] + new_imports + [''] + lines[import_end_idx:]
                            content = '\n'.join(lines)
                    
                    if content != original_content:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                        issues_fixed += 1
                        
                except Exception as e:
                    print(f"❌ Error processing {file_path}: {e}")
    
    print(f"✅ Fixed missing imports in {issues_fixed} files")

def install_tools():
    """Install required tools."""
    print("📦 Installing required tools...")
    try:
        subprocess.run(["pip", "install", "autoflake"], check=True)
        print("✅ autoflake installed")
    except subprocess.CalledProcessError:
        print("❌ Failed to install autoflake")

def main():
    """Main function."""
    print("🚀 Starting code quality fixes...")
    print("=" * 50)
    
    # Install tools
    install_tools()
    
    # Run fixes
    remove_unused_imports()
    fix_missing_imports()
    fix_line_lengths()
    fix_import_order()
    fix_common_issues()
    
    print("\n" + "=" * 50)
    print("🎯 Code quality fixes complete!")

if __name__ == "__main__":
    main()
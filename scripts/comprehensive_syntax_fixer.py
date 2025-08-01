#!/usr/bin/env python3
"""
Comprehensive syntax error fixer for the Prometheus AI platform.
Fixes common Python syntax issues automatically.
"""

import ast
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict

class SyntaxFixer:
    """Advanced syntax error fixer."""
    
    def __init__(self):
        self.fixes_applied = 0
        self.files_processed = 0
        
    def fix_indentation_errors(self, content: str) -> str:
        """Fix common indentation errors."""
        lines = content.split('\n')
        fixed_lines = []
        
        for i, line in enumerate(lines):
            # Skip empty lines
            if not line.strip():
                fixed_lines.append(line)
                continue
                
            # Fix mixed tabs and spaces
            line = line.expandtabs(4)
            
            # Fix common indentation patterns
            if i > 0:
                prev_line = lines[i-1].strip()
                
                # After colon, next line should be indented
                if prev_line.endswith(':') and line.strip() and not line.startswith(' '):
                    line = '    ' + line
                    
                # After try/except/finally/if/for/while/with/def/class
                if any(prev_line.startswith(keyword) for keyword in 
                       ['try:', 'except', 'finally:', 'if ', 'for ', 'while ', 'with ', 'def ', 'class ']):
                    if line.strip() and not line.startswith(' '):
                        line = '    ' + line
            
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def fix_bracket_errors(self, content: str) -> str:
        """Fix unmatched brackets and parentheses."""
        # Simple bracket matching and fixing
        stack = []
        bracket_map = {'(': ')', '[': ']', '{': '}'}
        
        lines = content.split('\n')
        for i, line in enumerate(lines):
            for char in line:
                if char in bracket_map:
                    stack.append((char, i))
                elif char in bracket_map.values():
                    if stack and bracket_map[stack[-1][0]] == char:
                        stack.pop()
        
        # If we have unmatched opening brackets, try to close them
        if stack:
            lines = content.split('\n')
            for bracket, line_num in reversed(stack):
                if line_num < len(lines):
                    closing = bracket_map[bracket]
                    # Add closing bracket at end of line or next appropriate line
                    if lines[line_num].strip():
                        lines[line_num] += closing
                    
            content = '\n'.join(lines)
        
        return content
    
    def fix_string_literals(self, content: str) -> str:
        """Fix unterminated string literals."""
        lines = content.split('\n')
        fixed_lines = []
        
        for line in lines:
            # Fix unterminated strings by adding closing quotes
            if line.count('"') % 2 == 1 and not line.strip().endswith('\\'):
                line += '"'
            if line.count("'") % 2 == 1 and not line.strip().endswith('\\'):
                line += "'"
            
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def fix_try_except_blocks(self, content: str) -> str:
        """Fix incomplete try/except blocks."""
        lines = content.split('\n')
        fixed_lines = []
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # If we find a try block without except/finally
            if line.strip().startswith('try:'):
                fixed_lines.append(line)
                i += 1
                
                # Look for the except/finally block
                found_except = False
                indent_level = len(line) - len(line.lstrip())
                
                while i < len(lines):
                    current_line = lines[i]
                    current_indent = len(current_line) - len(current_line.lstrip())
                    
                    if (current_line.strip().startswith('except') or 
                        current_line.strip().startswith('finally')):
                        found_except = True
                        break
                    
                    # If we hit a line at the same or lower indentation level
                    if (current_line.strip() and current_indent <= indent_level and 
                        not current_line.strip().startswith('except') and
                        not current_line.strip().startswith('finally')):
                        break
                    
                    fixed_lines.append(current_line)
                    i += 1
                
                # If no except/finally found, add a generic except
                if not found_except:
                    fixed_lines.append(' ' * (indent_level + 4) + 'pass')
                    fixed_lines.append(' ' * indent_level + 'except Exception:')
                    fixed_lines.append(' ' * (indent_level + 4) + 'pass')
                
                continue
            
            fixed_lines.append(line)
            i += 1
        
        return '\n'.join(fixed_lines)
    
    def fix_if_blocks(self, content: str) -> str:
        """Fix incomplete if blocks."""
        lines = content.split('\n')
        fixed_lines = []
        
        for i, line in enumerate(lines):
            fixed_lines.append(line)
            
            # If we find an if/for/while/with statement without a body
            if (line.strip().endswith(':') and 
                any(line.strip().startswith(keyword) for keyword in 
                    ['if ', 'for ', 'while ', 'with ', 'else:', 'elif '])):
                
                # Check if next line is properly indented
                if i + 1 < len(lines):
                    next_line = lines[i + 1]
                    if not next_line.strip() or not next_line.startswith(' '):
                        # Add a pass statement
                        indent = len(line) - len(line.lstrip()) + 4
                        fixed_lines.append(' ' * indent + 'pass')
        
        return '\n'.join(fixed_lines)
    
    def fix_function_definitions(self, content: str) -> str:
        """Fix incomplete function definitions."""
        lines = content.split('\n')
        fixed_lines = []
        
        for i, line in enumerate(lines):
            fixed_lines.append(line)
            
            # If we find a function definition without a body
            if line.strip().startswith('def ') and line.strip().endswith(':'):
                # Check if next line is properly indented
                if i + 1 < len(lines):
                    next_line = lines[i + 1]
                    if not next_line.strip() or not next_line.startswith(' '):
                        # Add a pass statement
                        indent = len(line) - len(line.lstrip()) + 4
                        fixed_lines.append(' ' * indent + 'pass')
        
        return '\n'.join(fixed_lines)
    
    def fix_syntax_errors(self, content: str) -> str:
        """Apply all syntax fixes."""
        # Apply fixes in order
        content = self.fix_indentation_errors(content)
        content = self.fix_bracket_errors(content)
        content = self.fix_string_literals(content)
        content = self.fix_try_except_blocks(content)
        content = self.fix_if_blocks(content)
        content = self.fix_function_definitions(content)
        
        return content
    
    def fix_file(self, file_path: Path) -> bool:
        """Fix syntax errors in a single file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                original_content = f.read()
            
            # Try to parse original content
            try:
                ast.parse(original_content)
                return True  # No syntax errors
            except SyntaxError:
                pass  # Continue with fixing
            
            # Apply fixes
            fixed_content = self.fix_syntax_errors(original_content)
            
            # Test if fixes worked
            try:
                ast.parse(fixed_content)
                
                # Write fixed content back
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(fixed_content)
                
                self.fixes_applied += 1
                print(f"✅ Fixed: {file_path}")
                return True
                
            except SyntaxError as e:
                print(f"⚠️  Could not auto-fix: {file_path} - {e}")
                return False
                
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
            return False
    
    def fix_directory(self, directory: Path) -> Dict[str, int]:
        """Fix all Python files in a directory."""
        results = {"fixed": 0, "failed": 0, "total": 0}
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    results["total"] += 1
                    
                    if self.fix_file(file_path):
                        results["fixed"] += 1
                    else:
                        results["failed"] += 1
        
        return results

def main():
    """Main function."""
    print("🔧 Comprehensive Python Syntax Fixer")
    print("=====================================")
    
    fixer = SyntaxFixer()
    
    # Fix services directory
    services_dir = Path("services")
    if services_dir.exists():
        print(f"\n🔍 Fixing syntax errors in {services_dir}...")
        results = fixer.fix_directory(services_dir)
        
        print(f"\n📊 Results:")
        print(f"  Total files: {results['total']}")
        print(f"  Fixed: {results['fixed']}")
        print(f"  Failed: {results['failed']}")
        
        if results['failed'] == 0:
            print("\n✅ All syntax errors fixed!")
        else:
            print(f"\n⚠️  {results['failed']} files still have syntax errors")
            print("These may require manual intervention.")
    
    else:
        print("❌ Services directory not found!")

if __name__ == "__main__":
    main()
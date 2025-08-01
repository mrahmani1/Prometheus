#!/usr/bin/env python3
"""
Check for potentially unused imports in Python files.
"""

import ast
import os

def find_unused_imports():
    unused_count = 0
    files_with_unused = []
    
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    tree = ast.parse(content)
                    
                    # Collect imports
                    imports = []
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                imports.append(alias.asname or alias.name)
                        elif isinstance(node, ast.ImportFrom):
                            for alias in node.names:
                                imports.append(alias.asname or alias.name)
                    
                    # Check if imports are used (simple heuristic)
                    unused_in_file = []
                    for imp in imports:
                        if imp != '*' and imp not in ['__future__']:
                            # Simple check: if import name doesn't appear elsewhere
                            import_lines = [line for line in content.split('\n') if line.strip().startswith(('import ', 'from '))]
                            non_import_content = '\n'.join([line for line in content.split('\n') if not line.strip().startswith(('import ', 'from ', '#'))])
                            
                            if imp not in non_import_content and len(imp) > 2:
                                unused_in_file.append(imp)
                    
                    if unused_in_file:
                        files_with_unused.append((filepath, unused_in_file))
                        unused_count += len(unused_in_file)
                        
                except Exception:
                    continue
    
    return unused_count, files_with_unused

if __name__ == "__main__":
    count, files = find_unused_imports()
    print(f'Found {count} potentially unused imports in {len(files)} files')
    for filepath, unused in files[:10]:
        print(f'{filepath}: {unused}')
#!/usr/bin/env python3
"""
Advanced import organization tool for Python files.
Organizes imports according to PEP 8 standards.
"""

from typing import List, Dict, Tuple, Set
import ast
import os
import re

class ImportOrganizer:
    def __init__(self):
        # Standard library modules (partial list)
        self.stdlib_modules = {
            'os', 'sys', 'json', 'csv', 'xml', 'urllib', 'http', 'datetime', 
            'time', 'collections', 'itertools', 'functools', 'operator',
            'math', 'random', 'statistics', 'pathlib', 'glob', 'shutil',
            'subprocess', 'threading', 'multiprocessing', 'asyncio',
            'logging', 'warnings', 'traceback', 'unittest', 'typing',
            'dataclasses', 'enum', 'abc', 're', 'string', 'textwrap',
            'io', 'pickle', 'sqlite3', 'hashlib', 'uuid', 'copy',
            'tempfile', 'argparse', 'configparser', 'inspect'
        }
        
        # Third-party modules commonly used in data science/ML
        self.third_party_modules = {
            'numpy', 'pandas', 'scipy', 'matplotlib', 'seaborn', 'plotly',
            'sklearn', 'torch', 'tensorflow', 'keras', 'transformers',
            'requests', 'flask', 'django', 'fastapi', 'sqlalchemy',
            'pytest', 'click', 'rich', 'tqdm', 'joblib', 'dask',
            'xarray', 'networkx', 'pillow', 'opencv', 'imageio'
        }

    def analyze_file(self, filepath: str) -> Dict:
        """Analyze import organization in a file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                
            tree = ast.parse(content)
            lines = content.split('\n')
            
            imports = []
            import_lines = []
            
            # Find all imports and their line numbers
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    line_no = node.lineno - 1  # Convert to 0-indexed
                    import_lines.append(line_no)
                    
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.append({
                                'type': 'import',
                                'module': alias.name,
                                'line_no': line_no,
                                'line_content': lines[line_no].strip()
                            })
                    else:  # ImportFrom
                        module = node.module or ''
                        for alias in node.names:
                            imports.append({
                                'type': 'from_import', 
                                'module': module,
                                'name': alias.name,
                                'line_no': line_no,
                                'line_content': lines[line_no].strip()
                            })
            
            # Categorize imports
            stdlib_imports = []
            third_party_imports = []
            local_imports = []
            
            for imp in imports:
                if self._is_local_import(imp):
                    local_imports.append(imp)
                elif self._is_stdlib_import(imp):
                    stdlib_imports.append(imp)
                else:
                    third_party_imports.append(imp)
            
            issues = self._find_organization_issues(
                stdlib_imports, third_party_imports, local_imports, import_lines
            )
            
            return {
                'filepath': filepath,
                'total_imports': len(imports),
                'stdlib_imports': len(stdlib_imports),
                'third_party_imports': len(third_party_imports),
                'local_imports': len(local_imports),
                'issues': issues,
                'needs_reorganization': len(issues) > 0
            }
            
        except Exception as e:
            return {
                'filepath': filepath,
                'error': str(e),
                'needs_reorganization': False
            }

    def _is_local_import(self, imp: Dict) -> bool:
        """Check if import is local/relative."""
        line = imp['line_content']
        return ('from .' in line or 'from ..' in line)

    def _is_stdlib_import(self, imp: Dict) -> bool:
        """Check if import is from standard library."""
        module = imp['module'].split('.')[0] if imp['module'] else ''
        return module in self.stdlib_modules

    def _find_organization_issues(self, stdlib: List, third_party: List, 
                                local: List, import_lines: List) -> List[str]:
        """Find import organization issues."""
        issues = []
        
        if not import_lines:
            return issues
            
        # Check if imports are grouped properly
        all_imports = stdlib + third_party + local
        
        # Sort by line number to check order
        all_imports.sort(key=lambda x: x['line_no'])
        
        # Check grouping
        current_group = None
        for imp in all_imports:
            if self._is_local_import(imp):
                group = 'local'
            elif self._is_stdlib_import(imp):
                group = 'stdlib'
            else:
                group = 'third_party'
                
            if current_group is None:
                current_group = group
            elif current_group != group:
                # Check if we've seen this group before
                prev_groups = [self._get_import_group(x) for x in all_imports 
                             if x['line_no'] < imp['line_no']]
                if group in prev_groups:
                    issues.append(f"Import groups not properly separated")
                    break
                current_group = group
        
        # Check for mixed import styles
        import_styles = set()
        for imp in all_imports:
            if imp['type'] == 'import':
                import_styles.add('import')
            else:
                import_styles.add('from_import')
        
        # Check alphabetical ordering within groups
        if len(stdlib) > 1:
            stdlib_modules = [imp['module'] for imp in stdlib]
            if stdlib_modules != sorted(stdlib_modules):
                issues.append("Standard library imports not alphabetically ordered")
        
        if len(third_party) > 1:
            tp_modules = [imp['module'] for imp in third_party]
            if tp_modules != sorted(tp_modules):
                issues.append("Third-party imports not alphabetically ordered")
                
        # Check for blank lines between groups
        if len(all_imports) > 1:
            consecutive_lines = 0
            for i in range(1, len(import_lines)):
                if import_lines[i] == import_lines[i-1] + 1:
                    consecutive_lines += 1
            
            if consecutive_lines == len(import_lines) - 1:
                issues.append("Missing blank lines between import groups")
        
        return issues

    def _get_import_group(self, imp: Dict) -> str:
        """Get the group category for an import."""
        if self._is_local_import(imp):
            return 'local'
        elif self._is_stdlib_import(imp):
            return 'stdlib'
        else:
            return 'third_party'

    def organize_imports(self, filepath: str) -> bool:
        """Organize imports in a file according to PEP 8."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                
            lines = content.split('\n')
            tree = ast.parse(content)
            
            # Find all imports
            imports = []
            import_line_numbers = set()
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    line_no = node.lineno - 1
                    import_line_numbers.add(line_no)
                    
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.append({
                                'type': 'import',
                                'module': alias.name,
                                'alias': alias.asname,
                                'line_no': line_no,
                                'original_line': lines[line_no]
                            })
                    else:
                        module = node.module or ''
                        for alias in node.names:
                            imports.append({
                                'type': 'from_import',
                                'module': module,
                                'name': alias.name,
                                'alias': alias.asname,
                                'line_no': line_no,
                                'original_line': lines[line_no]
                            })
            
            if not imports:
                return False
                
            # Categorize and sort imports
            stdlib_imports = []
            third_party_imports = []
            local_imports = []
            
            for imp in imports:
                if self._is_local_import_dict(imp):
                    local_imports.append(imp)
                elif self._is_stdlib_import_dict(imp):
                    stdlib_imports.append(imp)
                else:
                    third_party_imports.append(imp)
            
            # Sort each category
            stdlib_imports.sort(key=lambda x: (x['module'], x.get('name', '')))
            third_party_imports.sort(key=lambda x: (x['module'], x.get('name', '')))
            local_imports.sort(key=lambda x: (x['module'], x.get('name', '')))
            
            # Generate organized import lines
            organized_lines = []
            
            # Add standard library imports
            if stdlib_imports:
                organized_lines.extend(self._format_imports(stdlib_imports))
                organized_lines.append('')  # Blank line
            
            # Add third-party imports
            if third_party_imports:
                organized_lines.extend(self._format_imports(third_party_imports))
                organized_lines.append('')  # Blank line
                
            # Add local imports
            if local_imports:
                organized_lines.extend(self._format_imports(local_imports))
                organized_lines.append('')  # Blank line
            
            # Remove the last blank line if it exists
            if organized_lines and organized_lines[-1] == '':
                organized_lines.pop()
            
            # Replace import section in original content
            new_lines = []
            import_section_start = min(import_line_numbers)
            import_section_end = max(import_line_numbers)
            
            # Add content before imports
            new_lines.extend(lines[:import_section_start])
            
            # Add organized imports
            new_lines.extend(organized_lines)
            
            # Add content after imports
            new_lines.extend(lines[import_section_end + 1:])
            
            # Write back to file
            new_content = '\n'.join(new_lines)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
                
            return True
            
        except Exception as e:
            print(f"Error organizing imports in {filepath}: {e}")
            return False

    def _is_local_import_dict(self, imp: Dict) -> bool:
        """Check if import dict represents local import."""
        return imp['original_line'].strip().startswith(('from .', 'from ..'))

    def _is_stdlib_import_dict(self, imp: Dict) -> bool:
        """Check if import dict represents stdlib import."""
        module = imp['module'].split('.')[0] if imp['module'] else ''
        return module in self.stdlib_modules

    def _format_imports(self, imports: List[Dict]) -> List[str]:
        """Format imports into strings."""
        formatted = []
        
        # Group by type and module
        import_groups = {}
        from_import_groups = {}
        
        for imp in imports:
            if imp['type'] == 'import':
                key = imp['module']
                if key not in import_groups:
                    import_groups[key] = []
                import_groups[key].append(imp)
            else:
                key = imp['module']
                if key not in from_import_groups:
                    from_import_groups[key] = []
                from_import_groups[key].append(imp)
        
        # Format regular imports
        for module in sorted(import_groups.keys()):
            imp_list = import_groups[module]
            if len(imp_list) == 1:
                imp = imp_list[0]
                if imp['alias']:
                    formatted.append(f"import {imp['module']} as {imp['alias']}")
                else:
                    formatted.append(f"import {imp['module']}")
        
        # Format from imports
        for module in sorted(from_import_groups.keys()):
            imp_list = from_import_groups[module]
            names = []
            for imp in imp_list:
                if imp['alias']:
                    names.append(f"{imp['name']} as {imp['alias']}")
                else:
                    names.append(imp['name'])
            
            names.sort()
            if len(names) == 1:
                formatted.append(f"from {module} import {names[0]}")
            else:
                # Multi-line import for readability
                formatted.append(f"from {module} import (")
                for i, name in enumerate(names):
                    if i == len(names) - 1:
                        formatted.append(f"    {name}")
                    else:
                        formatted.append(f"    {name},")
                formatted.append(")")
        
        return formatted

def main():
    organizer = ImportOrganizer()
    
    files_analyzed = 0
    files_with_issues = 0
    files_fixed = 0
    
    print("Analyzing import organization...")
    
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                files_analyzed += 1
                
                analysis = organizer.analyze_file(filepath)
                
                if 'error' in analysis:
                    continue
                    
                if analysis['needs_reorganization']:
                    files_with_issues += 1
                    print(f"\nIssues in {filepath}:")
                    for issue in analysis['issues']:
                        print(f"  - {issue}")
                    
                    # Fix the file
                    if organizer.organize_imports(filepath):
                        files_fixed += 1
                        print(f"  [FIXED] Fixed imports in {filepath}")
                    else:
                        print(f"  [FAILED] Failed to fix imports in {filepath}")
    
    print(f"\n=== Import Organization Summary ===")
    print(f"Files analyzed: {files_analyzed}")
    print(f"Files with issues: {files_with_issues}")
    print(f"Files fixed: {files_fixed}")

if __name__ == "__main__":
    main()
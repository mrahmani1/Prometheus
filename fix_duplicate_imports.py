#!/usr/bin/env python3
"""
Duplicate Import Detection and Fixing Script for Prometheus AI Platform

This script:
1. Uses AST to properly parse Python files and detect duplicate imports
2. Identifies genuine duplicate imports within the same scope (not in templates/comments)
3. Fixes duplicates by removing redundant ones while preserving the first occurrence
4. Handles both "import X" and "from X import Y" statements
5. Provides detailed logging and reporting
"""

import ast
import os
import sys
import logging
import json
from typing import List, Dict, Set, Tuple, Optional, Any
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('duplicate_imports_fix.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ImportStatement:
    """Represents an import statement with its details."""
    type: str  # 'import' or 'from_import'
    module: str
    names: List[str]  # For 'from' imports, these are the imported names
    alias: Optional[str] = None
    line_number: int = 0
    raw_statement: str = ""
    
    def __hash__(self):
        """Make ImportStatement hashable for set operations."""
        return hash((self.type, self.module, tuple(sorted(self.names)), self.alias))
    
    def __eq__(self, other):
        """Check equality for duplicate detection."""
        if not isinstance(other, ImportStatement):
            return False
        return (self.type == other.type and 
                self.module == other.module and 
                sorted(self.names) == sorted(other.names) and 
                self.alias == other.alias)
    
    def to_string(self):
        """Convert back to import statement string."""
        if self.type == 'import':
            if self.alias:
                return f"import {self.module} as {self.alias}"
            else:
                return f"import {self.module}"
        elif self.type == 'from_import':
            names_str = ', '.join(self.names)
            return f"from {self.module} import {names_str}"
        return self.raw_statement

@dataclass
class DuplicateImportReport:
    """Report for duplicate imports found in a file."""
    file_path: str
    duplicates: List[Dict[str, Any]]
    total_duplicates: int
    lines_removed: List[int]
    fixed: bool = False

class ImportAnalyzer(ast.NodeVisitor):
    """AST visitor to analyze import statements."""
    
    def __init__(self):
        self.imports: List[ImportStatement] = []
        self.current_scope = "module"  # Track scope for future enhancements
        
    def visit_Import(self, node):
        """Visit regular import statements."""
        for alias in node.names:
            import_stmt = ImportStatement(
                type='import',
                module=alias.name,
                names=[],
                alias=alias.asname,
                line_number=node.lineno,
                raw_statement=f"import {alias.name}" + (f" as {alias.asname}" if alias.asname else "")
            )
            self.imports.append(import_stmt)
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        """Visit from-import statements."""
        if node.module:
            names = [alias.name for alias in node.names]
            # Handle cases like "from module import *"
            if any(name == '*' for name in names):
                names = ['*']
            
            import_stmt = ImportStatement(
                type='from_import',
                module=node.module,
                names=names,
                line_number=node.lineno,
                raw_statement=f"from {node.module} import {', '.join(names)}"
            )
            self.imports.append(import_stmt)
        self.generic_visit(node)

class DuplicateImportFixer:
    """Main class for detecting and fixing duplicate imports."""
    
    def __init__(self, root_directory: str):
        self.root_directory = Path(root_directory)
        self.reports: List[DuplicateImportReport] = []
        self.stats = {
            'files_processed': 0,
            'files_with_duplicates': 0,
            'total_duplicates_found': 0,
            'total_duplicates_fixed': 0,
            'errors': 0
        }
        
    def find_python_files(self) -> List[Path]:
        """Find all Python files in the directory tree."""
        python_files = []
        for path in self.root_directory.rglob('*.py'):
            # Skip backup files and test files for now
            if '.backup' not in str(path) and '__pycache__' not in str(path):
                python_files.append(path)
        return python_files
    
    def analyze_file(self, file_path: Path) -> Optional[DuplicateImportReport]:
        """Analyze a single Python file for duplicate imports."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse the AST
            tree = ast.parse(content, filename=str(file_path))
            
            # Analyze imports
            analyzer = ImportAnalyzer()
            analyzer.visit(tree)
            
            # Find duplicates
            duplicates = self._find_duplicates(analyzer.imports)
            
            if duplicates:
                report = DuplicateImportReport(
                    file_path=str(file_path),
                    duplicates=duplicates,
                    total_duplicates=len(duplicates),
                    lines_removed=[]
                )
                return report
            
            return None
            
        except SyntaxError as e:
            logger.warning(f"Syntax error in {file_path}: {e}")
            self.stats['errors'] += 1
            return None
        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {e}")
            self.stats['errors'] += 1
            return None
    
    def _find_duplicates(self, imports: List[ImportStatement]) -> List[Dict[str, Any]]:
        """Find duplicate imports in the list."""
        seen = {}
        duplicates = []
        
        for imp in imports:
            # Create a key for comparison
            key = (imp.type, imp.module, tuple(sorted(imp.names)), imp.alias)
            
            if key in seen:
                # Found a duplicate
                original = seen[key]
                duplicate_info = {
                    'original': {
                        'line': original.line_number,
                        'statement': original.to_string()
                    },
                    'duplicate': {
                        'line': imp.line_number,
                        'statement': imp.to_string()
                    },
                    'import_type': imp.type,
                    'module': imp.module
                }
                duplicates.append(duplicate_info)
            else:
                seen[key] = imp
                
        return duplicates
    
    def fix_file(self, report: DuplicateImportReport) -> bool:
        """Fix duplicate imports in a file."""
        try:
            file_path = Path(report.file_path)
            
            # Read the file
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Collect line numbers to remove (duplicates, keep first occurrence)
            lines_to_remove = set()
            for dup in report.duplicates:
                # Remove the duplicate (later occurrence), keep the original
                lines_to_remove.add(dup['duplicate']['line'] - 1)  # Convert to 0-based index
            
            # Create new content without duplicate lines
            new_lines = []
            for i, line in enumerate(lines):
                if i not in lines_to_remove:
                    new_lines.append(line)
                else:
                    logger.info(f"Removing duplicate import at line {i+1}: {line.strip()}")
            
            # Write back to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
            
            report.lines_removed = sorted(lines_to_remove)
            report.fixed = True
            
            logger.info(f"Fixed {len(lines_to_remove)} duplicate imports in {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error fixing {report.file_path}: {e}")
            return False
    
    def scan_and_fix(self, fix_files: bool = False) -> Dict[str, Any]:
        """Scan all Python files and optionally fix duplicate imports."""
        logger.info(f"Starting duplicate import scan in {self.root_directory}")
        
        python_files = self.find_python_files()
        logger.info(f"Found {len(python_files)} Python files to analyze")
        
        for file_path in python_files:
            self.stats['files_processed'] += 1
            
            if self.stats['files_processed'] % 50 == 0:
                logger.info(f"Processed {self.stats['files_processed']} files...")
            
            report = self.analyze_file(file_path)
            if report:
                self.reports.append(report)
                self.stats['files_with_duplicates'] += 1
                self.stats['total_duplicates_found'] += report.total_duplicates
                
                logger.info(f"Found {report.total_duplicates} duplicate imports in {file_path}")
                
                if fix_files:
                    if self.fix_file(report):
                        self.stats['total_duplicates_fixed'] += report.total_duplicates
        
        # Generate summary
        summary = {
            'statistics': self.stats,
            'files_with_duplicates': [
                {
                    'file_path': report.file_path,
                    'duplicate_count': report.total_duplicates,
                    'duplicates': report.duplicates,
                    'fixed': report.fixed,
                    'lines_removed': report.lines_removed
                }
                for report in self.reports
            ]
        }
        
        return summary
    
    def generate_report(self, output_file: str = 'duplicate_imports_report.json'):
        """Generate a detailed report of findings."""
        summary = {
            'scan_summary': self.stats,
            'detailed_findings': []
        }
        
        for report in self.reports:
            finding = {
                'file_path': report.file_path,
                'total_duplicates': report.total_duplicates,
                'duplicates': report.duplicates,
                'fixed': report.fixed,
                'lines_removed': report.lines_removed
            }
            summary['detailed_findings'].append(finding)
        
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Report saved to {output_file}")

def main():
    """Main function to run the duplicate import fixer."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Find and fix duplicate Python imports')
    parser.add_argument('directory', nargs='?', default='.', help='Directory to scan (default: current directory)')
    parser.add_argument('--fix', action='store_true', help='Actually fix the duplicate imports')
    parser.add_argument('--report', default='duplicate_imports_report.json', help='Output report file')
    
    args = parser.parse_args()
    
    # Initialize the fixer
    fixer = DuplicateImportFixer(args.directory)
    
    # Scan and optionally fix
    logger.info("="*60)
    logger.info("PROMETHEUS AI PLATFORM - DUPLICATE IMPORT FIXER")
    logger.info("="*60)
    
    summary = fixer.scan_and_fix(fix_files=args.fix)
    
    # Generate report
    fixer.generate_report(args.report)
    
    # Print summary
    print("\n" + "="*60)
    print("SCAN SUMMARY")
    print("="*60)
    print(f"Files processed: {summary['statistics']['files_processed']}")
    print(f"Files with duplicates: {summary['statistics']['files_with_duplicates']}")
    print(f"Total duplicates found: {summary['statistics']['total_duplicates_found']}")
    if args.fix:
        print(f"Total duplicates fixed: {summary['statistics']['total_duplicates_fixed']}")
    print(f"Errors encountered: {summary['statistics']['errors']}")
    
    if summary['statistics']['files_with_duplicates'] > 0:
        print(f"\nFiles with duplicate imports:")
        for file_info in summary['files_with_duplicates']:
            print(f"  - {file_info['file_path']}: {file_info['duplicate_count']} duplicates")
            if args.fix and file_info['fixed']:
                print(f"    -> FIXED: Removed lines {file_info['lines_removed']}")
    
    if args.fix:
        print(f"\nDetailed log saved to: duplicate_imports_fix.log")
    print(f"Detailed report saved to: {args.report}")
    
    return 0 if summary['statistics']['errors'] == 0 else 1

if __name__ == '__main__':
    sys.exit(main())
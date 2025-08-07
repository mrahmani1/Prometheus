import re
from collections import Counter
import sys

def check_html_file(filename):
    """Check HTML file for tag balance issues"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find all opening and closing tags
        opening_tags = re.findall(r'<([a-zA-Z][a-zA-Z0-9]*)[^>]*>', content)
        closing_tags = re.findall(r'</([a-zA-Z][a-zA-Z0-9]*)>', content)
        
        # Self-closing tags to ignore
        self_closing = {'img', 'input', 'br', 'hr', 'meta', 'link', 'area', 'base', 'col', 'embed', 'source', 'track', 'wbr'}
        
        # Filter out self-closing tags
        opening_tags = [tag for tag in opening_tags if tag not in self_closing]
        
        # Count occurrences
        open_count = Counter(opening_tags)
        close_count = Counter(closing_tags)
        
        print(f'\n=== {filename} ===')
        
        # Find mismatches
        errors = []
        for tag in open_count:
            diff = open_count[tag] - close_count.get(tag, 0)
            if diff != 0:
                if diff > 0:
                    errors.append(f'{tag}: {diff} unclosed')
                else:
                    errors.append(f'{tag}: {-diff} extra closing')
        
        for tag in close_count:
            if tag not in open_count:
                errors.append(f'{tag}: {close_count[tag]} extra closing')
        
        if errors:
            for error in errors:
                print(f'  {error}')
            return False
        else:
            print('  No tag balance issues found')
            return True
            
    except FileNotFoundError:
        print(f'\n=== {filename} ===')
        print('  File not found')
        return False
    except Exception as e:
        print(f'\n=== {filename} ===')
        print(f'  Error: {e}')
        return False

def fix_common_html_errors(filename):
    """Fix common HTML errors in a file"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Fix common issues
        # 1. Add missing DOCTYPE if not present
        if not content.strip().lower().startswith('<!doctype'):
            content = '<!DOCTYPE html>\n' + content
        
        # 2. Ensure proper meta charset
        if '<meta charset=' not in content.lower():
            # Add after <head> tag
            content = re.sub(r'(<head[^>]*>)', r'\1\n    <meta charset="UTF-8">', content, count=1, flags=re.IGNORECASE)
        
        # 3. Ensure viewport meta tag for responsiveness
        if 'viewport' not in content.lower():
            content = re.sub(r'(<head[^>]*>)', r'\1\n    <meta name="viewport" content="width=device-width, initial-scale=1.0">', content, count=1, flags=re.IGNORECASE)
        
        # Write back the fixed content
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f'Fixed common issues in {filename}')
        return True
        
    except Exception as e:
        print(f'Error fixing {filename}: {e}')
        return False

if __name__ == '__main__':
    files = [
        'advanced_insar_analysis.html',
        'advanced_tectonic_simulation.html',
        'earthquake_stress_visualization.html',
        'volcanic_system_simulator.html',
        'mantle_dynamics_simulator.html',
        'professional_tectonic_simulation.html',
        'scientific_tectonic_simulation.html',
        'volcano_magma_simulation.html'
    ]
    
    print("Checking HTML files for errors...")
    all_valid = True
    
    for filename in files:
        if not check_html_file(filename):
            all_valid = False
            # Try to fix common issues
            fix_common_html_errors(filename)
    
    if all_valid:
        print("\nAll HTML files are valid!")
    else:
        print("\nFixed common HTML issues. Re-checking...")
        for filename in files:
            check_html_file(filename)
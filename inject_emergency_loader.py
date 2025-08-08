#!/usr/bin/env python3
"""
Emergency Loader Injection Script
Injects immediate emergency loader at the top of all HTML files
"""

import os
import glob
import re

def inject_emergency_loader(file_path):
    """Inject emergency loader script at the beginning of HTML file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Skip if already injected
        if 'immediate_emergency_loader.js' in content:
            print(f"  OK: Already injected: {os.path.basename(file_path)}")
            return True
            
        # Emergency loader script tag
        emergency_script = '''    <!-- IMMEDIATE EMERGENCY LOADER - Must be first script -->
    <script src="immediate_emergency_loader.js"></script>
    
'''
        
        # Find the first <script> tag or <head> section
        head_pattern = r'(<head[^>]*>)'
        script_pattern = r'(    <script[^>]*>)'
        
        if re.search(head_pattern, content, re.IGNORECASE):
            # Insert after <head> tag
            content = re.sub(
                head_pattern, 
                r'\1\n' + emergency_script, 
                content, 
                count=1,
                flags=re.IGNORECASE
            )
        elif re.search(script_pattern, content):
            # Insert before first script tag
            content = re.sub(
                script_pattern,
                emergency_script + r'\1',
                content,
                count=1
            )
        else:
            # Insert at the beginning of the file if no head or script found
            content = emergency_script + '\n' + content
        
        # Write back to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"  SUCCESS: Injected: {os.path.basename(file_path)}")
        return True
        
    except Exception as e:
        print(f"  ERROR with {file_path}: {str(e)}")
        return False

def main():
    print("EMERGENCY: Injecting Immediate Emergency Loader into all HTML files...")
    
    # Find all HTML files
    html_files = glob.glob("*.html")
    
    if not html_files:
        print("  WARNING: No HTML files found")
        return
    
    success_count = 0
    total_files = len(html_files)
    
    for html_file in html_files:
        if inject_emergency_loader(html_file):
            success_count += 1
    
    print(f"\nSUCCESS: Emergency Loader Injection Complete!")
    print(f"   Successfully injected: {success_count}/{total_files} files")
    print(f"   The immediate emergency loader will now create libraries before any page code runs")

if __name__ == "__main__":
    main()
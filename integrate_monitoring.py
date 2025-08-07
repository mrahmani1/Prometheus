#!/usr/bin/env python3
"""
Prometheus Monitoring Integration Script
Automatically adds monitoring scripts to existing HTML files
"""

import os
import re
import shutil
from pathlib import Path
from datetime import datetime
import argparse

class PrometheusMonitoringIntegrator:
    def __init__(self, prometheus_dir="."):
        self.prometheus_dir = Path(prometheus_dir)
        self.monitoring_scripts = [
            'remote_monitor.js',
            'global_functions_utility.js', 
            'automatic_recovery_system.js'
        ]
        
        self.html_files_to_update = [
            'volcanic_system_simulator.html',
            'advanced_tectonic_simulation.html',
            'advanced_insar_analysis.html', 
            'earthquake_stress_visualization.html',
            'test_integration.html',
            'mantle_dynamics_simulator.html',
            'prometheus_navigation.html',
            'ml_dashboard.html',
            'scientific_tectonic_simulation.html',
            'professional_tectonic_simulation.html'
        ]
        
    def backup_file(self, file_path):
        """Create a backup of the original file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = file_path.with_suffix(f'.backup_{timestamp}{file_path.suffix}')
        shutil.copy2(file_path, backup_path)
        print(f"✓ Created backup: {backup_path.name}")
        return backup_path
    
    def check_monitoring_scripts_exist(self):
        """Verify all monitoring scripts exist"""
        missing_scripts = []
        for script in self.monitoring_scripts:
            script_path = self.prometheus_dir / script
            if not script_path.exists():
                missing_scripts.append(script)
        
        if missing_scripts:
            print("❌ Missing monitoring scripts:")
            for script in missing_scripts:
                print(f"   - {script}")
            return False
        
        print("✓ All monitoring scripts found")
        return True
    
    def has_monitoring_integration(self, html_content):
        """Check if HTML already has monitoring integration"""
        return 'remote_monitor.js' in html_content
    
    def find_script_insertion_point(self, html_content):
        """Find the best place to insert monitoring scripts"""
        # Look for existing script tags, preferably after holocene_volcanoes.js
        patterns = [
            r'(<script[^>]*src=["\']holocene_volcanoes\.js["\'][^>]*></script>)',
            r'(<script[^>]*src=["\']webgl_fix_and_minimize\.js["\'][^>]*></script>)',
            r'(<script[^>]*src=[^>]*three\.min\.js[^>]*></script>)',
            r'(<script[^>]*src=[^>]*Chart\.js[^>]*></script>)',
            r'(</head>)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, html_content, re.IGNORECASE)
            if match:
                return match.end()
        
        # Fallback: insert before closing body tag
        body_match = re.search(r'(</body>)', html_content, re.IGNORECASE)
        if body_match:
            return body_match.start()
            
        return len(html_content)
    
    def generate_monitoring_scripts_html(self):
        """Generate HTML for monitoring script inclusions"""
        return """
    <!-- Prometheus Remote Monitoring & Debugging System -->
    <script src="remote_monitor.js"></script>
    <script src="global_functions_utility.js"></script>
    <script src="automatic_recovery_system.js"></script>"""
    
    def add_monitoring_to_html(self, html_file_path):
        """Add monitoring scripts to a single HTML file"""
        print(f"\n📄 Processing: {html_file_path.name}")
        
        # Read the HTML file
        try:
            with open(html_file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
        except Exception as e:
            print(f"❌ Error reading file: {e}")
            return False
        
        # Check if already integrated
        if self.has_monitoring_integration(html_content):
            print("   ⚠️  Monitoring already integrated, skipping")
            return True
        
        # Create backup
        self.backup_file(html_file_path)
        
        # Find insertion point
        insertion_point = self.find_script_insertion_point(html_content)
        
        # Insert monitoring scripts
        monitoring_html = self.generate_monitoring_scripts_html()
        new_content = (html_content[:insertion_point] + 
                      monitoring_html + 
                      html_content[insertion_point:])
        
        # Write updated content
        try:
            with open(html_file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print("   ✓ Monitoring scripts added successfully")
            return True
        except Exception as e:
            print(f"   ❌ Error writing file: {e}")
            return False
    
    def integrate_all_files(self):
        """Integrate monitoring into all specified HTML files"""
        print("🚀 Starting Prometheus Monitoring Integration")
        print("=" * 50)
        
        # Check prerequisites
        if not self.check_monitoring_scripts_exist():
            return False
        
        # Find HTML files to update
        files_processed = 0
        files_successful = 0
        
        for html_filename in self.html_files_to_update:
            html_file_path = self.prometheus_dir / html_filename
            
            if html_file_path.exists():
                files_processed += 1
                if self.add_monitoring_to_html(html_file_path):
                    files_successful += 1
            else:
                print(f"⚠️  File not found: {html_filename}")
        
        print("\n" + "=" * 50)
        print("🎉 Integration Complete!")
        print(f"Files processed: {files_processed}")
        print(f"Files successful: {files_successful}")
        
        if files_successful > 0:
            print("\n📋 Next Steps:")
            print("1. Start the error collection server:")
            print("   python error_collection_server.py --debug")
            print("\n2. Open your HTML files in browser to verify monitoring")
            print("\n3. Access debug dashboard at: http://20.80.89.29:8080/")
            print("\n4. Test error capture with:")
            print("   window.prometheusMonitor.addCustomError('test', 'Integration test')")
        
        return files_successful == files_processed
    
    def remove_monitoring(self):
        """Remove monitoring integration from HTML files"""
        print("🧹 Removing Prometheus Monitoring Integration")
        print("=" * 50)
        
        files_processed = 0
        files_cleaned = 0
        
        for html_filename in self.html_files_to_update:
            html_file_path = self.prometheus_dir / html_filename
            
            if not html_file_path.exists():
                continue
                
            print(f"\n📄 Processing: {html_filename}")
            files_processed += 1
            
            try:
                with open(html_file_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                # Remove monitoring scripts section
                monitoring_pattern = r'\s*<!-- Prometheus Remote Monitoring & Debugging System -->[\s\S]*?<script src="automatic_recovery_system\.js"></script>'
                new_content = re.sub(monitoring_pattern, '', html_content)
                
                if new_content != html_content:
                    # Create backup before removal
                    self.backup_file(html_file_path)
                    
                    with open(html_file_path, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    print("   ✓ Monitoring scripts removed")
                    files_cleaned += 1
                else:
                    print("   ⚠️  No monitoring scripts found")
                    
            except Exception as e:
                print(f"   ❌ Error processing file: {e}")
        
        print(f"\n🎉 Removal complete! Files cleaned: {files_cleaned}/{files_processed}")
    
    def verify_integration(self):
        """Verify monitoring integration status"""
        print("🔍 Verifying Prometheus Monitoring Integration")
        print("=" * 50)
        
        integration_status = {}
        
        for html_filename in self.html_files_to_update:
            html_file_path = self.prometheus_dir / html_filename
            
            if not html_file_path.exists():
                integration_status[html_filename] = "FILE_NOT_FOUND"
                continue
                
            try:
                with open(html_file_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                has_monitoring = self.has_monitoring_integration(html_content)
                integration_status[html_filename] = "INTEGRATED" if has_monitoring else "NOT_INTEGRATED"
                
            except Exception as e:
                integration_status[html_filename] = f"ERROR: {e}"
        
        # Print results
        print("\n📊 Integration Status:")
        for filename, status in integration_status.items():
            if status == "INTEGRATED":
                print(f"   ✅ {filename} - {status}")
            elif status == "NOT_INTEGRATED":
                print(f"   ❌ {filename} - {status}")
            elif status == "FILE_NOT_FOUND":
                print(f"   ⚠️  {filename} - {status}")
            else:
                print(f"   🔥 {filename} - {status}")
        
        return integration_status

def main():
    parser = argparse.ArgumentParser(description='Integrate Prometheus Monitoring into HTML files')
    parser.add_argument('--dir', default='.', help='Prometheus directory path (default: current directory)')
    parser.add_argument('--action', choices=['integrate', 'remove', 'verify'], default='integrate',
                       help='Action to perform (default: integrate)')
    
    args = parser.parse_args()
    
    integrator = PrometheusMonitoringIntegrator(args.dir)
    
    if args.action == 'integrate':
        success = integrator.integrate_all_files()
        exit(0 if success else 1)
    elif args.action == 'remove':
        integrator.remove_monitoring()
    elif args.action == 'verify':
        integrator.verify_integration()

if __name__ == '__main__':
    main()
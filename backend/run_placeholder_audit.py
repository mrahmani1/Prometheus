#!/usr/bin/env python3
"""
Placeholder Detection Audit Runner
Simple script to run comprehensive placeholder detection and generate reports
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the backend directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from core.placeholder_audit_orchestrator import PlaceholderAuditOrchestrator


async def run_audit():
    """Run the placeholder detection audit"""
    print("🔍 Starting Comprehensive Placeholder Detection Audit")
    print("=" * 60)
    
    # Initialize orchestrator
    orchestrator = PlaceholderAuditOrchestrator()
    
    try:
        # Run comprehensive audit
        result = await orchestrator.run_comprehensive_audit(
            directories=['backend/modules', 'backend/core'],
            create_tasks=True,
            generate_reports=True
        )
        
        if result['status'] == 'success':
            print("\n✅ Audit completed successfully!")
            print(f"📊 Scan Results:")
            print(f"   • Modules scanned: {result['summary']['scan_summary']['modules_scanned']}")
            print(f"   • Placeholders found: {result['summary']['scan_summary']['placeholders_found']}")
            print(f"   • Modules affected: {result['summary']['scan_summary']['modules_affected']}")
            
            print(f"\n📈 Severity Breakdown:")
            for severity, count in result['summary']['severity_distribution'].items():
                if count > 0:
                    emoji = "🔴" if severity == "critical" else "🟠" if severity == "high" else "🟡" if severity == "medium" else "🟢"
                    print(f"   {emoji} {severity.title()}: {count}")
            
            print(f"\n⏱️  Effort Estimation:")
            print(f"   • Total hours: {result['summary']['effort_estimation']['total_hours']}")
            print(f"   • Estimated weeks: {result['summary']['effort_estimation']['total_weeks']}")
            print(f"   • Tasks created: {result['summary']['effort_estimation']['tasks_created']}")
            
            if result['summary']['top_priorities']:
                print(f"\n🚨 Top Priority Items:")
                for i, item in enumerate(result['summary']['top_priorities'][:3], 1):
                    print(f"   {i}. {item['module']} (line {item['line']}) - {item['type']}")
            
            print(f"\n📁 Generated Reports:")
            for report_file in result['report_files']:
                print(f"   • {report_file}")
            
            print(f"\n💡 Key Recommendations:")
            for rec in result['summary']['recommendations'][:3]:
                print(f"   • {rec}")
            
            print(f"\n🎯 Next Steps:")
            print(f"   1. Review the generated HTML report for detailed analysis")
            print(f"   2. Address critical placeholders immediately")
            print(f"   3. Set up development sprints for systematic replacement")
            print(f"   4. Monitor progress using the tracking system")
            
        else:
            print(f"❌ Audit failed: {result['error_message']}")
            return 1
            
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1
    
    return 0


def main():
    """Main entry point"""
    return asyncio.run(run_audit())


if __name__ == "__main__":
    sys.exit(main())
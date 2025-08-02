#!/usr/bin/env python3
"""
Quality Dashboard Generator for Prometheus AI Platform
Generates HTML dashboard for QA metrics visualization
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
import base64
import io
import json
import logging

from matplotlib.figure import Figure
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class QualityDashboardGenerator:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.reports_dir = self.project_root / "qa-reports"
        self.dashboard_dir = self.project_root / "dashboard"
        self.dashboard_dir.mkdir(exist_ok=True)

    def load_historical_reports(self, days: int = 30) -> List[Dict[str, Any]]:
        """Load historical QA reports"""
        reports = []
        cutoff_date = datetime.now() - timedelta(days=days)

        if not self.reports_dir.exists():
            return reports

        for report_file in self.reports_dir.glob("qa_orchestration_*.json"):
            try:
                # Extract timestamp from filename
                timestamp_str = report_file.stem.split("_")[-2:]
                timestamp_str = "_".join(timestamp_str)
                report_date = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")

                if report_date >= cutoff_date:
                    with open(report_file, "r") as f:
                        report_data = json.load(f)
                        report_data["file_timestamp"] = report_date
                        reports.append(report_data)
            except Exception as e:
                logger.warning(f"Could not load report {report_file}: {e}")

        # Sort by timestamp
        reports.sort(key=lambda x: x.get("file_timestamp", datetime.min))
        return reports

    def generate_score_trend_chart(self, reports: List[Dict[str, Any]]) -> str:
        """Generate overall score trend chart"""
        if not reports:
            return ""

        dates = [r["file_timestamp"] for r in reports]
        scores = [r.get("overall_score", 0) for r in reports]

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(dates, scores, marker="o", linewidth=2, markersize=6)
        ax.set_title("Overall QA Score Trend", fontsize=16, fontweight="bold")
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("QA Score (0-10)", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 10)

        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(dates) // 10)))
        plt.xticks(rotation=45)

        # Add trend line
        if len(scores) > 1:
            z = np.polyfit(range(len(scores)), scores, 1)
            p = np.poly1d(z)
            ax.plot(dates, p(range(len(scores))), "--", alpha=0.7, color="red")

        plt.tight_layout()

        # Convert to base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()

        return f"data:image/png;base64,{image_base64}"

    def generate_component_scores_chart(self, reports: List[Dict[str, Any]]) -> str:
        """Generate component scores comparison chart"""
        if not reports:
            return ""

        # Get latest report for component scores
        latest_report = reports[-1] if reports else {}
        component_results = latest_report.get("component_results", [])

        if not component_results:
            return ""

        components = [r["component"] for r in component_results]
        scores = [r["score"] for r in component_results]
        colors = [
            (
                "green"
                if r["status"] == "passed"
                else "red" if r["status"] == "failed" else "orange"
            )
            for r in component_results
        ]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(components, scores, color=colors, alpha=0.7)
        ax.set_title("Component QA Scores (Latest)", fontsize=16, fontweight="bold")
        ax.set_xlabel("Components", fontsize=12)
        ax.set_ylabel("Score (0-10)", fontsize=12)
        ax.set_ylim(0, 10)
        ax.grid(True, alpha=0.3, axis="y")

        # Add score labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.1,
                f"{score:.1f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        plt.xticks(rotation=45)
        plt.tight_layout()

        # Convert to base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()

        return f"data:image/png;base64,{image_base64}"

    def generate_issues_trend_chart(self, reports: List[Dict[str, Any]]) -> str:
        """Generate issues trend chart"""
        if not reports:
            return ""

        dates = [r["file_timestamp"] for r in reports]
        total_issues = [r.get("summary", {}).get("total_issues", 0) for r in reports]
        critical_issues = [
            r.get("summary", {}).get("critical_issues", 0) for r in reports
        ]

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(dates, total_issues, marker="o", label="Total Issues", linewidth=2)
        ax.plot(
            dates,
            critical_issues,
            marker="s",
            label="Critical Issues",
            linewidth=2,
            color="red",
        )
        ax.set_title("Issues Trend", fontsize=16, fontweight="bold")
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Number of Issues", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(dates) // 10)))
        plt.xticks(rotation=45)

        plt.tight_layout()

        # Convert to base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()

        return f"data:image/png;base64,{image_base64}"

    def get_latest_metrics(self, reports: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get latest QA metrics"""
        if not reports:
            return {}

        latest_report = reports[-1]
        return {
            "overall_score": latest_report.get("overall_score", 0),
            "overall_status": latest_report.get("overall_status", "unknown"),
            "total_duration": latest_report.get("total_duration", 0),
            "timestamp": latest_report.get("timestamp", ""),
            "summary": latest_report.get("summary", {}),
            "component_results": latest_report.get("component_results", []),
            "recommendations": latest_report.get("recommendations", []),
        }

    def generate_html_dashboard(self, reports: List[Dict[str, Any]]) -> str:
        """Generate HTML dashboard"""
        logger.info("Generating HTML dashboard")

        # Generate charts
        score_trend_chart = self.generate_score_trend_chart(reports)
        component_scores_chart = self.generate_component_scores_chart(reports)
        issues_trend_chart = self.generate_issues_trend_chart(reports)

        # Get latest metrics
        latest_metrics = self.get_latest_metrics(reports)

        # Generate HTML
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prometheus AI - QA Dashboard</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
        }}
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .metric-value {{
            font-size: 2.5em;
            font-weight: bold;
            margin: 10px 0;
        }}
        .metric-label {{
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .status-passed {{ color: #28a745; }}
        .status-failed {{ color: #dc3545; }}
        .status-warning {{ color: #ffc107; }}
        .chart-container {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .chart-container img {{
            width: 100%;
            height: auto;
        }}
        .components-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .component-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .component-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }}
        .component-name {{
            font-weight: bold;
            font-size: 1.1em;
        }}
        .component-score {{
            font-size: 1.5em;
            font-weight: bold;
        }}
        .recommendations {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .recommendations h3 {{
            margin-top: 0;
            color: #333;
        }}
        .recommendations ul {{
            padding-left: 20px;
        }}
        .recommendations li {{
            margin-bottom: 10px;
            line-height: 1.5;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            color: #666;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔬 Prometheus AI QA Dashboard</h1>
        <p>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-label">Overall Score</div>
            <div class="metric-value status-{latest_metrics.get('overall_status', 'unknown')}">{latest_metrics.get('overall_score', 0):.1f}/10</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Status</div>
            <div class="metric-value status-{latest_metrics.get('overall_status', 'unknown')}">{latest_metrics.get('overall_status', 'Unknown').upper()}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Total Issues</div>
            <div class="metric-value">{latest_metrics.get('summary', {}).get('total_issues', 0)}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Critical Issues</div>
            <div class="metric-value status-{'failed' if latest_metrics.get('summary', {}).get('critical_issues', 0) > 0 else 'passed'}">{latest_metrics.get('summary', {}).get('critical_issues', 0)}</div>
        </div>
    </div>
    
    <div class="chart-container">
        <h3>📈 Overall Score Trend</h3>
        {f'<img src="{score_trend_chart}" alt="Score Trend Chart">' if score_trend_chart else '<p>No trend data available</p>'}
    </div>
    
    <div class="chart-container">
        <h3>📊 Component Scores</h3>
        {f'<img src="{component_scores_chart}" alt="Component Scores Chart">' if component_scores_chart else '<p>No component data available</p>'}
    </div>
    
    <div class="chart-container">
        <h3>🐛 Issues Trend</h3>
        {f'<img src="{issues_trend_chart}" alt="Issues Trend Chart">' if issues_trend_chart else '<p>No issues data available</p>'}
    </div>
    
    <div class="components-grid">
        {self.generate_component_cards_html(latest_metrics.get('component_results', []))}
    </div>
    
    <div class="recommendations">
        <h3>💡 Recommendations</h3>
        {self.generate_recommendations_html(latest_metrics.get('recommendations', []))}
    </div>
    
    <div class="footer">
        <p>Generated by Prometheus AI QA System | Data retention: 30 days</p>
    </div>
</body>
</html>
        """

        return html_content

    def generate_component_cards_html(
        self, component_results: List[Dict[str, Any]]
    ) -> str:
        """Generate HTML for component cards"""
        if not component_results:
            return (
                '<div class="component-card"><p>No component data available</p></div>'
            )

        cards_html = ""
        for component in component_results:
            status_class = f"status-{component.get('status', 'unknown')}"
            cards_html += f"""
            <div class="component-card">
                <div class="component-header">
                    <div class="component-name">{component.get('component', 'Unknown').replace('_', ' ').title()}</div>
                    <div class="component-score {status_class}">{component.get('score', 0):.1f}/10</div>
                </div>
                <div>
                    <strong>Status:</strong> <span class="{status_class}">{component.get('status', 'unknown').upper()}</span><br>
                    <strong>Duration:</strong> {component.get('duration', 0):.1f}s<br>
                    <strong>Issues:</strong> {component.get('issues_count', 0)}
                </div>
            </div>
            """

        return cards_html

    def generate_recommendations_html(self, recommendations: List[str]) -> str:
        """Generate HTML for recommendations"""
        if not recommendations:
            return "<p>No recommendations at this time. All systems are performing well! 🎉</p>"

        recommendations_html = "<ul>"
        for rec in recommendations:
            recommendations_html += f"<li>{rec}</li>"
        recommendations_html += "</ul>"

        return recommendations_html

    def generate_dashboard(self) -> str:
        """Generate complete QA dashboard"""
        logger.info("Starting dashboard generation")

        # Load historical reports
        reports = self.load_historical_reports(days=30)
        logger.info(f"Loaded {len(reports)} historical reports")

        # Generate HTML dashboard
        html_content = self.generate_html_dashboard(reports)

        # Save dashboard
        dashboard_file = self.dashboard_dir / "qa_dashboard.html"
        with open(dashboard_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"Dashboard generated: {dashboard_file}")
        return str(dashboard_file)


def main():
    """Main execution function"""
    try:
        # Import numpy for trend analysis
        global np
        import numpy as np

    except ImportError:
        logger.warning("NumPy not available, trend analysis will be limited")
        import sys

        class MockNumPy:
            def polyfit(self, x, y, deg):
                return [0, 0]

            def poly1d(self, coeffs):
                return lambda x: [0] * len(x) if hasattr(x, "__len__") else 0

        sys.modules["numpy"] = MockNumPy()
        np = MockNumPy()

    generator = QualityDashboardGenerator()

    try:
        dashboard_file = generator.generate_dashboard()
        logger.info(f"QA Dashboard generated successfully: {dashboard_file}")
        print(f"Dashboard available at: file://{Path(dashboard_file).absolute()}")
    except Exception as e:
        logger.error(f"Dashboard generation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

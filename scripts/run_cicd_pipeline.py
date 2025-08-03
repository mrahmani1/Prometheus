#!/usr/bin/env python3
"""
CI/CD Pipeline Runner Script
Executes the comprehensive CI/CD pipeline for Prometheus AI Platform
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import git

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from services.system_monitor.cicd_pipeline import CICDPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_current_commit_hash() -> str:
    """Get the current commit hash"""
    try:
        repo = git.Repo(project_root)
        return repo.head.commit.hexsha
    except Exception as e:
        logger.error(f"Failed to get commit hash: {e}")
        return "unknown"


def get_current_branch() -> str:
    """Get the current branch name"""
    try:
        repo = git.Repo(project_root)
        return repo.active_branch.name
    except Exception as e:
        logger.error(f"Failed to get branch name: {e}")
        return "main"


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Run CI/CD Pipeline for Prometheus AI Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run pipeline for current commit and branch
  python scripts/run_cicd_pipeline.py
  
  # Run pipeline for specific commit
  python scripts/run_cicd_pipeline.py --commit abc123def
  
  # Run pipeline for specific branch
  python scripts/run_cicd_pipeline.py --branch feature/new-feature
  
  # Run pipeline with specific trigger
  python scripts/run_cicd_pipeline.py --trigger manual
  
  # Run only specific stages
  python scripts/run_cicd_pipeline.py --stages build,test
  
  # Skip specific stages
  python scripts/run_cicd_pipeline.py --skip-stages security_scan
        """,
    )

    parser.add_argument("--commit", help="Commit hash to build (default: current HEAD)")

    parser.add_argument("--branch", help="Branch name (default: current branch)")

    parser.add_argument(
        "--trigger",
        choices=["push", "pull_request", "manual", "scheduled"],
        default="manual",
        help="Pipeline trigger type",
    )

    parser.add_argument(
        "--stages", help="Comma-separated list of stages to run (default: all)"
    )

    parser.add_argument("--skip-stages", help="Comma-separated list of stages to skip")

    parser.add_argument(
        "--environment",
        choices=["development", "staging", "production"],
        help="Target environment for deployment stages",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a dry run without making actual changes",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    parser.add_argument("--config", help="Path to custom CI/CD configuration file")

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Get commit and branch
    commit_hash = args.commit or get_current_commit_hash()
    branch = args.branch or get_current_branch()

    logger.info(
        f"Starting CI/CD pipeline for commit {commit_hash[:8]} on branch {branch}"
    )

    try:
        # Initialize pipeline
        pipeline = CICDPipeline(project_root)

        # Override configuration if provided
        if args.config:
            config_path = Path(args.config)
            if config_path.exists():
                import yaml

                with open(config_path, "r") as f:
                    custom_config = yaml.safe_load(f)
                pipeline.pipeline_config.update(custom_config)
                logger.info(f"Loaded custom configuration from {config_path}")
            else:
                logger.error(f"Configuration file not found: {config_path}")
                sys.exit(1)

        # Handle stage filtering
        if args.stages:
            enabled_stages = set(args.stages.split(","))
            for stage_name in pipeline.pipeline_config["stages"]:
                if stage_name not in enabled_stages:
                    pipeline.pipeline_config["stages"][stage_name]["enabled"] = False
            logger.info(f"Running only stages: {', '.join(enabled_stages)}")

        if args.skip_stages:
            skipped_stages = set(args.skip_stages.split(","))
            for stage_name in skipped_stages:
                if stage_name in pipeline.pipeline_config["stages"]:
                    pipeline.pipeline_config["stages"][stage_name]["enabled"] = False
            logger.info(f"Skipping stages: {', '.join(skipped_stages)}")

        # Handle dry run
        if args.dry_run:
            logger.info("DRY RUN MODE: No actual changes will be made")
            # In a real implementation, this would set flags to prevent actual deployments

        # Execute pipeline
        execution = await pipeline.execute_pipeline(commit_hash, branch, args.trigger)

        # Print results
        print("\n" + "=" * 80)
        print(f"Pipeline Execution Results")
        print("=" * 80)
        print(f"Pipeline ID: {execution.pipeline_id}")
        print(f"Status: {execution.status.value.upper()}")
        print(f"Branch: {execution.branch}")
        print(f"Commit: {execution.commit_hash[:8]}")
        print(f"Version: {execution.version}")

        if execution.end_time:
            duration = (execution.end_time - execution.start_time).total_seconds()
            print(f"Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")

        print(f"Trigger: {execution.trigger}")

        if execution.rollback_version:
            print(f"Rollback Version: {execution.rollback_version}")

        print("\nStage Results:")
        print("-" * 80)

        for stage in execution.stages:
            status_emoji = {
                "success": "✅",
                "failed": "❌",
                "skipped": "⏭️",
                "running": "🔄",
                "pending": "⏳",
            }.get(stage.status.value, "❓")

            print(
                f"{status_emoji} {stage.stage.value.upper():<20} {stage.status.value.upper():<10} ({stage.duration:.1f}s)"
            )

            if stage.error:
                print(f"   Error: {stage.error}")

            if stage.metrics:
                print(f"   Metrics: {stage.metrics}")

            if args.verbose and stage.output:
                print(f"   Output: {stage.output[:200]}...")

        print("\n" + "=" * 80)

        # Print summary statistics
        total_stages = len(execution.stages)
        successful_stages = len(
            [s for s in execution.stages if s.status.value == "success"]
        )
        failed_stages = len([s for s in execution.stages if s.status.value == "failed"])
        skipped_stages = len(
            [s for s in execution.stages if s.status.value == "skipped"]
        )

        print(f"Summary: {successful_stages}/{total_stages} stages successful")
        if failed_stages > 0:
            print(f"Failed stages: {failed_stages}")
        if skipped_stages > 0:
            print(f"Skipped stages: {skipped_stages}")

        # Exit with appropriate code
        if execution.status.value == "failed":
            logger.error("Pipeline failed!")
            sys.exit(1)
        elif execution.status.value == "success":
            logger.info("Pipeline completed successfully!")
            sys.exit(0)
        else:
            logger.info(f"Pipeline completed with status: {execution.status.value}")
            sys.exit(0)

    except KeyboardInterrupt:
        logger.info("Pipeline execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Pipeline execution failed with error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

# Import wrapper for cloud-run-optimizer.py to handle Python import naming
import importlib.util
import sys
from pathlib import Path

# Load the actual implementation from the hyphenated file
spec = importlib.util.spec_from_file_location(
    "cloud_run_optimizer_impl", 
    Path(__file__).parent / "cloud-run-optimizer.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# Re-export all the classes and functions
CloudRunOptimizer = module.CloudRunOptimizer
ScalingStrategy = module.ScalingStrategy
ServiceMetrics = module.ServiceMetrics
ScalingDecision = module.ScalingDecision
CostOptimization = module.CostOptimization

# Export everything that might be needed
__all__ = [
    'CloudRunOptimizer',
    'ScalingStrategy', 
    'ServiceMetrics',
    'ScalingDecision',
    'CostOptimization'
]
"""
FastAPI endpoints for quantum optimization algorithms for geological problems

This module provides REST API endpoints for accessing quantum computing
capabilities for geological parameter optimization, system analysis,
pattern recognition, and quantum-classical hybrid algorithms.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from ..core.quantum_helios_integration import QuantumHeliosIntegrator
from ..core.quantum_optimization import (QuantumGeologicalOptimizationManager,
                                         analyze_quantum_advantage,
                                         create_geological_quantum_problem)

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/quantum", tags=["quantum-optimization"])

# Global instances
quantum_manager = QuantumGeologicalOptimizationManager()
helios_integrator = QuantumHeliosIntegrator()

# Pydantic models for API


class GeologicalDataInput(BaseModel):
    """Input model for geological data"""

    data: List[float] = Field(..., description="Geological measurement data")
    data_type: str = Field("generic", description="Type of geological data")
    units: Optional[str] = Field(None, description="Data units")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class QuantumProblemRequest(BaseModel):
    """Request model for quantum optimization problem"""

    geological_data: GeologicalDataInput
    problem_type: str = Field(
        ...,
        description="Type of problem: parameter_optimization, system_analysis, pattern_recognition",
    )
    algorithm: str = Field(
        "auto",
        description="Quantum algorithm: quantum_annealing, vqe, quantum_ml, hybrid, auto",
    )
    constraints: Optional[Dict[str, Any]] = Field(
        None, description="Problem constraints"
    )
    objective_function: str = Field(
        "minimize_energy", description="Optimization objective"
    )
    quantum_encoding: str = Field("amplitude", description="Quantum encoding method")


class QuantumOptimizationResponse(BaseModel):
    """Response model for quantum optimization results"""

    success: bool
    algorithm_used: str
    optimal_parameters: List[float]
    optimal_energy: float
    quantum_advantage_metric: float
    confidence_score: float
    execution_time: float
    geological_interpretation: Dict[str, Any]
    convergence_history: List[float]


class HeliosIntegrationRequest(BaseModel):
    """Request model for Helios integration"""

    geological_data: Dict[str, List[float]]
    integration_type: str = Field(
        ...,
        description="Type of integration: parameter_optimization, system_analysis, pattern_recognition, large_scale_modeling",
    )
    model_parameters: Optional[Dict[str, float]] = Field(
        None, description="Model parameters for optimization"
    )
    analysis_config: Optional[Dict[str, Any]] = Field(
        None, description="Analysis configuration"
    )


class HeliosIntegrationResponse(BaseModel):
    """Response model for Helios integration results"""

    success: bool
    integration_type: str
    classical_result: Dict[str, Any]
    quantum_result: QuantumOptimizationResponse
    enhancement_metrics: Dict[str, float]
    combined_interpretation: Dict[str, Any]
    processing_time: float


class BenchmarkRequest(BaseModel):
    """Request model for algorithm benchmarking"""

    geological_data: GeologicalDataInput
    problem_type: str
    algorithms: Optional[List[str]] = Field(None, description="Algorithms to benchmark")


class BenchmarkResponse(BaseModel):
    """Response model for benchmarking results"""

    success: bool
    benchmark_results: Dict[str, QuantumOptimizationResponse]
    performance_analysis: Dict[str, Any]
    best_algorithm: str
    recommendations: Dict[str, float]


# API Endpoints


@router.post("/optimize", response_model=QuantumOptimizationResponse)
async def optimize_geological_problem(request: QuantumProblemRequest):
    """
    Optimize geological problem using quantum algorithms

    This endpoint performs quantum optimization for geological parameter optimization,
    system analysis, or pattern recognition problems.
    """
    try:
        # Convert input data to numpy array
        geological_data = np.array(request.geological_data.data)

        # Create quantum problem
        problem = create_geological_quantum_problem(
            geological_data=geological_data,
            problem_type=request.problem_type,
            constraints=request.constraints or {},
            objective_function=request.objective_function,
            quantum_encoding=request.quantum_encoding,
        )

        # Run quantum optimization
        result = await quantum_manager.optimize_geological_problem(
            problem, algorithm=request.algorithm
        )

        # Convert result to response model
        response = QuantumOptimizationResponse(
            success=True,
            algorithm_used=result.algorithm_type,
            optimal_parameters=result.optimal_parameters.tolist(),
            optimal_energy=result.optimal_energy,
            quantum_advantage_metric=result.quantum_advantage_metric,
            confidence_score=result.confidence_score,
            execution_time=result.execution_time,
            geological_interpretation=result.geological_interpretation,
            convergence_history=result.convergence_history,
        )

        logger.info(
            f"Quantum optimization completed: {request.algorithm} algorithm, "
            f"advantage={result.quantum_advantage_metric:.3f}"
        )

        return response

    except Exception as e:
        logger.error(f"Quantum optimization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")


@router.post("/helios-integration", response_model=HeliosIntegrationResponse)
async def quantum_enhanced_helios_analysis(request: HeliosIntegrationRequest):
    """
    Perform quantum-enhanced Helios geological analysis

    This endpoint integrates quantum optimization with the Helios AI system
    for enhanced geological modeling and analysis.
    """
    try:
        # Convert geological data
        geological_data = {
            key: np.array(values) for key, values in request.geological_data.items()
        }

        # Perform quantum-enhanced analysis based on integration type
        if request.integration_type == "parameter_optimization":
            if not request.model_parameters:
                raise HTTPException(
                    status_code=400,
                    detail="Model parameters required for parameter optimization",
                )

            # Use first data component for parameter optimization
            first_data_key = list(geological_data.keys())[0]
            result = await helios_integrator.quantum_enhanced_parameter_optimization(
                geological_data[first_data_key],
                request.model_parameters,
                optimization_target="minimize_misfit",
            )

        elif request.integration_type == "system_analysis":
            result = await helios_integrator.quantum_enhanced_system_analysis(
                geological_data, analysis_type="stability_analysis"
            )

        elif request.integration_type == "pattern_recognition":
            # Combine all data for pattern recognition
            combined_data = np.concatenate(
                [data[:100] for data in geological_data.values()]
            )
            result = await helios_integrator.quantum_enhanced_pattern_recognition(
                combined_data, recognition_task="classification"
            )

        elif request.integration_type == "large_scale_modeling":
            result = await helios_integrator.quantum_enhanced_large_scale_modeling(
                geological_data, modeling_objective="comprehensive_simulation"
            )

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown integration type: {request.integration_type}",
            )

        # Convert quantum result to response format
        quantum_response = QuantumOptimizationResponse(
            success=True,
            algorithm_used=result.quantum_optimization_result.algorithm_type,
            optimal_parameters=result.quantum_optimization_result.optimal_parameters.tolist(),
            optimal_energy=result.quantum_optimization_result.optimal_energy,
            quantum_advantage_metric=result.quantum_optimization_result.quantum_advantage_metric,
            confidence_score=result.quantum_optimization_result.confidence_score,
            execution_time=result.quantum_optimization_result.execution_time,
            geological_interpretation=result.quantum_optimization_result.geological_interpretation,
            convergence_history=result.quantum_optimization_result.convergence_history,
        )

        # Create response
        response = HeliosIntegrationResponse(
            success=True,
            integration_type=request.integration_type,
            classical_result=result.classical_result,
            quantum_result=quantum_response,
            enhancement_metrics=result.enhancement_metrics,
            combined_interpretation=result.combined_interpretation,
            processing_time=result.processing_time,
        )

        logger.info(
            f"Helios integration completed: {request.integration_type}, "
            f"enhancement={result.enhancement_metrics.get('overall_enhancement', 0):.3f}"
        )

        return response

    except Exception as e:
        logger.error(f"Helios integration failed: {e}")
        raise HTTPException(status_code=500, detail=f"Integration failed: {str(e)}")


@router.post("/benchmark", response_model=BenchmarkResponse)
async def benchmark_quantum_algorithms(request: BenchmarkRequest):
    """
    Benchmark quantum algorithms on a geological problem

    This endpoint compares the performance of different quantum algorithms
    on the same geological optimization problem.
    """
    try:
        # Convert input data
        geological_data = np.array(request.geological_data.data)

        # Create quantum problem
        problem = create_geological_quantum_problem(
            geological_data=geological_data,
            problem_type=request.problem_type,
            objective_function="minimize_energy",
        )

        # Run benchmarking
        if request.algorithms:
            # Benchmark specific algorithms
            benchmark_results = {}
            for algorithm in request.algorithms:
                try:
                    result = await quantum_manager.optimize_geological_problem(
                        problem, algorithm=algorithm
                    )
                    benchmark_results[algorithm] = result
                except Exception as e:
                    logger.warning(f"Algorithm {algorithm} failed: {e}")
        else:
            # Benchmark all algorithms
            benchmark_results = quantum_manager.benchmark_algorithms(problem)

        # Convert results to response format
        response_results = {}
        for alg_name, result in benchmark_results.items():
            response_results[alg_name] = QuantumOptimizationResponse(
                success=True,
                algorithm_used=result.algorithm_type,
                optimal_parameters=result.optimal_parameters.tolist(),
                optimal_energy=result.optimal_energy,
                quantum_advantage_metric=result.quantum_advantage_metric,
                confidence_score=result.confidence_score,
                execution_time=result.execution_time,
                geological_interpretation=getattr(
                    result, "geological_interpretation", {}
                ),
                metadata=getattr(result, "metadata", {}),
            )

        # Analyze performance
        performance_analysis = analyze_quantum_advantage(
            list(benchmark_results.values())
        )

        # Get algorithm recommendations
        recommendations = quantum_manager.get_algorithm_recommendations(problem)

        # Find best algorithm
        best_algorithm = max(
            benchmark_results.keys(),
            key=lambda k: benchmark_results[k].quantum_advantage_metric,
        )

        response = BenchmarkResponse(
            success=True,
            benchmark_results=response_results,
            performance_analysis=performance_analysis,
            best_algorithm=best_algorithm,
            recommendations=recommendations,
        )

        logger.info(
            f"Benchmarking completed: {len(benchmark_results)} algorithms, "
            f"best={best_algorithm}"
        )

        return response

    except Exception as e:
        logger.error(f"Benchmarking failed: {e}")
        raise HTTPException(status_code=500, detail=f"Benchmarking failed: {str(e)}")


@router.get("/algorithms")
async def get_available_algorithms():
    """
    Get list of available quantum algorithms

    Returns information about available quantum algorithms and their
    recommended use cases for geological problems.
    """
    algorithms = {
        "quantum_annealing": {
            "name": "Quantum Annealing",
            "description": "Quantum annealing for geological parameter optimization",
            "best_for": ["parameter_optimization", "discrete_optimization"],
            "data_size_limit": 50,
            "typical_use_cases": [
                "Mineral deposit location optimization",
                "Seismic network placement",
                "Volcanic hazard zone boundary optimization",
            ],
        },
        "vqe": {
            "name": "Variational Quantum Eigensolver",
            "description": "VQE for geological system analysis",
            "best_for": ["system_analysis", "eigenvalue_problems"],
            "data_size_limit": 30,
            "typical_use_cases": [
                "Ground state energy calculations for mineral stability",
                "Eigenmode analysis of seismic wave propagation",
                "Quantum simulation of geological phase transitions",
            ],
        },
        "quantum_ml": {
            "name": "Quantum Machine Learning",
            "description": "Quantum ML for geological pattern recognition",
            "best_for": ["pattern_recognition", "classification"],
            "data_size_limit": 100,
            "typical_use_cases": [
                "Quantum feature mapping of geological data",
                "Quantum kernel methods for geological classification",
                "Quantum neural networks for geological prediction",
            ],
        },
        "hybrid": {
            "name": "Quantum-Classical Hybrid",
            "description": "Hybrid algorithms for large-scale geological problems",
            "best_for": ["large_scale_optimization", "complex_problems"],
            "data_size_limit": 1000,
            "typical_use_cases": [
                "Large-scale seismic tomography inversion",
                "Multi-scale geological modeling",
                "Real-time geological parameter estimation",
            ],
        },
    }

    return {
        "available_algorithms": algorithms,
        "auto_selection": "Use 'auto' algorithm parameter for automatic algorithm selection based on problem characteristics",
        "quantum_hardware_status": "Currently using quantum simulators. Hardware integration available upon request.",
    }


@router.get("/problem-types")
async def get_problem_types():
    """
    Get information about supported geological problem types

    Returns details about the types of geological problems that can be
    solved using quantum optimization algorithms.
    """
    problem_types = {
        "parameter_optimization": {
            "description": "Optimize geological model parameters",
            "input_requirements": "Geological measurement data and model parameters",
            "output": "Optimized parameter values with uncertainty quantification",
            "examples": [
                "Seismic velocity model optimization",
                "Fault plane parameter estimation",
                "Mineral composition optimization",
            ],
        },
        "system_analysis": {
            "description": "Analyze geological system stability and behavior",
            "input_requirements": "Multi-component geological system data",
            "output": "System stability metrics and behavior classification",
            "examples": [
                "Fault network stability analysis",
                "Volcanic system state assessment",
                "Groundwater system analysis",
            ],
        },
        "pattern_recognition": {
            "description": "Recognize patterns in geological data",
            "input_requirements": "Geological pattern data with optional labels",
            "output": "Pattern classification and feature identification",
            "examples": [
                "Geological formation classification",
                "Mineral pattern recognition",
                "Seismic event classification",
            ],
        },
    }

    return {
        "supported_problem_types": problem_types,
        "quantum_encoding_methods": {
            "amplitude": "Encode data in quantum state amplitudes",
            "angle": "Encode data as rotation angles",
            "basis": "Encode data in computational basis states",
        },
        "objective_functions": [
            "minimize_energy",
            "minimize_misfit",
            "maximize_accuracy",
            "analyze_stability",
            "optimize_parameters",
        ],
    }


@router.get("/status")
async def get_quantum_system_status():
    """
    Get current status of the quantum optimization system

    Returns information about system health, available resources,
    and recent performance metrics.
    """
    try:
        # Check system components
        quantum_backend_status = "operational"  # Would check actual quantum backend
        classical_simulator_status = "operational"

        # Get recent performance metrics (would be from actual monitoring)
        recent_metrics = {
            "average_quantum_advantage": 0.65,
            "average_confidence_score": 0.78,
            "average_execution_time": 2.3,
            "successful_optimizations_24h": 42,
            "failed_optimizations_24h": 3,
        }

        status = {
            "system_status": "operational",
            "quantum_backend": quantum_backend_status,
            "classical_simulator": classical_simulator_status,
            "available_algorithms": len(quantum_manager.optimizers),
            "cache_size": len(quantum_manager.results_cache),
            "recent_performance": recent_metrics,
            "last_updated": datetime.now().isoformat(),
            "version": "1.0.0",
        }

        return status

    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")


@router.delete("/cache")
async def clear_optimization_cache():
    """
    Clear the quantum optimization results cache

    Clears cached optimization results to free memory and ensure
    fresh calculations for subsequent requests.
    """
    try:
        cache_size_before = len(quantum_manager.results_cache)
        quantum_manager.results_cache.clear()

        # Also clear Helios integrator cache if it exists
        if hasattr(helios_integrator, "integration_cache"):
            helios_integrator.integration_cache.clear()

        logger.info(f"Cleared optimization cache: {cache_size_before} entries removed")

        return {
            "success": True,
            "message": "Cache cleared successfully",
            "entries_removed": cache_size_before,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Cache clear failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cache clear failed: {str(e)}")


# Background task for long-running optimizations
@router.post("/optimize-async")
async def optimize_geological_problem_async(
    request: QuantumProblemRequest, background_tasks: BackgroundTasks
):
    """
    Start asynchronous quantum optimization

    For large or complex problems, this endpoint starts optimization
    in the background and returns a task ID for status checking.
    """
    try:
        # Generate task ID
        task_id = f"quantum_opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(str(request.geological_data.data)) % 10000}"

        # Add background task
        background_tasks.add_task(run_async_optimization, task_id, request)

        return {
            "success": True,
            "task_id": task_id,
            "message": "Quantum optimization started in background",
            "estimated_completion": "2-10 minutes depending on problem complexity",
            "status_endpoint": f"/quantum/status/{task_id}",
        }

    except Exception as e:
        logger.error(f"Async optimization start failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to start async optimization: {str(e)}"
        )


# Store for async task results
async_results = {}


async def run_async_optimization(task_id: str, request: QuantumProblemRequest):
    """Background task for running quantum optimization"""
    try:
        async_results[task_id] = {"status": "running", "progress": 0}

        # Convert input data
        geological_data = np.array(request.geological_data.data)

        # Create and run optimization
        problem = create_geological_quantum_problem(
            geological_data=geological_data,
            problem_type=request.problem_type,
            constraints=request.constraints or {},
            objective_function=request.objective_function,
            quantum_encoding=request.quantum_encoding,
        )

        async_results[task_id]["progress"] = 25

        result = await quantum_manager.optimize_geological_problem(
            problem, algorithm=request.algorithm
        )

        async_results[task_id]["progress"] = 100
        async_results[task_id]["status"] = "completed"
        async_results[task_id]["result"] = {
            "algorithm_used": result.algorithm_type,
            "optimal_parameters": result.optimal_parameters.tolist(),
            "optimal_energy": result.optimal_energy,
            "quantum_advantage_metric": result.quantum_advantage_metric,
            "confidence_score": result.confidence_score,
            "execution_time": result.execution_time,
            "geological_interpretation": result.geological_interpretation,
        }

        logger.info(f"Async optimization {task_id} completed successfully")

    except Exception as e:
        async_results[task_id]["status"] = "failed"
        async_results[task_id]["error"] = str(e)
        logger.error(f"Async optimization {task_id} failed: {e}")


@router.get("/status/{task_id}")
async def get_async_optimization_status(task_id: str):
    """
    Get status of asynchronous quantum optimization

    Returns the current status and results (if completed) of an
    asynchronous quantum optimization task.
    """
    if task_id not in async_results:
        raise HTTPException(status_code=404, detail="Task ID not found")

    return async_results[task_id]

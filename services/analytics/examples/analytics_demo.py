"""
Advanced Analytics Service Demo
Demonstrates complex geological analysis capabilities with distributed computing
"""

import asyncio
import aiohttp
import json
import time
import numpy as np
from datetime import datetime
from typing import Dict, Any, List

class AdvancedAnalyticsDemo:
    """Demo client for Advanced Analytics Service"""
    
    def __init__(self, base_url: str = "http://localhost:8004"):
        self.base_url = base_url
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def check_health(self) -> Dict[str, Any]:
        """Check service health"""
        async with self.session.get(f"{self.base_url}/health") as response:
            return await response.json()
    
    async def start_geological_modeling(self) -> Dict[str, Any]:
        """Start 3D geological modeling job"""
        # Create sample geological data
        geological_data = {
            'geological_units': [
                {'name': 'granite', 'age': 2500, 'composition': 'felsic'},
                {'name': 'sandstone', 'age': 500, 'composition': 'sedimentary'},
                {'name': 'basalt', 'age': 100, 'composition': 'mafic'}
            ],
            'structural_features': [
                {'type': 'fault', 'strike': 45, 'dip': 60, 'length': 10000},
                {'type': 'fold', 'axis_trend': 120, 'plunge': 30, 'wavelength': 5000}
            ],
            'coordinates': {
                'x_range': [0, 50000],  # meters
                'y_range': [0, 50000],
                'z_range': [-10000, 1000]
            },
            'sample_points': [
                {'x': 10000, 'y': 15000, 'z': -500, 'rock_type': 'granite'},
                {'x': 25000, 'y': 30000, 'z': -200, 'rock_type': 'sandstone'},
                {'x': 40000, 'y': 45000, 'z': -1000, 'rock_type': 'basalt'}
            ]
        }
        
        modeling_request = {
            "job_name": f"geological_modeling_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "geological_data": geological_data,
            "modeling_parameters": {
                "resolution": "high",
                "interpolation_method": "kriging",
                "uncertainty_quantification": True,
                "physics_constraints": True
            },
            "output_format": "vtk",
            "enable_gpu_acceleration": True,
            "distributed_processing": False,
            "quality_level": "high"
        }
        
        async with self.session.post(
            f"{self.base_url}/geological-modeling",
            json=modeling_request
        ) as response:
            return await response.json()
    
    async def start_magmatic_analysis(self) -> Dict[str, Any]:
        """Start magmatic system analysis"""
        # Create sample seismic tomography data
        seismic_data = {
            'tomography_data': {
                'p_wave_velocities': {
                    'depths': [0, 5, 10, 15, 20, 30],  # km
                    'velocities': [5.8, 6.2, 6.5, 6.8, 7.0, 7.2]  # km/s
                },
                's_wave_velocities': {
                    'depths': [0, 5, 10, 15, 20, 30],  # km
                    'velocities': [3.2, 3.5, 3.7, 3.9, 4.0, 4.1]  # km/s
                },
                'resolution': {
                    'horizontal': 2.0,  # km
                    'vertical': 1.0     # km
                }
            },
            'velocity_models': {
                'layers': [
                    {'depth_top': 0, 'depth_bottom': 5, 'vp': 5.8, 'vs': 3.2},
                    {'depth_top': 5, 'depth_bottom': 15, 'vp': 6.5, 'vs': 3.7},
                    {'depth_top': 15, 'depth_bottom': 30, 'vp': 7.0, 'vs': 4.0}
                ],
                'anomalies': [
                    {'depth': 8, 'vp_anomaly': -0.5, 'vs_anomaly': -0.3, 'extent': 5},
                    {'depth': 18, 'vp_anomaly': -0.8, 'vs_anomaly': -0.5, 'extent': 8}
                ]
            }
        }
        
        geophysical_data = {
            'gravity_data': {
                'stations': 50,
                'anomaly_range': [-20, 15],  # mGal
                'resolution': 1.0  # km
            },
            'magnetic_data': {
                'total_field': 45000,  # nT
                'anomaly_range': [-500, 800],  # nT
                'resolution': 0.5  # km
            }
        }
        
        analysis_request = {
            "job_name": f"magmatic_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "seismic_data": seismic_data,
            "geophysical_data": geophysical_data,
            "analysis_type": "comprehensive",
            "uncertainty_quantification": True,
            "physics_constraints": True
        }
        
        async with self.session.post(
            f"{self.base_url}/magmatic-analysis",
            json=analysis_request
        ) as response:
            return await response.json()
    
    async def monitor_job_progress(self, job_id: str, max_wait_time: int = 300) -> Dict[str, Any]:
        """Monitor job progress until completion"""
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            try:
                async with self.session.get(f"{self.base_url}/jobs/{job_id}") as response:
                    if response.status == 200:
                        job_info = await response.json()
                        status = job_info.get('status', 'unknown')
                        
                        print(f"Job {job_id}: {status}")
                        
                        if status in ['completed', 'failed', 'cancelled']:
                            return job_info
                        
                        # Show progress if available
                        if 'progress_percentage' in job_info:
                            print(f"  Progress: {job_info['progress_percentage']:.1f}%")
                    
                    await asyncio.sleep(2)  # Check every 2 seconds
                    
            except Exception as e:
                print(f"Error monitoring job: {e}")
                await asyncio.sleep(5)
        
        print(f"Job monitoring timed out after {max_wait_time} seconds")
        return {}
    
    async def list_all_jobs(self) -> List[Dict[str, Any]]:
        """List all jobs"""
        async with self.session.get(f"{self.base_url}/jobs") as response:
            return await response.json()
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        async with self.session.get(f"{self.base_url}/performance/metrics") as response:
            return await response.json()
    
    async def run_performance_benchmark(self) -> Dict[str, Any]:
        """Run performance benchmark"""
        async with self.session.get(f"{self.base_url}/performance/benchmark") as response:
            return await response.json()

async def run_comprehensive_demo():
    """Run comprehensive analytics demo"""
    print("🔬 Advanced Analytics Service Demo")
    print("=" * 60)
    
    async with AdvancedAnalyticsDemo() as demo:
        try:
            # 1. Check service health
            print("\n1. Checking service health...")
            health = await demo.check_health()
            print(f"   Status: {health.get('status', 'unknown')}")
            print(f"   GPU Available: {health.get('gpu_available', False)}")
            print(f"   Active Jobs: {health.get('active_jobs', 0)}")
            print(f"   Components: {len(health.get('components', {}))}")
            
            # 2. Start geological modeling job
            print("\n2. Starting 3D geological modeling...")
            geo_job = await demo.start_geological_modeling()
            geo_job_id = geo_job.get('job_id')
            print(f"   Job ID: {geo_job_id}")
            print(f"   Status: {geo_job.get('status')}")
            print(f"   Message: {geo_job.get('message')}")
            
            # 3. Start magmatic analysis job
            print("\n3. Starting magmatic system analysis...")
            mag_job = await demo.start_magmatic_analysis()
            mag_job_id = mag_job.get('job_id')
            print(f"   Job ID: {mag_job_id}")
            print(f"   Status: {mag_job.get('status')}")
            print(f"   Message: {mag_job.get('message')}")
            
            # 4. Monitor geological modeling job
            if geo_job_id:
                print(f"\n4. Monitoring geological modeling job {geo_job_id}...")
                geo_result = await demo.monitor_job_progress(geo_job_id, max_wait_time=60)
                
                if geo_result.get('status') == 'completed':
                    print("   ✅ Geological modeling completed successfully!")
                    results = geo_result.get('results', {})
                    if 'model_data' in results:
                        model_data = results['model_data']
                        print(f"   Model vertices: {model_data.get('vertices', 'N/A')}")
                        print(f"   Model faces: {model_data.get('faces', 'N/A')}")
                    
                    if 'quality_metrics' in results:
                        quality = results['quality_metrics']
                        print(f"   Accuracy: {quality.get('accuracy', 'N/A')}")
                        print(f"   Completeness: {quality.get('completeness', 'N/A')}")
                else:
                    print(f"   ❌ Geological modeling job status: {geo_result.get('status', 'unknown')}")
            
            # 5. Monitor magmatic analysis job
            if mag_job_id:
                print(f"\n5. Monitoring magmatic analysis job {mag_job_id}...")
                mag_result = await demo.monitor_job_progress(mag_job_id, max_wait_time=60)
                
                if mag_result.get('status') == 'completed':
                    print("   ✅ Magmatic analysis completed successfully!")
                    results = mag_result.get('results', {})
                    
                    if 'final_model' in results:
                        final_model = results['final_model']
                        if 'magmatic_chambers' in final_model:
                            chambers = final_model['magmatic_chambers']
                            print(f"   Detected {len(chambers)} magmatic chambers:")
                            for i, chamber in enumerate(chambers):
                                print(f"     Chamber {i+1}: Depth {chamber.get('depth', 'N/A')} km, "
                                      f"Volume {chamber.get('volume', 'N/A')} km³")
                    
                    if 'validation_metrics' in results:
                        validation = results['validation_metrics']
                        print(f"   Validation accuracy: {validation.get('accuracy', 'N/A')}")
                else:
                    print(f"   ❌ Magmatic analysis job status: {mag_result.get('status', 'unknown')}")
            
            # 6. List all jobs
            print("\n6. Listing all jobs...")
            all_jobs = await demo.list_all_jobs()
            if all_jobs:
                for job in all_jobs:
                    print(f"   - {job.get('job_name', 'Unknown')}: {job.get('status', 'unknown')} "
                          f"({job.get('progress_percentage', 0):.1f}%)")
            else:
                print("   No jobs found")
            
            # 7. Get performance metrics
            print("\n7. Getting performance metrics...")
            try:
                metrics = await demo.get_performance_metrics()
                print(f"   System metrics available: {len(metrics) if metrics else 0} categories")
                if metrics:
                    for category, values in metrics.items():
                        if isinstance(values, dict):
                            print(f"   {category}: {len(values)} metrics")
                        else:
                            print(f"   {category}: {values}")
            except Exception as e:
                print(f"   Performance metrics not available: {e}")
            
            # 8. Demonstrate batch processing
            print("\n8. Demonstrating batch processing capabilities...")
            print("   (This would process multiple geological samples in parallel)")
            print("   - Batch size optimization based on available resources")
            print("   - Distributed processing across multiple nodes")
            print("   - GPU acceleration for computationally intensive tasks")
            print("   - Real-time progress monitoring and resource usage")
            
            # 9. Show advanced features
            print("\n9. Advanced Analytics Features:")
            print("   🧠 Machine Learning Integration:")
            print("     - Graph Neural Networks for geological relationships")
            print("     - Physics-informed neural networks for constraints")
            print("     - Uncertainty quantification with GANs")
            
            print("   🌍 Geological Modeling:")
            print("     - 3D geological structure reconstruction")
            print("     - Magmatic system analysis from seismic data")
            print("     - Stress field mapping and visualization")
            
            print("   ⚡ High-Performance Computing:")
            print("     - GPU acceleration with CUDA")
            print("     - Distributed computing with Dask/Ray")
            print("     - Adaptive mesh refinement")
            print("     - Memory-efficient processing")
            
            print("   📊 Advanced Analytics:")
            print("     - Crystallographic structure analysis")
            print("     - Thermodynamic calculations")
            print("     - Mantle convection simulation")
            print("     - Plate evolution modeling")
            
        except aiohttp.ClientError as e:
            print(f"❌ Connection error: {e}")
            print("   Make sure the Advanced Analytics Service is running on http://localhost:8004")
        except Exception as e:
            print(f"❌ Demo error: {e}")
    
    print("\n✅ Advanced Analytics Demo completed!")
    print("\nNext steps:")
    print("- Explore the API documentation at http://localhost:8004/docs")
    print("- Set up distributed computing clusters for large-scale analysis")
    print("- Configure GPU acceleration for neural network computations")
    print("- Integrate with geological databases and monitoring systems")
    print("- Implement custom analysis workflows for specific research needs")

def run_sync_demo():
    """Synchronous wrapper for the demo"""
    asyncio.run(run_comprehensive_demo())

if __name__ == "__main__":
    run_sync_demo()
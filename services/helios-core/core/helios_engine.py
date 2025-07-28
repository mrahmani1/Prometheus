"""Helios AI Engine - Core Physics-Informed Neural Network Engine"""

import asyncio
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
import torch
import numpy as np
import structlog

from .config import Settings
from .service_registry import ServiceRegistry
from .pinn_engine import PINNEngine, GeologicalProcess
from .temporal_processing import (
    TimeScale, GeologicalProcess as TemporalGeologicalProcess,
    create_default_temporal_scales
)
from models.ai_models import (
    PINNModelConfig, PhysicsConstraint, PhysicsConstraintType,
    NetworkArchitecture, ActivationFunction
)
from .geodynamic_modeling import IntegratedGeodynamicModeler
from .geological_simulation import LongTermGeologicalSimulator
from .mantle_dynamics import DeepEarthMantleDynamicsSimulator
from .gan_engine import GANEngine, GANConfig, GANEnsemble
from .uncertainty_quantification import GeologicalUncertaintyQuantifier, UncertaintyConfig
from .gan_pinn_integration import GANPINNIntegrationEngine, GANPINNConfig
from .insar_deformation_analyzer import InSARDeformationAnalyzer
from .earthquake_forecasting import EarthquakeForecastingEngine
from .transfer_learning_engine import TransferLearningEngine, GeologicalDomain, AdaptationConfig
from .domain_adaptation_utils import GeologicalDomainRegistry, AdaptationStrategySelector

logger = structlog.get_logger()


class HeliosEngine:
    """
    Core Helios AI Engine for Physics-Informed Neural Network processing
    Handles geodynamic simulations and real-time volcanic monitoring
    """
    
    def __init__(self, config: Settings, service_registry: ServiceRegistry):
        self.config = config
        self.service_registry = service_registry
        self.models: Dict[str, torch.nn.Module] = {}
        self.device = self._setup_device()
        self.is_initialized = False
        self.model_cache_dir = config.model_cache_dir
        
        # Initialize PINN Engine
        self.pinn_engine = PINNEngine(device=self.device)
        
        # Initialize Integrated Geodynamic Modeler
        self.geodynamic_modeler = None
        
        # Initialize InSAR Deformation Analyzer
        self.insar_analyzer = InSARDeformationAnalyzer(device=self.device)
        
        # Initialize Long-term Geological Simulator
        self.geological_simulator = None
        
        # Initialize Deep Earth Mantle Dynamics Simulator
        self.mantle_dynamics_simulator = None
        
        # Initialize GAN Engine for uncertainty quantification
        self.gan_engine = None
        
        # Initialize Uncertainty Quantifier
        self.uncertainty_quantifier = None
        
        # Initialize GAN-PINN Integration Engine
        self.gan_pinn_integration = None
        
        # Initialize Earthquake Forecasting Engine
        self.earthquake_forecasting = None
        
        # Initialize Transfer Learning Engine
        self.transfer_learning_engine = None
        
        # Initialize Domain Registry
        self.domain_registry = None
        
        # Initialize Strategy Selector
        self.strategy_selector = None
        
        # Physics constraints and parameters
        self.physics_constraints = {
            "navier_stokes": True,
            "heat_transfer": True,
            "elastic_deformation": True,
            "mass_conservation": True
        }
        
        # Simulation parameters
        self.simulation_params = {
            "default_resolution": config.default_simulation_resolution,
            "max_time": config.max_simulation_time,
            "solver_tolerance": config.physics_solver_tolerance
        }
        
    def _setup_device(self) -> torch.device:
        """Setup compute device (CPU/GPU)"""
        if self.config.gpu_enabled and torch.cuda.is_available():
            if self.config.gpu_device_ids:
                device_id = self.config.gpu_device_ids[0]
                device = torch.device(f"cuda:{device_id}")
            else:
                device = torch.device("cuda")
            logger.info("Using GPU device", device=str(device))
        else:
            device = torch.device("cpu")
            logger.info("Using CPU device")
        
        return device
    
    async def initialize(self):
        """Initialize the Helios engine"""
        try:
            logger.info("Initializing Helios AI Engine")
            
            # Create model cache directory
            os.makedirs(self.model_cache_dir, exist_ok=True)
            
            # Initialize core models
            await self._load_core_models()
            
            # Setup physics solvers
            await self._setup_physics_solvers()
            
            # Initialize database connections
            await self._setup_database_connections()
            
            # Initialize integrated geodynamic modeler
            self.geodynamic_modeler = IntegratedGeodynamicModeler(self.pinn_engine, self.device)
            
            # Initialize long-term geological simulator
            self.geological_simulator = LongTermGeologicalSimulator(self.pinn_engine, self.device)
            
            # Initialize deep Earth mantle dynamics simulator
            self.mantle_dynamics_simulator = DeepEarthMantleDynamicsSimulator(self.pinn_engine, self.device)
            
            # Initialize GAN engine for uncertainty quantification
            self.gan_engine = GANEngine(self.device, self.model_cache_dir + "/gan")
            
            # Initialize uncertainty quantifier
            self.uncertainty_quantifier = GeologicalUncertaintyQuantifier(self.gan_engine, self.device)
            
            # Initialize GAN-PINN integration engine
            self.gan_pinn_integration = GANPINNIntegrationEngine(self.pinn_engine, self.gan_engine, self.device)
            
            # Initialize earthquake forecasting engine
            self.earthquake_forecasting = EarthquakeForecastingEngine()
            
            # Initialize transfer learning components
            self.transfer_learning_engine = TransferLearningEngine()
            self.domain_registry = GeologicalDomainRegistry()
            self.strategy_selector = AdaptationStrategySelector()
            
            self.is_initialized = True
            logger.info("Helios AI Engine initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize Helios engine", error=str(e))
            raise
    
    async def _load_core_models(self):
        """Load core AI models"""
        try:
            # PINN Model for geological processes
            self.models["pinn_geological"] = await self._create_pinn_model(
                input_dim=6,  # x, y, z, t, temperature, pressure
                output_dim=4,  # velocity_x, velocity_y, velocity_z, temperature
                hidden_layers=[128, 256, 256, 128],
                physics_constraints=["navier_stokes", "heat_transfer"]
            )
            
            # CNN Model for seismic event detection
            self.models["cnn_seismic"] = await self._create_cnn_model(
                input_channels=3,  # 3-component seismogram
                num_classes=5,     # earthquake, explosion, volcanic, noise, unknown
                architecture="resnet18"
            )
            
            # GAN Model for uncertainty quantification
            self.models["gan_uncertainty"] = await self._create_gan_model(
                latent_dim=100,
                output_dim=64,  # geological scenario parameters
                generator_layers=[256, 512, 512, 256],
                discriminator_layers=[256, 512, 256, 1]
            )
            
            logger.info("Core models loaded successfully", 
                       model_count=len(self.models))
            
        except Exception as e:
            logger.error("Failed to load core models", error=str(e))
            raise
    
    async def _create_pinn_model(
        self, 
        input_dim: int, 
        output_dim: int, 
        hidden_layers: List[int],
        physics_constraints: List[str]
    ) -> torch.nn.Module:
        """Create Physics-Informed Neural Network model"""
        
        class PINNModel(torch.nn.Module):
            def __init__(self, input_dim, output_dim, hidden_layers):
                super().__init__()
                
                layers = []
                prev_dim = input_dim
                
                for hidden_dim in hidden_layers:
                    layers.extend([
                        torch.nn.Linear(prev_dim, hidden_dim),
                        torch.nn.Tanh(),
                        torch.nn.Dropout(0.1)
                    ])
                    prev_dim = hidden_dim
                
                layers.append(torch.nn.Linear(prev_dim, output_dim))
                self.network = torch.nn.Sequential(*layers)
                
                # Physics constraint weights
                self.physics_weights = torch.nn.Parameter(
                    torch.ones(len(physics_constraints))
                )
                
            def forward(self, x):
                return self.network(x)
            
            def physics_loss(self, x, y_pred):
                """Compute physics-informed loss"""
                # This would implement actual physics constraints
                # For now, return a placeholder
                return torch.tensor(0.0, requires_grad=True)
        
        model = PINNModel(input_dim, output_dim, hidden_layers)
        model.to(self.device)
        
        return model
    
    async def _create_cnn_model(
        self, 
        input_channels: int, 
        num_classes: int, 
        architecture: str
    ) -> torch.nn.Module:
        """Create CNN model for seismic event detection"""
        
        class SeismicCNN(torch.nn.Module):
            def __init__(self, input_channels, num_classes):
                super().__init__()
                
                self.conv_layers = torch.nn.Sequential(
                    torch.nn.Conv1d(input_channels, 64, kernel_size=7, padding=3),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool1d(2),
                    
                    torch.nn.Conv1d(64, 128, kernel_size=5, padding=2),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool1d(2),
                    
                    torch.nn.Conv1d(128, 256, kernel_size=3, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool1d(2),
                    
                    torch.nn.AdaptiveAvgPool1d(1)
                )
                
                self.classifier = torch.nn.Sequential(
                    torch.nn.Dropout(0.5),
                    torch.nn.Linear(256, 128),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.3),
                    torch.nn.Linear(128, num_classes)
                )
                
            def forward(self, x):
                features = self.conv_layers(x)
                features = features.view(features.size(0), -1)
                return self.classifier(features)
        
        model = SeismicCNN(input_channels, num_classes)
        model.to(self.device)
        
        return model
    
    async def _create_gan_model(
        self, 
        latent_dim: int, 
        output_dim: int,
        generator_layers: List[int],
        discriminator_layers: List[int]
    ) -> Dict[str, torch.nn.Module]:
        """Create GAN model for uncertainty quantification"""
        
        class Generator(torch.nn.Module):
            def __init__(self, latent_dim, output_dim, hidden_layers):
                super().__init__()
                
                layers = []
                prev_dim = latent_dim
                
                for hidden_dim in hidden_layers:
                    layers.extend([
                        torch.nn.Linear(prev_dim, hidden_dim),
                        torch.nn.ReLU(),
                        torch.nn.BatchNorm1d(hidden_dim)
                    ])
                    prev_dim = hidden_dim
                
                layers.extend([
                    torch.nn.Linear(prev_dim, output_dim),
                    torch.nn.Tanh()
                ])
                
                self.network = torch.nn.Sequential(*layers)
                
            def forward(self, z):
                return self.network(z)
        
        class Discriminator(torch.nn.Module):
            def __init__(self, input_dim, hidden_layers):
                super().__init__()
                
                layers = []
                prev_dim = input_dim
                
                for hidden_dim in hidden_layers[:-1]:
                    layers.extend([
                        torch.nn.Linear(prev_dim, hidden_dim),
                        torch.nn.LeakyReLU(0.2),
                        torch.nn.Dropout(0.3)
                    ])
                    prev_dim = hidden_dim
                
                layers.append(torch.nn.Linear(prev_dim, hidden_layers[-1]))
                layers.append(torch.nn.Sigmoid())
                
                self.network = torch.nn.Sequential(*layers)
                
            def forward(self, x):
                return self.network(x)
        
        generator = Generator(latent_dim, output_dim, generator_layers)
        discriminator = Discriminator(output_dim, discriminator_layers)
        
        generator.to(self.device)
        discriminator.to(self.device)
        
        return {
            "generator": generator,
            "discriminator": discriminator
        }
    
    async def _setup_physics_solvers(self):
        """Setup physics constraint solvers"""
        logger.info("Setting up physics solvers")
        # This would initialize numerical solvers for physics constraints
        # For now, just log that it's set up
        pass
    
    async def _setup_database_connections(self):
        """Setup database connections"""
        logger.info("Setting up database connections")
        # This would initialize database connection pools
        # For now, just log that it's set up
        pass
    
    async def is_healthy(self) -> bool:
        """Check if the engine is healthy"""
        try:
            if not self.is_initialized:
                return False
            
            # Check if models are loaded
            if not self.models:
                return False
            
            # Check device availability
            if self.device.type == "cuda" and not torch.cuda.is_available():
                return False
            
            return True
            
        except Exception as e:
            logger.error("Health check failed", error=str(e))
            return False
    
    async def is_ready(self) -> bool:
        """Check if the engine is ready to process requests"""
        return await self.is_healthy()
    
    async def get_loaded_models_count(self) -> int:
        """Get the number of loaded models"""
        return len(self.models)
    
    async def process_geological_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process geological data using PINN models"""
        try:
            if "pinn_geological" not in self.models:
                raise ValueError("Geological PINN model not loaded")
            
            model = self.models["pinn_geological"]
            
            # Convert input data to tensor
            input_tensor = torch.tensor(
                data["input_features"], 
                dtype=torch.float32, 
                device=self.device
            )
            
            # Run inference
            with torch.no_grad():
                predictions = model(input_tensor)
            
            return {
                "predictions": predictions.cpu().numpy().tolist(),
                "model_version": "pinn_geological_v1.0",
                "timestamp": data.get("timestamp"),
                "confidence": 0.95  # Placeholder
            }
            
        except Exception as e:
            logger.error("Failed to process geological data", error=str(e))
            raise
    
    async def detect_seismic_events(self, waveform_data: np.ndarray) -> Dict[str, Any]:
        """Detect seismic events using CNN model"""
        try:
            if "cnn_seismic" not in self.models:
                raise ValueError("Seismic CNN model not loaded")
            
            model = self.models["cnn_seismic"]
            
            # Convert waveform data to tensor
            input_tensor = torch.tensor(
                waveform_data, 
                dtype=torch.float32, 
                device=self.device
            ).unsqueeze(0)  # Add batch dimension
            
            # Run inference
            with torch.no_grad():
                logits = model(input_tensor)
                probabilities = torch.softmax(logits, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1)
            
            event_types = ["earthquake", "explosion", "volcanic", "noise", "unknown"]
            
            return {
                "event_type": event_types[predicted_class.item()],
                "confidence": probabilities.max().item(),
                "probabilities": {
                    event_type: prob.item() 
                    for event_type, prob in zip(event_types, probabilities[0])
                },
                "model_version": "cnn_seismic_v1.0"
            }
            
        except Exception as e:
            logger.error("Failed to detect seismic events", error=str(e))
            raise
    
    async def generate_uncertainty_scenarios(
        self, 
        num_scenarios: int = 100
    ) -> Dict[str, Any]:
        """Generate uncertainty scenarios using GAN model"""
        try:
            if "gan_uncertainty" not in self.models:
                raise ValueError("Uncertainty GAN model not loaded")
            
            gan_models = self.models["gan_uncertainty"]
            generator = gan_models["generator"]
            
            # Generate random noise
            noise = torch.randn(
                num_scenarios, 100, 
                device=self.device
            )
            
            # Generate scenarios
            with torch.no_grad():
                scenarios = generator(noise)
            
            return {
                "scenarios": scenarios.cpu().numpy().tolist(),
                "num_scenarios": num_scenarios,
                "model_version": "gan_uncertainty_v1.0"
            }
            
        except Exception as e:
            logger.error("Failed to generate uncertainty scenarios", error=str(e))
            raise
    
    async def create_pinn_model(
        self, 
        model_id: str,
        geological_process: GeologicalProcess,
        physics_constraints: List[PhysicsConstraint],
        network_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a new PINN model for geological processes
        
        Args:
            model_id: Unique identifier for the model
            geological_process: Type of geological process
            physics_constraints: List of physics constraints
            network_config: Network architecture configuration
            
        Returns:
            Model creation result
        """
        try:
            # Create network architecture
            architecture = NetworkArchitecture(
                layer_sizes=network_config.get("layer_sizes", [4, 64, 64, 64, 1]),
                activation_functions=[
                    ActivationFunction(af) for af in network_config.get(
                        "activation_functions", ["tanh", "tanh", "tanh", "tanh"]
                    )
                ],
                dropout_rates=network_config.get("dropout_rates", [0.0, 0.1, 0.1, 0.0]),
                batch_normalization=network_config.get("batch_normalization", False),
                residual_connections=network_config.get("residual_connections", True)
            )
            
            # Create PINN configuration
            pinn_config = PINNModelConfig(
                model_id=model_id,
                network_architecture=architecture,
                physics_constraints=physics_constraints,
                optimization_config=network_config.get("optimization_config", {
                    "optimizer": "adam",
                    "learning_rate": 0.001
                }),
                training_config=network_config.get("training_config", {
                    "max_epochs": 100,
                    "batch_size": 64
                })
            )
            
            # Create model using PINN engine
            model = self.pinn_engine.create_model(model_id, pinn_config, geological_process)
            
            logger.info(f"Created PINN model: {model_id} for process: {geological_process}")
            
            return {
                "model_id": model_id,
                "geological_process": geological_process.value,
                "physics_constraints": [pc.constraint_type.value for pc in physics_constraints],
                "status": "created",
                "parameters": sum(p.numel() for p in model.parameters())
            }
            
        except Exception as e:
            logger.error(f"Failed to create PINN model {model_id}", error=str(e))
            raise
    
    async def train_pinn_model(
        self, 
        model_id: str,
        training_data: Dict[str, Any],
        validation_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Train a PINN model
        
        Args:
            model_id: Model identifier
            training_data: Training data dictionary
            validation_data: Validation data dictionary (optional)
            
        Returns:
            Training results
        """
        try:
            trainer = self.pinn_engine.get_trainer(model_id)
            if trainer is None:
                raise ValueError(f"Model {model_id} not found")
            
            # Convert data to tensors
            train_inputs = torch.tensor(training_data["inputs"], dtype=torch.float32)
            train_targets = torch.tensor(training_data["targets"], dtype=torch.float32) if "targets" in training_data else None
            
            # Create data loader
            if train_targets is not None:
                train_dataset = torch.utils.data.TensorDataset(train_inputs, train_targets)
            else:
                train_dataset = torch.utils.data.TensorDataset(train_inputs)
            
            batch_size = trainer.config.training_config.get("batch_size", 64)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            # Validation data loader
            val_loader = None
            if validation_data:
                val_inputs = torch.tensor(validation_data["inputs"], dtype=torch.float32)
                val_targets = torch.tensor(validation_data["targets"], dtype=torch.float32) if "targets" in validation_data else None
                
                if val_targets is not None:
                    val_dataset = torch.utils.data.TensorDataset(val_inputs, val_targets)
                else:
                    val_dataset = torch.utils.data.TensorDataset(val_inputs)
                
                val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            # Train model
            history = trainer.train(train_loader, val_loader)
            
            logger.info(f"PINN model {model_id} training completed")
            
            return {
                "model_id": model_id,
                "status": "trained",
                "final_loss": history["total_loss"][-1] if history["total_loss"] else 0.0,
                "epochs": len(history["epoch"]),
                "history": history
            }
            
        except Exception as e:
            logger.error(f"Failed to train PINN model {model_id}", error=str(e))
            raise
    
    async def predict_with_pinn(
        self, 
        model_id: str,
        inputs: np.ndarray
    ) -> Dict[str, Any]:
        """
        Make predictions using a PINN model
        
        Args:
            model_id: Model identifier
            inputs: Input data array
            
        Returns:
            Prediction results
        """
        try:
            # Convert inputs to tensor
            input_tensor = torch.tensor(inputs, dtype=torch.float32)
            
            # Make predictions
            predictions = self.pinn_engine.predict(model_id, input_tensor)
            
            return {
                "model_id": model_id,
                "predictions": predictions.cpu().numpy().tolist(),
                "input_shape": list(inputs.shape),
                "output_shape": list(predictions.shape),
                "timestamp": np.datetime64('now').isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to make predictions with PINN model {model_id}", error=str(e))
            raise
    
    async def get_pinn_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get information about a PINN model"""
        try:
            return self.pinn_engine.get_model_info(model_id)
        except Exception as e:
            logger.error(f"Failed to get PINN model info for {model_id}", error=str(e))
            raise
    
    async def list_pinn_models(self) -> List[str]:
        """List all available PINN models"""
        return self.pinn_engine.list_models()
    
    async def create_temporal_pinn_model(
        self,
        model_id: str,
        input_dim: int,
        output_dim: int,
        branch_hidden_dims: List[int] = [64, 128, 64],
        attention_heads: int = 8
    ) -> Dict[str, Any]:
        """
        Create a multi-scale temporal PINN model
        
        Args:
            model_id: Unique identifier for the model
            input_dim: Input dimension (spatial coordinates + parameters)
            output_dim: Output dimension (predicted variables)
            branch_hidden_dims: Hidden layer dimensions for each temporal branch
            attention_heads: Number of attention heads for temporal attention
            
        Returns:
            Model creation result
        """
        try:
            temporal_model = self.pinn_engine.create_temporal_model(
                model_id=model_id,
                input_dim=input_dim,
                output_dim=output_dim,
                branch_hidden_dims=branch_hidden_dims,
                attention_heads=attention_heads
            )
            
            logger.info(f"Created temporal PINN model: {model_id}")
            
            return {
                "model_id": model_id,
                "model_type": "temporal_pinn",
                "input_dim": input_dim,
                "output_dim": output_dim,
                "num_temporal_scales": len(create_default_temporal_scales()),
                "status": "created",
                "parameters": sum(p.numel() for p in temporal_model.parameters())
            }
            
        except Exception as e:
            logger.error(f"Failed to create temporal PINN model {model_id}", error=str(e))
            raise
    
    async def predict_temporal_evolution(
        self,
        model_id: str,
        spatial_coordinates: np.ndarray,
        time_start: float,
        time_end: float,
        process_type: str,
        initial_conditions: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Predict temporal evolution of geological processes
        
        Args:
            model_id: Temporal PINN model identifier
            spatial_coordinates: Spatial coordinates [n_points, spatial_dim]
            time_start: Start time in seconds
            time_end: End time in seconds
            process_type: Type of geological process
            initial_conditions: Initial solution state
            
        Returns:
            Temporal evolution predictions
        """
        try:
            # Convert process type string to enum
            process_enum = TemporalGeologicalProcess(process_type)
            
            # Convert inputs to tensors
            inputs = torch.tensor(spatial_coordinates, dtype=torch.float32, device=self.device)
            initial_solution = None
            if initial_conditions is not None:
                initial_solution = torch.tensor(initial_conditions, dtype=torch.float32, device=self.device)
            
            # Make predictions with adaptive time stepping
            results = self.pinn_engine.predict_with_adaptive_stepping(
                model_id=model_id,
                inputs=inputs,
                time_start=time_start,
                time_end=time_end,
                process_type=process_enum,
                initial_solution=initial_solution
            )
            
            # Convert results to numpy arrays for JSON serialization
            return {
                "model_id": model_id,
                "process_type": process_type,
                "time_range": [time_start, time_end],
                "times": results["times"],
                "solutions": [sol.cpu().numpy().tolist() for sol in results["solutions"]],
                "time_steps": results["time_steps"],
                "total_steps": results["total_steps"],
                "final_solution": results["final_solution"].cpu().numpy().tolist()
            }
            
        except Exception as e:
            logger.error(f"Failed to predict temporal evolution with model {model_id}", error=str(e))
            raise
    
    async def create_gan_model(
        self,
        model_id: str,
        latent_dim: int = 100,
        output_dim: int = 64,
        generator_layers: List[int] = [256, 512, 256],
        discriminator_layers: List[int] = [256, 128, 1],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a GAN model for geological scenario generation
        
        Args:
            model_id: Unique identifier for the GAN model
            latent_dim: Dimension of latent noise vector
            output_dim: Dimension of generated geological scenarios
            generator_layers: Hidden layer sizes for generator
            discriminator_layers: Hidden layer sizes for discriminator
            **kwargs: Additional GAN configuration parameters
            
        Returns:
            GAN model creation result
        """
        try:
            # Create GAN configuration
            gan_config = GANConfig(
                model_id=model_id,
                latent_dim=latent_dim,
                output_dim=output_dim,
                generator_layers=generator_layers,
                discriminator_layers=discriminator_layers,
                **kwargs
            )
            
            # Create GAN model
            gan = self.gan_engine.create_gan(gan_config)
            
            logger.info(f"Created GAN model: {model_id}")
            
            return {
                "model_id": model_id,
                "model_type": "geological_gan",
                "latent_dim": latent_dim,
                "output_dim": output_dim,
                "generator_parameters": sum(p.numel() for p in gan.generator.parameters()),
                "discriminator_parameters": sum(p.numel() for p in gan.discriminator.parameters()),
                "status": "created"
            }
            
        except Exception as e:
            logger.error(f"Failed to create GAN model {model_id}", error=str(e))
            raise
    
    async def create_gan_ensemble(
        self,
        ensemble_id: str,
        num_models: int = 5,
        base_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a GAN ensemble for uncertainty quantification
        
        Args:
            ensemble_id: Unique identifier for the ensemble
            num_models: Number of GAN models in the ensemble
            base_config: Base configuration for ensemble models
            
        Returns:
            Ensemble creation result
        """
        try:
            if base_config is None:
                base_config = {
                    "latent_dim": 100,
                    "output_dim": 64,
                    "generator_layers": [256, 512, 256],
                    "discriminator_layers": [256, 128, 1]
                }
            
            # Create ensemble configurations with slight variations
            ensemble_configs = []
            for i in range(num_models):
                config = GANConfig(
                    model_id=f"{ensemble_id}_model_{i}",
                    latent_dim=base_config["latent_dim"],
                    output_dim=base_config["output_dim"],
                    generator_layers=base_config["generator_layers"],
                    discriminator_layers=base_config["discriminator_layers"],
                    learning_rate=0.0002 + i * 0.0001,  # Slight variation
                    beta1=0.5 + i * 0.01,  # Slight variation
                    noise_type="normal" if i % 2 == 0 else "uniform"  # Alternate noise types
                )
                ensemble_configs.append(config)
            
            # Create ensemble
            ensemble = self.gan_engine.create_ensemble(ensemble_id, ensemble_configs)
            
            logger.info(f"Created GAN ensemble: {ensemble_id} with {num_models} models")
            
            return {
                "ensemble_id": ensemble_id,
                "num_models": num_models,
                "model_ids": [config.model_id for config in ensemble_configs],
                "total_parameters": sum(
                    sum(p.numel() for p in gan.generator.parameters()) +
                    sum(p.numel() for p in gan.discriminator.parameters())
                    for gan in ensemble.gans
                ),
                "status": "created"
            }
            
        except Exception as e:
            logger.error(f"Failed to create GAN ensemble {ensemble_id}", error=str(e))
            raise
    
    async def generate_geological_scenarios(
        self,
        model_id: str,
        num_scenarios: int = 100,
        geological_constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate geological scenarios using GAN model
        
        Args:
            model_id: GAN model or ensemble identifier
            num_scenarios: Number of scenarios to generate
            geological_constraints: Optional geological constraints
            
        Returns:
            Generated geological scenarios
        """
        try:
            # Check if it's an ensemble or single model
            if model_id in self.gan_engine.ensembles:
                # Use uncertainty quantifier for ensemble
                results = self.uncertainty_quantifier.generate_geological_scenarios_with_uncertainty(
                    model_id, num_scenarios, geological_constraints
                )
                results["generation_type"] = "ensemble"
            else:
                # Single GAN model
                gan = self.gan_engine.get_gan(model_id)
                scenarios = gan.generate_scenarios(num_scenarios)
                plausibility = gan.assess_plausibility(scenarios)
                
                results = {
                    "scenarios": scenarios.cpu().numpy().tolist(),
                    "plausibility_scores": plausibility.cpu().numpy().tolist(),
                    "num_scenarios": num_scenarios,
                    "generation_type": "single_model",
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            logger.info(f"Generated {num_scenarios} geological scenarios using {model_id}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to generate geological scenarios with {model_id}", error=str(e))
            raise
    
    async def quantify_prediction_uncertainty(
        self,
        model_id: str,
        inputs: np.ndarray,
        method: str = "ensemble_variance",
        num_samples: int = 100,
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """
        Quantify uncertainty in geological predictions
        
        Args:
            model_id: Model identifier for uncertainty quantification
            inputs: Input data for predictions
            method: Uncertainty quantification method
            num_samples: Number of samples for uncertainty estimation
            confidence_level: Confidence level for intervals
            
        Returns:
            Uncertainty quantification results
        """
        try:
            # Convert method string to enum
            method_enum = UncertaintyMethod(method)
            
            # Create uncertainty configuration
            config = UncertaintyConfig(
                method=method_enum,
                num_samples=num_samples,
                confidence_level=confidence_level
            )
            
            # Convert inputs to tensor
            input_tensor = torch.tensor(inputs, dtype=torch.float32, device=self.device)
            
            # Quantify uncertainty
            uncertainty_result = self.uncertainty_quantifier.quantify_uncertainty(
                model_id, input_tensor, config
            )
            
            # Convert results for JSON serialization
            results = {
                "method": method,
                "confidence_level": confidence_level,
                "confidence_intervals": uncertainty_result.confidence_intervals,
                "epistemic_uncertainty": uncertainty_result.epistemic_uncertainty.tolist(),
                "aleatoric_uncertainty": uncertainty_result.aleatoric_uncertainty.tolist(),
                "total_uncertainty": uncertainty_result.total_uncertainty.tolist(),
                "calibration_metrics": uncertainty_result.calibration_metrics,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Quantified uncertainty for {model_id} using {method}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to quantify uncertainty for {model_id}", error=str(e))
            raise
    
    async def train_gan_model(
        self,
        model_id: str,
        training_data: List[Dict[str, Any]],
        num_epochs: int = 100,
        batch_size: int = 64,
        **training_kwargs
    ) -> Dict[str, Any]:
        """
        Train a GAN model on geological data
        
        Args:
            model_id: GAN model identifier
            training_data: List of geological data samples
            num_epochs: Number of training epochs
            batch_size: Training batch size
            **training_kwargs: Additional training parameters
            
        Returns:
            Training results
        """
        try:
            from .gan_training import GANTrainingManager, TrainingConfig
            
            # Create training manager
            training_manager = GANTrainingManager(self.device)
            
            # Get existing GAN
            gan = self.gan_engine.get_gan(model_id)
            
            # Create training configuration
            training_config = TrainingConfig(
                num_epochs=num_epochs,
                batch_size=batch_size,
                **training_kwargs
            )
            
            # Create trainer with existing GAN
            trainer_id = f"{model_id}_trainer"
            # Note: This would need modification to GANTrainingManager to accept existing GAN
            # For now, we'll use a simplified approach
            
            # Create data loader
            dataloader = training_manager.create_geological_dataset(training_data, batch_size)
            
            # Train GAN directly
            history = gan.train(dataloader, num_epochs)
            
            logger.info(f"GAN model {model_id} training completed")
            
            return {
                "model_id": model_id,
                "status": "trained",
                "epochs": num_epochs,
                "final_g_loss": history["g_loss"][-1] if history["g_loss"] else 0.0,
                "final_d_loss": history["d_loss"][-1] if history["d_loss"] else 0.0,
                "history": history
            }
            
        except Exception as e:
            logger.error(f"Failed to train GAN model {model_id}", error=str(e))
            raise
    
    async def get_gan_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get information about a GAN model"""
        try:
            if model_id in self.gan_engine.gans:
                gan = self.gan_engine.get_gan(model_id)
                return {
                    "model_id": model_id,
                    "model_type": "single_gan",
                    "config": gan.config.__dict__,
                    "metadata": gan.metadata.to_dict(),
                    "generator_parameters": sum(p.numel() for p in gan.generator.parameters()),
                    "discriminator_parameters": sum(p.numel() for p in gan.discriminator.parameters())
                }
            elif model_id in self.gan_engine.ensembles:
                ensemble = self.gan_engine.get_ensemble(model_id)
                return {
                    "ensemble_id": model_id,
                    "model_type": "gan_ensemble",
                    "num_models": len(ensemble.gans),
                    "model_ids": [gan.config.model_id for gan in ensemble.gans],
                    "total_parameters": sum(
                        sum(p.numel() for p in gan.generator.parameters()) +
                        sum(p.numel() for p in gan.discriminator.parameters())
                        for gan in ensemble.gans
                    )
                }
            else:
                raise ValueError(f"Model {model_id} not found")
                
        except Exception as e:
            logger.error(f"Failed to get GAN model info for {model_id}", error=str(e))
            raise
    
    async def list_gan_models(self) -> Dict[str, List[str]]:
        """List all available GAN models and ensembles"""
        return {
            "single_models": self.gan_engine.list_gans(),
            "ensembles": self.gan_engine.list_ensembles()
        }
    
    async def create_probabilistic_geological_model(
        self,
        integration_id: str,
        gan_model_id: str,
        pinn_model_id: str,
        coupling_method: str = "ensemble_sampling",
        num_ensemble_samples: int = 100,
        confidence_level: float = 0.95,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a probabilistic geological model integrating GAN and PINN
        
        Args:
            integration_id: Unique identifier for the integration
            gan_model_id: GAN model or ensemble identifier
            pinn_model_id: PINN model identifier
            coupling_method: Method for coupling GAN and PINN
            num_ensemble_samples: Number of ensemble samples
            confidence_level: Confidence level for uncertainty quantification
            **kwargs: Additional configuration parameters
            
        Returns:
            Integration creation result
        """
        try:
            # Create GAN-PINN configuration
            config = GANPINNConfig(
                integration_id=integration_id,
                gan_model_id=gan_model_id,
                pinn_model_id=pinn_model_id,
                coupling_method=coupling_method,
                num_ensemble_samples=num_ensemble_samples,
                confidence_level=confidence_level,
                **kwargs
            )
            
            # Create probabilistic model
            created_id = self.gan_pinn_integration.create_probabilistic_model(config)
            
            logger.info(f"Created probabilistic geological model: {integration_id}")
            
            return {
                "integration_id": created_id,
                "gan_model_id": gan_model_id,
                "pinn_model_id": pinn_model_id,
                "coupling_method": coupling_method,
                "num_ensemble_samples": num_ensemble_samples,
                "confidence_level": confidence_level,
                "status": "created",
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to create probabilistic geological model {integration_id}", error=str(e))
            raise
    
    async def predict_probabilistic_geological_outcomes(
        self,
        integration_id: str,
        inputs: np.ndarray,
        geological_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make probabilistic geological predictions with uncertainty quantification
        
        Args:
            integration_id: Integration identifier
            inputs: Input data for predictions
            geological_context: Optional geological context information
            
        Returns:
            Probabilistic prediction results with uncertainty
        """
        try:
            # Make probabilistic prediction
            results = self.gan_pinn_integration.predict_probabilistic(
                integration_id, inputs, geological_context
            )
            
            logger.info(f"Made probabilistic prediction for integration {integration_id}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to make probabilistic prediction for {integration_id}", error=str(e))
            raise
    
    async def train_adversarial_pinn_model(
        self,
        pinn_model_id: str,
        gan_model_id: str,
        training_data: List[Dict[str, Any]],
        num_epochs: int = 100,
        batch_size: int = 64,
        adversarial_weight: float = 0.1,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train PINN model with adversarial feedback from GAN
        
        Args:
            pinn_model_id: PINN model identifier
            gan_model_id: GAN model identifier
            training_data: Training data samples
            num_epochs: Number of training epochs
            batch_size: Training batch size
            adversarial_weight: Weight for adversarial loss
            **kwargs: Additional training parameters
            
        Returns:
            Training results
        """
        try:
            from .gan_training import GANTrainingManager
            
            # Create data loader
            training_manager = GANTrainingManager(self.device)
            dataloader = training_manager.create_geological_dataset(training_data, batch_size)
            
            # Train with adversarial feedback
            history = self.gan_pinn_integration.train_adversarial_pinn(
                pinn_model_id=pinn_model_id,
                gan_model_id=gan_model_id,
                training_data=dataloader,
                num_epochs=num_epochs,
                adversarial_weight=adversarial_weight,
                **kwargs
            )
            
            logger.info(f"Adversarial PINN training completed for {pinn_model_id}")
            
            return {
                "pinn_model_id": pinn_model_id,
                "gan_model_id": gan_model_id,
                "status": "trained",
                "epochs": num_epochs,
                "adversarial_weight": adversarial_weight,
                "final_pinn_loss": history["pinn_loss"][-1] if history["pinn_loss"] else 0.0,
                "final_adversarial_loss": history["adversarial_loss"][-1] if history["adversarial_loss"] else 0.0,
                "history": history,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to train adversarial PINN model {pinn_model_id}", error=str(e))
            raise
    
    async def assess_geological_hazard_probability(
        self,
        integration_id: str,
        location: Dict[str, float],
        time_horizon: float,
        hazard_types: List[str] = ["volcanic_eruption", "earthquake", "landslide"],
        scenario_count: int = 1000
    ) -> Dict[str, Any]:
        """
        Assess geological hazard probabilities using probabilistic models
        
        Args:
            integration_id: Integration identifier
            location: Location coordinates (latitude, longitude, depth)
            time_horizon: Time horizon for hazard assessment (years)
            hazard_types: Types of hazards to assess
            scenario_count: Number of scenarios for probability estimation
            
        Returns:
            Hazard probability assessment results
        """
        try:
            # Prepare input data
            inputs = np.array([[
                location.get("latitude", 0.0),
                location.get("longitude", 0.0),
                location.get("depth", 0.0),
                time_horizon
            ]])
            
            # Geological context
            geological_context = {
                "location": location,
                "time_horizon": time_horizon,
                "assessment_type": "hazard_probability"
            }
            
            # Make probabilistic prediction
            prediction_results = await self.predict_probabilistic_geological_outcomes(
                integration_id, inputs, geological_context
            )
            
            # Convert predictions to hazard probabilities
            hazard_probabilities = self._convert_predictions_to_hazard_probabilities(
                prediction_results, hazard_types, scenario_count
            )
            
            results = {
                "location": location,
                "time_horizon": time_horizon,
                "hazard_types": hazard_types,
                "hazard_probabilities": hazard_probabilities,
                "confidence_intervals": prediction_results.get("confidence_intervals", {}),
                "total_uncertainty": prediction_results.get("total_uncertainty", []),
                "scenario_count": scenario_count,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Assessed geological hazard probabilities for location {location}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to assess geological hazard probabilities", error=str(e))
            raise
    
    def _convert_predictions_to_hazard_probabilities(
        self,
        prediction_results: Dict[str, Any],
        hazard_types: List[str],
        scenario_count: int
    ) -> Dict[str, Dict[str, float]]:
        """
        Convert model predictions to hazard probabilities
        
        Args:
            prediction_results: Probabilistic prediction results
            hazard_types: Types of hazards to assess
            scenario_count: Number of scenarios
            
        Returns:
            Dictionary of hazard probabilities
        """
        mean_prediction = np.array(prediction_results.get("mean_prediction", []))
        std_prediction = np.array(prediction_results.get("std_prediction", []))
        
        hazard_probabilities = {}
        
        for i, hazard_type in enumerate(hazard_types):
            if i < len(mean_prediction[0]):
                # Convert prediction to probability (simplified approach)
                # In practice, this would use domain-specific conversion functions
                mean_val = mean_prediction[0][i]
                std_val = std_prediction[0][i] if len(std_prediction) > 0 else 0.1
                
                # Use sigmoid to convert to probability
                probability = 1 / (1 + np.exp(-mean_val))
                
                # Calculate confidence bounds
                lower_bound = 1 / (1 + np.exp(-(mean_val - 1.96 * std_val)))
                upper_bound = 1 / (1 + np.exp(-(mean_val + 1.96 * std_val)))
                
                hazard_probabilities[hazard_type] = {
                    "probability": float(probability),
                    "lower_bound": float(lower_bound),
                    "upper_bound": float(upper_bound),
                    "uncertainty": float(std_val)
                }
            else:
                # Default low probability if no prediction available
                hazard_probabilities[hazard_type] = {
                    "probability": 0.01,
                    "lower_bound": 0.001,
                    "upper_bound": 0.1,
                    "uncertainty": 0.05
                }
        
        return hazard_probabilities
    
    async def get_probabilistic_model_info(self, integration_id: str) -> Dict[str, Any]:
        """Get information about a probabilistic geological model"""
        try:
            return self.gan_pinn_integration.get_integration_info(integration_id)
        except Exception as e:
            logger.error(f"Failed to get probabilistic model info for {integration_id}", error=str(e))
            raise
    
    async def list_probabilistic_models(self) -> List[str]:
        """List all available probabilistic geological models"""
        return self.gan_pinn_integration.list_integrations()
    
    async def predict_multi_scale_temporal(
        self,
        model_id: str,
        spatial_coordinates: np.ndarray,
        time_coordinates: np.ndarray,
        target_scale: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Make multi-scale temporal predictions
        
        Args:
            model_id: Temporal PINN model identifier
            spatial_coordinates: Spatial coordinates [n_points, spatial_dim]
            time_coordinates: Time coordinates [n_points, 1]
            target_scale: Specific temporal scale to focus on
            
        Returns:
            Multi-scale temporal predictions
        """
        try:
            # Convert inputs to tensors
            inputs = torch.tensor(spatial_coordinates, dtype=torch.float32, device=self.device)
            time_coords = torch.tensor(time_coordinates, dtype=torch.float32, device=self.device)
            
            # Convert target scale if provided
            target_scale_enum = None
            if target_scale:
                target_scale_enum = TimeScale(target_scale)
            
            # Make predictions
            results = self.pinn_engine.predict_temporal(
                model_id=model_id,
                inputs=inputs,
                time_coords=time_coords,
                target_scale=target_scale_enum
            )
            
            # Convert results to numpy arrays
            output = {
                "model_id": model_id,
                "fused_prediction": results["fused_prediction"].cpu().numpy().tolist(),
                "branch_predictions": {
                    scale: pred.cpu().numpy().tolist() 
                    for scale, pred in results["branch_predictions"].items()
                },
                "input_shape": list(spatial_coordinates.shape),
                "time_shape": list(time_coordinates.shape)
            }
            
            if "target_scale_prediction" in results:
                output["target_scale_prediction"] = results["target_scale_prediction"].cpu().numpy().tolist()
            
            return output
            
        except Exception as e:
            logger.error(f"Failed to make multi-scale temporal predictions with model {model_id}", error=str(e))
            raise
    
    async def interpolate_geological_timeline(
        self,
        model_id: str,
        known_times: List[float],
        known_solutions: List[List[float]],
        target_times: List[float]
    ) -> Dict[str, Any]:
        """
        Interpolate geological data along timeline
        
        Args:
            model_id: Temporal PINN model identifier
            known_times: Known time points
            known_solutions: Known solutions at those times
            target_times: Target times for interpolation
            
        Returns:
            Interpolated solutions
        """
        try:
            # Convert to tensors
            known_times_tensor = torch.tensor(known_times, dtype=torch.float32)
            known_solutions_tensor = torch.tensor(known_solutions, dtype=torch.float32)
            target_times_tensor = torch.tensor(target_times, dtype=torch.float32)
            
            # Interpolate
            interpolated = self.pinn_engine.interpolate_temporal_solution(
                model_id=model_id,
                known_times=known_times_tensor,
                known_solutions=known_solutions_tensor,
                target_times=target_times_tensor
            )
            
            return {
                "model_id": model_id,
                "known_times": known_times,
                "target_times": target_times,
                "interpolated_solutions": interpolated.cpu().numpy().tolist(),
                "method": "temporal_interpolation"
            }
            
        except Exception as e:
            logger.error(f"Failed to interpolate geological timeline with model {model_id}", error=str(e))
            raise
    
    async def extrapolate_geological_future(
        self,
        model_id: str,
        known_times: List[float],
        known_solutions: List[List[float]],
        future_times: List[float],
        method: str = "polynomial"
    ) -> Dict[str, Any]:
        """
        Extrapolate geological processes into the future
        
        Args:
            model_id: Temporal PINN model identifier
            known_times: Known time points
            known_solutions: Known solutions at those times
            future_times: Future times for extrapolation
            method: Extrapolation method
            
        Returns:
            Extrapolated solutions
        """
        try:
            # Convert to tensors
            known_times_tensor = torch.tensor(known_times, dtype=torch.float32)
            known_solutions_tensor = torch.tensor(known_solutions, dtype=torch.float32)
            future_times_tensor = torch.tensor(future_times, dtype=torch.float32)
            
            # Extrapolate
            extrapolated = self.pinn_engine.extrapolate_temporal_solution(
                model_id=model_id,
                known_times=known_times_tensor,
                known_solutions=known_solutions_tensor,
                target_times=future_times_tensor,
                method=method
            )
            
            return {
                "model_id": model_id,
                "known_times": known_times,
                "future_times": future_times,
                "extrapolated_solutions": extrapolated.cpu().numpy().tolist(),
                "method": method,
                "extrapolation_type": "geological_future"
            }
            
        except Exception as e:
            logger.error(f"Failed to extrapolate geological future with model {model_id}", error=str(e))
            raise
    
    async def list_temporal_models(self) -> List[str]:
        """List all available temporal PINN models"""
        return self.pinn_engine.list_temporal_models()
    
    async def get_temporal_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get information about a temporal PINN model"""
        try:
            return self.pinn_engine.get_model_info(model_id)
        except Exception as e:
            logger.error(f"Failed to get temporal model info for {model_id}", error=str(e))
            raise

    async def process_integrated_geodynamic_data(
        self,
        seismic_events: List[Dict[str, Any]],
        deformation_measurements: List[Dict[str, Any]],
        geochemical_samples: List[Dict[str, Any]],
        station_locations: List[Dict[str, Any]],
        grid_bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
        resolution: float = 1.0
    ) -> Dict[str, Any]:
        """
        Process integrated geodynamic data using all analysis components
        
        Args:
            seismic_events: List of seismic event dictionaries
            deformation_measurements: List of deformation measurement dictionaries
            geochemical_samples: List of geochemical sample dictionaries
            station_locations: List of station location dictionaries
            grid_bounds: Spatial bounds for modeling
            resolution: Grid resolution in km
            
        Returns:
            Integrated modeling results
        """
        try:
            if not self.is_initialized:
                raise RuntimeError("Helios engine not initialized")
            
            if self.geodynamic_modeler is None:
                raise RuntimeError("Geodynamic modeler not initialized")
            
            logger.info("Processing integrated geodynamic data")
            
            # Convert dictionaries to data model objects
            from models.geological import (
                SeismicEvent, DeformationMeasurement, GeochemicalSample, GeologicalLocation
            )
            from datetime import datetime
            
            # Convert seismic events
            seismic_event_objects = []
            for event_dict in seismic_events:
                event = SeismicEvent.from_dict(event_dict)
                seismic_event_objects.append(event)
            
            # Convert deformation measurements
            deformation_objects = []
            for deform_dict in deformation_measurements:
                measurement = DeformationMeasurement.from_dict(deform_dict)
                deformation_objects.append(measurement)
            
            # Convert geochemical samples
            geochem_objects = []
            for sample_dict in geochemical_samples:
                sample = GeochemicalSample.from_dict(sample_dict)
                geochem_objects.append(sample)
            
            # Convert station locations
            station_objects = []
            for station_dict in station_locations:
                station = GeologicalLocation.from_dict(station_dict)
                station_objects.append(station)
            
            # Process integrated data
            results = self.geodynamic_modeler.process_integrated_data(
                seismic_event_objects,
                deformation_objects,
                geochem_objects,
                station_objects,
                grid_bounds,
                resolution
            )
            
            # Convert results to serializable format
            serializable_results = self._convert_results_to_serializable(results)
            
            logger.info("Integrated geodynamic data processing completed")
            return serializable_results
            
        except Exception as e:
            logger.error("Failed to process integrated geodynamic data", error=str(e))
            raise
    
    def _convert_results_to_serializable(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Convert numpy arrays and other non-serializable objects to serializable format"""
        
        serializable = {}
        
        for key, value in results.items():
            if hasattr(value, '__dict__'):
                # Convert dataclass objects
                if hasattr(value, 'to_dict'):
                    serializable[key] = value.to_dict()
                else:
                    # Convert dataclass to dict manually
                    obj_dict = {}
                    for attr_name, attr_value in value.__dict__.items():
                        if isinstance(attr_value, np.ndarray):
                            obj_dict[attr_name] = attr_value.tolist()
                        elif isinstance(attr_value, datetime):
                            obj_dict[attr_name] = attr_value.isoformat()
                        else:
                            obj_dict[attr_name] = attr_value
                    serializable[key] = obj_dict
            elif isinstance(value, np.ndarray):
                serializable[key] = value.tolist()
            elif isinstance(value, datetime):
                serializable[key] = value.isoformat()
            else:
                serializable[key] = value
        
        return serializable
    
    async def analyze_seismic_tomography(
        self,
        seismic_events: List[Dict[str, Any]],
        station_locations: List[Dict[str, Any]],
        grid_bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
        resolution: float = 1.0
    ) -> Dict[str, Any]:
        """
        Analyze seismic tomography data specifically
        
        Args:
            seismic_events: List of seismic event dictionaries
            station_locations: List of station location dictionaries
            grid_bounds: Spatial bounds for modeling
            resolution: Grid resolution in km
            
        Returns:
            Seismic tomography results
        """
        try:
            if not self.is_initialized or self.geodynamic_modeler is None:
                raise RuntimeError("Helios engine or geodynamic modeler not initialized")
            
            # Convert to data model objects
            from models.geological import SeismicEvent, GeologicalLocation
            
            seismic_event_objects = [SeismicEvent.from_dict(d) for d in seismic_events]
            station_objects = [GeologicalLocation.from_dict(d) for d in station_locations]
            
            # Process seismic tomography
            result = self.geodynamic_modeler.tomography_processor.process_seismic_data(
                seismic_event_objects, station_objects, grid_bounds, resolution
            )
            
            # Convert to serializable format
            return self._convert_results_to_serializable({"seismic_tomography": result})
            
        except Exception as e:
            logger.error("Failed to analyze seismic tomography", error=str(e))
            raise
    
    async def track_deformation(
        self,
        deformation_measurements: List[Dict[str, Any]],
        grid_bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
        resolution: float = 0.5
    ) -> Dict[str, Any]:
        """
        Track deformation with millimeter accuracy
        
        Args:
            deformation_measurements: List of deformation measurement dictionaries
            grid_bounds: Spatial bounds for modeling
            resolution: Grid resolution in km
            
        Returns:
            Deformation field results
        """
        try:
            if not self.is_initialized or self.geodynamic_modeler is None:
                raise RuntimeError("Helios engine or geodynamic modeler not initialized")
            
            # Convert to data model objects
            from models.geological import DeformationMeasurement
            
            measurement_objects = [DeformationMeasurement.from_dict(d) for d in deformation_measurements]
            
            # Track deformation
            result = self.geodynamic_modeler.deformation_tracker.track_deformation(
                measurement_objects, grid_bounds, resolution
            )
            
            # Convert to serializable format
            return self._convert_results_to_serializable({"deformation_field": result})
            
        except Exception as e:
            logger.error("Failed to track deformation", error=str(e))
            raise
    
    async def analyze_earthquake_catalog(
        self,
        seismic_events: List[Dict[str, Any]],
        time_window: Optional[Tuple[str, str]] = None,
        spatial_bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]] = None
    ) -> Dict[str, Any]:
        """
        Analyze earthquake catalog for stress and magma movement
        
        Args:
            seismic_events: List of seismic event dictionaries
            time_window: Time window for analysis (ISO format strings)
            spatial_bounds: Spatial bounds for analysis
            
        Returns:
            Earthquake analysis results
        """
        try:
            if not self.is_initialized or self.geodynamic_modeler is None:
                raise RuntimeError("Helios engine or geodynamic modeler not initialized")
            
            # Convert to data model objects
            from models.geological import SeismicEvent
            from datetime import datetime
            
            seismic_event_objects = [SeismicEvent.from_dict(d) for d in seismic_events]
            
            # Convert time window if provided
            time_window_dt = None
            if time_window:
                time_window_dt = (
                    datetime.fromisoformat(time_window[0]),
                    datetime.fromisoformat(time_window[1])
                )
            
            # Analyze earthquake catalog
            result = self.geodynamic_modeler.earthquake_analyzer.analyze_earthquake_catalog(
                seismic_event_objects, time_window_dt, spatial_bounds
            )
            
            # Convert to serializable format
            return self._convert_results_to_serializable({"earthquake_analysis": result})
            
        except Exception as e:
            logger.error("Failed to analyze earthquake catalog", error=str(e))
            raise
    
    async def analyze_geochemical_data(
        self,
        geochemical_samples: List[Dict[str, Any]],
        reference_compositions: Optional[Dict[str, Dict[str, float]]] = None
    ) -> Dict[str, Any]:
        """
        Analyze geochemical data for magma composition determination
        
        Args:
            geochemical_samples: List of geochemical sample dictionaries
            reference_compositions: Reference magma compositions for comparison
            
        Returns:
            Geochemical analysis results
        """
        try:
            if not self.is_initialized or self.geodynamic_modeler is None:
                raise RuntimeError("Helios engine or geodynamic modeler not initialized")
            
            # Convert to data model objects
            from models.geological import GeochemicalSample
            
            sample_objects = [GeochemicalSample.from_dict(d) for d in geochemical_samples]
            
            # Analyze geochemical data
            result = self.geodynamic_modeler.geochemical_analyzer.analyze_geochemical_data(
                sample_objects, reference_compositions
            )
            
            # Convert to serializable format
            return self._convert_results_to_serializable({"geochemical_analysis": result})
            
        except Exception as e:
            logger.error("Failed to analyze geochemical data", error=str(e))
            raise
    
    async def run_long_term_geological_simulation(
        self,
        simulation_config: Dict[str, Any],
        time_span: Tuple[str, str],
        time_scale: str = "millions_years"
    ) -> Dict[str, Any]:
        """
        Run long-term geological simulation
        
        Args:
            simulation_config: Configuration for simulation components
            time_span: Time span as ISO format strings [start, end]
            time_scale: Time scale for simulation
            
        Returns:
            Long-term geological simulation results
        """
        try:
            if not self.is_initialized or self.geological_simulator is None:
                raise RuntimeError("Helios engine or geological simulator not initialized")
            
            logger.info("Running long-term geological simulation")
            
            # Convert time span
            from datetime import datetime
            from .geological_simulation import SimulationTimeScale
            
            time_span_dt = (
                datetime.fromisoformat(time_span[0]),
                datetime.fromisoformat(time_span[1])
            )
            
            time_scale_enum = SimulationTimeScale(time_scale)
            
            # Run comprehensive simulation
            result = self.geological_simulator.run_comprehensive_simulation(
                simulation_config,
                time_span_dt,
                time_scale_enum
            )
            
            # Convert to serializable format
            serializable_result = self._convert_geological_simulation_result(result)
            
            logger.info("Long-term geological simulation completed")
            return serializable_result
            
        except Exception as e:
            logger.error("Failed to run long-term geological simulation", error=str(e))
            raise
    
    async def simulate_volcanic_evolution(
        self,
        initial_state_config: Dict[str, Any],
        time_span: Tuple[str, str],
        time_step_days: int = 365000,  # 1000 years
        external_forcing: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Simulate volcanic system evolution over geological timescales
        
        Args:
            initial_state_config: Initial volcanic system configuration
            time_span: Time span as ISO format strings
            time_step_days: Time step in days
            external_forcing: External forcing factors
            
        Returns:
            Volcanic evolution results
        """
        try:
            if not self.is_initialized or self.geological_simulator is None:
                raise RuntimeError("Helios engine or geological simulator not initialized")
            
            logger.info("Simulating volcanic system evolution")
            
            # Convert time span and time step
            from datetime import datetime, timedelta
            
            time_span_dt = (
                datetime.fromisoformat(time_span[0]),
                datetime.fromisoformat(time_span[1])
            )
            time_step = timedelta(days=time_step_days)
            
            # Create initial volcanic state
            initial_state = self.geological_simulator._create_initial_volcanic_state(
                initial_state_config
            )
            
            # Run volcanic evolution simulation
            evolution = self.geological_simulator.volcanic_modeler.simulate_volcanic_evolution(
                initial_state,
                time_span_dt,
                time_step,
                external_forcing
            )
            
            # Convert to serializable format
            serializable_evolution = []
            for state in evolution:
                state_dict = {
                    "timestamp": state.timestamp.isoformat(),
                    "stage": state.stage.value,
                    "eruption_rate": state.eruption_rate,
                    "thermal_state": state.thermal_state,
                    "volatile_budget": state.volatile_budget,
                    "crustal_thickness": state.crustal_thickness,
                    "system_age": state.system_age,
                    "num_reservoirs": len(state.magma_reservoirs)
                }
                serializable_evolution.append(state_dict)
            
            logger.info(f"Volcanic evolution simulation completed with {len(evolution)} time steps")
            return {"volcanic_evolution": serializable_evolution}
            
        except Exception as e:
            logger.error("Failed to simulate volcanic evolution", error=str(e))
            raise
    
    async def track_magma_generation(
        self,
        initial_conditions: Dict[str, Any],
        time_span: Tuple[str, str],
        spatial_domain: List[List[float]],
        time_step_days: int = 36500  # 100 years
    ) -> Dict[str, Any]:
        """
        Track magma generation from mantle to surface systems
        
        Args:
            initial_conditions: Initial mantle conditions
            time_span: Time span as ISO format strings
            spatial_domain: 3D spatial domain [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
            time_step_days: Time step in days
            
        Returns:
            Magma generation tracking results
        """
        try:
            if not self.is_initialized or self.geological_simulator is None:
                raise RuntimeError("Helios engine or geological simulator not initialized")
            
            logger.info("Tracking magma generation")
            
            # Convert parameters
            from datetime import datetime, timedelta
            
            time_span_dt = (
                datetime.fromisoformat(time_span[0]),
                datetime.fromisoformat(time_span[1])
            )
            time_step = timedelta(days=time_step_days)
            spatial_domain_tuple = tuple(tuple(bound) for bound in spatial_domain)
            
            # Run magma generation tracking
            generation_history = self.geological_simulator.magma_tracker.track_magma_generation(
                initial_conditions,
                time_span_dt,
                spatial_domain_tuple,
                time_step
            )
            
            # Convert to serializable format
            serializable_history = []
            for state in generation_history:
                state_dict = {
                    "timestamp": state.timestamp.isoformat(),
                    "mantle_temperature": state.mantle_temperature,
                    "mantle_composition": state.mantle_composition,
                    "melt_fraction": state.melt_fraction,
                    "generation_rate": state.generation_rate,
                    "ascent_velocity": state.ascent_velocity,
                    "crystallization_degree": state.crystallization_degree,
                    "volatile_content": state.volatile_content,
                    "depth_range": state.depth_range
                }
                serializable_history.append(state_dict)
            
            logger.info(f"Magma generation tracking completed with {len(generation_history)} time steps")
            return {"magma_generation_history": serializable_history}
            
        except Exception as e:
            logger.error("Failed to track magma generation", error=str(e))
            raise
    
    async def simulate_hydrothermal_circulation(
        self,
        initial_conditions: Dict[str, Any],
        heat_source: Dict[str, Any],
        time_span: Tuple[str, str],
        spatial_domain: List[List[float]],
        time_step_days: int = 3650  # 10 years
    ) -> Dict[str, Any]:
        """
        Simulate hydrothermal system circulation
        
        Args:
            initial_conditions: Initial fluid and rock conditions
            heat_source: Heat source parameters
            time_span: Time span as ISO format strings
            spatial_domain: 3D spatial domain
            time_step_days: Time step in days
            
        Returns:
            Hydrothermal circulation results
        """
        try:
            if not self.is_initialized or self.geological_simulator is None:
                raise RuntimeError("Helios engine or geological simulator not initialized")
            
            logger.info("Simulating hydrothermal circulation")
            
            # Convert parameters
            from datetime import datetime, timedelta
            
            time_span_dt = (
                datetime.fromisoformat(time_span[0]),
                datetime.fromisoformat(time_span[1])
            )
            time_step = timedelta(days=time_step_days)
            spatial_domain_tuple = tuple(tuple(bound) for bound in spatial_domain)
            
            # Run hydrothermal circulation simulation
            circulation_history = self.geological_simulator.hydrothermal_modeler.simulate_hydrothermal_circulation(
                initial_conditions,
                heat_source,
                time_span_dt,
                spatial_domain_tuple,
                time_step
            )
            
            # Convert to serializable format
            serializable_history = []
            for state in circulation_history:
                state_dict = {
                    "timestamp": state.timestamp.isoformat(),
                    "fluid_temperature": state.fluid_temperature,
                    "fluid_composition": state.fluid_composition,
                    "circulation_rate": state.circulation_rate,
                    "heat_flux": state.heat_flux,
                    "permeability": state.permeability,
                    "porosity": state.porosity,
                    "pressure": state.pressure,
                    "mineral_precipitation": state.mineral_precipitation
                }
                serializable_history.append(state_dict)
            
            logger.info(f"Hydrothermal circulation simulation completed with {len(circulation_history)} time steps")
            return {"hydrothermal_evolution": serializable_history}
            
        except Exception as e:
            logger.error("Failed to simulate hydrothermal circulation", error=str(e))
            raise
    
    async def reconstruct_tectonic_plates(
        self,
        initial_configuration: Dict[str, Any],
        time_span: Tuple[str, str],
        time_step_days: int = 365000000,  # 1 million years
        reference_frame: str = "hotspot"
    ) -> Dict[str, Any]:
        """
        Reconstruct tectonic plate movements through deep time
        
        Args:
            initial_configuration: Initial plate configuration
            time_span: Time span as ISO format strings
            time_step_days: Time step in days
            reference_frame: Reference frame for reconstruction
            
        Returns:
            Tectonic plate reconstruction results
        """
        try:
            if not self.is_initialized or self.geological_simulator is None:
                raise RuntimeError("Helios engine or geological simulator not initialized")
            
            logger.info("Reconstructing tectonic plate motion")
            
            # Convert parameters
            from datetime import datetime, timedelta
            
            time_span_dt = (
                datetime.fromisoformat(time_span[0]),
                datetime.fromisoformat(time_span[1])
            )
            time_step = timedelta(days=time_step_days)
            
            # Run tectonic reconstruction
            reconstruction = self.geological_simulator.tectonic_reconstructor.reconstruct_plate_motion(
                initial_configuration,
                time_span_dt,
                time_step,
                reference_frame
            )
            
            # Convert to serializable format
            serializable_reconstruction = []
            for state in reconstruction:
                state_dict = {
                    "timestamp": state.timestamp.isoformat(),
                    "plate_positions": {k: v.tolist() for k, v in state.plate_positions.items()},
                    "plate_velocities": {k: v.tolist() for k, v in state.plate_velocities.items()},
                    "boundary_types": state.boundary_types,
                    "subduction_angles": state.subduction_angles,
                    "spreading_rates": state.spreading_rates,
                    "continental_positions": {k: v.tolist() for k, v in state.continental_positions.items()}
                }
                serializable_reconstruction.append(state_dict)
            
            logger.info(f"Tectonic reconstruction completed with {len(reconstruction)} time steps")
            return {"tectonic_reconstruction": serializable_reconstruction}
            
        except Exception as e:
            logger.error("Failed to reconstruct tectonic plates", error=str(e))
            raise
    
    def _convert_geological_simulation_result(self, result) -> Dict[str, Any]:
        """Convert geological simulation result to serializable format"""
        
        from datetime import datetime
        
        serializable = {
            "simulation_id": result.simulation_id,
            "time_range": [result.time_range[0].isoformat(), result.time_range[1].isoformat()],
            "time_scale": result.time_scale.value,
            "simulation_metadata": result.simulation_metadata,
            "processing_timestamp": result.processing_timestamp.isoformat()
        }
        
        # Convert volcanic evolution
        if result.volcanic_evolution:
            serializable["volcanic_evolution"] = []
            for state in result.volcanic_evolution:
                state_dict = {
                    "timestamp": state.timestamp.isoformat(),
                    "stage": state.stage.value,
                    "eruption_rate": state.eruption_rate,
                    "thermal_state": state.thermal_state,
                    "volatile_budget": state.volatile_budget,
                    "crustal_thickness": state.crustal_thickness,
                    "system_age": state.system_age,
                    "num_reservoirs": len(state.magma_reservoirs)
                }
                serializable["volcanic_evolution"].append(state_dict)
        
        # Convert magma generation history
        if result.magma_generation_history:
            serializable["magma_generation_history"] = []
            for state in result.magma_generation_history:
                state_dict = {
                    "timestamp": state.timestamp.isoformat(),
                    "mantle_temperature": state.mantle_temperature,
                    "melt_fraction": state.melt_fraction,
                    "generation_rate": state.generation_rate,
                    "ascent_velocity": state.ascent_velocity,
                    "crystallization_degree": state.crystallization_degree,
                    "volatile_content": state.volatile_content,
                    "depth_range": state.depth_range
                }
                serializable["magma_generation_history"].append(state_dict)
        
        # Convert hydrothermal evolution
        if result.hydrothermal_evolution:
            serializable["hydrothermal_evolution"] = []
            for state in result.hydrothermal_evolution:
                state_dict = {
                    "timestamp": state.timestamp.isoformat(),
                    "fluid_temperature": state.fluid_temperature,
                    "circulation_rate": state.circulation_rate,
                    "heat_flux": state.heat_flux,
                    "permeability": state.permeability,
                    "mineral_precipitation": state.mineral_precipitation
                }
                serializable["hydrothermal_evolution"].append(state_dict)
        
        # Convert tectonic reconstruction
        if result.tectonic_reconstruction:
            serializable["tectonic_reconstruction"] = []
            for state in result.tectonic_reconstruction:
                state_dict = {
                    "timestamp": state.timestamp.isoformat(),
                    "plate_positions": {k: v.tolist() for k, v in state.plate_positions.items()},
                    "plate_velocities": {k: v.tolist() for k, v in state.plate_velocities.items()},
                    "boundary_types": state.boundary_types,
                    "spreading_rates": state.spreading_rates
                }
                serializable["tectonic_reconstruction"].append(state_dict)
        
        return serializable
    
    async def run_mantle_dynamics_simulation(
        self,
        simulation_config: Dict[str, Any],
        time_span: Tuple[str, str]
    ) -> Dict[str, Any]:
        """
        Run comprehensive deep Earth and mantle dynamics simulation
        
        Args:
            simulation_config: Configuration for mantle dynamics components
            time_span: Time span as ISO format strings
            
        Returns:
            Mantle dynamics simulation results
        """
        try:
            if not self.is_initialized or self.mantle_dynamics_simulator is None:
                raise RuntimeError("Helios engine or mantle dynamics simulator not initialized")
            
            logger.info("Running mantle dynamics simulation")
            
            # Convert time span
            from datetime import datetime
            
            time_span_dt = (
                datetime.fromisoformat(time_span[0]),
                datetime.fromisoformat(time_span[1])
            )
            
            # Run comprehensive mantle simulation
            result = self.mantle_dynamics_simulator.run_comprehensive_mantle_simulation(
                simulation_config, time_span_dt
            )
            
            # Convert to serializable format
            serializable_result = self._convert_mantle_dynamics_result(result)
            
            logger.info("Mantle dynamics simulation completed")
            return serializable_result
            
        except Exception as e:
            logger.error("Failed to run mantle dynamics simulation", error=str(e))
            raise
    
    async def simulate_mantle_plume_evolution(
        self,
        initiation_location: List[float],
        initial_conditions: Dict[str, Any],
        time_span: Tuple[str, str],
        time_step_days: int = 3650000  # 10,000 years
    ) -> Dict[str, Any]:
        """
        Simulate mantle plume evolution from initiation to surface expression
        
        Args:
            initiation_location: Plume initiation location [x, y, z] in meters
            initial_conditions: Initial plume conditions
            time_span: Time span as ISO format strings
            time_step_days: Time step in days
            
        Returns:
            Mantle plume evolution results
        """
        try:
            if not self.is_initialized or self.mantle_dynamics_simulator is None:
                raise RuntimeError("Helios engine or mantle dynamics simulator not initialized")
            
            logger.info("Simulating mantle plume evolution")
            
            # Convert parameters
            from datetime import datetime, timedelta
            import numpy as np
            
            time_span_dt = (
                datetime.fromisoformat(time_span[0]),
                datetime.fromisoformat(time_span[1])
            )
            time_step = timedelta(days=time_step_days)
            initiation_array = np.array(initiation_location)
            
            # Run plume evolution simulation
            plume_evolution = self.mantle_dynamics_simulator.plume_modeler.simulate_plume_evolution(
                initiation_array, initial_conditions, time_span_dt, time_step
            )
            
            # Convert to serializable format
            serializable_evolution = []
            for state in plume_evolution:
                state_dict = {
                    "timestamp": state.timestamp.isoformat(),
                    "plume_id": state.plume_id,
                    "stage": state.stage.value,
                    "head_location": state.head_location.tolist(),
                    "tail_location": state.tail_location.tolist(),
                    "head_radius": state.head_radius,
                    "tail_radius": state.tail_radius,
                    "temperature_anomaly": state.temperature_anomaly,
                    "ascent_velocity": state.ascent_velocity,
                    "buoyancy_flux": state.buoyancy_flux,
                    "volume_flux": state.volume_flux,
                    "composition": state.composition,
                    "surface_expression": (
                        state.surface_expression.to_dict() 
                        if state.surface_expression else None
                    )
                }
                serializable_evolution.append(state_dict)
            
            logger.info(f"Mantle plume evolution simulation completed with {len(plume_evolution)} time steps")
            return {"plume_evolution": serializable_evolution}
            
        except Exception as e:
            logger.error("Failed to simulate mantle plume evolution", error=str(e))
            raise
    
    async def simulate_core_mantle_boundary(
        self,
        initial_conditions: Dict[str, Any],
        time_span: Tuple[str, str],
        time_step_days: int = 365000000  # 1 million years
    ) -> Dict[str, Any]:
        """
        Simulate core-mantle boundary thermal and chemical interactions
        
        Args:
            initial_conditions: Initial CMB conditions
            time_span: Time span as ISO format strings
            time_step_days: Time step in days
            
        Returns:
            CMB dynamics results
        """
        try:
            if not self.is_initialized or self.mantle_dynamics_simulator is None:
                raise RuntimeError("Helios engine or mantle dynamics simulator not initialized")
            
            logger.info("Simulating core-mantle boundary dynamics")
            
            # Convert parameters
            from datetime import datetime, timedelta
            
            time_span_dt = (
                datetime.fromisoformat(time_span[0]),
                datetime.fromisoformat(time_span[1])
            )
            time_step = timedelta(days=time_step_days)
            
            # Run CMB dynamics simulation
            cmb_evolution = self.mantle_dynamics_simulator.cmb_modeler.simulate_cmb_dynamics(
                initial_conditions, time_span_dt, time_step
            )
            
            # Convert to serializable format
            serializable_evolution = []
            for state in cmb_evolution:
                state_dict = {
                    "timestamp": state.timestamp.isoformat(),
                    "temperature_cmb": state.temperature_cmb,
                    "heat_flux_cmb": state.heat_flux_cmb,
                    "chemical_composition": state.chemical_composition,
                    "d_prime_thickness": state.d_prime_thickness,
                    "seismic_velocity_anomaly": state.seismic_velocity_anomaly,
                    "thermal_boundary_layer_thickness": state.thermal_boundary_layer_thickness,
                    "ultra_low_velocity_zones": state.ultra_low_velocity_zones,
                    "post_perovskite_stability": state.post_perovskite_stability
                }
                serializable_evolution.append(state_dict)
            
            logger.info(f"CMB dynamics simulation completed with {len(cmb_evolution)} time steps")
            return {"cmb_evolution": serializable_evolution}
            
        except Exception as e:
            logger.error("Failed to simulate core-mantle boundary", error=str(e))
            raise
    
    async def simulate_hotspot_volcanism(
        self,
        hotspot_location: Dict[str, float],
        hotspot_type: str,
        plume_properties: Dict[str, Any],
        plate_motion: Dict[str, Any],
        time_span: Tuple[str, str],
        time_step_days: int = 3650000  # 10,000 years
    ) -> Dict[str, Any]:
        """
        Simulate hotspot volcanism and track formation
        
        Args:
            hotspot_location: Hotspot location dictionary
            hotspot_type: Type of hotspot ("deep_plume", "shallow_plume", etc.)
            plume_properties: Plume properties
            plate_motion: Plate motion parameters
            time_span: Time span as ISO format strings
            time_step_days: Time step in days
            
        Returns:
            Hotspot volcanism results
        """
        try:
            if not self.is_initialized or self.mantle_dynamics_simulator is None:
                raise RuntimeError("Helios engine or mantle dynamics simulator not initialized")
            
            logger.info("Simulating hotspot volcanism")
            
            # Convert parameters
            from datetime import datetime, timedelta
            from models.geological import GeologicalLocation
            from .mantle_dynamics import HotspotType
            
            time_span_dt = (
                datetime.fromisoformat(time_span[0]),
                datetime.fromisoformat(time_span[1])
            )
            time_step = timedelta(days=time_step_days)
            
            location = GeologicalLocation.from_dict(hotspot_location)
            hotspot_type_enum = HotspotType(hotspot_type)
            
            # Run hotspot volcanism simulation
            hotspot_evolution = self.mantle_dynamics_simulator.hotspot_modeler.simulate_hotspot_volcanism(
                location, hotspot_type_enum, plume_properties, plate_motion, time_span_dt, time_step
            )
            
            # Convert to serializable format
            serializable_evolution = []
            for state in hotspot_evolution:
                state_dict = {
                    "timestamp": state.timestamp.isoformat(),
                    "hotspot_id": state.hotspot_id,
                    "hotspot_type": state.hotspot_type.value,
                    "location": state.location.to_dict(),
                    "volcanic_flux": state.volcanic_flux,
                    "temperature": state.temperature,
                    "composition": state.composition,
                    "age_progression": state.age_progression,
                    "track_direction": state.track_direction.tolist(),
                    "plume_source_depth": state.plume_source_depth,
                    "crustal_thickness_anomaly": state.crustal_thickness_anomaly
                }
                serializable_evolution.append(state_dict)
            
            logger.info(f"Hotspot volcanism simulation completed with {len(hotspot_evolution)} time steps")
            return {"hotspot_evolution": serializable_evolution}
            
        except Exception as e:
            logger.error("Failed to simulate hotspot volcanism", error=str(e))
            raise
    
    def _convert_mantle_dynamics_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Convert mantle dynamics result to serializable format"""
        
        serializable = {
            "simulation_id": result["simulation_id"],
            "time_range": result["time_range"],
            "components_simulated": result["components_simulated"],
            "processing_timestamp": result["processing_timestamp"]
        }
        
        results = result["results"]
        serializable_results = {}
        
        # Convert CMB evolution
        if "cmb_evolution" in results:
            cmb_evolution = []
            for state in results["cmb_evolution"]:
                state_dict = {
                    "timestamp": state.timestamp.isoformat(),
                    "temperature_cmb": state.temperature_cmb,
                    "heat_flux_cmb": state.heat_flux_cmb,
                    "chemical_composition": state.chemical_composition,
                    "d_prime_thickness": state.d_prime_thickness,
                    "seismic_velocity_anomaly": state.seismic_velocity_anomaly,
                    "post_perovskite_stability": state.post_perovskite_stability
                }
                cmb_evolution.append(state_dict)
            serializable_results["cmb_evolution"] = cmb_evolution
        
        # Convert plume evolution
        if "plume_evolution" in results:
            plume_results = []
            for plume_result in results["plume_evolution"]:
                plume_evolution = []
                for state in plume_result["evolution"]:
                    state_dict = {
                        "timestamp": state.timestamp.isoformat(),
                        "plume_id": state.plume_id,
                        "stage": state.stage.value,
                        "head_location": state.head_location.tolist(),
                        "temperature_anomaly": state.temperature_anomaly,
                        "ascent_velocity": state.ascent_velocity,
                        "volume_flux": state.volume_flux,
                        "surface_expression": (
                            state.surface_expression.to_dict() 
                            if state.surface_expression else None
                        )
                    }
                    plume_evolution.append(state_dict)
                
                plume_results.append({
                    "plume_id": plume_result["plume_id"],
                    "evolution": plume_evolution
                })
            serializable_results["plume_evolution"] = plume_results
        
        # Convert convection evolution
        if "convection_evolution" in results:
            convection_evolution = []
            for state in results["convection_evolution"]:
                state_dict = {
                    "timestamp": state.timestamp.isoformat(),
                    "convection_regime": state.convection_regime.value,
                    "rayleigh_number": state.rayleigh_number,
                    "nusselt_number": state.nusselt_number,
                    "heat_transport_efficiency": state.heat_transport_efficiency,
                    "num_upwelling_locations": len(state.upwelling_locations),
                    "num_downwelling_locations": len(state.downwelling_locations),
                    "slab_penetration_depth": state.slab_penetration_depth
                }
                convection_evolution.append(state_dict)
            serializable_results["convection_evolution"] = convection_evolution
        
        # Convert hotspot evolution
        if "hotspot_evolution" in results:
            hotspot_results = []
            for hotspot_result in results["hotspot_evolution"]:
                hotspot_evolution = []
                for state in hotspot_result["evolution"]:
                    state_dict = {
                        "timestamp": state.timestamp.isoformat(),
                        "hotspot_id": state.hotspot_id,
                        "hotspot_type": state.hotspot_type.value,
                        "location": state.location.to_dict(),
                        "volcanic_flux": state.volcanic_flux,
                        "temperature": state.temperature,
                        "composition": state.composition,
                        "crustal_thickness_anomaly": state.crustal_thickness_anomaly
                    }
                    hotspot_evolution.append(state_dict)
                
                hotspot_results.append({
                    "hotspot_id": hotspot_result["hotspot_id"],
                    "evolution": hotspot_evolution
                })
            serializable_results["hotspot_evolution"] = hotspot_results
        
        # Convert LIP evolution
        if "lip_evolution" in results:
            lip_results = []
            for lip_result in results["lip_evolution"]:
                lip_evolution = []
                for state in lip_result["evolution"]:
                    state_dict = {
                        "timestamp": state.timestamp.isoformat(),
                        "lip_id": state.lip_id,
                        "formation_stage": state.formation_stage,
                        "total_volume": state.total_volume,
                        "eruption_rate": state.eruption_rate,
                        "areal_extent": state.areal_extent,
                        "duration": state.duration,
                        "environmental_impact": state.environmental_impact
                    }
                    lip_evolution.append(state_dict)
                
                lip_results.append({
                    "lip_id": lip_result["lip_id"],
                    "evolution": lip_evolution
                })
            serializable_results["lip_evolution"] = lip_results
        
        serializable["results"] = serializable_results
        return serializable

    async def process_insar_scene(
        self,
        scene_data: Dict[str, Any],
        apply_atmospheric_correction: bool = True,
        apply_noise_reduction: bool = True
    ) -> Dict[str, Any]:
        """
        Process InSAR scene data for deformation analysis.
        
        Args:
            scene_data: InSAR scene data dictionary
            apply_atmospheric_correction: Whether to apply atmospheric correction
            apply_noise_reduction: Whether to apply noise reduction
            
        Returns:
            Processing results including deformation patterns and measurements
        """
        try:
            logger.info("Processing InSAR scene", scene_id=scene_data.get('scene_id'))
            
            # Import here to avoid circular imports
            from .insar_deformation_analyzer import InSARScene
            
            # Convert dictionary to InSAR scene object
            scene = InSARScene(
                scene_id=scene_data['scene_id'],
                timestamp=datetime.fromisoformat(scene_data['timestamp']),
                satellite=scene_data['satellite'],
                orbit_direction=scene_data['orbit_direction'],
                look_angle=scene_data['look_angle'],
                heading_angle=scene_data['heading_angle'],
                wavelength=scene_data['wavelength'],
                spatial_resolution=scene_data['spatial_resolution'],
                coverage_area=tuple(scene_data['coverage_area'])
            )
            
            # Add data arrays if provided
            if 'unwrapped_phase' in scene_data:
                scene.unwrapped_phase = np.array(scene_data['unwrapped_phase'])
            if 'coherence' in scene_data:
                scene.coherence = np.array(scene_data['coherence'])
            if 'interferogram' in scene_data:
                scene.interferogram = np.array(scene_data['interferogram'])
            
            # Add metadata
            scene.metadata = scene_data.get('metadata', {})
            
            # Process the scene
            results = self.insar_analyzer.process_insar_scene(
                scene,
                apply_atmospheric_correction=apply_atmospheric_correction,
                apply_noise_reduction=apply_noise_reduction
            )
            
            # Convert results to serializable format
            serializable_results = self._convert_insar_results(results)
            
            logger.info("InSAR scene processing completed", 
                       scene_id=scene_data.get('scene_id'),
                       status=results['status'])
            
            return serializable_results
            
        except Exception as e:
            logger.error("Failed to process InSAR scene", error=str(e))
            raise

    async def analyze_insar_deformation_patterns(
        self,
        scenes_data: List[Dict[str, Any]],
        temporal_analysis: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze deformation patterns across multiple InSAR scenes.
        
        Args:
            scenes_data: List of InSAR scene data dictionaries
            temporal_analysis: Whether to perform temporal analysis
            
        Returns:
            Comprehensive deformation analysis results
        """
        try:
            logger.info("Analyzing InSAR deformation patterns", 
                       num_scenes=len(scenes_data))
            
            # Process each scene
            scene_results = []
            for scene_data in scenes_data:
                result = await self.process_insar_scene(scene_data)
                scene_results.append(result)
            
            # Perform temporal analysis if requested
            temporal_results = None
            if temporal_analysis and len(scene_results) > 1:
                temporal_results = self._analyze_temporal_deformation(scene_results)
            
            # Aggregate results
            analysis_results = {
                'analysis_id': f"insar_analysis_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                'num_scenes_processed': len(scene_results),
                'scene_results': scene_results,
                'temporal_analysis': temporal_results,
                'processing_timestamp': datetime.utcnow().isoformat()
            }
            
            logger.info("InSAR deformation pattern analysis completed",
                       num_scenes=len(scenes_data))
            
            return analysis_results
            
        except Exception as e:
            logger.error("Failed to analyze InSAR deformation patterns", error=str(e))
            raise

    async def train_insar_models(
        self,
        training_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Train InSAR analysis models (atmospheric correction and deformation pattern recognition).
        
        Args:
            training_config: Training configuration including data paths and parameters
            
        Returns:
            Training results and model performance metrics
        """
        try:
            logger.info("Starting InSAR model training")
            
            # Extract training parameters
            epochs = training_config.get('epochs', 100)
            batch_size = training_config.get('batch_size', 16)
            learning_rate = training_config.get('learning_rate', 1e-4)
            
            # Load training data (this would typically load from files)
            training_scenes = self._load_insar_training_data(
                training_config.get('training_data_path')
            )
            validation_scenes = self._load_insar_training_data(
                training_config.get('validation_data_path')
            )
            
            training_results = {}
            
            # Train atmospheric correction model if requested
            if training_config.get('train_atmospheric_model', True):
                logger.info("Training atmospheric correction model")
                atm_history = self.insar_analyzer.train_atmospheric_model(
                    training_scenes, validation_scenes,
                    epochs=epochs, batch_size=batch_size, learning_rate=learning_rate
                )
                training_results['atmospheric_model'] = atm_history
            
            # Train deformation pattern model if requested
            if training_config.get('train_deformation_model', True):
                logger.info("Training deformation pattern model")
                
                # Prepare training data with ground truth patterns
                deformation_training_data = self._prepare_deformation_training_data(
                    training_scenes, training_config.get('ground_truth_patterns', [])
                )
                deformation_validation_data = self._prepare_deformation_training_data(
                    validation_scenes, training_config.get('validation_patterns', [])
                )
                
                def_history = self.insar_analyzer.train_deformation_model(
                    deformation_training_data, deformation_validation_data,
                    epochs=epochs, batch_size=batch_size, learning_rate=learning_rate
                )
                training_results['deformation_model'] = def_history
            
            # Save models if requested
            if training_config.get('save_models', True):
                model_save_path = training_config.get('model_save_path', self.model_cache_dir)
                self.insar_analyzer.save_models(
                    atmospheric_model_path=f"{model_save_path}/atmospheric_model.pth",
                    deformation_model_path=f"{model_save_path}/deformation_model.pth"
                )
            
            results = {
                'training_id': f"insar_training_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                'training_results': training_results,
                'training_config': training_config,
                'completion_timestamp': datetime.utcnow().isoformat()
            }
            
            logger.info("InSAR model training completed")
            return results
            
        except Exception as e:
            logger.error("Failed to train InSAR models", error=str(e))
            raise

    async def load_insar_models(
        self,
        atmospheric_model_path: Optional[str] = None,
        deformation_model_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load pre-trained InSAR analysis models.
        
        Args:
            atmospheric_model_path: Path to atmospheric correction model
            deformation_model_path: Path to deformation pattern model
            
        Returns:
            Model loading status
        """
        try:
            logger.info("Loading InSAR models")
            
            # Use default paths if not provided
            if atmospheric_model_path is None:
                atmospheric_model_path = f"{self.model_cache_dir}/atmospheric_model.pth"
            if deformation_model_path is None:
                deformation_model_path = f"{self.model_cache_dir}/deformation_model.pth"
            
            # Load models
            self.insar_analyzer.load_models(
                atmospheric_model_path=atmospheric_model_path if os.path.exists(atmospheric_model_path) else None,
                deformation_model_path=deformation_model_path if os.path.exists(deformation_model_path) else None
            )
            
            results = {
                'atmospheric_model_loaded': os.path.exists(atmospheric_model_path),
                'deformation_model_loaded': os.path.exists(deformation_model_path),
                'loading_timestamp': datetime.utcnow().isoformat()
            }
            
            logger.info("InSAR models loaded", **results)
            return results
            
        except Exception as e:
            logger.error("Failed to load InSAR models", error=str(e))
            raise

    def _convert_insar_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Convert InSAR processing results to serializable format"""
        serializable = {
            'scene_id': results['scene_id'],
            'timestamp': results['timestamp'].isoformat() if isinstance(results['timestamp'], datetime) else results['timestamp'],
            'processing_timestamp': results['processing_timestamp'].isoformat() if isinstance(results['processing_timestamp'], datetime) else results['processing_timestamp'],
            'status': results['status']
        }
        
        # Add error message if present
        if 'error_message' in results:
            serializable['error_message'] = results['error_message']
        
        # Convert atmospheric correction results
        if results.get('atmospheric_correction'):
            atm_correction = results['atmospheric_correction']
            serializable['atmospheric_correction'] = {
                'correction_method': atm_correction.correction_method,
                'correction_quality': atm_correction.correction_quality,
                'turbulence_strength': atm_correction.turbulence_strength
            }
        
        # Convert deformation patterns
        if results.get('deformation_patterns'):
            patterns = []
            for pattern in results['deformation_patterns']:
                pattern_dict = {
                    'pattern_id': pattern.pattern_id,
                    'pattern_type': pattern.pattern_type,
                    'center_location': pattern.center_location.to_dict(),
                    'spatial_extent': pattern.spatial_extent,
                    'magnitude': pattern.magnitude,
                    'confidence': pattern.confidence
                }
                if pattern.temporal_evolution:
                    pattern_dict['temporal_evolution'] = [
                        (ts.isoformat(), val) for ts, val in pattern.temporal_evolution
                    ]
                if pattern.uncertainty:
                    pattern_dict['uncertainty'] = pattern.uncertainty
                if pattern.associated_features:
                    pattern_dict['associated_features'] = pattern.associated_features
                
                patterns.append(pattern_dict)
            
            serializable['deformation_patterns'] = patterns
        
        # Convert displacement measurements
        if results.get('displacement_measurements'):
            measurements = []
            for measurement in results['displacement_measurements']:
                measurement_dict = {
                    'timestamp': measurement.timestamp.isoformat(),
                    'location': measurement.location.to_dict(),
                    'displacement_vector': measurement.displacement_vector.tolist(),
                    'measurement_type': measurement.measurement_type,
                    'uncertainty': measurement.uncertainty.tolist(),
                    'reference_frame': measurement.reference_frame
                }
                measurements.append(measurement_dict)
            
            serializable['displacement_measurements'] = measurements
        
        # Add uncertainty estimates
        if results.get('uncertainty_estimates'):
            serializable['uncertainty_estimates'] = results['uncertainty_estimates']
        
        # Add processing flags
        serializable['noise_reduction_applied'] = results.get('noise_reduction_applied', False)
        
        return serializable

    def _analyze_temporal_deformation(self, scene_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze temporal evolution of deformation across multiple scenes"""
        try:
            # Extract timestamps and measurements
            timestamps = []
            max_displacements = []
            pattern_counts = []
            
            for result in scene_results:
                if result['status'] == 'success':
                    timestamps.append(datetime.fromisoformat(result['timestamp']))
                    
                    # Find maximum displacement
                    measurements = result.get('displacement_measurements', [])
                    if measurements:
                        displacements = [
                            np.linalg.norm(np.array(m['displacement_vector'])) 
                            for m in measurements
                        ]
                        max_displacements.append(max(displacements))
                    else:
                        max_displacements.append(0.0)
                    
                    pattern_counts.append(len(result.get('deformation_patterns', [])))
            
            if len(timestamps) < 2:
                return {'error': 'Insufficient data for temporal analysis'}
            
            # Calculate deformation rate
            time_span_days = (timestamps[-1] - timestamps[0]).days
            if time_span_days > 0:
                deformation_rate = (max_displacements[-1] - max_displacements[0]) / time_span_days * 365.25
            else:
                deformation_rate = 0.0
            
            # Calculate statistics
            temporal_analysis = {
                'time_span_days': time_span_days,
                'num_acquisitions': len(timestamps),
                'max_displacement': max(max_displacements) if max_displacements else 0.0,
                'min_displacement': min(max_displacements) if max_displacements else 0.0,
                'mean_displacement': np.mean(max_displacements) if max_displacements else 0.0,
                'deformation_rate_m_per_year': deformation_rate,
                'average_patterns_per_scene': np.mean(pattern_counts) if pattern_counts else 0.0,
                'displacement_trend': 'increasing' if deformation_rate > 0.001 else 'decreasing' if deformation_rate < -0.001 else 'stable'
            }
            
            return temporal_analysis
            
        except Exception as e:
            logger.error("Failed to analyze temporal deformation", error=str(e))
            return {'error': str(e)}

    def _load_insar_training_data(self, data_path: Optional[str]) -> List:
        """Load InSAR training data from file path"""
        # This is a placeholder - in a real implementation, this would load
        # actual InSAR scenes from files (e.g., HDF5, NetCDF, GeoTIFF)
        if not data_path or not os.path.exists(data_path):
            logger.warning("Training data path not found, using synthetic data")
            return []
        
        # For now, return empty list - would implement actual data loading
        return []

    def _prepare_deformation_training_data(self, scenes: List, ground_truth_patterns: List) -> List:
        """Prepare training data for deformation pattern recognition"""
        # This would pair scenes with their ground truth deformation patterns
        # For now, return empty list - would implement actual data preparation
        return []

    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up Helios engine")
        
        # Clear models from memory
        for model_name in list(self.models.keys()):
            del self.models[model_name]
        
        # Clear CUDA cache if using GPU
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        
        self.is_initialized = False
        logger.info("Helios engine cleanup complete")   
 
    # Earthquake Forecasting Methods
    
    async def analyze_earthquake_patterns(
        self,
        events: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze earthquake patterns using machine learning models.
        
        Args:
            events: List of seismic event dictionaries
            
        Returns:
            Pattern analysis results
        """
        try:
            from ..models.geological import SeismicEvent, GeologicalLocation, EventType
            
            # Convert dictionaries to SeismicEvent objects
            seismic_events = []
            for event_data in events:
                location = GeologicalLocation(
                    latitude=event_data["latitude"],
                    longitude=event_data["longitude"],
                    elevation=event_data.get("elevation", 0.0),
                    depth=event_data.get("depth")
                )
                
                event = SeismicEvent(
                    timestamp=datetime.fromisoformat(event_data["timestamp"]),
                    location=location,
                    magnitude=event_data["magnitude"],
                    depth=event_data["depth"],
                    event_type=EventType(event_data.get("event_type", "earthquake")),
                    confidence=event_data.get("confidence", 1.0),
                    agency=event_data.get("agency"),
                    event_id=event_data.get("event_id")
                )
                seismic_events.append(event)
            
            # Analyze patterns using earthquake forecasting engine
            results = self.earthquake_forecasting.analyze_earthquake_patterns(seismic_events)
            
            logger.info(f"Analyzed earthquake patterns for {len(seismic_events)} events")
            
            return {
                "status": "success",
                "num_events": len(seismic_events),
                "analysis_results": results,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to analyze earthquake patterns", error=str(e))
            raise
    
    async def create_statistical_seismicity_model(
        self,
        events: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Create statistical seismicity models with confidence intervals.
        
        Args:
            events: List of seismic event dictionaries
            
        Returns:
            Statistical model results
        """
        try:
            from ..models.geological import SeismicEvent, GeologicalLocation, EventType
            
            # Convert dictionaries to SeismicEvent objects
            seismic_events = []
            for event_data in events:
                location = GeologicalLocation(
                    latitude=event_data["latitude"],
                    longitude=event_data["longitude"],
                    elevation=event_data.get("elevation", 0.0),
                    depth=event_data.get("depth")
                )
                
                event = SeismicEvent(
                    timestamp=datetime.fromisoformat(event_data["timestamp"]),
                    location=location,
                    magnitude=event_data["magnitude"],
                    depth=event_data["depth"],
                    event_type=EventType(event_data.get("event_type", "earthquake")),
                    confidence=event_data.get("confidence", 1.0),
                    agency=event_data.get("agency"),
                    event_id=event_data.get("event_id")
                )
                seismic_events.append(event)
            
            # Create statistical model using earthquake forecasting engine
            results = self.earthquake_forecasting.create_statistical_seismicity_model(seismic_events)
            
            logger.info(f"Created statistical seismicity model from {len(seismic_events)} events")
            
            return {
                "status": "success",
                "num_events": len(seismic_events),
                "model_results": results,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to create statistical seismicity model", error=str(e))
            raise
    
    async def perform_temporal_clustering_analysis(
        self,
        events: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Perform temporal clustering analysis for earthquake sequences.
        
        Args:
            events: List of seismic event dictionaries
            
        Returns:
            Clustering analysis results
        """
        try:
            from ..models.geological import SeismicEvent, GeologicalLocation, EventType
            
            # Convert dictionaries to SeismicEvent objects
            seismic_events = []
            for event_data in events:
                location = GeologicalLocation(
                    latitude=event_data["latitude"],
                    longitude=event_data["longitude"],
                    elevation=event_data.get("elevation", 0.0),
                    depth=event_data.get("depth")
                )
                
                event = SeismicEvent(
                    timestamp=datetime.fromisoformat(event_data["timestamp"]),
                    location=location,
                    magnitude=event_data["magnitude"],
                    depth=event_data["depth"],
                    event_type=EventType(event_data.get("event_type", "earthquake")),
                    confidence=event_data.get("confidence", 1.0),
                    agency=event_data.get("agency"),
                    event_id=event_data.get("event_id")
                )
                seismic_events.append(event)
            
            # Perform clustering analysis using earthquake forecasting engine
            clusters = self.earthquake_forecasting.perform_temporal_clustering_analysis(seismic_events)
            
            # Convert clusters to serializable format
            cluster_results = []
            for cluster in clusters:
                cluster_data = {
                    "cluster_id": cluster.cluster_id,
                    "cluster_type": cluster.cluster_type,
                    "num_events": len(cluster.events),
                    "spatial_extent_km": cluster.spatial_extent,
                    "temporal_extent_days": cluster.temporal_extent,
                    "magnitude_range": cluster.magnitude_range,
                    "centroid_location": {
                        "latitude": cluster.centroid_location.latitude,
                        "longitude": cluster.centroid_location.longitude,
                        "elevation": cluster.centroid_location.elevation
                    },
                    "centroid_time": cluster.centroid_time.isoformat(),
                    "events": [
                        {
                            "event_id": event.event_id,
                            "timestamp": event.timestamp.isoformat(),
                            "magnitude": event.magnitude,
                            "depth": event.depth,
                            "latitude": event.location.latitude,
                            "longitude": event.location.longitude
                        }
                        for event in cluster.events
                    ]
                }
                cluster_results.append(cluster_data)
            
            logger.info(f"Performed temporal clustering analysis, found {len(clusters)} clusters")
            
            return {
                "status": "success",
                "num_events": len(seismic_events),
                "num_clusters": len(clusters),
                "clusters": cluster_results,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to perform temporal clustering analysis", error=str(e))
            raise
    
    async def generate_probabilistic_hazard_assessment(
        self,
        events: List[Dict[str, Any]],
        target_region: Dict[str, float],
        forecast_period_days: int = 30
    ) -> Dict[str, Any]:
        """
        Generate probabilistic hazard assessment with uncertainty quantification.
        
        Args:
            events: List of seismic event dictionaries
            target_region: Geographic region for assessment
            forecast_period_days: Forecast period in days
            
        Returns:
            Probabilistic hazard assessment results
        """
        try:
            from ..models.geological import SeismicEvent, GeologicalLocation, EventType
            
            # Convert dictionaries to SeismicEvent objects
            seismic_events = []
            for event_data in events:
                location = GeologicalLocation(
                    latitude=event_data["latitude"],
                    longitude=event_data["longitude"],
                    elevation=event_data.get("elevation", 0.0),
                    depth=event_data.get("depth")
                )
                
                event = SeismicEvent(
                    timestamp=datetime.fromisoformat(event_data["timestamp"]),
                    location=location,
                    magnitude=event_data["magnitude"],
                    depth=event_data["depth"],
                    event_type=EventType(event_data.get("event_type", "earthquake")),
                    confidence=event_data.get("confidence", 1.0),
                    agency=event_data.get("agency"),
                    event_id=event_data.get("event_id")
                )
                seismic_events.append(event)
            
            # Generate probabilistic hazard assessment
            forecast_result = self.earthquake_forecasting.generate_probabilistic_hazard_assessment(
                seismic_events, target_region, forecast_period_days
            )
            
            # Convert forecast result to serializable format
            result_data = {
                "forecast_time": forecast_result.forecast_time.isoformat(),
                "forecast_period": {
                    "start": forecast_result.forecast_period[0].isoformat(),
                    "end": forecast_result.forecast_period[1].isoformat()
                },
                "target_region": forecast_result.target_region,
                "probability_estimates": forecast_result.probability_estimates,
                "confidence_intervals": {
                    mag_range: {"lower": ci[0], "upper": ci[1]}
                    for mag_range, ci in forecast_result.confidence_intervals.items()
                },
                "hazard_level": forecast_result.hazard_level.value,
                "model_used": forecast_result.model_used.value,
                "uncertainty_metrics": forecast_result.uncertainty_metrics,
                "contributing_factors": forecast_result.contributing_factors
            }
            
            logger.info(f"Generated probabilistic hazard assessment for {len(seismic_events)} events, "
                       f"hazard level: {forecast_result.hazard_level.value}")
            
            return {
                "status": "success",
                "num_events": len(seismic_events),
                "forecast_period_days": forecast_period_days,
                "forecast_result": result_data,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to generate probabilistic hazard assessment", error=str(e))
            raise
    
    async def get_earthquake_forecasting_config(self) -> Dict[str, Any]:
        """
        Get earthquake forecasting engine configuration.
        
        Returns:
            Configuration parameters
        """
        try:
            return {
                "status": "success",
                "config": self.earthquake_forecasting.config,
                "available_models": [
                    "random_forest",
                    "gradient_boosting", 
                    "statistical_seismicity",
                    "temporal_clustering",
                    "etas"
                ],
                "supported_forecast_periods": [1, 7, 30, 365],  # days
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to get earthquake forecasting config", error=str(e))
            raise
    
    async def update_earthquake_forecasting_config(
        self,
        config_updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update earthquake forecasting engine configuration.
        
        Args:
            config_updates: Configuration updates to apply
            
        Returns:
            Updated configuration
        """
        try:
            # Update configuration
            for key, value in config_updates.items():
                if key in self.earthquake_forecasting.config:
                    self.earthquake_forecasting.config[key].update(value)
            
            logger.info("Updated earthquake forecasting configuration", updates=config_updates)
            
            return {
                "status": "success",
                "updated_config": self.earthquake_forecasting.config,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to update earthquake forecasting config", error=str(e))
            raise    

    # Transfer Learning Methods
    
    async def adapt_model_to_domain(
        self,
        source_model_id: str,
        target_domain_id: str,
        adaptation_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Adapt a source model to a target geological domain
        
        Args:
            source_model_id: Identifier of the source model
            target_domain_id: Identifier of the target domain
            adaptation_config: Configuration for adaptation process
            
        Returns:
            Adaptation results
        """
        try:
            # Get target domain
            target_domain = self.domain_registry.get_domain(target_domain_id)
            if not target_domain:
                raise ValueError(f"Target domain {target_domain_id} not found")
            
            # Get source model
            source_model = self.pinn_engine.get_model(source_model_id)
            if source_model is None:
                raise ValueError(f"Source model {source_model_id} not found")
            
            # Create adaptation configuration
            config = AdaptationConfig(
                source_domain=None,  # Would be determined from model metadata
                target_domain=target_domain,
                adaptation_method=adaptation_config.get("adaptation_method"),
                learning_rate=adaptation_config.get("learning_rate", 1e-4),
                num_epochs=adaptation_config.get("num_epochs", 100),
                batch_size=adaptation_config.get("batch_size", 32),
                regularization_strength=adaptation_config.get("regularization_strength", 0.01),
                few_shot_samples=adaptation_config.get("few_shot_samples", 10),
                validation_split=adaptation_config.get("validation_split", 0.2)
            )
            
            # Generate dummy target data (in practice, would come from data ingestion)
            target_data = self._generate_domain_data(target_domain, config.batch_size * 20)
            
            # Perform adaptation
            result = self.transfer_learning_engine.adapt_to_domain(
                source_model, target_domain, target_data, config
            )
            
            # Store adapted model
            adapted_model_id = f"{source_model_id}_adapted_{target_domain_id}"
            self.pinn_engine.register_model(adapted_model_id, result.adapted_model)
            
            logger.info(f"Model adaptation completed: {source_model_id} -> {adapted_model_id}")
            
            return {
                "source_model_id": source_model_id,
                "adapted_model_id": adapted_model_id,
                "target_domain": target_domain_id,
                "adaptation_method": config.adaptation_method,
                "adaptation_metrics": result.adaptation_metrics,
                "validation_scores": result.validation_scores,
                "confidence_score": result.confidence_score,
                "status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Failed to adapt model {source_model_id} to domain {target_domain_id}", error=str(e))
            raise
    
    async def few_shot_adapt_model(
        self,
        source_model_id: str,
        target_domain_id: str,
        few_shot_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Adapt a model using few-shot learning for data-sparse regions
        
        Args:
            source_model_id: Identifier of the source model
            target_domain_id: Identifier of the target domain
            few_shot_config: Configuration for few-shot learning
            
        Returns:
            Few-shot adaptation results
        """
        try:
            # Get target domain
            target_domain = self.domain_registry.get_domain(target_domain_id)
            if not target_domain:
                raise ValueError(f"Target domain {target_domain_id} not found")
            
            # Get source model
            source_model = self.pinn_engine.get_model(source_model_id)
            if source_model is None:
                raise ValueError(f"Source model {source_model_id} not found")
            
            # Create few-shot configuration
            config = AdaptationConfig(
                source_domain=None,
                target_domain=target_domain,
                adaptation_method="few_shot",
                learning_rate=few_shot_config.get("learning_rate", 1e-3),
                num_epochs=few_shot_config.get("num_epochs", 50),
                batch_size=few_shot_config.get("batch_size", 16),
                few_shot_samples=few_shot_config.get("few_shot_samples", 10),
                validation_split=few_shot_config.get("validation_split", 0.2)
            )
            
            # Generate small dataset for few-shot learning
            few_shot_data = self._generate_domain_data(target_domain, config.few_shot_samples * 5)
            
            # Perform few-shot adaptation
            result = self.transfer_learning_engine.few_shot_adapt(
                source_model, target_domain, few_shot_data, config
            )
            
            # Store adapted model
            adapted_model_id = f"{source_model_id}_fewshot_{target_domain_id}"
            self.pinn_engine.register_model(adapted_model_id, result.adapted_model)
            
            logger.info(f"Few-shot adaptation completed: {source_model_id} -> {adapted_model_id}")
            
            return {
                "source_model_id": source_model_id,
                "adapted_model_id": adapted_model_id,
                "target_domain": target_domain_id,
                "few_shot_samples": config.few_shot_samples,
                "adaptation_metrics": result.adaptation_metrics,
                "validation_scores": result.validation_scores,
                "confidence_score": result.confidence_score,
                "status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Failed to perform few-shot adaptation for model {source_model_id}", error=str(e))
            raise
    
    async def get_adaptation_strategy_recommendation(
        self,
        target_domain_id: str
    ) -> Dict[str, Any]:
        """
        Get adaptation strategy recommendations for a target domain
        
        Args:
            target_domain_id: Identifier of the target domain
            
        Returns:
            Strategy recommendations
        """
        try:
            # Get target domain
            target_domain = self.domain_registry.get_domain(target_domain_id)
            if not target_domain:
                raise ValueError(f"Target domain {target_domain_id} not found")
            
            # Get strategy recommendations
            recommendations = self.strategy_selector.get_strategy_recommendations(target_domain)
            
            # Get general adaptation recommendations
            general_recommendations = self.transfer_learning_engine.get_adaptation_recommendations(target_domain)
            
            return {
                "target_domain": target_domain_id,
                "region_name": target_domain.region_name,
                "data_sparsity_level": target_domain.data_sparsity_level,
                "geological_setting": target_domain.geological_setting,
                "strategy_recommendations": recommendations,
                "general_recommendations": general_recommendations,
                "recommended_strategy": self.strategy_selector.select_strategy(target_domain)
            }
            
        except Exception as e:
            logger.error(f"Failed to get adaptation recommendations for domain {target_domain_id}", error=str(e))
            raise
    
    async def register_geological_domain(
        self,
        domain_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Register a new geological domain
        
        Args:
            domain_data: Domain information dictionary
            
        Returns:
            Registration result
        """
        try:
            from .domain_adaptation_utils import GeologicalSetting, TectonicContext, DataCharacteristics
            
            # Validate and create domain
            geological_setting = GeologicalSetting(domain_data["geological_setting"])
            tectonic_context = TectonicContext(domain_data["tectonic_context"])
            
            data_characteristics = DataCharacteristics(**domain_data["data_characteristics"])
            
            # Register domain
            self.domain_registry.register_domain(
                domain_id=domain_data["domain_id"],
                region_name=domain_data["region_name"],
                geological_setting=geological_setting,
                tectonic_context=tectonic_context,
                data_characteristics=data_characteristics,
                available_data_types=domain_data["available_data_types"],
                data_sparsity_level=domain_data["data_sparsity_level"]
            )
            
            logger.info(f"Registered geological domain: {domain_data['domain_id']}")
            
            return {
                "domain_id": domain_data["domain_id"],
                "region_name": domain_data["region_name"],
                "status": "registered",
                "total_domains": len(self.domain_registry.domains)
            }
            
        except Exception as e:
            logger.error(f"Failed to register geological domain", error=str(e))
            raise
    
    async def get_similar_domains(
        self,
        target_domain_id: str,
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Find domains similar to the target domain
        
        Args:
            target_domain_id: Target domain identifier
            top_k: Number of similar domains to return
            
        Returns:
            Similar domains information
        """
        try:
            similar_domains = self.domain_registry.get_similar_domains(target_domain_id, top_k)
            
            result = []
            for domain_id, similarity_score in similar_domains:
                domain = self.domain_registry.get_domain(domain_id)
                if domain:
                    result.append({
                        "domain_id": domain_id,
                        "region_name": domain.region_name,
                        "similarity_score": similarity_score,
                        "geological_setting": domain.geological_setting,
                        "tectonic_context": domain.tectonic_context,
                        "data_sparsity_level": domain.data_sparsity_level
                    })
            
            return {
                "target_domain": target_domain_id,
                "similar_domains": result,
                "total_found": len(result)
            }
            
        except Exception as e:
            logger.error(f"Failed to find similar domains for {target_domain_id}", error=str(e))
            raise
    
    async def validate_adapted_model(
        self,
        adapted_model_id: str,
        validation_domain_id: str
    ) -> Dict[str, Any]:
        """
        Validate an adapted model's performance
        
        Args:
            adapted_model_id: Identifier of the adapted model
            validation_domain_id: Domain to validate against
            
        Returns:
            Validation results
        """
        try:
            # Get adapted model
            adapted_model = self.pinn_engine.get_model(adapted_model_id)
            if adapted_model is None:
                raise ValueError(f"Adapted model {adapted_model_id} not found")
            
            # Get validation domain
            validation_domain = self.domain_registry.get_domain(validation_domain_id)
            if not validation_domain:
                raise ValueError(f"Validation domain {validation_domain_id} not found")
            
            # Get original model (assume it's the base model without adaptation suffix)
            original_model_id = adapted_model_id.split('_adapted_')[0].split('_fewshot_')[0]
            original_model = self.pinn_engine.get_model(original_model_id)
            
            # Generate validation data
            validation_data = self._generate_domain_data(validation_domain, 200)
            
            # Perform validation
            validation_scores = self.transfer_learning_engine.validator.validate_adapted_model(
                adapted_model, validation_data, original_model, validation_domain
            )
            
            logger.info(f"Model validation completed for {adapted_model_id}")
            
            return {
                "adapted_model_id": adapted_model_id,
                "validation_domain": validation_domain_id,
                "validation_scores": validation_scores,
                "validation_summary": {
                    "overall_performance": "good" if validation_scores.get('mse_improvement', 0) > 0 else "needs_improvement",
                    "physics_consistency": validation_scores.get('physics_consistency', 0),
                    "geological_plausibility": validation_scores.get('geological_plausibility', 0),
                    "improvement_over_original": validation_scores.get('mse_improvement', 0)
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to validate adapted model {adapted_model_id}", error=str(e))
            raise
    
    async def get_adaptation_history(self) -> Dict[str, Any]:
        """
        Get history of all adaptation processes
        
        Returns:
            Adaptation history
        """
        try:
            history_data = []
            for entry in self.transfer_learning_engine.adaptation_history:
                history_data.append({
                    "target_domain": entry['target_domain'],
                    "strategy": entry['strategy'],
                    "validation_scores": entry['result'].validation_scores,
                    "adaptation_metrics": entry['result'].adaptation_metrics,
                    "confidence_score": entry['result'].confidence_score
                })
            
            return {
                "adaptation_history": history_data,
                "total_adaptations": len(history_data),
                "successful_adaptations": sum(1 for entry in history_data if entry['confidence_score'] > 0.5)
            }
            
        except Exception as e:
            logger.error("Failed to get adaptation history", error=str(e))
            raise
    
    async def list_geological_domains(self) -> List[Dict[str, Any]]:
        """
        List all registered geological domains
        
        Returns:
            List of domain information
        """
        try:
            domains = []
            for domain_id in self.domain_registry.list_domains():
                domain = self.domain_registry.get_domain(domain_id)
                if domain:
                    domains.append({
                        "domain_id": domain.domain_id,
                        "region_name": domain.region_name,
                        "geological_setting": domain.geological_setting,
                        "tectonic_context": domain.tectonic_context,
                        "data_sparsity_level": domain.data_sparsity_level,
                        "available_data_types": domain.available_data_types
                    })
            
            return domains
            
        except Exception as e:
            logger.error("Failed to list geological domains", error=str(e))
            raise
    
    def _generate_domain_data(self, domain: GeologicalDomain, num_samples: int):
        """
        Generate synthetic data for a geological domain (helper method)
        
        Args:
            domain: Geological domain
            num_samples: Number of samples to generate
            
        Returns:
            DataLoader with synthetic data
        """
        from torch.utils.data import DataLoader, TensorDataset
        
        # Generate synthetic data based on domain characteristics
        num_features = len(domain.available_data_types) * 5  # 5 features per data type
        
        # Add noise based on domain data quality
        noise_level = 1.0 - domain.data_characteristics.get('data_quality', 0.8)
        
        data = torch.randn(num_samples, num_features) + noise_level * torch.randn(num_samples, num_features)
        targets = torch.randn(num_samples, 1)  # Simplified target
        
        dataset = TensorDataset(data, targets)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        return dataloader
    
    def get_pinn_model(self):
        """Get the main PINN model for transfer learning"""
        return self.models.get("pinn_geological")
    
    async def get_transfer_learning_status(self) -> Dict[str, Any]:
        """
        Get status of transfer learning components
        
        Returns:
            Transfer learning status
        """
        try:
            return {
                "transfer_learning_engine": self.transfer_learning_engine is not None,
                "domain_registry": self.domain_registry is not None,
                "strategy_selector": self.strategy_selector is not None,
                "registered_domains": len(self.domain_registry.domains) if self.domain_registry else 0,
                "adaptation_history": len(self.transfer_learning_engine.adaptation_history) if self.transfer_learning_engine else 0,
                "available_strategies": list(self.transfer_learning_engine.adaptation_strategies.keys()) if self.transfer_learning_engine else []
            }
            
        except Exception as e:
            logger.error("Failed to get transfer learning status", error=str(e))
            raise
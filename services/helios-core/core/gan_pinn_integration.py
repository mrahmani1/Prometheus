"""
GAN-PINN Integration for Probabilistic Geological Outcomes

This module implements the coupling mechanisms between Generative Adversarial Networks
and Physics-Informed Neural Networks to provide probabilistic outcomes for geological
predictions with uncertainty quantification.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import structlog
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gan_engine import GANEngine, GeologicalGAN
from .pinn_engine import PINNEngine

logger = structlog.get_logger()

@dataclass
class GANPINNConfig:
    """Configuration for GAN-PINN integration"""

    integration_id: str
    gan_model_id: str
    pinn_model_id: str
    coupling_method: str = (
        "ensemble_sampling"  # ensemble_sampling, adversarial_training, hybrid
    )
    num_ensemble_samples: int = 100
    confidence_level: float = 0.95
    physics_constraint_weight: float = 1.0
    adversarial_weight: float = 0.1

    def __post_init__(self):
        if not self.integration_id:
            raise ValueError("Integration ID cannot be empty")
        if not self.gan_model_id:
            raise ValueError("GAN model ID cannot be empty")
        if not self.pinn_model_id:
            raise ValueError("PINN model ID cannot be empty")
        if not 0 < self.confidence_level < 1:
            raise ValueError("Confidence level must be between 0 and 1")

class ProbabilisticPINNPredictor:
    """
    Probabilistic PINN predictor using GAN-generated scenarios
    """

    def __init__(
        self, pinn_engine: PINNEngine, gan_engine: GANEngine, device: torch.device
    ):
        self.pinn_engine = pinn_engine
        self.gan_engine = gan_engine
        self.device = device

        logger.info("Probabilistic PINN Predictor initialized")

    def predict_with_uncertainty(
        self,
        pinn_model_id: str,
        gan_ensemble_id: str,
        inputs: torch.Tensor,
        num_scenarios: int = 100,
        physics_constraints: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Make probabilistic predictions using PINN with GAN-generated scenarios

        Args:
            pinn_model_id: PINN model identifier
            gan_ensemble_id: GAN ensemble identifier
            inputs: Input data for predictions
            num_scenarios: Number of scenarios to generate
            physics_constraints: Optional physics constraints to enforce

        Returns:
            Probabilistic prediction results with uncertainty
        """
        try:
            # Get GAN ensemble
            gan_ensemble = self.gan_engine.get_ensemble(gan_ensemble_id)

            # Generate geological scenarios
            scenario_list = gan_ensemble.generate_ensemble_scenarios(num_scenarios)

            # Make PINN predictions for each scenario
            predictions = []

            for scenarios in scenario_list:
                # For each GAN model's scenarios
                scenario_predictions = []

                for scenario in scenarios:
                    # Combine input with scenario parameters
                    combined_input = self._combine_input_with_scenario(inputs, scenario)

                    # Make PINN prediction
                    pred = self.pinn_engine.predict(pinn_model_id, combined_input)

                    # Apply physics constraints if specified
                    if physics_constraints:
                        pred = self._apply_physics_constraints(
                            pred, physics_constraints
                        )

                    scenario_predictions.append(pred)

                # Stack predictions for this GAN model
                model_predictions = torch.stack(scenario_predictions, dim=0)
                predictions.append(model_predictions)

            # Combine predictions from all GAN models
            all_predictions = torch.cat(
                predictions, dim=0
            )  # [total_scenarios, batch_size, output_dim]

            # Calculate statistics
            mean_prediction = torch.mean(all_predictions, dim=0)
            std_prediction = torch.std(all_predictions, dim=0)

            # Calculate confidence intervals
            confidence_intervals = self._calculate_confidence_intervals(
                all_predictions, confidence_level=0.95
            )

            # Separate epistemic and aleatoric uncertainty
            epistemic_uncertainty, aleatoric_uncertainty = self._decompose_uncertainty(
                all_predictions, scenario_list
            )

            return {
                "mean_prediction": mean_prediction.cpu().numpy().tolist(),
                "std_prediction": std_prediction.cpu().numpy().tolist(),
                "confidence_intervals": confidence_intervals,
                "epistemic_uncertainty": epistemic_uncertainty.tolist(),
                "aleatoric_uncertainty": aleatoric_uncertainty.tolist(),
                "total_uncertainty": (
                    epistemic_uncertainty + aleatoric_uncertainty
                ).tolist(),
                "num_scenarios": num_scenarios,
                "num_gan_models": len(scenario_list),
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error("Failed to make probabilistic PINN prediction", error=str(e))
            raise

    def _combine_input_with_scenario(
        self, inputs: torch.Tensor, scenario: torch.Tensor
    ) -> torch.Tensor:
        """
        Combine input data with generated scenario parameters

        Args:
            inputs: Original input data [batch_size, input_dim]
            scenario: Generated scenario parameters [scenario_dim]

        Returns:
            Combined input tensor [batch_size, input_dim + scenario_dim]
        """
        batch_size = inputs.size(0)

        # Expand scenario to match batch size
        expanded_scenario = scenario.unsqueeze(0).expand(batch_size, -1)

        # Concatenate inputs with scenario
        combined = torch.cat([inputs, expanded_scenario], dim=1)

        return combined

    def _apply_physics_constraints(
        self, predictions: torch.Tensor, constraints: List[str]
    ) -> torch.Tensor:
        """Apply physics constraints to predictions"""
        constrained_predictions = predictions.clone()

        for constraint in constraints:
            if constraint == "temperature_positive":
                # Ensure temperature is positive (above absolute zero)
                constrained_predictions[:, 0] = torch.clamp(
                    constrained_predictions[:, 0], min=273.15
                )
            elif constraint == "pressure_positive":
                # Ensure pressure is positive
                constrained_predictions[:, 1] = torch.clamp(
                    constrained_predictions[:, 1], min=0.0
                )
            elif constraint == "mass_conservation":
                # Apply mass conservation constraint (simplified)
                # In practice, this would be more complex
                total_mass = torch.sum(
                    constrained_predictions[:, 2:5], dim=1, keepdim=True
                )
                constrained_predictions[:, 2:5] = constrained_predictions[:, 2:5] / (
                    total_mass + 1e-8
                )

        return constrained_predictions

    def _calculate_confidence_intervals(
        self, predictions: torch.Tensor, confidence_level: float = 0.95
    ) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for predictions"""
        alpha = 1 - confidence_level
        lower_quantile = alpha / 2
        upper_quantile = 1 - alpha / 2

        confidence_intervals = {}

        for i in range(predictions.size(2)):  # For each output dimension
            lower = (
                torch.quantile(predictions[:, :, i], lower_quantile, dim=0)
                .mean()
                .item()
            )
            upper = (
                torch.quantile(predictions[:, :, i], upper_quantile, dim=0)
                .mean()
                .item()
            )
            confidence_intervals[f"output_{i}"] = (lower, upper)

        return confidence_intervals

    def _decompose_uncertainty(
        self, all_predictions: torch.Tensor, scenario_list: List[torch.Tensor]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decompose uncertainty into epistemic and aleatoric components

        Args:
            all_predictions: All predictions [total_scenarios, batch_size, output_dim]
            scenario_list: List of scenarios from each GAN model

        Returns:
            Tuple of (epistemic_uncertainty, aleatoric_uncertainty)
        """
        # Epistemic uncertainty: variance across different GAN models
        num_models = len(scenario_list)
        scenarios_per_model = scenario_list[0].size(0)

        # Reshape predictions by model
        model_predictions = all_predictions.view(
            num_models,
            scenarios_per_model,
            all_predictions.size(1),
            all_predictions.size(2),
        )

        # Calculate mean prediction for each model
        model_means = torch.mean(
            model_predictions, dim=1
        )  # [num_models, batch_size, output_dim]

        # Epistemic uncertainty: variance across model means
        epistemic_uncertainty = torch.var(model_means, dim=0).cpu().numpy()

        # Aleatoric uncertainty: average within-model variance
        model_variances = torch.var(
            model_predictions, dim=1
        )  # [num_models, batch_size, output_dim]
        aleatoric_uncertainty = torch.mean(model_variances, dim=0).cpu().numpy()

        return epistemic_uncertainty, aleatoric_uncertainty

class AdversarialPINNTrainer:
    """
    Adversarial training for PINN models using GAN feedback
    """

    def __init__(
        self, pinn_engine: PINNEngine, gan_engine: GANEngine, device: torch.device
    ):
        self.pinn_engine = pinn_engine
        self.gan_engine = gan_engine
        self.device = device

        logger.info("Adversarial PINN Trainer initialized")

    def train_pinn_with_gan_feedback(
        self,
        pinn_model_id: str,
        gan_model_id: str,
        training_data: torch.utils.data.DataLoader,
        num_epochs: int = 100,
        adversarial_weight: float = 0.1,
    ) -> Dict[str, List[float]]:
        """
        Train PINN model with GAN feedback for improved uncertainty quantification

        Args:
            pinn_model_id: PINN model identifier
            gan_model_id: GAN model identifier
            training_data: Training data loader
            num_epochs: Number of training epochs
            adversarial_weight: Weight for adversarial loss component

        Returns:
            Training history
        """
        try:
            # Get models
            pinn_trainer = self.pinn_engine.get_trainer(pinn_model_id)
            gan = self.gan_engine.get_gan(gan_model_id)

            if pinn_trainer is None:
                raise ValueError(f"PINN model {pinn_model_id} not found")

            # Training history
            history = {
                "pinn_loss": [],
                "physics_loss": [],
                "adversarial_loss": [],
                "total_loss": [],
                "epoch": [],
            }

            # Optimizer for PINN
            optimizer = torch.optim.Adam(pinn_trainer.model.parameters(), lr=0.001)

            logger.info(f"Starting adversarial PINN training for {num_epochs} epochs")

            for epoch in range(num_epochs):
                epoch_losses = {
                    "pinn_loss": [],
                    "physics_loss": [],
                    "adversarial_loss": [],
                    "total_loss": [],
                }

                for batch_idx, batch_data in enumerate(training_data):
                    if len(batch_data) == 2:
                        inputs, targets = batch_data
                        inputs, targets = inputs.to(self.device), targets.to(
                            self.device
                        )
                    else:
                        inputs = batch_data[0].to(self.device)
                        targets = None

                    optimizer.zero_grad()

                    # PINN forward pass
                    pinn_predictions = pinn_trainer.model(inputs)

                    # Standard PINN loss (data fitting + physics)
                    if targets is not None:
                        data_loss = nn.MSELoss()(pinn_predictions, targets)
                    else:
                        data_loss = torch.tensor(0.0, device=self.device)

                    # Physics loss
                    physics_loss = pinn_trainer._compute_physics_loss(
                        inputs, pinn_predictions
                    )

                    # Adversarial loss: encourage predictions that GAN discriminator finds plausible
                    adversarial_loss = self._compute_adversarial_loss(
                        pinn_predictions, gan
                    )

                    # Total loss
                    total_loss = (
                        data_loss + physics_loss + adversarial_weight * adversarial_loss
                    )

                    total_loss.backward()
                    optimizer.step()

                    # Record losses
                    epoch_losses["pinn_loss"].append(data_loss.item())
                    epoch_losses["physics_loss"].append(physics_loss.item())
                    epoch_losses["adversarial_loss"].append(adversarial_loss.item())
                    epoch_losses["total_loss"].append(total_loss.item())

                # Average losses for epoch
                for key in epoch_losses:
                    history[key].append(np.mean(epoch_losses[key]))
                history["epoch"].append(epoch)

                # Log progress
                if epoch % 10 == 0:
                    logger.info(
                        f"Epoch {epoch}/{num_epochs}",
                        pinn_loss=f"{history['pinn_loss'][-1]:.4f}",
                        physics_loss=f"{history['physics_loss'][-1]:.4f}",
                        adversarial_loss=f"{history['adversarial_loss'][-1]:.4f}",
                        total_loss=f"{history['total_loss'][-1]:.4f}",
                    )

            logger.info("Adversarial PINN training completed")
            return history

        except Exception as e:
            logger.error("Failed to train PINN with GAN feedback", error=str(e))
            raise

    def _compute_adversarial_loss(
        self, pinn_predictions: torch.Tensor, gan: GeologicalGAN
    ) -> torch.Tensor:
        """
        Compute adversarial loss encouraging PINN predictions to be plausible

        Args:
            pinn_predictions: PINN model predictions
            gan: GAN model for plausibility assessment

        Returns:
            Adversarial loss tensor
        """
        # Assess plausibility of PINN predictions using GAN discriminator
        plausibility_scores = gan.discriminator(pinn_predictions)

        # Adversarial loss: encourage high plausibility scores
        # We want the discriminator to think PINN predictions are "real"
        target_scores = torch.ones_like(plausibility_scores)
        adversarial_loss = nn.BCELoss()(plausibility_scores, target_scores)

        return adversarial_loss

class EnsemblePredictionSystem:
    """
    Ensemble prediction system combining multiple GAN outputs with PINN models
    """

    def __init__(
        self, pinn_engine: PINNEngine, gan_engine: GANEngine, device: torch.device
    ):
        self.pinn_engine = pinn_engine
        self.gan_engine = gan_engine
        self.device = device

        # Registry of ensemble configurations
        self.ensemble_configs: Dict[str, GANPINNConfig] = {}

        logger.info("Ensemble Prediction System initialized")

    def create_ensemble_predictor(self, config: GANPINNConfig) -> str:
        """
        Create an ensemble predictor configuration

        Args:
            config: GAN-PINN integration configuration

        Returns:
            Integration ID
        """
        if config.integration_id in self.ensemble_configs:
            raise ValueError(
                f"Ensemble predictor {config.integration_id} already exists"
            )

        # Validate that models exist
        if (
            config.gan_model_id not in self.gan_engine.gans
            and config.gan_model_id not in self.gan_engine.ensembles
        ):
            raise ValueError(f"GAN model {config.gan_model_id} not found")

        if config.pinn_model_id not in self.pinn_engine.models:
            raise ValueError(f"PINN model {config.pinn_model_id} not found")

        self.ensemble_configs[config.integration_id] = config

        logger.info(f"Created ensemble predictor: {config.integration_id}")
        return config.integration_id

    def predict_ensemble(
        self,
        integration_id: str,
        inputs: torch.Tensor,
        geological_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Make ensemble predictions combining GAN scenarios with PINN physics

        Args:
            integration_id: Ensemble predictor identifier
            inputs: Input data for predictions
            geological_context: Optional geological context information

        Returns:
            Ensemble prediction results with uncertainty quantification
        """
        try:
            config = self.ensemble_configs[integration_id]

            # Get models
            if config.gan_model_id in self.gan_engine.ensembles:
                gan_ensemble = self.gan_engine.get_ensemble(config.gan_model_id)
                use_ensemble = True
            else:
                gan_model = self.gan_engine.get_gan(config.gan_model_id)
                use_ensemble = False

            # Generate scenarios
            if use_ensemble:
                scenario_list = gan_ensemble.generate_ensemble_scenarios(
                    config.num_ensemble_samples
                )
            else:
                scenarios = gan_model.generate_scenarios(config.num_ensemble_samples)
                scenario_list = [scenarios]

            # Make PINN predictions for each scenario
            ensemble_predictions = []
            plausibility_scores = []

            for scenarios in scenario_list:
                model_predictions = []
                model_plausibility = []

                for scenario in scenarios:
                    # Combine input with scenario
                    combined_input = self._combine_inputs(
                        inputs, scenario, geological_context
                    )

                    # PINN prediction
                    pinn_pred = self.pinn_engine.predict(
                        config.pinn_model_id, combined_input
                    )
                    model_predictions.append(pinn_pred)

                    # Assess plausibility
                    if use_ensemble:
                        # Use first GAN model for plausibility (could be improved)
                        plausibility = gan_ensemble.gans[0].assess_plausibility(
                            pinn_pred
                        )
                    else:
                        plausibility = gan_model.assess_plausibility(pinn_pred)

                    model_plausibility.append(plausibility)

                ensemble_predictions.append(torch.stack(model_predictions))
                plausibility_scores.append(torch.stack(model_plausibility))

            # Combine all predictions
            all_predictions = torch.cat(ensemble_predictions, dim=0)
            all_plausibility = torch.cat(plausibility_scores, dim=0)

            # Calculate ensemble statistics
            mean_prediction = torch.mean(all_predictions, dim=0)
            std_prediction = torch.std(all_predictions, dim=0)
            mean_plausibility = torch.mean(all_plausibility, dim=0)

            # Calculate confidence intervals
            confidence_intervals = {}
            for i in range(mean_prediction.size(1)):
                lower = (
                    torch.quantile(all_predictions[:, :, i], 0.025, dim=0).mean().item()
                )
                upper = (
                    torch.quantile(all_predictions[:, :, i], 0.975, dim=0).mean().item()
                )
                confidence_intervals[f"output_{i}"] = (lower, upper)

            # Uncertainty decomposition
            epistemic_uncertainty = self._calculate_epistemic_uncertainty(
                ensemble_predictions
            )
            aleatoric_uncertainty = self._calculate_aleatoric_uncertainty(
                ensemble_predictions
            )

            return {
                "integration_id": integration_id,
                "mean_prediction": mean_prediction.cpu().numpy().tolist(),
                "std_prediction": std_prediction.cpu().numpy().tolist(),
                "confidence_intervals": confidence_intervals,
                "mean_plausibility": mean_plausibility.cpu().numpy().tolist(),
                "epistemic_uncertainty": epistemic_uncertainty.tolist(),
                "aleatoric_uncertainty": aleatoric_uncertainty.tolist(),
                "total_uncertainty": (
                    epistemic_uncertainty + aleatoric_uncertainty
                ).tolist(),
                "num_scenarios": config.num_ensemble_samples,
                "num_models": len(scenario_list),
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(
                f"Failed to make ensemble prediction for {integration_id}", error=str(e)
            )
            raise

    def _combine_inputs(
        self,
        inputs: torch.Tensor,
        scenario: torch.Tensor,
        geological_context: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Combine inputs with scenario and geological context"""
        batch_size = inputs.size(0)

        # Expand scenario to match batch size
        expanded_scenario = scenario.unsqueeze(0).expand(batch_size, -1)

        # Start with inputs and scenario
        combined = torch.cat([inputs, expanded_scenario], dim=1)

        # Add geological context if provided
        if geological_context:
            context_features = self._extract_context_features(
                geological_context, batch_size
            )
            combined = torch.cat([combined, context_features], dim=1)

        return combined

    def _extract_context_features(
        self, geological_context: Dict[str, Any], batch_size: int
    ) -> torch.Tensor:
        """Extract numerical features from geological context"""
        features = []

        # Location features
        if "location" in geological_context:
            location = geological_context["location"]
            lat = location.get("latitude", 0.0) / 90.0  # Normalize to [-1, 1]
            lon = location.get("longitude", 0.0) / 180.0  # Normalize to [-1, 1]
            depth = min(location.get("depth", 0.0) / 100000.0, 1.0)  # Normalize depth
            features.extend([lat, lon, depth])

        # Temporal features
        if "time" in geological_context:
            # Convert time to numerical features (simplified)
            time_val = geological_context["time"]
            if isinstance(time_val, (int, float)):
                features.append(time_val)
            else:
                features.append(0.0)

        # Geological setting features
        if "setting" in geological_context:
            setting = geological_context["setting"]
            # One-hot encode geological settings (simplified)
            settings = ["volcanic", "tectonic", "sedimentary", "metamorphic"]
            setting_vector = [1.0 if setting == s else 0.0 for s in settings]
            features.extend(setting_vector)

        # Convert to tensor and expand for batch
        if features:
            feature_tensor = torch.tensor(
                features, dtype=torch.float32, device=self.device
            )
            return feature_tensor.unsqueeze(0).expand(batch_size, -1)
        else:
            # Return empty tensor if no context features
            return torch.zeros(batch_size, 0, device=self.device)

    def _calculate_epistemic_uncertainty(
        self, ensemble_predictions: List[torch.Tensor]
    ) -> np.ndarray:
        """Calculate epistemic uncertainty from ensemble predictions"""
        if len(ensemble_predictions) == 1:
            # Single model - no epistemic uncertainty
            return np.zeros(ensemble_predictions[0].shape[1:])

        # Calculate mean prediction for each model
        model_means = [torch.mean(preds, dim=0) for preds in ensemble_predictions]
        model_means_tensor = torch.stack(model_means, dim=0)

        # Epistemic uncertainty is variance across model means
        epistemic_uncertainty = torch.var(model_means_tensor, dim=0).cpu().numpy()

        return epistemic_uncertainty

    def _calculate_aleatoric_uncertainty(
        self, ensemble_predictions: List[torch.Tensor]
    ) -> np.ndarray:
        """Calculate aleatoric uncertainty from ensemble predictions"""
        # Calculate within-model variance for each model
        model_variances = [torch.var(preds, dim=0) for preds in ensemble_predictions]

        # Average the variances
        aleatoric_uncertainty = (
            torch.mean(torch.stack(model_variances, dim=0), dim=0).cpu().numpy()
        )

        return aleatoric_uncertainty

    def calculate_confidence_intervals(
        self,
        integration_id: str,
        predictions: torch.Tensor,
        confidence_level: Optional[float] = None,
    ) -> Dict[str, Tuple[float, float]]:
        """
        Calculate confidence intervals for predictions

        Args:
            integration_id: Integration identifier
            predictions: Prediction tensor [num_samples, batch_size, output_dim]
            confidence_level: Confidence level (uses config default if None)

        Returns:
            Dictionary of confidence intervals
        """
        config = self.ensemble_configs[integration_id]
        if confidence_level is None:
            confidence_level = config.confidence_level

        alpha = 1 - confidence_level
        lower_quantile = alpha / 2
        upper_quantile = 1 - alpha / 2

        confidence_intervals = {}

        for i in range(predictions.size(2)):
            lower = (
                torch.quantile(predictions[:, :, i], lower_quantile, dim=0)
                .mean()
                .item()
            )
            upper = (
                torch.quantile(predictions[:, :, i], upper_quantile, dim=0)
                .mean()
                .item()
            )
            confidence_intervals[f"output_{i}"] = (lower, upper)

        return confidence_intervals

    def get_ensemble_info(self, integration_id: str) -> Dict[str, Any]:
        """Get information about an ensemble predictor"""
        if integration_id not in self.ensemble_configs:
            raise ValueError(f"Ensemble predictor {integration_id} not found")

        config = self.ensemble_configs[integration_id]

        return {
            "integration_id": integration_id,
            "gan_model_id": config.gan_model_id,
            "pinn_model_id": config.pinn_model_id,
            "coupling_method": config.coupling_method,
            "num_ensemble_samples": config.num_ensemble_samples,
            "confidence_level": config.confidence_level,
            "physics_constraint_weight": config.physics_constraint_weight,
            "adversarial_weight": config.adversarial_weight,
        }

    def list_ensemble_predictors(self) -> List[str]:
        """List all available ensemble predictors"""
        return list(self.ensemble_configs.keys())

class GANPINNIntegrationEngine:
    """
    Main engine for GAN-PINN integration and probabilistic geological modeling
    """

    def __init__(
        self, pinn_engine: PINNEngine, gan_engine: GANEngine, device: torch.device
    ):
        self.pinn_engine = pinn_engine
        self.gan_engine = gan_engine
        self.device = device

        # Initialize components
        self.probabilistic_predictor = ProbabilisticPINNPredictor(
            pinn_engine, gan_engine, device
        )
        self.adversarial_trainer = AdversarialPINNTrainer(
            pinn_engine, gan_engine, device
        )
        self.ensemble_system = EnsemblePredictionSystem(pinn_engine, gan_engine, device)

        logger.info("GAN-PINN Integration Engine initialized")

    def create_probabilistic_model(self, config: GANPINNConfig) -> str:
        """Create a probabilistic geological model"""
        return self.ensemble_system.create_ensemble_predictor(config)

    def predict_probabilistic(
        self,
        integration_id: str,
        inputs: np.ndarray,
        geological_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make probabilistic predictions with uncertainty quantification"""
        input_tensor = torch.tensor(inputs, dtype=torch.float32, device=self.device)
        return self.ensemble_system.predict_ensemble(
            integration_id, input_tensor, geological_context
        )

    def train_adversarial_pinn(
        self,
        pinn_model_id: str,
        gan_model_id: str,
        training_data: torch.utils.data.DataLoader,
        **kwargs,
    ) -> Dict[str, List[float]]:
        """Train PINN with adversarial feedback from GAN"""
        return self.adversarial_trainer.train_pinn_with_gan_feedback(
            pinn_model_id, gan_model_id, training_data, **kwargs
        )

    def get_integration_info(self, integration_id: str) -> Dict[str, Any]:
        """Get information about a GAN-PINN integration"""
        return self.ensemble_system.get_ensemble_info(integration_id)

    def list_integrations(self) -> List[str]:
        """List all available GAN-PINN integrations"""
        return self.ensemble_system.list_ensemble_predictors()

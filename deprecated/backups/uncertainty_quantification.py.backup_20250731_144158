"""
Uncertainty Quantification Methods for Geological Predictions

This module implements various uncertainty quantification techniques specifically
designed for geological modeling and prediction, with emphasis on GAN-based
ensemble methods.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime
import structlog
from scipy import stats
try:
    from sklearn.calibration import calibration_curve
except ImportError:
    # Fallback for older sklearn versions
    try:
        from sklearn.metrics import calibration_curve
    except ImportError:
        # If calibration_curve is not available, create a dummy function
        def calibration_curve(y_true, y_prob, n_bins=5):
            """Dummy calibration curve function"""
            return np.linspace(0, 1, n_bins), np.linspace(0, 1, n_bins)
import warnings

from models.ai_models import UncertaintyQuantification, UncertaintyMethod, ModelPrediction
from models.geological import GeologicalLocation, SeismicEvent, DeformationMeasurement
from .gan_engine import GANEngine, GANEnsemble

logger = structlog.get_logger()


@dataclass
class UncertaintyConfig:
    """Configuration for uncertainty quantification methods"""
    method: UncertaintyMethod
    num_samples: int = 100
    confidence_level: float = 0.95
    calibration_bins: int = 10
    monte_carlo_dropout_rate: float = 0.1
    ensemble_size: int = 5
    bootstrap_samples: int = 1000
    
    def __post_init__(self):
        if not 0 < self.confidence_level < 1:
            raise ValueError("Confidence level must be between 0 and 1")
        if self.num_samples <= 0:
            raise ValueError("Number of samples must be positive")
        if self.calibration_bins <= 0:
            raise ValueError("Calibration bins must be positive")


class MonteCarloDropout:
    """Monte Carlo Dropout for uncertainty quantification"""
    
    def __init__(self, model: nn.Module, dropout_rate: float = 0.1):
        self.model = model
        self.dropout_rate = dropout_rate
        self._enable_dropout()
    
    def _enable_dropout(self):
        """Enable dropout layers during inference"""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    
    def predict_with_uncertainty(
        self, 
        inputs: torch.Tensor, 
        num_samples: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with uncertainty using Monte Carlo Dropout
        
        Args:
            inputs: Input tensor
            num_samples: Number of Monte Carlo samples
            
        Returns:
            Tuple of (mean_predictions, uncertainty)
        """
        self.model.eval()
        self._enable_dropout()
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                pred = self.model(inputs)
                predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)  # [num_samples, batch_size, output_dim]
        
        mean_pred = torch.mean(predictions, dim=0)
        uncertainty = torch.std(predictions, dim=0)
        
        return mean_pred, uncertainty


class BayesianNeuralNetwork:
    """Bayesian Neural Network for uncertainty quantification"""
    
    def __init__(self, model: nn.Module, prior_std: float = 1.0):
        self.model = model
        self.prior_std = prior_std
        self.posterior_samples = []
    
    def variational_inference(
        self, 
        dataloader: torch.utils.data.DataLoader, 
        num_epochs: int = 100,
        learning_rate: float = 0.001
    ):
        """
        Perform variational inference to approximate posterior
        
        Args:
            dataloader: Training data loader
            num_epochs: Number of training epochs
            learning_rate: Learning rate for optimization
        """
        # This is a simplified implementation
        # In practice, would use libraries like Pyro or Edward
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        for epoch in range(num_epochs):
            total_loss = 0
            
            for batch_idx, (data, targets) in enumerate(dataloader):
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(data)
                
                # Likelihood loss
                likelihood_loss = nn.MSELoss()(outputs, targets)
                
                # KL divergence (regularization)
                kl_loss = 0
                for param in self.model.parameters():
                    kl_loss += torch.sum((param / self.prior_std) ** 2)
                kl_loss *= 0.5
                
                # Total loss
                loss = likelihood_loss + kl_loss / len(dataloader.dataset)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 20 == 0:
                logger.info(f"BNN Epoch {epoch}, Loss: {total_loss:.4f}")
    
    def sample_posterior(self, num_samples: int = 100):
        """Sample from posterior distribution"""
        # Simplified posterior sampling
        # In practice, would use MCMC or variational methods
        samples = []
        
        for _ in range(num_samples):
            # Add noise to parameters to simulate posterior sampling
            sample_model = type(self.model)(self.model)
            
            with torch.no_grad():
                for param in sample_model.parameters():
                    noise = torch.randn_like(param) * 0.1
                    param.add_(noise)
            
            samples.append(sample_model)
        
        self.posterior_samples = samples
        return samples
    
    def predict_with_uncertainty(
        self, 
        inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Make predictions with uncertainty using posterior samples"""
        if not self.posterior_samples:
            raise ValueError("Must sample posterior first")
        
        predictions = []
        
        with torch.no_grad():
            for model_sample in self.posterior_samples:
                pred = model_sample(inputs)
                predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)
        
        mean_pred = torch.mean(predictions, dim=0)
        uncertainty = torch.std(predictions, dim=0)
        
        return mean_pred, uncertainty


class ConformalPrediction:
    """Conformal prediction for uncertainty quantification"""
    
    def __init__(self, model: nn.Module, calibration_data: torch.utils.data.DataLoader):
        self.model = model
        self.calibration_data = calibration_data
        self.quantiles = None
    
    def calibrate(self, confidence_level: float = 0.95):
        """
        Calibrate the model using conformal prediction
        
        Args:
            confidence_level: Desired confidence level
        """
        self.model.eval()
        residuals = []
        
        with torch.no_grad():
            for data, targets in self.calibration_data:
                predictions = self.model(data)
                residual = torch.abs(predictions - targets)
                residuals.append(residual)
        
        residuals = torch.cat(residuals, dim=0)
        
        # Calculate quantiles for each output dimension
        alpha = 1 - confidence_level
        quantile_level = 1 - alpha
        
        self.quantiles = torch.quantile(residuals, quantile_level, dim=0)
        
        logger.info(f"Conformal prediction calibrated with {confidence_level} confidence")
    
    def predict_with_intervals(
        self, 
        inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Make predictions with prediction intervals
        
        Args:
            inputs: Input tensor
            
        Returns:
            Tuple of (predictions, lower_bounds, upper_bounds)
        """
        if self.quantiles is None:
            raise ValueError("Must calibrate model first")
        
        self.model.eval()
        
        with torch.no_grad():
            predictions = self.model(inputs)
            
            lower_bounds = predictions - self.quantiles
            upper_bounds = predictions + self.quantiles
        
        return predictions, lower_bounds, upper_bounds


class GeologicalUncertaintyQuantifier:
    """
    Main class for geological uncertainty quantification using multiple methods
    """
    
    def __init__(self, gan_engine: GANEngine, device: torch.device):
        self.gan_engine = gan_engine
        self.device = device
        
        # Available uncertainty methods
        self.methods = {
            UncertaintyMethod.MONTE_CARLO_DROPOUT: self._monte_carlo_dropout,
            UncertaintyMethod.BAYESIAN_NEURAL_NETWORK: self._bayesian_neural_network,
            UncertaintyMethod.ENSEMBLE_VARIANCE: self._ensemble_variance,
            UncertaintyMethod.CONFORMAL_PREDICTION: self._conformal_prediction,
            UncertaintyMethod.DEEP_ENSEMBLES: self._deep_ensembles
        }
        
        logger.info("Geological Uncertainty Quantifier initialized")
    
    def quantify_uncertainty(
        self,
        model: Union[nn.Module, str],
        inputs: torch.Tensor,
        config: UncertaintyConfig,
        targets: Optional[torch.Tensor] = None
    ) -> UncertaintyQuantification:
        """
        Quantify uncertainty using specified method
        
        Args:
            model: Model or model ID for uncertainty quantification
            inputs: Input data
            config: Uncertainty quantification configuration
            targets: Target values (for calibration methods)
            
        Returns:
            Uncertainty quantification results
        """
        if config.method not in self.methods:
            raise ValueError(f"Unsupported uncertainty method: {config.method}")
        
        method_func = self.methods[config.method]
        return method_func(model, inputs, config, targets)
    
    def _monte_carlo_dropout(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        config: UncertaintyConfig,
        targets: Optional[torch.Tensor] = None
    ) -> UncertaintyQuantification:
        """Monte Carlo Dropout uncertainty quantification"""
        mc_dropout = MonteCarloDropout(model, config.monte_carlo_dropout_rate)
        
        mean_pred, uncertainty = mc_dropout.predict_with_uncertainty(
            inputs, config.num_samples
        )
        
        # Calculate confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(
            mean_pred, uncertainty, config.confidence_level
        )
        
        # Separate epistemic and aleatoric uncertainty (simplified)
        epistemic_uncertainty = uncertainty.cpu().numpy() * 0.7  # Model uncertainty
        aleatoric_uncertainty = uncertainty.cpu().numpy() * 0.3  # Data uncertainty
        total_uncertainty = uncertainty.cpu().numpy()
        
        return UncertaintyQuantification(
            method=UncertaintyMethod.MONTE_CARLO_DROPOUT,
            confidence_intervals=confidence_intervals,
            epistemic_uncertainty=epistemic_uncertainty,
            aleatoric_uncertainty=aleatoric_uncertainty,
            total_uncertainty=total_uncertainty
        )
    
    def _bayesian_neural_network(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        config: UncertaintyConfig,
        targets: Optional[torch.Tensor] = None
    ) -> UncertaintyQuantification:
        """Bayesian Neural Network uncertainty quantification"""
        bnn = BayesianNeuralNetwork(model)
        
        # Sample from posterior
        bnn.sample_posterior(config.num_samples)
        
        mean_pred, uncertainty = bnn.predict_with_uncertainty(inputs)
        
        # Calculate confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(
            mean_pred, uncertainty, config.confidence_level
        )
        
        # For BNN, most uncertainty is epistemic
        epistemic_uncertainty = uncertainty.cpu().numpy() * 0.8
        aleatoric_uncertainty = uncertainty.cpu().numpy() * 0.2
        total_uncertainty = uncertainty.cpu().numpy()
        
        return UncertaintyQuantification(
            method=UncertaintyMethod.BAYESIAN_NEURAL_NETWORK,
            confidence_intervals=confidence_intervals,
            epistemic_uncertainty=epistemic_uncertainty,
            aleatoric_uncertainty=aleatoric_uncertainty,
            total_uncertainty=total_uncertainty
        )
    
    def _ensemble_variance(
        self,
        model: Union[str, GANEnsemble],
        inputs: torch.Tensor,
        config: UncertaintyConfig,
        targets: Optional[torch.Tensor] = None
    ) -> UncertaintyQuantification:
        """Ensemble variance uncertainty quantification using GANs"""
        if isinstance(model, str):
            # Get GAN ensemble by ID
            ensemble = self.gan_engine.get_ensemble(model)
        else:
            ensemble = model
        
        # Generate scenarios and quantify uncertainty
        uncertainty_result = ensemble.quantify_uncertainty(config.num_samples)
        
        return uncertainty_result
    
    def _conformal_prediction(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        config: UncertaintyConfig,
        targets: Optional[torch.Tensor] = None
    ) -> UncertaintyQuantification:
        """Conformal prediction uncertainty quantification"""
        if targets is None:
            raise ValueError("Conformal prediction requires calibration targets")
        
        # Create calibration dataset
        calibration_dataset = torch.utils.data.TensorDataset(inputs, targets)
        calibration_loader = torch.utils.data.DataLoader(
            calibration_dataset, batch_size=32, shuffle=False
        )
        
        conformal = ConformalPrediction(model, calibration_loader)
        conformal.calibrate(config.confidence_level)
        
        predictions, lower_bounds, upper_bounds = conformal.predict_with_intervals(inputs)
        
        # Calculate uncertainty from prediction intervals
        uncertainty = (upper_bounds - lower_bounds) / 2
        
        # Calculate confidence intervals
        confidence_intervals = {}
        for i in range(predictions.shape[1]):
            confidence_intervals[f"param_{i}"] = (
                lower_bounds[:, i].mean().item(),
                upper_bounds[:, i].mean().item()
            )
        
        # For conformal prediction, uncertainty is mostly aleatoric
        epistemic_uncertainty = uncertainty.cpu().numpy() * 0.2
        aleatoric_uncertainty = uncertainty.cpu().numpy() * 0.8
        total_uncertainty = uncertainty.cpu().numpy()
        
        return UncertaintyQuantification(
            method=UncertaintyMethod.CONFORMAL_PREDICTION,
            confidence_intervals=confidence_intervals,
            epistemic_uncertainty=epistemic_uncertainty,
            aleatoric_uncertainty=aleatoric_uncertainty,
            total_uncertainty=total_uncertainty
        )
    
    def _deep_ensembles(
        self,
        model: Union[List[nn.Module], str],
        inputs: torch.Tensor,
        config: UncertaintyConfig,
        targets: Optional[torch.Tensor] = None
    ) -> UncertaintyQuantification:
        """Deep ensembles uncertainty quantification"""
        if isinstance(model, str):
            # Get ensemble models (simplified - would need model registry)
            raise NotImplementedError("Model registry lookup not implemented")
        
        models = model
        predictions = []
        
        # Get predictions from all models
        with torch.no_grad():
            for m in models:
                m.eval()
                pred = m(inputs)
                predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)  # [num_models, batch_size, output_dim]
        
        mean_pred = torch.mean(predictions, dim=0)
        uncertainty = torch.std(predictions, dim=0)
        
        # Calculate confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(
            mean_pred, uncertainty, config.confidence_level
        )
        
        # For deep ensembles, uncertainty is mostly epistemic
        epistemic_uncertainty = uncertainty.cpu().numpy() * 0.9
        aleatoric_uncertainty = uncertainty.cpu().numpy() * 0.1
        total_uncertainty = uncertainty.cpu().numpy()
        
        return UncertaintyQuantification(
            method=UncertaintyMethod.DEEP_ENSEMBLES,
            confidence_intervals=confidence_intervals,
            epistemic_uncertainty=epistemic_uncertainty,
            aleatoric_uncertainty=aleatoric_uncertainty,
            total_uncertainty=total_uncertainty
        )
    
    def _calculate_confidence_intervals(
        self,
        predictions: torch.Tensor,
        uncertainty: torch.Tensor,
        confidence_level: float
    ) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals from predictions and uncertainty"""
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        
        confidence_intervals = {}
        
        for i in range(predictions.shape[1]):
            mean_val = predictions[:, i].mean().item()
            std_val = uncertainty[:, i].mean().item()
            
            lower = mean_val - z_score * std_val
            upper = mean_val + z_score * std_val
            
            confidence_intervals[f"param_{i}"] = (lower, upper)
        
        return confidence_intervals
    
    def calibrate_uncertainty(
        self,
        predictions: List[ModelPrediction],
        true_values: torch.Tensor,
        num_bins: int = 10
    ) -> Dict[str, float]:
        """
        Calibrate uncertainty estimates using reliability diagrams
        
        Args:
            predictions: List of model predictions with uncertainty
            true_values: True values for calibration
            num_bins: Number of bins for calibration curve
            
        Returns:
            Calibration metrics
        """
        confidences = []
        accuracies = []
        
        for pred in predictions:
            # Extract confidence scores
            confidence = pred.confidence_score
            confidences.append(confidence)
            
            # Calculate accuracy (simplified binary classification)
            # In practice, would need proper accuracy metric for regression
            pred_values = torch.tensor(pred.values)
            true_vals = true_values[len(accuracies)]
            
            # Use relative error as accuracy measure
            relative_error = torch.abs(pred_values - true_vals) / (torch.abs(true_vals) + 1e-8)
            accuracy = (relative_error < 0.1).float().mean().item()  # Within 10% error
            accuracies.append(accuracy)
        
        confidences = np.array(confidences)
        accuracies = np.array(accuracies)
        
        # Calculate calibration curve
        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                accuracies > 0.5,  # Binary accuracy
                confidences,
                n_bins=num_bins
            )
            
            # Expected Calibration Error (ECE)
            bin_boundaries = np.linspace(0, 1, num_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            ece = 0
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = accuracies[in_bin].mean()
                    avg_confidence_in_bin = confidences[in_bin].mean()
                    ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            # Maximum Calibration Error (MCE)
            mce = 0
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
                
                if in_bin.sum() > 0:
                    accuracy_in_bin = accuracies[in_bin].mean()
                    avg_confidence_in_bin = confidences[in_bin].mean()
                    mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
            
            return {
                "expected_calibration_error": ece,
                "maximum_calibration_error": mce,
                "average_confidence": confidences.mean(),
                "average_accuracy": accuracies.mean()
            }
            
        except Exception as e:
            logger.warning(f"Calibration calculation failed: {e}")
            return {
                "expected_calibration_error": 0.0,
                "maximum_calibration_error": 0.0,
                "average_confidence": confidences.mean(),
                "average_accuracy": accuracies.mean()
            }
    
    def generate_geological_scenarios_with_uncertainty(
        self,
        ensemble_id: str,
        num_scenarios: int,
        geological_constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate geological scenarios with uncertainty quantification
        
        Args:
            ensemble_id: GAN ensemble identifier
            num_scenarios: Number of scenarios to generate
            geological_constraints: Optional geological constraints
            
        Returns:
            Generated scenarios with uncertainty metrics
        """
        ensemble = self.gan_engine.get_ensemble(ensemble_id)
        
        # Generate scenarios from ensemble
        scenarios_list = ensemble.generate_ensemble_scenarios(num_scenarios)
        
        # Calculate ensemble statistics
        all_scenarios = torch.stack(scenarios_list, dim=0)  # [num_models, num_scenarios, output_dim]
        
        mean_scenarios = torch.mean(all_scenarios, dim=0)
        std_scenarios = torch.std(all_scenarios, dim=0)
        
        # Apply geological constraints if provided
        if geological_constraints:
            mean_scenarios, std_scenarios = self._apply_geological_constraints(
                mean_scenarios, std_scenarios, geological_constraints
            )
        
        # Calculate plausibility scores
        plausibility_scores = []
        for scenarios in scenarios_list:
            for gan in ensemble.gans:
                scores = gan.assess_plausibility(scenarios)
                plausibility_scores.append(scores)
        
        avg_plausibility = torch.mean(torch.stack(plausibility_scores), dim=0)
        
        return {
            "scenarios": mean_scenarios.cpu().numpy().tolist(),
            "uncertainty": std_scenarios.cpu().numpy().tolist(),
            "plausibility_scores": avg_plausibility.cpu().numpy().tolist(),
            "num_scenarios": num_scenarios,
            "ensemble_size": len(scenarios_list),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _apply_geological_constraints(
        self,
        scenarios: torch.Tensor,
        uncertainties: torch.Tensor,
        constraints: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply geological constraints to generated scenarios"""
        # This is a simplified implementation
        # In practice, would apply complex geological constraints
        
        constrained_scenarios = scenarios.clone()
        constrained_uncertainties = uncertainties.clone()
        
        # Example constraints
        if "temperature_range" in constraints:
            temp_min, temp_max = constraints["temperature_range"]
            temp_idx = constraints.get("temperature_index", 0)
            
            # Clamp temperature values
            constrained_scenarios[:, temp_idx] = torch.clamp(
                constrained_scenarios[:, temp_idx], temp_min, temp_max
            )
            
            # Adjust uncertainty based on constraint violation
            violations = (scenarios[:, temp_idx] < temp_min) | (scenarios[:, temp_idx] > temp_max)
            constrained_uncertainties[violations, temp_idx] *= 1.5  # Increase uncertainty for violations
        
        if "pressure_temperature_correlation" in constraints:
            # Apply pressure-temperature correlation constraints
            pressure_idx = constraints.get("pressure_index", 1)
            temp_idx = constraints.get("temperature_index", 0)
            
            # Simple linear correlation constraint
            expected_pressure = constrained_scenarios[:, temp_idx] * 0.1  # Simplified relationship
            pressure_diff = torch.abs(constrained_scenarios[:, pressure_idx] - expected_pressure)
            
            # Increase uncertainty for scenarios that violate correlation
            violation_mask = pressure_diff > constraints.get("correlation_tolerance", 0.5)
            constrained_uncertainties[violation_mask, pressure_idx] *= 2.0
        
        return constrained_scenarios, constrained_uncertainties
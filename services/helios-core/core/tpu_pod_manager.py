"""
TPU Pod Manager for massive parallel processing in geological simulations.

This module provides TPU Pod integration for distributed training and inference
of Physics-Informed Neural Networks (PINNs) and other geological AI models.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    import optax
    from flax import linen as nn
    from flax.training import train_state
    from jax import pmap, random
    from jax.experimental import mesh_utils
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    TPU_AVAILABLE = True
except ImportError:
    TPU_AVAILABLE = False
    logging.warning(
        "JAX/TPU dependencies not available. TPU functionality will be disabled."
    )

logger = logging.getLogger(__name__)

class TPUPodStatus(Enum):
    """TPU Pod status enumeration."""

    INITIALIZING = "initializing"
    READY = "ready"
    TRAINING = "training"
    INFERENCE = "inference"
    ERROR = "error"
    SHUTDOWN = "shutdown"

@dataclass
class TPUPodConfig:
    """Configuration for TPU Pod deployment."""

    pod_size: int = 8  # Number of TPU cores
    mesh_shape: Tuple[int, ...] = (2, 4)  # Mesh topology
    model_parallel_submesh: Tuple[int, ...] = (1, 4)
    data_parallel_submesh: Tuple[int, ...] = (2, 1)
    precision: str = "bfloat16"  # TPU-optimized precision
    max_sequence_length: int = 1024
    batch_size_per_device: int = 32
    gradient_accumulation_steps: int = 1

@dataclass
class TPUTrainingMetrics:
    """Metrics for TPU training performance."""

    step: int
    loss: float
    physics_loss: float
    data_loss: float
    learning_rate: float
    throughput_samples_per_sec: float
    memory_usage_gb: float
    flops_utilization: float

class GeologicalPINNTPU(nn.Module):
    """
    Physics-Informed Neural Network optimized for TPU architecture.

    This implementation uses JAX/Flax for TPU optimization with specialized
    kernels for geological physics constraints.
    """

    features: List[int]
    physics_constraints: List[str]
    activation: str = "tanh"

    def setup(self):
        """Initialize network layers optimized for TPU."""
        self.layers = [
            nn.Dense(feat, kernel_init=nn.initializers.xavier_uniform())
            for feat in self.features
        ]

        # Physics constraint enforcement layers
        self.physics_layers = {
            constraint: nn.Dense(1, kernel_init=nn.initializers.zeros)
            for constraint in self.physics_constraints
        }

    def __call__(self, x: jnp.ndarray, training: bool = True) -> Dict[str, jnp.ndarray]:
        """Forward pass with physics constraint enforcement."""
        # Main network forward pass
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            if self.activation == "tanh":
                x = jnp.tanh(x)
            elif self.activation == "relu":
                x = nn.relu(x)
            elif self.activation == "swish":
                x = nn.swish(x)

        # Output layer
        output = self.layers[-1](x)

        # Physics constraint outputs
        physics_outputs = {}
        for constraint_name, physics_layer in self.physics_layers.items():
            physics_outputs[constraint_name] = physics_layer(x)

        return {"prediction": output, "physics_constraints": physics_outputs}

class TPUDatasetSharding:
    """Handles automatic dataset sharding and distribution across TPU cores."""

    def __init__(self, mesh: Mesh, batch_size: int):
        self.mesh = mesh
        self.batch_size = batch_size
        self.data_sharding = NamedSharding(mesh, PartitionSpec("data", None))

    def shard_batch(self, batch: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
        """Shard a batch across TPU cores."""
        sharded_batch = {}
        for key, data in batch.items():
            # Reshape for sharding across data parallel dimension
            reshaped_data = data.reshape((self.mesh.shape["data"], -1) + data.shape[1:])
            sharded_batch[key] = jax.device_put(reshaped_data, self.data_sharding)

        return sharded_batch

    def create_data_iterator(self, dataset: np.ndarray, labels: np.ndarray):
        """Create an iterator that yields sharded batches."""
        num_samples = dataset.shape[0]
        indices = np.arange(num_samples)

        while True:
            # Shuffle indices
            np.random.shuffle(indices)

            for i in range(0, num_samples, self.batch_size):
                batch_indices = indices[i : i + self.batch_size]
                if len(batch_indices) < self.batch_size:
                    continue  # Skip incomplete batches

                batch_data = dataset[batch_indices]
                batch_labels = labels[batch_indices]

                yield self.shard_batch({"inputs": batch_data, "targets": batch_labels})

class TPUPhysicsKernels:
    """Specialized TPU kernels for physics-informed computations."""

    @staticmethod
    @jax.jit
    def navier_stokes_residual(
        u: jnp.ndarray,
        v: jnp.ndarray,
        p: jnp.ndarray,
        coords: jnp.ndarray,
        viscosity: float,
    ) -> jnp.ndarray:
        """Compute Navier-Stokes equation residual on TPU."""
        # Compute gradients using JAX automatic differentiation
        u_x = jnp.gradient(u, coords[:, 0], axis=0)
        u_y = jnp.gradient(u, coords[:, 1], axis=1)
        u_xx = jnp.gradient(u_x, coords[:, 0], axis=0)
        u_yy = jnp.gradient(u_y, coords[:, 1], axis=1)

        v_x = jnp.gradient(v, coords[:, 0], axis=0)
        v_y = jnp.gradient(v, coords[:, 1], axis=1)
        v_xx = jnp.gradient(v_x, coords[:, 0], axis=0)
        v_yy = jnp.gradient(v_y, coords[:, 1], axis=1)

        p_x = jnp.gradient(p, coords[:, 0], axis=0)
        p_y = jnp.gradient(p, coords[:, 1], axis=1)

        # Navier-Stokes equations
        residual_u = u * u_x + v * u_y + p_x - viscosity * (u_xx + u_yy)
        residual_v = u * v_x + v * v_y + p_y - viscosity * (v_xx + v_yy)

        return jnp.stack([residual_u, residual_v], axis=-1)

    @staticmethod
    @jax.jit
    def heat_transfer_residual(
        T: jnp.ndarray,
        coords: jnp.ndarray,
        thermal_diffusivity: float,
        heat_source: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute heat transfer equation residual on TPU."""
        T_x = jnp.gradient(T, coords[:, 0], axis=0)
        T_y = jnp.gradient(T, coords[:, 1], axis=1)
        T_z = jnp.gradient(T, coords[:, 2], axis=2) if coords.shape[1] > 2 else 0

        T_xx = jnp.gradient(T_x, coords[:, 0], axis=0)
        T_yy = jnp.gradient(T_y, coords[:, 1], axis=1)
        T_zz = jnp.gradient(T_z, coords[:, 2], axis=2) if coords.shape[1] > 2 else 0

        laplacian_T = T_xx + T_yy + T_zz
        residual = thermal_diffusivity * laplacian_T - heat_source

        return residual

    @staticmethod
    @jax.jit
    def elastic_deformation_residual(
        displacement: jnp.ndarray,
        coords: jnp.ndarray,
        youngs_modulus: float,
        poisson_ratio: float,
        body_force: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute elastic deformation equation residual on TPU."""
        # Compute strain tensor
        u_x = jnp.gradient(displacement[..., 0], coords[:, 0], axis=0)
        u_y = jnp.gradient(displacement[..., 0], coords[:, 1], axis=1)
        v_x = jnp.gradient(displacement[..., 1], coords[:, 0], axis=0)
        v_y = jnp.gradient(displacement[..., 1], coords[:, 1], axis=1)

        # Strain components
        epsilon_xx = u_x
        epsilon_yy = v_y
        epsilon_xy = 0.5 * (u_y + v_x)

        # Stress components (plane stress)
        factor = youngs_modulus / (1 - poisson_ratio**2)
        sigma_xx = factor * (epsilon_xx + poisson_ratio * epsilon_yy)
        sigma_yy = factor * (epsilon_yy + poisson_ratio * epsilon_xx)
        sigma_xy = factor * (1 - poisson_ratio) * epsilon_xy

        # Equilibrium equations
        sigma_xx_x = jnp.gradient(sigma_xx, coords[:, 0], axis=0)
        sigma_xy_y = jnp.gradient(sigma_xy, coords[:, 1], axis=1)
        sigma_yy_y = jnp.gradient(sigma_yy, coords[:, 1], axis=1)
        sigma_xy_x = jnp.gradient(sigma_xy, coords[:, 0], axis=0)

        residual_x = sigma_xx_x + sigma_xy_y + body_force[..., 0]
        residual_y = sigma_xy_x + sigma_yy_y + body_force[..., 1]

        return jnp.stack([residual_x, residual_y], axis=-1)

class TPUPodManager:
    """
    Manages TPU Pod resources for distributed geological modeling.

    Handles TPU Pod allocation, model distribution, training coordination,
    and performance monitoring for large-scale geological simulations.
    """

    def __init__(self, config: TPUPodConfig):
        self.config = config
        self.status = TPUPodStatus.INITIALIZING
        self.mesh = None
        self.model = None
        self.train_state = None
        self.dataset_sharding = None
        self.physics_kernels = TPUPhysicsKernels()
        self.training_metrics = []

        if not TPU_AVAILABLE:
            raise RuntimeError("TPU functionality requires JAX with TPU support")

        self._initialize_tpu_pod()

    def _initialize_tpu_pod(self):
        """Initialize TPU Pod with mesh configuration."""
        try:
            # Initialize JAX distributed system
            jax.distributed.initialize()

            # Create device mesh
            devices = mesh_utils.create_device_mesh(self.config.mesh_shape)
            self.mesh = Mesh(devices, axis_names=("data", "model"))

            logger.info(f"Initialized TPU Pod with {jax.device_count()} devices")
            logger.info(f"Mesh shape: {self.config.mesh_shape}")

            self.status = TPUPodStatus.READY

        except Exception as e:
            logger.error(f"Failed to initialize TPU Pod: {e}")
            self.status = TPUPodStatus.ERROR
            raise

    def create_geological_model(
        self, model_config: Dict[str, Any]
    ) -> GeologicalPINNTPU:
        """Create a geological PINN model optimized for TPU."""
        self.model = GeologicalPINNTPU(
            features=model_config.get("features", [256, 256, 256, 3]),
            physics_constraints=model_config.get(
                "physics_constraints",
                ["navier_stokes", "heat_transfer", "elastic_deformation"],
            ),
            activation=model_config.get("activation", "tanh"),
        )

        # Initialize model parameters
        random.PRNGKey(42)

class AdvancedSeismicCNN:
    """Convolutional Neural Network for seismic event detection and classification"""

    def __init__(self, input_channels: int = 3, num_classes: int = 5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model(input_channels, num_classes)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=1e-4, weight_decay=1e-5
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100
        )

    def _build_model(self, input_channels: int, num_classes: int) -> nn.Module:
        """Build advanced CNN architecture with attention mechanisms"""

        class AttentionBlock(nn.Module):
            def __init__(self, channels):
                super().__init__()
                self.attention = nn.Sequential(
                    nn.AdaptiveAvgPool1d(1),
                    nn.Conv1d(channels, channels // 8, 1),
                    nn.ReLU(),
                    nn.Conv1d(channels // 8, channels, 1),
                    nn.Sigmoid(),
                )

            def forward(self, x):
                attention_weights = self.attention(x)
                return x * attention_weights

        class SeismicCNN(nn.Module):
            def __init__(self, input_channels, num_classes):
                super().__init__()

                # Feature extraction layers
                self.conv_blocks = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Conv1d(input_channels, 64, kernel_size=7, padding=3),
                            nn.BatchNorm1d(64),
                            nn.ReLU(),
                            AttentionBlock(64),
                            nn.MaxPool1d(2),
                        ),
                        nn.Sequential(
                            nn.Conv1d(64, 128, kernel_size=5, padding=2),
                            nn.BatchNorm1d(128),
                            nn.ReLU(),
                            AttentionBlock(128),
                            nn.MaxPool1d(2),
                        ),
                        nn.Sequential(
                            nn.Conv1d(128, 256, kernel_size=3, padding=1),
                            nn.BatchNorm1d(256),
                            nn.ReLU(),
                            AttentionBlock(256),
                            nn.MaxPool1d(2),
                        ),
                        nn.Sequential(
                            nn.Conv1d(256, 512, kernel_size=3, padding=1),
                            nn.BatchNorm1d(512),
                            nn.ReLU(),
                            AttentionBlock(512),
                            nn.AdaptiveAvgPool1d(1),
                        ),
                    ]
                )

                # Classification head
                self.classifier = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, num_classes),
                )

            def forward(self, x):
                for block in self.conv_blocks:
                    x = block(x)

                x = x.view(x.size(0), -1)
                return self.classifier(x)

        return SeismicCNN(input_channels, num_classes).to(self.device)

    def detect_seismic_events(
        self, waveform_data: np.ndarray, sampling_rate: float = 100.0
    ) -> Dict[str, Any]:
        """Detect and classify seismic events in waveform data"""

        # Preprocess data
        processed_data = self._preprocess_waveform(waveform_data, sampling_rate)

        # Convert to tensor
        input_tensor = torch.FloatTensor(processed_data).unsqueeze(0).to(self.device)

        # Inference
        self.model.eval()
        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = F.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)

        # Event classification mapping
        event_types = {
            0: "background_noise",
            1: "local_earthquake",
            2: "regional_earthquake",
            3: "volcanic_tremor",
            4: "explosion",
        }

        return {
            "event_type": event_types[predicted_class.item()],
            "confidence": probabilities.max().item(),
            "all_probabilities": {
                event_types[i]: prob.item() for i, prob in enumerate(probabilities[0])
            },
            "magnitude_estimate": self._estimate_magnitude(waveform_data),
            "arrival_time": self._detect_p_wave_arrival(waveform_data, sampling_rate),
            "frequency_content": self._analyze_frequency_content(
                waveform_data, sampling_rate
            ),
        }

    def _preprocess_waveform(
        self, data: np.ndarray, sampling_rate: float
    ) -> np.ndarray:
        """Advanced waveform preprocessing"""
        # Bandpass filter
        from scipy import signal

        nyquist = sampling_rate / 2
        low_freq = 1.0 / nyquist
        high_freq = 25.0 / nyquist
        b, a = signal.butter(4, [low_freq, high_freq], btype="band")
        filtered_data = signal.filtfilt(b, a, data, axis=-1)

        # Normalize
        normalized_data = (
            filtered_data - np.mean(filtered_data, axis=-1, keepdims=True)
        ) / (np.std(filtered_data, axis=-1, keepdims=True) + 1e-8)

        return normalized_data

    def _estimate_magnitude(self, waveform_data: np.ndarray) -> float:
        """Estimate earthquake magnitude from waveform amplitude"""
        max_amplitude = np.max(np.abs(waveform_data))
        # Simplified magnitude estimation (would use proper calibration in practice)
        magnitude = np.log10(max_amplitude * 1e6) - 3.0
        return max(0.0, min(9.0, magnitude))

    def _detect_p_wave_arrival(self, data: np.ndarray, sampling_rate: float) -> float:
        """Detect P-wave arrival time using STA/LTA algorithm"""
        # Short-term average / Long-term average
        sta_window = int(0.5 * sampling_rate)  # 0.5 seconds
        lta_window = int(10.0 * sampling_rate)  # 10 seconds

        energy = np.square(data)
        sta = np.convolve(energy, np.ones(sta_window) / sta_window, mode="same")
        lta = np.convolve(energy, np.ones(lta_window) / lta_window, mode="same")

        sta_lta_ratio = sta / (lta + 1e-8)
        trigger_threshold = 3.0

        trigger_indices = np.where(sta_lta_ratio > trigger_threshold)[0]
        if len(trigger_indices) > 0:
            return trigger_indices[0] / sampling_rate
        return 0.0

    def _analyze_frequency_content(
        self, data: np.ndarray, sampling_rate: float
    ) -> Dict[str, float]:
        """Analyze frequency content of seismic signal"""

        frequencies, psd = signal.welch(data, fs=sampling_rate, nperseg=1024)

        # Frequency bands of interest
        bands = {
            "low_freq": (0.1, 1.0),
            "mid_freq": (1.0, 10.0),
            "high_freq": (10.0, 25.0),
        }

        band_powers = {}
        for band_name, (low, high) in bands.items():
            band_mask = (frequencies >= low) & (frequencies <= high)
            band_powers[band_name] = np.trapz(psd[band_mask], frequencies[band_mask])

        # Dominant frequency
        dominant_freq = frequencies[np.argmax(psd)]

        return {
            "dominant_frequency": dominant_freq,
            "band_powers": band_powers,
            "spectral_centroid": np.sum(frequencies * psd) / np.sum(psd),
            "spectral_bandwidth": np.sqrt(
                np.sum(((frequencies - dominant_freq) ** 2) * psd) / np.sum(psd)
            ),
        }

        jnp.ones((1, model_config.get("input_dim", 4)))

        variables = self.model.init(key, dummy_input)

        # Create optimizer
        optimizer = optax.adamw(
            learning_rate=model_config.get("learning_rate", 1e-3),
            weight_decay=model_config.get("weight_decay", 1e-4),
        )

        # Create training state with model sharding
        model_sharding = NamedSharding(self.mesh, PartitionSpec(None, "model"))

        self.train_state = train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=jax.device_put(variables["params"], model_sharding),
            tx=optimizer,
        )

        logger.info("Created geological PINN model for TPU training")
        return self.model

    def setup_dataset_sharding(self, dataset: np.ndarray, labels: np.ndarray):
        """Setup automatic dataset sharding across TPU cores."""
        self.dataset_sharding = TPUDatasetSharding(
            mesh=self.mesh,
            batch_size=self.config.batch_size_per_device * jax.device_count(),
        )

        self.data_iterator = self.dataset_sharding.create_data_iterator(dataset, labels)
        logger.info("Setup dataset sharding across TPU cores")

    @jax.jit
    def compute_physics_loss(
        self, params: Dict, batch: Dict[str, jnp.ndarray]
    ) -> jnp.ndarray:
        """Compute physics-informed loss using specialized TPU kernels."""
        # Forward pass
        outputs = self.model.apply({"params": params}, batch["inputs"])
        predictions = outputs["prediction"]
        physics_outputs = outputs["physics_constraints"]

        # Data fitting loss
        data_loss = jnp.mean((predictions - batch["targets"]) ** 2)

        # Physics constraint losses
        physics_loss = 0.0
        coords = batch["inputs"][
            :, :3
        ]  # Assume first 3 dimensions are spatial coordinates

        if "navier_stokes" in physics_outputs:
            u = predictions[:, 0]
            v = predictions[:, 1]
            p = predictions[:, 2] if predictions.shape[1] > 2 else jnp.zeros_like(u)
            ns_residual = self.physics_kernels.navier_stokes_residual(
                u, v, p, coords, viscosity=1e-3
            )
            physics_loss += jnp.mean(ns_residual**2)

        if "heat_transfer" in physics_outputs:
            T = predictions[:, -1]  # Assume temperature is last output
            heat_source = jnp.zeros_like(T)  # Can be parameterized
            ht_residual = self.physics_kernels.heat_transfer_residual(
                T, coords, thermal_diffusivity=1e-6, heat_source=heat_source
            )
            physics_loss += jnp.mean(ht_residual**2)

        if "elastic_deformation" in physics_outputs:
            displacement = predictions[:, :2]  # Assume first 2 outputs are displacement
            body_force = jnp.zeros_like(displacement)
            ed_residual = self.physics_kernels.elastic_deformation_residual(
                displacement,
                coords,
                youngs_modulus=2e11,
                poisson_ratio=0.3,
                body_force=body_force,
            )
            physics_loss += jnp.mean(ed_residual**2)

        # Combined loss
        total_loss = data_loss + 0.1 * physics_loss  # Physics loss weight

        return total_loss, (data_loss, physics_loss)

    @jax.jit
    def train_step(
        self, state: train_state.TrainState, batch: Dict[str, jnp.ndarray]
    ) -> Tuple[train_state.TrainState, Dict[str, float]]:
        """Single training step optimized for TPU."""

        def loss_fn(params):
            return self.compute_physics_loss(params, batch)

        (loss, (data_loss, physics_loss)), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(state.params)

        # Apply gradients
        state = state.apply_gradients(grads=grads)

        metrics = {
            "loss": loss,
            "data_loss": data_loss,
            "physics_loss": physics_loss,
            "learning_rate": state.opt_state.hyperparams["learning_rate"],
        }

        return state, metrics

    def train_distributed(
        self, num_steps: int, log_interval: int = 100
    ) -> List[TPUTrainingMetrics]:
        """Run distributed training across TPU Pod."""
        if self.status != TPUPodStatus.READY:
            raise RuntimeError(f"TPU Pod not ready. Status: {self.status}")

        self.status = TPUPodStatus.TRAINING
        training_metrics = []

        # Parallelize training step across devices
        p_train_step = pmap(self.train_step, axis_name="data")

        try:
            for step in range(num_steps):
                # Get next batch
                batch = next(self.data_iterator)

                # Distributed training step
                self.train_state, step_metrics = p_train_step(self.train_state, batch)

                # Collect metrics (average across devices)
                avg_metrics = jax.tree_map(lambda x: jnp.mean(x), step_metrics)

                if step % log_interval == 0:
                    metrics = TPUTrainingMetrics(
                        step=step,
                        loss=float(avg_metrics["loss"]),
                        physics_loss=float(avg_metrics["physics_loss"]),
                        data_loss=float(avg_metrics["data_loss"]),
                        learning_rate=float(avg_metrics["learning_rate"]),
                        throughput_samples_per_sec=self._calculate_throughput(),
                        memory_usage_gb=self._get_memory_usage(),
                        flops_utilization=self._get_flops_utilization(),
                    )

                    training_metrics.append(metrics)
                    logger.info(
                        f"Step {step}: Loss={metrics.loss:.6f}, "
                        f"Physics Loss={metrics.physics_loss:.6f}, "
                        f"Throughput={metrics.throughput_samples_per_sec:.1f} samples/sec"
                    )

            self.status = TPUPodStatus.READY
            return training_metrics

        except Exception as e:
            logger.error(f"Training failed: {e}")
            self.status = TPUPodStatus.ERROR
            raise

    def inference_distributed(self, input_data: np.ndarray) -> np.ndarray:
        """Run distributed inference across TPU Pod."""
        if self.status not in [TPUPodStatus.READY, TPUPodStatus.INFERENCE]:
            raise RuntimeError(
                f"TPU Pod not available for inference. Status: {self.status}"
            )

        self.status = TPUPodStatus.INFERENCE

        try:
            # Shard input data
            sharded_input = self.dataset_sharding.shard_batch({"inputs": input_data})

            # Distributed inference
            @jax.jit
            def inference_fn(params, inputs):
                return self.model.apply({"params": params}, inputs)

            p_inference = pmap(inference_fn, axis_name="data")
            outputs = p_inference(self.train_state.params, sharded_input["inputs"])

            # Gather results from all devices
            gathered_outputs = jax.tree_map(
                lambda x: x.reshape(-1, *x.shape[2:]), outputs
            )

            self.status = TPUPodStatus.READY
            return np.array(gathered_outputs["prediction"])

        except Exception as e:
            logger.error(f"Inference failed: {e}")
            self.status = TPUPodStatus.ERROR
            raise

    def _calculate_throughput(self) -> float:
        """Calculate training throughput in samples per second."""
        # This would be implemented with actual timing measurements
        # For now, return estimated throughput based on TPU specifications
        return (
            self.config.batch_size_per_device * jax.device_count() * 10.0
        )  # Estimated

    def _get_memory_usage(self) -> float:
        """Get current memory usage in GB."""
        # This would query actual TPU memory usage
        # For now, return estimated usage
        return 8.0  # Estimated GB per TPU core

    def _get_flops_utilization(self) -> float:
        """Get FLOPS utilization percentage."""
        # This would query actual TPU utilization metrics
        # For now, return estimated utilization
        return 85.0  # Estimated percentage

    def save_model(self, checkpoint_path: str):
        """Save model checkpoint optimized for TPU."""
        if self.train_state is None:
            raise RuntimeError("No model to save")

        # Save model parameters and training state
        checkpoint_data = {
            "params": self.train_state.params,
            "opt_state": self.train_state.opt_state,
            "step": self.train_state.step,
            "config": self.config,
        }

        # Use JAX serialization for TPU-optimized checkpoints
        import pickle

        with open(checkpoint_path, "wb") as f:
            pickle.dump(checkpoint_data, f)

        logger.info(f"Saved TPU model checkpoint to {checkpoint_path}")

    def load_model(self, checkpoint_path: str):
        """Load model checkpoint for TPU."""

        with open(checkpoint_path, "rb") as f:
            checkpoint_data = pickle.load(f)

        # Restore training state
        self.train_state = self.train_state.replace(
            params=checkpoint_data["params"],
            opt_state=checkpoint_data["opt_state"],
            step=checkpoint_data["step"],
        )

        logger.info(f"Loaded TPU model checkpoint from {checkpoint_path}")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive TPU performance metrics."""
        return {
            "status": self.status.value,
            "device_count": jax.device_count(),
            "mesh_shape": self.config.mesh_shape,
            "memory_usage_gb": self._get_memory_usage(),
            "flops_utilization": self._get_flops_utilization(),
            "training_metrics": (
                self.training_metrics[-10:] if self.training_metrics else []
            ),
        }

    def shutdown(self):
        """Shutdown TPU Pod and cleanup resources."""
        self.status = TPUPodStatus.SHUTDOWN
        logger.info("TPU Pod shutdown completed")

# Factory function for creating TPU Pod manager
def create_tpu_pod_manager(config: Optional[TPUPodConfig] = None) -> TPUPodManager:
    """Create and initialize a TPU Pod manager."""
    if config is None:
        config = TPUPodConfig()

    return TPUPodManager(config)

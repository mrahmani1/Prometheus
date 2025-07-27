"""
CNN Training Module for Seismic Event Detection

This module provides training capabilities for the CNN-based seismic event
detection system, including data preparation, model training, validation,
and performance evaluation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path
import json
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

from .seismic_cnn_processor import SeismicCNN, SeismicEventType
from ..models.geological import GeologicalLocation


logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for CNN training"""
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 100
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    weight_decay: float = 1e-4
    scheduler_step_size: int = 30
    scheduler_gamma: float = 0.1
    device: str = "cpu"
    save_best_model: bool = True
    model_save_path: str = "best_seismic_model.pth"


class SeismicDataset(Dataset):
    """
    PyTorch Dataset for seismic waveform data.
    
    Handles loading and preprocessing of seismic waveforms with labels
    for training the CNN model.
    """
    
    def __init__(
        self,
        waveforms: np.ndarray,
        labels: Dict[str, np.ndarray],
        transform=None,
        augment: bool = False
    ):
        """
        Initialize seismic dataset.
        
        Args:
            waveforms: Array of shape (N, 3, L) where N is number of samples,
                      3 is number of components (Z, N, E), L is sequence length
            labels: Dictionary containing different label types:
                   - 'event_type': Event type classification labels
                   - 'detection': Binary detection labels
                   - 'magnitude': Magnitude regression labels
                   - 'confidence': Confidence labels
            transform: Optional transform to apply to waveforms
            augment: Whether to apply data augmentation
        """
        self.waveforms = waveforms
        self.labels = labels
        self.transform = transform
        self.augment = augment
        
        # Validate data consistency
        num_samples = len(waveforms)
        for label_type, label_data in labels.items():
            if len(label_data) != num_samples:
                raise ValueError(f"Mismatch in number of samples: waveforms={num_samples}, {label_type}={len(label_data)}")
    
    def __len__(self):
        return len(self.waveforms)
    
    def __getitem__(self, idx):
        waveform = self.waveforms[idx].copy()
        
        # Apply data augmentation if enabled
        if self.augment:
            waveform = self._apply_augmentation(waveform)
        
        # Apply transform if provided
        if self.transform:
            waveform = self.transform(waveform)
        
        # Convert to tensor
        waveform = torch.FloatTensor(waveform)
        
        # Prepare labels
        labels = {}
        for label_type, label_data in self.labels.items():
            if label_type == 'event_type':
                labels[label_type] = torch.LongTensor([label_data[idx]])[0]
            else:
                labels[label_type] = torch.FloatTensor([label_data[idx]])[0]
        
        return waveform, labels
    
    def _apply_augmentation(self, waveform: np.ndarray) -> np.ndarray:
        """Apply data augmentation techniques"""
        augmented = waveform.copy()
        
        # Random amplitude scaling (±20%)
        if np.random.random() < 0.5:
            scale_factor = np.random.uniform(0.8, 1.2)
            augmented *= scale_factor
        
        # Random noise addition
        if np.random.random() < 0.3:
            noise_level = np.random.uniform(0.01, 0.05)
            noise = np.random.randn(*augmented.shape) * noise_level
            augmented += noise
        
        # Random time shift (±10% of length)
        if np.random.random() < 0.4:
            shift_samples = int(np.random.uniform(-0.1, 0.1) * augmented.shape[1])
            if shift_samples != 0:
                augmented = np.roll(augmented, shift_samples, axis=1)
        
        # Random polarity flip
        if np.random.random() < 0.2:
            augmented *= -1
        
        return augmented


class SeismicCNNTrainer:
    """
    Trainer class for the seismic CNN model.
    
    Handles the complete training pipeline including data loading,
    model training, validation, and evaluation.
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize model
        self.model = SeismicCNN(
            input_length=3000,  # 30 seconds at 100 Hz
            num_channels=3,
            num_classes=len(SeismicEventType),
            dropout_rate=0.3,
            use_attention=True
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Initialize scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config.scheduler_step_size,
            gamma=config.scheduler_gamma
        )
        
        # Loss functions for multi-task learning
        self.classification_loss = nn.CrossEntropyLoss()
        self.detection_loss = nn.BCELoss()
        self.regression_loss = nn.MSELoss()
        
        # Training history
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'learning_rates': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def prepare_data(
        self,
        waveforms: np.ndarray,
        labels: Dict[str, np.ndarray],
        augment_training: bool = True
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare training and validation data loaders.
        
        Args:
            waveforms: Seismic waveform data
            labels: Labels for multi-task learning
            augment_training: Whether to apply augmentation to training data
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Create dataset
        dataset = SeismicDataset(waveforms, labels, augment=False)
        
        # Split into train and validation
        val_size = int(len(dataset) * self.config.validation_split)
        train_size = len(dataset) - val_size
        
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Apply augmentation only to training data
        if augment_training:
            train_dataset.dataset.augment = True
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,  # Set to 0 for compatibility
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")
        
        return train_loader, val_loader
    
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute multi-task loss.
        
        Args:
            outputs: Model outputs
            targets: Target labels
            
        Returns:
            Tuple of (total_loss, loss_components)
        """
        losses = {}
        
        # Classification loss
        if 'event_type' in outputs and 'event_type' in targets:
            losses['classification'] = self.classification_loss(
                outputs['event_type'], targets['event_type']
            )
        
        # Detection loss
        if 'detection' in outputs and 'detection' in targets:
            losses['detection'] = self.detection_loss(
                outputs['detection'].squeeze(), targets['detection']
            )
        
        # Magnitude regression loss
        if 'magnitude' in outputs and 'magnitude' in targets:
            # Only compute loss for samples with valid magnitude labels
            valid_mask = ~torch.isnan(targets['magnitude'])
            if valid_mask.any():
                losses['magnitude'] = self.regression_loss(
                    outputs['magnitude'].squeeze()[valid_mask],
                    targets['magnitude'][valid_mask]
                )
        
        # Confidence loss
        if 'confidence' in outputs and 'confidence' in targets:
            losses['confidence'] = self.regression_loss(
                outputs['confidence'].squeeze(), targets['confidence']
            )
        
        # Combine losses with weights
        loss_weights = {
            'classification': 1.0,
            'detection': 1.0,
            'magnitude': 0.5,
            'confidence': 0.5
        }
        
        total_loss = sum(
            loss_weights.get(name, 1.0) * loss
            for name, loss in losses.items()
        )
        
        # Convert to float for logging
        loss_components = {name: loss.item() for name, loss in losses.items()}
        
        return total_loss, loss_components
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for batch_idx, (waveforms, targets) in enumerate(train_loader):
            # Move data to device
            waveforms = waveforms.to(self.device)
            targets = {k: v.to(self.device) for k, v in targets.items()}
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(waveforms)
            
            # Compute loss
            loss, loss_components = self.compute_loss(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update statistics
            total_loss += loss.item()
            
            # Calculate accuracy for classification task
            if 'event_type' in outputs and 'event_type' in targets:
                _, predicted = torch.max(outputs['event_type'], 1)
                correct_predictions += (predicted == targets['event_type']).sum().item()
                total_samples += targets['event_type'].size(0)
            
            # Log progress
            if batch_idx % 10 == 0:
                logger.debug(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for waveforms, targets in val_loader:
                # Move data to device
                waveforms = waveforms.to(self.device)
                targets = {k: v.to(self.device) for k, v in targets.items()}
                
                # Forward pass
                outputs = self.model(waveforms)
                
                # Compute loss
                loss, _ = self.compute_loss(outputs, targets)
                total_loss += loss.item()
                
                # Calculate accuracy
                if 'event_type' in outputs and 'event_type' in targets:
                    _, predicted = torch.max(outputs['event_type'], 1)
                    correct_predictions += (predicted == targets['event_type']).sum().item()
                    total_samples += targets['event_type'].size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
        
        return avg_loss, accuracy
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader
    ) -> Dict[str, List[float]]:
        """
        Complete training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            
        Returns:
            Training history dictionary
        """
        logger.info("Starting training...")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.config.num_epochs):
            epoch_start_time = datetime.now()
            
            # Training phase
            train_loss, train_accuracy = self.train_epoch(train_loader)
            
            # Validation phase
            val_loss, val_accuracy = self.validate_epoch(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['train_accuracy'].append(train_accuracy)
            self.training_history['val_accuracy'].append(val_accuracy)
            self.training_history['learning_rates'].append(current_lr)
            
            # Check for best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                
                if self.config.save_best_model:
                    self.save_model(self.config.model_save_path, epoch, val_loss)
                    logger.info(f"Saved best model with validation loss: {val_loss:.4f}")
            else:
                self.patience_counter += 1
            
            # Log epoch results
            epoch_time = (datetime.now() - epoch_start_time).total_seconds()
            logger.info(
                f"Epoch {epoch+1}/{self.config.num_epochs} "
                f"({epoch_time:.1f}s) - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Train Acc: {train_accuracy:.3f}, "
                f"Val Acc: {val_accuracy:.3f}, "
                f"LR: {current_lr:.6f}"
            )
            
            # Early stopping
            if self.patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        logger.info("Training completed!")
        return self.training_history
    
    def save_model(self, path: str, epoch: int, val_loss: float):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'training_history': self.training_history,
            'config': self.config.__dict__
        }
        
        torch.save(checkpoint, path)
    
    def load_model(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.training_history = checkpoint.get('training_history', {})
        
        logger.info(f"Loaded model from {path}")
        return checkpoint
    
    def evaluate_model(
        self,
        test_loader: DataLoader,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate trained model on test data.
        
        Args:
            test_loader: Test data loader
            save_results: Whether to save evaluation results
            
        Returns:
            Dictionary containing evaluation metrics
        """
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_outputs = []
        
        with torch.no_grad():
            for waveforms, targets in test_loader:
                waveforms = waveforms.to(self.device)
                targets = {k: v.to(self.device) for k, v in targets.items()}
                
                outputs = self.model(waveforms)
                
                # Store predictions and targets
                if 'event_type' in outputs:
                    _, predicted = torch.max(outputs['event_type'], 1)
                    all_predictions.extend(predicted.cpu().numpy())
                    all_targets.extend(targets['event_type'].cpu().numpy())
                
                all_outputs.append({
                    k: v.cpu().numpy() for k, v in outputs.items()
                })
        
        # Calculate metrics
        results = {}
        
        if all_predictions and all_targets:
            # Classification metrics
            results['classification_report'] = classification_report(
                all_targets, all_predictions,
                target_names=[event_type.value for event_type in SeismicEventType],
                output_dict=True
            )
            
            results['confusion_matrix'] = confusion_matrix(all_targets, all_predictions)
            
            # Overall accuracy
            results['accuracy'] = np.mean(np.array(all_predictions) == np.array(all_targets))
        
        # Save results if requested
        if save_results:
            results_path = "evaluation_results.json"
            with open(results_path, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                json_results = {}
                for key, value in results.items():
                    if isinstance(value, np.ndarray):
                        json_results[key] = value.tolist()
                    else:
                        json_results[key] = value
                
                json.dump(json_results, f, indent=2)
            
            logger.info(f"Evaluation results saved to {results_path}")
        
        return results
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('Training History')
            
            # Loss plot
            axes[0, 0].plot(self.training_history['train_loss'], label='Train Loss')
            axes[0, 0].plot(self.training_history['val_loss'], label='Validation Loss')
            axes[0, 0].set_title('Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # Accuracy plot
            axes[0, 1].plot(self.training_history['train_accuracy'], label='Train Accuracy')
            axes[0, 1].plot(self.training_history['val_accuracy'], label='Validation Accuracy')
            axes[0, 1].set_title('Accuracy')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            # Learning rate plot
            axes[1, 0].plot(self.training_history['learning_rates'])
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True)
            
            # Loss difference plot
            if len(self.training_history['train_loss']) > 0:
                loss_diff = np.array(self.training_history['val_loss']) - np.array(self.training_history['train_loss'])
                axes[1, 1].plot(loss_diff)
                axes[1, 1].set_title('Validation - Training Loss')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Loss Difference')
                axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Training history plot saved to {save_path}")
            
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")


def create_synthetic_training_data(
    num_samples: int = 1000,
    sequence_length: int = 3000,
    sampling_rate: float = 100.0
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Create synthetic training data for demonstration purposes.
    
    Args:
        num_samples: Number of training samples to generate
        sequence_length: Length of each waveform sequence
        sampling_rate: Sampling rate in Hz
        
    Returns:
        Tuple of (waveforms, labels)
    """
    logger.info(f"Generating {num_samples} synthetic training samples...")
    
    waveforms = np.zeros((num_samples, 3, sequence_length))
    event_types = np.zeros(num_samples, dtype=int)
    detections = np.zeros(num_samples)
    magnitudes = np.full(num_samples, np.nan)
    confidences = np.zeros(num_samples)
    
    event_type_list = list(SeismicEventType)
    
    for i in range(num_samples):
        # Random event type
        event_type_idx = np.random.randint(0, len(event_type_list))
        event_type = event_type_list[event_type_idx]
        
        # Generate waveform based on event type
        if event_type == SeismicEventType.NOISE:
            # Pure noise
            waveforms[i] = np.random.randn(3, sequence_length) * 0.1
            detections[i] = 0.0
            confidences[i] = 0.1
        else:
            # Event + noise
            noise = np.random.randn(3, sequence_length) * 0.1
            
            # Event parameters based on type
            if event_type == SeismicEventType.EARTHQUAKE:
                amplitude = np.random.uniform(1.0, 3.0)
                frequency = np.random.uniform(3.0, 15.0)
                duration = np.random.uniform(5.0, 30.0)
                magnitude = np.random.uniform(2.0, 6.0)
            elif event_type == SeismicEventType.LONG_PERIOD:
                amplitude = np.random.uniform(0.5, 2.0)
                frequency = np.random.uniform(1.0, 5.0)
                duration = np.random.uniform(10.0, 60.0)
                magnitude = np.random.uniform(1.0, 4.0)
            elif event_type == SeismicEventType.EXPLOSION:
                amplitude = np.random.uniform(2.0, 4.0)
                frequency = np.random.uniform(8.0, 25.0)
                duration = np.random.uniform(1.0, 10.0)
                magnitude = np.random.uniform(1.5, 4.5)
            else:
                amplitude = np.random.uniform(0.5, 2.5)
                frequency = np.random.uniform(2.0, 12.0)
                duration = np.random.uniform(3.0, 45.0)
                magnitude = np.random.uniform(1.0, 5.0)
            
            # Generate event signal
            event_samples = int(duration * sampling_rate)
            event_start = np.random.randint(0, max(1, sequence_length - event_samples))
            
            t = np.arange(event_samples) / sampling_rate
            event_signal = amplitude * np.exp(-t / (duration / 3)) * np.sin(2 * np.pi * frequency * t)
            
            # Add to each component with different amplitudes
            waveforms[i, 0, event_start:event_start + event_samples] = event_signal  # Z
            waveforms[i, 1, event_start:event_start + event_samples] = event_signal * 0.8  # N
            waveforms[i, 2, event_start:event_start + event_samples] = event_signal * 0.6  # E
            
            # Add noise
            waveforms[i] += noise
            
            detections[i] = 1.0
            magnitudes[i] = magnitude
            confidences[i] = np.random.uniform(0.7, 0.95)
        
        event_types[i] = event_type_idx
    
    labels = {
        'event_type': event_types,
        'detection': detections,
        'magnitude': magnitudes,
        'confidence': confidences
    }
    
    logger.info("Synthetic training data generated successfully")
    return waveforms, labels


def main():
    """Main training function"""
    # Configuration
    config = TrainingConfig(
        batch_size=16,
        learning_rate=0.001,
        num_epochs=50,
        validation_split=0.2,
        early_stopping_patience=10,
        device="cpu",  # Change to "cuda" if GPU available
        save_best_model=True,
        model_save_path="seismic_cnn_model.pth"
    )
    
    # Generate synthetic training data
    waveforms, labels = create_synthetic_training_data(
        num_samples=500,  # Small dataset for demo
        sequence_length=3000,
        sampling_rate=100.0
    )
    
    # Initialize trainer
    trainer = SeismicCNNTrainer(config)
    
    # Prepare data
    train_loader, val_loader = trainer.prepare_data(
        waveforms, labels, augment_training=True
    )
    
    # Train model
    history = trainer.train(train_loader, val_loader)
    
    # Plot training history
    trainer.plot_training_history("training_history.png")
    
    # Evaluate model
    results = trainer.evaluate_model(val_loader)
    
    print("Training completed!")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"Final validation accuracy: {results.get('accuracy', 'N/A'):.3f}")


if __name__ == "__main__":
    main()
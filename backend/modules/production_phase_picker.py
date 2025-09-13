"""
Production-Grade Phase Picking Algorithms
Advanced algorithms for P and S wave detection with real seismic data
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import warnings

try:
    from scipy.signal import butter, filtfilt, hilbert, find_peaks
    from scipy.ndimage import gaussian_filter1d
    from scipy.stats import kurtosis, skew
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class PhasePickConfig:
    """Configuration for phase picking algorithms"""
    sample_rate: float = 100.0
    
    # STA/LTA parameters
    sta_window: float = 0.5  # seconds
    lta_window: float = 10.0  # seconds
    trigger_on: float = 3.0
    trigger_off: float = 1.5
    
    # Frequency filtering
    p_freq_range: Tuple[float, float] = (1.0, 15.0)  # Hz
    s_freq_range: Tuple[float, float] = (0.5, 10.0)  # Hz
    
    # Advanced parameters
    min_phase_separation: float = 1.0  # seconds
    max_picks_per_trace: int = 10
    quality_threshold: float = 0.6
    
    # Neural network parameters
    use_neural_picker: bool = True
    model_confidence_threshold: float = 0.7


class STALTAPhaseDetector:
    """Classic STA/LTA phase detection algorithm"""
    
    def __init__(self, config: PhasePickConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def detect_phases(self, waveform: np.ndarray) -> Dict[str, List[int]]:
        """Detect P and S phases using STA/LTA algorithm"""
        
        if not SCIPY_AVAILABLE:
            self.logger.warning("SciPy not available - using simplified detection")
            return self._simple_detection(waveform)
        
        # Filter for P waves
        p_filtered = self._bandpass_filter(waveform, self.config.p_freq_range)
        p_picks = self._sta_lta_picker(p_filtered, phase_type='P')
        
        # Filter for S waves
        s_filtered = self._bandpass_filter(waveform, self.config.s_freq_range)
        s_picks = self._sta_lta_picker(s_filtered, phase_type='S')
        
        # Remove picks that are too close together
        p_picks = self._remove_close_picks(p_picks)
        s_picks = self._remove_close_picks(s_picks)
        
        # Ensure S picks come after P picks
        s_picks = self._filter_s_after_p(p_picks, s_picks)
        
        return {
            'p_picks': p_picks[:self.config.max_picks_per_trace],
            's_picks': s_picks[:self.config.max_picks_per_trace]
        }
    
    def _bandpass_filter(self, data: np.ndarray, freq_range: Tuple[float, float]) -> np.ndarray:
        """Apply bandpass filter to waveform"""
        
        nyquist = self.config.sample_rate / 2
        low = freq_range[0] / nyquist
        high = freq_range[1] / nyquist
        
        # Ensure frequencies are within valid range
        low = max(0.01, min(low, 0.99))
        high = max(low + 0.01, min(high, 0.99))
        
        b, a = butter(4, [low, high], btype='band')
        return filtfilt(b, a, data)
    
    def _sta_lta_picker(self, data: np.ndarray, phase_type: str = 'P') -> List[int]:
        """STA/LTA phase picker implementation"""
        
        sta_samples = int(self.config.sta_window * self.config.sample_rate)
        lta_samples = int(self.config.lta_window * self.config.sample_rate)
        
        # Calculate characteristic function (energy)
        char_func = data ** 2
        
        # Calculate STA and LTA
        sta = np.zeros(len(char_func))
        lta = np.zeros(len(char_func))
        
        # Initialize LTA
        for i in range(lta_samples, len(char_func)):
            lta[i] = np.mean(char_func[i-lta_samples:i])
        
        # Initialize STA
        for i in range(lta_samples + sta_samples, len(char_func)):
            sta[i] = np.mean(char_func[i-sta_samples:i])
        
        # Calculate STA/LTA ratio
        ratio = np.zeros(len(char_func))
        valid_indices = lta > 0
        ratio[valid_indices] = sta[valid_indices] / lta[valid_indices]
        
        # Find triggers
        picks = []
        triggered = False
        
        for i in range(lta_samples + sta_samples, len(ratio)):
            if not triggered and ratio[i] > self.config.trigger_on:
                # Trigger on - find precise pick time
                pick_time = self._refine_pick_time(data, i, phase_type)
                if pick_time is not None:
                    picks.append(pick_time)
                triggered = True
            elif triggered and ratio[i] < self.config.trigger_off:
                triggered = False
        
        return picks
    
    def _refine_pick_time(self, data: np.ndarray, trigger_index: int, phase_type: str) -> Optional[int]:
        """Refine pick time using additional criteria"""
        
        # Search window around trigger
        search_window = int(2.0 * self.config.sample_rate)  # 2 seconds
        start_idx = max(0, trigger_index - search_window)
        end_idx = min(len(data), trigger_index + search_window // 2)
        
        search_data = data[start_idx:end_idx]
        
        if len(search_data) < 10:
            return None
        
        # Method 1: Maximum gradient (for P waves)
        if phase_type == 'P':
            gradient = np.gradient(search_data)
            max_grad_idx = np.argmax(np.abs(gradient))
            refined_pick = start_idx + max_grad_idx
        
        # Method 2: Maximum amplitude change (for S waves)
        else:
            # Use envelope for S waves
            envelope = np.abs(hilbert(search_data))
            envelope_grad = np.gradient(envelope)
            max_grad_idx = np.argmax(envelope_grad)
            refined_pick = start_idx + max_grad_idx
        
        # Quality check
        if self._assess_pick_quality(data, refined_pick, phase_type) > self.config.quality_threshold:
            return refined_pick
        
        return None
    
    def _assess_pick_quality(self, data: np.ndarray, pick_index: int, phase_type: str) -> float:
        """Assess quality of a phase pick"""
        
        # Define windows before and after pick
        before_window = int(5.0 * self.config.sample_rate)  # 5 seconds
        after_window = int(5.0 * self.config.sample_rate)
        
        before_start = max(0, pick_index - before_window)
        after_end = min(len(data), pick_index + after_window)
        
        if pick_index <= before_start or pick_index >= after_end:
            return 0.0
        
        before_data = data[before_start:pick_index]
        after_data = data[pick_index:after_end]
        
        if len(before_data) < 10 or len(after_data) < 10:
            return 0.0
        
        # Calculate quality metrics
        quality_score = 0.0
        
        # 1. Signal-to-noise ratio
        noise_power = np.var(before_data)
        signal_power = np.var(after_data)
        
        if noise_power > 0:
            snr = signal_power / noise_power
            quality_score += min(snr / 10.0, 0.3)  # Max 0.3 points
        
        # 2. Amplitude increase
        before_rms = np.sqrt(np.mean(before_data**2))
        after_rms = np.sqrt(np.mean(after_data**2))
        
        if before_rms > 0:
            amplitude_ratio = after_rms / before_rms
            quality_score += min(amplitude_ratio / 5.0, 0.3)  # Max 0.3 points
        
        # 3. Frequency content change (for P vs S discrimination)
        if SCIPY_AVAILABLE:
            before_freq = self._dominant_frequency(before_data)
            after_freq = self._dominant_frequency(after_data)
            
            if phase_type == 'P' and after_freq > before_freq:
                quality_score += 0.2
            elif phase_type == 'S' and after_freq < before_freq:
                quality_score += 0.2
        
        # 4. Sharpness of onset
        gradient = np.gradient(data[max(0, pick_index-10):pick_index+10])
        max_gradient = np.max(np.abs(gradient))
        quality_score += min(max_gradient / np.std(data), 0.2)  # Max 0.2 points
        
        return min(quality_score, 1.0)
    
    def _dominant_frequency(self, data: np.ndarray) -> float:
        """Calculate dominant frequency of signal"""
        
        fft_data = np.fft.fft(data)
        freqs = np.fft.fftfreq(len(data), 1.0 / self.config.sample_rate)
        
        # Only consider positive frequencies
        positive_freqs = freqs[:len(freqs)//2]
        positive_fft = np.abs(fft_data[:len(fft_data)//2])
        
        if len(positive_fft) == 0:
            return 1.0
        
        dominant_idx = np.argmax(positive_fft)
        return abs(positive_freqs[dominant_idx])
    
    def _remove_close_picks(self, picks: List[int]) -> List[int]:
        """Remove picks that are too close together"""
        
        if len(picks) <= 1:
            return picks
        
        min_separation_samples = int(self.config.min_phase_separation * self.config.sample_rate)
        filtered_picks = [picks[0]]
        
        for pick in picks[1:]:
            if pick - filtered_picks[-1] >= min_separation_samples:
                filtered_picks.append(pick)
        
        return filtered_picks
    
    def _filter_s_after_p(self, p_picks: List[int], s_picks: List[int]) -> List[int]:
        """Ensure S picks come after P picks"""
        
        if not p_picks:
            return s_picks
        
        filtered_s_picks = []
        
        for s_pick in s_picks:
            # Find closest preceding P pick
            preceding_p_picks = [p for p in p_picks if p < s_pick]
            
            if preceding_p_picks:
                closest_p = max(preceding_p_picks)
                # Ensure minimum separation between P and S
                min_ps_separation = int(1.0 * self.config.sample_rate)  # 1 second
                
                if s_pick - closest_p >= min_ps_separation:
                    filtered_s_picks.append(s_pick)
        
        return filtered_s_picks
    
    def _simple_detection(self, waveform: np.ndarray) -> Dict[str, List[int]]:
        """Simplified detection when SciPy is not available"""
        
        # Simple energy-based detection
        energy = waveform ** 2
        
        # Smooth energy
        window_size = int(1.0 * self.config.sample_rate)
        smoothed_energy = np.convolve(energy, np.ones(window_size)/window_size, mode='same')
        
        # Find peaks
        threshold = np.mean(smoothed_energy) + 3 * np.std(smoothed_energy)
        
        picks = []
        for i in range(1, len(smoothed_energy) - 1):
            if (smoothed_energy[i] > threshold and 
                smoothed_energy[i] > smoothed_energy[i-1] and 
                smoothed_energy[i] > smoothed_energy[i+1]):
                picks.append(i)
        
        # Assume first half are P picks, second half are S picks
        mid_point = len(picks) // 2
        
        return {
            'p_picks': picks[:mid_point],
            's_picks': picks[mid_point:]
        }


class NeuralPhaseDetector:
    """Neural network-based phase detection"""
    
    def __init__(self, config: PhasePickConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model = None
        
        if TORCH_AVAILABLE:
            self._initialize_model()
    
    def _initialize_model(self):
        """Initialize neural network model for phase detection"""
        
        class PhasePickerNet(nn.Module):
            def __init__(self, input_size=6000):  # 60 seconds at 100 Hz
                super().__init__()
                
                # Convolutional layers for feature extraction
                self.conv_layers = nn.Sequential(
                    nn.Conv1d(1, 32, kernel_size=7, padding=3),
                    nn.BatchNorm1d(32),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    
                    nn.Conv1d(32, 64, kernel_size=5, padding=2),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    
                    nn.Conv1d(64, 128, kernel_size=3, padding=1),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                )
                
                # Calculate size after convolutions
                conv_output_size = input_size // 8  # Three MaxPool1d(2) layers
                
                # LSTM for temporal modeling
                self.lstm = nn.LSTM(128, 64, batch_first=True, bidirectional=True)
                
                # Output heads
                self.p_head = nn.Sequential(
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, 1),
                    nn.Sigmoid()
                )
                
                self.s_head = nn.Sequential(
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, 1),
                    nn.Sigmoid()
                )
            
            def forward(self, x):
                # x shape: (batch, 1, time)
                
                # Convolutional feature extraction
                conv_out = self.conv_layers(x)  # (batch, 128, time/8)
                
                # Reshape for LSTM
                conv_out = conv_out.transpose(1, 2)  # (batch, time/8, 128)
                
                # LSTM processing
                lstm_out, _ = self.lstm(conv_out)  # (batch, time/8, 128)
                
                # Phase predictions
                p_prob = self.p_head(lstm_out)  # (batch, time/8, 1)
                s_prob = self.s_head(lstm_out)  # (batch, time/8, 1)
                
                return {
                    'p_probability': p_prob.squeeze(-1),
                    's_probability': s_prob.squeeze(-1)
                }
        
        self.model = PhasePickerNet()
        self.model.eval()  # Set to evaluation mode
        
        self.logger.info("Neural phase picker model initialized")
    
    def detect_phases(self, waveform: np.ndarray) -> Dict[str, List[int]]:
        """Detect phases using neural network"""
        
        if not TORCH_AVAILABLE or self.model is None:
            self.logger.warning("Neural network not available - using fallback")
            return self._fallback_detection(waveform)
        
        try:
            # Prepare input
            input_tensor = torch.FloatTensor(waveform).unsqueeze(0).unsqueeze(0)  # (1, 1, time)
            
            with torch.no_grad():
                predictions = self.model(input_tensor)
            
            # Extract predictions
            p_probs = predictions['p_probability'].squeeze().numpy()
            s_probs = predictions['s_probability'].squeeze().numpy()
            
            # Find peaks above threshold
            p_picks = self._find_probability_peaks(p_probs, 'P')
            s_picks = self._find_probability_peaks(s_probs, 'S')
            
            # Scale picks back to original time resolution
            scale_factor = len(waveform) / len(p_probs)
            p_picks = [int(pick * scale_factor) for pick in p_picks]
            s_picks = [int(pick * scale_factor) for pick in s_picks]
            
            return {
                'p_picks': p_picks,
                's_picks': s_picks,
                'p_probabilities': p_probs.tolist(),
                's_probabilities': s_probs.tolist()
            }
            
        except Exception as e:
            self.logger.error(f"Neural phase detection failed: {e}")
            return self._fallback_detection(waveform)
    
    def _find_probability_peaks(self, probabilities: np.ndarray, phase_type: str) -> List[int]:
        """Find peaks in probability time series"""
        
        if not SCIPY_AVAILABLE:
            # Simple peak finding
            peaks = []
            threshold = self.config.model_confidence_threshold
            
            for i in range(1, len(probabilities) - 1):
                if (probabilities[i] > threshold and
                    probabilities[i] > probabilities[i-1] and
                    probabilities[i] > probabilities[i+1]):
                    peaks.append(i)
            
            return peaks
        
        # Use scipy for more sophisticated peak finding
        peaks, properties = find_peaks(
            probabilities,
            height=self.config.model_confidence_threshold,
            distance=int(self.config.min_phase_separation * self.config.sample_rate / 8)  # Account for downsampling
        )
        
        return peaks.tolist()
    
    def _fallback_detection(self, waveform: np.ndarray) -> Dict[str, List[int]]:
        """Fallback detection when neural network fails"""
        
        # Use STA/LTA as fallback
        sta_lta_detector = STALTAPhaseDetector(self.config)
        return sta_lta_detector.detect_phases(waveform)


class HybridPhaseDetector:
    """Hybrid phase detector combining multiple algorithms"""
    
    def __init__(self, config: PhasePickConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize individual detectors
        self.sta_lta_detector = STALTAPhaseDetector(config)
        self.neural_detector = NeuralPhaseDetector(config) if config.use_neural_picker else None
    
    def detect_phases(self, waveform: np.ndarray) -> Dict:
        """Detect phases using hybrid approach"""
        
        results = {
            'p_picks': [],
            's_picks': [],
            'confidence_scores': {},
            'detection_methods': []
        }
        
        # STA/LTA detection
        sta_lta_results = self.sta_lta_detector.detect_phases(waveform)
        
        # Neural network detection
        neural_results = None
        if self.neural_detector:
            neural_results = self.neural_detector.detect_phases(waveform)
        
        # Combine results
        if neural_results and len(neural_results.get('p_picks', [])) > 0:
            # Use neural network results as primary
            results['p_picks'] = neural_results['p_picks']
            results['s_picks'] = neural_results['s_picks']
            results['detection_methods'].append('neural_network')
            
            # Add confidence scores if available
            if 'p_probabilities' in neural_results:
                results['p_probabilities'] = neural_results['p_probabilities']
                results['s_probabilities'] = neural_results['s_probabilities']
            
            # Validate with STA/LTA
            validated_picks = self._validate_picks_with_sta_lta(
                results, sta_lta_results, waveform
            )
            results.update(validated_picks)
            results['detection_methods'].append('sta_lta_validation')
            
        else:
            # Fall back to STA/LTA
            results['p_picks'] = sta_lta_results['p_picks']
            results['s_picks'] = sta_lta_results['s_picks']
            results['detection_methods'].append('sta_lta_primary')
        
        # Calculate confidence scores
        results['confidence_scores'] = self._calculate_confidence_scores(results, waveform)
        
        return results
    
    def _validate_picks_with_sta_lta(self, neural_results: Dict, 
                                   sta_lta_results: Dict, 
                                   waveform: np.ndarray) -> Dict:
        """Validate neural network picks with STA/LTA"""
        
        validated_results = {
            'p_picks': [],
            's_picks': []
        }
        
        # Validation window (±2 seconds)
        validation_window = int(2.0 * self.config.sample_rate)
        
        # Validate P picks
        for neural_pick in neural_results.get('p_picks', []):
            is_validated = False
            
            for sta_lta_pick in sta_lta_results.get('p_picks', []):
                if abs(neural_pick - sta_lta_pick) <= validation_window:
                    validated_results['p_picks'].append(neural_pick)
                    is_validated = True
                    break
            
            # If not validated by STA/LTA, check if it's a high-confidence neural pick
            if not is_validated and 'p_probabilities' in neural_results:
                pick_idx = neural_pick // (len(waveform) // len(neural_results['p_probabilities']))
                pick_idx = min(pick_idx, len(neural_results['p_probabilities']) - 1)
                
                if neural_results['p_probabilities'][pick_idx] > 0.9:  # Very high confidence
                    validated_results['p_picks'].append(neural_pick)
        
        # Validate S picks similarly
        for neural_pick in neural_results.get('s_picks', []):
            is_validated = False
            
            for sta_lta_pick in sta_lta_results.get('s_picks', []):
                if abs(neural_pick - sta_lta_pick) <= validation_window:
                    validated_results['s_picks'].append(neural_pick)
                    is_validated = True
                    break
            
            if not is_validated and 's_probabilities' in neural_results:
                pick_idx = neural_pick // (len(waveform) // len(neural_results['s_probabilities']))
                pick_idx = min(pick_idx, len(neural_results['s_probabilities']) - 1)
                
                if neural_results['s_probabilities'][pick_idx] > 0.9:
                    validated_results['s_picks'].append(neural_pick)
        
        return validated_results
    
    def _calculate_confidence_scores(self, results: Dict, waveform: np.ndarray) -> Dict:
        """Calculate confidence scores for picks"""
        
        confidence_scores = {
            'p_confidence': 0.0,
            's_confidence': 0.0
        }
        
        # P wave confidence
        if results['p_picks']:
            p_confidences = []
            for pick in results['p_picks']:
                quality = self.sta_lta_detector._assess_pick_quality(waveform, pick, 'P')
                p_confidences.append(quality)
            
            confidence_scores['p_confidence'] = np.mean(p_confidences) if p_confidences else 0.0
        
        # S wave confidence
        if results['s_picks']:
            s_confidences = []
            for pick in results['s_picks']:
                quality = self.sta_lta_detector._assess_pick_quality(waveform, pick, 'S')
                s_confidences.append(quality)
            
            confidence_scores['s_confidence'] = np.mean(s_confidences) if s_confidences else 0.0
        
        return confidence_scores


def create_production_phase_picker(config: Optional[PhasePickConfig] = None) -> HybridPhaseDetector:
    """Create production-grade phase picker"""
    
    if config is None:
        config = PhasePickConfig()
    
    return HybridPhaseDetector(config)
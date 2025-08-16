"""
Speech processing and timing analysis
"""

import logging
from typing import Dict, List, Tuple
import numpy as np
import librosa
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

logger = logging.getLogger(__name__)

class SpeechProcessor:
    """
    Process speech timing and rate features
    """
    
    def __init__(self, 
                 energy_threshold: float = 0.01,
                 min_pause_duration: float = 0.1,
                 max_pause_duration: float = 2.0):
        """
        Initialize speech processor
        
        Args:
            energy_threshold: Threshold for speech detection
            min_pause_duration: Minimum pause duration (seconds)
            max_pause_duration: Maximum pause duration (seconds)
        """
        self.energy_threshold = energy_threshold
        self.min_pause_duration = min_pause_duration
        self.max_pause_duration = max_pause_duration
    
    def estimate_speech_rate(self, audio_data: np.ndarray, sr: int) -> float:
        """
        Estimate speech rate in words per minute
        
        Args:
            audio_data: Audio signal
            sr: Sample rate
            
        Returns:
            Estimated speech rate (words per minute)
        """
        try:
            # Calculate energy envelope
            energy = librosa.feature.rms(y=audio_data)[0]
            
            # Smooth energy envelope
            energy_smooth = gaussian_filter1d(energy, sigma=2)
            
            # Find speech segments
            speech_segments = energy_smooth > self.energy_threshold
            
            # Calculate total speech duration
            speech_duration = np.sum(speech_segments) * librosa.get_duration(y=audio_data, sr=sr) / len(energy_smooth)
            
            # Estimate words based on speech duration
            # Average speaking rate is ~150 words per minute
            # Average word duration is ~0.4 seconds
            avg_word_duration = 0.4  # seconds per word
            estimated_words = speech_duration / avg_word_duration
            
            # Calculate words per minute
            total_duration = len(audio_data) / sr
            speech_rate_wpm = (estimated_words / total_duration) * 60
            
            return speech_rate_wpm
            
        except Exception as e:
            logger.warning(f"Error estimating speech rate: {e}")
            return 150.0  # Default average rate
    
    def detect_pauses(self, audio_data: np.ndarray, sr: int) -> Dict:
        """
        Detect pauses in speech
        
        Args:
            audio_data: Audio signal
            sr: Sample rate
            
        Returns:
            Dictionary containing pause features
        """
        try:
            # Calculate energy envelope
            energy = librosa.feature.rms(y=audio_data)[0]
            
            # Smooth energy envelope
            energy_smooth = gaussian_filter1d(energy, sigma=3)
            
            # Find pause segments (low energy)
            pause_segments = energy_smooth < self.energy_threshold
            
            # Find pause boundaries
            pause_starts = []
            pause_ends = []
            
            in_pause = False
            for i, is_pause in enumerate(pause_segments):
                if is_pause and not in_pause:
                    pause_starts.append(i)
                    in_pause = True
                elif not is_pause and in_pause:
                    pause_ends.append(i)
                    in_pause = False
            
            # Handle case where audio ends in pause
            if in_pause:
                pause_ends.append(len(pause_segments) - 1)
            
            # Calculate pause durations
            pause_durations = []
            for start, end in zip(pause_starts, pause_ends):
                duration = (end - start) * librosa.get_duration(y=audio_data, sr=sr) / len(energy_smooth)
                
                # Filter pauses by duration
                if self.min_pause_duration <= duration <= self.max_pause_duration:
                    pause_durations.append(duration)
            
            # Calculate pause statistics
            if pause_durations:
                pause_count = len(pause_durations)
                total_pause_duration = np.sum(pause_durations)
                avg_pause_duration = np.mean(pause_durations)
                pause_std = np.std(pause_durations)
                
                # Calculate pause ratio
                total_duration = len(audio_data) / sr
                pause_ratio = total_pause_duration / total_duration
                
            else:
                pause_count = 0
                total_pause_duration = 0
                avg_pause_duration = 0
                pause_std = 0
                pause_ratio = 0
            
            return {
                'pause_count': pause_count,
                'total_pause_duration': total_pause_duration,
                'avg_pause_duration': avg_pause_duration,
                'pause_std': pause_std,
                'pause_ratio': pause_ratio,
                'pause_durations': pause_durations
            }
            
        except Exception as e:
            logger.warning(f"Error detecting pauses: {e}")
            return {
                'pause_count': 0,
                'total_pause_duration': 0,
                'avg_pause_duration': 0,
                'pause_std': 0,
                'pause_ratio': 0,
                'pause_durations': []
            }
    
    def calculate_rate_variability(self, audio_data: np.ndarray, sr: int) -> Dict:
        """
        Calculate speech rate variability over time
        
        Args:
            audio_data: Audio signal
            sr: Sample rate
            
        Returns:
            Dictionary containing rate variability features
        """
        try:
            # Calculate energy envelope
            energy = librosa.feature.rms(y=audio_data)[0]
            
            # Smooth energy envelope
            energy_smooth = gaussian_filter1d(energy, sigma=2)
            
            # Find speech segments
            speech_segments = energy_smooth > self.energy_threshold
            
            # Calculate local speech rates using sliding windows
            window_size = int(5 * sr / 512)  # 5-second windows
            hop_size = int(1 * sr / 512)     # 1-second hop
            
            local_rates = []
            
            for i in range(0, len(energy_smooth) - window_size, hop_size):
                window = speech_segments[i:i + window_size]
                
                # Calculate speech duration in window
                speech_duration = np.sum(window) * window_size * librosa.get_duration(y=audio_data, sr=sr) / len(energy_smooth)
                
                # Estimate words in window
                avg_word_duration = 0.4
                estimated_words = speech_duration / avg_word_duration
                
                # Calculate local rate (words per minute)
                window_duration = window_size * librosa.get_duration(y=audio_data, sr=sr) / len(energy_smooth)
                local_rate = (estimated_words / window_duration) * 60
                
                local_rates.append(local_rate)
            
            if local_rates:
                rate_std = np.std(local_rates)
                rate_mean = np.mean(local_rates)
                rate_cv = rate_std / max(rate_mean, 1e-8)  # Coefficient of variation
                rate_range = np.max(local_rates) - np.min(local_rates)
            else:
                rate_std = 0
                rate_mean = 0
                rate_cv = 0
                rate_range = 0
            
            return {
                'std': rate_std,
                'mean': rate_mean,
                'cv': rate_cv,
                'range': rate_range,
                'local_rates': local_rates
            }
            
        except Exception as e:
            logger.warning(f"Error calculating rate variability: {e}")
            return {
                'std': 0,
                'mean': 0,
                'cv': 0,
                'range': 0,
                'local_rates': []
            }
    
    def detect_speech_onsets(self, audio_data: np.ndarray, sr: int) -> List[float]:
        """
        Detect speech onset times
        
        Args:
            audio_data: Audio signal
            sr: Sample rate
            
        Returns:
            List of onset times in seconds
        """
        try:
            # Calculate onset strength
            onset_env = librosa.onset.onset_strength(y=audio_data, sr=sr)
            
            # Detect onset times
            onset_times = librosa.onset.onset_detect(
                onset_envelope=onset_env,
                sr=sr,
                units='time',
                threshold=0.1
            )
            
            return onset_times.tolist()
            
        except Exception as e:
            logger.warning(f"Error detecting speech onsets: {e}")
            return []
    
    def analyze_speech_rhythm(self, audio_data: np.ndarray, sr: int) -> Dict:
        """
        Analyze speech rhythm and timing patterns
        
        Args:
            audio_data: Audio signal
            sr: Sample rate
            
        Returns:
            Dictionary containing rhythm features
        """
        try:
            # Detect speech onsets
            onset_times = self.detect_speech_onsets(audio_data, sr)
            
            if len(onset_times) < 2:
                return {
                    'rhythm_regularity': 0,
                    'onset_intervals': [],
                    'interval_std': 0,
                    'interval_cv': 0
                }
            
            # Calculate intervals between onsets
            onset_intervals = np.diff(onset_times)
            
            # Calculate rhythm regularity
            interval_mean = np.mean(onset_intervals)
            interval_std = np.std(onset_intervals)
            interval_cv = interval_std / max(interval_mean, 1e-8)
            
            # Rhythm regularity (inverse of coefficient of variation)
            rhythm_regularity = 1.0 / max(interval_cv, 1e-8)
            
            return {
                'rhythm_regularity': rhythm_regularity,
                'onset_intervals': onset_intervals.tolist(),
                'interval_std': interval_std,
                'interval_cv': interval_cv,
                'interval_mean': interval_mean
            }
            
        except Exception as e:
            logger.warning(f"Error analyzing speech rhythm: {e}")
            return {
                'rhythm_regularity': 0,
                'onset_intervals': [],
                'interval_std': 0,
                'interval_cv': 0,
                'interval_mean': 0
            }
    
    def detect_filled_pauses(self, audio_data: np.ndarray, sr: int) -> Dict:
        """
        Detect filled pauses (um, uh, etc.) using spectral features
        
        Args:
            audio_data: Audio signal
            sr: Sample rate
            
        Returns:
            Dictionary containing filled pause features
        """
        try:
            # Extract MFCCs
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
            
            # Calculate MFCC statistics
            mfcc_mean = np.mean(mfccs, axis=1)
            mfcc_std = np.std(mfccs, axis=1)
            
            # Simple heuristic for filled pause detection
            # Filled pauses often have low spectral energy and specific MFCC patterns
            energy = librosa.feature.rms(y=audio_data)[0]
            low_energy_segments = energy < self.energy_threshold * 0.5
            
            # Count potential filled pause segments
            filled_pause_count = np.sum(low_energy_segments)
            
            # Calculate filled pause ratio
            total_frames = len(energy)
            filled_pause_ratio = filled_pause_count / max(total_frames, 1)
            
            return {
                'filled_pause_count': filled_pause_count,
                'filled_pause_ratio': filled_pause_ratio,
                'mfcc_mean': mfcc_mean.tolist(),
                'mfcc_std': mfcc_std.tolist()
            }
            
        except Exception as e:
            logger.warning(f"Error detecting filled pauses: {e}")
            return {
                'filled_pause_count': 0,
                'filled_pause_ratio': 0,
                'mfcc_mean': [],
                'mfcc_std': []
            }
    
    def extract_all_speech_features(self, audio_data: np.ndarray, sr: int) -> Dict:
        """
        Extract all speech timing features
        
        Args:
            audio_data: Audio signal
            sr: Sample rate
            
        Returns:
            Dictionary containing all speech features
        """
        features = {}
        
        # Extract all speech features
        features['speech_rate'] = self.estimate_speech_rate(audio_data, sr)
        features.update(self.detect_pauses(audio_data, sr))
        features.update(self.calculate_rate_variability(audio_data, sr))
        features.update(self.analyze_speech_rhythm(audio_data, sr))
        features.update(self.detect_filled_pauses(audio_data, sr))
        
        return features

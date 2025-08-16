"""
Vocal feature extraction using Praat
"""

import logging
from typing import Dict
import numpy as np
import parselmouth
from parselmouth.praat import call

logger = logging.getLogger(__name__)

class VocalFeatureExtractor:
    """
    Extract vocal features (pitch, jitter, shimmer) using Praat
    """
    
    def __init__(self):
        """Initialize the vocal feature extractor"""
        pass
    
    def extract_pitch_features(self, sound: parselmouth.Sound) -> Dict:
        """
        Extract pitch (fundamental frequency) features
        
        Args:
            sound: Parselmouth Sound object
            
        Returns:
            Dictionary containing pitch features
        """
        try:
            # Extract pitch
            pitch = call(sound, "To Pitch", 0.0, 75, 600)
            
            # Get pitch statistics
            pitch_mean = call(pitch, "Get mean", 0, 0, "Hertz")
            pitch_std = call(pitch, "Get standard deviation", 0, 0, "Hertz")
            pitch_min = call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic")
            pitch_max = call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic")
            pitch_range = pitch_max - pitch_min
            
            # Pitch quantiles
            pitch_q25 = call(pitch, "Get quantile", 0, 0, 0.25, "Hertz")
            pitch_q75 = call(pitch, "Get quantile", 0, 0, 0.75, "Hertz")
            pitch_iqr = pitch_q75 - pitch_q25
            
            # Pitch slope (change over time)
            pitch_slope = self._calculate_pitch_slope(pitch)
            
            return {
                'pitch_mean': pitch_mean,
                'pitch_std': pitch_std,
                'pitch_min': pitch_min,
                'pitch_max': pitch_max,
                'pitch_range': pitch_range,
                'pitch_q25': pitch_q25,
                'pitch_q75': pitch_q75,
                'pitch_iqr': pitch_iqr,
                'pitch_slope': pitch_slope,
                'pitch_cv': pitch_std / max(pitch_mean, 1e-8)  # Coefficient of variation
            }
            
        except Exception as e:
            logger.warning(f"Error extracting pitch features: {e}")
            return {
                'pitch_mean': 0,
                'pitch_std': 0,
                'pitch_min': 0,
                'pitch_max': 0,
                'pitch_range': 0,
                'pitch_q25': 0,
                'pitch_q75': 0,
                'pitch_iqr': 0,
                'pitch_slope': 0,
                'pitch_cv': 0
            }
    
    def extract_jitter_features(self, sound: parselmouth.Sound) -> Dict:
        """
        Extract jitter (frequency perturbation) features
        
        Args:
            sound: Parselmouth Sound object
            
        Returns:
            Dictionary containing jitter features
        """
        try:
            # Extract point process (glottal pulses)
            point_process = call(sound, "To PointProcess (periodic, cc)...", 75, 600)
            
            # Calculate jitter measures
            jitter_local = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            jitter_rap = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
            jitter_ppq5 = call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
            jitter_ddp = call(point_process, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
            
            # Additional jitter measures
            jitter_abs = call(point_process, "Get jitter (absolute)", 0, 0, 0.0001, 0.02, 1.3)
            jitter_sma = call(point_process, "Get jitter (sma)", 0, 0, 0.0001, 0.02, 1.3)
            
            return {
                'jitter_local': jitter_local,
                'jitter_rap': jitter_rap,
                'jitter_ppq5': jitter_ppq5,
                'jitter_ddp': jitter_ddp,
                'jitter_abs': jitter_abs,
                'jitter_sma': jitter_sma
            }
            
        except Exception as e:
            logger.warning(f"Error extracting jitter features: {e}")
            return {
                'jitter_local': 0,
                'jitter_rap': 0,
                'jitter_ppq5': 0,
                'jitter_ddp': 0,
                'jitter_abs': 0,
                'jitter_sma': 0
            }
    
    def extract_shimmer_features(self, sound: parselmouth.Sound) -> Dict:
        """
        Extract shimmer (amplitude perturbation) features
        
        Args:
            sound: Parselmouth Sound object
            
        Returns:
            Dictionary containing shimmer features
        """
        try:
            # Extract point process
            point_process = call(sound, "To PointProcess (periodic, cc)...", 75, 600)
            
            # Calculate shimmer measures
            shimmer_local = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            shimmer_apq3 = call([sound, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            shimmer_apq5 = call([sound, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            shimmer_apq11 = call([sound, point_process], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            shimmer_dda = call([sound, point_process], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            
            # Additional shimmer measures
            shimmer_abs = call([sound, point_process], "Get shimmer (absolute)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            
            return {
                'shimmer_local': shimmer_local,
                'shimmer_apq3': shimmer_apq3,
                'shimmer_apq5': shimmer_apq5,
                'shimmer_apq11': shimmer_apq11,
                'shimmer_dda': shimmer_dda,
                'shimmer_abs': shimmer_abs
            }
            
        except Exception as e:
            logger.warning(f"Error extracting shimmer features: {e}")
            return {
                'shimmer_local': 0,
                'shimmer_apq3': 0,
                'shimmer_apq5': 0,
                'shimmer_apq11': 0,
                'shimmer_dda': 0,
                'shimmer_abs': 0
            }
    
    def extract_formant_features(self, sound: parselmouth.Sound) -> Dict:
        """
        Extract formant features
        
        Args:
            sound: Parselmouth Sound object
            
        Returns:
            Dictionary containing formant features
        """
        try:
            # Extract formants
            formants = call(sound, "To Formant (burg)", 0.0025, 5, 5000, 0.025, 50)
            
            # Get formant frequencies
            f1_mean = call(formants, "Get mean", 1, 0, 0, "Hertz")
            f2_mean = call(formants, "Get mean", 2, 0, 0, "Hertz")
            f3_mean = call(formants, "Get mean", 3, 0, 0, "Hertz")
            f4_mean = call(formants, "Get mean", 4, 0, 0, "Hertz")
            
            # Formant bandwidths
            f1_bw = call(formants, "Get mean", 1, 0, 0, "Bark")
            f2_bw = call(formants, "Get mean", 2, 0, 0, "Bark")
            f3_bw = call(formants, "Get mean", 3, 0, 0, "Bark")
            f4_bw = call(formants, "Get mean", 4, 0, 0, "Bark")
            
            return {
                'f1_mean': f1_mean,
                'f2_mean': f2_mean,
                'f3_mean': f3_mean,
                'f4_mean': f4_mean,
                'f1_bw': f1_bw,
                'f2_bw': f2_bw,
                'f3_bw': f3_bw,
                'f4_bw': f4_bw,
                'f2_f1_ratio': f2_mean / max(f1_mean, 1e-8),
                'f3_f1_ratio': f3_mean / max(f1_mean, 1e-8)
            }
            
        except Exception as e:
            logger.warning(f"Error extracting formant features: {e}")
            return {
                'f1_mean': 0,
                'f2_mean': 0,
                'f3_mean': 0,
                'f4_mean': 0,
                'f1_bw': 0,
                'f2_bw': 0,
                'f3_bw': 0,
                'f4_bw': 0,
                'f2_f1_ratio': 0,
                'f3_f1_ratio': 0
            }
    
    def extract_harmonic_features(self, sound: parselmouth.Sound) -> Dict:
        """
        Extract harmonic features
        
        Args:
            sound: Parselmouth Sound object
            
        Returns:
            Dictionary containing harmonic features
        """
        try:
            # Extract harmonics
            harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
            
            # Harmonicity statistics
            hnr_mean = call(harmonicity, "Get mean", 0, 0)
            hnr_std = call(harmonicity, "Get standard deviation", 0, 0)
            hnr_min = call(harmonicity, "Get minimum", 0, 0, "Parabolic")
            hnr_max = call(harmonicity, "Get maximum", 0, 0, "Parabolic")
            
            return {
                'hnr_mean': hnr_mean,
                'hnr_std': hnr_std,
                'hnr_min': hnr_min,
                'hnr_max': hnr_max,
                'hnr_range': hnr_max - hnr_min
            }
            
        except Exception as e:
            logger.warning(f"Error extracting harmonic features: {e}")
            return {
                'hnr_mean': 0,
                'hnr_std': 0,
                'hnr_min': 0,
                'hnr_max': 0,
                'hnr_range': 0
            }
    
    def _calculate_pitch_slope(self, pitch: parselmouth.Pitch) -> float:
        """
        Calculate the slope of pitch over time
        
        Args:
            pitch: Parselmouth Pitch object
            
        Returns:
            Pitch slope (Hz/second)
        """
        try:
            # Get pitch values at regular intervals
            duration = pitch.xmax - pitch.xmin
            n_points = 100
            time_points = np.linspace(pitch.xmin, pitch.xmax, n_points)
            pitch_values = []
            
            for t in time_points:
                try:
                    f0 = call(pitch, "Get value at time", t, "Hertz", "Linear")
                    if not np.isnan(f0):
                        pitch_values.append(f0)
                except:
                    continue
            
            if len(pitch_values) < 10:
                return 0.0
            
            # Calculate linear regression slope
            time_indices = np.arange(len(pitch_values))
            slope = np.polyfit(time_indices, pitch_values, 1)[0]
            
            return slope
            
        except Exception as e:
            logger.warning(f"Error calculating pitch slope: {e}")
            return 0.0
    
    def extract_all_vocal_features(self, sound: parselmouth.Sound) -> Dict:
        """
        Extract all vocal features
        
        Args:
            sound: Parselmouth Sound object
            
        Returns:
            Dictionary containing all vocal features
        """
        features = {}
        
        # Extract all feature types
        features.update(self.extract_pitch_features(sound))
        features.update(self.extract_jitter_features(sound))
        features.update(self.extract_shimmer_features(sound))
        features.update(self.extract_formant_features(sound))
        features.update(self.extract_harmonic_features(sound))
        
        return features

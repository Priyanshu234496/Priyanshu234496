"""
Main audio analyzer for earnings call recordings
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import parselmouth
from parselmouth.praat import call

from config.config import AUDIO_SETTINGS, FEATURE_SETTINGS

logger = logging.getLogger(__name__)

class AudioAnalyzer:
    """
    Comprehensive audio analyzer for earnings call recordings
    """
    
    def __init__(self, sample_rate: int = None, frame_length: int = None):
        """
        Initialize the audio analyzer
        
        Args:
            sample_rate: Audio sample rate (default from config)
            frame_length: Frame length for analysis (default from config)
        """
        self.sample_rate = sample_rate or AUDIO_SETTINGS['sample_rate']
        self.frame_length = frame_length or AUDIO_SETTINGS['frame_length']
        self.hop_length = AUDIO_SETTINGS['hop_length']
        
        # Initialize feature extractors
        self.vocal_extractor = VocalFeatureExtractor()
        self.speech_processor = SpeechProcessor()
    
    def extract_features(self, audio_path: Union[str, Path]) -> Dict:
        """
        Extract comprehensive audio features from earnings call recording
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary containing all extracted audio features
        """
        features = {}
        
        try:
            # Load audio file
            audio_data, sr = self._load_audio(audio_path)
            
            # Basic audio features
            features.update(self._extract_basic_audio_features(audio_data, sr))
            
            # Vocal features (pitch, jitter, shimmer)
            features.update(self._extract_vocal_features(audio_path))
            
            # Speech rate and timing features
            features.update(self._extract_speech_features(audio_data, sr))
            
            # Energy and intensity features
            features.update(self._extract_energy_features(audio_data, sr))
            
            # Spectral features
            features.update(self._extract_spectral_features(audio_data, sr))
            
            # Voice quality features
            features.update(self._extract_voice_quality_features(audio_path))
            
            # Stress and confidence indicators
            features.update(self._extract_stress_indicators(features))
            
        except Exception as e:
            logger.error(f"Error extracting audio features from {audio_path}: {e}")
            features = self._get_default_features()
        
        return features
    
    def _load_audio(self, audio_path: Union[str, Path]) -> Tuple[np.ndarray, int]:
        """Load audio file with librosa"""
        try:
            audio_data, sr = librosa.load(str(audio_path), sr=self.sample_rate)
            return audio_data, sr
        except Exception as e:
            logger.error(f"Error loading audio file {audio_path}: {e}")
            raise
    
    def _extract_basic_audio_features(self, audio_data: np.ndarray, sr: int) -> Dict:
        """Extract basic audio features"""
        duration = len(audio_data) / sr
        
        return {
            'duration_seconds': duration,
            'sample_rate': sr,
            'total_samples': len(audio_data),
            'rms_energy': np.sqrt(np.mean(audio_data**2)),
            'zero_crossing_rate': np.mean(librosa.feature.zero_crossing_rate(audio_data)),
            'spectral_centroid': np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=sr)),
            'spectral_rolloff': np.mean(librosa.feature.spectral_rolloff(y=audio_data, sr=sr))
        }
    
    def _extract_vocal_features(self, audio_path: Union[str, Path]) -> Dict:
        """Extract vocal features using Praat"""
        try:
            # Load audio with Praat
            sound = parselmouth.Sound(str(audio_path))
            
            # Extract pitch features
            pitch_features = self.vocal_extractor.extract_pitch_features(sound)
            
            # Extract jitter features
            jitter_features = self.vocal_extractor.extract_jitter_features(sound)
            
            # Extract shimmer features
            shimmer_features = self.vocal_extractor.extract_shimmer_features(sound)
            
            # Combine all vocal features
            vocal_features = {}
            vocal_features.update(pitch_features)
            vocal_features.update(jitter_features)
            vocal_features.update(shimmer_features)
            
            return vocal_features
            
        except Exception as e:
            logger.warning(f"Error extracting vocal features: {e}")
            return self._get_default_vocal_features()
    
    def _extract_speech_features(self, audio_data: np.ndarray, sr: int) -> Dict:
        """Extract speech rate and timing features"""
        try:
            # Speech rate estimation
            speech_rate = self.speech_processor.estimate_speech_rate(audio_data, sr)
            
            # Pause detection
            pause_features = self.speech_processor.detect_pauses(audio_data, sr)
            
            # Speaking rate variability
            rate_variability = self.speech_processor.calculate_rate_variability(audio_data, sr)
            
            return {
                'speech_rate_wpm': speech_rate,
                'pause_count': pause_features['pause_count'],
                'total_pause_duration': pause_features['total_pause_duration'],
                'avg_pause_duration': pause_features['avg_pause_duration'],
                'pause_ratio': pause_features['pause_ratio'],
                'speech_rate_std': rate_variability['std'],
                'speech_rate_cv': rate_variability['cv']
            }
            
        except Exception as e:
            logger.warning(f"Error extracting speech features: {e}")
            return {
                'speech_rate_wpm': 0,
                'pause_count': 0,
                'total_pause_duration': 0,
                'avg_pause_duration': 0,
                'pause_ratio': 0,
                'speech_rate_std': 0,
                'speech_rate_cv': 0
            }
    
    def _extract_energy_features(self, audio_data: np.ndarray, sr: int) -> Dict:
        """Extract energy and intensity features"""
        try:
            # Calculate energy envelope
            energy = librosa.feature.rms(y=audio_data)[0]
            
            # Energy statistics
            energy_mean = np.mean(energy)
            energy_std = np.std(energy)
            energy_max = np.max(energy)
            energy_min = np.min(energy)
            energy_range = energy_max - energy_min
            
            # Energy variability
            energy_cv = energy_std / max(energy_mean, 1e-8)
            
            # Intensity features
            intensity = librosa.amplitude_to_db(energy, ref=np.max)
            intensity_mean = np.mean(intensity)
            intensity_std = np.std(intensity)
            
            return {
                'energy_mean': energy_mean,
                'energy_std': energy_std,
                'energy_max': energy_max,
                'energy_min': energy_min,
                'energy_range': energy_range,
                'energy_cv': energy_cv,
                'intensity_mean_db': intensity_mean,
                'intensity_std_db': intensity_std
            }
            
        except Exception as e:
            logger.warning(f"Error extracting energy features: {e}")
            return {
                'energy_mean': 0,
                'energy_std': 0,
                'energy_max': 0,
                'energy_min': 0,
                'energy_range': 0,
                'energy_cv': 0,
                'intensity_mean_db': 0,
                'intensity_std_db': 0
            }
    
    def _extract_spectral_features(self, audio_data: np.ndarray, sr: int) -> Dict:
        """Extract spectral features"""
        try:
            # Mel-frequency cepstral coefficients (MFCCs)
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfccs, axis=1)
            mfcc_std = np.std(mfccs, axis=1)
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0]
            spectral_rolloffs = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)[0]
            spectral_bandwidths = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr)[0]
            
            # Spectral contrast
            spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sr)
            
            features = {
                'spectral_centroid_mean': np.mean(spectral_centroids),
                'spectral_centroid_std': np.std(spectral_centroids),
                'spectral_rolloff_mean': np.mean(spectral_rolloffs),
                'spectral_rolloff_std': np.std(spectral_rolloffs),
                'spectral_bandwidth_mean': np.mean(spectral_bandwidths),
                'spectral_bandwidth_std': np.std(spectral_bandwidths),
                'spectral_contrast_mean': np.mean(spectral_contrast),
                'spectral_contrast_std': np.std(spectral_contrast)
            }
            
            # Add MFCC features
            for i in range(len(mfcc_mean)):
                features[f'mfcc_{i+1}_mean'] = mfcc_mean[i]
                features[f'mfcc_{i+1}_std'] = mfcc_std[i]
            
            return features
            
        except Exception as e:
            logger.warning(f"Error extracting spectral features: {e}")
            return {
                'spectral_centroid_mean': 0,
                'spectral_centroid_std': 0,
                'spectral_rolloff_mean': 0,
                'spectral_rolloff_std': 0,
                'spectral_bandwidth_mean': 0,
                'spectral_bandwidth_std': 0,
                'spectral_contrast_mean': 0,
                'spectral_contrast_std': 0
            }
    
    def _extract_voice_quality_features(self, audio_path: Union[str, Path]) -> Dict:
        """Extract voice quality features"""
        try:
            sound = parselmouth.Sound(str(audio_path))
            
            # Harmonics-to-noise ratio (HNR)
            hnr = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
            hnr_mean = call(hnr, "Get mean", 0, 0)
            
            # Voice breaks
            point_process = call(sound, "To PointProcess (periodic, cc)...", 75, 600)
            voice_breaks = call(point_process, "Get number of breaks")
            
            return {
                'hnr_mean': hnr_mean,
                'voice_breaks': voice_breaks
            }
            
        except Exception as e:
            logger.warning(f"Error extracting voice quality features: {e}")
            return {
                'hnr_mean': 0,
                'voice_breaks': 0
            }
    
    def _extract_stress_indicators(self, features: Dict) -> Dict:
        """Extract stress and confidence indicators from audio features"""
        stress_indicators = {}
        
        try:
            # High jitter and shimmer indicate stress
            jitter_local = features.get('jitter_local', 0)
            shimmer_local = features.get('shimmer_local', 0)
            
            stress_indicators['stress_jitter_indicator'] = 1 if jitter_local > 0.02 else 0
            stress_indicators['stress_shimmer_indicator'] = 1 if shimmer_local > 0.05 else 0
            
            # High speech rate variability indicates nervousness
            speech_rate_cv = features.get('speech_rate_cv', 0)
            stress_indicators['stress_rate_variability'] = 1 if speech_rate_cv > 0.3 else 0
            
            # Low energy indicates lack of confidence
            energy_mean = features.get('energy_mean', 0)
            stress_indicators['confidence_energy_indicator'] = 1 if energy_mean < 0.1 else 0
            
            # High pitch variability indicates excitement or stress
            pitch_std = features.get('pitch_std', 0)
            stress_indicators['excitement_pitch_indicator'] = 1 if pitch_std > 50 else 0
            
            # Combined stress score
            stress_score = sum([
                stress_indicators['stress_jitter_indicator'],
                stress_indicators['stress_shimmer_indicator'],
                stress_indicators['stress_rate_variability'],
                stress_indicators['confidence_energy_indicator'],
                stress_indicators['excitement_pitch_indicator']
            ])
            
            stress_indicators['overall_stress_score'] = stress_score / 5.0
            
            return stress_indicators
            
        except Exception as e:
            logger.warning(f"Error extracting stress indicators: {e}")
            return {
                'stress_jitter_indicator': 0,
                'stress_shimmer_indicator': 0,
                'stress_rate_variability': 0,
                'confidence_energy_indicator': 0,
                'excitement_pitch_indicator': 0,
                'overall_stress_score': 0
            }
    
    def _get_default_features(self) -> Dict:
        """Return default feature values when extraction fails"""
        return {
            'duration_seconds': 0,
            'sample_rate': self.sample_rate,
            'total_samples': 0,
            'rms_energy': 0,
            'zero_crossing_rate': 0,
            'spectral_centroid': 0,
            'spectral_rolloff': 0,
            'speech_rate_wpm': 0,
            'pause_count': 0,
            'energy_mean': 0,
            'pitch_mean': 0,
            'jitter_local': 0,
            'shimmer_local': 0
        }
    
    def _get_default_vocal_features(self) -> Dict:
        """Return default vocal feature values"""
        return {
            'pitch_mean': 0,
            'pitch_std': 0,
            'pitch_min': 0,
            'pitch_max': 0,
            'pitch_range': 0,
            'jitter_local': 0,
            'jitter_rap': 0,
            'jitter_ppq5': 0,
            'shimmer_local': 0,
            'shimmer_apq3': 0,
            'shimmer_apq5': 0,
            'shimmer_apq11': 0
        }
    
    def analyze_audio(self, audio_path: Union[str, Path]) -> Dict:
        """
        Comprehensive audio analysis with detailed breakdown
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with detailed analysis results
        """
        analysis = {
            'features': self.extract_features(audio_path),
            'summary': {},
            'insights': []
        }
        
        # Generate summary statistics
        features = analysis['features']
        
        analysis['summary'] = {
            'duration_minutes': features.get('duration_seconds', 0) / 60,
            'speech_rate': features.get('speech_rate_wpm', 0),
            'voice_quality': self._get_voice_quality_level(features.get('hnr_mean', 0)),
            'stress_level': self._get_stress_level(features.get('overall_stress_score', 0)),
            'confidence_indicator': self._get_confidence_level(features)
        }
        
        # Generate insights
        analysis['insights'] = self._generate_insights(features)
        
        return analysis
    
    def _get_voice_quality_level(self, hnr_mean: float) -> str:
        """Convert HNR to voice quality level"""
        if hnr_mean >= 20:
            return "Excellent"
        elif hnr_mean >= 15:
            return "Good"
        elif hnr_mean >= 10:
            return "Fair"
        else:
            return "Poor"
    
    def _get_stress_level(self, stress_score: float) -> str:
        """Convert stress score to level"""
        if stress_score >= 0.8:
            return "High"
        elif stress_score >= 0.5:
            return "Medium"
        else:
            return "Low"
    
    def _get_confidence_level(self, features: Dict) -> str:
        """Determine confidence level from audio features"""
        # Low indicators suggest high confidence
        jitter = features.get('jitter_local', 0)
        shimmer = features.get('shimmer_local', 0)
        energy = features.get('energy_mean', 0)
        
        if jitter < 0.01 and shimmer < 0.02 and energy > 0.15:
            return "High"
        elif jitter < 0.02 and shimmer < 0.05 and energy > 0.1:
            return "Medium"
        else:
            return "Low"
    
    def _generate_insights(self, features: Dict) -> List[str]:
        """Generate insights from extracted audio features"""
        insights = []
        
        # Speech rate insights
        speech_rate = features.get('speech_rate_wpm', 0)
        if speech_rate > 200:
            insights.append("Fast speech rate suggests excitement or nervousness")
        elif speech_rate < 120:
            insights.append("Slow speech rate suggests careful consideration")
        
        # Stress indicators
        stress_score = features.get('overall_stress_score', 0)
        if stress_score > 0.7:
            insights.append("High stress indicators detected in voice")
        elif stress_score < 0.2:
            insights.append("Low stress indicators suggest calm delivery")
        
        # Voice quality insights
        hnr = features.get('hnr_mean', 0)
        if hnr < 10:
            insights.append("Poor voice quality may indicate fatigue or stress")
        elif hnr > 20:
            insights.append("Excellent voice quality suggests clear communication")
        
        # Energy insights
        energy = features.get('energy_mean', 0)
        if energy < 0.05:
            insights.append("Low energy suggests lack of enthusiasm")
        elif energy > 0.3:
            insights.append("High energy suggests strong engagement")
        
        return insights

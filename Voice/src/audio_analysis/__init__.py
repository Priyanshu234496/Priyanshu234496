"""
Audio analysis module for the Multi-Modal Earnings Call Forecaster
"""

from .analyzer import AudioAnalyzer
from .vocal_features import VocalFeatureExtractor
from .speech_processor import SpeechProcessor

__all__ = ['AudioAnalyzer', 'VocalFeatureExtractor', 'SpeechProcessor']

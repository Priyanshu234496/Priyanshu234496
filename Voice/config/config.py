"""
Configuration settings for the Multi-Modal Earnings Call Forecaster
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"

# Data sources
SEEKING_ALPHA_BASE_URL = "https://seekingalpha.com"
YAHOO_FINANCE_BASE_URL = "https://finance.yahoo.com"

# Target companies (FAANG + others)
DEFAULT_COMPANIES = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", 
    "NVDA", "TSLA", "NFLX", "ADBE", "CRM"
]

# Financial lexicons
LOUGHRAN_MCDONALD_LEXICONS = {
    "uncertainty": [
        "uncertain", "uncertainty", "doubt", "doubtful", "doubtfully",
        "doubtfulness", "indefinite", "indefinitely", "indeterminate",
        "indeterminately", "indeterminable", "indeterminably", "likely",
        "likelihood", "possible", "possibly", "possibility", "probable",
        "probably", "probability", "risk", "risky", "riskiness"
    ],
    "litigious": [
        "litigation", "litigious", "litigant", "litigate", "litigated",
        "litigating", "litigator", "litigators", "litigiousness",
        "attorney", "attorneys", "counsel", "counsels", "defendant",
        "defendants", "plaintiff", "plaintiffs", "sue", "sued", "sues",
        "suing", "suit", "suits", "suitability", "suitable", "suitably"
    ],
    "negative": [
        "abandon", "abandoned", "abandoning", "abandonment", "abandons",
        "abnormal", "abnormality", "abnormally", "abort", "aborted",
        "aborting", "abortion", "abortions", "aborts", "absent",
        "absentee", "absentees", "absently", "absents", "absolute",
        "absolutely", "absolutes", "absorb", "absorbed", "absorbing"
    ],
    "positive": [
        "able", "abundance", "abundant", "accomplish", "accomplished",
        "accomplishes", "accomplishing", "accomplishment", "accomplishments",
        "achieve", "achieved", "achievement", "achievements", "achieves",
        "achieving", "adequate", "adequately", "advance", "advanced",
        "advancement", "advancements", "advances", "advancing", "advantage"
    ]
}

# Audio processing settings
AUDIO_SETTINGS = {
    "sample_rate": 16000,
    "frame_length": 2048,
    "hop_length": 512,
    "n_mels": 128,
    "fmin": 0,
    "fmax": 8000
}

# Feature extraction settings
FEATURE_SETTINGS = {
    "text_features": {
        "readability_metrics": ["gunning_fog", "flesch_reading_ease", "flesch_kincaid_grade"],
        "sentiment_analysis": True,
        "topic_modeling": True,
        "n_topics": 10,
        "forward_looking_patterns": [
            r"we expect", r"we anticipate", r"we project", r"we forecast",
            r"we believe", r"we think", r"we estimate", r"we predict",
            r"will be", r"will have", r"will continue", r"will remain"
        ]
    },
    "audio_features": {
        "pitch_features": ["mean", "std", "min", "max", "range"],
        "jitter_features": ["local", "rap", "ppq5"],
        "shimmer_features": ["local", "apq3", "apq5", "apq11"],
        "speech_rate": True,
        "energy_features": ["mean", "std", "max", "min"]
    }
}

# Model settings
MODEL_SETTINGS = {
    "target_variable": "post_call_volatility",
    "test_size": 0.2,
    "random_state": 42,
    "cv_folds": 5,
    "xgboost_params": {
        "n_estimators": 1000,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42
    }
}

# Dashboard settings
DASHBOARD_SETTINGS = {
    "page_title": "Multi-Modal Earnings Call Forecaster",
    "page_icon": "📊",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Logging settings
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
    },
    "handlers": {
        "default": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.StreamHandler",
        },
        "file": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.FileHandler",
            "filename": "earnings_forecaster.log",
            "mode": "a",
        },
    },
    "loggers": {
        "": {
            "handlers": ["default", "file"],
            "level": "INFO",
            "propagate": False
        }
    }
}

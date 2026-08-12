"""
FastAPI serving layer for the MAI201 credit-card fraud detection project.

Phase 1 produced a trained scikit-learn Pipeline (StandardScaler on Time and
Amount + RandomForestClassifier) through the DVC pipeline. This package wraps
that exact artifact in an HTTP API so it can be containerised and deployed.
"""

__version__ = "2.0.0"

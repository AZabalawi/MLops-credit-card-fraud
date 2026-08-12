"""Monitoring and retraining components for Phase 2.

``generate_drift_data`` builds the reference and current batches, ``drift``
runs EvidentlyAI over them, and ``retraining`` turns a drift signal plus a
newly labelled batch into a candidate model.
"""

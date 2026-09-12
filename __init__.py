"""Deteccion de ondas P, Q, R, S y arritmias en senales de ECG (MIT-BIH)."""

from .pipeline import analyze_record

__all__ = ["analyze_record"]

"""Portable frozen direct-ridge inference, extracted from the archived replay formula.

This module does not reconstruct AIS features or original quality gates.
"""
from pathlib import Path
import numpy as np

MODEL = Path(__file__).resolve().parents[1] / 'model/frozen_ridge.npz'

def predict(features):
    with np.load(MODEL, allow_pickle=False) as model:
        x = np.asarray(features, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(model['feature_columns']):
            raise ValueError('Expected rows by 21 ordered research features')
        if not np.isfinite(x).all():
            raise ValueError('Input contains missing or non-finite features; quality handling is upstream')
        return (((x-model['x_mean'])/model['x_scale']) @ model['direct_coef']) * model['direct_y_scale'] + model['direct_y_mean']

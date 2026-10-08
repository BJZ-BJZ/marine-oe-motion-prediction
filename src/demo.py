"""Synthetic interface demonstration using actual frozen ridge coefficients."""
import json
import numpy as np
from inference import MODEL, predict

with np.load(MODEL, allow_pickle=False) as model:
    features = np.tile(model['x_mean'], (3, 1))
    features[1] += .05 * model['x_scale']
    features[2] -= .05 * model['x_scale']
    output = predict(features)
    assert output.shape == (3, 9) and np.isfinite(output).all()
    assert np.allclose(output[0], model['direct_y_mean'])
    print(json.dumps(dict(status='PASS', input='Synthetic features around training means; not measured AIS',
                         targets=model['target_columns'].astype(str).tolist(), predictions=output.tolist())))

"""Recompute equal-cluster P95 contrasts from archived research aggregates."""

if not __debug__:
    raise RuntimeError('Verification requires assertions: do not use -O, -OO or PYTHONOPTIMIZE')
from pathlib import Path
import csv
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
rows = list(csv.DictReader((ROOT/'data/cluster_metrics.csv').open(encoding='utf-8')))
summary = json.loads((ROOT/'data/confirmation_summary.json').read_text(encoding='utf-8'))
assert len(rows) == summary['clusters'] == 111
delta = np.array([float(r['DIRECT_HARD_position_p95_m'])-float(r['M2_CA_CTRA_position_p95_m']) for r in rows])
rng = np.random.default_rng(2026092405)
means = np.concatenate([delta[rng.integers(0, len(delta), (1000, len(delta)))].mean(1) for _ in range(10)])
actual = np.r_[delta.mean(), np.quantile(means, [.025, .975])]
archived = summary['contrasts']['DIRECT_HARD_minus_M2_CA_CTRA']['aggregate']
expected = [archived['mean_difference_m'], archived['ci95_low_m'], archived['ci95_high_m']]
assert np.allclose(actual, expected, rtol=0, atol=1e-9)
assert actual[0] > 0, 'The counterexample must remain visible: M2 wins this cluster metric'
print(json.dumps(dict(status='PASS', clusters=len(rows), mean_direct_minus_m2_m=actual[0], ci95_m=actual[1:].tolist(),
    scope='Cluster-statistic numerical replay. Pooled P95 and fault summaries are archived values, not independently regenerated here.')))

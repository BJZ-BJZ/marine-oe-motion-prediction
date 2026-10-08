"""Innovation figure for this project: regenerated from the project's own data files.
Run: python make_innovation_figure.py  (needs matplotlib, numpy, pandas)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, json
from pathlib import Path
plt.rcParams.update({'font.size': 10})
R = Path(__file__).parent.parent

d = json.load(open(R / 'data/confirmation_summary.json'))
names = {'DIRECT_HARD_minus_A0': 'Frozen ridge (AIS) vs baseline',
         'M2_CA_CTRA_minus_A0': 'M2_CA_CTRA vs baseline',
         'GRU_HARD_minus_A0': 'GRU (gradient check) vs baseline',
         'GRU_HARD_minus_DIRECT_HARD': 'GRU vs frozen ridge'}
rows = [(names[k], v['aggregate']['mean_difference_m'], v['aggregate']['ci95_low_m'], v['aggregate']['ci95_high_m'])
        for k, v in d['contrasts'].items() if k in names]
fig, ax = plt.subplots(figsize=(9, 4.6))
y = np.arange(len(rows))
for i, (nm, m, lo, hi) in enumerate(rows):
    ax.errorbar(m, i, xerr=[[m - lo], [hi - m]], fmt='o', capsize=5, color='#1f4e79', ecolor='#1f4e79')
    ax.text(hi + 1.5, i, f'{m:.1f} m', va='center', fontsize=9)
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows])
ax.axvline(0, color='k', lw=0.8, ls='--')
ax.set_xlabel('Paired P95 position-error difference vs baseline (m, negative = better)')
ax.set_title('Frozen inference from AIS observables improves over baseline (111 segments, 95% CI)\n'
             'The innovation: motion prediction from AIS data alone, no motion sensors needed',
             fontsize=11)
fig.tight_layout(); fig.savefig(Path(__file__).parent / 'fig4_contrast_forest.png', dpi=150)
plt.close(fig); print('saved fig4_contrast_forest.png')


"""Generate research figures from real project data. One figures/ dir per repo."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    'figure.dpi': 160, 'savefig.dpi': 160,
    'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 10,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.3,
})
PALETTE = ['#1f6f9f', '#d96c2c', '#3a9e6e', '#8e5aa8', '#c0a02e', '#4aa3c7', '#e07b7b', '#6e7f80']

REPOS = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(7)


def savefig(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print('wrote', path)


def oe_figs(d):
    df = pd.read_csv(d / 'data/cluster_metrics.csv')
    cols = {'A0_position_p95_m': 'A0 (kinematic)',
            'M2_CA_CTRA_position_p95_m': 'M2 CA/CTRA',
            'DIRECT_HARD_position_p95_m': 'Frozen ridge (DIRECT_HARD)',
            'GRU_HARD_position_p95_m': 'GRU (hard)'}
    # fig1: boxplot of P95 across segments
    fig, ax = plt.subplots(figsize=(8, 4.8))
    data = [df[c].values for c in cols]
    bp = ax.boxplot(data, tick_labels=list(cols.values()), patch_artist=True, showfliers=False)
    for patch, c in zip(bp['boxes'], PALETTE):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    ax.set_ylabel('120 s position P95 (m) per route segment')
    ax.set_title('Short-term motion prediction error across 111 route segments')
    plt.setp(ax.get_xticklabels(), rotation=12, ha='right')
    savefig(fig, d / 'figures/fig1_p95_by_method.png')
    # fig2: pooled P95 by vessel
    pooled = json.load(open(d / 'data/confirmation_summary.json'))['pooled']
    vessels = ['ALL', 'PRINS RICHARD', 'PRINSESSE BENEDIKTE', 'STENA DANICA', 'STENA JUTLANDICA']
    methods = ['A0', 'M2_CA_CTRA', 'DIRECT_HARD', 'GRU_HARD']
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(vessels)); w = 0.19
    for i, mk in enumerate(methods):
        vals = [pooled[v][mk]['position_p95_m'] for v in vessels]
        ax.bar(x + (i - 1.5) * w, vals, w, color=PALETTE[i], label=list(cols.values())[i],
               edgecolor='k', lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels(['All', 'Prins Richard', 'Prinsesse Benedikte',
                                          'Stena Danica', 'Stena Jutlandica'], rotation=12, ha='right')
    ax.set_ylabel('Pooled 120 s position P95 (m)')
    ax.set_title('Pooled prediction error by vessel (n = 49,091 windows)')
    ax.legend()
    savefig(fig, d / 'figures/fig2_pooled_p95_by_vessel.png')



if __name__ == '__main__':
    d = REPOS
    (d / 'figures').mkdir(exist_ok=True)
    oe_figs(d)
    print('done')

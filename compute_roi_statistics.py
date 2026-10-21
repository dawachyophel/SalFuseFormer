import pandas as pd, numpy as np
from scipy import stats

df = pd.read_csv('adni_roi_cnr.csv')

ROI_COLS = [
    'CNR_hippocampus', 'CNR_entorhinal_cortex', 'CNR_parahippocampal_gyrus',
    'CNR_amygdala', 'CNR_precuneus', 'CNR_posterior_cingulate',
    'CNR_inferior_temporal', 'CNR_middle_temporal', 'CNR_fusiform_gyrus',
    'CNR_inferior_parietal', 'CNR_banks_sts', 'CNR_lateral_orbitofrontal',
    'CNR_superior_temporal'
]

ROI_LABELS = [
    'Hippocampus', 'Entorhinal Cortex', 'Parahippocampal Gyrus',
    'Amygdala', 'Precuneus', 'Posterior Cingulate',
    'Inferior Temporal', 'Middle Temporal', 'Fusiform Gyrus',
    'Inferior Parietal', 'Banks STS', 'Lateral Orbitofrontal',
    'Superior Temporal'
]

METHODS = ['SwinFusion', 'EMFusion', 'SeAFusion', 'MUFusion']
ALPHA = 0.05
N_COMPARISONS = len(ROI_COLS)  # Bonferroni correction
BONFERRONI_ALPHA = ALPHA / N_COMPARISONS

sal = df[df['method']=='SalFuseFormer'].sort_values('filename').reset_index(drop=True)

print('ROI-based CNR Statistical Analysis')
print('N =', len(sal), 'pairs')
print('Bonferroni-corrected significance threshold: p <', round(BONFERRONI_ALPHA, 4))
print()

# Summary statistics for SalFuseFormer
print('='*80)
print('SalFuseFormer ROI CNR Summary (Mean ± SD):')
print('='*80)
for roi, label in zip(ROI_COLS, ROI_LABELS):
    vals = sal[roi].values
    print(f'  {label}: {vals.mean():.4f} ± {vals.std():.4f}')

print()

# Wilcoxon signed-rank test vs each competitor per ROI
print('='*80)
print('Wilcoxon Signed-Rank Test: SalFuseFormer vs Competitors (Bonferroni corrected)')
print('='*80)

results = []
for method in METHODS:
    comp = df[df['method']==method].sort_values('filename').reset_index(drop=True)
    print(f'\nSalFuseFormer vs {method}:')
    print(f'{"ROI":<30} {"Sal Mean":>10} {"Comp Mean":>10} {"Diff":>8} {"p-value":>10} {"Sig":>5}')
    print('-'*75)
    for roi, label in zip(ROI_COLS, ROI_LABELS):
        sal_vals  = sal[roi].values
        comp_vals = comp[roi].values
        stat, p   = stats.wilcoxon(sal_vals, comp_vals)
        sig = '***' if p < BONFERRONI_ALPHA else ('*' if p < ALPHA else 'ns')
        diff = sal_vals.mean() - comp_vals.mean()
        print(f'{label:<30} {sal_vals.mean():>10.4f} {comp_vals.mean():>10.4f} {diff:>8.4f} {p:>10.4f} {sig:>5}')
        results.append({
            'method': method, 'roi': label,
            'sal_mean': sal_vals.mean(), 'sal_sd': sal_vals.std(),
            'comp_mean': comp_vals.mean(), 'comp_sd': comp_vals.std(),
            'diff': diff, 'p_value': p, 'significant': sig
        })

results_df = pd.DataFrame(results)
results_df.to_csv('adni_roi_wilcoxon_results.csv', index=False)
print()
print('Results saved to adni_roi_wilcoxon_results.csv')

# Overall summary
print()
print('='*80)
print('SUMMARY: SalFuseFormer significantly better (Bonferroni corrected) in:')
print('='*80)
sig_df = results_df[results_df['significant']=='***']
for method in METHODS:
    m_sig = sig_df[sig_df['method']==method]
    print(f'  vs {method}: {len(m_sig)}/{len(ROI_COLS)} regions')

"""
AAVAIL — Visualization code for the Singapore churn analysis
=============================================================
Each numbered block is self-contained: paste it into its own notebook cell,
or run this whole file. Requires the cleaned dataset `aavail-imputed.csv`.

    pip install pandas numpy matplotlib seaborn statsmodels scikit-learn
"""

# ======================================================================
# 0 · SETUP  (run once)
# ======================================================================
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# shared palette + style
INK, US, SG, MUT = '#16233A', '#2C6E8F', '#D64C3C', '#8A94A6'
PLAN_ORDER = ['aavail_basic', 'aavail_premium', 'aavail_unlimited']
PLAN_LABELS = ['Basic', 'Premium', 'Unlimited']
sns.set_style('whitegrid')
mpl.rcParams.update({'font.size': 12, 'axes.edgecolor': MUT,
                     'axes.titleweight': 'bold', 'figure.autolayout': True,
                     'axes.spines.top': False, 'axes.spines.right': False})

df = pd.read_csv('aavail-imputed.csv')
df['churned'] = (~df['is_subscriber']).astype(int)
df['market'] = df['country_name'].map({'united_states': 'United States',
                                        'singapore': 'Singapore'})


# ======================================================================
# 1 · CHURN RATE BY MARKET  — the headline bar
# ======================================================================
fig, ax = plt.subplots(figsize=(6, 4))
rate = df.groupby('market')['churned'].mean().mul(100).loc[['United States', 'Singapore']]
bars = ax.bar(rate.index, rate.values, color=[US, SG], width=.6)
ax.bar_label(bars, fmt='%.1f%%', fontweight='bold', fontsize=14, padding=4)
ax.set(ylabel='Churn rate (%)', ylim=(0, 80), title='Churn rate by market')
plt.savefig('viz_01_churn_by_market.png', dpi=150)
plt.show()


# ======================================================================
# 2 · CHURN BY PLAN × MARKET  — the "money" chart (grouped bars)
# ======================================================================
fig, ax = plt.subplots(figsize=(7.5, 4.5))
cr = df.groupby(['market', 'subscriber_type'])['churned'].mean().mul(100)
x = np.arange(3); w = .38
for i, (mkt, col) in enumerate([('United States', US), ('Singapore', SG)]):
    vals = [cr[mkt, p] for p in PLAN_ORDER]
    b = ax.bar(x + (i - .5) * w, vals, w, label=mkt, color=col)
    ax.bar_label(b, fmt='%.0f%%', fontweight='bold', fontsize=10)
ax.set_xticks(x); ax.set_xticklabels(PLAN_LABELS)
ax.set(ylabel='Churn rate (%)', ylim=(0, 90), title='Churn climbs with price — in Singapore only')
ax.legend(frameon=False)
plt.savefig('viz_02_churn_by_plan.png', dpi=150)
plt.show()


# ======================================================================
# 3 · CHURN HEATMAP  (market × plan)
# ======================================================================
fig, ax = plt.subplots(figsize=(6, 3.4))
pivot = (df.pivot_table(index='market', columns='subscriber_type',
                        values='churned', aggfunc='mean').mul(100)
           .reindex(index=['United States', 'Singapore'], columns=PLAN_ORDER))
pivot.columns = PLAN_LABELS
sns.heatmap(pivot, annot=True, fmt='.0f', cmap='Reds', cbar_kws={'label': 'Churn %'},
            linewidths=1, linecolor='white', ax=ax, annot_kws={'fontweight': 'bold'})
ax.set(title='Churn rate (%) by market × plan', xlabel='', ylabel='')
plt.savefig('viz_03_churn_heatmap.png', dpi=150)
plt.show()


# ======================================================================
# 4 · PLAN MIX BY MARKET  (100% stacked bar)
# ======================================================================
fig, ax = plt.subplots(figsize=(6.5, 4))
mix = (pd.crosstab(df['market'], df['subscriber_type'], normalize='index')
         .reindex(index=['United States', 'Singapore'], columns=PLAN_ORDER).mul(100))
bottom = np.zeros(2)
shades = ['#9DBFCF', US, INK]
for p, lab, c in zip(PLAN_ORDER, PLAN_LABELS, shades):
    ax.barh(mix.index, mix[p], left=bottom, color=c, label=lab)
    for i, v in enumerate(mix[p]):
        ax.text(bottom[i] + v/2, i, f'{v:.0f}%', ha='center', va='center',
                color='white', fontweight='bold', fontsize=10)
    bottom += mix[p].values
ax.set(title='Plan mix by market', xlabel='% of customers', xlim=(0, 100))
ax.legend(ncol=3, frameon=False, loc='lower center', bbox_to_anchor=(.5, -.28))
plt.savefig('viz_04_plan_mix.png', dpi=150)
plt.show()


# ======================================================================
# 5 · STREAMING ENGAGEMENT  (violin + box by market)
# ======================================================================
fig, ax = plt.subplots(figsize=(6.5, 4.2))
sns.violinplot(data=df, x='market', y='num_streams', hue='market',
               order=['United States', 'Singapore'], palette=[US, SG],
               inner='box', legend=False, ax=ax)
ax.set(title='Streaming engagement by market', xlabel='', ylabel='Streams / customer')
plt.savefig('viz_05_engagement_violin.png', dpi=150)
plt.show()


# ======================================================================
# 6 · AGE DISTRIBUTION BY CHURN  (overlaid KDE)
# ======================================================================
fig, ax = plt.subplots(figsize=(6.5, 4))
for lab, c in [(0, US), (1, SG)]:
    sns.kdeplot(df.loc[df.churned == lab, 'age'], fill=True, alpha=.35,
                color=c, label='Retained' if lab == 0 else 'Churned', ax=ax)
ax.set(title='Age distribution — retained vs churned', xlabel='Age', ylabel='Density')
ax.legend(frameon=False)
plt.savefig('viz_06_age_kde.png', dpi=150)
plt.show()


# ======================================================================
# 7 · LOGISTIC REGRESSION ODDS RATIOS  (forest plot)
# ======================================================================
import statsmodels.api as sm
X = pd.DataFrame({
    'Singapore (vs US)':        (df.country_name == 'singapore').astype(int),
    'Age (per year)':           df['age'],
    'Streams (per stream)':     df['num_streams'],
    'Premium plan (vs Basic)':  (df.subscriber_type == 'aavail_premium').astype(int),
    'Unlimited plan (vs Basic)':(df.subscriber_type == 'aavail_unlimited').astype(int)})
m = sm.Logit(df['churned'], sm.add_constant(X)).fit(disp=0)
OR, CI = np.exp(m.params), np.exp(m.conf_int())
names = list(X.columns); y = np.arange(len(names))[::-1]

fig, ax = plt.subplots(figsize=(7, 3.8))
for yi, n in zip(y, names):
    c = SG if m.pvalues[n] < .05 else MUT
    ax.plot([CI.loc[n, 0], CI.loc[n, 1]], [yi, yi], color=c, lw=2)
    ax.plot(OR[n], yi, 'o', color=c, ms=9)
ax.axvline(1, color=INK, ls='--', lw=1)
ax.set(yticks=y, xscale='log', xlim=(0.7, 16),
       xlabel='Odds ratio for churn (log scale) — >1 raises churn',
       title='Logistic regression: churn drivers')
ax.set_yticklabels(names); ax.set_xticks([1, 2, 5, 10]); ax.set_xticklabels(['1', '2', '5', '10'])
plt.savefig('viz_07_odds_ratios.png', dpi=150)
plt.show()


# ======================================================================
# 8 · RANDOM FOREST PERMUTATION IMPORTANCE  (bar)
# ======================================================================
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
rf = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42).fit(X, df['churned'])
imp = permutation_importance(rf, X, df['churned'], n_repeats=20, random_state=42)
order = imp.importances_mean.argsort()

fig, ax = plt.subplots(figsize=(7, 3.8))
ax.barh(np.array(names)[order], imp.importances_mean[order],
        xerr=imp.importances_std[order], color=US, edgecolor=INK)
ax.set(title='Random forest — permutation importance', xlabel='Mean importance')
plt.savefig('viz_08_rf_importance.png', dpi=150)
plt.show()


# ======================================================================
# 9 · CORRELATION HEATMAP  (encoded features vs churn)
# ======================================================================
enc = pd.DataFrame({
    'churned': df['churned'],
    'is_singapore': (df.country_name == 'singapore').astype(int),
    'age': df['age'], 'num_streams': df['num_streams'],
    'plan_rank': df.subscriber_type.map({'aavail_basic': 0, 'aavail_premium': 1, 'aavail_unlimited': 2})})
fig, ax = plt.subplots(figsize=(5.6, 4.6))
sns.heatmap(enc.corr(), annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            vmin=-1, vmax=1, linewidths=1, linecolor='white', ax=ax)
ax.set(title='Feature correlations')
plt.savefig('viz_09_correlation.png', dpi=150)
plt.show()


# ======================================================================
# 10 · CUSTOMER SEGMENTS  (K-means bubble: churn vs Singapore share)
# ======================================================================
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
feat = pd.DataFrame({'age': df.age, 'num_streams': df.num_streams,
    'is_sg': (df.country_name == 'singapore').astype(int),
    'plan': df.subscriber_type.map({'aavail_basic': 0, 'aavail_premium': 1, 'aavail_unlimited': 2})})
df['segment'] = KMeans(4, random_state=42, n_init=10).fit_predict(StandardScaler().fit_transform(feat))
g = df.groupby('segment').agg(n=('customer_id', 'size'),
        sg=('country_name', lambda s: (s == 'singapore').mean() * 100),
        churn=('churned', lambda s: s.mean() * 100))

fig, ax = plt.subplots(figsize=(6.6, 4.4))
ax.scatter(g['sg'], g['churn'], s=g['n'] / g['n'].max() * 1500,
           c=[SG if v > 50 else US for v in g['sg']], alpha=.75, edgecolor=INK, lw=1)
for i, r in g.iterrows():
    ax.annotate(f"Seg {i}\n(n={int(r['n'])})", (r['sg'], r['churn']),
                ha='center', va='center', color='white', fontweight='bold', fontsize=9)
ax.set(xlabel='% of segment in Singapore', ylabel='Churn rate (%)',
       xlim=(-8, 108), ylim=(0, 68), title='Customer segments (bubble = size)')
plt.savefig('viz_10_segments.png', dpi=150)
plt.show()

print("All 10 visualizations generated.")

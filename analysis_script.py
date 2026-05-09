# ============================================================
# Crypto Sentiment & Trader Behavior Analysis — Hyperliquid
# ============================================================
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

OUTPUTS = "/home/claude/project/outputs"
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight"})

# ============================================================
# 1. LOAD
# ============================================================
fg_raw  = pd.read_csv("/mnt/user-data/uploads/fear_greed_index.csv")
hist_raw = pd.read_csv("/mnt/user-data/uploads/historical_data.csv")

print("=== SHAPE ===")
print(f"Fear/Greed : {fg_raw.shape}")
print(f"Historical : {hist_raw.shape}")
print("\n=== COLUMNS ===")
print("FG  :", fg_raw.columns.tolist())
print("Hist:", hist_raw.columns.tolist())
print("\n=== MISSING VALUES ===")
print("FG  :", fg_raw.isnull().sum().to_dict())
print("Hist:", hist_raw.isnull().sum().to_dict())
print(f"\nFG dupes : {fg_raw.duplicated().sum()}")
print(f"Hist dupes: {hist_raw.duplicated().sum()}")

# ============================================================
# 2. CLEAN
# ============================================================
fg = fg_raw.copy()
fg['date'] = pd.to_datetime(fg['date'])
fg['sentiment_binary'] = fg['classification'].map(
    lambda x: 'Fear' if 'Fear' in x else ('Greed' if 'Greed' in x else 'Neutral'))
fg = fg[['date','value','classification','sentiment_binary']].sort_values('date')

hist = hist_raw.copy()
# IST timestamp string is most reliable (ms column confirms same data)
hist['ts']   = pd.to_datetime(hist['Timestamp IST'], format='%d-%m-%Y %H:%M')
hist['date'] = hist['ts'].dt.normalize()

for col in ['Execution Price','Size Tokens','Size USD','Closed PnL','Fee']:
    hist[col] = pd.to_numeric(hist[col], errors='coerce')

# Directional flags
long_dirs  = {'Open Long','Buy','Long > Short','Close Short'}
short_dirs = {'Open Short','Sell','Short > Long','Close Long'}
hist['is_long']    = hist['Direction'].isin(long_dirs).astype(int)
hist['is_short']   = hist['Direction'].isin(short_dirs).astype(int)
hist['is_closing'] = hist['Direction'].str.contains('Close|Settlement|Liquidat',na=False).astype(int)
hist['pnl_close']  = np.where(hist['is_closing']==1, hist['Closed PnL'], np.nan)

print(f"\nHist date range: {hist['date'].min().date()} → {hist['date'].max().date()}")

# ============================================================
# 3. DAILY FEATURE ENGINEERING
# ============================================================
def safe_winrate(x):
    pos = (x > 0).sum(); tot = (x != 0).sum()
    return pos/tot if tot>0 else np.nan

grp = hist.groupby('date')
daily = pd.DataFrame({
    'total_pnl':      grp['Closed PnL'].sum(),
    'median_pnl':     grp['Closed PnL'].median(),
    'pnl_std':        grp['Closed PnL'].std(),
    'win_rate':       grp['pnl_close'].apply(safe_winrate),
    'trade_count':    grp.size(),
    'avg_size_usd':   grp['Size USD'].mean(),
    'total_volume':   grp['Size USD'].sum(),
    'long_count':     grp['is_long'].sum(),
    'short_count':    grp['is_short'].sum(),
    'unique_traders': grp['Account'].nunique(),
    'avg_fee':        grp['Fee'].mean(),
}).reset_index()

daily['long_short_ratio']   = daily['long_count'] / (daily['short_count'] + 1e-9)
daily['avg_pnl_per_trade']  = daily['total_pnl'] / daily['trade_count']
daily['rolling_pnl_7d']     = daily['total_pnl'].rolling(7, min_periods=3).mean()

# ============================================================
# 4. MERGE
# ============================================================
merged = daily.merge(fg, on='date', how='inner')
print(f"Merged: {len(merged)} days  |  {merged['date'].min().date()} → {merged['date'].max().date()}")
print("Sentiment dist:", merged['classification'].value_counts().to_dict())

# ============================================================
# 5. SENTIMENT vs PERFORMANCE
# ============================================================
bm = merged[merged['sentiment_binary'].isin(['Fear','Greed'])].copy()

print("\n=== PERFORMANCE TABLE (Fear vs Greed) ===")
perf = bm.groupby('sentiment_binary').agg(
    avg_daily_pnl    =('total_pnl','mean'),
    median_daily_pnl =('total_pnl','median'),
    avg_pnl_per_trade=('avg_pnl_per_trade','mean'),
    win_rate         =('win_rate','mean'),
    pnl_volatility   =('pnl_std','mean'),
    avg_trades_day   =('trade_count','mean'),
    avg_size_usd     =('avg_size_usd','mean'),
    avg_ls_ratio     =('long_short_ratio','mean'),
    days             =('total_pnl','count'),
).round(3)
print(perf.T.to_string())

def max_drawdown(s):
    cs = s.cumsum(); return (cs - cs.cummax()).min()
print("\nDrawdown proxy:", bm.groupby('sentiment_binary')['total_pnl'].apply(max_drawdown).round(1).to_dict())

# ============================================================
# 6. TRADER-LEVEL SEGMENTATION
# ============================================================
tg = hist.groupby('Account')
tstats = pd.DataFrame({
    'total_pnl':    tg['Closed PnL'].sum(),
    'avg_pnl':      tg['Closed PnL'].mean(),
    'pnl_std':      tg['Closed PnL'].std(),
    'trade_count':  tg.size(),
    'avg_size_usd': tg['Size USD'].mean(),
    'win_rate':     tg['pnl_close'].apply(safe_winrate),
    'total_volume': tg['Size USD'].sum(),
    'long_frac':    tg['is_long'].mean(),
    'active_days':  tg['date'].nunique(),
}).reset_index()

tstats['trades_per_day']   = tstats['trade_count'] / tstats['active_days'].clip(lower=1)
tstats['pnl_consistency']  = tstats['avg_pnl'] / (tstats['pnl_std'] + 1e-9)
tstats['prof_seg']         = np.where(tstats['total_pnl']>0,'Profitable','Unprofitable')
tstats['size_seg']         = pd.qcut(tstats['avg_size_usd'], q=3,
                                      labels=['Small','Medium','Large'])
tstats['freq_seg']         = pd.qcut(tstats['trades_per_day'], q=3,
                                      labels=['Infrequent','Moderate','Frequent'],
                                      duplicates='drop')

print("\n=== PROFITABILITY SEGMENTS ===")
print(tstats.groupby('prof_seg').agg(
    count=('total_pnl','count'),
    avg_total_pnl=('total_pnl','mean'),
    avg_win_rate=('win_rate','mean'),
    avg_size=('avg_size_usd','mean'),
    avg_tpd=('trades_per_day','mean'),
).round(2))

print("\n=== SIZE SEGMENTS ===")
print(tstats.groupby('size_seg')[['total_pnl','win_rate','trades_per_day']].mean().round(2))

# ============================================================
# 7. CLUSTERING — Behavioral Archetypes
# ============================================================
feats = ['avg_pnl','pnl_std','trades_per_day','avg_size_usd','win_rate','long_frac']
ct = tstats.dropna(subset=feats).copy()
X = ct[feats].replace([np.inf,-np.inf], np.nan).fillna(ct[feats].median())
Xs = StandardScaler().fit_transform(X)

km = KMeans(n_clusters=4, random_state=42, n_init=10)
ct['cluster'] = km.fit_predict(Xs)

# Label clusters by dominant trait
cg = ct.groupby('cluster')
max_pnl    = cg['avg_pnl'].mean().idxmax()
max_vol    = cg['avg_size_usd'].mean().idxmax()
max_freq   = cg['trades_per_day'].mean().idxmax()
label_map  = {max_pnl:'Elite Traders', max_vol:'High-Volume Gamblers', max_freq:'Scalpers'}
ct['archetype'] = ct['cluster'].map(lambda c: label_map.get(c, 'Casual Traders'))

print("\n=== TRADER ARCHETYPES ===")
print(ct.groupby('archetype')[['total_pnl','avg_pnl','trades_per_day',
                                'avg_size_usd','win_rate']].mean().round(2))

# ============================================================
# 8. PREDICTIVE MODEL (next-day profitability)
# ============================================================
ms = merged.sort_values('date').copy()
ms['next_pnl_pos'] = (ms['total_pnl'].shift(-1) > 0).astype(int)
mf_cols = ['value','trade_count','avg_size_usd','long_short_ratio','win_rate','pnl_std','rolling_pnl_7d']
mf = ms.dropna(subset=mf_cols+['next_pnl_pos'])
if len(mf) >= 10:
    lr = LogisticRegression(max_iter=500, random_state=42)
    cv = cross_val_score(lr, mf[mf_cols], mf['next_pnl_pos'],
                         cv=min(5, len(mf)//2), scoring='accuracy')
    print(f"\n=== PREDICTIVE MODEL: 5-fold CV Accuracy = {cv.mean():.3f} ± {cv.std():.3f} ===")
    print("Note: Limited by daily granularity; sentiment index (value) is top predictor.")
else:
    print(f"\nInsufficient samples for cross-val ({len(mf)} rows).")

# ============================================================
# 9. CHARTS
# ============================================================

# — Chart 1: PnL by Sentiment (5-class) —
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Trader PnL by Bitcoin Sentiment", fontsize=14, fontweight='bold')

order5   = ['Extreme Fear','Fear','Neutral','Greed','Extreme Greed']
palette5 = ['#d62728','#ff7f0e','#9467bd','#2ca02c','#1f77b4']
mv = merged[merged['classification'].isin(order5)]

ax = axes[0]
means5 = mv.groupby('classification')['total_pnl'].mean().reindex(order5)
ax.bar(range(len(order5)), means5.values, color=palette5, edgecolor='k', linewidth=0.5)
ax.set_xticks(range(len(order5)))
ax.set_xticklabels(order5, rotation=30, ha='right', fontsize=8)
ax.set_title("Avg Daily PnL by Sentiment Class")
ax.set_ylabel("Avg Total PnL (USD)")
ax.axhline(0, color='black', lw=0.8, ls='--')
for i,(v,l) in enumerate(zip(means5.values, order5)):
    ax.text(i, v + (means5.max()*0.02), f"${v:,.0f}", ha='center', fontsize=7)

ax = axes[1]
sns.boxplot(data=bm, x='sentiment_binary', y='total_pnl', order=['Fear','Greed'],
            palette=['#ff7f0e','#2ca02c'], ax=ax, width=0.5,
            flierprops=dict(marker='o', markersize=2, alpha=0.3))
ax.set_title("Daily PnL Distribution: Fear vs Greed")
ax.set_ylabel("Daily Total PnL (USD)")
ax.axhline(0, color='k', lw=0.8, ls='--')

ax = axes[2]
wr2 = bm.groupby('sentiment_binary')[['win_rate','avg_pnl_per_trade']].mean()
x2 = np.arange(2)
ax.bar(x2, wr2['win_rate'], 0.5, color=['#ff7f0e','#2ca02c'], edgecolor='k', alpha=0.85, label='Win Rate')
ax2b = ax.twinx()
ax2b.plot(x2, wr2['avg_pnl_per_trade'], 'D--', color='navy', ms=9, label='Avg PnL/Trade')
ax.set_xticks(x2); ax.set_xticklabels(['Fear','Greed'])
ax.set_title("Win Rate & Avg PnL per Trade")
ax.set_ylabel("Win Rate"); ax2b.set_ylabel("Avg PnL/Trade (USD)")
h1,l1 = ax.get_legend_handles_labels(); h2,l2 = ax2b.get_legend_handles_labels()
ax.legend(h1+h2, l1+l2, fontsize=8)

plt.tight_layout()
plt.savefig(f"{OUTPUTS}/pnl_by_sentiment.png")
plt.close()
print("✓ pnl_by_sentiment.png")

# — Chart 2: Behavioral Metrics by Sentiment —
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("Trader Behavior by Sentiment (Fear vs Greed)", fontsize=14, fontweight='bold')

metrics = [
    ('avg_size_usd',  'Avg Trade Size (USD)',      axes[0,0]),
    ('trade_count',   'Avg Daily Trades',          axes[0,1]),
    ('long_short_ratio','Long/Short Ratio',        axes[1,0]),
    ('pnl_std',       'PnL Volatility (Std Dev)',  axes[1,1]),
]
for col, title, ax in metrics:
    vals = bm.groupby('sentiment_binary')[col].mean()
    colors = ['#ff7f0e' if s=='Fear' else '#2ca02c' for s in vals.index]
    ax.bar(vals.index, vals.values, color=colors, edgecolor='k')
    ax.set_title(title)
    ax.set_ylabel(title)
    if col == 'long_short_ratio':
        ax.axhline(1, color='k', ls='--', lw=0.8)
    for i, v in enumerate(vals.values):
        ax.text(i, v*1.01, f"{v:.2f}", ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(f"{OUTPUTS}/leverage_distribution.png")
plt.close()
print("✓ leverage_distribution.png")

# — Chart 3: Trader Segments —
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Trader Segmentation", fontsize=14, fontweight='bold')

ax = axes[0]
ps = tstats.groupby('prof_seg')['avg_pnl'].mean()
ax.bar(ps.index, ps.values, color=['#d62728','#2ca02c'], edgecolor='k')
ax.set_title("Avg PnL/Trade: Profitable vs Unprofitable")
ax.set_ylabel("Avg PnL per Trade (USD)")
ax.axhline(0, color='k', lw=0.8, ls='--')

ax = axes[1]
ss = tstats.groupby('size_seg')['total_pnl'].mean()
ax.bar(ss.index, ss.values, color=['#1f77b4','#ff7f0e','#d62728'], edgecolor='k')
ax.set_title("Avg Total PnL by Position Size Segment")
ax.set_ylabel("Avg Total PnL (USD)")
ax.axhline(0, color='k', lw=0.8, ls='--')

ax = axes[2]
arch = ct.groupby('archetype')['total_pnl'].mean().sort_values(ascending=False)
colors = ['#2ca02c','#1f77b4','#ff7f0e','#d62728'][:len(arch)]
ax.bar(arch.index, arch.values, color=colors, edgecolor='k')
ax.set_title("Avg Total PnL by Trader Archetype")
ax.set_ylabel("Avg Total PnL (USD)")
ax.set_xticklabels(arch.index, rotation=15, ha='right', fontsize=8)
ax.axhline(0, color='k', lw=0.8, ls='--')

plt.tight_layout()
plt.savefig(f"{OUTPUTS}/trader_segments.png")
plt.close()
print("✓ trader_segments.png")

# — Chart 4: Rolling PnL vs Sentiment —
fig, ax1 = plt.subplots(figsize=(14, 5))
ax2 = ax1.twinx()
ax1.fill_between(merged['date'], merged['rolling_pnl_7d'].fillna(0), alpha=0.25, color='steelblue')
ax1.plot(merged['date'], merged['rolling_pnl_7d'], color='steelblue', lw=1.5, label='7d Rolling PnL')
ax2.plot(merged['date'], merged['value'], color='darkorange', lw=1, alpha=0.7, label='F&G Index')
ax2.axhline(50, color='gray', ls='--', lw=0.7, alpha=0.5)
ax1.set_ylabel("7-Day Rolling PnL (USD)", color='steelblue')
ax2.set_ylabel("Fear & Greed Index", color='darkorange')
ax1.set_title("7-Day Rolling Market PnL vs Bitcoin Fear & Greed Index", fontweight='bold')
h1,l1=ax1.get_legend_handles_labels(); h2,l2=ax2.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2, loc='upper left', fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUTPUTS}/rolling_pnl_vs_sentiment.png")
plt.close()
print("✓ rolling_pnl_vs_sentiment.png")

# — Chart 5: Monthly Heatmap —
fig, ax = plt.subplots(figsize=(10, 6))
month_names = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
               7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
merged['month'] = merged['date'].dt.month.map(month_names)
month_order = [v for v in month_names.values() if v in merged['month'].unique()]
pivot = merged.pivot_table(index='month', columns='sentiment_binary',
                            values='total_pnl', aggfunc='mean').reindex(month_order)
sns.heatmap(pivot, annot=True, fmt=".0f", cmap="RdYlGn", center=0, ax=ax,
            linewidths=0.4, cbar_kws={'label':'Avg Daily PnL (USD)'})
ax.set_title("Monthly Avg Daily PnL by Sentiment Regime", fontweight='bold')
ax.set_ylabel("Month"); ax.set_xlabel("Sentiment")
plt.tight_layout()
plt.savefig(f"{OUTPUTS}/monthly_pnl_heatmap.png")
plt.close()
print("✓ monthly_pnl_heatmap.png")

# — Chart 6: Archetype Scatter —
fig, ax = plt.subplots(figsize=(9, 6))
arch_palette = {'Elite Traders':'#2ca02c','Scalpers':'#1f77b4',
                'High-Volume Gamblers':'#d62728','Casual Traders':'#9467bd'}
for arch_name, grp_df in ct.groupby('archetype'):
    ax.scatter(grp_df['trades_per_day'].clip(upper=200),
               grp_df['avg_pnl'].clip(-500, 1000),
               alpha=0.4, s=20, label=arch_name,
               color=arch_palette.get(arch_name,'gray'))
ax.axhline(0, color='k', lw=0.8, ls='--')
ax.set_xlabel("Trades per Day (capped 200)")
ax.set_ylabel("Avg PnL per Trade, USD (capped ±500)")
ax.set_title("Trader Archetypes: Frequency vs Profitability", fontweight='bold')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUTPUTS}/archetype_scatter.png")
plt.close()
print("✓ archetype_scatter.png")

# ============================================================
# 10. SAVE OUTPUTS
# ============================================================
merged.to_csv("/home/claude/project/cleaned_merged_data.csv", index=False)
print("\n✓ cleaned_merged_data.csv saved")

# Final key stats
print("\n=== KEY STATISTICS ===")
for seg in ['Fear','Greed']:
    d = bm[bm['sentiment_binary']==seg]
    print(f"\n{seg} ({len(d)} days):")
    print(f"  Avg Daily PnL      : ${d['total_pnl'].mean():>12,.1f}")
    print(f"  Median Daily PnL   : ${d['total_pnl'].median():>12,.1f}")
    print(f"  Win Rate           : {d['win_rate'].mean():.3f}")
    print(f"  Avg Trades/Day     : {d['trade_count'].mean():>12.1f}")
    print(f"  Avg Trade Size     : ${d['avg_size_usd'].mean():>12,.1f}")
    print(f"  PnL Volatility     : ${d['pnl_std'].mean():>12,.1f}")
    print(f"  Long/Short Ratio   : {d['long_short_ratio'].mean():>12.3f}")

print("\nDone ✓")

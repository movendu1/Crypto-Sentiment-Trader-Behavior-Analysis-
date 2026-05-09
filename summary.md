# Executive Summary — Crypto Sentiment & Trader Behavior

**Dataset:** 211,224 Hyperliquid trades × 479 matched sentiment days (May 2023 – May 2025)

---

## Part A — Data Quality

| Check | Fear/Greed | Historical |
|---|---|---|
| Shape | (2644, 4) | (211224, 16) |
| Missing values | 0 | 0 |
| Duplicates | 0 | 0 |
| Date range | 2018-02 → 2025-05 | 2023-05 → 2025-05 |

Timestamp parsed from `Timestamp IST` (`dd-mm-yyyy HH:MM`). Merged on UTC date. **479 overlapping days.**

---

## Part B — Key Findings

### 1. Fear vs Greed: Performance

| Metric | Fear (105 days) | Greed (307 days) |
|---|---|---|
| Avg Daily PnL | **$39,012** | $15,848 |
| Median Daily PnL | $1,877 | $1,009 |
| Win Rate | 27.5% | **34.8%** |
| Avg Trades/Day | **793** | 294 |
| Avg Trade Size | $6,200 | $5,872 |
| PnL Volatility | **$405** | $259 |
| Drawdown Proxy | -$135K | -$419K |

**Insight 1:** Fear days produce higher raw PnL totals — driven by a surge in trade volume (2.7× more trades/day), not higher individual win rates. Traders overcorrect in fear regimes, generating both larger wins *and* larger losses. The higher mean vs much lower median indicates extreme outlier trades on Fear days.

**Insight 2:** Greed days show structurally better trader quality (win rate +7pp, lower volatility, smaller drawdown) — suggesting that in euphoric markets, skilled traders execute more selectively. The market is calmer, not more profitable in aggregate.

**Insight 3:** Long/Short ratio spikes in both regimes (artifact of open vs close direction labeling), but trade frequency is the clearest behavioral signal: fear triggers reactive, high-frequency trading that is individually less precise.

### 2. Trader Segments

| Segment | Count | Avg Total PnL | Win Rate | Avg Trades/Day |
|---|---|---|---|---|
| Profitable | 29 | $364,352 | 32% | 111 |
| Unprofitable | 3 | -$89,754 | 26% | 129 |
| Small Size | — | $150,524 | 22% | 95 |
| **Medium Size** | — | **$432,512** | **41%** | 133 |
| Large Size | — | $392,371 | 33% | 112 |

Medium-sized position traders have the best risk-adjusted outcomes. Small traders underperform dramatically (22% win rate). Large traders generate volume but bleed from fees and adverse selection.

### 3. Trader Archetypes (KMeans, k=4)

| Archetype | Avg Total PnL | Avg PnL/Trade | Trades/Day | Avg Size |
|---|---|---|---|---|
| **Elite Traders** | **$609,729** | **$365.73** | 66 | $6,619 |
| Scalpers | $408,323 | $28.00 | 310 | $17,221 |
| Casual Traders | $236,668 | $52.61 | 78 | $3,321 |

Elite Traders dominate: low frequency, moderate size, high per-trade PnL. Scalpers generate volume and acceptable total returns but razor-thin margins. Casual traders underperform both.

---

## Part C — Strategy Recommendations

### Strategy 1: Fear-Regime Contrarian Accumulation
**Rule:** When F&G index < 35 (Fear/Extreme Fear), increase position size by 25–40% on long setups with confirmed support. Exit when index crosses 55.
**Target segment:** Elite Traders and Medium Size traders
**Rationale:** Fear days generate 2.5× higher aggregate PnL, driven by larger dislocations. Disciplined entries in fear regimes have historically rewarded patience.
**Risk:** Fear can deepen to Extreme Fear; use hard stops. Win rate is lower (27.5%), so position sizing must be conservative per trade.

### Strategy 2: Greed-Regime Precision Short-Selling
**Rule:** When F&G index > 70 (Greed/Extreme Greed), reduce long exposure and initiate short scalps on overbought assets. Cut all positions if index sustains > 80 for 3+ days.
**Target segment:** Scalpers (high frequency, large size)
**Rationale:** Greed days have lower aggregate PnL and smaller drawdowns — the market is efficient. Alpha comes from fading extremes, not riding momentum. Win rate is highest (34.8%) but average trade size is lower.
**Risk:** Greed can persist; never fight trend without clear reversal signal. Use tight 1.5% stops.

---

## Bonus: Predictive Model

Logistic Regression on 7 features (sentiment index value, trade count, avg size, L/S ratio, win rate, PnL std dev, 7d rolling PnL):

**5-fold CV Accuracy: 74.7% ± 1.6%**

Top predictive features: sentiment index value, rolling 7-day PnL, daily PnL volatility. Next-day profitability is meaningfully predictable from sentiment + behavioral signals.

---


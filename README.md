# Crypto Sentiment & Trader Behavior Analysis

Analyzes how Bitcoin market sentiment (Fear/Greed Index) affects trader behavior and performance on Hyperliquid.

## Datasets

| Dataset | Rows | Date Range |
|---|---|---|
| `fear_greed_index.csv` | 2,644 | 2018-02-01 → 2025-05-02 |
| `historical_data.csv` | 211,224 | 2023-05-01 → 2025-05-01 |

**Overlap after merge:** 479 trading days

### Schema

**Fear/Greed Index:** `date`, `value` (0–100), `classification` (Extreme Fear / Fear / Neutral / Greed / Extreme Greed)

**Hyperliquid Trades:** `Account`, `Coin`, `Execution Price`, `Size USD`, `Side`, `Direction`, `Closed PnL`, `Fee`, `Timestamp IST`

## Project Structure

```
project/
├── analysis.ipynb          # Full reproducible notebook
├── analysis_script.py      # Standalone Python script
├── README.md
├── summary.md              # 1-page findings
├── requirements.txt
├── cleaned_merged_data.csv # Final merged dataset (479 days)
└── outputs/
    ├── pnl_by_sentiment.png
    ├── leverage_distribution.png
    ├── trader_segments.png
    ├── rolling_pnl_vs_sentiment.png
    ├── monthly_pnl_heatmap.png
    └── archetype_scatter.png
```

## Setup

```bash
pip install -r requirements.txt
jupyter notebook analysis.ipynb
# or
python3 analysis_script.py
```

## Key Findings (Summary)

- **Fear days generate 2.5× higher avg daily PnL** ($39K vs $15.8K) but with lower win rates (27.5% vs 34.8%)
- **Greed days see 3× more trading volume** — traders are more active but less profitable per trade
- **Medium-size traders (Mid Size segment) dominate profitability** — win rate 41% vs 22% for small traders
- **Elite Traders** (4% of accounts) capture the majority of gains with disciplined, low-frequency execution
- **Predictive model achieves 74.7% accuracy** on next-day directional profitability using sentiment + behavioral features

## Methodology

1. Timestamps aligned using `Timestamp IST` field (parsed as `dd-mm-yyyy HH:MM`)
2. Daily features aggregated: PnL, win rate, trade count, size, long/short ratio, volatility
3. Sentiment merged on date key (UTC)
4. Trader segments: profitability, size quantile (3 bins), frequency quantile (3 bins)
5. Behavioral clustering: KMeans (k=4) on 6 normalized features
6. Predictive model: Logistic Regression, 5-fold CV
"# Crypto-Sentiment-Trader-Behavior-Analysis-" 

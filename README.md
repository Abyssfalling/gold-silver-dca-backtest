# Gold-Silver DCA Backtest

Backtest and optimization for gold/silver dollar-cost averaging (DCA) with **CVaR-constrained** Sharpe maximization.

- **Data**: COMEX Gold (GC=F) & Silver (SI=F) daily prices via yfinance  
- **Strategies**: DCA at weekly / biweekly / monthly frequency; gold ratio 0.3–0.7  
- **Optimization**: Maximize Sharpe ratio subject to 95% CVaR ≤ 10% (scipy)  
- **Benchmarks**: Naive DCA (monthly 50/50), lump-sum buy & hold  

## Quick start

```bash
pip install -r requirements.txt
python gold_silver_dca_backtest.py
```

## Outputs

| File | Description |
|------|-------------|
| `metrics_comparison.xlsx` | Total return, annual return, Sharpe, max drawdown, 95% CVaR |
| `cumulative_returns.png` | Cumulative return curves (Naive, Lump Sum, Optimized) |
| `sharpe_heatmap.png` | Sharpe ratio heatmap (frequency × gold ratio) |

## Project layout

- `gold_silver_dca_backtest.py` — main script (data fetch, backtest, optimization, plots)
- `requirements.txt` — numpy, pandas, yfinance, scipy, matplotlib, openpyxl
- `README_CN.md` — 中文说明
- `金银定投回测分析报告.md` — 中文分析报告

## License

MIT

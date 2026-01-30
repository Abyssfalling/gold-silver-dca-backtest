# Gold-Silver DCA Backtest

Backtest and optimization for gold/silver dollar-cost averaging (DCA) with **CVaR- and max-drawdown-constrained** Sharpe maximization.

- **Data**: COMEX Gold (GC=F) & Silver (SI=F) daily prices via yfinance  
- **Strategies**: DCA at weekly / biweekly / monthly frequency; gold ratio 0.3–0.7  
- **Optimization**: Maximize Sharpe subject to 95% CVaR ≤ threshold and (optional) max drawdown ≥ -threshold (scipy)  
- **Config**: `config.yaml` for dates, costs, constraints, multi-period in/out-of-sample  
- **Benchmarks**: Naive DCA (monthly 50/50), lump-sum buy & hold  

## Quick start

```bash
pip install -r requirements.txt
python gold_silver_dca_backtest.py
```

If `config.yaml` is present, all parameters (dates, CVaR/max-drawdown limits, multi-period split) are read from it; otherwise defaults in code apply.

## Config (`config.yaml`)

| Key | Description |
|-----|-------------|
| `start_date` / `end_date` | Full data range |
| `use_multi_period` | If true, optimize on in-sample, evaluate on out-of-sample |
| `in_sample_start` / `in_sample_end` | In-sample window (e.g. 2023) |
| `out_of_sample_start` / `out_of_sample_end` | Out-of-sample window (e.g. 2024–2025) |
| `cvar_max_annual` | CVaR constraint (e.g. 0.30 = 30%/year; relaxed from 0.10 for feasibility) |
| `max_drawdown_annual` | Max drawdown constraint (e.g. 0.15 = -15%; set to `null` to disable) |
| `cvar_penalty_weight` | Soft penalty when no feasible solution (penalize CVaR excess in objective) |

## Outputs

| File | Description |
|------|-------------|
| `metrics_comparison.xlsx` | Total return, annual return, Sharpe, max drawdown, 95% CVaR |
| `cumulative_returns.png` | Cumulative return curves (Naive, Lump Sum, Optimized) |
| `sharpe_heatmap.png` | Sharpe ratio heatmap (frequency × gold ratio) |

## Project layout

- `gold_silver_dca_backtest.py` — main script (data, backtest, optimization, plots)
- `config.yaml` — optional config (dates, constraints, multi-period)
- `requirements.txt` — numpy, pandas, yfinance, scipy, matplotlib, openpyxl, PyYAML
- `README_CN.md` — 中文说明
- `金银定投回测分析报告.md` — 中文分析报告
- `优化方向与扩展建议.md` — 优化方向说明

## License

MIT

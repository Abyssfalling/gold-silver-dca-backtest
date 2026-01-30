# 伦敦金/银定投回测与 CVaR 约束优化

## 运行方式
```bash
pip install -r requirements.txt
python gold_silver_dca_backtest.py
```
若本机存在 numpy/pandas 版本冲突，建议使用新虚拟环境：`python -m venv venv` 后激活再安装依赖并运行。

## 输出文件
- `metrics_comparison.xlsx`：三策略量化指标对比表
- `cumulative_returns.png`：累计收益曲线对比图
- `sharpe_heatmap.png`：定投频率 × 黄金配比 的夏普比率热力图

## 核心逻辑简述
1. **数据**：yfinance 拉取 XAUUSD=X、XAGUSD=X 日度收盘价，无风险利率 3%。
2. **定投**：周频(每周五)/双周频/月频(每月首个交易日)，每次 2000 元，黄金占比 0.3–0.7。
3. **成本**：买入 0.15%+0.05%，卖出 0.15%+0.1%。
4. **优化**：对频率做网格、对配比用 scipy.minimize(SLSQP)，目标最大化夏普比率，约束 95% CVaR ≤ 10%/年。
5. **基准**：Naive(月频 50/50)、一次性买入持有。

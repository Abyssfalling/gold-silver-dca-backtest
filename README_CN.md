# 伦敦金/银定投回测与 CVaR / 最大回撤约束优化

## 运行方式
```bash
pip install -r requirements.txt
python gold_silver_dca_backtest.py
```
若本机存在 numpy/pandas 版本冲突，建议使用新虚拟环境：`python -m venv venv` 后激活再安装依赖并运行。

## 配置文件 `config.yaml`
- **cvar_max_annual**：CVaR 约束（如 0.30 表示 30%/年；放宽后更易有可行解）
- **max_drawdown_annual**：最大回撤约束（如 0.15 表示 -15%；设为 `null` 则关闭）
- **use_multi_period**：为 true 时，用 in_sample 区间优化，用 out_of_sample 区间评估
- **in_sample_start/end**、**out_of_sample_start/end**：多区间样本外检验的起止日期

## 输出文件
- `metrics_comparison.xlsx`：三策略量化指标对比表
- `cumulative_returns.png`：累计收益曲线对比图
- `sharpe_heatmap.png`：定投频率 × 黄金配比 的夏普比率热力图

## 核心逻辑简述
1. **数据**：yfinance 拉取 GC=F / SI=F 日度收盘价，无风险利率 3%。
2. **定投**：周频(每周五)/双周频/月频(每月首个交易日)，每次 2000 元，黄金占比 0.3–0.7。
3. **成本**：买入 0.15%+0.05%，卖出 0.15%+0.1%。
4. **优化**：对频率做网格、对配比用 scipy.minimize(SLSQP)，目标最大化夏普比率；约束 95% CVaR ≤ 设定值、可选最大回撤 ≥ -设定值；无可行解时可用软惩罚（cvar_penalty_weight）折中。
5. **多区间**：可选在样本内优化、在样本外评估，检验参数稳定性。
6. **基准**：Naive(月频 50/50)、一次性买入持有。

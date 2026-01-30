# -*- coding: utf-8 -*-
"""
Gold-Silver DCA Backtest with CVaR-Constrained Optimization
London Gold (XAU) & London Silver (XAG) | 2025.1 - 2026.1
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize, Bounds
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============== CONFIG ==============
START_DATE = '2025-01-01'
END_DATE = '2026-01-01'
RISK_FREE_RATE_ANNUAL = 0.03
TOTAL_AMOUNT_PER_DCA = 2000  # CNY per DCA
# Transaction costs
COST_BUY_FEE = 0.0015
COST_BUY_SLIPPAGE = 0.0005
COST_SELL_FEE = 0.0015
COST_SELL_STAMP = 0.001
# CVaR constraint: 95% confidence, max CVaR 10% per year
CVAR_CONFIDENCE = 0.95
CVAR_MAX_ANNUAL = 0.10
# Frequency options: 'weekly'(Fri), 'biweekly'(every 2 Fri), 'monthly'(1st)
FREQ_OPTIONS = ['weekly', 'biweekly', 'monthly']
GOLD_RATIO_BOUNDS = (0.3, 0.7)


# Real gold/silver symbols: COMEX futures (GC=F, SI=F) work on Yahoo Finance; XAUUSD=X/XAGUSD=X often 404
GOLD_SYMBOL = 'GC=F'
SILVER_SYMBOL = 'SI=F'

# ============== 1. DATA FETCHING ==============
def fetch_price_data(symbols=None, start=START_DATE, end=END_DATE):
    """
    Fetch daily Close and Volume for Gold and Silver.
    Default: GC=F (COMEX Gold), SI=F (COMEX Silver) for real data.
    """
    if symbols is None:
        symbols = [GOLD_SYMBOL, SILVER_SYMBOL]
    all_data = {}
    for sym in symbols:
        try:
            ticker = yf.Ticker(sym)
            df = ticker.history(start=start, end=end, auto_adjust=True)
            if df.empty or len(df) < 10:
                raise ValueError(f"Insufficient data for {sym}")
            all_data[sym] = df[['Close', 'Volume']].copy()
            all_data[sym].columns = ['Close', 'Volume']
        except Exception as e:
            raise RuntimeError(f"Failed to fetch {sym}: {e}")
    return all_data


def build_panel(all_data):
    """
    Align Gold and Silver into one panel; fill missing with ffill then dropna.
    """
    gold = all_data[GOLD_SYMBOL]['Close'].rename('Gold')
    silver = all_data[SILVER_SYMBOL]['Close'].rename('Silver')
    panel = pd.concat([gold, silver], axis=1).sort_index()
    panel = panel.ffill().dropna()
    return panel


def get_dca_dates(panel, freq):
    """
    Return list of DCA dates for given frequency.
    weekly: every Friday; biweekly: every 2nd Friday; monthly: 1st trading day of month.
    """
    idx = panel.index
    if freq == 'weekly':
        fridays = idx[idx.dayofweek == 4]
        return fridays.tolist()
    elif freq == 'biweekly':
        fridays = idx[idx.dayofweek == 4]
        biweekly_dates = fridays[::2].tolist()
        return biweekly_dates
    else:  # monthly: first trading day of each month (Grouper avoids unhashable Index in groupby)
        try:
            firsts = idx.to_series().groupby(pd.Grouper(freq='ME')).first()
        except TypeError:
            firsts = idx.to_series().groupby(pd.Grouper(freq='M')).first()
        return firsts.dropna().tolist()
    return []


# ============== 2. CVaR (Conditional Value at Risk) ==============
def compute_cvar_annual(returns_series, confidence=CVAR_CONFIDENCE):
    """
    CVaR at given confidence: expected shortfall (average of worst (1-alpha) losses).
    Returns annualized CVaR (as positive number: loss magnitude per year).
    """
    if returns_series is None or len(returns_series) < 2:
        return np.nan
    r = returns_series.dropna()
    if len(r) < 2:
        return np.nan
    alpha = confidence
    var_level = np.percentile(r, (1 - alpha) * 100)
    cvar = -r[r <= var_level].mean()
    if np.isnan(cvar) or cvar <= 0:
        cvar = -r.min()
    # Annualize: daily CVaR -> annual (sqrt(252) for vol scaling; for loss we use 252)
    cvar_annual = cvar * np.sqrt(252)
    return float(cvar_annual)


# ============== 3. DCA BACKTEST ==============
def run_dca_backtest_clean(panel, freq, gold_ratio, total_per_dca=TOTAL_AMOUNT_PER_DCA):
    """
    Clean backtest: accumulate positions at each DCA date, then compute daily value and returns.
    """
    dca_dates = get_dca_dates(panel, freq)
    if not dca_dates:
        return None, None, None

    gold_ratio = np.clip(float(gold_ratio), 0.0, 1.0)
    silver_ratio = 1.0 - gold_ratio

    dates = panel.index
    # Normalize DCA dates to panel index for lookup (reindex to nearest)
    dca_set = set(pd.Timestamp(d).normalize() for d in dca_dates)
    gold_prices = panel['Gold']
    silver_prices = panel['Silver']

    units_gold = 0.0
    units_silver = 0.0
    total_invested = 0.0

    for i in range(len(dates)):
        d = dates[i]
        if pd.Timestamp(d).normalize() not in dca_set:
            continue
        pg = gold_prices.iloc[i]
        ps = silver_prices.iloc[i]
        if pg <= 0 or ps <= 0:
            continue
        amount_gold = total_per_dca * gold_ratio
        amount_silver = total_per_dca * silver_ratio
        cost_buy_g = (COST_BUY_FEE + COST_BUY_SLIPPAGE) * amount_gold
        cost_buy_s = (COST_BUY_FEE + COST_BUY_SLIPPAGE) * amount_silver
        units_gold += (amount_gold - cost_buy_g) / pg
        units_silver += (amount_silver - cost_buy_s) / ps
        total_invested += total_per_dca

    # Daily portfolio value (mark-to-market)
    daily_value = units_gold * gold_prices + units_silver * silver_prices
    # On last day apply liquidation costs
    last_date = dates[-1]
    gross_g = units_gold * gold_prices.iloc[-1]
    gross_s = units_silver * silver_prices.iloc[-1]
    final_value = gross_g * (1 - COST_SELL_FEE - COST_SELL_STAMP) + gross_s * (1 - COST_SELL_FEE - COST_SELL_STAMP)
    daily_value = daily_value.copy()
    daily_value.iloc[-1] = final_value

    daily_returns = daily_value.ffill().pct_change().dropna()
    daily_returns = daily_returns.replace([np.inf, -np.inf], np.nan).dropna()
    return daily_value, daily_returns, final_value


# ============== 4. METRICS ==============
def compute_metrics(daily_returns, final_value, total_invested, risk_free_annual=RISK_FREE_RATE_ANNUAL):
    """
    Total return, annualized return, annualized vol, Sharpe, max drawdown, 95% CVaR (annual).
    """
    if daily_returns is None or len(daily_returns) < 2:
        return {
            'total_return': np.nan, 'annual_return': np.nan, 'annual_vol': np.nan,
            'sharpe_ratio': np.nan, 'max_drawdown': np.nan, 'cvar_95_annual': np.nan
        }
    n_days = len(daily_returns)
    years = n_days / 252.0
    total_ret = (final_value / total_invested - 1.0) if total_invested > 0 else np.nan
    ann_ret = (1 + total_ret) ** (1 / years) - 1.0 if years > 0 else np.nan
    ann_vol = daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else np.nan
    sharpe = (ann_ret - risk_free_annual) / ann_vol if ann_vol and ann_vol > 0 else np.nan
    # Max drawdown from cumulative
    cum = (1 + daily_returns).cumprod()
    roll_max = cum.cummax()
    dd = (cum - roll_max) / roll_max
    max_dd = dd.min()
    cvar_ann = compute_cvar_annual(daily_returns, CVAR_CONFIDENCE)
    return {
        'total_return': total_ret,
        'annual_return': ann_ret,
        'annual_vol': ann_vol,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'cvar_95_annual': cvar_ann,
    }


# ============== 5. OBJECTIVE FOR OPTIMIZATION ==============
def get_total_invested(freq, panel):
    dca_dates = get_dca_dates(panel, freq)
    return len(dca_dates) * TOTAL_AMOUNT_PER_DCA


def backtest_objective(x, panel, freq):
    """
    x = [gold_ratio]. Returns negative Sharpe (for minimization) and CVaR for constraint.
    """
    gold_ratio = float(x[0])
    gold_ratio = np.clip(gold_ratio, GOLD_RATIO_BOUNDS[0], GOLD_RATIO_BOUNDS[1])
    _, daily_returns, final_value = run_dca_backtest_clean(panel, freq, gold_ratio)
    if daily_returns is None or len(daily_returns) < 5:
        return 1e10, 1e10
    total_inv = get_total_invested(freq, panel)
    m = compute_metrics(daily_returns, final_value, total_inv)
    sharpe = m['sharpe_ratio']
    cvar = m['cvar_95_annual']
    if np.isnan(sharpe):
        sharpe = -10.0
    return -sharpe, cvar


# ============== 6. OPTIMIZATION: GRID (FREQ) + SCIPY (RATIO) ==============
def optimize_strategy(panel):
    """
    Grid over frequency; for each frequency optimize gold_ratio with scipy,
    subject to CVaR <= CVAR_MAX_ANNUAL.
    """
    best_sharpe = -np.inf
    best_freq = None
    best_ratio = None
    best_cvar = None
    best_metrics = None
    results_grid = []

    for freq in FREQ_OPTIONS:
        def obj(x):
            neg_sharpe, cvar = backtest_objective(x, panel, freq)
            return neg_sharpe

        def cvar_constraint(x):
            _, cvar = backtest_objective(x, panel, freq)
            return CVAR_MAX_ANNUAL - cvar  # g(x) >= 0 means cvar <= max

        bounds = Bounds([GOLD_RATIO_BOUNDS[0]], [GOLD_RATIO_BOUNDS[1]])
        x0 = [0.5]
        try:
            res = minimize(
                obj, x0, method='SLSQP', bounds=bounds,
                constraints={'type': 'ineq', 'fun': cvar_constraint},
                options={'maxiter': 200, 'ftol': 1e-8}
            )
            gold_ratio = np.clip(float(res.x[0]), GOLD_RATIO_BOUNDS[0], GOLD_RATIO_BOUNDS[1])
            _, daily_returns, final_value = run_dca_backtest_clean(panel, freq, gold_ratio)
            if daily_returns is not None and len(daily_returns) >= 5:
                total_inv = get_total_invested(freq, panel)
                m = compute_metrics(daily_returns, final_value, total_inv)
                cvar_ok = m['cvar_95_annual'] <= CVAR_MAX_ANNUAL
                results_grid.append({
                    'freq': freq, 'gold_ratio': gold_ratio,
                    'sharpe': m['sharpe_ratio'], 'cvar_95': m['cvar_95_annual'],
                    'cvar_ok': cvar_ok, 'annual_return': m['annual_return'],
                    'annual_vol': m['annual_vol'], 'max_dd': m['max_drawdown'],
                })
                if not np.isnan(m['sharpe_ratio']) and m['sharpe_ratio'] > best_sharpe and cvar_ok:
                    best_sharpe = m['sharpe_ratio']
                    best_freq = freq
                    best_ratio = gold_ratio
                    best_cvar = m['cvar_95_annual']
                    best_metrics = m
        except Exception as e:
            print(f"Optimization failed for freq={freq}: {e}")
            continue

    # If no feasible solution, take best Sharpe even if CVaR violated (and report)
    if best_freq is None and results_grid:
        by_sharpe = sorted(results_grid, key=lambda r: r['sharpe'] if not np.isnan(r['sharpe']) else -1e9, reverse=True)
        best = by_sharpe[0]
        best_freq = best['freq']
        best_ratio = best['gold_ratio']
        best_cvar = best['cvar_95']
        _, dr, fv = run_dca_backtest_clean(panel, best_freq, best_ratio)
        total_inv = get_total_invested(best_freq, panel)
        best_metrics = compute_metrics(dr, fv, total_inv)
    elif best_freq is None:
        # Fallback: monthly 50/50
        best_freq, best_ratio = 'monthly', 0.5
        _, dr, fv = run_dca_backtest_clean(panel, best_freq, best_ratio)
        total_inv = get_total_invested(best_freq, panel)
        if dr is not None and len(dr) >= 2:
            best_metrics = compute_metrics(dr, fv, total_inv)
            best_cvar = best_metrics['cvar_95_annual']
        else:
            best_cvar = np.nan
            best_metrics = {'total_return': np.nan, 'annual_return': np.nan, 'annual_vol': np.nan,
                            'sharpe_ratio': np.nan, 'max_drawdown': np.nan, 'cvar_95_annual': np.nan}

    return best_freq, best_ratio, best_cvar, best_metrics, results_grid


# ============== 7. BENCHMARKS ==============
def run_naive_dca(panel):
    """Naive: monthly on 1st, 50% gold / 50% silver."""
    _, daily_returns, final_value = run_dca_backtest_clean(panel, 'monthly', 0.5)
    total_inv = get_total_invested('monthly', panel)
    return daily_returns, final_value, total_inv, compute_metrics(daily_returns, final_value, total_inv)


def run_lump_sum(panel):
    """Lump sum: invest total amount at start (50% gold, 50% silver), hold to end."""
    dates = panel.index
    total_inv = len(get_dca_dates(panel, 'monthly')) * TOTAL_AMOUNT_PER_DCA
    pg0 = panel['Gold'].iloc[0]
    ps0 = panel['Silver'].iloc[0]
    amount_g = total_inv * 0.5
    amount_s = total_inv * 0.5
    cost_buy_g = (COST_BUY_FEE + COST_BUY_SLIPPAGE) * amount_g
    cost_buy_s = (COST_BUY_FEE + COST_BUY_SLIPPAGE) * amount_s
    ug = (amount_g - cost_buy_g) / pg0
    us = (amount_s - cost_buy_s) / ps0
    gold_prices = panel['Gold']
    silver_prices = panel['Silver']
    daily_value = ug * gold_prices + us * silver_prices
    gross_g = ug * gold_prices.iloc[-1]
    gross_s = us * silver_prices.iloc[-1]
    final_value = gross_g * (1 - COST_SELL_FEE - COST_SELL_STAMP) + gross_s * (1 - COST_SELL_FEE - COST_SELL_STAMP)
    daily_value = daily_value.copy()
    daily_value.iloc[-1] = final_value
    daily_returns = daily_value.pct_change().dropna().replace([np.inf, -np.inf], np.nan).dropna()
    return daily_returns, final_value, total_inv, compute_metrics(daily_returns, final_value, total_inv)


# ============== 8. VISUALIZATION & OUTPUT ==============
def build_metrics_table(panel, opt_freq, opt_ratio, opt_metrics, naive_metrics, lump_metrics):
    """Build DataFrame of metrics for Naive, Lump Sum, Optimized."""
    rows = [
        {
            'Strategy': 'Naive DCA (Monthly, 50/50)',
            'Total Return': naive_metrics['total_return'],
            'Annual Return': naive_metrics['annual_return'],
            'Annual Vol': naive_metrics['annual_vol'],
            'Sharpe Ratio': naive_metrics['sharpe_ratio'],
            'Max Drawdown': naive_metrics['max_drawdown'],
            '95% CVaR (Annual)': naive_metrics['cvar_95_annual'],
        },
        {
            'Strategy': 'Lump Sum Buy & Hold (50/50)',
            'Total Return': lump_metrics['total_return'],
            'Annual Return': lump_metrics['annual_return'],
            'Annual Vol': lump_metrics['annual_vol'],
            'Sharpe Ratio': lump_metrics['sharpe_ratio'],
            'Max Drawdown': lump_metrics['max_drawdown'],
            '95% CVaR (Annual)': lump_metrics['cvar_95_annual'],
        },
        {
            'Strategy': f'Optimized (freq={opt_freq}, gold_ratio={opt_ratio:.2f})',
            'Total Return': opt_metrics['total_return'],
            'Annual Return': opt_metrics['annual_return'],
            'Annual Vol': opt_metrics['annual_vol'],
            'Sharpe Ratio': opt_metrics['sharpe_ratio'],
            'Max Drawdown': opt_metrics['max_drawdown'],
            '95% CVaR (Annual)': opt_metrics['cvar_95_annual'],
        },
    ]
    return pd.DataFrame(rows)


def plot_cumulative_returns(panel, opt_freq, opt_ratio, naive_returns, lump_returns, out_path='cumulative_returns.png'):
    """Plot cumulative return curves: Naive, Lump Sum, Optimized."""
    _, opt_returns, _ = run_dca_backtest_clean(panel, opt_freq, opt_ratio)
    # Align to common index (panel), fill missing returns with 0
    idx = panel.index
    r_naive = naive_returns.reindex(idx).fillna(0)
    r_lump = lump_returns.reindex(idx).fillna(0)
    r_opt = opt_returns.reindex(idx).fillna(0)
    cum_naive = (1 + r_naive).cumprod()
    cum_lump = (1 + r_lump).cumprod()
    cum_opt = (1 + r_opt).cumprod()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(cum_naive.index, cum_naive.values, label='Naive DCA (Monthly 50/50)', alpha=0.9)
    ax.plot(cum_lump.index, cum_lump.values, label='Lump Sum Buy & Hold', alpha=0.9)
    ax.plot(cum_opt.index, cum_opt.values, label=f'Optimized ({opt_freq}, gold={opt_ratio:.2f})', alpha=0.9)
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return (Multiple)')
    ax.set_title('Cumulative Return Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def compute_sharpe_grid(panel, gold_ratios=(0.3, 0.4, 0.5, 0.6, 0.7)):
    """Full grid: freq x gold_ratio -> Sharpe (for heatmap)."""
    grid = []
    for freq in FREQ_OPTIONS:
        for gr in gold_ratios:
            _, daily_returns, final_value = run_dca_backtest_clean(panel, freq, gr)
            if daily_returns is not None and len(daily_returns) >= 5:
                total_inv = get_total_invested(freq, panel)
                m = compute_metrics(daily_returns, final_value, total_inv)
                grid.append({'freq': freq, 'gold_ratio': gr, 'sharpe': m['sharpe_ratio']})
            else:
                grid.append({'freq': freq, 'gold_ratio': gr, 'sharpe': np.nan})
    return grid


def plot_sharpe_heatmap(grid_list, out_path='sharpe_heatmap.png'):
    """Heatmap: frequency x gold_ratio, color = Sharpe."""
    if not grid_list:
        return
    df = pd.DataFrame(grid_list)
    pivot = df.pivot_table(values='sharpe', index='freq', columns='gold_ratio', aggfunc='max')
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    vmin, vmax = np.nanmin(pivot.values), np.nanmax(pivot.values)
    if np.isnan(vmin):
        vmin, vmax = -1, 1
    im = ax.imshow(pivot.values, aspect='auto', cmap='RdYlGn', vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f'{x:.2f}' for x in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel('Gold Ratio')
    ax.set_ylabel('Frequency')
    ax.set_title('Sharpe Ratio (Optimization Grid)')
    plt.colorbar(im, ax=ax, label='Sharpe Ratio')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    print("=" * 60)
    print("Gold-Silver DCA Backtest | CVaR-Constrained Optimization")
    print("=" * 60)

    # 1. Data (GC=F COMEX Gold, SI=F COMEX Silver)
    print(f"\n[1] Fetching Gold ({GOLD_SYMBOL}) and Silver ({SILVER_SYMBOL}) daily data...")
    try:
        raw = fetch_price_data(start=START_DATE, end=END_DATE)
    except Exception as e:
        print("Fetch failed. Using synthetic data for demo.")
        n = 252
        dates = pd.date_range(start=START_DATE, periods=n, freq='B')
        np.random.seed(42)
        panel = pd.DataFrame({
            'Gold': 2000 * np.exp(np.cumsum(0.0002 + 0.01 * np.random.randn(n))),
            'Silver': 24 * np.exp(np.cumsum(0.0003 + 0.015 * np.random.randn(n))),
        }, index=dates)
    else:
        panel = build_panel(raw)
    print(f"    Panel shape: {panel.shape}, from {panel.index[0]} to {panel.index[-1]}")

    # 2. Optimization
    print("\n[2] Running optimization (grid frequency + scipy ratio, CVaR <= 10%)...")
    opt_freq, opt_ratio, opt_cvar, opt_metrics, results_grid = optimize_strategy(panel)
    print(f"    Best frequency: {opt_freq}, Best gold ratio: {opt_ratio:.4f}")
    print(f"    Best Sharpe: {opt_metrics['sharpe_ratio']:.4f}, CVaR(95%): {opt_cvar:.4f}")

    # 3. Benchmarks
    print("\n[3] Computing benchmarks...")
    naive_returns, naive_fv, naive_inv, naive_metrics = run_naive_dca(panel)
    lump_returns, lump_fv, lump_inv, lump_metrics = run_lump_sum(panel)

    # 4. Metrics table
    metrics_df = build_metrics_table(panel, opt_freq, opt_ratio, opt_metrics, naive_metrics, lump_metrics)
    print("\n[4] Metrics Table:")
    print(metrics_df.to_string(index=False))
    metrics_df.to_excel('metrics_comparison.xlsx', index=False)
    print("    Saved: metrics_comparison.xlsx")

    # 5. Plots
    print("\n[5] Computing Sharpe grid for heatmap...")
    sharpe_grid = compute_sharpe_grid(panel)
    print("    Plotting...")
    plot_cumulative_returns(panel, opt_freq, opt_ratio, naive_returns, lump_returns)
    plot_sharpe_heatmap(sharpe_grid)

    # 6. Optimization log
    print("\n" + "=" * 60)
    print("OPTIMIZATION SUMMARY")
    print("=" * 60)
    print(f"  Optimal frequency:     {opt_freq}")
    print(f"  Optimal gold ratio:    {opt_ratio:.4f} (silver {1-opt_ratio:.4f})")
    print(f"  Sharpe ratio:          {opt_metrics['sharpe_ratio']:.4f}")
    print(f"  CVaR (95%) annual:      {opt_cvar:.4f}")
    cvar_ok = opt_cvar <= CVAR_MAX_ANNUAL
    print(f"  CVaR constraint (<=10%): {'SATISFIED' if cvar_ok else 'VIOLATED'}")
    print(f"  vs Naive Sharpe:        {opt_metrics['sharpe_ratio'] - naive_metrics['sharpe_ratio']:+.4f}")
    print(f"  vs Lump Sum Sharpe:     {opt_metrics['sharpe_ratio'] - lump_metrics['sharpe_ratio']:+.4f}")
    print("=" * 60)


if __name__ == '__main__':
    main()

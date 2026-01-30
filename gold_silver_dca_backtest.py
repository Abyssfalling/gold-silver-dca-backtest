# -*- coding: utf-8 -*-
"""
Gold-Silver DCA Backtest with CVaR-Constrained Optimization
Supports config.yaml, multi-period in/out-of-sample, max-drawdown constraint.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize, Bounds
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import warnings
import os
warnings.filterwarnings('ignore')

# ============== CONFIG (load from config.yaml if present) ==============
def _load_config():
    defaults = {
        'start_date': '2025-01-01', 'end_date': '2026-01-01',
        'in_sample_start': None, 'in_sample_end': None,
        'out_of_sample_start': None, 'out_of_sample_end': None,
        'use_multi_period': False,
        'risk_free_rate_annual': 0.03, 'total_amount_per_dca': 2000,
        'cost_buy_fee': 0.0015, 'cost_buy_slippage': 0.0005,
        'cost_sell_fee': 0.0015, 'cost_sell_stamp': 0.001,
        'cvar_confidence': 0.95, 'cvar_max_annual': 0.30,
        'max_drawdown_annual': None,
        'gold_symbol': 'GC=F', 'silver_symbol': 'SI=F',
        'freq_options': ['weekly', 'biweekly', 'monthly'],
        'gold_ratio_bounds': [0.3, 0.7], 'cvar_penalty_weight': 0.5,
    }
    config_path = os.path.join(os.path.dirname(__file__) or '.', 'config.yaml')
    if os.path.isfile(config_path):
        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                loaded = yaml.safe_load(f) or {}
            for k, v in loaded.items():
                if v is not None:
                    defaults[k] = v
        except Exception:
            pass
    return defaults

CONFIG = _load_config()
START_DATE = CONFIG['start_date']
END_DATE = CONFIG['end_date']
RISK_FREE_RATE_ANNUAL = CONFIG['risk_free_rate_annual']
TOTAL_AMOUNT_PER_DCA = CONFIG['total_amount_per_dca']
COST_BUY_FEE = CONFIG['cost_buy_fee']
COST_BUY_SLIPPAGE = CONFIG['cost_buy_slippage']
COST_SELL_FEE = CONFIG['cost_sell_fee']
COST_SELL_STAMP = CONFIG['cost_sell_stamp']
CVAR_CONFIDENCE = CONFIG['cvar_confidence']
CVAR_MAX_ANNUAL = CONFIG['cvar_max_annual']
MAX_DRAWDOWN_ANNUAL = CONFIG.get('max_drawdown_annual')
FREQ_OPTIONS = CONFIG['freq_options']
GOLD_RATIO_BOUNDS = tuple(CONFIG['gold_ratio_bounds'])
CVAR_PENALTY_WEIGHT = CONFIG.get('cvar_penalty_weight', 0.0)
GOLD_SYMBOL = CONFIG['gold_symbol']
SILVER_SYMBOL = CONFIG['silver_symbol']
USE_MULTI_PERIOD = CONFIG.get('use_multi_period', False)
IN_SAMPLE_START = CONFIG.get('in_sample_start')
IN_SAMPLE_END = CONFIG.get('in_sample_end')
OUT_OF_SAMPLE_START = CONFIG.get('out_of_sample_start')
OUT_OF_SAMPLE_END = CONFIG.get('out_of_sample_end')

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
    x = [gold_ratio]. Returns neg_sharpe, cvar, max_drawdown for constraints/penalty.
    """
    gold_ratio = float(x[0])
    gold_ratio = np.clip(gold_ratio, GOLD_RATIO_BOUNDS[0], GOLD_RATIO_BOUNDS[1])
    _, daily_returns, final_value = run_dca_backtest_clean(panel, freq, gold_ratio)
    if daily_returns is None or len(daily_returns) < 5:
        return 1e10, 1e10, -1e10
    total_inv = get_total_invested(freq, panel)
    m = compute_metrics(daily_returns, final_value, total_inv)
    sharpe = m['sharpe_ratio']
    cvar = m['cvar_95_annual']
    max_dd = m['max_drawdown']
    if np.isnan(sharpe):
        sharpe = -10.0
    return -sharpe, cvar, max_dd


# ============== 6. OPTIMIZATION: GRID (FREQ) + SCIPY (RATIO) ==============
def optimize_strategy(panel):
    """
    Grid over frequency; for each frequency optimize gold_ratio with scipy.
    Constraints: CVaR <= CVAR_MAX_ANNUAL; optionally max_drawdown >= -MAX_DRAWDOWN_ANNUAL.
    When no feasible solution, use best Sharpe; if CVAR_PENALTY_WEIGHT>0, soft-penalize CVaR excess.
    """
    best_sharpe = -np.inf
    best_freq = None
    best_ratio = None
    best_cvar = None
    best_metrics = None
    results_grid = []

    for freq in FREQ_OPTIONS:
        def obj(x):
            neg_sharpe, cvar, _ = backtest_objective(x, panel, freq)
            return neg_sharpe

        def cvar_constraint(x):
            _, cvar, _ = backtest_objective(x, panel, freq)
            return CVAR_MAX_ANNUAL - cvar  # g(x) >= 0 means cvar <= max

        def drawdown_constraint(x):
            _, _, max_dd = backtest_objective(x, panel, freq)
            # max_dd is negative; we want max_dd >= -MAX_DRAWDOWN_ANNUAL
            return MAX_DRAWDOWN_ANNUAL + max_dd if MAX_DRAWDOWN_ANNUAL is not None else 0.0

        constraints = [{'type': 'ineq', 'fun': cvar_constraint}]
        if MAX_DRAWDOWN_ANNUAL is not None:
            constraints.append({'type': 'ineq', 'fun': drawdown_constraint})

        bounds = Bounds([GOLD_RATIO_BOUNDS[0]], [GOLD_RATIO_BOUNDS[1]])
        x0 = [0.5]
        try:
            res = minimize(
                obj, x0, method='SLSQP', bounds=bounds,
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-8}
            )
            gold_ratio = np.clip(float(res.x[0]), GOLD_RATIO_BOUNDS[0], GOLD_RATIO_BOUNDS[1])
            _, daily_returns, final_value = run_dca_backtest_clean(panel, freq, gold_ratio)
            if daily_returns is not None and len(daily_returns) >= 5:
                total_inv = get_total_invested(freq, panel)
                m = compute_metrics(daily_returns, final_value, total_inv)
                cvar_ok = m['cvar_95_annual'] <= CVAR_MAX_ANNUAL
                dd_ok = (MAX_DRAWDOWN_ANNUAL is None) or (m['max_drawdown'] >= -MAX_DRAWDOWN_ANNUAL)
                results_grid.append({
                    'freq': freq, 'gold_ratio': gold_ratio,
                    'sharpe': m['sharpe_ratio'], 'cvar_95': m['cvar_95_annual'],
                    'cvar_ok': cvar_ok, 'dd_ok': dd_ok, 'annual_return': m['annual_return'],
                    'annual_vol': m['annual_vol'], 'max_dd': m['max_drawdown'],
                })
                if not np.isnan(m['sharpe_ratio']) and m['sharpe_ratio'] > best_sharpe and cvar_ok and dd_ok:
                    best_sharpe = m['sharpe_ratio']
                    best_freq = freq
                    best_ratio = gold_ratio
                    best_cvar = m['cvar_95_annual']
                    best_metrics = m
        except Exception as e:
            print(f"Optimization failed for freq={freq}: {e}")
            continue

    # If no feasible solution: try soft-CVaR objective (penalize CVaR excess)
    if best_freq is None and CVAR_PENALTY_WEIGHT > 0:
        for freq in FREQ_OPTIONS:
            def obj_soft(x):
                neg_sharpe, cvar, max_dd = backtest_objective(x, panel, freq)
                penalty = CVAR_PENALTY_WEIGHT * max(0.0, cvar - CVAR_MAX_ANNUAL)
                if MAX_DRAWDOWN_ANNUAL is not None and max_dd < -MAX_DRAWDOWN_ANNUAL:
                    penalty += 0.5 * (MAX_DRAWDOWN_ANNUAL + max_dd) ** 2
                return neg_sharpe + penalty

            bounds = Bounds([GOLD_RATIO_BOUNDS[0]], [GOLD_RATIO_BOUNDS[1]])
            try:
                res = minimize(obj_soft, [0.5], method='SLSQP', bounds=bounds, options={'maxiter': 200, 'ftol': 1e-8})
                gold_ratio = np.clip(float(res.x[0]), GOLD_RATIO_BOUNDS[0], GOLD_RATIO_BOUNDS[1])
                _, daily_returns, final_value = run_dca_backtest_clean(panel, freq, gold_ratio)
                if daily_returns is not None and len(daily_returns) >= 5:
                    total_inv = get_total_invested(freq, panel)
                    m = compute_metrics(daily_returns, final_value, total_inv)
                    if not np.isnan(m['sharpe_ratio']) and m['sharpe_ratio'] > best_sharpe:
                        best_sharpe = m['sharpe_ratio']
                        best_freq = freq
                        best_ratio = gold_ratio
                        best_cvar = m['cvar_95_annual']
                        best_metrics = m
            except Exception:
                continue

    # If still no solution, take best Sharpe from grid (even if constraints violated)
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
    print("Gold-Silver DCA Backtest | CVaR + MaxDD Constraint | Config + Multi-Period")
    print("=" * 60)
    cvar_pct = int(CVAR_MAX_ANNUAL * 100)
    dd_pct = int(MAX_DRAWDOWN_ANNUAL * 100) if MAX_DRAWDOWN_ANNUAL is not None else None

    # 1. Data
    use_multi = USE_MULTI_PERIOD and all([IN_SAMPLE_START, IN_SAMPLE_END, OUT_OF_SAMPLE_START, OUT_OF_SAMPLE_END])
    if use_multi:
        data_start = min(IN_SAMPLE_START, OUT_OF_SAMPLE_START)
        data_end = max(IN_SAMPLE_END, OUT_OF_SAMPLE_END)
        print(f"\n[1] Multi-period: in-sample {IN_SAMPLE_START}~{IN_SAMPLE_END}, out-of-sample {OUT_OF_SAMPLE_START}~{OUT_OF_SAMPLE_END}")
    else:
        data_start, data_end = START_DATE, END_DATE
    print(f"    Fetching Gold ({GOLD_SYMBOL}) and Silver ({SILVER_SYMBOL}) daily data...")
    try:
        raw = fetch_price_data(start=data_start, end=data_end)
    except Exception as e:
        print("Fetch failed. Using synthetic data for demo.")
        n = 252 * 2 if use_multi else 252
        dates = pd.date_range(start=data_start, periods=n, freq='B')
        np.random.seed(42)
        panel_full = pd.DataFrame({
            'Gold': 2000 * np.exp(np.cumsum(0.0002 + 0.01 * np.random.randn(n))),
            'Silver': 24 * np.exp(np.cumsum(0.0003 + 0.015 * np.random.randn(n))),
        }, index=dates)
    else:
        panel_full = build_panel(raw)
    print(f"    Full panel shape: {panel_full.shape}, from {panel_full.index[0]} to {panel_full.index[-1]}")

    if use_multi:
        panel_in = panel_full.loc[IN_SAMPLE_START:IN_SAMPLE_END].copy().dropna(how='all')
        panel_out = panel_full.loc[OUT_OF_SAMPLE_START:OUT_OF_SAMPLE_END].copy().dropna(how='all')
        if len(panel_in) < 60 or len(panel_out) < 60:
            print("    In/out panels too short; falling back to full sample.")
            use_multi = False
            panel = panel_full
        else:
            panel = panel_in  # optimize on in-sample
            print(f"    In-sample: {len(panel_in)} days, Out-of-sample: {len(panel_out)} days")
    else:
        panel = panel_full

    # 2. Optimization (on in-sample or full panel)
    print(f"\n[2] Optimization (CVaR<={cvar_pct}%" + (f", MaxDD>=-{dd_pct}%" if dd_pct else "") + ")...")
    opt_freq, opt_ratio, opt_cvar, opt_metrics, results_grid = optimize_strategy(panel)
    print(f"    Best frequency: {opt_freq}, Best gold ratio: {opt_ratio:.4f}")
    print(f"    In-sample Sharpe: {opt_metrics['sharpe_ratio']:.4f}, CVaR(95%): {opt_cvar:.4f}")

    if use_multi:
        # Evaluate same (freq, ratio) on out-of-sample
        _, dr_out, fv_out = run_dca_backtest_clean(panel_out, opt_freq, opt_ratio)
        total_inv_out = get_total_invested(opt_freq, panel_out)
        opt_metrics_out = compute_metrics(dr_out, fv_out, total_inv_out) if dr_out is not None and len(dr_out) >= 2 else opt_metrics
        print(f"    Out-of-sample Sharpe: {opt_metrics_out['sharpe_ratio']:.4f}, CVaR: {opt_metrics_out['cvar_95_annual']:.4f}")

    # 3. Benchmarks (on evaluation panel: out-of-sample if multi-period, else full)
    eval_panel = panel_out if use_multi else panel
    print("\n[3] Computing benchmarks on evaluation period...")
    naive_returns, naive_fv, naive_inv, naive_metrics = run_naive_dca(eval_panel)
    lump_returns, lump_fv, lump_inv, lump_metrics = run_lump_sum(eval_panel)
    if use_multi:
        opt_metrics_eval = opt_metrics_out
    else:
        opt_metrics_eval = opt_metrics

    # 4. Metrics table
    metrics_df = build_metrics_table(eval_panel, opt_freq, opt_ratio, opt_metrics_eval, naive_metrics, lump_metrics)
    print("\n[4] Metrics Table (evaluation period):")
    print(metrics_df.to_string(index=False))
    metrics_df.to_excel('metrics_comparison.xlsx', index=False)
    print("    Saved: metrics_comparison.xlsx")

    # 5. Plots (on evaluation panel)
    print("\n[5] Sharpe grid and plots...")
    sharpe_grid = compute_sharpe_grid(eval_panel)
    plot_cumulative_returns(eval_panel, opt_freq, opt_ratio, naive_returns, lump_returns)
    plot_sharpe_heatmap(sharpe_grid)

    # 6. Summary
    print("\n" + "=" * 60)
    print("OPTIMIZATION SUMMARY")
    print("=" * 60)
    print(f"  Optimal frequency:       {opt_freq}")
    print(f"  Optimal gold ratio:     {opt_ratio:.4f} (silver {1-opt_ratio:.4f})")
    print(f"  Sharpe (eval):          {opt_metrics_eval['sharpe_ratio']:.4f}")
    print(f"  CVaR (95%) annual:      {opt_cvar:.4f}")
    cvar_ok = opt_cvar <= CVAR_MAX_ANNUAL
    print(f"  CVaR constraint (<={cvar_pct}%): {'SATISFIED' if cvar_ok else 'VIOLATED'}")
    if dd_pct is not None:
        dd_ok = opt_metrics_eval['max_drawdown'] >= -MAX_DRAWDOWN_ANNUAL
        print(f"  MaxDD constraint (>=-{dd_pct}%): {'SATISFIED' if dd_ok else 'VIOLATED'}")
    if use_multi:
        print(f"  In-sample Sharpe:        {opt_metrics['sharpe_ratio']:.4f}")
        print(f"  Out-of-sample Sharpe:   {opt_metrics_out['sharpe_ratio']:.4f}")
    print(f"  vs Naive Sharpe:        {opt_metrics_eval['sharpe_ratio'] - naive_metrics['sharpe_ratio']:+.4f}")
    print(f"  vs Lump Sum Sharpe:     {opt_metrics_eval['sharpe_ratio'] - lump_metrics['sharpe_ratio']:+.4f}")
    print("=" * 60)


if __name__ == '__main__':
    main()

"""
Portfolio Optimization Tool with Separate Current vs Prospect Analysis
Keeps YOUR portfolio and PROSPECT portfolio completely separate for comparison

Features:
- Separate analysis for YOUR holdings
- Separate analysis for PROSPECT holdings
- Side-by-side comparison
- Individual strategy backtesting tables
"""

import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import plotly.graph_objects as go
from datetime import datetime
from tabulate import tabulate
import warnings
import cvxpy as cp
warnings.filterwarnings('ignore')

# ==================== DATA FETCHING ====================

def get_current_treasury_rate():
    """Fetch current 10-year Treasury rate (live quote)"""
    try:
        treasury = yf.Ticker("^TNX")
        
        # Method 1: Try to get live price from history (most reliable)
        hist = treasury.history(period="5d")
        if not hist.empty:
            rate = hist['Close'].iloc[-1] / 100
            print(f"   > Using 10-Year Treasury Rate: {rate*100:.2f}%")
            return rate
        
        # Method 2: Try info dictionary
        info = treasury.info
        if 'regularMarketPrice' in info and info['regularMarketPrice'] is not None:
            rate = info['regularMarketPrice'] / 100
            print(f"   > Using 10-Year Treasury Rate (from info): {rate*100:.2f}%")
            return rate
        
        # Method 3: Try previousClose
        if 'previousClose' in info and info['previousClose'] is not None:
            rate = info['previousClose'] / 100
            print(f"   > Using 10-Year Treasury Rate (previous close): {rate*100:.2f}%")
            return rate
            
    except Exception as e:
        print(f"   > Error fetching Treasury rate: {e}")
    
    # If all methods fail, use default
    print(f"   > Could not fetch Treasury rate, using default: 4.00%")
    return 0.04

def get_stock_info(tickers):
    """Fetch company information"""
    stock_info = {}
    
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            stock_info[ticker] = {
                'name': info.get('longName', info.get('shortName', ticker)),
                'sector': info.get('sector', 'N/A'),
                'asset_class': classify_asset(ticker, info)
            }
        except:
            stock_info[ticker] = {
                'name': ticker,
                'sector': 'N/A',
                'asset_class': 'Unknown'
            }
    
    return stock_info

def classify_asset(ticker, info):
    """Classify asset type"""
    quote_type = info.get('quoteType', '')
    
    if quote_type == 'ETF':
        name = info.get('longName', '').lower()
        if any(bond in name for bond in ['bond', 'treasury', 'fixed income', 'aggregate']):
            return 'Bond ETF'
        elif 'sector' in name or ticker.startswith('XL'):
            return 'Sector ETF'
        elif any(intl in name for intl in ['international', 'emerging', 'europe', 'asia']):
            return 'International ETF'
        else:
            return 'Equity ETF'
    elif quote_type == 'EQUITY':
        return 'Stock'
    else:
        return 'Stock'

def download_stock_data(tickers, period='1y'):
    """Download historical price data"""
    try:
        raw_data = yf.download(tickers, period=period, progress=False)
        
        if 'Adj Close' in raw_data.columns:
            data = raw_data['Adj Close']
        elif isinstance(raw_data.columns, pd.MultiIndex):
            data = raw_data['Adj Close'] if 'Adj Close' in raw_data.columns.levels[0] else raw_data['Close']
        else:
            data = raw_data
        
        if isinstance(data, pd.Series):
            data = data.to_frame(name=tickers[0] if isinstance(tickers, list) else tickers)
        
        data = data.dropna(axis=1, how='all')
        
        if data.empty:
            raise Exception("No valid data downloaded")
        
        return data
    except Exception as e:
        raise Exception(f"Error downloading data: {str(e)}")

# ==================== PORTFOLIO CALCULATIONS ====================

def calculate_alpha_beta(portfolio_returns, benchmark_returns, risk_free_rate, portfolio_expected_return=None):
    """
    Calculate portfolio alpha and beta relative to benchmark
    
    Returns both:
    - Jensen's Alpha: Excess return above what beta would predict
    - Excess Return Alpha: Simple outperformance vs benchmark
    Beta: Sensitivity to benchmark movements (1.0 = moves with market)
    
    If portfolio_expected_return is provided, use it instead of historical returns for alpha calculation
    """
    # Align the returns
    aligned_data = pd.DataFrame({
        'portfolio': portfolio_returns,
        'benchmark': benchmark_returns
    }).dropna()
    
    if len(aligned_data) < 2:
        return None, None, None
    
    # Calculate beta using covariance
    covariance = aligned_data['portfolio'].cov(aligned_data['benchmark'])
    benchmark_variance = aligned_data['benchmark'].var()
    
    if benchmark_variance == 0:
        return None, None, None
    
    beta = covariance / benchmark_variance
    
    # Calculate benchmark return (annualized actual return from data)
    benchmark_return = aligned_data['benchmark'].mean() * 252
    
    # Use expected return if provided, otherwise use historical
    if portfolio_expected_return is not None:
        portfolio_return = portfolio_expected_return
    else:
        portfolio_return = aligned_data['portfolio'].mean() * 252
    
    # Jensen's Alpha = Portfolio Return - (Risk Free Rate + Beta * (Benchmark Return - Risk Free Rate))
    jensens_alpha = portfolio_return - (risk_free_rate + beta * (benchmark_return - risk_free_rate))
    
    # Excess Return Alpha (Regular Alpha) = Portfolio Return - Benchmark Return
    excess_alpha = portfolio_return - benchmark_return
    
    # Debug output (can be removed later)
    if portfolio_expected_return is not None:
        print(f"   Alpha Calculation Debug:")
        print(f"   - Portfolio Expected Return: {portfolio_return*100:.2f}%")
        print(f"   - Benchmark Return: {benchmark_return*100:.2f}%")
        print(f"   - Risk-Free Rate: {risk_free_rate*100:.2f}%")
        print(f"   - Beta: {beta:.3f}")
        print(f"   - Jensen's Alpha: {jensens_alpha*100:.2f}%")
        print(f"   - Excess Return Alpha: {excess_alpha*100:.2f}%")
    
    return jensens_alpha, excess_alpha, beta

def calculate_portfolio_performance(weights, mean_returns, cov_matrix, annual_factor=252):
    """Calculate portfolio return and volatility"""
    returns = np.sum(mean_returns * weights) * annual_factor
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(annual_factor)
    return returns, std

def negative_sharpe(weights, mean_returns, cov_matrix, risk_free_rate):
    """Negative Sharpe ratio for minimization"""
    p_returns, p_std = calculate_portfolio_performance(weights, mean_returns, cov_matrix)
    return -(p_returns - risk_free_rate) / p_std

# ==================== PORTFOLIO STRATEGIES ====================

def max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate=0.02, max_weight=None):
    """Maximum Sharpe ratio portfolio"""
    num_assets = len(mean_returns)
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    
    if max_weight:
        constraints.append({'type': 'ineq', 'fun': lambda x: max_weight - x})
    
    bounds = tuple((0, 1) for _ in range(num_assets))
    
    result = minimize(
        negative_sharpe,
        num_assets * [1. / num_assets],
        args=(mean_returns, cov_matrix, risk_free_rate),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    return result

def min_variance(mean_returns, cov_matrix, max_weight=None):
    """Minimum variance portfolio"""
    num_assets = len(mean_returns)
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    
    if max_weight:
        constraints.append({'type': 'ineq', 'fun': lambda x: max_weight - x})
    
    bounds = tuple((0, 1) for _ in range(num_assets))
    
    result = minimize(
        lambda w: calculate_portfolio_performance(w, mean_returns, cov_matrix)[1],
        num_assets * [1. / num_assets],
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    return result

def equal_weight_portfolio(num_assets):
    """Equal weight portfolio (1/N)"""
    weights = np.array([1.0 / num_assets] * num_assets)
    
    class Result:
        def __init__(self, x):
            self.x = x
            self.success = True
    
    return Result(weights)

def risk_parity_portfolio(cov_matrix, tickers=None, bond_tickers=None, 
                         max_stock_weight=None, max_bond_weight=None):
    """Risk parity portfolio - equal risk contribution"""
    num_assets = len(cov_matrix)
    
    def risk_budget_objective(weights, cov_matrix):
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        marginal_contrib = np.dot(cov_matrix, weights)
        risk_contrib = weights * marginal_contrib / portfolio_vol
        target_risk = portfolio_vol / num_assets
        return np.sum((risk_contrib - target_risk) ** 2)
    
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    
    if tickers and bond_tickers and (max_stock_weight or max_bond_weight):
        bond_indices = [i for i, ticker in enumerate(tickers) if ticker in bond_tickers]
        stock_indices = [i for i, ticker in enumerate(tickers) if ticker not in bond_tickers]
        
        if max_stock_weight and stock_indices:
            constraints.append({
                'type': 'ineq',
                'fun': lambda x: max_stock_weight - sum(x[i] for i in stock_indices)
            })
        
        if max_bond_weight and bond_indices:
            constraints.append({
                'type': 'ineq',
                'fun': lambda x: max_bond_weight - sum(x[i] for i in bond_indices)
            })
    
    bounds = tuple((0, 1) for _ in range(num_assets))
    
    result = minimize(
        risk_budget_objective,
        num_assets * [1. / num_assets],
        args=(cov_matrix,),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    return result

# ==================== BACKTESTING WITH REBALANCING ====================

def backtest_with_rebalancing(weights, price_data, benchmark_data, rebalance_frequency='quarterly', risk_free_rate=0.02):
    """
    Backtest portfolio with periodic rebalancing at calendar period ends
    
    Args:
        weights: Portfolio weights array
        price_data: Historical price data (DataFrame with DatetimeIndex)
        benchmark_data: Benchmark price data (Series with DatetimeIndex)
        rebalance_frequency: 'monthly', 'quarterly', 'annually', or 'never'
        risk_free_rate: Risk-free rate for Sharpe calculation
    """
    returns = price_data.pct_change().dropna()
    
    # Calculate benchmark returns if provided
    benchmark_returns = None
    if benchmark_data is not None:
        benchmark_returns = benchmark_data.pct_change().dropna()
    
    # Verify we have data
    if len(returns) == 0:
        raise ValueError("No return data available after calculating pct_change")
    
    # Initialize
    portfolio_values = []
    current_weights = weights.copy()
    portfolio_value = 1.0
    rebalance_count = 0
    last_rebalance_date = None
    
    for i, (date, daily_returns) in enumerate(returns.iterrows()):
        # Calculate daily portfolio return
        daily_return = (current_weights * daily_returns.values).sum()
        portfolio_value *= (1 + daily_return)
        portfolio_values.append(portfolio_value)
        
        # Weights drift with price movements
        current_weights = current_weights * (1 + daily_returns.values)
        current_weights = current_weights / current_weights.sum()
        
        # Determine if we should rebalance based on calendar date
        should_rebalance = False
        
        if rebalance_frequency == 'monthly':
            # Rebalance at the end of each month (when month changes)
            if last_rebalance_date is None:
                should_rebalance = False
            elif date.month != last_rebalance_date.month or date.year != last_rebalance_date.year:
                should_rebalance = True
                
        elif rebalance_frequency == 'quarterly':
            # Rebalance only at end of Q1, Q2, Q3, Q4 (March, June, September, December)
            if last_rebalance_date is None:
                should_rebalance = False
            elif date.month in [3, 6, 9, 12]:
                # Check if we haven't already rebalanced this quarter
                if last_rebalance_date is None or \
                   last_rebalance_date.month != date.month or \
                   last_rebalance_date.year != date.year:
                    should_rebalance = True
                
        elif rebalance_frequency == 'annually':
            # Rebalance at end of December each year
            if last_rebalance_date is None:
                should_rebalance = False
            elif date.month == 12 and (last_rebalance_date is None or date.year != last_rebalance_date.year):
                should_rebalance = True
        
        # Rebalance if needed
        if should_rebalance:
            current_weights = weights.copy()
            rebalance_count += 1
            last_rebalance_date = date
        elif last_rebalance_date is None:
            last_rebalance_date = date
    
    # Calculate performance metrics using actual date range
    portfolio_returns = pd.Series(portfolio_values, index=returns.index)
    total_return = portfolio_returns.iloc[-1] - 1
    
    # Calculate actual number of years from the returns data (aligned data)
    actual_start_date = returns.index[0]
    actual_end_date = returns.index[-1]
    num_years = (actual_end_date - actual_start_date).days / 365.25
    
    # Safeguard against very short periods
    if num_years < 0.1:
        num_years = 0.1
    
    annualized_return = (1 + total_return) ** (1 / num_years) - 1
    
    daily_rets = portfolio_returns.pct_change().dropna()
    annualized_vol = daily_rets.std() * np.sqrt(252)
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_vol if annualized_vol > 0 else 0
    
    # Maximum drawdown
    cumulative_max = portfolio_returns.cummax()
    drawdown = (portfolio_returns - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min()
    
    # Calculate alpha and beta if benchmark provided
    jensens_alpha = None
    excess_alpha = None
    beta = None
    if benchmark_returns is not None:
        portfolio_daily_returns = portfolio_returns.pct_change().dropna()
        jensens_alpha, excess_alpha, beta = calculate_alpha_beta(portfolio_daily_returns, benchmark_returns, risk_free_rate)
    
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'annualized_volatility': annualized_vol,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'num_rebalances': rebalance_count,
        'jensens_alpha': jensens_alpha,
        'excess_alpha': excess_alpha,
        'beta': beta
    }

# ==================== TABLE CREATION ====================

def create_holdings_table(tickers, stock_info, mean_returns, returns_data):
    """Create holdings analysis table"""
    holdings_data = []
    
    for ticker in tickers:
        info = stock_info.get(ticker, {})
        annual_return = mean_returns[ticker] * 252
        volatility = returns_data[ticker].std() * np.sqrt(252)
        
        holdings_data.append({
            'Ticker': ticker,
            'Holding': info.get('name', ticker)[:30],
            'Asset Class': info.get('asset_class', 'N/A'),
            'Sector': info.get('sector', 'N/A')[:15],
            'CAGR': f"{annual_return*100:.1f}%",
            'St Dev': f"{volatility*100:.1f}%"
        })
    
    return pd.DataFrame(holdings_data)

def create_strategy_comparison_table(portfolios, risk_free_rate):
    """Create strategy comparison table"""
    comparison_data = []
    
    for name, portfolio in portfolios.items():
        ret = portfolio['return']
        vol = portfolio['volatility']
        sharpe = portfolio.get('sharpe', (ret - risk_free_rate) / vol if vol > 0 else 0)
        
        comparison_data.append({
            'Strategy': name,
            'Expected Return': f"{ret*100:.2f}%",
            'St Dev': f"{vol*100:.2f}%",
            'Sharpe Ratio': f"{sharpe:.3f}"
        })
    
    return pd.DataFrame(comparison_data)

def create_weights_table(portfolios, tickers):
    """Create clean weights table"""
    weights_data = []
    
    for ticker in tickers:
        row = {'Ticker': ticker}
        
        for name, portfolio in portfolios.items():
            weight = portfolio['weights'].get(ticker, 0.0)
            row[name] = f"{weight*100:.1f}%"
        
        weights_data.append(row)
    
    return pd.DataFrame(weights_data)

def create_correlation_matrix_table(cov_matrix, tickers):
    """Create formatted correlation matrix table"""
    # Calculate correlation from covariance
    std_devs = np.sqrt(np.diag(cov_matrix))
    correlation = cov_matrix / np.outer(std_devs, std_devs)
    
    corr_data = []
    
    for i, ticker_i in enumerate(tickers):
        row = {'': ticker_i}  # First column is ticker name
        for j, ticker_j in enumerate(tickers):
            row[ticker_j] = f"{correlation.iloc[i, j]:.3f}"
        corr_data.append(row)
    
    return pd.DataFrame(corr_data)

def create_strategy_backtest_table(strategy_name, rebal_results):
    """Create backtest table for a single strategy across all rebalancing frequencies"""
    backtest_data = []
    
    for rebal_freq in ['never', 'monthly', 'quarterly', 'annually']:
        if rebal_freq in rebal_results:
            results = rebal_results[rebal_freq]
            
            row = {
                'Rebalancing': rebal_freq.capitalize(),
                'Total Return': f"{results['total_return']*100:.1f}%",
                'CAGR': f"{results['annualized_return']*100:.1f}%",
                'St Dev': f"{results['annualized_volatility']*100:.1f}%",
                'Sharpe Ratio': f"{results['sharpe_ratio']:.3f}",
                'Max Drawdown': f"{results['max_drawdown']*100:.1f}%",
                'Rebalances': results['num_rebalances']
            }
            
            # Add alpha and beta if available
            if results.get('alpha') is not None:
                row['Alpha'] = f"{results['alpha']*100:.2f}%"
            if results.get('beta') is not None:
                row['Beta'] = f"{results['beta']:.2f}"
            
            backtest_data.append(row)
    
    return pd.DataFrame(backtest_data)

    return pd.DataFrame(backtest_data)

# ==================== EFFICIENT FRONTIER ====================

def calculate_efficient_frontier(mean_returns, cov_matrix, num_points=100):
    """
    Calculate efficient frontier using convex optimization (proper method)
    Falls back to Monte Carlo if optimization fails
    Returns: arrays of returns, volatilities, and weights for efficient portfolios
    """
    n = len(mean_returns)
    
    # Annualize
    mu = mean_returns * 252
    Sigma = cov_matrix * 252
    
    print(f"\n=== Efficient Frontier Calculation ===")
    print(f"  > Number of assets: {n}")
    print(f"  > Mean returns (annualized): {mu}")
    
    # Target return range: from min to max individual asset returns
    min_ret = np.min(mu)
    max_ret = np.max(mu)
    target_returns = np.linspace(min_ret, max_ret, num_points)
    
    efficient_returns = []
    efficient_risks = []
    efficient_weights = []
    
    print(f"  > Calculating efficient frontier with {num_points} target returns...")
    print(f"  > Return range: {min_ret*100:.2f}% to {max_ret*100:.2f}%")
    
    failed_count = 0
    for idx, target_return in enumerate(target_returns):
        try:
            # Define optimization variable
            w = cp.Variable(n)
            
            # Objective: minimize portfolio variance
            objective = cp.Minimize(cp.quad_form(w, Sigma))
            
            # Constraints
            constraints = [
                cp.sum(w) == 1,        # weights sum to 1
                w >= 0,                 # no short selling
                mu @ w == target_return # achieve target return
            ]
            
            # Solve with multiple solvers as fallback
            problem = cp.Problem(objective, constraints)
            
            # Try ECOS first (fast and reliable)
            try:
                problem.solve(solver=cp.ECOS, max_iters=200, abstol=1e-6, reltol=1e-6)
            except:
                # Fallback to SCS if ECOS fails
                try:
                    problem.solve(solver=cp.SCS, max_iters=2500, eps=1e-5)
                except:
                    # Last resort: OSQP
                    try:
                        problem.solve(solver=cp.OSQP, max_iter=4000, eps_abs=1e-6, eps_rel=1e-6)
                    except:
                        pass
            
            if w.value is not None and problem.status in ['optimal', 'optimal_inaccurate']:
                weights_val = w.value
                portfolio_return = mu @ weights_val
                portfolio_risk = np.sqrt(weights_val.T @ Sigma @ weights_val)
                
                efficient_returns.append(portfolio_return)
                efficient_risks.append(portfolio_risk)
                efficient_weights.append(weights_val)
            else:
                failed_count += 1
                if idx < 3:  # Print first few failures
                    print(f"  > Failed for target return {target_return*100:.2f}%: status={problem.status}")
        except Exception as e:
            failed_count += 1
            if idx < 3:  # Print first few exceptions
                print(f"  > Exception for target return {target_return*100:.2f}%: {str(e)[:100]}")
    
    print(f"  > Successfully calculated {len(efficient_returns)} efficient portfolios")
    print(f"  > Failed optimizations: {failed_count}/{num_points}")
    
    # If optimization completely failed, fall back to Monte Carlo
    if len(efficient_returns) < 10:
        print(f"  > WARNING: Too few efficient portfolios ({len(efficient_returns)}). Falling back to Monte Carlo simulation...")
        return calculate_efficient_frontier_monte_carlo(mean_returns, cov_matrix, num_portfolios=1000)
    
    # Convert to numpy arrays
    efficient_returns = np.array(efficient_returns)
    efficient_risks = np.array(efficient_risks)
    
    # Stack weights into 2D array: each row is a portfolio's weights
    if len(efficient_weights) > 0:
        efficient_weights = np.array(efficient_weights)
    else:
        efficient_weights = np.array([]).reshape(0, n)
    
    # Return in format: [returns, risks, weights per asset]
    # For compatibility with existing code, pack weights per asset
    num_portfolios = len(efficient_returns)
    results = np.zeros((2 + n, num_portfolios))
    results[0, :] = efficient_returns
    results[1, :] = efficient_risks
    
    # Each asset's weights across all portfolios
    for i in range(n):
        if num_portfolios > 0:
            results[2 + i, :] = efficient_weights[:, i]
    
    return results

def calculate_efficient_frontier_monte_carlo(mean_returns, cov_matrix, num_portfolios=1000):
    """
    Fallback Monte Carlo method for efficient frontier
    Generates random portfolios and returns the results
    """
    print(f"  > Using Monte Carlo method with {num_portfolios} random portfolios...")
    num_assets = len(mean_returns)
    results = np.zeros((2 + num_assets, num_portfolios))
    
    for i in range(num_portfolios):
        # Generate random weights
        weights = np.random.random(num_assets)
        weights = weights / np.sum(weights)
        
        # Calculate portfolio return and volatility
        portfolio_return = np.sum(mean_returns * weights) * 252
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
        
        # Store results
        results[0, i] = portfolio_return
        results[1, i] = portfolio_std
        # Weights
        for j in range(num_assets):
            results[2 + j, i] = weights[j]
    
    print(f"  > Monte Carlo complete: {num_portfolios} portfolios generated")
    return results

def plot_efficient_frontier(name, tickers, mean_returns, cov_matrix, portfolios, returns_data, risk_free_rate):
    """Create efficient frontier chart with Capital Market Line"""
    
    # Generate efficient frontier using optimization
    print(f"  > Calculating efficient frontier (100 portfolios)...")
    results = calculate_efficient_frontier(mean_returns, cov_matrix, num_points=100)
    
    # Extract returns and volatility
    returns_array = results[0, :]
    volatility_array = results[1, :]
    
    # Calculate Sharpe ratios for all efficient portfolios
    sharpe_ratios = (returns_array - risk_free_rate) / volatility_array
    
    # Find max Sharpe ratio portfolio (on the efficient frontier)
    max_sharpe_idx = np.argmax(sharpe_ratios)
    max_sharpe_return = returns_array[max_sharpe_idx]
    max_sharpe_risk = volatility_array[max_sharpe_idx]
    max_sharpe_ratio = sharpe_ratios[max_sharpe_idx]
    
    # Find min volatility portfolio (leftmost point)
    min_vol_idx = np.argmin(volatility_array)
    
    # Capital Market Line (CML) - tangent line from risk-free rate through max Sharpe portfolio
    cml_x = np.linspace(0, max_sharpe_risk * 1.5, 100)
    cml_y = risk_free_rate + max_sharpe_ratio * cml_x
    
    # Create plot
    fig = go.Figure()
    
    # Plot Capital Market Line (green dashed line)
    fig.add_trace(go.Scatter(
        x=cml_x,
        y=cml_y,
        mode='lines',
        name='Capital Market Line (CML)',
        line=dict(color='green', width=2, dash='dash'),
        hovertemplate='CML<br>Risk: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>'
    ))
    
    # Plot efficient frontier curve (smooth blue line)
    fig.add_trace(go.Scatter(
        x=volatility_array,
        y=returns_array,
        mode='lines',
        name='Efficient Frontier',
        line=dict(color='blue', width=3),
        hovertemplate='Efficient Frontier<br>Risk: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>'
    ))
    
    # Highlight Max Sharpe portfolio on efficient frontier
    fig.add_trace(go.Scatter(
        x=[max_sharpe_risk],
        y=[max_sharpe_return],
        mode='markers',
        name='Max Sharpe (Tangency)',
        marker=dict(
            size=20,
            color='red',
            symbol='circle',
            line=dict(width=3, color='darkred')
        ),
        hovertemplate=f'Max Sharpe Portfolio<br>Return: {max_sharpe_return:.2%}<br>Risk: {max_sharpe_risk:.2%}<br>Sharpe: {max_sharpe_ratio:.3f}<extra></extra>'
    ))
    
    # Highlight Minimum Volatility portfolio
    fig.add_trace(go.Scatter(
        x=[volatility_array[min_vol_idx]],
        y=[returns_array[min_vol_idx]],
        mode='markers',
        name='Min Volatility',
        marker=dict(
            size=18,
            color='green',
            symbol='circle',
            line=dict(width=3, color='darkgreen')
        ),
        hovertemplate=f'Min Volatility<br>Return: {returns_array[min_vol_idx]:.2%}<br>Risk: {volatility_array[min_vol_idx]:.2%}<extra></extra>'
    ))
    
    # Add strategy portfolios (larger, fully opaque for visibility)
    # Max Sharpe is shown as the tangency point on EF
    # Other strategies (including Min Variance) shown separately
    strategy_colors = {
        'Max Sharpe': 'red',
        'Min Variance': 'orange', 
        'Equal Weight': 'purple',
        'Risk Parity': 'brown',
        'Current Portfolio': 'cyan'
    }
    
    for strategy_name, portfolio in portfolios.items():
        # Skip only Max Sharpe - it's already shown as the tangency point
        if strategy_name == 'Max Sharpe':
            continue
            
        ret = portfolio['return']
        vol = portfolio['volatility']
        color = strategy_colors.get(strategy_name, 'gray')
        
        fig.add_trace(go.Scatter(
            x=[vol],
            y=[ret],
            mode='markers',
            name=strategy_name,
            marker=dict(
                size=16,  # Larger for visibility
                color=color, 
                symbol='circle',
                line=dict(width=2, color='black'),  # Black outline
                opacity=1.0  # Fully opaque
            ),
            hovertemplate=f'{strategy_name}<br>Return: {ret:.2%}<br>Risk: {vol:.2%}<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(
            text=f'Efficient Frontier - {name}',
            x=0.5,
            xanchor='center',
            font=dict(size=16)
        ),
        xaxis_title='Volatility (Risk)',
        yaxis_title='Expected Return',
        hovermode='closest',
        template='plotly_white',
        height=600,
        width=800,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='gray',
            borderwidth=1
        ),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    # Format axes as percentages
    fig.update_xaxes(tickformat='.1%', gridcolor='lightgray')
    fig.update_yaxes(tickformat='.1%', gridcolor='lightgray')
    
    fig.show()


# ==================== ANALYSIS FUNCTION ====================

def run_portfolio_analysis(name, tickers, current_weights_dict, bond_tickers, max_stock_pct, max_bond_pct, 
                          price_data, stock_info, mean_returns, returns, cov_matrix, risk_free_rate, max_weight, benchmark_data=None):
    """
    Run complete portfolio analysis for a given set of holdings
    Returns portfolios dict and backtest results
    """
    
    # Get actual period info from price data
    start_date = price_data.index[0]
    end_date = price_data.index[-1]
    num_years = (end_date - start_date).days / 365.25
    num_trading_days = len(price_data)
    
    print(f"\n{'=' * 90}")
    print(f"{name} - HOLDINGS ANALYSIS")
    print(f"Period: {start_date.date()} to {end_date.date()} ({num_years:.2f} years, {num_trading_days} trading days)")
    print(f"{'=' * 90}\n")
    
    holdings_df = create_holdings_table(tickers, stock_info, mean_returns, returns)
    print(tabulate(holdings_df, headers='keys', tablefmt='grid', showindex=False))
    
    # Calculate strategies
    portfolios = {}
    
    # Max Sharpe
    max_sharpe = max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate, max_weight)
    ret, vol = calculate_portfolio_performance(max_sharpe.x, mean_returns, cov_matrix)
    portfolios['Max Sharpe Ratio'] = {
        'weights': dict(zip(tickers, max_sharpe.x)),
        'return': ret,
        'volatility': vol,
        'sharpe': (ret - risk_free_rate) / vol
    }
    
    # Min Variance
    min_var = min_variance(mean_returns, cov_matrix, max_weight)
    ret, vol = calculate_portfolio_performance(min_var.x, mean_returns, cov_matrix)
    portfolios['Min Variance'] = {
        'weights': dict(zip(tickers, min_var.x)),
        'return': ret,
        'volatility': vol
    }
    
    # Equal Weight
    equal_w = equal_weight_portfolio(len(tickers))
    ret, vol = calculate_portfolio_performance(equal_w.x, mean_returns, cov_matrix)
    portfolios['Equal Weight'] = {
        'weights': dict(zip(tickers, equal_w.x)),
        'return': ret,
        'volatility': vol
    }
    
    # Risk Parity
    risk_par = risk_parity_portfolio(
        cov_matrix, 
        tickers=tickers, 
        bond_tickers=bond_tickers,
        max_stock_weight=max_stock_pct,
        max_bond_weight=max_bond_pct
    )
    ret, vol = calculate_portfolio_performance(risk_par.x, mean_returns, cov_matrix)
    portfolios['Risk Parity'] = {
        'weights': dict(zip(tickers, risk_par.x)),
        'return': ret,
        'volatility': vol
    }
    
    # Current Portfolio
    if current_weights_dict:
        current_weights_array = np.array([current_weights_dict.get(t, 0.0) for t in tickers])
        ret, vol = calculate_portfolio_performance(current_weights_array, mean_returns, cov_matrix)
        portfolios['Current Portfolio'] = {
            'weights': current_weights_dict,
            'return': ret,
            'volatility': vol,
            'sharpe': (ret - risk_free_rate) / vol
        }
    
    # Display strategy comparison
    print(f"\n{'=' * 90}")
    print(f"{name} - PORTFOLIO STRATEGIES COMPARISON")
    print(f"{'=' * 90}\n")
    
    comparison_df = create_strategy_comparison_table(portfolios, risk_free_rate)
    print(tabulate(comparison_df, headers='keys', tablefmt='grid', showindex=False))
    
    # Display weights
    print(f"\n{'=' * 90}")
    print(f"{name} - PORTFOLIO WEIGHTS")
    print(f"{'=' * 90}\n")
    
    weights_df = create_weights_table(portfolios, tickers)
    print(tabulate(weights_df, headers='keys', tablefmt='grid', showindex=False))
    
    # Display correlation matrix
    print(f"\n{'=' * 90}")
    print(f"{name} - CORRELATION MATRIX")
    print(f"{'=' * 90}")
    print("Shows how assets move together: 1.0 = perfect correlation, 0.0 = no correlation, -1.0 = inverse")
    print()
    
    corr_df = create_correlation_matrix_table(cov_matrix, tickers)
    print(tabulate(corr_df, headers='keys', tablefmt='grid', showindex=False))
    
    # Backtest all strategies
    print(f"\n{'=' * 90}")
    print(f"{name} - HISTORICAL BACKTESTING")
    print(f"{'=' * 90}\n")
    
    all_backtest_results = {}
    
    for strategy_name, portfolio in portfolios.items():
        weights = np.array([portfolio['weights'].get(t, 0.0) for t in tickers])
        all_backtest_results[strategy_name] = {}
        
        for rebal_freq in ['never', 'monthly', 'quarterly', 'annually']:
            results = backtest_with_rebalancing(weights, price_data, benchmark_data, rebal_freq, risk_free_rate)
            all_backtest_results[strategy_name][rebal_freq] = results
        
        # Print individual strategy table
        print(f"\n{strategy_name}:")
        print("-" * 90)
        strategy_df = create_strategy_backtest_table(strategy_name, all_backtest_results[strategy_name])
        print(tabulate(strategy_df, headers='keys', tablefmt='grid', showindex=False))
    
    return portfolios, all_backtest_results

# ==================== MAIN ====================

def main():
    print("=" * 90)
    print("PORTFOLIO OPTIMIZATION TOOL - SEPARATE ANALYSIS")
    print("=" * 90)
    print()
    
    # Get benchmark ticker
    print("=" * 90)
    print("BENCHMARK SELECTION")
    print("=" * 90)
    print("\nEnter a benchmark ticker to compare against (e.g., SPY, QQQ, VTI)")
    print("This will be used to calculate Alpha and Beta for your portfolios")
    
    benchmark_ticker = input("\nBenchmark ticker [default: SPY]: ").strip().upper() or 'SPY'
    print(f"\n> Using {benchmark_ticker} as benchmark")
    
    # Get YOUR portfolio inputs
    print("\n" + "=" * 90)
    print("YOUR PORTFOLIO SETUP")
    print("=" * 90)
    
    tickers_input = input("\nEnter YOUR stock tickers (comma-separated): ")
    your_tickers = [t.strip().upper() for t in tickers_input.split(',')]
    
    period = input("Enter time period (1mo, 3mo, 6mo, 1y, 2y, 5y) [default: 1y]: ").strip() or '1y'
    
    max_weight_input = input("YOUR max position size per stock (e.g., 0.40) [press Enter to skip]: ").strip()
    your_max_weight = float(max_weight_input) if max_weight_input else None
    
    your_current_weights_dict = {}
    current_input = input("\nEnter YOUR current weights? (yes/no): ").strip().lower()
    
    if current_input == 'yes':
        print("\nEnter YOUR CURRENT weights:")
        for ticker in your_tickers:
            weight_input = input(f"  {ticker}: ").strip()
            if weight_input:
                try:
                    your_current_weights_dict[ticker] = float(weight_input)
                except:
                    pass
        
        if your_current_weights_dict:
            total = sum(your_current_weights_dict.values())
            print(f"\n  > Your allocation total: {total*100:.2f}%")
    
    bond_input = input("\nEnter YOUR bond/ETF tickers (e.g., TLT,AGG,BND) [press Enter to skip]: ").strip()
    your_bond_tickers = [b.strip().upper() for b in bond_input.split(',')] if bond_input else []
    
    your_max_stock_pct = None
    your_max_bond_pct = None
    
    if your_bond_tickers:
        stock_pct_input = input("YOUR max allocation to stocks (e.g., 0.60) [press Enter for no limit]: ").strip()
        your_max_stock_pct = float(stock_pct_input) if stock_pct_input else None
        
        bond_pct_input = input("YOUR max allocation to bonds (e.g., 0.40) [press Enter for no limit]: ").strip()
        your_max_bond_pct = float(bond_pct_input) if bond_pct_input else None
    
    # Get PROSPECT portfolio inputs
    print("\n" + "=" * 90)
    print("PROSPECT PORTFOLIO SETUP")
    print("=" * 90)
    
    prospect_input = input("\nDo you want to analyze a PROSPECT portfolio? (yes/no): ").strip().lower()
    
    prospect_tickers = []
    prospect_current_weights_dict = {}
    prospect_bond_tickers = []
    prospect_max_weight = None
    prospect_max_stock_pct = None
    prospect_max_bond_pct = None
    
    if prospect_input == 'yes':
        tickers_input = input("\nEnter PROSPECT's tickers (comma-separated): ")
        prospect_tickers = [t.strip().upper() for t in tickers_input.split(',')]
        
        max_weight_input = input("PROSPECT's max position size per stock (e.g., 0.40) [press Enter to skip]: ").strip()
        prospect_max_weight = float(max_weight_input) if max_weight_input else None
        
        print("\nEnter PROSPECT's CURRENT weights:")
        for ticker in prospect_tickers:
            weight_input = input(f"  {ticker}: ").strip()
            if weight_input:
                try:
                    prospect_current_weights_dict[ticker] = float(weight_input)
                except:
                    pass
        
        if prospect_current_weights_dict:
            total = sum(prospect_current_weights_dict.values())
            print(f"\n  > Prospect allocation total: {total*100:.2f}%")
        
        bond_input = input("\nEnter PROSPECT's bond/ETF tickers [press Enter to skip]: ").strip()
        prospect_bond_tickers = [b.strip().upper() for b in bond_input.split(',')] if bond_input else []
        
        if prospect_bond_tickers:
            stock_pct_input = input("PROSPECT's max allocation to stocks (e.g., 0.60) [press Enter for no limit]: ").strip()
            prospect_max_stock_pct = float(stock_pct_input) if stock_pct_input else None
            
            bond_pct_input = input("PROSPECT's max allocation to bonds (e.g., 0.40) [press Enter for no limit]: ").strip()
            prospect_max_bond_pct = float(bond_pct_input) if bond_pct_input else None
    
    # Get Treasury rate
    print(f"\n> Downloading data...")
    risk_free_rate = get_current_treasury_rate()
    
    # Download benchmark data
    print(f"\n> Downloading benchmark data ({benchmark_ticker})...")
    try:
        benchmark_price_data = download_stock_data([benchmark_ticker], period)
        benchmark_price_data = benchmark_price_data[benchmark_ticker] if benchmark_ticker in benchmark_price_data.columns else benchmark_price_data.iloc[:, 0]
        print(f"> Benchmark data: {benchmark_price_data.index[0].date()} to {benchmark_price_data.index[-1].date()}")
    except Exception as e:
        print(f"> WARNING: Could not download benchmark {benchmark_ticker}: {e}")
        print(f"> Continuing without benchmark (Alpha/Beta will not be calculated)")
        benchmark_price_data = None
    
    # Download data for YOUR portfolio
    print(f"\n> Downloading YOUR portfolio data for {period}...")
    your_price_data = download_stock_data(your_tickers, period)
    your_tickers = list(your_price_data.columns)
    
    print(f"> Data period: {your_price_data.index[0].date()} to {your_price_data.index[-1].date()}")
    print(f"> Trading days: {len(your_price_data)}")
    
    print(f"> Fetching company information...")
    your_stock_info = get_stock_info(your_tickers)
    
    your_returns = your_price_data.pct_change().dropna()
    your_mean_returns = your_returns.mean()
    your_cov_matrix = your_returns.cov()
    
    # Align benchmark to YOUR portfolio dates
    your_benchmark_data = None
    if benchmark_price_data is not None:
        your_benchmark_data = benchmark_price_data.reindex(your_price_data.index, method='ffill').dropna()
        print(f"> Aligned benchmark to YOUR dates: {len(your_benchmark_data)} days")
    
    # Run YOUR analysis
    your_portfolios, your_backtest_results = run_portfolio_analysis(
        "YOUR PORTFOLIO",
        your_tickers,
        your_current_weights_dict,
        your_bond_tickers,
        your_max_stock_pct,
        your_max_bond_pct,
        your_price_data,
        your_stock_info,
        your_mean_returns,
        your_returns,
        your_cov_matrix,
        risk_free_rate,
        your_max_weight,
        your_benchmark_data
    )
    
    # Show YOUR efficient frontier
    print(f"\n{'=' * 90}")
    show_your_chart = input("\nWould you like to see YOUR portfolio efficient frontier? (yes/no): ").strip().lower()
    
    if show_your_chart == 'yes':
        print("\n> Generating YOUR efficient frontier chart...")
        plot_efficient_frontier(
            "YOUR PORTFOLIO",
            your_tickers,
            your_mean_returns,
            your_cov_matrix,
            your_portfolios,
            your_returns,
            risk_free_rate
        )
    
    # Run PROSPECT analysis if provided
    if prospect_tickers:
        print(f"\n\n{'=' * 90}")
        print("PROSPECT PORTFOLIO ANALYSIS")
        print(f"{'=' * 90}")
        
        print(f"\n> Downloading PROSPECT portfolio data for {period}...")
        prospect_price_data = download_stock_data(prospect_tickers, period)
        prospect_tickers = list(prospect_price_data.columns)
        
        print(f"> Data period: {prospect_price_data.index[0].date()} to {prospect_price_data.index[-1].date()}")
        print(f"> Trading days: {len(prospect_price_data)}")
        
        # CRITICAL: Force PROSPECT to use EXACT same date range as YOUR portfolio
        print(f"> Forcing alignment to YOUR portfolio date range...")
        print(f"> YOUR portfolio dates: {your_price_data.index[0].date()} to {your_price_data.index[-1].date()} ({len(your_price_data)} days)")
        
        # Reindex prospect data to match YOUR dates exactly
        prospect_price_data_aligned = pd.DataFrame(index=your_price_data.index)
        
        for ticker in prospect_tickers:
            if ticker in prospect_price_data.columns:
                # Reindex this ticker to YOUR dates, forward filling gaps
                prospect_price_data_aligned[ticker] = prospect_price_data[ticker].reindex(
                    your_price_data.index, 
                    method='ffill'
                )
        
        # Drop any tickers that are all NaN (no data in this period)
        prospect_price_data_aligned = prospect_price_data_aligned.dropna(axis=1, how='all')
        prospect_tickers = list(prospect_price_data_aligned.columns)
        
        # Check if we still have data after alignment
        if len(prospect_price_data_aligned.columns) == 0:
            print("> ERROR: No prospect tickers have data in YOUR portfolio's date range!")
            return
        
        # Replace prospect_price_data with aligned version
        prospect_price_data = prospect_price_data_aligned
        
        print(f"> Aligned prospect data: {prospect_price_data.index[0].date()} to {prospect_price_data.index[-1].date()}")
        print(f"> Aligned trading days: {len(prospect_price_data)} (should match YOUR: {len(your_price_data)})")
        print(f"> Tickers after alignment: {prospect_tickers}")
        
        # Check for NaN values that might cause issues in backtest
        nan_counts = prospect_price_data.isna().sum()
        if nan_counts.sum() > 0:
            print(f"> WARNING: NaN values found in aligned data:")
            for ticker in nan_counts[nan_counts > 0].index:
                pct_nan = (nan_counts[ticker] / len(prospect_price_data)) * 100
                print(f">   {ticker}: {nan_counts[ticker]} NaN values ({pct_nan:.1f}% of data)")
            
            # Find the first date where ALL tickers have valid data
            first_valid_idx = prospect_price_data.apply(lambda x: x.first_valid_index()).max()
            
            if first_valid_idx != prospect_price_data.index[0]:
                print(f"> Some tickers don't have data until: {first_valid_idx.date()}")
                print(f"> Options:")
                print(f">   1. Drop those tickers")
                print(f">   2. Only use period where all tickers have data")
                print(f"> Choosing option 2: Using common data period for fair comparison")
                
                # Trim to period where all tickers have data
                prospect_price_data = prospect_price_data.loc[first_valid_idx:]
                
                print(f"> PROSPECT final data range: {prospect_price_data.index[0].date()} to {prospect_price_data.index[-1].date()}")
                print(f"> PROSPECT final trading days: {len(prospect_price_data)}")
                
                # Calculate actual years
                actual_years = (prospect_price_data.index[-1] - prospect_price_data.index[0]).days / 365.25
                print(f"> PROSPECT analysis period: {actual_years:.2f} years")
                print(f"> NOTE: This is shorter than YOUR portfolio due to newer tickers in PROSPECT")
        
        print(f"> Fetching company information...")
        prospect_stock_info = get_stock_info(prospect_tickers)
        
        prospect_returns = prospect_price_data.pct_change().dropna()
        prospect_mean_returns = prospect_returns.mean()
        prospect_cov_matrix = prospect_returns.cov()
        
        # Align benchmark to PROSPECT portfolio dates
        prospect_benchmark_data = None
        if benchmark_price_data is not None:
            prospect_benchmark_data = benchmark_price_data.reindex(prospect_price_data.index, method='ffill').dropna()
            print(f"> Aligned benchmark to PROSPECT dates: {len(prospect_benchmark_data)} days")
        
        prospect_portfolios, prospect_backtest_results = run_portfolio_analysis(
            "PROSPECT PORTFOLIO",
            prospect_tickers,
            prospect_current_weights_dict,
            prospect_bond_tickers,
            prospect_max_stock_pct,
            prospect_max_bond_pct,
            prospect_price_data,
            prospect_stock_info,
            prospect_mean_returns,
            prospect_returns,
            prospect_cov_matrix,
            risk_free_rate,
            prospect_max_weight,
            prospect_benchmark_data
        )
        
        # Show PROSPECT efficient frontier
        print(f"\n{'=' * 90}")
        show_prospect_chart = input("\nWould you like to see PROSPECT portfolio efficient frontier? (yes/no): ").strip().lower()
        
        if show_prospect_chart == 'yes':
            print("\n> Generating PROSPECT efficient frontier chart...")
            plot_efficient_frontier(
                "PROSPECT PORTFOLIO",
                prospect_tickers,
                prospect_mean_returns,
                prospect_cov_matrix,
                prospect_portfolios,
                prospect_returns,
                risk_free_rate
            )
        
        # Value proposition comparison
        print(f"\n\n{'=' * 90}")
        print("VALUE PROPOSITION - YOUR vs PROSPECT")
        print(f"{'=' * 90}\n")
        
        print("Comparing best strategies (Quarterly Rebalancing):\n")
        
        comparison_data = []
        
        for strategy in ['Max Sharpe Ratio', 'Current Portfolio']:
            if strategy in your_backtest_results and strategy in prospect_backtest_results:
                your_results = your_backtest_results[strategy]['quarterly']
                prospect_results = prospect_backtest_results[strategy]['quarterly']
                
                comparison_data.append({
                    'Strategy': strategy,
                    'Your Return': f"{your_results['annualized_return']*100:.1f}%",
                    'Prospect Return': f"{prospect_results['annualized_return']*100:.1f}%",
                    'Your Sharpe': f"{your_results['sharpe_ratio']:.3f}",
                    'Prospect Sharpe': f"{prospect_results['sharpe_ratio']:.3f}",
                    'Your Alpha': f"{your_results.get('alpha', 0)*100:.2f}%" if your_results.get('alpha') is not None else 'N/A',
                    'Prospect Alpha': f"{prospect_results.get('alpha', 0)*100:.2f}%" if prospect_results.get('alpha') is not None else 'N/A',
                    'Your Beta': f"{your_results.get('beta', 0):.2f}" if your_results.get('beta') is not None else 'N/A',
                    'Prospect Beta': f"{prospect_results.get('beta', 0):.2f}" if prospect_results.get('beta') is not None else 'N/A',
                    'Your Drawdown': f"{your_results['max_drawdown']*100:.1f}%",
                    'Prospect Drawdown': f"{prospect_results['max_drawdown']*100:.1f}%"
                })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            print(tabulate(comparison_df, headers='keys', tablefmt='grid', showindex=False))
    
    print(f"\n\n{'=' * 90}")
    print("ANALYSIS COMPLETE")
    print(f"{'=' * 90}\n")

if __name__ == '__main__':
    main()

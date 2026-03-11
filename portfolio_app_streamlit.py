"""
Portfolio Optimization Web App using Streamlit
Run with: streamlit run portfolio_app_streamlit.py
"""

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import plotly.graph_objects as go
from datetime import datetime
import sys
import os

# Import functions from your portfolio optimizer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from portfolio_ef import (
    get_current_treasury_rate,
    download_stock_data,
    get_stock_info,
    calculate_alpha_beta,
    calculate_portfolio_performance,
    max_sharpe_ratio,
    min_variance,
    equal_weight_portfolio,
    risk_parity_portfolio,
    backtest_with_rebalancing,
    calculate_efficient_frontier,
    plot_efficient_frontier
)

# Page config
st.set_page_config(
    page_title="Portfolio Optimizer",
    page_icon="📊",
    layout="wide"
)

# Initialize session state for weights
if 'weights_submitted' not in st.session_state:
    st.session_state.weights_submitted = False
if 'current_weights' not in st.session_state:
    st.session_state.current_weights = {}
if 'prospect_weights_submitted' not in st.session_state:
    st.session_state.prospect_weights_submitted = False
if 'prospect_weights' not in st.session_state:
    st.session_state.prospect_weights = {}
if 'analysis_started' not in st.session_state:
    st.session_state.analysis_started = False

# Title
st.title("Portfolio Optimization Tool")
st.markdown("### Advanced Portfolio Analysis with Modern Portfolio Theory")

# Sidebar for inputs
st.sidebar.header("Portfolio Configuration")

# Benchmark selection
benchmark = st.sidebar.text_input("Benchmark Ticker", value="SPY", help="e.g., SPY, QQQ, VTI")

# YOUR Portfolio Section
st.sidebar.subheader("YOUR Portfolio")
your_tickers_input = st.sidebar.text_input(
    "Stock Tickers (comma-separated)", 
    value="AAPL, MSFT, GOOGL, AMZN, JPM, JNJ, XOM, PG",
    key="your_tickers"
)

period = st.sidebar.selectbox("Time Period", ["1y", "2y", "3y", "4y", "5y"], index=3)

your_max_weight = st.sidebar.number_input(
    "Max Position Size (%)", 
    min_value=1.0, 
    max_value=100.0, 
    value=17.0, 
    step=0.01, 
    format="%.2f",
    key="your_max"
) / 100

your_bond_tickers = st.sidebar.text_input("Bond/ETF Tickers (optional)", value="AGG", key="your_bonds")

# Current portfolio weights (optional)
use_current_weights = st.sidebar.checkbox("Include Current Portfolio weights?")

# Show reset button if weights have been submitted
if use_current_weights and st.session_state.weights_submitted:
    if st.sidebar.button("Reset Weights", help="Clear submitted weights and enter new ones"):
        st.session_state.weights_submitted = False
        st.session_state.current_weights = {}
        st.rerun()

col1, col2 = st.sidebar.columns(2)
with col1:
    your_max_stock = st.number_input(
        "Max Stocks %", 
        min_value=0.0, 
        max_value=100.0, 
        value=90.0, 
        step=0.01, 
        format="%.2f",
        key="your_stock_max"
    ) / 100
with col2:
    your_max_bond = st.number_input(
        "Max Bonds %", 
        min_value=0.0, 
        max_value=100.0, 
        value=10.0, 
        step=0.01, 
        format="%.2f",
        key="your_bond_max"
    ) / 100

# PROSPECT Portfolio (optional)
st.sidebar.subheader("PROSPECT Portfolio (Optional)")
analyze_prospect = st.sidebar.checkbox("Analyze PROSPECT Portfolio")

prospect_tickers_input = None
use_prospect_current_weights = False
if analyze_prospect:
    prospect_tickers_input = st.sidebar.text_input(
        "PROSPECT Tickers", 
        value="NVDA, AMD, AVGO, ASML, TSM, QCOM, MU, INTC",
        key="prospect_tickers"
    )
    prospect_max_weight = st.sidebar.number_input(
        "PROSPECT Max Position %", 
        min_value=1.0, 
        max_value=100.0, 
        value=17.0, 
        step=0.01, 
        format="%.2f",
        key="prospect_max"
    ) / 100
    prospect_bond_tickers = st.sidebar.text_input("PROSPECT Bonds", value="BND", key="prospect_bonds")
    
    col3, col4 = st.sidebar.columns(2)
    with col3:
        prospect_max_stock = st.number_input(
            "PROSPECT Max Stocks %", 
            min_value=0.0, 
            max_value=100.0, 
            value=90.0, 
            step=0.01, 
            format="%.2f",
            key="prospect_stock_max"
        ) / 100
    with col4:
        prospect_max_bond = st.number_input(
            "PROSPECT Max Bonds %", 
            min_value=0.0, 
            max_value=100.0, 
            value=10.0, 
            step=0.01, 
            format="%.2f",
            key="prospect_bond_max"
        ) / 100
    
    # PROSPECT current weights option
    use_prospect_current_weights = st.sidebar.checkbox("Include PROSPECT Current Portfolio weights?", key="prospect_use_current")
    
    # Show reset button if PROSPECT weights have been submitted
    if use_prospect_current_weights and st.session_state.prospect_weights_submitted:
        if st.sidebar.button("Reset PROSPECT Weights", help="Clear submitted PROSPECT weights and enter new ones"):
            st.session_state.prospect_weights_submitted = False
            st.session_state.prospect_weights = {}
            st.rerun()

# Run Analysis Button
if st.sidebar.button("Run Analysis", type="primary", use_container_width=True):
    st.session_state.analysis_started = True

# Add reset button if analysis has been started
if st.session_state.analysis_started:
    if st.sidebar.button("Start New Analysis", help="Clear all data and start fresh"):
        st.session_state.analysis_started = False
        st.session_state.weights_submitted = False
        st.session_state.current_weights = {}
        st.session_state.prospect_weights_submitted = False
        st.session_state.prospect_weights = {}
        st.rerun()

# Main content area
if st.session_state.analysis_started:
    try:
        # Parse tickers
        your_tickers = [t.strip().upper() for t in your_tickers_input.split(',')]
        
        # Initialize current_weights_dict
        current_weights_dict = {}
        
        # If using current weights, collect them first
        if use_current_weights:
            # Check if we already have submitted weights
            if st.session_state.weights_submitted and st.session_state.current_weights:
                # Use previously submitted weights
                current_weights_dict = st.session_state.current_weights.copy()
                st.info(f"Using Current Portfolio weights (Total: {sum(current_weights_dict.values())*100:.1f}%)")
            else:
                # Need to collect weights - show form
                st.header("Step 1: Enter Your Current Portfolio Weights")
                st.markdown("Please enter the percentage allocation for each ticker, then click Submit to continue.")
                
                with st.form("current_weights_form", clear_on_submit=False):
                    cols = st.columns(4)
                    temp_weights = {}
                    
                    for idx, ticker in enumerate(your_tickers):
                        with cols[idx % 4]:
                            default_val = st.session_state.current_weights.get(ticker, 0.0) * 100
                            weight = st.number_input(
                                f"{ticker} (%)", 
                                min_value=0.0, 
                                max_value=100.0, 
                                value=default_val, 
                                step=0.01,
                                format="%.2f",
                                key=f"weight_{ticker}"
                            ) / 100
                            temp_weights[ticker] = weight
                    
                    # Show current total
                    current_total = sum(temp_weights.values()) * 100
                    st.markdown(f"**Current Total: {current_total:.2f}%**")
                    
                    submitted = st.form_submit_button("Submit Weights & Continue to Analysis", type="primary")
                    
                if submitted:
                    total_weight = sum(temp_weights.values())
                    
                    if abs(total_weight - 1.0) > 0.01:
                        st.error(f"Weights must sum to 100% (currently {total_weight*100:.2f}%). Please adjust and click Submit again.")
                        st.stop()
                    else:
                        st.session_state.current_weights = temp_weights.copy()
                        st.session_state.weights_submitted = True
                        current_weights_dict = temp_weights.copy()
                        st.success("Weights saved successfully! Continuing to analysis...")
                        # Continue to analysis below
                
                if not submitted:
                    # If form not submitted yet, stop here
                    st.info("Please fill in the weights above and click Submit to continue.")
                    st.stop()
        
        # Continue with analysis
        st.header("Portfolio Analysis Results")
        
        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Download data
        status_text.text("Downloading data...")
        progress_bar.progress(10)
        
        risk_free_rate = get_current_treasury_rate()
        
        # Download benchmark
        benchmark_price_data = download_stock_data([benchmark], period)
        benchmark_price_data = benchmark_price_data[benchmark] if benchmark in benchmark_price_data.columns else benchmark_price_data.iloc[:, 0]
        
        progress_bar.progress(20)
        
        # Download YOUR portfolio data
        status_text.text("Downloading YOUR portfolio data...")
        your_price_data = download_stock_data(your_tickers, period)
        your_tickers = list(your_price_data.columns)
        
        progress_bar.progress(30)
        
        # Get stock info
        your_stock_info = get_stock_info(your_tickers)
        
        # Calculate returns
        your_returns = your_price_data.pct_change().dropna()
        your_mean_returns = your_returns.mean()
        your_cov_matrix = your_returns.cov()
        
        progress_bar.progress(50)
        
        # Align benchmark
        your_benchmark_data = benchmark_price_data.reindex(your_price_data.index, method='ffill').dropna()
        
        # Calculate benchmark return for display
        benchmark_total_return = (your_benchmark_data.iloc[-1] / your_benchmark_data.iloc[0]) - 1
        num_years = (your_benchmark_data.index[-1] - your_benchmark_data.index[0]).days / 365.25
        benchmark_cagr = (1 + benchmark_total_return) ** (1 / num_years) - 1
        
        # Debug output
        print(f"Benchmark ({benchmark}) Calculation:")
        print(f"  Start Date: {your_benchmark_data.index[0]}")
        print(f"  End Date: {your_benchmark_data.index[-1]}")
        print(f"  Start Price: ${your_benchmark_data.iloc[0]:.2f}")
        print(f"  End Price: ${your_benchmark_data.iloc[-1]:.2f}")
        print(f"  Total Return: {benchmark_total_return*100:.2f}%")
        print(f"  Time Period: {num_years:.2f} years")
        print(f"  CAGR: {benchmark_cagr*100:.2f}%")
        
        # Display key metrics at the top
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Benchmark", benchmark, delta=None)
        with col2:
            st.metric("Benchmark CAGR", f"{benchmark_cagr*100:.2f}%")
        with col3:
            st.metric("10Y Treasury (Current)", f"{risk_free_rate*100:.2f}%")
        
        st.markdown("---")
        
        # Calculate strategies
        status_text.text("Calculating optimal portfolios...")
        
        portfolios = {}
        
        # Max Sharpe
        result = max_sharpe_ratio(your_mean_returns, your_cov_matrix, risk_free_rate, your_max_weight)
        weights = result.x  # Extract the weights array from scipy Result
        ret, vol = calculate_portfolio_performance(weights, your_mean_returns, your_cov_matrix)
        portfolios['Max Sharpe'] = {
            'weights': {t: w for t, w in zip(your_tickers, weights)},
            'return': ret,
            'volatility': vol
        }
        
        # Min Variance
        result = min_variance(your_mean_returns, your_cov_matrix, your_max_weight)
        weights = result.x  # Extract the weights array from scipy Result
        ret, vol = calculate_portfolio_performance(weights, your_mean_returns, your_cov_matrix)
        portfolios['Min Variance'] = {
            'weights': {t: w for t, w in zip(your_tickers, weights)},
            'return': ret,
            'volatility': vol
        }
        
        # Equal Weight
        result = equal_weight_portfolio(len(your_tickers))
        weights = result.x  # Extract the weights array
        ret, vol = calculate_portfolio_performance(weights, your_mean_returns, your_cov_matrix)
        portfolios['Equal Weight'] = {
            'weights': {t: w for t, w in zip(your_tickers, weights)},
            'return': ret,
            'volatility': vol
        }
        
        # Risk Parity
        bond_tickers_list = [t.strip().upper() for t in your_bond_tickers.split(',') if t.strip()]
        result = risk_parity_portfolio(
            your_cov_matrix, 
            your_tickers, 
            bond_tickers_list, 
            your_max_stock, 
            your_max_bond
        )
        weights = result.x  # Extract the weights array
        ret, vol = calculate_portfolio_performance(weights, your_mean_returns, your_cov_matrix)
        portfolios['Risk Parity'] = {
            'weights': {t: w for t, w in zip(your_tickers, weights)},
            'return': ret,
            'volatility': vol
        }
        
        # Current Portfolio (if weights provided)
        if use_current_weights and current_weights_dict:
            total_weight = sum(current_weights_dict.values())
            if abs(total_weight - 1.0) < 0.01:  # Check if weights sum to 100%
                current_weights_array = np.array([current_weights_dict.get(t, 0) for t in your_tickers])
                ret, vol = calculate_portfolio_performance(current_weights_array, your_mean_returns, your_cov_matrix)
                portfolios['Current Portfolio'] = {
                    'weights': current_weights_dict,
                    'return': ret,
                    'volatility': vol
                }
        
        progress_bar.progress(70)
        
        # Calculate Alpha and Beta for each strategy using Expected Returns
        status_text.text("Calculating Alpha and Beta...")
        
        # Calculate benchmark returns
        benchmark_returns = your_benchmark_data.pct_change().dropna()
        
        # Store backtest results for all strategies and rebalancing frequencies
        backtest_results = {}
        
        for strategy_name, portfolio in portfolios.items():
            weights = np.array([portfolio['weights'].get(t, 0) for t in your_tickers])
            
            # Calculate Alpha/Beta using Expected Return (not backtest return)
            # Get portfolio daily returns based on weights
            portfolio_daily_returns = (your_returns * weights).sum(axis=1)
            
            # Calculate both Alphas and Beta using the strategy's Expected Return
            jensens_alpha, excess_alpha, beta = calculate_alpha_beta(
                portfolio_daily_returns,
                benchmark_returns,
                risk_free_rate,
                portfolio_expected_return=portfolio['return']  # Use Expected Return from strategy
            )
            
            portfolio['jensens_alpha'] = jensens_alpha
            portfolio['excess_alpha'] = excess_alpha
            portfolio['beta'] = beta
            
            # Run backtests for display
            backtest_results[strategy_name] = {}
            
            # Run backtest for all rebalancing frequencies
            for rebal_freq in ['never', 'monthly', 'quarterly', 'annually']:
                result = backtest_with_rebalancing(
                    weights, 
                    your_price_data, 
                    your_benchmark_data, 
                    rebal_freq, 
                    risk_free_rate
                )
                backtest_results[strategy_name][rebal_freq] = result
        
        # Display results
        status_text.text("Generating results...")
        
        # Strategy Comparison Table
        st.header("Portfolio Strategies Comparison")
        
        comparison_data = []
        for name, portfolio in portfolios.items():
            row = {
                'Strategy': name,
                'Expected Return': f"{portfolio['return']*100:.2f}%",
                'St Dev': f"{portfolio['volatility']*100:.2f}%",
                'Sharpe Ratio': f"{(portfolio['return'] - risk_free_rate) / portfolio['volatility']:.3f}"
            }
            
            if portfolio.get('jensens_alpha') is not None:
                row["Jensen's Alpha"] = f"{portfolio['jensens_alpha']*100:.2f}%"
            if portfolio.get('excess_alpha') is not None:
                row['Excess Alpha'] = f"{portfolio['excess_alpha']*100:.2f}%"
            if portfolio.get('beta') is not None:
                row['Beta'] = f"{portfolio['beta']:.2f}"
            
            comparison_data.append(row)
        
        # Create DataFrame and reset index to start at 1
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.index = comparison_df.index + 1
        st.dataframe(comparison_df, use_container_width=True)
        
        progress_bar.progress(75)
        
        # Portfolio Weights
        st.header("Portfolio Weights")
        
        weights_data = []
        for ticker in your_tickers:
            row = {'Ticker': ticker}
            for name, portfolio in portfolios.items():
                row[name] = f"{portfolio['weights'].get(ticker, 0)*100:.1f}%"
            weights_data.append(row)
        
        weights_df = pd.DataFrame(weights_data)
        weights_df.index = weights_df.index + 1
        st.dataframe(weights_df, use_container_width=True)
        
        progress_bar.progress(80)
        
        # Correlation Matrix
        st.header("Correlation Matrix")
        st.markdown("Shows how assets move together: 1.0 = perfect correlation, 0.0 = no correlation, -1.0 = inverse")
        
        # Calculate correlation
        correlation = your_cov_matrix / np.outer(np.sqrt(np.diag(your_cov_matrix)), np.sqrt(np.diag(your_cov_matrix)))
        corr_df = pd.DataFrame(correlation, index=your_tickers, columns=your_tickers)
        corr_df = corr_df.round(3)
        
        st.dataframe(corr_df, use_container_width=True)
        
        progress_bar.progress(85)
        
        # Historical Backtesting Tables
        st.header("Historical Backtesting")
        st.markdown("Performance across different rebalancing frequencies")
        
        for strategy_name in portfolios.keys():
            st.subheader(f"{strategy_name}:")
            
            backtest_table_data = []
            for rebal_freq in ['never', 'monthly', 'quarterly', 'annually']:
                result = backtest_results[strategy_name][rebal_freq]
                
                row = {
                    'Rebalancing': rebal_freq.capitalize(),
                    'Total Return': f"{result['total_return']*100:.1f}%",
                    'CAGR': f"{result['annualized_return']*100:.1f}%",
                    'St Dev': f"{result['annualized_volatility']*100:.1f}%",
                    'Sharpe Ratio': f"{result['sharpe_ratio']:.3f}",
                    'Max Drawdown': f"{result['max_drawdown']*100:.1f}%",
                    'Rebalances': result['num_rebalances']
                }
                
                if result.get('jensens_alpha') is not None:
                    row["Jensen's Alpha"] = f"{result['jensens_alpha']*100:.2f}%"
                if result.get('excess_alpha') is not None:
                    row['Excess Alpha'] = f"{result['excess_alpha']*100:.2f}%"
                if result.get('beta') is not None:
                    row['Beta'] = f"{result['beta']:.2f}"
                
                backtest_table_data.append(row)
            
            backtest_df = pd.DataFrame(backtest_table_data)
            backtest_df.index = backtest_df.index + 1
            st.dataframe(backtest_df, use_container_width=True)
        
        progress_bar.progress(90)
        
        # Efficient Frontier
        st.header("Efficient Frontier")
        status_text.text("Generating efficient frontier chart...")
        
        # Calculate efficient frontier by varying target returns
        min_ret = your_mean_returns.min() * 252
        max_ret = your_mean_returns.max() * 252
        target_returns = np.linspace(min_ret, max_ret, 100)
        
        efficient_portfolios = []
        
        for target in target_returns:
            try:
                # Minimize volatility for target return
                def portfolio_volatility(weights):
                    return np.sqrt(np.dot(weights.T, np.dot(your_cov_matrix * 252, weights)))
                
                constraints = [
                    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # weights sum to 1
                    {'type': 'eq', 'fun': lambda x: np.sum(your_mean_returns * x) * 252 - target}  # target return
                ]
                
                bounds = tuple((0, 1) for _ in range(len(your_tickers)))
                
                result = minimize(
                    portfolio_volatility,
                    len(your_tickers) * [1. / len(your_tickers)],
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints
                )
                
                if result.success:
                    vol = portfolio_volatility(result.x)
                    efficient_portfolios.append((vol, target))
            except:
                pass
        
        # Create plot
        fig = go.Figure()
        
        if len(efficient_portfolios) > 0:
            vols, rets = zip(*efficient_portfolios)
            
            # Plot efficient frontier curve
            fig.add_trace(go.Scatter(
                x=vols,
                y=rets,
                mode='markers',
                name='Efficient Frontier',
                marker=dict(
                    size=4,
                    color='red',
                    opacity=0.6
                ),
                hovertemplate='Return: %{y:.2%}<br>Risk: %{x:.2%}<extra></extra>',
                showlegend=False
            ))
        
        # Min volatility point (green star) - leftmost point
        if len(efficient_portfolios) > 0:
            min_vol_idx = np.argmin([v for v, r in efficient_portfolios])
            min_vol_point = efficient_portfolios[min_vol_idx]
            
            fig.add_trace(go.Scatter(
                x=[min_vol_point[0]],
                y=[min_vol_point[1]],
                mode='markers',
                name='Min Volatility',
                marker=dict(size=25, color='green', symbol='star', line=dict(width=2, color='darkgreen')),
                hovertemplate=f'Min Volatility<br>Return: {min_vol_point[1]:.2%}<br>Risk: {min_vol_point[0]:.2%}<extra></extra>'
            ))
        
        # Add strategy portfolios
        colors = ['red', 'orange', 'purple', 'brown']
        symbols = ['circle', 'square', 'diamond', 'cross']
        strategy_names = ['Max Sharpe', 'Min Variance', 'Equal Weight', 'Risk Parity']
        
        for i, name in enumerate(strategy_names):
            if name in portfolios:
                portfolio = portfolios[name]
                fig.add_trace(go.Scatter(
                    x=[portfolio['volatility']],
                    y=[portfolio['return']],
                    mode='markers',
                    name=name,
                    marker=dict(size=12, color=colors[i], symbol=symbols[i], line=dict(width=2, color='black')),
                    hovertemplate=f'{name}<br>Return: %{{y:.2%}}<br>Risk: %{{x:.2%}}<extra></extra>'
                ))
        
        fig.update_layout(
            title='Efficient Frontier',
            xaxis_title='Risk (Standard Deviation)',
            yaxis_title='Expected Return',
            height=600,
            template='plotly_white',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            )
        )
        
        fig.update_xaxes(tickformat='.1%')
        fig.update_yaxes(tickformat='.1%')
        
        st.plotly_chart(fig, use_container_width=True)
        
        progress_bar.progress(100)
        status_text.text("Analysis complete!")
        
        # Success message
        st.success("Portfolio optimization complete! Scroll up to view results.")
        
        # PROSPECT Portfolio Analysis (if enabled)
        if analyze_prospect and prospect_tickers_input:
            st.markdown("---")  # Simple horizontal line
            st.header("PROSPECT PORTFOLIO ANALYSIS")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Parse prospect tickers
                prospect_tickers = [t.strip().upper() for t in prospect_tickers_input.split(',')]
                
                # Initialize prospect_current_weights_dict
                prospect_current_weights_dict = {}
                
                # If using prospect current weights, show input fields FIRST and wait for submission
                if use_prospect_current_weights and not st.session_state.prospect_weights_submitted:
                    st.header("Step 1: Enter PROSPECT Current Portfolio Weights")
                    
                    with st.form("prospect_weights_form"):
                        st.markdown("**Enter the percentage allocation for each PROSPECT ticker:**")
                        cols = st.columns(4)
                        temp_prospect_weights = {}
                        
                        for idx, ticker in enumerate(prospect_tickers):
                            with cols[idx % 4]:
                                default_val = st.session_state.prospect_weights.get(ticker, 0.0) * 100
                                weight = st.number_input(
                                    f"{ticker} (%)", 
                                    min_value=0.0, 
                                    max_value=100.0, 
                                    value=default_val, 
                                    step=0.01,
                                    format="%.2f",
                                    key=f"prospect_weight_{ticker}"
                                ) / 100
                                temp_prospect_weights[ticker] = weight
                        
                        st.markdown("---")
                        prospect_submitted = st.form_submit_button("Submit PROSPECT Weights & Continue", type="primary", use_container_width=True)
                        
                        if prospect_submitted:
                            prospect_total_weight = sum(temp_prospect_weights.values())
                            
                            if abs(prospect_total_weight - 1.0) > 0.01:
                                st.error(f"PROSPECT weights must sum to 100% (currently {prospect_total_weight*100:.1f}%). Please adjust and resubmit.")
                            else:
                                st.session_state.prospect_weights = temp_prospect_weights.copy()
                                st.session_state.prospect_weights_submitted = True
                                st.success("PROSPECT weights saved! Continuing with analysis...")
                                st.rerun()
                    
                    # Stop here until weights are submitted
                    st.stop()
                
                # If weights were submitted or not needed, use them
                if use_prospect_current_weights and st.session_state.prospect_weights_submitted:
                    prospect_current_weights_dict = st.session_state.prospect_weights.copy()
                    st.success(f"Using PROSPECT Current Portfolio weights (Total: {sum(prospect_current_weights_dict.values())*100:.1f}%)")
                
                status_text.text("Downloading PROSPECT portfolio data...")
                progress_bar.progress(10)
                
                # Download PROSPECT data
                prospect_price_data = download_stock_data(prospect_tickers, period)
                prospect_tickers = list(prospect_price_data.columns)
                
                progress_bar.progress(20)
                
                # Get stock info
                prospect_stock_info = get_stock_info(prospect_tickers)
                
                # Calculate returns
                prospect_returns = prospect_price_data.pct_change().dropna()
                prospect_mean_returns = prospect_returns.mean()
                prospect_cov_matrix = prospect_returns.cov()
                
                progress_bar.progress(30)
                
                # Align benchmark
                prospect_benchmark_data = benchmark_price_data.reindex(prospect_price_data.index, method='ffill').dropna()
                
                # Calculate strategies
                status_text.text("Calculating PROSPECT optimal portfolios...")
                
                prospect_portfolios = {}
                
                # Max Sharpe
                result = max_sharpe_ratio(prospect_mean_returns, prospect_cov_matrix, risk_free_rate, prospect_max_weight)
                weights = result.x
                ret, vol = calculate_portfolio_performance(weights, prospect_mean_returns, prospect_cov_matrix)
                prospect_portfolios['Max Sharpe'] = {
                    'weights': {t: w for t, w in zip(prospect_tickers, weights)},
                    'return': ret,
                    'volatility': vol
                }
                
                # Min Variance
                result = min_variance(prospect_mean_returns, prospect_cov_matrix, prospect_max_weight)
                weights = result.x
                ret, vol = calculate_portfolio_performance(weights, prospect_mean_returns, prospect_cov_matrix)
                prospect_portfolios['Min Variance'] = {
                    'weights': {t: w for t, w in zip(prospect_tickers, weights)},
                    'return': ret,
                    'volatility': vol
                }
                
                # Equal Weight
                result = equal_weight_portfolio(len(prospect_tickers))
                weights = result.x
                ret, vol = calculate_portfolio_performance(weights, prospect_mean_returns, prospect_cov_matrix)
                prospect_portfolios['Equal Weight'] = {
                    'weights': {t: w for t, w in zip(prospect_tickers, weights)},
                    'return': ret,
                    'volatility': vol
                }
                
                # Risk Parity
                prospect_bond_tickers_list = [t.strip().upper() for t in prospect_bond_tickers.split(',') if t.strip()]
                result = risk_parity_portfolio(
                    prospect_cov_matrix, 
                    prospect_tickers, 
                    prospect_bond_tickers_list, 
                    prospect_max_stock, 
                    prospect_max_bond
                )
                weights = result.x
                ret, vol = calculate_portfolio_performance(weights, prospect_mean_returns, prospect_cov_matrix)
                prospect_portfolios['Risk Parity'] = {
                    'weights': {t: w for t, w in zip(prospect_tickers, weights)},
                    'return': ret,
                    'volatility': vol
                }
                
                # PROSPECT Current Portfolio (if weights provided)
                if use_prospect_current_weights and prospect_current_weights_dict:
                    prospect_total_weight = sum(prospect_current_weights_dict.values())
                    if abs(prospect_total_weight - 1.0) < 0.01:  # Check if weights sum to 100%
                        prospect_current_weights_array = np.array([prospect_current_weights_dict.get(t, 0) for t in prospect_tickers])
                        ret, vol = calculate_portfolio_performance(prospect_current_weights_array, prospect_mean_returns, prospect_cov_matrix)
                        prospect_portfolios['Current Portfolio'] = {
                            'weights': prospect_current_weights_dict,
                            'return': ret,
                            'volatility': vol
                        }
                
                progress_bar.progress(50)
                
                # Calculate benchmark returns for PROSPECT
                prospect_benchmark_returns = prospect_benchmark_data.pct_change().dropna()
                
                # Run backtests for PROSPECT
                status_text.text("Running PROSPECT backtests...")
                prospect_backtest_results = {}
                
                for strategy_name, portfolio in prospect_portfolios.items():
                    weights = np.array([portfolio['weights'].get(t, 0) for t in prospect_tickers])
                    
                    # Calculate Alpha/Beta using Expected Return (not backtest return)
                    portfolio_daily_returns = (prospect_returns * weights).sum(axis=1)
                    
                    # Calculate both Alphas and Beta using the strategy's Expected Return
                    jensens_alpha, excess_alpha, beta = calculate_alpha_beta(
                        portfolio_daily_returns,
                        prospect_benchmark_returns,
                        risk_free_rate,
                        portfolio_expected_return=portfolio['return']  # Use Expected Return
                    )
                    
                    portfolio['jensens_alpha'] = jensens_alpha
                    portfolio['excess_alpha'] = excess_alpha
                    portfolio['beta'] = beta
                    
                    # Run backtests for display
                    prospect_backtest_results[strategy_name] = {}
                    
                    for rebal_freq in ['never', 'monthly', 'quarterly', 'annually']:
                        result = backtest_with_rebalancing(
                            weights, 
                            prospect_price_data, 
                            prospect_benchmark_data, 
                            rebal_freq, 
                            risk_free_rate
                        )
                        prospect_backtest_results[strategy_name][rebal_freq] = result
                
                progress_bar.progress(70)
                
                # Display PROSPECT results
                st.header("PROSPECT Portfolio Strategies Comparison")
                
                prospect_comparison_data = []
                for name, portfolio in prospect_portfolios.items():
                    row = {
                        'Strategy': name,
                        'Expected Return': f"{portfolio['return']*100:.2f}%",
                        'St Dev': f"{portfolio['volatility']*100:.2f}%",
                        'Sharpe Ratio': f"{(portfolio['return'] - risk_free_rate) / portfolio['volatility']:.3f}"
                    }
                    
                    if portfolio.get('jensens_alpha') is not None:
                        row["Jensen's Alpha"] = f"{portfolio['jensens_alpha']*100:.2f}%"
                    if portfolio.get('excess_alpha') is not None:
                        row['Excess Alpha'] = f"{portfolio['excess_alpha']*100:.2f}%"
                    if portfolio.get('beta') is not None:
                        row['Beta'] = f"{portfolio['beta']:.2f}"
                    
                    prospect_comparison_data.append(row)
                
                # Create DataFrame and reset index to start at 1
                prospect_comparison_df = pd.DataFrame(prospect_comparison_data)
                prospect_comparison_df.index = prospect_comparison_df.index + 1
                st.dataframe(prospect_comparison_df, use_container_width=True)
                
                # PROSPECT Portfolio Weights
                st.header("PROSPECT Portfolio Weights")
                
                prospect_weights_data = []
                for ticker in prospect_tickers:
                    row = {'Ticker': ticker}
                    for name, portfolio in prospect_portfolios.items():
                        row[name] = f"{portfolio['weights'].get(ticker, 0)*100:.1f}%"
                    prospect_weights_data.append(row)
                
                prospect_weights_df = pd.DataFrame(prospect_weights_data)
                prospect_weights_df.index = prospect_weights_df.index + 1
                st.dataframe(prospect_weights_df, use_container_width=True)
                
                progress_bar.progress(80)
                
                # PROSPECT Correlation Matrix
                st.header("PROSPECT Correlation Matrix")
                
                prospect_correlation = prospect_cov_matrix / np.outer(np.sqrt(np.diag(prospect_cov_matrix)), np.sqrt(np.diag(prospect_cov_matrix)))
                prospect_corr_df = pd.DataFrame(prospect_correlation, index=prospect_tickers, columns=prospect_tickers)
                prospect_corr_df = prospect_corr_df.round(3)
                
                st.dataframe(prospect_corr_df, use_container_width=True)
                
                # PROSPECT Historical Backtesting
                st.header("PROSPECT Historical Backtesting")
                
                for strategy_name in prospect_portfolios.keys():
                    st.subheader(f"{strategy_name}:")
                    
                    prospect_backtest_table_data = []
                    for rebal_freq in ['never', 'monthly', 'quarterly', 'annually']:
                        result = prospect_backtest_results[strategy_name][rebal_freq]
                        
                        row = {
                            'Rebalancing': rebal_freq.capitalize(),
                            'Total Return': f"{result['total_return']*100:.1f}%",
                            'CAGR': f"{result['annualized_return']*100:.1f}%",
                            'St Dev': f"{result['annualized_volatility']*100:.1f}%",
                            'Sharpe Ratio': f"{result['sharpe_ratio']:.3f}",
                            'Max Drawdown': f"{result['max_drawdown']*100:.1f}%",
                            'Rebalances': result['num_rebalances']
                        }
                        
                        if result.get('jensens_alpha') is not None:
                            row["Jensen's Alpha"] = f"{result['jensens_alpha']*100:.2f}%"
                        if result.get('excess_alpha') is not None:
                            row['Excess Alpha'] = f"{result['excess_alpha']*100:.2f}%"
                        if result.get('beta') is not None:
                            row['Beta'] = f"{result['beta']:.2f}"
                        
                        prospect_backtest_table_data.append(row)
                    
                    prospect_backtest_df = pd.DataFrame(prospect_backtest_table_data)
                    prospect_backtest_df.index = prospect_backtest_df.index + 1
                    st.dataframe(prospect_backtest_df, use_container_width=True)
                
                progress_bar.progress(90)
                
                # PROSPECT Efficient Frontier
                st.header("PROSPECT Efficient Frontier")
                status_text.text("Generating PROSPECT efficient frontier...")
                
                # Calculate efficient frontier for PROSPECT
                min_ret_p = prospect_mean_returns.min() * 252
                max_ret_p = prospect_mean_returns.max() * 252
                target_returns_p = np.linspace(min_ret_p, max_ret_p, 100)
                
                efficient_portfolios_p = []
                
                for target in target_returns_p:
                    try:
                        def portfolio_volatility_p(weights):
                            return np.sqrt(np.dot(weights.T, np.dot(prospect_cov_matrix * 252, weights)))
                        
                        constraints = [
                            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                            {'type': 'eq', 'fun': lambda x: np.sum(prospect_mean_returns * x) * 252 - target}
                        ]
                        
                        bounds = tuple((0, 1) for _ in range(len(prospect_tickers)))
                        
                        result = minimize(
                            portfolio_volatility_p,
                            len(prospect_tickers) * [1. / len(prospect_tickers)],
                            method='SLSQP',
                            bounds=bounds,
                            constraints=constraints
                        )
                        
                        if result.success:
                            vol = portfolio_volatility_p(result.x)
                            efficient_portfolios_p.append((vol, target))
                    except:
                        pass
                
                # Create PROSPECT plot
                fig_p = go.Figure()
                
                if len(efficient_portfolios_p) > 0:
                    vols_p, rets_p = zip(*efficient_portfolios_p)
                    
                    fig_p.add_trace(go.Scatter(
                        x=vols_p,
                        y=rets_p,
                        mode='markers',
                        name='Efficient Frontier',
                        marker=dict(size=4, color='red', opacity=0.6),
                        hovertemplate='Return: %{y:.2%}<br>Risk: %{x:.2%}<extra></extra>',
                        showlegend=False
                    ))
                    
                    # Min volatility point
                    min_vol_idx_p = np.argmin([v for v, r in efficient_portfolios_p])
                    min_vol_point_p = efficient_portfolios_p[min_vol_idx_p]
                    
                    fig_p.add_trace(go.Scatter(
                        x=[min_vol_point_p[0]],
                        y=[min_vol_point_p[1]],
                        mode='markers',
                        name='Min Volatility',
                        marker=dict(size=25, color='green', symbol='star', line=dict(width=2, color='darkgreen')),
                        hovertemplate=f'Min Volatility<br>Return: {min_vol_point_p[1]:.2%}<br>Risk: {min_vol_point_p[0]:.2%}<extra></extra>'
                    ))
                
                # Add PROSPECT strategy portfolios
                colors = ['red', 'orange', 'purple', 'brown']
                symbols = ['circle', 'square', 'diamond', 'cross']
                strategy_names = ['Max Sharpe', 'Min Variance', 'Equal Weight', 'Risk Parity']
                
                for i, name in enumerate(strategy_names):
                    if name in prospect_portfolios:
                        portfolio = prospect_portfolios[name]
                        fig_p.add_trace(go.Scatter(
                            x=[portfolio['volatility']],
                            y=[portfolio['return']],
                            mode='markers',
                            name=name,
                            marker=dict(size=12, color=colors[i], symbol=symbols[i], line=dict(width=2, color='black')),
                            hovertemplate=f'{name}<br>Return: %{{y:.2%}}<br>Risk: %{{x:.2%}}<extra></extra>'
                        ))
                
                fig_p.update_layout(
                    title='PROSPECT Efficient Frontier',
                    xaxis_title='Risk (Standard Deviation)',
                    yaxis_title='Expected Return',
                    height=600,
                    template='plotly_white',
                    showlegend=True,
                    legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
                )
                
                fig_p.update_xaxes(tickformat='.1%')
                fig_p.update_yaxes(tickformat='.1%')
                
                st.plotly_chart(fig_p, use_container_width=True)
                
                progress_bar.progress(95)
                
                # Value Proposition Comparison
                st.markdown("---")  # Simple horizontal line
                st.header("VALUE PROPOSITION - YOUR vs PROSPECT")
                st.markdown("Comparing best strategies (Quarterly Rebalancing)")
                
                value_prop_data = []
                
                for strategy_name in ['Max Sharpe', 'Current Portfolio'] if 'Current Portfolio' in portfolios else ['Max Sharpe']:
                    if strategy_name in portfolios and strategy_name in prospect_portfolios:
                        your_results = backtest_results[strategy_name]['quarterly']
                        prospect_results = prospect_backtest_results[strategy_name]['quarterly']
                        
                        value_prop_data.append({
                            'Strategy': strategy_name,
                            'Your Return': f"{your_results['annualized_return']*100:.1f}%",
                            'Prospect Return': f"{prospect_results['annualized_return']*100:.1f}%",
                            'Your Sharpe': f"{your_results['sharpe_ratio']:.3f}",
                            'Prospect Sharpe': f"{prospect_results['sharpe_ratio']:.3f}",
                            "Your Jensen's α": f"{your_results.get('jensens_alpha', 0)*100:.2f}%" if your_results.get('jensens_alpha') is not None else 'N/A',
                            "Prospect Jensen's α": f"{prospect_results.get('jensens_alpha', 0)*100:.2f}%" if prospect_results.get('jensens_alpha') is not None else 'N/A',
                            'Your Excess α': f"{your_results.get('excess_alpha', 0)*100:.2f}%" if your_results.get('excess_alpha') is not None else 'N/A',
                            'Prospect Excess α': f"{prospect_results.get('excess_alpha', 0)*100:.2f}%" if prospect_results.get('excess_alpha') is not None else 'N/A',
                            'Your Beta': f"{your_results.get('beta', 0):.2f}" if your_results.get('beta') is not None else 'N/A',
                            'Prospect Beta': f"{prospect_results.get('beta', 0):.2f}" if prospect_results.get('beta') is not None else 'N/A',
                            'Your Drawdown': f"{your_results['max_drawdown']*100:.1f}%",
                            'Prospect Drawdown': f"{prospect_results['max_drawdown']*100:.1f}%"
                        })
                
                value_prop_df = pd.DataFrame(value_prop_data)
                value_prop_df.index = value_prop_df.index + 1
                st.dataframe(value_prop_df, use_container_width=True)
                
                progress_bar.progress(100)
                status_text.text("PROSPECT analysis complete!")
                
            except Exception as e:
                st.error(f"Error analyzing PROSPECT portfolio: {str(e)}")
                st.exception(e)
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.exception(e)

else:
    # Landing page
    st.info("Configure your portfolio in the sidebar and click 'Run Analysis' to begin")
    
    st.markdown("""
    ### Features:
    - Modern Portfolio Theory optimization
    - Multiple Strategies: Max Sharpe, Min Variance, Equal Weight, Risk Parity
    - Interactive Charts: Efficient Frontier with 10,000 Monte Carlo simulations
    - Benchmark Comparison: Calculate Alpha and Beta vs any benchmark
    - Backtesting: Test strategies with different rebalancing frequencies
    - Correlation Analysis: Understand asset relationships
    
    ### How to Use:
    1. Enter your stock tickers in the sidebar (comma-separated)
    2. Choose time period and constraints
    3. Optionally add a PROSPECT portfolio for comparison
    4. Click "Run Analysis"
    5. View optimized portfolios and efficient frontier
    
    ### Example Tickers:
    - Tech: AAPL, MSFT, GOOGL, NVDA, AMD
    - Diversified: AAPL, JPM, JNJ, XOM, PG, COST, GE, LLY
    - Bonds: AGG, BND, TLT, IEF
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Built with Streamlit")

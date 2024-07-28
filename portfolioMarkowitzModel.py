import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as optimization
import streamlit as st
from datetime import date, timedelta

# Constants
NUM_TRADING_DAYS = 252
NUM_PORTFOLIOS = 15000

# Indices and their stock symbols
indices = {
    'nifty_50': [
        'RELIANCE.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS', 'TCS.NS', 'KOTAKBANK.NS', 
        'HINDUNILVR.NS', 'ITC.NS', 'LT.NS', 'AXISBANK.NS', 'SBIN.NS', 'BAJFINANCE.NS', 
        'BHARTIARTL.NS', 'HCLTECH.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'SUNPHARMA.NS', 
        'ULTRACEMCO.NS', 'TATAMOTORS.NS', 'TATASTEEL.NS', 'WIPRO.NS', 'NESTLEIND.NS', 
        'TECHM.NS', 'POWERGRID.NS', 'ADANIPORTS.NS', 'GRASIM.NS', 'BAJAJFINSV.NS', 
        'TITAN.NS', 'INDUSINDBK.NS', 'BPCL.NS', 'HDFCLIFE.NS', 'DIVISLAB.NS', 
        'HEROMOTOCO.NS', 'DRREDDY.NS', 'EICHERMOT.NS', 'JSWSTEEL.NS', 'TATACONSUM.NS', 
        'SHREECEM.NS', 'BRITANNIA.NS', 'ONGC.NS', 'HINDALCO.NS', 'COALINDIA.NS', 
        'BAJAJ-AUTO.NS', 'NTPC.NS', 'M&M.NS', 'SBILIFE.NS', 'ICICIPRULI.NS', 
        'CIPLA.NS', 'ADANIGREEN.NS', 'ADANIENT.NS', 'APOLLOHOSP.NS', 'TATASTEEL.NS', 
        'TATAMOTORS.NS', 'SHRIRAMFIN.NS', 'MRF.NS'
    ],
    'nifty_bank': [
        'AXISBANK.NS', 'BANDHANBNK.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'IDFCFIRSTB.NS',
        'INDUSINDBK.NS', 'KOTAKBANK.NS', 'PNB.NS', 'SBIN.NS', 'AUBANK.NS', 
        'BANKBARODA.NS', 'FEDERALBNK.NS'
    ],
    'nifty_fmcg': [
        'BRITANNIA.NS', 'COLPAL.NS', 'DABUR.NS', 'GODREJCP.NS', 'HINDUNILVR.NS',
        'ITC.NS', 'MARICO.NS', 'NESTLEIND.NS', 'TATACONSUM.NS', 'UBL.NS',
        'PGHH.NS', 'RADICO.NS', 'UNITDSPR.NS', 'VBL.NS'
    ],
    'nifty_it': [
        'COFORGE.NS', 'HCLTECH.NS', 'INFY.NS', 'LTIM.NS', 'MPHASIS.NS',
        'PERSISTENT.NS', 'TCS.NS', 'TECHM.NS', 'WIPRO.NS', 'LTTS.NS'
    ],
    'nifty_auto': [
        'ASHOKLEY.NS', 'BAJAJ-AUTO.NS', 'BALKRISIND.NS', 'BHARATFORG.NS', 'EICHERMOT.NS',
        'HEROMOTOCO.NS', 'M&M.NS', 'MARUTI.NS', 'TATAMOTORS.NS', 'TVSMOTOR.NS',
        'APOLLOTYRE.NS', 'BOSCHLTD.NS', 'EXIDEIND.NS', 'MRF.NS', 'MOTHERSON.NS', 
        'TATAMTRDVR.NS'
    ]
}

# Default dates
default_end_date = date.today() - timedelta(days=2)
default_start_date = default_end_date - timedelta(days=3*365)

def download_data(stock_symbols, start_date, end_date):
    stock_data = {}
    for stock in stock_symbols:
        ticker = yf.Ticker(stock)
        stock_data[stock] = ticker.history(start=start_date, end=end_date)['Close']
    return pd.DataFrame(stock_data)

def show_data(data):
    st.line_chart(data)

def calculate_return(data):
    log_return = np.log(data / data.shift(1))
    return log_return[1:]

def show_statistics(returns):
    st.write("Mean Returns:", returns.mean() * NUM_TRADING_DAYS)
    st.write("Covariance Matrix:", returns.cov() * NUM_TRADING_DAYS)

def show_mean_variance(returns, weights):
    portfolio_return = np.sum(returns.mean() * weights) * NUM_TRADING_DAYS
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * NUM_TRADING_DAYS, weights)))
    st.write("Expected portfolio mean (return): ", portfolio_return)
    st.write("Expected portfolio volatility (standard deviation): ", portfolio_volatility)

def show_portfolios(returns, volatilities):
    plt.figure(figsize=(10, 6))
    plt.scatter(volatilities, returns, c=(returns - RISK_FREE_RETURN) / volatilities, marker='o')
    plt.title("Portfolio Mean-Variance Analysis")
    plt.grid(True)
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.colorbar(label='Sharpe Ratio')
    st.pyplot(plt)

def generate_portfolios(returns):
    portfolio_means = []
    portfolio_risks = []
    portfolio_weights = []

    for _ in range(NUM_PORTFOLIOS):
        w = np.random.random(len(returns.columns))
        w /= np.sum(w)
        portfolio_weights.append(w)
        portfolio_means.append(np.sum(returns.mean() * w) * NUM_TRADING_DAYS)
        portfolio_risks.append(np.sqrt(np.dot(w.T, np.dot(returns.cov() * NUM_TRADING_DAYS, w))))

    return np.array(portfolio_weights), np.array(portfolio_means), np.array(portfolio_risks)

def statistics(weights, returns):
    portfolio_return = np.sum(returns.mean() * weights) * NUM_TRADING_DAYS
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * NUM_TRADING_DAYS, weights)))
    return np.array([portfolio_return, portfolio_volatility, (portfolio_return - RISK_FREE_RETURN) / portfolio_volatility])

def min_function_sharpe(weights, returns):
    return -statistics(weights, returns)[2]

def optimize_portfolio(weights, returns):
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(len(returns.columns)))
    return optimization.minimize(fun=min_function_sharpe, x0=weights[0], args=returns, method='SLSQP', bounds=bounds, constraints=constraints)

def print_optimal_portfolio(optimum, returns):
    optimal_weights = optimum['x'].round(3)
    expected_ret, volatility, sharpe_ratio = statistics(optimal_weights, returns)
    st.write("Expected return : ", expected_ret)
    st.write("Volatility : ", volatility)
    st.write("Sharpe ratio: ", sharpe_ratio)
    return optimal_weights

def show_optimal_portfolio(opt, rets, portfolio_rets, portfolio_vols):
    plt.figure(figsize=(10, 6))
    plt.scatter(portfolio_vols, portfolio_rets, c=(portfolio_rets - RISK_FREE_RETURN) / portfolio_vols, marker='o')
    plt.title("Optimal Portfolio vs. Random Portfolios")
    plt.grid(True)
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.colorbar(label='Sharpe Ratio')
    plt.plot(statistics(opt['x'], rets)[1], statistics(opt['x'], rets)[0], 'g*', markersize=20.0)
    st.pyplot(plt)

# Streamlit App
st.title("Optimal Portfolio Construction Using the Markowitz Model")

# Sidebar inputs
index_list = list(indices.keys())
index_choice = st.sidebar.selectbox("Choose an index", index_list)
use_custom_stocks = st.sidebar.checkbox("Use custom stocks")

if use_custom_stocks:
    custom_stocks_input = st.sidebar.text_area("Enter custom stock symbols separated by commas")
    custom_stocks = [stock.strip().upper() for stock in custom_stocks_input.split(',')]
    
    # Append .NS to Indian stocks (those without a market identifier)
    custom_stocks = [stock if '.' in stock else stock + '.NS' for stock in custom_stocks]

    valid_stocks = []
    invalid_stocks = []

    for stock in custom_stocks:
        ticker = yf.Ticker(stock)
        try:
            _ = ticker.history(period="1d")  # Check if the stock symbol is valid
            valid_stocks.append(stock)
        except Exception as e:
            invalid_stocks.append(stock)

    if invalid_stocks:
        st.sidebar.error(f"The following stock symbols are invalid: {', '.join(invalid_stocks)}. Please check and correct them.")
    else:
        stocks_choice = valid_stocks
        st.sidebar.write("Using the following custom stocks:")
        st.sidebar.write(stocks_choice)
else:
    stocks_choice = indices[index_choice]

RISK_FREE_RETURN = st.sidebar.number_input("Risk-Free Return", value=0.07)
start_date = st.sidebar.date_input("Start Date", value=default_start_date)
end_date = st.sidebar.date_input("End Date", value=default_end_date)

# Main content
if st.sidebar.button("Generate Portfolio"):
    if not use_custom_stocks or (use_custom_stocks and not invalid_stocks):
        dataset = download_data(stocks_choice, start_date, end_date)
        show_data(dataset)
        log_daily_returns = calculate_return(dataset)
        pweights, means, risks = generate_portfolios(log_daily_returns)
        show_portfolios(means, risks)
        optimum = optimize_portfolio(pweights, log_daily_returns)
        optimal_weights = print_optimal_portfolio(optimum, log_daily_returns)
        show_optimal_portfolio(optimum, log_daily_returns, means, risks)
        
        st.write("\nStock allocation in the optimal portfolio (showing only stocks with weights above zero):")
        non_zero_weights = {stock: weight for stock, weight in zip(stocks_choice, optimal_weights) if weight > 0}
        sorted_non_zero_weights = dict(sorted(non_zero_weights.items(), key=lambda item: item[1], reverse=True))
        for stock, weight in sorted_non_zero_weights.items():
            st.write(f"<span style='color: red'>{stock}: {weight:.3f}</span>", unsafe_allow_html=True)

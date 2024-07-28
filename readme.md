# Optimal Portfolio Construction Using the Markowitz Model

## Overview

This Streamlit app implements the Markowitz model to help users construct an optimal investment portfolio. The app leverages historical stock price data to compute various portfolio metrics, including returns, volatility, and Sharpe ratios. Users can select from predefined indices or input custom stock symbols to analyze. The app also visualizes portfolios on a mean-variance frontier and highlights the optimal portfolio based on the Sharpe ratio.

## Features

- **Data Download**: Fetches historical stock price data using Yahoo Finance.
- **Return Calculation**: Computes log returns from historical prices.
- **Portfolio Generation**: Creates a large number of random portfolios to explore the mean-variance space.
- **Optimization**: Finds the optimal portfolio that maximizes the Sharpe ratio.
- **Visualization**: Plots the mean-variance frontier and highlights the optimal portfolio.

## How to Use

1. **Choose an Index or Enter Custom Stocks**:
   - Select from predefined indices (Nifty 50, Nifty Bank, Nifty FMCG, Nifty IT, Nifty Auto).
   - Optionally, enter custom stock symbols in the sidebar (ensure they are valid).

2. **Set Parameters**:
   - Input the risk-free return rate.
   - Select the start and end dates for historical data.

3. **Generate Portfolio**:
   - Click the "Generate Portfolio" button to:
     - Download data and show historical price charts.
     - Calculate and display log returns statistics.
     - Generate and plot portfolios with their expected returns and volatilities.
     - Optimize the portfolio and show its metrics.
     - Display the optimal portfolio allocation.

4. **View Results**:
   - Visualize the portfolios on the mean-variance frontier.
   - See the optimal portfolio highlighted.
   - Check the stock allocation of the optimal portfolio.

## Output

- **Historical Data**: Line charts of stock prices.
- **Return Statistics**: Mean returns and covariance matrix.
- **Portfolios**: Scatter plot of random portfolios with color-coded Sharpe ratios.
- **Optimal Portfolio**: Expected return, volatility, and Sharpe ratio.
- **Stock Allocation**: List of stocks in the optimal portfolio with their weights.

## Deployment

The app is deployed and accessible online. You can interact with it and perform portfolio optimization directly on Streamlit.


[**Access the Streamlit App**](https://markowitzmodel-portfolio.streamlit.app/)

## Installation and Running Locally

To run this app locally:

1. Clone the repository:
2. Navigate to the project directory:
3. Install the required packages:
4. Run the Streamlit app:

## Requirements

Python 3.x
numpy
pandas
matplotlib
scipy
yfinance
streamlit

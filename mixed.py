
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
import pandas as pd

# Title of the Streamlit app
st.title('Portfolio Performance Analysis')
st.markdown('## 📈 Analyze Your Portfolio Performance')

class Stock:
    historical_prices = {}

    def __init__(self, symbol):
        self.symbol = symbol
        if symbol not in Stock.historical_prices:
            data = yf.download(symbol, start="2020-01-01", end=datetime.today().strftime('%Y-%m-%d'))
            Stock.historical_prices[symbol] = data

    def cur_price(self, cur_date, end_date):
        try:
            price = self.historical_prices[self.symbol].loc[cur_date:end_date]['Adj Close'].iloc[-1]
            return price 
        except KeyError:
            st.write("Data not available for the given date. Change the date to get data")

    def monthly_ret(self, cur_date, end_date):
        try:           
            end_date = datetime.strptime(cur_date, "%Y-%m-%d")
            start_date = end_date - timedelta(days=30)
            prices = self.historical_prices[self.symbol].loc[start_date:end_date]['Adj Close']
            returns = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0] * 100
            return returns
        except KeyError:
            st.write("Data not available for the given date. Change the date to get data")

    def daily_ret(self, cur_date, end_date):
        try:
            end_date = datetime.strptime(cur_date, "%Y-%m-%d")
            start_date = end_date - timedelta(days=1)
            price_end = self.historical_prices[self.symbol].loc[start_date:end_date]['Adj Close'].iloc[-1]
            price_start = self.historical_prices[self.symbol].loc[start_date:end_date]['Adj Close'].iloc[0]
            returns = (price_end - price_start) / price_start * 100
            return returns
        except KeyError:
            st.write("Data not available for the given date. Change the date to get data")

    def last_30_days_price(self, cur_date, end_date):
        end_date = datetime.strptime(cur_date, "%Y-%m-%d")
        start_date = end_date - timedelta(days=30)
        prices = self.historical_prices[self.symbol].loc[start_date:end_date]['Adj Close']
        return prices.to_numpy()

    def daily_returns_for_every_day(self, start_date, end_date):
        prices = self.historical_prices[self.symbol].loc[start_date:end_date]['Adj Close']
        daily_returns = prices.pct_change().dropna()
        return daily_returns.to_numpy()

    def final_daily_return(self, daily_returns, investment):
        final_return = (investment * (1 + daily_returns)).cumprod()[-1]
        return final_return

# Function to select top performing stocks
def select_top_stocks(stocks, cur_date, end_date):
    selected_stocks = []
    for symbol in stocks:
        stock = Stock(symbol)
        monthly_ret = stock.monthly_ret(cur_date, end_date)
        if monthly_ret > 0:
            selected_stocks.append(symbol)
    return selected_stocks

# Function to calculate Maximum Drawdown (MaxDD)
def calculate_max_drawdown(returns):
    peak = returns[0]
    max_dd = 0
    for r in returns:
        if r > peak:
            peak = r
        else:
            dd = (peak - r) / peak
            if dd > max_dd:
                max_dd = dd
    return max_dd * 100  # Convert to percentage

# Function to calculate Drawdown Volatility (DDVol)
def calculate_dd_volatility(returns):
    drawdowns = []
    peak = returns[0]
    for r in returns:
        if r > peak:
            peak = r
        drawdown = (peak - r) / peak
        drawdowns.append(drawdown)
    dd_volatility = np.std(drawdowns) * np.sqrt(252) * 100  # Annualized standard deviation
    return dd_volatility

# Function to calculate portfolio performance
def calculate_performance(returns, initial_investment, start_date, end_date):
    if initial_investment == 0:
        st.error("Error: Initial investment cannot be zero.")
        return None, None, None, None
    
    start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
    end_datetime = datetime.strptime(end_date, "%Y-%m-%d")
    t = (end_datetime - start_datetime).days / 365  # Number of years

    final_value = initial_investment
    for daily_return in returns:
        value = final_value * (1 + daily_return / 100)
        final_value = value

    CAGR = (((final_value / initial_investment) ** (1 / t)) - 1) * 100
    volatility = (np.std(returns) * np.sqrt(252)) * 100
    sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252)
    max_dd = calculate_max_drawdown(returns)
    dd_volatility = calculate_dd_volatility(returns)
    return CAGR, volatility, sharpe_ratio, max_dd, dd_volatility

# Function to calculate portfolio performance with selected stocks
def selected_stock_calculate_performance(returns, initial_value, start_date, end_date, portfolio_daily_returns):
    if initial_value == 0:
        st.error("Error: Initial investment cannot be zero.")
        return None, None, None
    
    start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
    end_datetime = datetime.strptime(end_date, "%Y-%m-%d")
    t = (end_datetime - start_datetime).days / 365  # Number of years
    
    final_value = sum(returns)  # Calculate final value based on sum of individual stock returns
    CAGR = (((final_value / initial_value) ** (1 / t)) - 1) * 100
    
    # Calculate the aggregated daily returns of the selected stocks portfolio
    portfolio_daily_returns = np.array(portfolio_daily_returns).mean(axis=0)
    
    volatility = (np.std(portfolio_daily_returns) * np.sqrt(252)) * 100
    sharpe_ratio = (np.mean(portfolio_daily_returns) / np.std(portfolio_daily_returns)) * np.sqrt(252)
    
    return CAGR, volatility, sharpe_ratio

# User input for initial investment, start date, and end date
st.sidebar.subheader("Portfolio Configuration")
initial_value = st.sidebar.number_input("Initial Equity", min_value=1)
start_date = st.sidebar.date_input("Start Date")
end_date = st.sidebar.date_input("End Date")

# Calculation and display of portfolio performance
if st.button("Calculate"):
    start_date = start_date.strftime('%Y-%m-%d')
    end_date = end_date.strftime('%Y-%m-%d')
    stocks_list = ['COALINDIA.NS', 'UPL.NS', 'ICICIBANK.NS', 'NTPC.NS', 'HEROMOTOCO.NS', 'AXISBANK.NS', 
                'HDFCLIFE.NS', 'BAJAJFINSV.NS', 'ONGC.NS', 'APOLLOHOSP.NS', 'SBIN.NS', 'KOTAKBANK.NS', 'SBILIFE.NS', 
                'BRITANNIA.NS', 'SUNPHARMA.NS', 'BAJAJ-AUTO.NS', 'MARUTI.NS', 'LT.NS', 'RELIANCE.NS', 'BPCL.NS', 
                'TATACONSUM.NS', 'CIPLA.NS', 'M&M.NS', 'BAJFINANCE.NS', 'ITC.NS', 'ADANIPORTS.NS', 'NESTLEIND.NS', 
                'HDFCBANK.NS', 'DIVISLAB.NS', 'TATAMOTORS.NS', 'INDUSINDBK.NS', 'EICHERMOT.NS', 'HINDUNILVR.NS', 'ASIANPAINT.NS', 'DRREDDY.NS', 
                'ULTRACEMCO.NS', 'BHARTIARTL.NS', 'TITAN.NS', 'TCS.NS', 'LTIM.NS', 'TATASTEEL.NS', 'POWERGRID.NS',
                    'HCLTECH.NS', 'TECHM.NS', 'INFY.NS', 'WIPRO.NS', 'JSWSTEEL.NS', 'ADANIENT.NS', 'GRASIM.NS', 'HINDALCO.NS']
    selected_stocks = select_top_stocks(stocks_list, start_date, end_date)

    portfolio_returns = []
    portfolio_dailyreturns = []
    for symbol in selected_stocks:
        stock = Stock(symbol)
        daily_returns = stock.daily_returns_for_every_day(start_date, end_date)
        final_return = stock.final_daily_return(daily_returns, initial_value / len(selected_stocks))
        portfolio_returns.append(final_return)
        portfolio_dailyreturns.append(daily_returns)

    CAGR, volatility, sharpe_ratio, max_dd, dd_volatility = calculate_performance(stock.daily_returns_for_every_day(start_date, end_date), initial_value, start_date, end_date)
    nifty_data = {'CAGR (%)': [round(CAGR, 2)], 'Volatility (%)': [round(volatility, 2)], 
                  'Sharpe Ratio': [round(sharpe_ratio, 2)], 'MaxDD (%)': [round(max_dd, 2)],
                  'DDVol (%)': [round(dd_volatility, 2)], "Start Date": start_date, "End Date": end_date}
    df_nifty = pd.DataFrame(nifty_data)

    CAGR1, volatility1, sharpe_ratio1 = selected_stock_calculate_performance(portfolio_returns, initial_value, start_date, end_date, portfolio_dailyreturns)
    selected_stocks_performance = {'CAGR (%)': [round(CAGR1, 2)], 'Volatility (%)': [round(volatility1, 2)], 'Sharpe Ratio': [round(sharpe_ratio1, 2)], "Start Date": start_date, "End Date": end_date}
    df_selected_stocks = pd.DataFrame.from_records([selected_stocks_performance, nifty_data], index=["Strategy", "Benchmark"])

    # Display performance metrics and selected stocks
    st.subheader("Performance Metrics")
    st.table(df_selected_stocks)

    st.subheader("Selected Stocks")
    for i, symbol in enumerate(selected_stocks, start=1):
        st.write(f"{i}. {symbol}")

    # Plotting daily returns
    st.subheader("Equity Curve")

    dates = pd.date_range(start=start_date, end=end_date)
    nifty_returns = stock.daily_returns_for_every_day(start_date, end_date)
    nifty_returns_df = pd.DataFrame({"Date": dates[:len(nifty_returns)], "Nifty50": nifty_returns})

    min_length = min(len(portfolio_dailyreturns[0]), len(nifty_returns))
    portfolio_returns_df = pd.DataFrame({"Date": dates[:min_length], 
                                         "Portfolio": [(sum(sublist) / len(portfolio_dailyreturns)) for sublist in zip(*portfolio_dailyreturns[:min_length])]})

    scale_factor = initial_value / 1000

    # Plotting combined daily returns
    st.line_chart(data=portfolio_returns_df.set_index("Date").join(nifty_returns_df.set_index("Date")) * scale_factor, use_container_width=True)

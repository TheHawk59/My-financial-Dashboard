# -*- coding: utf-8 -*-
###############################################################################
# FINANCIAL DASHBOARD EXAMPLE - v3
###############################################################################

#==============================================================================
# Initiating
#==============================================================================

# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf
import streamlit as st
import plotly.express as px

#==============================================================================
# HOT FIX FOR YFINANCE .INFO METHOD
# Ref: https://github.com/ranaroussi/yfinance/issues/1729
#==============================================================================

import requests
import urllib

class YFinance:
    user_agent_key = "User-Agent"
    user_agent_value = ("Mozilla/5.0 (Windows NT 6.1; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/58.0.3029.110 Safari/537.36")
    
    def __init__(self, ticker):
        self.yahoo_ticker = ticker

    def __str__(self):
        return self.yahoo_ticker

    def _get_yahoo_cookie(self):
        cookie = None

        headers = {self.user_agent_key: self.user_agent_value}
        response = requests.get("https://fc.yahoo.com",
                                headers=headers,
                                allow_redirects=True)

        if not response.cookies:
            raise Exception("Failed to obtain Yahoo auth cookie.")

        cookie = list(response.cookies)[0]

        return cookie

    def _get_yahoo_crumb(self, cookie):
        crumb = None

        headers = {self.user_agent_key: self.user_agent_value}

        crumb_response = requests.get(
            "https://query1.finance.yahoo.com/v1/test/getcrumb",
            headers=headers,
            cookies={cookie.name: cookie.value},
            allow_redirects=True,
        )
        crumb = crumb_response.text

        if crumb is None:
            raise Exception("Failed to retrieve Yahoo crumb.")

        return crumb

    @property
    def info(self):
        # Yahoo modules doc informations :
        # https://cryptocointracker.com/yahoo-finance/yahoo-finance-api
        cookie = self._get_yahoo_cookie()
        crumb = self._get_yahoo_crumb(cookie)
        info = {}
        ret = {}

        headers = {self.user_agent_key: self.user_agent_value}

        yahoo_modules = ("assetProfile,"  # longBusinessSummary
                         "summaryDetail,"
                         "financialData,"
                         "indexTrend,"
                         "defaultKeyStatistics")

        url = ("https://query1.finance.yahoo.com/v10/finance/"
               f"quoteSummary/{self.yahoo_ticker}"
               f"?modules={urllib.parse.quote_plus(yahoo_modules)}"
               f"&ssl=true&crumb={urllib.parse.quote_plus(crumb)}")

        info_response = requests.get(url,
                                     headers=headers,
                                     cookies={cookie.name: cookie.value},
                                     allow_redirects=True)

        info = info_response.json()
        info = info['quoteSummary']['result'][0]

        for mainKeys in info.keys():
            for key in info[mainKeys].keys():
                if isinstance(info[mainKeys][key], dict):
                    try:
                        ret[key] = info[mainKeys][key]['raw']
                    except (KeyError, TypeError):
                        pass
                else:
                    ret[key] = info[mainKeys][key]

        return ret

#==============================================================================
# Header
#==============================================================================

def render_header():
    """
    This function render the header of the dashboard with the following items:
        - Title
        - Dashboard description
        - 3 selection boxes to select: Ticker, Start Date, End Date
    """
    
    # Add dashboard title and description
    st.title("MY FINANCIAL DASHBOARD ⭐")
    col1, col2 = st.columns([1,5])
    col1.write("Data source: Yahoo Finance")
    #col2.image(r"C:\Users\mperan\OneDrive - IESEG\Desktop\FP_Sessions\Indiv Project\img\yahoo_finance.png", width=100)
    #col2.image('.\img\yahoo_finance.png', width=100)
    
    # Add the ticker selection on the sidebar
    # Get the list of stock tickers from S&P500
    global ticker_list 
    ticker_list = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol']
    period_list = ['1mo', '3mo', '6mo', '1y', '2y', '5y', 'ytd', 'max']
    
    # Add the selection boxes
    col1, col2, col3, col4 = st.columns(4)  # Create 4 columns
    # Ticker name
    global ticker  # Set this variable as global, so the functions in all of the tabs can read it
    ticker = col1.selectbox("Ticker", ticker_list, key=1)
    # Begin and end dates
    global start_date, end_date  # Set this variable as global, so all functions can read it
    start_date = col2.date_input("Start date", datetime.today().date() - timedelta(days=30), key=2)
    end_date = col3.date_input("End date", datetime.today().date(), key=3)
    # Fix the time period
    global time_period
    time_period = col4.selectbox("Time period", period_list)
 
    
    def TimePeriodCondition(time_period):
        if time_period == '1mo':
            start_date = end_date - timedelta(days=30)
        elif time_period == '3mo':
            start_date = end_date - timedelta(days=90)
        elif time_period == '6mo':
            start_date = end_date - timedelta(days=180)
        elif time_period == '1y':
            start_date = end_date - timedelta(days=365)
        elif time_period == '2y':
            start_date = end_date - timedelta(days=730)
        elif time_period == '5y':
            start_date = end_date - timedelta(days=1825)
        elif time_period == 'ytd':
            start_date = "2023-01-01"
        elif time_period == 'max':
            start_date = "1980-12-25" #datetime.date(1980, 12, 25)
        return start_date
    
    start_date = TimePeriodCondition(time_period)
    
    # Add the update button
    global update_button 
    update_button = st.button("Update Data")
    
#==============================================================================
# Tab 1
#==============================================================================

def render_tab1():
    """
    This function render the Tab 1 - Company Profile of the dashboard.
    """
    # st.write(YFinance(ticker).info)
    col1, col2 = st.columns([3,5])  # Create 2 columns
    
    def GetStockData(ticker, start_date, end_date):
        stock_df = yf.Ticker(ticker).history(start=start_date, end=end_date)
        stock_df.reset_index(inplace=True)  # Drop the indexes
        stock_df['Date'] = stock_df['Date'].dt.date  # Convert date-time to date
        return stock_df
    
    if ticker != '':
        stock_price = GetStockData(ticker, start_date, end_date)
        # st.write('**Stock price data**')
        # st.dataframe(stock_price, hide_index=True, use_container_width=True)
   
        fig = go.Figure()
    
        fig.add_trace(go.Scatter(
            x=stock_price['Date'],
            y=stock_price['Open'],
            mode='lines',
            line=dict(color='blue'),  # Set line color for 'Open'
            name=f'{ticker} Open Price'
        ))
    
        fig.add_trace(go.Scatter(
            x=stock_price['Date'],
            y=stock_price['High'],
            mode='lines',
            line=dict(color='green'),  # Set line color for 'High'
            name=f'{ticker} High Price'
        ))
    
        fig.add_trace(go.Scatter(
            x=stock_price['Date'],
            y=stock_price['Low'],
            mode='lines',
            line=dict(color='red'),  # Set line color for 'Low'
            name=f'{ticker} Low Price'
        ))
    
        fig.add_trace(go.Scatter(
            x=stock_price['Date'],
            y=stock_price['Close'],
            mode='lines',
            line=dict(color='purple'),  # Set line color for 'Close'
            name=f'{ticker} Close Price'
        ))
    
        fig.update_layout(
            title=f'{ticker} Stock Price Line Chart',
            xaxis_title='Date',
            yaxis_title='Stock Price',
            showlegend=True,
        )
    
        col2.plotly_chart(fig, use_container_width=True)


            
    # Get the company information
    @st.cache_data
    def GetCompanyInfo(ticker):
        """
        This function get the company information from Yahoo Finance.
        """        
        return YFinance(ticker).info
        #return yf.Ticker(ticker).info
    
    # If the ticker is already selected
    if ticker != '':
        # Get the company information in list format
        info = GetCompanyInfo(ticker)
        
        # Show the company profile using markdown + HTML
        st.header('Company Profile:')
        st.markdown(
            f"""
            <div style="text-align: justify;">
                {info['address1']}<br>
                {info['city']}, {info['state']} {info['zip']}<br>
                {info['country']}<br>
                Phone: {info['phone']}<br>
                Website: <a href="{info['website']}" target="_blank">{info['website']}</a><br>
                Sector: <strong>{info['sector']}</strong><br>
                Industry: <strong>{info['industry']}</strong><br>
                Full Time Employees: <strong>{info['fullTimeEmployees']}</strong><br>
            </div><br>
            """,
            unsafe_allow_html=True
        )
        
        # Show the company description using markdown + HTML
        st.header('Summary:')
        st.markdown('<div style="text-align: justify;">' + \
                    info['longBusinessSummary'] + \
                    '</div><br>',
                    unsafe_allow_html=True)
        
        # Show some statistics as a DataFrame
        info_keys = {'previousClose':'Previous Close',
                     'open'         :'Open',
                     'bid'          :'Bid',
                     'ask'          :'Ask',
                     'marketCap'    :'Market Cap',
                     'volume'       :'Volume'}
        company_stats = {}  # Dictionary
        for key in info_keys:
            company_stats.update({info_keys[key]:info[key]})
        company_stats = pd.DataFrame({'Value':pd.Series(company_stats)})  # Convert to DataFrame
        col1.dataframe(company_stats)
        


#==============================================================================
# Tab 2
#==============================================================================

def render_tab2():
    """
    This function render the Tab 2 - Chart of the dashboard.
    """
    
    # Insert interval filter
    # ival = st.selectbox("Interval", ('1d', '1w', '1mo', '1y'))
 
    
    
    # Add table to show stock data
    @st.cache_data
    def GetStockData(ticker, start_date, end_date):
        stock_df = yf.Ticker(ticker).history(start=start_date, end=end_date)
        stock_df.reset_index(inplace=True)  # Drop the indexes
        stock_df['Date'] = stock_df['Date'].dt.date  # Convert date-time to date
        return stock_df
    
    # Add a check box to show/hid data
    # If the ticker name is selected and the check box is checked, show data
    chart_type = st.selectbox("Chart Type", ("Line Chart", "Candlestick Chart"))
    if ticker != '':
        
## This code doesn't work

        # if ival == '1d':
        #     stock_price = yf.Ticker(ticker).history(interval='1d', start=start_date, end=end_date)
        # elif ival == '5d':
        #     stock_price = yf.Ticker(ticker).history(interval='5d', start=start_date, end=end_date)
        # elif ival == '1w':
        #     stock_price = yf.Ticker(ticker).history(interval='1w', start=start_date, end=end_date)
        # elif ival == '1mo':
        #     stock_price = yf.Ticker(ticker).history(interval='1mo', start=start_date, end=end_date)
        # elif ival == '1y':
        #     stock_price = yf.Ticker(ticker).history(interval='1y', start=start_date, end=end_date)
            
        # if interval == '1d':
        #     stock_price = yf.Ticker(ticker).history(interval='1d', start=start_date, end=end_date)
        # elif interval == '5d':
        #     stock_price = GetStockData(ticker, interval='5d', start_date, end_date)
        # elif interval == '1w':
        #     stock_price = GetStockData(ticker, interval='1w', start_date, end_date)
        # elif interval == '1mo':
        #     stock_price = GetStockData(ticker, interval='1mo', start_date, end_date)
        # elif interval == '1y':
        #     stock_price = GetStockData(ticker, interval='1y', start_date, end_date)
#####
        
        stock_price = GetStockData(ticker, start_date, end_date)
        
        if chart_type == "Candlestick Chart":
            # Add a candle stick chart using plotly
            # If the candlechart is selected show plot
                st.write('**Candlestick Chart**')       
                fig = go.Figure(data=[go.Candlestick(
                    x=stock_price['Date'],
                    open=stock_price['Open'],
                    high=stock_price['High'],
                    low=stock_price['Low'],
                    close=stock_price['Close'])])
                st.plotly_chart(fig, use_container_width=True) 
                
        # Add a line chart using plotly
        elif chart_type == "Line Chart":
            st.write('**Line Plot**')       
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=stock_price['Date'],
                y=stock_price['Open'],
                mode='lines',
                name=f'{ticker} Open Price'
            ))
            
            fig.add_trace(go.Scatter(
                x=stock_price['Date'],
                y=stock_price['High'],
                mode='lines',
                name=f'{ticker} High Price'
            ))
            
            fig.add_trace(go.Scatter(
                x=stock_price['Date'],
                y=stock_price['Low'],
                mode='lines',
                name=f'{ticker} Low Price'
            ))
            
            fig.add_trace(go.Scatter(
                x=stock_price['Date'],
                y=stock_price['Close'],
                mode='lines',
                name=f'{ticker} Close Price'
            ))
            
            fig.update_layout(
                title=f'{ticker} Stock Price Line Plot',
                xaxis_title='Date',
                yaxis_title='Stock Price',
                showlegend=True,
            )
            
            st.plotly_chart(fig, use_container_width=True)

            
            
#==============================================================================
# Tab 3
#==============================================================================

def render_tab3():
    """
    This function render the Tab 3 - Financials of the dashboard.
    """
        
    # insert lists: Income Statement, Balance Sheet, Cash Flow, Annual, Quarterly    
    col1, col2 = st.columns(2)
    fin_type =col1.selectbox("Financial report", ('Income Statement', 'Balance Sheet', 'Cash Flow'), key=10)
    period = col2.selectbox("Period", ('Annual', 'Quarterly'), key=11) 
    
    if ticker !='':
        # Define the financials to display
        if fin_type == 'Income Statement':
            data = yf.Ticker(ticker).income_stmt
            if period == 'Quarterly':
                data = yf.Ticker(ticker).quarterly_income_stmt
        elif fin_type == 'Balance Sheet':
            data = yf.Ticker(ticker).balance_sheet
            if period == 'Quarterly':
                data = yf.Ticker(ticker).quarterly_balance_sheet
        elif fin_type == 'Cash Flow':
            data = yf.Ticker(ticker).cashflow
            if period == 'Quarterly':
                data = yf.Ticker(ticker).quarterly_cashflow

                
        st.dataframe(data)
        
#==============================================================================
# Tab 4
#==============================================================================

def render_tab4():
    """
    This function render the Tab 4 - Monte_Carlo Simulation of the dashboard.
    """

    col_1, col_2 = st.columns(2)  # Create 2 columns
    # Ticker name
    sim = col_1.selectbox("Number of simulations", ('200', '500', '1000'), key=5)
    time = col_2.selectbox("Time horizon (days", ('30', '60', '90'), key=6)    
        # Define the stock ticker symbol and date range for historical data
    if ticker != '':
        # Fetch historical data for the specified date range
        stock = yf.Ticker(ticker).history(start=start_date, end=end_date)
        
        # Calculate daily returns
        daily_returns = stock['Close'].pct_change().dropna()
        
        # Calculate daily volatility as the standard deviation of daily returns
        daily_volatility = daily_returns.std()
        
        # Define the number of simulation runs and time horizon (days)
        num_simulations = int(sim)
        time_horizon = int(time)
        
        # Perform Monte Carlo simulation for the next time period
        simulated_prices = np.zeros((num_simulations, time_horizon))
        
        for i in range(num_simulations):
            daily_returns_simulated = np.random.normal(0, daily_volatility, time_horizon)
            price_path = np.cumprod(1 + daily_returns_simulated)
            simulated_prices[i, :] = stock['Close'].iloc[-1] * price_path
        
        # Plot the simulated stock price paths
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i in range(num_simulations):
            ax.plot(simulated_prices[i, :], lw=1, alpha=0.5)
        
        ax.set_title(f'{ticker} Stock Price Simulations for the Next {time_horizon} Days')
        ax.set_xlabel('Time (Days)')
        ax.set_ylabel('Stock Price')
        ax.grid(True)
        
        # Display the plot in Streamlit app
        st.pyplot(fig)

        
        # Calculate the VaR at a 95% confidence level
        ending_prices_200th_day = simulated_prices[:, -1]
        confidence_level = 95  # 95% confidence level
        var_95 = np.percentile(ending_prices_200th_day, 100 - confidence_level)
        
        st.write(f'Value at Risk (VaR) at {confidence_level}% confidence: {var_95:.2f}')
            
#==============================================================================
# Tab 5
#==============================================================================

def render_tab5():
    """
    This function render the Tab 5 - Stock Comparison
    """


    if ticker != "":
        
        # Get user input for comparison
        compare_ticker = st.selectbox("Ticker to compare with", ticker_list, key=15)
        
        col1, col2=st.columns(2)
        
        stock_data_ticker = yf.download(ticker, start=start_date, end=end_date, progress=False)['Close'].reset_index()
        stock_data_compare = yf.download(compare_ticker, start=start_date, end=end_date, progress=False)['Close'].reset_index()
        
        # Plot stock comparison for Ticker 1
        fig_ticker = go.Figure()
        fig_ticker.add_trace(go.Scatter(x=stock_data_ticker['Date'], y=stock_data_ticker['Close'], name=ticker, mode='lines'))
        fig_ticker.update_layout(title=f"{ticker} Stock Price", xaxis_title="Date", yaxis_title="Stock Price")
        col1.plotly_chart(fig_ticker, use_container_width=True)
        
        # Plot stock comparison for Ticker 2
        fig_compare = go.Figure()
        fig_compare.add_trace(go.Scatter(x=stock_data_compare['Date'], y=stock_data_compare['Close'], name=compare_ticker, mode='lines'))
        fig_compare.update_layout(title=f"{compare_ticker} Stock Price", xaxis_title="Date", yaxis_title="Stock Price")
        col2.plotly_chart(fig_compare, use_container_width=True)
#==============================================================================
# Main body
#==============================================================================
      
# Render the header
render_header()

# Render the tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Summary", "Chart", "Financials", "Monte Carlo Simulation", "Stock Comparison"])

with tab1:
    render_tab1()
with tab2:
    render_tab2()
with tab3:
    render_tab3()
with tab4:
    render_tab4()
with tab5:
    render_tab5()
    
# Customize the dashboard with CSS
st.markdown(
    """
    <style>
        .stApp {
            background: #F0F8FF;
        }
    </style>
    """,
    unsafe_allow_html=True,
)
    
###############################################################################
# END
###############################################################################
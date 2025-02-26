#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


# Create a simple DataFrame with pandas
data = {
    "Stocks": ["AAPL", "JPM", "MSFT"],
    "Price": [150.75, 100.50, 200],
    "Shares": [10, 15, 25.5]
}
print("Data dictionary: ", data)

df=pd.DataFrame(data)
print("DataFrame: ", df)


# In[3]:


# Use NumPy for numerical operations
prices = np.array([150.75, 152.30, 148.90, 151, 153.45, 164, 159.99])
print(prices)

avg=np.mean(prices)
print("Average price: ", avg)

price_sum=np.sum(prices)
print("sum of prices: ", price_sum)

price_min=np.min(prices)
print("Minimum price: ", price_min)

price_max=np.max(prices)
print("Maximum price: ",price_max)

price_median=np.median(prices)
print("Median of prices: ", price_median)

price_std_dev=np.std(prices)
print("Standard Deviation: ", price_std_dev)


# In[4]:


# Plot with matplotlib

plt.figure(figsize=(8,4))
plt.plot(prices, marker="o", linestyle=":", color="red")
plt.title("Stock Prices over time")
plt.xlabel("Time")
plt.ylabel("Prices")
plt.grid(True)
plt.show()


# 
# # Mini "Hello Finance" Project: Simulate and Plot a Random Walk
# 
# Random Walk: Simple model for simulating stock price movement. Assumption is that price changes are random and unpredicatable. Read more about Random Walks here: https://en.wikipedia.org/wiki/Random_walk

# In[5]:


# Use seed - This ensures reproducibility, same random numbers each run
np.random.seed(42) # If you want truly random numbers each run, remove this line
random_walk_no_sum=np.random.normal(0,1,250)
print(f"First 20 values of random_walk_no_sum:\n{random_walk_no_sum[0:20]}")

random_walk = np.random.normal(0, 1, 250).cumsum()
print(f"First 20 values of random_walk:\n{random_walk[0:20]}")


# In[6]:


mean_walk_no_sum = np.mean(random_walk_no_sum)
std_walk_no_sum = np.std(random_walk_no_sum)
print(f"Mean of random walk_no_sum: {mean_walk_no_sum:.2f}")
print(f"Standard Deviation of random walk_no_sum: {std_walk_no_sum:.2f}")

mean_walk = np.mean(random_walk)
std_walk = np.std(random_walk)
print(f"Mean of random walk (Cumulative Sum): {mean_walk:.2f}")
print(f"Standard Deviation of random walk (Cumulative Sum): {std_walk:.2f}")


# In[7]:


plt.figure(figsize=(20,8))
plt.xlabel("Time (days)")
plt.ylabel("Simulated stock price")
plt.title("Simulated Random Walk (Stock Price Simulation)")
plt.plot(random_walk,marker="p",linestyle="-",color="green")
plt.grid(True)
plt.show()


# In[8]:


plt.figure(figsize=(20,8))
plt.xlabel("Time (days)")
plt.ylabel("Simulated stock price")
plt.title("Simulated Random Walk (No CumSum)")
plt.plot(random_walk_no_sum,marker="p",linestyle="--",color="green")
plt.grid(True)
plt.show()


# In[9]:


plt.figure(figsize=(20,8))
plt.xlabel("Time (days)")
plt.ylabel("Value")
plt.title("Random Walk No CumSum vs Random Walk with CumSum")
plt.plot(random_walk_no_sum,marker=".",linestyle="-",color="purple",label="Random Walk No CumSum")
plt.plot(random_walk,marker=".",linestyle="-",color="orange",label="Random Walk With CumSum")
plt.grid(True)
plt.legend()
plt.show()


# # Reading Real Financial Data from CSV files
# 
# Read stock price data of AAPL, MSFT, JPM from csv files. Perform some analysis on the data. Merge the data and plot them using Matplotlib and Plotly.

# In[10]:


aapl=pd.read_csv("./datasets/AAPL.csv")
jpm=pd.read_csv("./datasets/JPM.csv")
msft=pd.read_csv("./datasets/MSFT.csv")

print("AAPL Data: \n",aapl.head())
print("JPM Data: \n",jpm.head())
print("MSFT Data: \n",msft.head())


# In[11]:


# Display a summary of AAPL's Data
print(aapl.describe)


# In[12]:


# Display summary and other details of MSFT's data
print(msft.describe)
print("Shape: ",msft.shape)
print(f"Columns: {msft.columns}")
print(f"Number of Rows: {len(msft)}")
print(f"Data Types: {msft.dtypes}")
print(f"Date Range: {msft['Date'].min()} to {msft['Date'].max()}")
print(f"Number of Unique values: {msft['High'].nunique()}")
print(f"\nUnique values: {msft['High'].unique()}\n")
print(f"Value Counts: {msft['Low'].value_counts()}")


# In[13]:


# Merge the closing prices of each stock based on the 'Date' column
merged_data = pd.merge(
    aapl[["Date", "Close"]],
    jpm[["Date", "Close"]],
    on="Date",
    suffixes=("_AAPL", "_JPM")
)


# In[14]:


merged_data = pd.merge(
    merged_data, 
    msft[["Date", "Close"]],
    on="Date",
)


# In[15]:


merged_data


# In[16]:


merged_data.rename(columns={"Close": "Close_MSFT"}, inplace=True) # inplace=True modifies original Dataframe
print(merged_data.head(10)) # top 10 rows


# In[17]:


plt.figure(figsize=(20, 15))
plt.plot(merged_data["Date"], merged_data["Close_AAPL"], marker="o", linestyle="--", color="red", label="AAPL")
plt.plot(merged_data["Date"], merged_data["Close_JPM"], marker="s", linestyle="--", color="blue", label="JPM")
plt.plot(merged_data["Date"], merged_data["Close_MSFT"], marker="^", linestyle="--", color="green", label="MSFT")
plt.grid(True)
plt.legend()
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.title("Stock Closing Prices")
plt.xticks(rotation=45)
plt.show()


# In[18]:


import plotly.graph_objects as go


# In[19]:


fig=go.Figure()

fig.add_trace(
  go.Scatter(
    x=merged_data["Date"],
    y=merged_data["Close_AAPL"],
    mode="lines+markers",
    name="AAPL"
  )
)

fig.add_trace(
    go.Scatter(
        x=merged_data["Date"],
        y=merged_data["Close_JPM"],
        mode="lines+markers",
        name="JPM"
    )
)

fig.add_trace(
    go.Scatter(
        x=merged_data["Date"],
        y=merged_data["Close_MSFT"],
        mode="lines+markers",
        name="MSFT"
    )
)

fig.update_layout(
    title="Closing Prices Comparison: AAPL, JPM, and MSFT",
    xaxis_title="Date",
    yaxis_title="Closing Price",
    xaxis_tickangle=45
)

fig.show()


# # Function to load and clean financial data

# In[20]:


import mplfinance as mpf


# In[21]:


def load_and_clean(file):
  df=pd.read_csv(file, parse_dates=["Date"])
  for col in ["Open","High","Low","Close"]:
    df[col]=pd.to_numeric(df[col],errors="coerce")
  df.dropna(inplace=True)
  # print(df["Date"].is_unique) # To check if "Date" column is unique
  df.set_index("Date", inplace=True)
  return df

aapl=load_and_clean("./datasets/AAPL.csv")
jpm=load_and_clean("./datasets/JPM.csv")
msft=load_and_clean("./datasets/MSFT.csv")


# In[22]:


ohlc_aapl = aapl[["Open", "High", "Low", "Close"]]
ohlc_jpm = jpm[["Open", "High", "Low", "Close"]]
ohlc_msft = msft[["Open", "High", "Low", "Close"]]


# In[23]:


mpf.plot(
    ohlc_aapl,
    type="ohlc",
    style="charles",
    title="AAPL OHLC Chart",
    ylabel="Price",
    figscale=3.0,
)


# In[24]:


mpf.plot(
  ohlc_aapl,
  type="candle",
  style="charles",
  title="AAPL CandleStick Chart",
  ylabel="Price",
  figscale=(3.0)
)


# In[25]:


mpf.plot(
    ohlc_jpm,
    type="ohlc",
    style="charles",
    title="JPM OHLC Chart",
    ylabel="Price",
    figscale=3.0,
)


# In[26]:


mpf.plot(
    ohlc_jpm,
    type="candle",
    style="charles",
    title="JPM OHLC Chart",
    ylabel="Price",
    figscale=3.0,
)


# In[27]:


mpf.plot(
    ohlc_msft,
    type="ohlc",
    style="charles",
    title="MSFT OHLC Chart",
    ylabel="Price",
    figscale=3.0,
)


# In[28]:


mpf.plot(
    ohlc_msft,
    type="candle",
    style="charles",
    title="MSFT OHLC Chart",
    ylabel="Price",
    figscale=3.0,
)


# In[29]:


# Use Plotly for OHLC and Candlestick Charts
fig = go.Figure()
fig.add_trace(
    go.Ohlc(
        x=aapl.index,
        open=aapl["Open"],
        high=aapl["High"],
        low=aapl["Low"],
        close=aapl["Close"],
        name="AAPl OHLC"
    )
)
fig.update_layout(
    title="AAPL OHLC",
    width=2000,
    height=1000
)
fig.show()


# In[30]:


# Use Plotly for OHLC and Candlestick Charts
fig = go.Figure()
fig.add_trace(
    go.Ohlc(
        x=jpm.index,
        open=jpm["Open"],
        high=jpm["High"],
        low=jpm["Low"],
        close=jpm["Close"],
        name="JPM OHLC"
    )
)
fig.update_layout(
    title="JPM OHLC",
    width=2000,
    height=1000
)
fig.show()


# In[31]:


# Use Plotly for OHLC and Candlestick Charts
fig = go.Figure()
fig.add_trace(
    go.Ohlc(
        x=msft.index,
        open=msft["Open"],
        high=msft["High"],
        low=msft["Low"],
        close=msft["Close"],
        name="MSFT OHLC"
    )
)
fig.update_layout(
    title="MSFT OHLC",
    width=2000,
    height=1000
)
fig.show()


# In[ ]:





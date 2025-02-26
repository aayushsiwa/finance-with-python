import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import plotly.graph_objects as go


def load_and_clean(file):
    df = pd.read_csv(file, parse_dates=["Date"])
    for col in ["Open", "High", "Low", "Close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(inplace=True)
    df.set_index("Date", inplace=True)
    return df


st.title("Stock Market Data Analysis")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = load_and_clean(uploaded_file)
    st.write("### Data Preview")
    st.dataframe(df.head())

    st.write("### Summary Statistics")
    st.write(df.describe())

    st.write("### Stock Price Trends")
    fig, ax = plt.subplots(figsize=(12, 6))
    df["Close"].plot(ax=ax, marker="o", linestyle="--", color="blue")
    ax.set_title("Closing Prices Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price")
    ax.grid(True)
    st.pyplot(fig)

    st.write("### OHLC Chart")
    fig = go.Figure()
    fig.add_trace(
        go.Ohlc(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="OHLC",
        )
    )
    fig.update_layout(title="OHLC Chart", width=1000, height=600)
    st.plotly_chart(fig)

    st.write("### Candlestick Chart")
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Candlestick",
        )
    )
    fig.update_layout(title="Candlestick Chart", width=1000, height=600)
    st.plotly_chart(fig)

st.write("Upload stock data to analyze and visualize market trends!")

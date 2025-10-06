import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from prophet import Prophet
from datetime import datetime, timedelta

# Streamlit UI
st.set_page_config(page_title="Business & Stock Sentiment Forecast", layout="wide")
st.title("📈 Business & Stock Market Sentiment Forecasting using News & Tweets")

# Sidebar inputs
st.sidebar.header("User Inputs")
keyword = st.sidebar.text_input("Enter keyword (e.g., 'Pakistan economy', 'stock market')", "Pakistan economy")
num_articles = st.sidebar.slider("Number of news articles to analyze", 5, 50, 20)

# Initialize News API
newsapi = NewsApiClient(api_key="ae2a696db51448d6b612cb7453abee0a")

# Fetch latest news
with st.spinner("Fetching latest news..."):
    all_articles = newsapi.get_everything(q=keyword, language='en', sort_by='relevancy', page_size=num_articles)
    articles = all_articles['articles']

# Sentiment analysis
analyzer = SentimentIntensityAnalyzer()
news_data = []
for article in articles:
    sentiment = analyzer.polarity_scores(article['title'])
    news_data.append({
        'Published At': article['publishedAt'][:10],
        'Title': article['title'],
        'Sentiment Score': sentiment['compound']
    })

df = pd.DataFrame(news_data)
df['Published At'] = pd.to_datetime(df['Published At'])

# Plotly visualization
st.subheader("📰 News Sentiment Trend")
fig = px.line(df, x='Published At', y='Sentiment Score', title='Daily Sentiment Scores', markers=True)
st.plotly_chart(fig, use_container_width=True)

# Forecast future sentiment using Prophet
st.subheader("🔮 Forecasting Future Sentiment")

df_prophet = df.groupby('Published At')['Sentiment Score'].mean().reset_index()
df_prophet.columns = ['ds', 'y']

if len(df_prophet) > 2:
    model = Prophet()
    model.fit(df_prophet)

    future = model.make_future_dataframe(periods=7)
    forecast = model.predict(future)

    fig_forecast = px.line(forecast, x='ds', y='yhat', title='Forecasted Sentiment for Next 7 Days')
    st.plotly_chart(fig_forecast, use_container_width=True)
else:
    st.warning("Not enough data points to forecast sentiment. Try increasing the number of articles.")

# Display data
st.subheader("📊 News Sentiment Data")
st.dataframe(df)


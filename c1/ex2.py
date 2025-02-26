# [Using the Coindesk API to collect bitcoin headlines for sentiment analysis.]
# The CoinDesk API provides access to the latest Bitcoin news without the need for authentication.
import json
import requests
from textblob import TextBlob

# Fetch the latest Bitcoin news from CoinDesk
url = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN"
response = requests.get(url)
data = response.json()

for article in data["Data"]:
  headline = article["title"]
  analysis = TextBlob(headline)
  sentiment = analysis.sentiment.polarity
  print(f"Headline: {headline}")
  print(f"Sentiment Score: {sentiment}\n")

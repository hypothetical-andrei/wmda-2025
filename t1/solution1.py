### **Exercise 1: Extracting and Cleaning Data from an API**
import requests
import pandas as pd
from pprint import pprint

cities = ["New York", "London", "Tokyo", "Paris", "Berlin"]
#  url = f"https://wttr.in/{city}?format=%C+%t"
search_url = "https://geocoding-api.open-meteo.com/v1/search?name={city}"
weather_url = "https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"

for city in cities:
  response= requests.get(search_url.format(city=city))
  data = response.json()
  main_result = data['results'][0]
  lat = main_result['latitude']
  lon = main_result['longitude']
  data = requests.get(weather_url.format(lat=lat, lon=lon)).json()
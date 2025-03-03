### **Exercise 1: Extracting and Cleaning Data from an API**
import requests
import pandas as pd

# Step 1: Define cities to fetch weather data for
cities = ["New York", "London", "Tokyo", "Paris", "Berlin"]

# Step 2: Fetch weather data from wttr.in (public API, no API key needed)
weather_data = []

for city in cities:
    url = f"https://wttr.in/{city}?format=%C+%t"  # Fetch condition and temperature
    response = requests.get(url)

    if response.status_code == 200:
        data = response.text.strip()
        condition, temperature = data.rsplit(" ", 1)  # Extract weather condition and temperature
        temperature = temperature.replace("°C", "")  # Clean temperature value

        weather_data.append([city, condition, temperature])
    else:
        print(f"Failed to retrieve data for {city}")

# Step 3: Convert data into a structured Pandas DataFrame
df = pd.DataFrame(weather_data, columns=["City", "Weather Condition", "Temperature (°C)"])

# Step 4: Save the cleaned data as a CSV file
df.to_csv("cleaned_weather_data.csv", index=False)

# Display the final cleaned dataset
print("Cleaned Weather Data:\n")
print(df)

# [Scraping product prices from an e-commerce website.]
import requests
from bs4 import BeautifulSoup

# Example e-commerce website (replace with an actual public e-commerce URL)
URL = "https://webscraper.io/test-sites/e-commerce/allinone/computers/laptops"

# Fetch the webpage content
response = requests.get(URL)
soup = BeautifulSoup(response.text, "html.parser")

# Extract product names and prices
products = soup.find_all("div", class_="thumbnail")

print("Scraped Product Prices:\n")

for product in products[:10]:  # Limit to first 10 products
    name = product.find("a", class_="title").text.strip()
    price = product.find("h4", class_="price").text.strip()
    print(f"Product: {name}")
    print(f"Price: {price}\n")

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import time

# Step 1: Set the path to ChromeDriver (Modify this path based on your system)
chromedriver_path = "./chromedriver-linux64/chromedriver"  # Change this to the actual path of your ChromeDriver

# Step 2: Configure Selenium options
chrome_options = Options()
chrome_options.add_argument("--headless")  # Run in headless mode
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--window-size=1920,1080")

# Step 3: Set up the ChromeDriver service with the specified path
service = Service(chromedriver_path)
driver = webdriver.Chrome(service=service, options=chrome_options)

# Step 4: Open a webpage (Modify with the actual e-commerce site)
url = "https://webscraper.io/test-sites/e-commerce/allinone/computers/laptops"
driver.get(url)

# Wait for page to load
time.sleep(5)

# Step 5: Inject JavaScript to extract product names and prices
script = """
const products = document.querySelectorAll('.product-wrapper');  // Modify the selector
let productData = [];
products.forEach(product => {
    let name = product.querySelector('h4 a') ? product.querySelector('h4 a').innerText : 'N/A';
    let price = product.querySelector('h4.price') ? product.querySelector('h4.price').innerText : 'N/A';
    productData.push({name, price});
});
return productData;
"""

# Execute JavaScript in the browser and retrieve data
product_list = driver.execute_script(script)

# Step 6: Print scraped data
print("\nScraped Product Prices:")
for product in product_list:
    print(f"Product: {product['name']} | Price: {product['price']}")

# Step 7: Close the browser
driver.quit()

### **Tutorial: Hands-on Data Processing & AI Problem Framing**
#### **Exercises (10 minutes each)**

---

### **Exercise 1: Extracting and Cleaning Data from an API**
**Objective:** Fetch and clean real-world data from a public API.

#### **Steps:**
1. Fetch weather data using the OpenWeatherMap API (no API key required, use: `https://wttr.in/?format=%C+%t`)
2. Parse the API response and extract relevant information (e.g., temperature, weather condition).
3. Clean the data by removing unnecessary characters and formatting it into a structured Pandas DataFrame.
4. Save the cleaned dataset as a CSV file.

**Expected Output:** A structured DataFrame with columns like `City`, `Temperature`, `Weather Condition`.

---

### **Exercise 2: Web Scraping a Product Listings Page**
**Objective:** Scrape and structure data from a public website.

#### **Steps:**
1. Use BeautifulSoup to scrape product names and prices from a test e-commerce site: [Web Scraper Test Site](https://webscraper.io/test-sites/e-commerce/allinone/computers/laptops).
2. Extract **product names** and **prices** from the page.
3. Store the extracted data in a Pandas DataFrame.
4. Remove duplicate product entries (if any).

**Expected Output:** A DataFrame with `Product Name` and `Price` columns containing at least 10 products.

---

### **Exercise 3: Implementing a Simple Recommendation System**
**Objective:** Build a basic collaborative filtering recommendation system using the **MovieLens dataset**.

#### **Steps:**
1. Load the **MovieLens 100K dataset** (`u.data` file) using Pandas.
2. Preprocess the dataset by filtering out users who have rated fewer than 10 movies.
3. Compute the **average rating per movie** and sort movies based on popularity.
4. Recommend the **top 5 most popular movies** for new users.

**Expected Output:** A ranked list of the top 5 movies based on user ratings.

---

### **Exercise 4: Feature Engineering for Classification**
**Objective:** Create new features and scale them for machine learning.

#### **Steps:**
1. Load the **Titanic dataset** from Seaborn (`sns.load_dataset('titanic')`).
2. Create a **new feature**: `family_size = sibsp + parch + 1` (Total family members onboard).
3. Encode categorical variables (`sex`, `embarked`) using **one-hot encoding**.
4. Scale the numerical features (`age`, `fare`, `family_size`) using **MinMaxScaler**.

**Expected Output:** A cleaned and transformed DataFrame ready for classification.

---

### **Exercise 5: Applying a Classification Model**
**Objective:** Train a simple classification model using the Titanic dataset.

#### **Steps:**
1. Use the preprocessed Titanic dataset from **Exercise 4**.
2. Split the data into **train and test sets** (`train_test_split`).
3. Train a **Logistic Regression** model to predict survival.
4. Evaluate the model using **accuracy, precision, and recall**.

**Expected Output:** A classification report showing model performance.

---

### **Bonus Exercise (If Time Permits): Hyperparameter Tuning for Classification**
**Objective:** Improve model performance using hyperparameter tuning.

#### **Steps:**
1. Use **GridSearchCV** to find the best parameters for a Logistic Regression model.
2. Tune `C` (regularization strength) and `penalty` (L1 vs. L2).
3. Compare the accuracy before and after tuning.

**Expected Output:** A summary of the best hyperparameters and improved model performance.

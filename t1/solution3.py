### **Exercise 3: Implementing a Simple Recommendation System**
import pandas as pd

# Step 1: Load the MovieLens 100K dataset
url = "https://files.grouplens.org/datasets/movielens/ml-100k/u.data"
column_names = ["user_id", "movie_id", "rating", "timestamp"]
ratings = pd.read_csv(url, sep="\t", names=column_names, usecols=["user_id", "movie_id", "rating"])

url_movies = "https://files.grouplens.org/datasets/movielens/ml-100k/u.item"
movies = pd.read_csv(url_movies, sep="|", encoding="latin-1", names=["movie_id", "title"], usecols=[0, 1])

# Step 2: Merge movie titles with ratings dataset
ratings = ratings.merge(movies, on="movie_id")

# Step 3: Filter out users who have rated fewer than 10 movies
user_ratings_count = ratings.groupby("user_id").size()
valid_users = user_ratings_count[user_ratings_count >= 10].index
filtered_ratings = ratings[ratings["user_id"].isin(valid_users)]

# Step 4: Compute the average rating per movie
movie_avg_ratings = filtered_ratings.groupby("title")["rating"].mean()

# Step 5: Get the top 5 most popular movies
top_movies = movie_avg_ratings.sort_values(ascending=False).head(5)

# Step 6: Display the recommended movies
print("Top 5 Most Popular Movies:\n")
print(top_movies)

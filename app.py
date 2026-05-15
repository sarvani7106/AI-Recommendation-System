import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
data = pd.read_csv("movies.csv")

# Combine important features
data["features"] = data["genre"]

# Convert text data into numbers
cv = CountVectorizer()

count_matrix = cv.fit_transform(data["features"])

# Calculate similarity between movies
similarity = cosine_similarity(count_matrix)

# Recommendation function
def recommend(movie_name):
    movie_name = movie_name.lower()

    # Find movie index
    matching_movies = data[data["title"].str.lower() == movie_name]

    if matching_movies.empty:
        print("Movie not found!")
        return

    movie_index = matching_movies.index[0]

    # Get similarity scores
    scores = list(enumerate(similarity[movie_index]))

    # Sort by similarity
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

    print(f"\nRecommended movies for '{movie_name.title()}':\n")

    # Show top recommendations
    for movie in sorted_scores[1:6]:
        index = movie[0]
        print(data.iloc[index]["title"])

# User input
movie = input("Enter your favorite movie: ")

recommend(movie)
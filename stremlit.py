import streamlit as st
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# --- Data Collection and Pre-Processing ---
@st.cache_data
def load_data():
    movies_data = pd.read_csv('movies.csv.csv')
    
    # Handling missing values
    selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
    for feature in selected_features:
        movies_data[feature] = movies_data[feature].fillna('')
        
    # Combining features for similarity
    combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + movies_data['tagline'] + ' ' + movies_data['cast'] + ' ' + movies_data['director']
    
    # Vectorizing the features using TF-IDF
    vectorizer = TfidfVectorizer()
    feature_vectors = vectorizer.fit_transform(combined_features)
    
    # Calculating similarity
    similarity = cosine_similarity(feature_vectors)
    
    return movies_data, similarity

# --- Load Data ---
movies_data, similarity = load_data()

# --- Streamlit UI ---
st.title("🎬 Movie Recommendation System")

# Dropdown for movie selection
list_of_all_titles = movies_data['title'].tolist()
selected_movie = st.selectbox('Select a movie:', list_of_all_titles)

# Or, allow the user to type the movie name
movie_name = st.text_input('Or, type a movie name:', '')

# Button to generate recommendations
if st.button("Recommend"):
    
    if not selected_movie and not movie_name:
        st.warning("Please enter or select a movie name.")
    else:
        # If a movie name is typed, use it, otherwise use the dropdown selection
        movie_name = selected_movie if selected_movie else movie_name
        
        # Find close match to the movie name
        find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
        
        if not find_close_match:
            st.error("Movie not found in the dataset. Try again.")
        else:
            close_match = find_close_match[0]
            st.write(f"Using movie: {close_match}")
            
            # Get the index of the movie
            index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
            st.write(f"Index of the movie: {index_of_the_movie}")  # <-- **Printing the index here**
            
            # Get similarity scores
            similarity_score = list(enumerate(similarity[index_of_the_movie]))
            
            # Sort the movies based on their similarity score
            sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)
            
            # Get the top 10 similar movies
            top_movies = sorted_similar_movies[1:11]  # Excluding the movie itself
            movie_titles = [movies_data.iloc[movie[0]]['title'] for movie in top_movies]
            similarities = [movie[1] for movie in top_movies]
            
            # Display similar movies
            st.subheader('Movies Suggested for You:')
            for i, movie in enumerate(movie_titles, 1):
                st.write(f"{i}. {movie}")
            
            # Plot the top 10 similar movies
            st.subheader(f"Top 10 Movies Similar to '{movie_name}'")
            plt.figure(figsize=(10, 5))
            plt.barh(movie_titles[::-1], similarities[::-1], color='skyblue')
            plt.xlabel("Cosine Similarity Score")
            plt.title(f"Top 10 Movies Similar to '{movie_name}'")
            plt.tight_layout()
            st.pyplot(plt)

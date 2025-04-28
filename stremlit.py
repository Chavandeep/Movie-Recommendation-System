import streamlit as st
import pandas as pd
import difflib
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Your TMDB API Key
API_KEY = '43a476b1b6a29938820e3eac8a0f423e'  # <-- Replace this with your API key

# --- Function to fetch movie details (poster, rating, overview) ---
def fetch_movie_details(movie_title):
    try:
        query = movie_title.replace(' ', '%20')
        url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={query}"
        response = requests.get(url)
        data = response.json()
        if data['results']:
            movie_data = data['results'][0]
            poster_path = movie_data.get('poster_path')
            full_poster_path = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None
            rating = movie_data.get('vote_average', 'N/A')
            overview = movie_data.get('overview', 'No description available.')
            return full_poster_path, rating, overview
        else:
            return None, 'N/A', 'No description available.'
    except:
        return None, 'N/A', 'No description available.'

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

# Add a placeholder at the beginning of the movie list
list_of_all_titles_with_placeholder = ['Select a movie'] + list_of_all_titles

selected_movie = st.selectbox('Select a movie:', list_of_all_titles_with_placeholder)

# Allow the user to type the movie name
typed_movie_name = st.text_input('Or, type a movie name:', '')

# Button to generate recommendations
if st.button("Recommend"):
    # If user neither selects nor types
    if (selected_movie == 'Select a movie') and (typed_movie_name.strip() == ''):
        st.warning("Please enter or select a movie name.")
    else:
        # Priority: typed name first, else selected dropdown
        movie_name = typed_movie_name if typed_movie_name.strip() != '' else selected_movie
        
        find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

        if not find_close_match:
            st.error("Movie not found in the dataset. Try again.")
        else:
            close_match = find_close_match[0]
            st.success(f"Using movie: **{close_match}**")
            index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
            similarity_score = list(enumerate(similarity[index_of_the_movie]))
            sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)
            
            top_movies = sorted_similar_movies[1:11]  # Exclude the movie itself
            movie_titles = [movies_data.iloc[movie[0]]['title'] for movie in top_movies]
            similarities = [movie[1] for movie in top_movies]
            
            # --- Display similar movies in LIST format ---
            st.subheader('Movies Suggested for You:')
            
            for idx, title in enumerate(movie_titles):
                poster_url, rating, overview = fetch_movie_details(title)
                with st.container():
                    st.markdown(f"### {idx+1}. {title}")
                    cols = st.columns([1, 4])  # Poster and Info
                    
                    with cols[0]:
                        if poster_url:
                            st.image(poster_url, width=120)
                        else:
                            st.image('https://via.placeholder.com/120x180?text=No+Image', width=120)
                    
                    with cols[1]:
                        st.markdown(f"**Rating:** {rating} ⭐")
                        st.markdown(f"**Description:** {overview}")
                        st.markdown("---")  # Divider between movies

            # --- Plot the top 10 similar movies (Cosine Similarity Score Bar Chart) ---
            st.subheader(f"Top 10 Movies Similar to '{movie_name}'")
            plt.figure(figsize=(10, 5))
            plt.barh(movie_titles[::-1], similarities[::-1], color='skyblue')
            plt.xlabel("Cosine Similarity Score")
            plt.title(f"Top 10 Movies Similar to '{movie_name}'")
            plt.tight_layout()
            st.pyplot(plt)


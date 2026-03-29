import streamlit as st
from recommender import (
    load_data,
    build_content_model,
    get_content_recommendations,
    get_popular_movies,
)

st.set_page_config(page_title="Movie Recommendation System", layout="wide")


def main():
    st.title("🎬 Movie Recommendation System")
    st.write("Recommend movies using Content-Based Filtering + Popularity")

    try:
        movies, ratings = load_data()
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return

    content_data = build_content_model(movies)

    menu = st.sidebar.selectbox(
        "Choose Recommendation Type",
        [
            "Content-Based Recommendation",
            "Popular Movies"
        ]
    )

    if menu == "Content-Based Recommendation":
        st.subheader("Content-Based Recommendation")
        movie_titles = movies["title"].dropna().unique().tolist()
        selected_movie = st.selectbox("Select a movie", sorted(movie_titles))
        top_n = st.slider("Top N Recommendations", 3, 10, 5)

        if st.button("Get Content Recommendations"):
            result = get_content_recommendations(selected_movie, movies, content_data, top_n)
            if result.empty:
                st.warning("No recommendations found.")
            else:
                st.dataframe(result, use_container_width=True)

    else:
        st.subheader("Popular Movies")
        top_n = st.slider("Top N Popular Movies", 3, 10, 5, key="popular_top_n")
        result = get_popular_movies(movies, ratings, top_n)
        st.dataframe(result, use_container_width=True)

    st.markdown("---")
    st.subheader("Dataset Preview")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Movies")
        st.dataframe(movies.head(10), use_container_width=True)

    with col2:
        st.write("Ratings")
        st.dataframe(ratings.head(10), use_container_width=True)


if __name__ == "__main__":
    main()
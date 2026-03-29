import streamlit as st
from recommender import (
    load_data,
    build_content_model,
    build_collaborative_model,
    get_content_recommendations,
    get_user_recommendations,
    get_hybrid_recommendations,
    get_popular_movies,
)

st.set_page_config(page_title="Hybrid Movie Recommendation System", layout="wide")


def main():
    st.title("🎬 Hybrid Movie Recommendation System")
    st.write("Recommend movies using Content-Based Filtering + Collaborative Filtering + Popularity")

    try:
        movies, ratings = load_data()
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return

    content_data = build_content_model(movies)
    svd_model, trainset = build_collaborative_model(ratings)

    menu = st.sidebar.selectbox(
        "Choose Recommendation Type",
        [
            "Content-Based Recommendation",
            "User-Based Recommendation",
            "Hybrid Recommendation",
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

    elif menu == "User-Based Recommendation":
        st.subheader("Collaborative Filtering Recommendation")
        user_ids = sorted(ratings["userId"].unique().tolist())
        selected_user = st.selectbox("Select User ID", user_ids)
        top_n = st.slider("Top N Recommendations", 3, 10, 5, key="user_top_n")

        if st.button("Get User Recommendations"):
            result = get_user_recommendations(selected_user, movies, ratings, svd_model, top_n)
            if result.empty:
                st.warning("No recommendations found for this user.")
            else:
                st.dataframe(result, use_container_width=True)

    elif menu == "Hybrid Recommendation":
        st.subheader("Hybrid Recommendation")
        user_ids = sorted(ratings["userId"].unique().tolist())
        movie_titles = movies["title"].dropna().unique().tolist()

        selected_user = st.selectbox("Select User ID", user_ids, key="hybrid_user")
        selected_movie = st.selectbox("Select a movie", sorted(movie_titles), key="hybrid_movie")
        top_n = st.slider("Top N Recommendations", 3, 10, 5, key="hybrid_top_n")

        if st.button("Get Hybrid Recommendations"):
            result = get_hybrid_recommendations(
                user_id=selected_user,
                movie_title=selected_movie,
                movies=movies,
                ratings=ratings,
                content_data=content_data,
                svd_model=svd_model,
                top_n=top_n
            )
            if result.empty:
                st.warning("No hybrid recommendations found.")
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
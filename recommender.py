import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_data():
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")
    return movies, ratings


def build_content_model(movies: pd.DataFrame):
    movies = movies.copy()
    movies["genres"] = movies["genres"].fillna("")
    movies["description"] = movies["description"].fillna("")
    movies["content"] = movies["genres"] + " " + movies["description"]

    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies["content"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    indices = pd.Series(movies.index, index=movies["title"]).drop_duplicates()

    return {
        "movies": movies,
        "cosine_sim": cosine_sim,
        "indices": indices
    }


def get_content_recommendations(movie_title, movies, content_data, top_n=5):
    indices = content_data["indices"]
    cosine_sim = content_data["cosine_sim"]
    model_movies = content_data["movies"]

    if movie_title not in indices:
        return pd.DataFrame()

    idx = indices[movie_title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n + 10]

    movie_indices = [i[0] for i in sim_scores]
    scores = [i[1] for i in sim_scores]

    result = model_movies.iloc[movie_indices][["movieId", "title", "genres"]].copy()
    result["content_score"] = scores
    result = result.head(top_n)

    return result.reset_index(drop=True)


def get_popular_movies(movies, ratings, top_n=5):
    stats = ratings.groupby("movieId").agg(
        avg_rating=("rating", "mean"),
        rating_count=("rating", "count")
    ).reset_index()

    merged = movies.merge(stats, on="movieId", how="left")
    merged["avg_rating"] = merged["avg_rating"].fillna(0)
    merged["rating_count"] = merged["rating_count"].fillna(0)

    merged["popularity_score"] = (merged["avg_rating"] * 0.7) + (
        merged["rating_count"] / max(merged["rating_count"].max(), 1) * 0.3
    )

    merged = merged.sort_values(by="popularity_score", ascending=False)

    return merged[
        ["movieId", "title", "genres", "avg_rating", "rating_count", "popularity_score"]
    ].head(top_n).reset_index(drop=True)
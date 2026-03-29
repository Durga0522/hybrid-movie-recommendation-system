import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD


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


def build_collaborative_model(ratings: pd.DataFrame):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)
    trainset = data.build_full_trainset()

    model = SVD()
    model.fit(trainset)

    return model, trainset


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


def get_user_recommendations(user_id, movies, ratings, svd_model, top_n=5):
    watched_movies = ratings[ratings["userId"] == user_id]["movieId"].tolist()
    unwatched_movies = movies[~movies["movieId"].isin(watched_movies)].copy()

    if unwatched_movies.empty:
        return pd.DataFrame()

    predicted_scores = []
    for movie_id in unwatched_movies["movieId"]:
        pred = svd_model.predict(user_id, movie_id)
        predicted_scores.append(pred.est)

    unwatched_movies["predicted_rating"] = predicted_scores
    unwatched_movies = unwatched_movies.sort_values(by="predicted_rating", ascending=False)

    return unwatched_movies[["movieId", "title", "genres", "predicted_rating"]].head(top_n).reset_index(drop=True)


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

    return merged[["movieId", "title", "genres", "avg_rating", "rating_count", "popularity_score"]].head(top_n).reset_index(drop=True)


def get_hybrid_recommendations(user_id, movie_title, movies, ratings, content_data, svd_model, top_n=5):
    content_recs = get_content_recommendations(movie_title, movies, content_data, top_n=20)

    if content_recs.empty:
        return pd.DataFrame()

    watched_movies = ratings[ratings["userId"] == user_id]["movieId"].tolist()

    content_recs = content_recs[~content_recs["movieId"].isin(watched_movies)].copy()

    if content_recs.empty:
        return pd.DataFrame()

    collaborative_scores = []
    for movie_id in content_recs["movieId"]:
        pred = svd_model.predict(user_id, movie_id)
        collaborative_scores.append(pred.est)

    content_recs["predicted_rating"] = collaborative_scores

    stats = ratings.groupby("movieId").agg(
        avg_rating=("rating", "mean"),
        rating_count=("rating", "count")
    ).reset_index()

    content_recs = content_recs.merge(stats, on="movieId", how="left")
    content_recs["avg_rating"] = content_recs["avg_rating"].fillna(0)
    content_recs["rating_count"] = content_recs["rating_count"].fillna(0)

    max_content = max(content_recs["content_score"].max(), 1)
    max_pred = max(content_recs["predicted_rating"].max(), 1)
    max_count = max(content_recs["rating_count"].max(), 1)

    content_recs["norm_content"] = content_recs["content_score"] / max_content
    content_recs["norm_pred"] = content_recs["predicted_rating"] / max_pred
    content_recs["norm_count"] = content_recs["rating_count"] / max_count

    content_recs["hybrid_score"] = (
        0.4 * content_recs["norm_content"] +
        0.4 * content_recs["norm_pred"] +
        0.2 * content_recs["norm_count"]
    )

    content_recs = content_recs.sort_values(by="hybrid_score", ascending=False)

    return content_recs[
        ["movieId", "title", "genres", "content_score", "predicted_rating", "avg_rating", "rating_count", "hybrid_score"]
    ].head(top_n).reset_index(drop=True)
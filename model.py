import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os

DATA_PATH = os.path.join("data", "tmdb_5000_movies.csv")
MODEL_PATH = "tfidf_matrix.pkl"
VECT_PATH = "vectorizer.pkl"

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df["overview"] = df["overview"].fillna("").astype(str)
    df["title"] = df["title"].fillna("").astype(str)
    df["title_lower"] = df["title"].str.lower()
    return df

def build_and_save_model(df):
    tfidf = TfidfVectorizer(stop_words="english", max_features=20000)
    tfidf_matrix = tfidf.fit_transform(df["overview"])

    joblib.dump(tfidf, VECT_PATH)
    joblib.dump(tfidf_matrix, MODEL_PATH)

    return tfidf, tfidf_matrix

def load_model():
    if os.path.exists(VECT_PATH) and os.path.exists(MODEL_PATH):
        tfidf = joblib.load(VECT_PATH)
        tfidf_matrix = joblib.load(MODEL_PATH)
        return tfidf, tfidf_matrix
    return None, None

def ensure_model(df):
    tfidf, tfidf_matrix = load_model()
    if tfidf is None:
        print("Building TF-IDF model...")
        tfidf, tfidf_matrix = build_and_save_model(df)
    return tfidf, tfidf_matrix

def recommend(title, df, tfidf, tfidf_matrix, top_n=5):
    title = title.lower().strip()
    row = df[df["title_lower"] == title]

    if row.empty:
        raise ValueError(f"Movie '{title}' not found!")

    idx = row.index[0]
    movie_vec = tfidf.transform([df.loc[idx, "overview"]])
    cosines = cosine_similarity(movie_vec, tfidf_matrix).flatten()

    top_indices = np.argsort(-cosines)[1:top_n+1]

    recommendations = []
    for i in top_indices:
        recommendations.append({
            "title": df.loc[i, "title"],
            "overview": df.loc[i, "overview"],
            "score": float(cosines[i])
        })
    return recommendations

if __name__ == "__main__":
    df = load_data()
    tfidf, tfidf_matrix = ensure_model(df)
    print("Model built!")

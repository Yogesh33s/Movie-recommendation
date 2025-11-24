# app.py
import os
import streamlit as st
import pandas as pd
from model import ensure_model, recommend

st.set_page_config(page_title="Movie Recommender", layout="wide")

# ---------- Data loading ----------
LOCAL_UPLOADED_CSV = "/mnt/data/tmdb_5000_movies.csv"
DATA_PATH_IN_REPO = os.path.join("data", "tmdb_5000_movies.csv")

def get_data_path():
    if os.path.exists(LOCAL_UPLOADED_CSV):
        return LOCAL_UPLOADED_CSV
    return DATA_PATH_IN_REPO

def load_df_from_path(path):
    df = pd.read_csv(path)
    df["overview"] = df["overview"].fillna("").astype(str)
    df["title"] = df["title"].fillna("").astype(str)
    df["title_lower"] = df["title"].str.lower()
    return df

data_path = get_data_path()
try:
    df = load_df_from_path(data_path)
except Exception as e:
    st.error(f"Could not load dataset from `{data_path}`: {e}")
    st.stop()

# ---------- Model ----------
tfidf, tfidf_matrix = ensure_model(df)

# ---------- Header ----------
st.title("🎬 Movie Recommendation System")
st.write("Content-Based | TF-IDF | Cosine Similarity")

# ---------- Input: Only manual text input (no dropdown) ----------
st.write("Type the movie title exactly as in the dataset (or paste it).")
manual_input = st.text_input("Enter movie title:")

# ---------- Number of recommendations box (display like 05) ----------
# We'll use a small text box, validate and convert to integer in range 1-10.
num_text = st.text_input("Number of recommendations (01-10)", value="05", max_chars=2)
# sanitize input
try:
    num_val = int(num_text)
except:
    num_val = 5

# clamp the number
if num_val < 1:
    num_val = 1
if num_val > 10:
    num_val = 10

# display formatted value back to user (two digits)
# Note: we don't overwrite until user updates again; but show the chosen value below
st.write(f"Selected: [{num_val:02d}]")

# ---------- Recommend button with spinner ----------
if st.button("Recommend"):
    chosen = manual_input.strip()
    if chosen == "":
        st.warning("Please type a movie title in the input box.")
    else:
        with st.spinner("Searching...."):
            try:
                recs = recommend(chosen, df, tfidf, tfidf_matrix, top_n=num_val)
            except Exception as e:
                st.error(str(e))
                recs = []

        if recs:
            st.success(f"Top {num_val} recommendations for **{chosen}**")
            for i, r in enumerate(recs, start=1):
                st.subheader(f"{i}. {r['title']}  —  score {r['score']:.3f}")
                st.write(r["overview"][:500] + ("..." if len(r["overview"]) > 500 else ""))
                st.markdown("---")
        else:
            st.info("No recommendations found.")

# ---------- Optional: dataset sample ----------
with st.expander("Show dataset sample"):
    st.dataframe(df.head(10))

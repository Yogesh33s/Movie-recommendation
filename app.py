# app.py
import os
import streamlit as st
import wikipedia
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

st.set_page_config(page_title="Movie Recommender — Wikipedia (fast)", layout="wide")

# -----------------------------
# Fast Wikipedia recommender (cached)
# -----------------------------
# Cache fetched summaries to avoid repeated network calls
@st.cache_data(show_spinner=False)
def fetch_wiki_summary(title, sentences=2):
    try:
        return wikipedia.summary(title, sentences=sentences)
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def fast_recommend_from_wikipedia(query_title, top_n=5, max_candidates=20):
    """
    Fast Wikipedia-only recommendations:
      - get source summary for query_title
      - use wikipedia.search to get candidate titles (fast)
      - fetch short summaries for top candidates
      - compute TF-IDF on source + candidates and return top_n similar
    """
    out = []
    try:
        source_summary = wikipedia.summary(query_title, sentences=4)
    except Exception as e:
        return out, f"Wikipedia lookup failed for '{query_title}': {e}"

    # get candidate titles quickly (no heavy page() calls)
    try:
        candidate_titles = wikipedia.search(query_title, results=max_candidates)
    except Exception as e:
        candidate_titles = []

    # remove exact match from candidates (we already have source_summary)
    candidate_titles = [t for t in candidate_titles if t.lower() != query_title.lower()]

    # fetch short summaries (cached)
    candidate_texts = []
    candidate_tuples = []
    for t in candidate_titles:
        s = fetch_wiki_summary(t, sentences=2)
        if s:
            candidate_texts.append(s)
            candidate_tuples.append((t, s))

    if not candidate_tuples:
        return out, "No candidate summaries found on Wikipedia."

    # TF-IDF and similarity (fit on small set)
    texts = [source_summary] + candidate_texts
    tfidf = TfidfVectorizer(stop_words="english", max_features=8000)
    X = tfidf.fit_transform(texts)

    source_vec = X[0]
    cand_vecs = X[1:]
    cosines = cosine_similarity(source_vec, cand_vecs).flatten()
    idxs = cosines.argsort()[::-1][:top_n]

    for i in idxs:
        title, summary = candidate_tuples[i]
        score = float(cosines[i])
        out.append({"title": title, "overview": summary, "score": score})

    return out, None

# -----------------------------
# Optional local CSV-based recommender
# (only used if user enables it)
# -----------------------------
LOCAL_CSV_PATH = "/mnt/data/tmdb_5000_movies.csv"  # local file you uploaded earlier

@st.cache_data(show_spinner=False)
def load_local_csv(path):
    df = pd.read_csv(path)
    df["overview"] = df.get("overview", "").fillna("").astype(str)
    df["title"] = df.get("title", "").fillna("").astype(str)
    df["title_lower"] = df["title"].str.lower()
    return df

@st.cache_data(show_spinner=False)
def build_tfidf_from_df(df):
    tfidf = TfidfVectorizer(stop_words="english", max_features=20000)
    matrix = tfidf.fit_transform(df["overview"].astype(str).fillna(""))
    return tfidf, matrix

def recommend_from_local_csv(title, df, tfidf, matrix, top_n=5):
    title_l = title.lower().strip()
    row = df[df["title_lower"] == title_l]
    if row.empty:
        raise ValueError(f"Movie '{title}' not found in local dataset.")
    idx = row.index[0]
    movie_vec = tfidf.transform([df.loc[idx, "overview"]])
    cosines = cosine_similarity(movie_vec, matrix).flatten()
    top_indices = cosines.argsort()[::-1][1:top_n+1]  # skip self
    recs = []
    for i in top_indices:
        recs.append({
            "title": df.loc[i, "title"],
            "overview": df.loc[i, "overview"],
            "score": float(cosines[i])
        })
    return recs

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🎬 Movie Recommendation — Wikipedia-based (fast)")
st.write("This app finds movie recommendations using **Wikipedia** summaries (fast, cached). Optional: use a local CSV if you explicitly enable it.")

st.markdown("### 1) Industry (affects heuristics; optional)")
industry = st.radio("Industry", options=["All", "Hollywood", "Bollywood"], index=0, horizontal=True)

st.markdown("### 2) Search (typos accepted)")
user_input = st.text_input("Enter movie title (typos OK):").strip()

st.markdown("### 3) Number of recommendations")
num_text = st.text_input("Enter number (1-10)", value="05", max_chars=2)
try:
    num_val = int(num_text)
except:
    num_val = 5
num_val = max(1, min(10, num_val))
st.write(f"Selected: [{num_val:02d}]")

st.markdown("---")
st.write("Optional: Use local uploaded dataset (only if you want CSV-based recommendations).")
use_local = st.checkbox("Use local CSV dataset (fallback)", value=False)

# If local requested but file missing, warn
local_df = None
local_tfidf = None
local_matrix = None
if use_local:
    if os.path.exists(LOCAL_CSV_PATH):
        try:
            with st.spinner("Loading local dataset..."):
                local_df = load_local_csv(LOCAL_CSV_PATH)
                local_tfidf, local_matrix = build_tfidf_from_df(local_df)
        except Exception as e:
            st.error(f"Failed to load local CSV: {e}")
            use_local = False
    else:
        st.warning(f"Local CSV not found at {LOCAL_CSV_PATH}. Uncheck 'Use local CSV dataset' to use Wikipedia-only.")
        use_local = False

# Recommend action
if st.button("Recommend"):
    if user_input == "":
        st.warning("Please enter a movie title.")
    else:
        with st.spinner("Searching...."):
            # If user opted to use local CSV and it's available
            if use_local and local_df is not None:
                try:
                    recs = recommend_from_local_csv(user_input, local_df, local_tfidf, local_matrix, top_n=num_val)
                    src = f"Local CSV: {os.path.basename(LOCAL_CSV_PATH)}"
                except Exception as e:
                    # fallback to Wikipedia if local fails
                    recs = []
                    st.info(f"Local CSV fallback: {e} — switching to Wikipedia-based recommendations.")
                    recs, err = fast_recommend_from_wikipedia(user_input, top_n=num_val, max_candidates=20)
                    if err:
                        st.error(err)
                    src = "Wikipedia (fallback)"
            else:
                # Use fast Wikipedia recommender
                recs, err = fast_recommend_from_wikipedia(user_input, top_n=num_val, max_candidates=20)
                if err:
                    st.error(err)
                    recs = []
                src = "Wikipedia"

        # Display results
        if recs:
            st.success(f"Top {num_val} recommendations for **{user_input}** — source: {src}")
            for i, r in enumerate(recs, start=1):
                st.subheader(f"{i}. {r['title']}  —  score {r['score']:.3f}")
                st.write(r.get("overview", "")[:700] + ("..." if len(r.get("overview", "")) > 700 else ""))
                st.markdown("---")
        else:
            st.info("No recommendations found. Try another title or try enabling the local CSV (if you have it).")

st.markdown("---")
st.caption(f"Local dataset file (optional, not required): `{LOCAL_CSV_PATH}`")

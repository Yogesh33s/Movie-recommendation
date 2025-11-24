import streamlit as st
import wikipedia
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommendation — Wikipedia", layout="wide")

# -------------------------------------------------------------
# Cached Wikipedia summary fetch
# -------------------------------------------------------------
@st.cache_data(show_spinner=False)
def fetch_wiki_summary(title, sentences=2):
    try:
        return wikipedia.summary(title, sentences=sentences)
    except Exception:
        return None


# -------------------------------------------------------------
# FAST Wikipedia Recommendation (NO CSV)
# -------------------------------------------------------------
@st.cache_data(show_spinner=False)
def fast_recommend_from_wikipedia(query_title, top_n=5, max_candidates=20):
    out = []
    try:
        source_summary = wikipedia.summary(query_title, sentences=4)
    except Exception as e:
        return out, f"Wikipedia lookup failed for '{query_title}': {e}"

    try:
        candidate_titles = wikipedia.search(query_title, results=max_candidates)
    except Exception as e:
        candidate_titles = []

    candidate_titles = [t for t in candidate_titles if t.lower() != query_title.lower()]

    candidate_texts = []
    candidate_tuples = []

    for t in candidate_titles:
        s = fetch_wiki_summary(t, sentences=2)
        if s:
            candidate_texts.append(s)
            candidate_tuples.append((t, s))

    if not candidate_tuples:
        return out, "No related Wikipedia pages found."

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
        out.append({
            "title": title,
            "overview": summary,
            "score": score
        })

    return out, None


# -------------------------------------------------------------
# UI Layout
# -------------------------------------------------------------
st.title("🎬 Movie Recommendation — Wikipedia-based")

st.write(
    "This system uses **Wikipedia searches and summaries** to find closely related movies or titles. "
    "No dataset or CSV is required."
)

# ---------------- Industry Filter (Heuristic only) -----------
st.subheader("1) Industry (affects heuristics slightly)")
industry = st.radio("Industry", ["All", "Hollywood", "Bollywood"], horizontal=True)

# ---------------------- Movie Input --------------------------
st.subheader("2) Search (typos accepted)")
movie_query = st.text_input("Enter movie name")

# ---------------------- Number of Recs ------------------------
st.subheader("3) Number of recommendations")
num_text = st.text_input("Enter number (1-10):", value="05", max_chars=2)

try:
    num_val = int(num_text)
except:
    num_val = 5

num_val = max(1, min(num_val, 10))
st.write(f"Selected: **[{num_val:02d}]**")

# ---------------------- Recommend Button ----------------------
if st.button("Recommend"):
    if movie_query.strip() == "":
        st.error("Please enter a movie name.")
    else:
        with st.spinner("Searching Wikipedia..."):
            results, err = fast_recommend_from_wikipedia(movie_query, top_n=num_val)

        if err:
            st.error(err)
        else:
            st.success(f"Top {num_val} recommendations for: **{movie_query}**")

            for i, r in enumerate(results, start=1):
                st.markdown(f"### {i}. **{r['title']}** — score `{r['score']:.3f}`")
                st.write(r["overview"])
                st.write("---")

# ---------------- Note ----------------
st.caption("Using Wikipedia only — no CSV file required.")

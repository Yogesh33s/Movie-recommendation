# app.py
import os
import streamlit as st
import pandas as pd
from model import load_data, ensure_model, recommend

st.set_page_config(page_title="Movie Recommender", layout="wide")

# ----------  Local fallback for the uploaded CSV (useful for quick testing) ----------
# If you uploaded the CSV file here during our chat, it's available at:
LOCAL_UPLOADED_CSV = "/mnt/data/tmdb_5000_movies.csv"
DATA_PATH_IN_REPO = os.path.join("data", "tmdb_5000_movies.csv")

def get_data_path():
    # Prefer the local uploaded copy if present (for quick testing)
    if os.path.exists(LOCAL_UPLOADED_CSV):
        return LOCAL_UPLOADED_CSV
    return DATA_PATH_IN_REPO

# load_data from model.py uses DATA_PATH constant; we will override by temporarily patching it.
# Simpler approach: read CSV here and pass df to the rest of the app.
def load_df_from_path(path):
    df = pd.read_csv(path)
    df["overview"] = df["overview"].fillna("").astype(str)
    df["title"] = df["title"].fillna("").astype(str)
    df["title_lower"] = df["title"].str.lower()
    return df

# ---------- Load dataset ----------
data_path = get_data_path()
try:
    df = load_df_from_path(data_path)
except Exception as e:
    st.error(f"Could not load dataset from `{data_path}`: {e}")
    st.stop()

# ---------- Build / load TF-IDF model ----------
# We reuse ensure_model from model.py but it expects the same dataframe columns.
tfidf, tfidf_matrix = ensure_model(df)

# ---------- UI: Dark mode toggle via CSS ----------
# Use a checkbox to switch between dark and light CSS injected into the page.
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

def local_css(css: str):
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

dark_css = """
/* Basic dark theme overrides (best-effort) */
.reportview-container, .main, .block-container {
    background-color: #0e1117 !important;
    color: #e6eef8 !important;
}
.stButton>button, .st-bx {
    color: #e6eef8 !important;
}
.stTextInput>div>div>input, textarea, .stSelectbox>div>div>div>select {
    background-color: #0f1720 !important;
    color: #e6eef8 !important;
}
header, .stAppHeader {
    background-color: #0e1117 !important;
}
.st-expander, .st-card {
    background-color: #0f1720 !important;
    color: #e6eef8 !important;
}
"""

light_css = """
/* Minimal light theme - revert to default */
.reportview-container, .main, .block-container {
    background-color: white !important;
    color: black !important;
}
"""

# Render toggle in top-right area using columns for alignment
col1, col2, col3 = st.columns([6, 2, 2])
with col3:
    dm = st.checkbox("Dark mode", value=st.session_state.dark_mode, key="dark_mode_checkbox")
    st.session_state.dark_mode = dm

if st.session_state.dark_mode:
    local_css(dark_css)
else:
    local_css(light_css)

# ---------- Page header ----------
st.title("🎬 Movie Recommendation System")
st.write("Content-Based | TF-IDF | Cosine Similarity")

# ---------- Autocomplete-like search (searchable selectbox) ----------
titles = df["title"].tolist()
# We'll provide a selectbox which is searchable in Streamlit
st.write("Type or search the movie title in the dropdown (search is supported).")
selected_title = st.selectbox("Pick a movie (searchable)", options=[""] + titles, index=0)

# Also provide a manual text input (keeps previous behaviour)
manual_input = st.text_input("Or type a movie title manually (optional)")

# Final chosen title preference: manual input if not empty else selectbox
chosen = manual_input.strip() if manual_input.strip() else (selected_title if selected_title else "")

# Number of recommendations
top_n = st.slider("Number of recommendations", min_value=1, max_value=10, value=5)

# Recommend button (with spinner)
if st.button("Recommend"):
    if chosen == "":
        st.warning("Please select or type a movie title.")
    else:
        # Use spinner with custom text "Searching...."
        with st.spinner("Searching...."):
            try:
                recs = recommend(chosen, df, tfidf, tfidf_matrix, top_n=top_n)
            except Exception as e:
                st.error(str(e))
                recs = []

        if recs:
            st.success(f"Top {top_n} recommendations for **{chosen}**")
            # Display recommendations as simple cards
            for i, r in enumerate(recs, start=1):
                st.subheader(f"{i}. {r['title']}  —  score {r['score']:.3f}")
                st.write(r["overview"][:500] + ("..." if len(r["overview"]) > 500 else ""))
                st.markdown("---")
        else:
            st.info("No recommendations found.")

# ---------- Show dataset sample ----------
with st.expander("Show dataset sample"):
    st.dataframe(df.head(10))

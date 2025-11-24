import streamlit as st
import pandas as pd
from model import load_data, ensure_model, recommend

st.set_page_config(page_title="Movie Recommender", layout="wide")

st.title("🎬 Movie Recommendation System")
st.write("Content-Based | TF-IDF | Cosine Similarity")

# Load data
df = load_data()

# Load / build model
tfidf, tfidf_matrix = ensure_model(df)

# Movie input
movie_name = st.text_input("Enter a movie title:")

if st.button("Recommend"):
    if movie_name.strip() == "":
        st.warning("Please enter a movie name")
    else:
        try:
            recs = recommend(movie_name, df, tfidf, tfidf_matrix)
            st.success(f"Top recommendations for: **{movie_name}**")

            for r in recs:
                st.subheader(r["title"])
                st.write(f"**Score:** {r['score']:.4f}")
                st.write(r["overview"])
                st.markdown("---")
        
        except Exception as e:
            st.error(str(e))


# Show sample data
with st.expander("Show dataset sample"):
    st.dataframe(df.head())

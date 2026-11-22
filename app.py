import ast
from collections import Counter
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

# 1. Page Configuration
st.set_page_config(
    page_title="Book Recommendation Engine | C964 Capstone",
    page_icon="📚",
    layout="wide"
)

# 2. Load Data & Model Artifacts (Cached for instant performance)
@st.cache_data
def load_data():
    df = pd.read_csv("data/books_cleaned.csv")
    df["genre_list"] = df["genre_list"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else []
    )
    return df


@st.cache_resource
def load_models():
    tfidf_vectorizer = joblib.load("models/tfidf_vectorizer.joblib")
    tfidf_matrix = joblib.load("models/tfidf_matrix.joblib")
    knn_model = joblib.load("models/knn_model.joblib")
    return tfidf_vectorizer, tfidf_matrix, knn_model


df = load_data()
tfidf_vectorizer, tfidf_matrix, knn_model = load_models()

# 3. Helper Function for Recommendations
def get_recommendations(book_title, num_recs=5):
    idx = df[df["book_title"] == book_title].index[0]
    distances, indices = knn_model.kneighbors(
        tfidf_matrix[idx], n_neighbors=num_recs + 1
    )

    rec_list = []
    for i in range(1, len(indices.flatten())):
        rec_idx = indices.flatten()[i]
        similarity = (1 - distances.flatten()[i]) * 100
        rec_row = df.loc[rec_idx]
        rec_list.append({
            "book_title": rec_row["book_title"],
            "book_authors": rec_row["book_authors"],
            "book_rating": rec_row.get("book_rating", "N/A"),
            "book_pages": rec_row.get("book_pages", "N/A"),
            "genres": ", ".join(rec_row["genre_list"][:4]),
            "book_desc": rec_row["book_desc"],
            "image_url": rec_row.get("image_url", ""),
            "similarity": round(similarity, 1)
        })
    return df.loc[idx], pd.DataFrame(rec_list)


# 4. Main App Layout & Tabs
st.title("📚 Book Recommendation Engine")
st.markdown(
    "Powered by **TF-IDF Vectorization** and **K-Nearest Neighbors (KNN)** across "
    f"**{len(df):,}** curated books."
)

tab1, tab2 = st.tabs(["🔍 Book Recommender", "📊 Data Insights & Model Metrics"])

# --- TAB 1: Interactive Recommender ---
with tab1:
    col_search, col_count = st.columns([3, 1])
    with col_search:
        selected_book = st.selectbox(
            "Select or type a book title you enjoyed:",
            options=df["book_title"].tolist(),
            index=0
        )
    with col_count:
        num_recommendations = st.slider("Number of recommendations:", min_value=5, max_value=20, value=5)

    if st.button("Get Recommendations", type="primary"):
        source_book, recs_df = get_recommendations(selected_book, num_recommendations)

        # Display Selected Book Summary
        st.subheader("Selected Title")
        s_col1, s_col2 = st.columns([1, 5])
        with s_col1:
            if pd.notna(source_book.get("image_url")) and str(source_book["image_url"]).startswith("http"):
                st.image(source_book["image_url"], width=110)
        with s_col2:
            st.markdown(f"**{source_book['book_title']}** by *{source_book['book_authors']}*")
            st.caption(
                f"⭐ Rating: {source_book.get('book_rating', 'N/A')} | "
                f"📄 Pages: {source_book.get('book_pages', 'N/A')} | "
                f"🏷️ Genres: {', '.join(source_book['genre_list'][:5])}"
            )
            st.write(str(source_book["book_desc"])[:350] + "...")

        st.divider()
        st.subheader(f"Top {num_recommendations} Recommended Books")

        # Display Recommended Books
        for idx, row in recs_df.iterrows():
            with st.container(border=True):
                r_col1, r_col2 = st.columns([1, 5])
                with r_col1:
                    if pd.notna(row["image_url"]) and str(row["image_url"]).startswith("http"):
                        st.image(row["image_url"], width=100)
                with r_col2:
                    st.markdown(
                        f"### {idx + 1}. {row['book_title']} "
                        f"(`{row['similarity']}% Match`)"
                    )
                    st.markdown(f"**Author:** {row['book_authors']} | **Rating:** ⭐ {row['book_rating']} | **Pages:** {row['book_pages']}")
                    st.markdown(f"**Genres:** {row['genres']}")
                    with st.expander("Read Synopsis"):
                        st.write(row["book_desc"])

        # Chart 3 (Dynamic): Similarity Score Comparison for Current Query
        st.divider()
        st.subheader("Diagnostic Visualization: Recommendation Similarity Scores")
        fig3, ax3 = plt.subplots(figsize=(8, 3.5))
        sns.barplot(
            data=recs_df,
            x="similarity",
            y="book_title",
            palette="viridis",
            ax=ax3
        )
        ax3.set_xlabel("Cosine Similarity Match (%)")
        ax3.set_ylabel("")
        ax3.set_xlim(0, 100)
        ax3.set_title(f"KNN Cosine Similarity Scores for '{selected_book}'")
        st.pyplot(fig3)

# --- TAB 2: Dataset & Model Visualizations ---
with tab2:
    st.header("Descriptive & Diagnostic Data Visualizations")
    st.markdown(
        "These visualizations summarize the cleaned Goodreads catalog and validate "
        "feature distributions used by the TF-IDF + KNN recommendation pipeline."
    )

    chart_col1, chart_col2 = st.columns(2)

    # Chart 1: Distribution of Book Ratings (Histogram)
    with chart_col1:
        st.subheader("1. Distribution of Average Book Ratings")
        valid_ratings = df[df["book_rating"] > 0]["book_rating"]
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.histplot(valid_ratings, bins=30, kde=True, color="#2b5c8f", ax=ax1)
        ax1.axvline(valid_ratings.mean(), color="red", linestyle="--", label=f"Mean: {valid_ratings.mean():.2f}")
        ax1.set_xlabel("Average Goodreads Rating (1–5)")
        ax1.set_ylabel("Number of Books")
        ax1.legend()
        st.pyplot(fig1)

    # Chart 2: Top 10 Most Common Genres (Bar Chart)
    with chart_col2:
        st.subheader("2. Top 10 Most Frequent Genres in Catalog")
        all_genres = [genre for sublist in df["genre_list"] for genre in sublist]
        top_genres = pd.DataFrame(Counter(all_genres).most_common(10), columns=["Genre", "Count"])
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.barplot(data=top_genres, x="Count", y="Genre", palette="Blues_r", ax=ax2)
        ax2.set_xlabel("Number of Books")
        ax2.set_ylabel("")
        st.pyplot(fig2)

    st.divider()
    st.subheader("Model Evaluation Summary (Task 2 Metrics)")
    m1, m2, m3 = st.columns(3)
    m1.metric("Catalog Size", f"{len(df):,} Books")
    m2.metric("TF-IDF Vocabulary Size", f"{tfidf_matrix.shape[1]:,} Features")
    m3.metric("Algorithm & Metric", "KNN (Cosine Distance)")
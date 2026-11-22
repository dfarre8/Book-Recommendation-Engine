import ast
import os
import random
import joblib
from langdetect import detect, DetectorFactory, LangDetectException
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# 1. Configuration & Paths
RAW_DATA_PATH = "data/books_data.csv"  # Update to match your exact CSV filename
CLEAN_DATA_PATH = "data/books_cleaned.csv"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


def parse_genres(genre_str):
    """Safely converts stringified genre lists like "['Fantasy', 'Fiction']" into a Python list."""
    if pd.isna(genre_str):
        return []
    try:
        parsed = ast.literal_eval(genre_str)
        return parsed if isinstance(parsed, list) else []
    except (ValueError, SyntaxError):
        return [g.strip() for g in str(genre_str).split(",") if g.strip()]

DetectorFactory.seed = 0


def is_english(text):
    """Returns True if langdetect identifies the book description as English."""
    if not isinstance(text, str) or len(text.strip()) < 20:
        return False
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False


def load_and_clean_data(filepath):
    print("Loading raw dataset...")
    df = pd.read_csv(filepath)


    # Expected columns: book_title, book_authors, book_desc, genres, book_rating
    required_cols = ["book_title", "book_authors", "book_desc", "genres"]
    df = df.dropna(subset=required_cols).copy()

    # Filter out books with no ISBN (reduces TF-IDF noise)
    if "book_isbn" in df.columns:
        df = df[df["book_isbn"].notnull()]

    # Remove duplicate book titles and reset index
    df = df.drop_duplicates(subset=["book_title"]).reset_index(drop=True)

    # Parse genres into clean lists and drop books with empty genre lists
    df["genre_list"] = df["genres"].apply(parse_genres)
    df = df[df["genre_list"].map(len) > 0].reset_index(drop=True)

    # Clean numeric columns for display & charts later
    if "book_rating" in df.columns:
        df["book_rating"] = pd.to_numeric(df["book_rating"], errors="coerce").fillna(0.0)
    if "book_pages" in df.columns:
        df["book_pages"] = pd.to_numeric(df["book_pages"], errors="coerce").fillna(0).astype(int)

    # Filter out non-English descriptions using lingua-py
    print(f"Filtering non-English descriptions from {len(df):,} books (this may take 15-30 seconds)...")
    df = df[df["book_desc"].apply(is_english)].reset_index(drop=True)
    print(f"Remaining English books: {len(df):,}")


    # Build the 'metadata soup' (weighting genres & author slightly higher by repeating them)
    def create_soup(row):
        genres_text = " ".join([g.replace(" ", "").lower() for g in row["genre_list"]])
        author_text = str(row["book_authors"]).replace(" ", "").lower()
        desc_text = str(row["book_desc"]).lower()
        return f"{genres_text} {genres_text} {author_text} {desc_text}"

    df["soup"] = df.apply(create_soup, axis=1)

    print(f"Cleaned dataset contains {len(df):,} books.")
    df.to_csv(CLEAN_DATA_PATH, index=False)
    return df


def train_recommender(df):
    print("Vectorizing text metadata with TF-IDF...")
    tfidf = TfidfVectorizer(
        stop_words="english",
        max_features=15000,
        ngram_range=(1, 2),
        min_df=3
    )
    tfidf_matrix = tfidf.fit_transform(df["soup"])

    print("Training K-Nearest Neighbors (KNN) model...")
    # n_neighbors=6 because index 0 will be the queried book itself
    knn = NearestNeighbors(n_neighbors=6, metric="cosine", algorithm="brute")
    knn.fit(tfidf_matrix)

    return tfidf, tfidf_matrix, knn


def evaluate_precision_at_k(df, tfidf_matrix, knn, k=5, sample_size=200):
    """
    Calculates Precision@K based on primary genre overlap.
    A recommendation counts as a 'hit' if it shares at least 1 genre with the query book.
    """
    print(f"\nEvaluating Precision@{k} across {sample_size} random books...")
    random.seed(42)
    sample_indices = random.sample(range(len(df)), min(sample_size, len(df)))

    precisions = []
    for idx in sample_indices:
        query_genres = set(df.loc[idx, "genre_list"])
        if not query_genres:
            continue

        distances, indices = knn.kneighbors(tfidf_matrix[idx], n_neighbors=k + 1)
        rec_indices = indices.flatten()[1:]  # Exclude the query book itself

        hits = 0
        for rec_idx in rec_indices:
            rec_genres = set(df.loc[rec_idx, "genre_list"])
            if query_genres.intersection(rec_genres):
                hits += 1

        precisions.append(hits / k)

    mean_precision = np.mean(precisions) * 100
    print(f"Model Precision@{k}: {mean_precision:.2f}%")
    return mean_precision


def test_sample_recommendation(df, tfidf_matrix, knn, query_title="The Hunger Games"):
    """Sanity check to print recommendations in the PyCharm console."""
    matches = df[df["book_title"].str.contains(query_title, case=False, na=False)]
    if matches.empty:
        print(f"\nSample title '{query_title}' not found in dataset.")
        return

    idx = matches.index[0]
    distances, indices = knn.kneighbors(tfidf_matrix[idx], n_neighbors=6)

    print(f"\nTop 5 Recommendations for '{df.loc[idx, 'book_title']}' by {df.loc[idx, 'book_authors']}:")
    for i in range(1, len(indices.flatten())):
        rec_idx = indices.flatten()[i]
        similarity = (1 - distances.flatten()[i]) * 100
        print(f"  {i}. {df.loc[rec_idx, 'book_title']} — {df.loc[rec_idx, 'book_authors']} (Similarity: {similarity:.1f}%)")


if __name__ == "__main__":
    # Run the full pipeline
    books_df = load_and_clean_data(RAW_DATA_PATH)
    tfidf_vectorizer, tfidf_mat, knn_model = train_recommender(books_df)

    # Evaluate accuracy and run a test query
    precision_score = evaluate_precision_at_k(books_df, tfidf_mat, knn_model, k=5, sample_size=250)
    test_sample_recommendation(books_df, tfidf_mat, knn_model, query_title="The Hunger Games")

    # Save artifacts for Streamlit app.py
    print("\nSaving model artifacts to 'models/' directory...")
    joblib.dump(tfidf_vectorizer, os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
    joblib.dump(tfidf_mat, os.path.join(MODELS_DIR, "tfidf_matrix.joblib"))
    joblib.dump(knn_model, os.path.join(MODELS_DIR, "knn_model.joblib"))
    print("Done! Ready for Streamlit.")
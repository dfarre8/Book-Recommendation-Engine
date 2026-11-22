import ast
from collections import Counter
import os
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 1. Configuration & Output Directory
CLEAN_DATA_PATH = "data/books_cleaned.csv"
MODELS_DIR = "models"
CHARTS_DIR = "charts"
os.makedirs(CHARTS_DIR, exist_ok=True)

# Set consistent visual styling
sns.set_theme(style="whitegrid", context="notebook")


def load_assets():
    print("Loading cleaned dataset and trained models...")
    df = pd.read_csv(CLEAN_DATA_PATH)
    df["genre_list"] = df["genre_list"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else []
    )
    tfidf_matrix = joblib.load(os.path.join(MODELS_DIR, "tfidf_matrix.joblib"))
    knn_model = joblib.load(os.path.join(MODELS_DIR, "knn_model.joblib"))
    return df, tfidf_matrix, knn_model


def chart_1_rating_distribution(df):
    """Descriptive Chart 1: Histogram of Average Book Ratings."""
    valid_ratings = df[df["book_rating"] > 0]["book_rating"]
    mean_val = valid_ratings.mean()
    median_val = valid_ratings.median()

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(valid_ratings, bins=30, kde=True, color="#2b5c8f", edgecolor="white", ax=ax)
    ax.axvline(mean_val, color="#d9534f", linestyle="--", linewidth=1.8, label=f"Mean Rating: {mean_val:.2f}")
    ax.axvline(median_val, color="#f0ad4e", linestyle=":", linewidth=1.8, label=f"Median Rating: {median_val:.2f}")

    ax.set_title("Chart 1: Distribution of Goodreads Average Book Ratings", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Average Rating (1.0 – 5.0 Scale)", fontsize=11)
    ax.set_ylabel("Number of Books", fontsize=11)
    ax.legend()

    out_path = os.path.join(CHARTS_DIR, "chart1_rating_distribution.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")


def chart_2_top_genres(df):
    """Descriptive Chart 2: Horizontal Bar Chart of Top 10 Genres."""
    all_genres = [genre for sublist in df["genre_list"] for genre in sublist]
    top_genres = pd.DataFrame(Counter(all_genres).most_common(10), columns=["Genre", "Book Count"])

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=top_genres,
        x="Book Count",
        y="Genre",
        hue="Genre",
        palette="Blues_r",
        legend=False,
        ax=ax
    )

    # Add exact count labels at the end of each bar
    for container in ax.containers:
        ax.bar_label(container, fmt="{:,.0f}", padding=4, fontsize=9)

    ax.set_xlim(0, top_genres["Book Count"].max() * 1.12)
    ax.set_title("Chart 2: Top 10 Most Frequent Genres in Catalog", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Number of Books Tagged", fontsize=11)
    ax.set_ylabel("")

    out_path = os.path.join(CHARTS_DIR, "chart2_top_genres.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")


def chart_3_sample_similarity(df, tfidf_matrix, knn_model, sample_title="The Hunger Games"):
    """Diagnostic Chart 3: Cosine Similarity Scores for Top 5 KNN Recommendations."""
    matches = df[df["book_title"].str.contains(sample_title, case=False, na=False)]
    idx = matches.index[0] if not matches.empty else 0
    actual_title = df.loc[idx, "book_title"]

    distances, indices = knn_model.kneighbors(tfidf_matrix[idx], n_neighbors=6)

    rec_titles = []
    sim_scores = []
    for i in range(1, len(indices.flatten())):
        rec_idx = indices.flatten()[i]
        title_str = str(df.loc[rec_idx, "book_title"])
        # Truncate overly long titles for clean chart labels
        short_title = title_str[:38] + "..." if len(title_str) > 38 else title_str
        rec_titles.append(short_title)
        sim_scores.append((1 - distances.flatten()[i]) * 100)

    rec_df = pd.DataFrame({"Recommended Book": rec_titles, "Similarity (%)": sim_scores})

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(
        data=rec_df,
        x="Similarity (%)",
        y="Recommended Book",
        hue="Recommended Book",
        palette="viridis",
        legend=False,
        ax=ax
    )

    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", padding=4, fontsize=9)

    ax.set_xlim(0, 100)
    ax.set_title(f"Chart 3: Top 5 KNN Similarity Matches for '{actual_title}'", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Cosine Similarity Score (%)", fontsize=11)
    ax.set_ylabel("")

    out_path = os.path.join(CHARTS_DIR, "chart3_knn_similarity.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    books_df, tfidf_mat, knn = load_assets()
    chart_1_rating_distribution(books_df)
    chart_2_top_genres(books_df)
    chart_3_sample_similarity(books_df, tfidf_mat, knn, sample_title="The Hunger Games")
    print("\nAll three visualizations exported to the 'charts/' folder at 300 DPI!")
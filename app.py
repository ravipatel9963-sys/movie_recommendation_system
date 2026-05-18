import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")


st.set_page_config(
    page_title="Movie Recommendation Engine",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# custom styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.stApp {
    background: #f5f5f5;
    color: #1a1a1a;
}
h1, h2, h3 {
    font-family: 'Playfair Display', serif !important;
    color: #1a1a1a !important;
}
.card {
    background: #fff;
    border: 1px solid #e0e0e0;
    border-radius: 14px;
    padding: 1.4rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.card-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.2rem;
    color: #1a1a1a;
    font-weight: 700;
    margin-bottom: 0.3rem;
}
.card-meta {
    font-size: 0.82rem;
    color: #555577;
    margin-bottom: 0.5rem;
}
.genre-tag {
    display: inline-block;
    background: rgba(229,9,20,0.10);
    color: #e50914;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.75rem;
    margin-right: 4px;
    font-weight: 500;
}
.stButton > button {
    background: linear-gradient(135deg, #e50914, #b20710) !important;
    color: #fff !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 8px !important;
    width: 100% !important;
}
label { color: #333 !important; }
div[data-testid="stSidebar"] {
    background: #f0f0f0 !important;
    border-right: 1px solid #ddd !important;
}
div[data-testid="stSidebar"] * { color: #1a1a1a !important; }
.stTabs [data-baseweb="tab-list"] { background: #fff; border-radius: 8px; }
.stTabs [data-baseweb="tab"] { color: #444 !important; }
.stTabs [aria-selected="true"] {
    color: #e50914 !important;
    border-bottom-color: #e50914 !important;
}
[data-testid="stMetricValue"] { color: #1a1a1a !important; }
[data-testid="stMetricLabel"] { color: #555 !important; }
.stDataFrame { background: #fff; }
.stTextInput input {
    background: #fff !important;
    color: #1a1a1a !important;
    border: 1px solid #ccc !important;
}
.stSelectbox > div > div {
    background: #fff !important;
    color: #1a1a1a !important;
}
</style>
""", unsafe_allow_html=True)


# load and clean the dataset
@st.cache_data
def load_data():
    data = pd.read_csv("cleaned_imdb.csv")
    data = data.dropna(subset=["Title", "Genre", "Rating", "Year", "Director"])
    for col in ["Genre", "Director", "Actors", "Description"]:
        data[col] = data[col].fillna("")
    data["Metascore"] = data["Metascore"].fillna(data["Metascore"].median())
    data["Votes"] = data["Votes"].fillna(0)
    return data.reset_index(drop=True)


# build the TF-IDF + numeric similarity matrices
@st.cache_resource
def build_content_model(_data):
    df = _data.copy()

    # combine text features into one string per movie
    df["combined"] = (
        df["Genre"].str.replace(",", " ") + " " +
        df["Director"].str.replace(" ", "") + " " +
        df["Actors"].str.replace(",", "").str.replace(" ", "") + " " +
        df["Description"]
    )

    tfidf = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 2))
    text_matrix = tfidf.fit_transform(df["combined"])

    scaler = MinMaxScaler()
    num_cols = ["Rating", "Year", "Votes", "Runtime_(Minutes)"]
    num_matrix = scaler.fit_transform(df[num_cols].fillna(0))

    return text_matrix, num_matrix


def get_similar_movies(title, df, text_mat, num_mat,
                       genres=None, year_range=(2006, 2016),
                       min_rating=0.0, director="Any", n=8):
    match = df["Title"].str.lower() == title.lower()
    if not match.any():
        return None

    i = df[match].index[0]
    text_sim = cosine_similarity(text_mat[i], text_mat).flatten()
    num_sim = cosine_similarity(num_mat[i].reshape(1, -1), num_mat).flatten()

    # weighted blend: 70% text, 30% numeric
    combined = 0.70 * text_sim + 0.30 * num_sim
    combined[i] = -1  # exclude the query movie itself

    result = df.copy()
    result["score"] = combined

    if genres:
        result = result[result["Genre"].apply(
            lambda g: any(x.lower() in g.lower() for x in genres)
        )]
    result = result[
        (result["Year"] >= year_range[0]) &
        (result["Year"] <= year_range[1]) &
        (result["Rating"] >= min_rating)
    ]
    if director != "Any":
        result = result[result["Director"] == director]

    return result.sort_values("score", ascending=False).head(n)


def search_by_preference(df, genres=None, year_range=(2006, 2016),
                          min_rating=0.0, director="Any", n=8):
    result = df.copy()

    if genres:
        result = result[result["Genre"].apply(
            lambda g: any(x.lower() in g.lower() for x in genres)
        )]
    result = result[
        (result["Year"] >= year_range[0]) &
        (result["Year"] <= year_range[1]) &
        (result["Rating"] >= min_rating)
    ]
    if director != "Any":
        result = result[result["Director"] == director]

    if result.empty:
        return result

    result = result.copy()
    max_votes = result["Votes"].max()
    result["rank_score"] = (
        0.6 * result["Rating"] / 10 +
        0.4 * (np.log1p(result["Votes"]) / np.log1p(max_votes + 1))
    )
    return result.sort_values("rank_score", ascending=False).head(n)


# SVD + KNN for rating-based recommendations
@st.cache_resource
def build_rating_model(_data):
    df = _data.copy()
    features = ["Rating", "Votes", "Metascore", "Runtime_(Minutes)", "Year"]
    X = df[features].fillna(0).values

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # keep n_components safely below number of features
    n = min(4, X_scaled.shape[1] - 1)
    svd = TruncatedSVD(n_components=n, random_state=42)
    latent = svd.fit_transform(X_scaled)

    knn = NearestNeighbors(n_neighbors=20, metric="cosine", algorithm="brute")
    knn.fit(latent)

    return svd, latent, knn, X_scaled


def svd_recommend(title, df, latent_mat, n=8, genres=None, min_rating=0.0):
    match = df["Title"].str.lower() == title.lower()
    if not match.any():
        return None
    i = df[match].index[0]
    scores = cosine_similarity(latent_mat[i].reshape(1, -1), latent_mat).flatten()
    scores[i] = -1

    result = df.copy()
    result["svd_score"] = scores
    if genres:
        result = result[result["Genre"].apply(
            lambda g: any(x.lower() in g.lower() for x in genres)
        )]
    result = result[result["Rating"] >= min_rating]
    return result.sort_values("svd_score", ascending=False).head(n)


def knn_recommend(title, df, latent_mat, knn_model, n=8, genres=None, min_rating=0.0):
    match = df["Title"].str.lower() == title.lower()
    if not match.any():
        return None
    i = df[match].index[0]
    dists, idxs = knn_model.kneighbors(
        latent_mat[i].reshape(1, -1),
        n_neighbors=min(n * 3 + 1, len(df))
    )
    neighbors = [j for j in idxs[0] if j != i]
    result = df.iloc[neighbors].copy()
    result["knn_score"] = 1 - dists[0][1:len(neighbors) + 1]
    if genres:
        result = result[result["Genre"].apply(
            lambda g: any(x.lower() in g.lower() for x in genres)
        )]
    result = result[result["Rating"] >= min_rating]
    return result.head(n)


def run_accuracy_check(df, latent_mat):
    y = df["Rating"].values
    model = Ridge(alpha=1.0)
    y_hat = cross_val_predict(model, latent_mat, y, cv=5)
    rmse = np.sqrt(mean_squared_error(y, y_hat))
    mae = mean_absolute_error(y, y_hat)
    return rmse, mae, y, y_hat


# Random Forest for revenue prediction
@st.cache_resource
def build_revenue_model(_data):
    df = _data.dropna(subset=["Revenue(Crores)"]).copy()
    feat_cols = ["Rating", "Votes", "Metascore", "Runtime_(Minutes)", "Year"]
    df = df.dropna(subset=feat_cols)

    X = df[feat_cols]
    y = df["Revenue(Crores)"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rf = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = rf.score(X_test, y_test)

    importance = pd.Series(
        rf.feature_importances_, index=feat_cols
    ).sort_values(ascending=False)

    return rf, rmse, mae, r2, importance, X_test, y_test, preds, feat_cols


# initialise everything
df = load_data()
text_mat, num_mat = build_content_model(df)
svd_obj, latent_mat, knn_obj, scaled_feats = build_rating_model(df)
rf_obj, rf_rmse, rf_mae, rf_r2, feat_importance, X_test_rf, y_test_rf, y_pred_rf, RF_COLS = build_revenue_model(df)

GENRES = sorted([
    "Action", "Adventure", "Animation", "Biography", "Comedy",
    "Crime", "Drama", "Fantasy", "Horror", "Mystery", "Sci-Fi", "Thriller"
])
DIRECTORS = ["Any"] + sorted(df["Director"].unique().tolist())
TITLES = sorted(df["Title"].unique().tolist())


# page header with IMDb badge
st.markdown("""
<div style="display:flex; align-items:center; gap:16px; margin-bottom:6px">
  <div style="background:#f5c518; border-radius:6px; padding:4px 10px;
              font-weight:900; font-size:1.4rem; color:#000;
              font-family:Arial Black,sans-serif; line-height:1.2">IMDb</div>
  <h1 style="color:#1a1a1a; font-family:'Playfair Display',serif; margin:0">
    🎬 Movie Recommendation Engine
  </h1>
</div>
""", unsafe_allow_html=True)
st.caption("Content-based filtering using genre, cast, director & numeric features")

# quick stats row
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Movies", len(df))
col2.metric("Directors", df["Director"].nunique())
col3.metric("Genres", len(GENRES))
col4.metric("Year Range", f"{int(df['Year'].min())} - {int(df['Year'].max())}")
st.markdown("---")


# sidebar filters
with st.sidebar:
    st.markdown("### Filters")
    mode = st.radio("Search Mode", ["🎯 Find Similar Movies", "🔍 Browse by Preference"])
    selected_genres = st.multiselect("Genre", GENRES, default=[])
    year_range = st.slider(
        "Year Range",
        int(df["Year"].min()), int(df["Year"].max()), (2008, 2016)
    )
    min_rating = st.slider("Min Rating", 1.0, 9.0, 6.0, step=0.1)
    selected_director = st.selectbox("Director", DIRECTORS)
    num_results = st.slider("Number of Results", 3, 20, 8)
    st.markdown("---")
    st.caption("Model: TF-IDF + Cosine Similarity\nFeatures: Genre, Director, Cast, Description, Rating, Year, Votes")


tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Recommendations",
    "📊 Data Explorer",
    "🧠 How It Works",
    "⭐ Rating-Based ML",
    "💰 Revenue Prediction"
])


# tab 1 - recommendations
with tab1:
    if "Find Similar" in mode:
        st.subheader("Find Similar Movies")
        picked = st.selectbox("Pick a movie you enjoyed:", TITLES)

        if st.button("Get Recommendations 🚀"):
            seed_row = df[df["Title"] == picked].iloc[0]

            with st.expander(f"About: {picked}", expanded=True):
                left, right = st.columns([3, 1])
                with left:
                    st.markdown(
                        f"**Director:** {seed_row['Director']}  |  "
                        f"**Year:** {int(seed_row['Year'])}  |  "
                        f"**Genre:** {seed_row['Genre']}"
                    )
                    st.markdown(f"**Cast:** {seed_row['Actors']}")
                    st.write(seed_row["Description"])
                with right:
                    st.metric("IMDb Rating", f"⭐ {seed_row['Rating']}")
                    st.metric("Runtime", f"{int(seed_row['Runtime_(Minutes)'])} min")
                    st.metric("Votes", f"{int(seed_row['Votes']):,}")

            recs = get_similar_movies(
                picked, df, text_mat, num_mat,
                selected_genres if selected_genres else None,
                year_range, min_rating, selected_director, num_results
            )

            if recs is None or recs.empty:
                st.warning("Nothing matched - try loosening the filters.")
            else:
                st.success(f"Found {len(recs)} recommendations!")
                for rank, (_, row) in enumerate(recs.iterrows(), 1):
                    pct = int(row["score"] * 100)
                    tags = " ".join(
                        f'<span class="genre-tag">{g.strip()}</span>'
                        for g in row["Genre"].split(",")
                    )
                    st.markdown(f"""
                    <div class="card">
                        <div class="card-title">#{rank} &nbsp; {row['Title']}</div>
                        <div class="card-meta">
                            🎬 {row['Director']} &nbsp;|&nbsp;
                            📅 {int(row['Year'])} &nbsp;|&nbsp;
                            ⭐ {row['Rating']} &nbsp;|&nbsp;
                            ⏱️ {int(row['Runtime_(Minutes)'])} min
                        </div>
                        <div style="margin-bottom:0.5rem">{tags}</div>
                        <div style="font-size:0.9rem; color:#333; margin-bottom:0.6rem">
                            {str(row['Description'])[:180]}...
                        </div>
                        <div style="font-size:0.8rem; color:#777">Match: {pct}%</div>
                        <div style="background:#e0e0e0; border-radius:4px; height:6px; margin-top:4px">
                            <div style="background:linear-gradient(90deg,#e50914,#b20710);
                                        width:{pct}%; height:6px; border-radius:4px"></div>
                        </div>
                    </div>""", unsafe_allow_html=True)

    else:
        st.subheader("Browse by Preference")
        st.info("Adjust the filters in the sidebar, then hit Search.")

        if st.button("🔍 Search"):
            results = search_by_preference(
                df,
                selected_genres if selected_genres else None,
                year_range, min_rating, selected_director, num_results
            )
            if results.empty:
                st.warning("No results - try broadening your filters.")
            else:
                st.success(f"{len(results)} movies found!")
                for rank, (_, row) in enumerate(results.iterrows(), 1):
                    tags = " ".join(
                        f'<span class="genre-tag">{g.strip()}</span>'
                        for g in row["Genre"].split(",")
                    )
                    st.markdown(f"""
                    <div class="card">
                        <div class="card-title">#{rank} &nbsp; {row['Title']}</div>
                        <div class="card-meta">
                            🎬 {row['Director']} &nbsp;|&nbsp;
                            📅 {int(row['Year'])} &nbsp;|&nbsp;
                            ⭐ {row['Rating']}
                        </div>
                        <div style="margin-bottom:0.5rem">{tags}</div>
                        <div style="font-size:0.9rem; color:#333">
                            {str(row['Description'])[:200]}...
                        </div>
                    </div>""", unsafe_allow_html=True)


# tab 2 - data explorer
with tab2:
    st.subheader("Dataset Explorer")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x="Rating", nbins=30,
                           title="Rating Distribution",
                           color_discrete_sequence=["#e50914"])
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#333", title_font_color="#1a1a1a"
        )
        fig.update_xaxes(gridcolor="#eee")
        fig.update_yaxes(gridcolor="#eee")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        genre_counts = df["Main_Genre"].value_counts().reset_index()
        genre_counts.columns = ["Genre", "Count"]
        fig = px.bar(genre_counts, x="Count", y="Genre", orientation="h",
                     color="Count", color_continuous_scale=["#f5c518", "#e50914"],
                     title="Movies by Genre")
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#333", title_font_color="#1a1a1a",
            coloraxis_showscale=False, yaxis={"categoryorder": "total ascending"}
        )
        fig.update_xaxes(gridcolor="#eee")
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig = px.box(df, x="Main_Genre", y="Rating", color="Main_Genre",
                     title="Rating Spread by Genre",
                     color_discrete_sequence=px.colors.qualitative.Set3)
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#333", title_font_color="#1a1a1a",
            showlegend=False, xaxis_tickangle=-35
        )
        fig.update_xaxes(gridcolor="#eee")
        fig.update_yaxes(gridcolor="#eee")
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        by_year = df.groupby("Year").size().reset_index(name="Count")
        fig = px.line(by_year, x="Year", y="Count",
                      title="Movies Released Per Year",
                      markers=True, color_discrete_sequence=["#e50914"])
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#333", title_font_color="#1a1a1a"
        )
        fig.update_xaxes(gridcolor="#eee")
        fig.update_yaxes(gridcolor="#eee")
        st.plotly_chart(fig, use_container_width=True)

    fig = px.scatter(df, x="Rating", y="Revenue(Crores)", color="Main_Genre",
                     hover_name="Title", size="Votes",
                     title="Rating vs Revenue  (bubble size = Votes)",
                     color_discrete_sequence=px.colors.qualitative.Pastel)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#333", title_font_color="#1a1a1a"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Raw Data")
    search_q = st.text_input("🔎 Filter by title")
    view_df = df if not search_q else df[df["Title"].str.contains(search_q, case=False, na=False)]
    st.dataframe(
        view_df[["Title", "Genre", "Director", "Year", "Rating",
                 "Runtime_(Minutes)", "Votes", "Revenue(Crores)"]],
        use_container_width=True, height=400
    )


# tab 3 - model explanation
with tab3:
    st.subheader("How the Recommendation Model Works")

    st.markdown("""
The recommender uses **Content-Based Filtering** - it finds movies that are
similar to the one you pick based on their metadata.

**Two similarity signals are blended together:**

| Signal | Weight | What it looks at |
|--------|--------|-----------------|
| TF-IDF Text Similarity | 70% | Genre, Director, Cast, Description (5000 features, 1-2 word grams) |
| Numeric Similarity | 30% | Rating, Year, Votes, Runtime (MinMax scaled, cosine distance) |

The final score for any candidate movie is:

```
score = 0.70 x text_similarity + 0.30 x numeric_similarity
```

The main advantage of content-based filtering is that it does not need user
history to work - it only needs the movie's own attributes.
""")

    st.markdown("#### Top Directors by Average Rating (min. 2 films)")
    dir_stats = (
        df.groupby("Director")
        .agg(avg_rating=("Rating", "mean"), film_count=("Title", "count"))
        .reset_index()
    )
    dir_stats = dir_stats[dir_stats["film_count"] >= 2].sort_values(
        "avg_rating", ascending=False
    ).head(15)

    fig = px.bar(dir_stats, x="avg_rating", y="Director", orientation="h",
                 color="avg_rating", color_continuous_scale=["#f5c518", "#e50914"],
                 hover_data=["film_count"], title="Top 15 Directors")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#333", title_font_color="#1a1a1a",
        coloraxis_showscale=False, yaxis={"categoryorder": "total ascending"}
    )
    fig.update_xaxes(range=[7, 9], gridcolor="#eee")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Average Rating per Genre")
    genre_avg = (
        df.groupby("Main_Genre")["Rating"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    fig = px.bar(genre_avg, x="Main_Genre", y="Rating",
                 color="Rating", color_continuous_scale=["#f5c518", "#e50914"],
                 title="Genre vs Avg Rating")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#333", title_font_color="#1a1a1a",
        coloraxis_showscale=False
    )
    fig.update_yaxes(range=[5, 8], gridcolor="#eee")
    st.plotly_chart(fig, use_container_width=True)


# tab 4 - SVD / KNN rating-based model
with tab4:
    st.subheader("⭐ Rating-Based Recommendations")
    st.markdown(
        "These models recommend movies based on numeric signals - "
        "**Rating, Votes, Metascore, Runtime and Year** - rather than text content."
    )

    left_exp, right_exp = st.columns(2)
    with left_exp:
        st.markdown("""
**SVD (Singular Value Decomposition)**
Breaks down the feature matrix into latent factors (hidden patterns).
Good at surfacing non-obvious connections between movies - e.g. two films
that share similar popularity curves even across different genres.
Uses cosine similarity on the compressed latent vectors.
        """)
    with right_exp:
        st.markdown("""
**KNN (K-Nearest Neighbours)**
Straightforward distance-based approach - finds the K movies whose
latent-space representation is closest to the query movie.
Very interpretable; each recommendation is a direct neighbour.
        """)

    st.markdown("---")
    st.markdown("### Accuracy Check")
    st.caption("Runs a Ridge regression on top of SVD latent features using 5-fold CV to predict IMDb ratings.")

    if st.button("Run Accuracy Check 📊"):
        with st.spinner("Running cross-validation..."):
            rmse_val, mae_val, actual_ratings, pred_ratings = run_accuracy_check(df, latent_mat)

        a1, a2, a3 = st.columns(3)
        a1.metric("RMSE", f"{rmse_val:.4f}", help="Lower is better")
        a2.metric("MAE", f"{mae_val:.4f}", help="Mean absolute error")
        a3.metric("Rating Scale", "1 - 10")

        fig = px.scatter(
            x=actual_ratings, y=pred_ratings,
            labels={"x": "Actual Rating", "y": "Predicted Rating"},
            title="Actual vs Predicted (5-fold CV)",
            color_discrete_sequence=["#e50914"], opacity=0.5
        )
        fig.add_shape(
            type="line",
            x0=actual_ratings.min(), y0=actual_ratings.min(),
            x1=actual_ratings.max(), y1=actual_ratings.max(),
            line=dict(color="#f5c518", dash="dash", width=2)
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#333", title_font_color="#1a1a1a"
        )
        fig.update_xaxes(gridcolor="#eee")
        fig.update_yaxes(gridcolor="#eee")
        st.plotly_chart(fig, use_container_width=True)

        err = np.abs(actual_ratings - pred_ratings)
        fig2 = px.histogram(x=err, nbins=30,
                            title="Error Distribution",
                            labels={"x": "|Actual - Predicted|"},
                            color_discrete_sequence=["#f5c518"])
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#333", title_font_color="#1a1a1a"
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.markdown("### Get Recommendations")

    rb1, rb2, rb3 = st.columns([3, 1, 1])
    with rb1:
        rb_title = st.selectbox("Choose a movie:", TITLES, key="rb_title")
    with rb2:
        rb_model = st.selectbox("Model", ["SVD", "KNN"], key="rb_model")
    with rb3:
        rb_n = st.slider("Results", 3, 15, 8, key="rb_n")

    rb_genres = st.multiselect("Genre filter (optional)", GENRES, key="rb_genre")
    rb_min_rating = st.slider("Min Rating", 1.0, 9.0, 5.0, 0.1, key="rb_rating")

    if st.button("⭐ Recommend"):
        seed = df[df["Title"] == rb_title].iloc[0]
        with st.expander(f"Seed: {rb_title}", expanded=True):
            st.markdown(
                f"**Director:** {seed['Director']}  |  "
                f"**Year:** {int(seed['Year'])}  |  "
                f"**Genre:** {seed['Genre']}"
            )
            st.markdown(
                f"**Rating:** ⭐ {seed['Rating']}  |  "
                f"**Votes:** {int(seed['Votes']):,}  |  "
                f"**Metascore:** {seed['Metascore']}"
            )

        gf = rb_genres if rb_genres else None
        if rb_model == "SVD":
            recs = svd_recommend(rb_title, df, latent_mat, rb_n, gf, rb_min_rating)
            score_key = "svd_score"
            label = "SVD Score"
        else:
            recs = knn_recommend(rb_title, df, latent_mat, knn_obj, rb_n, gf, rb_min_rating)
            score_key = "knn_score"
            label = "KNN Similarity"

        if recs is None or recs.empty:
            st.warning("No results - try loosening filters.")
        else:
            st.success(f"{len(recs)} recommendations via {rb_model}")
            for rank, (_, row) in enumerate(recs.iterrows(), 1):
                pct = int(row[score_key] * 100)
                tags = " ".join(
                    f'<span class="genre-tag">{g.strip()}</span>'
                    for g in row["Genre"].split(",")
                )
                st.markdown(f"""
                <div class="card">
                    <div class="card-title">#{rank} &nbsp; {row['Title']}</div>
                    <div class="card-meta">
                        🎬 {row['Director']} &nbsp;|&nbsp;
                        📅 {int(row['Year'])} &nbsp;|&nbsp;
                        ⭐ {row['Rating']} &nbsp;|&nbsp;
                        ⏱️ {int(row['Runtime_(Minutes)'])} min &nbsp;|&nbsp;
                        🗳️ {int(row['Votes']):,} votes
                    </div>
                    <div style="margin-bottom:0.5rem">{tags}</div>
                    <div style="font-size:0.9rem; color:#333; margin-bottom:0.6rem">
                        {str(row['Description'])[:180]}...
                    </div>
                    <div style="font-size:0.8rem; color:#777">{label}: {pct}%</div>
                    <div style="background:#e0e0e0; border-radius:4px; height:6px; margin-top:4px">
                        <div style="background:linear-gradient(90deg,#f5c518,#e50914);
                                    width:{pct}%; height:6px; border-radius:4px"></div>
                    </div>
                </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Feature Correlation Charts")
    fa1, fa2 = st.columns(2)
    with fa1:
        fig = px.scatter(df, x="Votes", y="Rating", color="Main_Genre",
                         hover_name="Title", log_x=True,
                         title="Votes vs Rating",
                         color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#333", title_font_color="#1a1a1a"
        )
        fig.update_xaxes(gridcolor="#eee")
        fig.update_yaxes(gridcolor="#eee")
        st.plotly_chart(fig, use_container_width=True)

    with fa2:
        fig = px.scatter(df, x="Metascore", y="Rating", color="Main_Genre",
                         hover_name="Title", trendline="ols",
                         title="Metascore vs IMDb Rating",
                         color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#333", title_font_color="#1a1a1a"
        )
        fig.update_xaxes(gridcolor="#eee")
        fig.update_yaxes(gridcolor="#eee")
        st.plotly_chart(fig, use_container_width=True)


# tab 5 - revenue prediction with random forest
with tab5:
    st.subheader("💰 Box Office Revenue Prediction")
    st.markdown(
        "Uses a **Random Forest Regressor** (200 trees, max depth 8) trained on "
        "Rating, Votes, Metascore, Runtime and Year to predict revenue in crores."
    )

    st.markdown("### Model Performance")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("RMSE", f"{rf_rmse:.1f} Cr")
    m2.metric("MAE", f"{rf_mae:.1f} Cr")
    m3.metric("R2 Score", f"{rf_r2:.3f}")
    m4.metric("Trees", "200")

    p1, p2 = st.columns(2)
    with p1:
        fig = px.scatter(
            x=y_test_rf, y=y_pred_rf,
            labels={"x": "Actual Revenue (Cr)", "y": "Predicted Revenue (Cr)"},
            title="Actual vs Predicted Revenue",
            color_discrete_sequence=["#e50914"], opacity=0.6
        )
        fig.add_shape(
            type="line",
            x0=float(y_test_rf.min()), y0=float(y_test_rf.min()),
            x1=float(y_test_rf.max()), y1=float(y_test_rf.max()),
            line=dict(color="#f5c518", dash="dash", width=2)
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#333", title_font_color="#1a1a1a"
        )
        fig.update_xaxes(gridcolor="#eee")
        fig.update_yaxes(gridcolor="#eee")
        st.plotly_chart(fig, use_container_width=True)

    with p2:
        fig = px.bar(
            x=feat_importance.values, y=feat_importance.index,
            orientation="h",
            labels={"x": "Importance", "y": "Feature"},
            title="Feature Importance",
            color=feat_importance.values,
            color_continuous_scale=["#f5c518", "#e50914"]
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#333", title_font_color="#1a1a1a",
            coloraxis_showscale=False,
            yaxis={"categoryorder": "total ascending"}
        )
        fig.update_xaxes(gridcolor="#eee")
        st.plotly_chart(fig, use_container_width=True)

    err = np.abs(y_test_rf - y_pred_rf)
    fig = px.histogram(x=err, nbins=30,
                       title="Prediction Error Distribution",
                       labels={"x": "Error (Crores)"},
                       color_discrete_sequence=["#f5c518"])
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#333", title_font_color="#1a1a1a"
    )
    fig.update_xaxes(gridcolor="#eee")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### Predict Revenue for a Movie")
    pred_pick = st.selectbox("Select a movie:", TITLES, key="rf_pick")

    if st.button("💰 Predict"):
        row = df[df["Title"] == pred_pick].iloc[0]
        meta = row["Metascore"] if pd.notna(row["Metascore"]) else df["Metascore"].median()
        inp = pd.DataFrame([{
            "Rating": row["Rating"],
            "Votes": row["Votes"],
            "Metascore": meta,
            "Runtime_(Minutes)": row["Runtime_(Minutes)"],
            "Year": row["Year"]
        }])
        predicted = rf_obj.predict(inp)[0]
        actual = row["Revenue(Crores)"] if pd.notna(row["Revenue(Crores)"]) else None

        r1, r2, r3 = st.columns(3)
        r1.metric("Predicted Revenue", f"Rs. {predicted:,.1f} Cr")
        if actual:
            r2.metric("Actual Revenue", f"Rs. {actual:,.1f} Cr")
            diff = predicted - actual
            r3.metric(
                "Difference",
                f"Rs. {abs(diff):,.1f} Cr",
                delta=f"{'Over' if diff > 0 else 'Under'} by {abs(diff):.1f} Cr",
                delta_color="inverse"
            )
        else:
            r2.metric("Actual Revenue", "Not available")

        st.markdown(f"""
        <div class="card">
            <div class="card-title">{pred_pick}</div>
            <div class="card-meta">
                ⭐ {row['Rating']} &nbsp;|&nbsp;
                🗳️ {int(row['Votes']):,} votes &nbsp;|&nbsp;
                Metascore: {int(meta)} &nbsp;|&nbsp;
                ⏱️ {int(row['Runtime_(Minutes)'])} min &nbsp;|&nbsp;
                📅 {int(row['Year'])}
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Custom Revenue Estimator")
    st.caption("Plug in your own values to see what a hypothetical movie might earn.")

    ci1, ci2, ci3 = st.columns(3)
    with ci1:
        c_rating = st.slider("IMDb Rating", 1.0, 10.0, 7.5, 0.1, key="c_rating")
        c_year = st.slider("Year", 2006, 2025, 2015, key="c_year")
    with ci2:
        c_votes = st.number_input("Votes", min_value=1000, max_value=2000000,
                                  value=150000, step=10000, key="c_votes")
        c_meta = st.slider("Metascore", 1, 100, 65, key="c_meta")
    with ci3:
        c_runtime = st.slider("Runtime (min)", 60, 240, 120, key="c_runtime")

    if st.button("🚀 Estimate Revenue"):
        custom = pd.DataFrame([{
            "Rating": c_rating,
            "Votes": c_votes,
            "Metascore": c_meta,
            "Runtime_(Minutes)": c_runtime,
            "Year": c_year
        }])
        est = rf_obj.predict(custom)[0]
        st.success(f"Estimated Revenue: Rs. {est:,.1f} Crores")

        avg = df[RF_COLS].mean()
        compare = pd.DataFrame({
            "Feature": RF_COLS,
            "Your Input": [c_rating, c_votes, c_meta, c_runtime, c_year],
            "Dataset Average": avg.values.round(1)
        })
        fig = px.bar(compare, x="Feature", y=["Your Input", "Dataset Average"],
                     barmode="group", title="Your Input vs Dataset Average",
                     color_discrete_sequence=["#e50914", "#f5c518"])
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#333", title_font_color="#1a1a1a"
        )
        fig.update_xaxes(gridcolor="#eee")
        fig.update_yaxes(gridcolor="#eee")
        st.plotly_chart(fig, use_container_width=True)

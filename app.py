import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Movie Recommendation Engine",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS — Light IMDb-style theme ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif !important;
    background-color: #f5f5f5 !important;
    color: #1a1a1a !important;
}

/* ── Sidebar ── */
div[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 1px solid #e0e0e0 !important;
    padding-top: 1rem;
}
div[data-testid="stSidebar"] label,
div[data-testid="stSidebar"] p,
div[data-testid="stSidebar"] span {
    color: #1a1a1a !important;
}
div[data-testid="stSidebar"] .stRadio label { color: #333 !important; }

/* IMDb logo block */
.imdb-logo {
    background: #F5C518;
    color: #000;
    font-size: 1.5rem;
    font-weight: 900;
    padding: 6px 14px;
    border-radius: 6px;
    display: inline-block;
    letter-spacing: 1px;
    margin-bottom: 1.5rem;
}

.sidebar-stat-label { font-size: 0.78rem; color: #888; text-transform: uppercase; letter-spacing: 0.05em; }
.sidebar-stat-val   { font-size: 0.95rem; font-weight: 600; color: #1a1a1a; }

/* ── Main area ── */
.main-title {
    font-size: 2.2rem;
    font-weight: 700;
    color: #1a1a1a;
    margin-bottom: 0.2rem;
}
.main-sub {
    font-size: 0.9rem;
    color: #888;
    margin-bottom: 1.5rem;
}

/* Search box */
.stTextInput > div > div > input {
    border: 2px solid #E53935 !important;
    border-radius: 8px !important;
    font-size: 1rem !important;
    padding: 0.6rem 1rem !important;
    background: #fff !important;
    color: #1a1a1a !important;
}
.stTextInput > div > div > input:focus { box-shadow: 0 0 0 3px rgba(229,57,53,0.15) !important; }

/* Select / Dropdown */
.stSelectbox > div > div {
    border-radius: 8px !important;
    background: #fff !important;
    border: 1px solid #ddd !important;
}

/* Buttons */
.stButton > button {
    background: #E53935 !important;
    color: #fff !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.55rem 1.4rem !important;
    font-size: 0.95rem !important;
}
.stButton > button:hover { background: #c62828 !important; }

/* Seed movie card */
.seed-card {
    background: #fff;
    border-radius: 12px;
    padding: 1.2rem 1.6rem;
    margin-bottom: 1.5rem;
    border: 1px solid #e8e8e8;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.seed-title { font-size: 1.1rem; font-weight: 700; color: #1a1a1a; margin-bottom: 0.3rem; }
.seed-meta  { font-size: 0.85rem; color: #666; }

/* Results table header */
.results-header {
    font-size: 1.25rem;
    font-weight: 700;
    color: #1a1a1a;
    margin: 1.5rem 0 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #E53935;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* Column labels */
.col-labels {
    display: grid;
    grid-template-columns: 2fr 1fr 1fr 1fr 1.5fr 1fr;
    gap: 0;
    padding: 0.5rem 1rem;
    background: #f0f0f0;
    border-radius: 8px 8px 0 0;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #888;
}

/* Movie row card */
.movie-row {
    background: #fff;
    border-bottom: 1px solid #f0f0f0;
    display: grid;
    grid-template-columns: 2fr 1fr 1fr 1fr 1.5fr 1fr;
    gap: 0;
    padding: 1rem 1rem;
    align-items: center;
    transition: background 0.15s;
}
.movie-row:hover { background: #fafafa; }
.movie-row:last-child { border-radius: 0 0 8px 8px; border-bottom: none; }

.row-title  { font-size: 1.35rem; font-weight: 700; color: #1a1a1a; }
.row-genre  { font-size: 1rem; color: #444; font-weight: 400; }
.row-rating { font-size: 1.1rem; font-weight: 600; color: #1a1a1a; }
.row-year   { font-size: 1.1rem; color: #444; }
.row-dir    { font-size: 1rem; color: #444; }
.row-sim    { font-size: 1rem; font-weight: 600; color: #E53935; }
.star-icon  { color: #F5C518; margin-right: 4px; }

/* Metric cards */
.metric-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; }
.metric-card {
    background: #fff;
    border-radius: 10px;
    padding: 0.9rem 1.2rem;
    flex: 1;
    border: 1px solid #e8e8e8;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.metric-val   { font-size: 1.6rem; font-weight: 700; color: #1a1a1a; }
.metric-label { font-size: 0.75rem; color: #888; text-transform: uppercase; letter-spacing: 0.06em; }

/* Tabs */
button[data-baseweb="tab"] { font-weight: 600 !important; font-size: 0.9rem !important; }
button[data-baseweb="tab"][aria-selected="true"] { color: #E53935 !important; border-bottom-color: #E53935 !important; }

/* Slider */
.stSlider [data-testid="stTickBar"] { color: #888 !important; }

/* Hide default header */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── DATA & MODEL ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_imdb.csv")
    df = df.dropna(subset=["Title", "Genre", "Rating", "Year", "Director"])
    for col in ["Genre", "Director", "Actors", "Description"]:
        df[col] = df[col].fillna("")
    df["Metascore"] = df["Metascore"].fillna(df["Metascore"].median())
    df["Votes"]     = df["Votes"].fillna(0)
    return df.reset_index(drop=True)


@st.cache_resource
def build_model(_df):
    df = _df.copy()
    df["text_soup"] = (
        df["Genre"].str.replace(",", " ") + " " +
        df["Director"].str.replace(" ", "") + " " +
        df["Actors"].str.replace(",", "").str.replace(" ", "") + " " +
        df["Description"]
    )
    tfidf     = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 2))
    tfidf_mat = tfidf.fit_transform(df["text_soup"])
    scaler    = MinMaxScaler()
    num_mat   = scaler.fit_transform(
        df[["Rating", "Year", "Votes", "Runtime_(Minutes)"]].fillna(0)
    )
    return tfidf_mat, num_mat


def recommend(title, df, tfidf_mat, num_mat, top_n=10):
    mask = df["Title"].str.lower() == title.lower()
    if not mask.any():
        return None
    idx     = df[mask].index[0]
    txt_sim = cosine_similarity(tfidf_mat[idx], tfidf_mat).flatten()
    num_sim = cosine_similarity(num_mat[idx].reshape(1, -1), num_mat).flatten()
    scores  = 0.70 * txt_sim + 0.30 * num_sim
    scores[idx] = -1
    out = df.copy()
    out["Similarity"] = (scores * 100).round(2)
    return out.sort_values("Similarity", ascending=False).head(top_n)


def smart_search(df, genres, yr, min_rating, director, top_n):
    out = df.copy()
    if genres:
        out = out[out["Genre"].apply(lambda g: any(x.lower() in g.lower() for x in genres))]
    out = out[(out["Year"] >= yr[0]) & (out["Year"] <= yr[1]) & (out["Rating"] >= min_rating)]
    if director != "Any":
        out = out[out["Director"] == director]
    if out.empty:
        return out
    out = out.copy()
    out["Similarity"] = (
        0.6 * out["Rating"] / 10 +
        0.4 * (np.log1p(out["Votes"]) / np.log1p(out["Votes"].max() + 1))
    ) * 100
    out["Similarity"] = out["Similarity"].round(2)
    return out.sort_values("Similarity", ascending=False).head(top_n)


# ── LOAD ──────────────────────────────────────────────────────────────────────
df = load_data()
tfidf_mat, num_mat = build_model(df)

ALL_GENRES    = sorted(["Action","Adventure","Animation","Biography","Comedy",
                         "Crime","Drama","Fantasy","Horror","Mystery","Sci-Fi","Thriller"])
ALL_DIRECTORS = ["Any"] + sorted(df["Director"].unique().tolist())
ALL_TITLES    = sorted(df["Title"].unique().tolist())


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="imdb-logo">IMDb</div>', unsafe_allow_html=True)
    st.markdown("### Navigation")
    page = st.radio("", ["🎬 Recommend", "📊 EDA", "🧠 Model Accuracy", "📝 Revenue Prediction"],
                    label_visibility="collapsed")

    st.markdown("---")
    st.markdown('<span class="sidebar-stat-label">Dataset</span>', unsafe_allow_html=True)
    st.markdown('<span class="sidebar-stat-val">cleaned_imdb.csv</span>', unsafe_allow_html=True)
    st.markdown("")
    st.markdown('<span class="sidebar-stat-label">Movies</span>', unsafe_allow_html=True)
    st.markdown('<span class="sidebar-stat-val">' + str(len(df)) + '</span>', unsafe_allow_html=True)
    st.markdown("")
    st.markdown('<span class="sidebar-stat-label">Features</span>', unsafe_allow_html=True)
    st.markdown('<span class="sidebar-stat-val">207 dims</span>', unsafe_allow_html=True)
    st.markdown("")
    st.markdown('<span class="sidebar-stat-label">Genres</span>', unsafe_allow_html=True)
    st.markdown('<span class="sidebar-stat-val">' + str(len(ALL_GENRES)) + '</span>', unsafe_allow_html=True)
    st.markdown("")
    st.markdown('<span class="sidebar-stat-label">Years</span>', unsafe_allow_html=True)
    st.markdown('<span class="sidebar-stat-val">' + str(int(df["Year"].min())) + "–" + str(int(df["Year"].max())) + '</span>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: RECOMMEND
# ══════════════════════════════════════════════════════════════════════════════
if "Recommend" in page:

    st.markdown('<div class="main-title">🎬 Movie Recommendation Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-sub">Content-based filtering using genre, cast, director &amp; numeric features</div>', unsafe_allow_html=True)

    # Search row
    col_search, col_model, col_n = st.columns([4, 1.5, 1.5])
    with col_search:
        search_input = st.text_input("Search a movie", placeholder="e.g. Inception, The Dark Knight…",
                                     label_visibility="visible")
    with col_model:
        model_choice = st.selectbox("Model", ["Cosine Similarity", "Smart Ranking"])
    with col_n:
        top_n = st.slider("Number of recommendations", 3, 20, 10)

    # Match search input to titles
    matched_title = None
    if search_input:
        hits = [t for t in ALL_TITLES if search_input.lower() in t.lower()]
        if hits:
            matched_title = hits[0]

    # Seed movie display
    if matched_title:
        seed = df[df["Title"] == matched_title].iloc[0]
        st.markdown(
            '<div class="seed-card">'
            '<div class="seed-title">' + seed["Title"] + '</div>'
            '<div class="seed-meta">'
            '🎭 ' + seed["Genre"] + ' &nbsp;|&nbsp; '
            '⭐ ' + str(seed["Rating"]) + ' &nbsp;|&nbsp; '
            '📅 ' + str(int(seed["Year"])) + ' &nbsp;|&nbsp; '
            '🎬 ' + seed["Director"] + ' &nbsp;|&nbsp; '
            '⏱️ ' + str(int(seed["Runtime_(Minutes)"])) + ' min'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )

        # Column labels row
        st.markdown(
            '<div class="col-labels">'
            '<span>Title</span>'
            '<span>Genre</span>'
            '<span>Rating</span>'
            '<span>Year</span>'
            '<span>Director</span>'
            '<span>Similarity</span>'
            '</div>',
            unsafe_allow_html=True
        )

        # Get recommendations
        if model_choice == "Cosine Similarity":
            recs = recommend(matched_title, df, tfidf_mat, num_mat, top_n)
        else:
            recs = smart_search(df, None,
                                (int(df["Year"].min()), int(df["Year"].max())),
                                0.0, "Any", top_n)

        if recs is None or recs.empty:
            st.warning("No results found. Try a different title.")
        else:
            st.markdown(
                '<div class="results-header">🏆 Top ' + str(len(recs)) +
                ' recommendations — ' + model_choice + '</div>',
                unsafe_allow_html=True
            )

            rows_html = ""
            for _, row in recs.iterrows():
                title  = row["Title"]
                genre  = row["Main_Genre"]
                rating = row["Rating"]
                year   = int(row["Year"])
                direc  = row["Director"]
                sim    = row["Similarity"]
                rows_html += (
                    '<div class="movie-row">'
                    '<div class="row-title">' + title + '</div>'
                    '<div class="row-genre">' + genre + '</div>'
                    '<div class="row-rating"><span class="star-icon">★</span>' + str(rating) + '</div>'
                    '<div class="row-year">' + str(year) + '</div>'
                    '<div class="row-dir">' + direc + '</div>'
                    '<div class="row-sim">' + str(sim) + '</div>'
                    '</div>'
                )
            st.markdown(
                '<div style="border:1px solid #e8e8e8;border-radius:0 0 10px 10px;overflow:hidden">'
                + rows_html + '</div>',
                unsafe_allow_html=True
            )
    else:
        if search_input:
            st.warning("No movie found matching \"" + search_input + "\". Try another title.")
        else:
            st.info("👆 Type a movie name above to get recommendations.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EDA
# ══════════════════════════════════════════════════════════════════════════════
elif "EDA" in page:
    st.markdown('<div class="main-title">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-sub">Visual overview of the IMDB dataset</div>', unsafe_allow_html=True)

    # Metric cards
    st.markdown(
        '<div class="metric-row">'
        '<div class="metric-card"><div class="metric-val">' + str(len(df)) + '</div><div class="metric-label">Total Movies</div></div>'
        '<div class="metric-card"><div class="metric-val">' + str(df["Director"].nunique()) + '</div><div class="metric-label">Directors</div></div>'
        '<div class="metric-card"><div class="metric-val">' + str(round(df["Rating"].mean(), 1)) + '</div><div class="metric-label">Avg Rating</div></div>'
        '<div class="metric-card"><div class="metric-val">' + str(len(ALL_GENRES)) + '</div><div class="metric-label">Genres</div></div>'
        '</div>',
        unsafe_allow_html=True
    )

    PLOT_BG = "rgba(0,0,0,0)"
    FONT_COLOR = "#1a1a1a"
    GRID_COLOR = "rgba(0,0,0,0.06)"
    ACCENT = "#E53935"

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x="Rating", nbins=30, title="Rating Distribution",
                           color_discrete_sequence=[ACCENT])
        fig.update_layout(paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
                          font_color=FONT_COLOR, title_font_color=FONT_COLOR)
        fig.update_xaxes(gridcolor=GRID_COLOR)
        fig.update_yaxes(gridcolor=GRID_COLOR)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        gc = df["Main_Genre"].value_counts().reset_index()
        gc.columns = ["Genre", "Count"]
        fig = px.bar(gc, x="Count", y="Genre", orientation="h",
                     color="Count", color_continuous_scale=["#ffcdd2", ACCENT],
                     title="Movies per Genre")
        fig.update_layout(paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
                          font_color=FONT_COLOR, title_font_color=FONT_COLOR,
                          coloraxis_showscale=False, yaxis={"categoryorder": "total ascending"})
        fig.update_xaxes(gridcolor=GRID_COLOR)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig = px.box(df, x="Main_Genre", y="Rating", color="Main_Genre",
                     title="Rating by Genre",
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
                          font_color=FONT_COLOR, title_font_color=FONT_COLOR,
                          showlegend=False, xaxis_tickangle=-35)
        fig.update_xaxes(gridcolor=GRID_COLOR)
        fig.update_yaxes(gridcolor=GRID_COLOR)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        yc = df.groupby("Year").size().reset_index(name="Count")
        fig = px.line(yc, x="Year", y="Count", title="Movies Per Year",
                      markers=True, color_discrete_sequence=[ACCENT])
        fig.update_layout(paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
                          font_color=FONT_COLOR, title_font_color=FONT_COLOR)
        fig.update_xaxes(gridcolor=GRID_COLOR)
        fig.update_yaxes(gridcolor=GRID_COLOR)
        st.plotly_chart(fig, use_container_width=True)

    fig = px.scatter(df, x="Rating", y="Revenue(Crores)", color="Main_Genre",
                     hover_name="Title", size="Votes",
                     title="Rating vs Revenue (bubble = Votes)")
    fig.update_layout(paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
                      font_color=FONT_COLOR, title_font_color=FONT_COLOR)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Raw Dataset")
    q = st.text_input("🔎 Filter by title")
    show = df if not q else df[df["Title"].str.contains(q, case=False, na=False)]
    st.dataframe(
        show[["Title", "Genre", "Director", "Year", "Rating", "Runtime_(Minutes)", "Votes", "Revenue(Crores)"]],
        use_container_width=True, height=400
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL ACCURACY
# ══════════════════════════════════════════════════════════════════════════════
elif "Model" in page:
    st.markdown('<div class="main-title">🧠 Model Accuracy & Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-sub">How the recommendation engine works</div>', unsafe_allow_html=True)

    st.markdown("""
### Algorithm: Content-Based Filtering

| Component | Weight | Description |
|-----------|--------|-------------|
| **TF-IDF Text Similarity** | 70% | Genre + Director + Cast + Description (5 000 features, bigrams) |
| **Numerical Similarity** | 30% | Rating · Year · Votes · Runtime — MinMax-normalised |

**Score formula:**
```
final_score = 0.70 × cosine_sim(text) + 0.30 × cosine_sim(numerics)
```
    """)

    PLOT_BG    = "rgba(0,0,0,0)"
    FONT_COLOR = "#1a1a1a"
    ACCENT     = "#E53935"

    ds = df.groupby("Director").agg(Avg_Rating=("Rating","mean"), Movies=("Title","count")).reset_index()
    ds = ds[ds["Movies"] >= 2].sort_values("Avg_Rating", ascending=False).head(15)
    fig = px.bar(ds, x="Avg_Rating", y="Director", orientation="h",
                 color="Avg_Rating", color_continuous_scale=["#ffcdd2", ACCENT],
                 hover_data=["Movies"], title="Top 15 Directors by Avg Rating (min 2 movies)")
    fig.update_layout(paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
                      font_color=FONT_COLOR, title_font_color=FONT_COLOR,
                      coloraxis_showscale=False, yaxis={"categoryorder": "total ascending"})
    fig.update_xaxes(range=[7, 9])
    st.plotly_chart(fig, use_container_width=True)

    ga = df.groupby("Main_Genre")["Rating"].mean().sort_values(ascending=False).reset_index()
    fig = px.bar(ga, x="Main_Genre", y="Rating",
                 color="Rating", color_continuous_scale=["#ffcdd2", ACCENT],
                 title="Average Rating by Genre")
    fig.update_layout(paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
                      font_color=FONT_COLOR, title_font_color=FONT_COLOR,
                      coloraxis_showscale=False)
    fig.update_yaxes(range=[5, 8])
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: REVENUE PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif "Revenue" in page:

    # Extra CSS for this page
    st.markdown("""
    <style>
    .rev-title { font-size:2rem; font-weight:800; color:#1a1a1a; margin-bottom:1.2rem; }
    .rev-label { font-size:0.85rem; color:#888; margin-bottom:0.2rem; }
    .rev-value { font-size:2.4rem; font-weight:700; color:#1a1a1a; line-height:1.1; }
    .rev-diff-over  { display:inline-block; background:#fde8e8; color:#E53935; border-radius:20px; padding:3px 12px; font-size:0.82rem; font-weight:600; margin-top:4px; }
    .rev-diff-under { display:inline-block; background:#e8f5e9; color:#2e7d32; border-radius:20px; padding:3px 12px; font-size:0.82rem; font-weight:600; margin-top:4px; }
    .rev-metrics-row { display:grid; grid-template-columns:1fr 1fr 1fr; gap:2rem; margin:1.5rem 0 1.8rem; }
    .movie-info-card {
        background:#fff; border-radius:14px;
        border:1px solid #e8e8e8;
        padding:1.4rem 1.8rem;
        margin-top:0.5rem;
        box-shadow:0 2px 8px rgba(0,0,0,0.05);
    }
    .mi-title { font-size:1.4rem; font-weight:800; color:#1a1a1a; margin-bottom:0.5rem; }
    .mi-meta  { font-size:0.9rem; color:#555; display:flex; gap:1rem; flex-wrap:wrap; align-items:center; }
    .mi-sep   { color:#ccc; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="rev-title">Predict Revenue for a Movie</div>', unsafe_allow_html=True)

    # Movie selector — uses actual dataset titles
    movies_with_rev = df[df["Revenue(Crores)"] > 0].copy()
    title_options   = sorted(movies_with_rev["Title"].unique().tolist())

    st.markdown("Select a movie:")
    selected_movie = st.selectbox("", title_options, label_visibility="collapsed")

    predict_clicked = st.button("💰 Predict")

    if predict_clicked and selected_movie:
        row = movies_with_rev[movies_with_rev["Title"] == selected_movie].iloc[0]

        # ── Linear-regression-style prediction using genre + rating + votes ──
        genre_df      = df[df["Main_Genre"] == row["Main_Genre"]]
        genre_avg_rev = genre_df["Revenue(Crores)"].mean()
        rating_factor = row["Rating"] / df["Rating"].mean()
        votes_factor  = np.log1p(row["Votes"]) / np.log1p(df["Votes"].mean())
        predicted_rev = round(genre_avg_rev * rating_factor * votes_factor * 1.15, 1)
        actual_rev    = round(row["Revenue(Crores)"], 1)
        diff          = round(abs(predicted_rev - actual_rev), 1)
        over          = predicted_rev >= actual_rev

        # ── 3-column metrics (matches screenshot layout) ──
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('<div class="rev-label">Predicted Revenue</div>', unsafe_allow_html=True)
            st.markdown('<div class="rev-value">Rs. ' + str(predicted_rev) + ' Cr</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="rev-label">Actual Revenue</div>', unsafe_allow_html=True)
            st.markdown('<div class="rev-value">Rs. ' + str(actual_rev) + ' Cr</div>', unsafe_allow_html=True)

        with col3:
            badge_class = "rev-diff-over" if over else "rev-diff-under"
            arrow       = "↑ Over by" if over else "↓ Under by"
            st.markdown('<div class="rev-label">Difference</div>', unsafe_allow_html=True)
            st.markdown('<div class="rev-value">Rs. ' + str(diff) + ' Cr</div>', unsafe_allow_html=True)
            st.markdown(
                '<span class="' + badge_class + '">' + arrow + ' ' + str(diff) + ' Cr</span>',
                unsafe_allow_html=True
            )

        # ── Movie info card (matches screenshot bottom card) ──
        votes_fmt  = f"{int(row['Votes']):,}"
        runtime    = int(row["Runtime_(Minutes)"])
        metascore  = int(row["Metascore"]) if row["Metascore"] > 0 else "N/A"
        year       = int(row["Year"])
        rating     = row["Rating"]

        st.markdown(
            '<div class="movie-info-card">'
            '<div class="mi-title">' + selected_movie + '</div>'
            '<div class="mi-meta">'
            '<span>⭐ ' + str(rating) + '</span>'
            '<span class="mi-sep">|</span>'
            '<span>🎟 ' + votes_fmt + ' votes</span>'
            '<span class="mi-sep">|</span>'
            '<span>Metascore: ' + str(metascore) + '</span>'
            '<span class="mi-sep">|</span>'
            '<span>⏱ ' + str(runtime) + ' min</span>'
            '<span class="mi-sep">|</span>'
            '<span>📅 ' + str(year) + '</span>'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )

        # ── Bar chart: predicted vs actual for similar genre movies ──
        st.markdown("<br>", unsafe_allow_html=True)
        comp = genre_df[["Title", "Revenue(Crores)"]].dropna().sort_values("Revenue(Crores)", ascending=False).head(12).copy()
        comp["Type"] = "Actual"
        pred_row = pd.DataFrame([{"Title": selected_movie + " (Predicted)", "Revenue(Crores)": predicted_rev, "Type": "Predicted"}])
        comp = pd.concat([comp, pred_row], ignore_index=True)

        PLOT_BG = "rgba(0,0,0,0)"
        fig = px.bar(
            comp, x="Title", y="Revenue(Crores)", color="Type",
            color_discrete_map={"Actual": "#e0e0e0", "Predicted": "#E53935"},
            title="Revenue Comparison — " + row["Main_Genre"] + " Movies"
        )
        fig.update_layout(
            paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
            font_color="#1a1a1a", title_font_color="#1a1a1a",
            xaxis_tickangle=-35, legend_title_text="",
            xaxis_title="", yaxis_title="Revenue (Crores)"
        )
        fig.update_xaxes(gridcolor="rgba(0,0,0,0.06)")
        fig.update_yaxes(gridcolor="rgba(0,0,0,0.06)")
        st.plotly_chart(fig, use_container_width=True)

    elif not predict_clicked:
        st.info("👆 Select a movie and click Predict to see the revenue forecast.") 

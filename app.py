import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Movie Recommendation Engine",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif !important;
    background-color: #f0f2f6 !important;
    color: #1a1a1a !important;
}

/* ── Sidebar ── */
div[data-testid="stSidebar"] {
    background-color: #f0f2f6 !important;
    border-right: 1px solid #ddd !important;
}
div[data-testid="stSidebar"] * { color: #1a1a1a !important; }
div[data-testid="stSidebar"] h1 { font-size: 1.4rem !important; font-weight: 800 !important; }

/* Slider accent red */
div[data-testid="stSlider"] [data-testid="stTickBar"],
div[data-testid="stSlider"] .st-emotion-cache-1dp5vir,
div[data-testid="stSlider"] [role="slider"] {
    background-color: #E53935 !important;
    color: #E53935 !important;
}
.stSlider [data-baseweb="slider"] [role="slider"] { background: #E53935 !important; border-color: #E53935 !important; }

/* Multiselect tag red */
.stMultiSelect [data-baseweb="tag"] {
    background-color: #E53935 !important;
    color: #fff !important;
    border-radius: 20px !important;
}
.stMultiSelect [data-baseweb="tag"] span { color: #fff !important; }

/* Buttons */
.stButton > button {
    background-color: #E53935 !important;
    color: #fff !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.6rem !important;
    font-size: 1rem !important;
}
.stButton > button:hover { background-color: #c62828 !important; }

/* Tabs */
button[data-baseweb="tab"] {
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    color: #555 !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #E53935 !important;
    border-bottom: 2px solid #E53935 !important;
}

/* Movie result card */
.rec-card {
    background: #ffffff;
    border-radius: 12px;
    padding: 1.4rem 1.8rem;
    margin-bottom: 1rem;
    border: 1px solid #e0e0e0;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.rec-rank  { font-size: 0.85rem; color: #888; font-weight: 500; margin-bottom: 0.2rem; }
.rec-title { font-size: 1.55rem; font-weight: 800; color: #1a1a1a; margin-bottom: 0.4rem; }
.rec-meta  { font-size: 0.88rem; color: #555; margin-bottom: 0.7rem; display: flex; gap: 0.6rem; align-items: center; flex-wrap: wrap; }
.rec-badge {
    display: inline-block;
    background: #fde8e8;
    color: #c62828;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.78rem;
    font-weight: 500;
    margin-right: 4px;
    margin-bottom: 6px;
}
.rec-desc { font-size: 0.92rem; color: #444; line-height: 1.6; margin-top: 0.6rem; }

/* Seed card */
.seed-card {
    background: #fff;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin: 1rem 0 1.5rem;
    border: 1px solid #e0e0e0;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.seed-label { font-size: 0.78rem; color: #888; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.2rem; }
.seed-big   { font-size: 2rem; font-weight: 800; color: #1a1a1a; }
.seed-mid   { font-size: 1.4rem; font-weight: 700; color: #1a1a1a; }

/* Stat numbers on homepage */
.stat-label { font-size: 0.85rem; color: #888; margin-bottom: 0.1rem; }
.stat-value { font-size: 2.5rem; font-weight: 800; color: #1a1a1a; }

/* IMDb logo */
.imdb-badge {
    background: #F5C518;
    color: #000;
    font-weight: 900;
    font-size: 1.1rem;
    padding: 4px 10px;
    border-radius: 5px;
    margin-right: 0.5rem;
    letter-spacing: 1px;
    display: inline-block;
}

/* Metric display */
.big-metric-label { font-size: 0.85rem; color: #888; margin-bottom: 0.2rem; }
.big-metric-value { font-size: 2.2rem; font-weight: 800; color: #1a1a1a; }

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── DATA & MODEL ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_imdb.csv")
    df = df.dropna(subset=["Title","Genre","Rating","Year","Director"])
    for col in ["Genre","Director","Actors","Description"]:
        df[col] = df[col].fillna("")
    df["Metascore"] = df["Metascore"].fillna(df["Metascore"].median())
    df["Votes"]     = df["Votes"].fillna(0)
    df["Revenue(Crores)"] = df["Revenue(Crores)"].fillna(0)
    return df.reset_index(drop=True)

@st.cache_resource
def build_tfidf(_df):
    df = _df.copy()
    df["text_soup"] = (
        df["Genre"].str.replace(",", " ") + " " +
        df["Director"].str.replace(" ", "") + " " +
        df["Actors"].str.replace(",","").str.replace(" ","") + " " +
        df["Description"]
    )
    tfidf     = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1,2))
    tfidf_mat = tfidf.fit_transform(df["text_soup"])
    scaler    = MinMaxScaler()
    num_mat   = scaler.fit_transform(df[["Rating","Year","Votes","Runtime_(Minutes)"]].fillna(0))
    return tfidf_mat, num_mat

def get_recs(title, df, tfidf_mat, num_mat,
             genres=None, yr=(2006,2016), min_rating=0.0, director="Any", top_n=8):
    mask = df["Title"].str.lower() == title.lower()
    if not mask.any():
        return None
    idx     = df[mask].index[0]
    txt_sim = cosine_similarity(tfidf_mat[idx], tfidf_mat).flatten()
    num_sim = cosine_similarity(num_mat[idx].reshape(1,-1), num_mat).flatten()
    scores  = 0.70 * txt_sim + 0.30 * num_sim
    scores[idx] = -1
    out = df.copy()
    out["_score"] = scores
    if genres:
        out = out[out["Genre"].apply(lambda g: any(x.lower() in g.lower() for x in genres))]
    out = out[(out["Year"] >= yr[0]) & (out["Year"] <= yr[1]) & (out["Rating"] >= min_rating)]
    if director != "Any":
        out = out[out["Director"] == director]
    return out.sort_values("_score", ascending=False).head(top_n)

def browse_prefs(df, genres=None, yr=(2006,2016), min_rating=0.0, director="Any", top_n=8):
    out = df.copy()
    if genres:
        out = out[out["Genre"].apply(lambda g: any(x.lower() in g.lower() for x in genres))]
    out = out[(out["Year"] >= yr[0]) & (out["Year"] <= yr[1]) & (out["Rating"] >= min_rating)]
    if director != "Any":
        out = out[out["Director"] == director]
    if out.empty:
        return out
    out = out.copy()
    vmax = out["Votes"].max() + 1
    out["_score"] = 0.6*out["Rating"]/10 + 0.4*(np.log1p(out["Votes"])/np.log1p(vmax))
    return out.sort_values("_score", ascending=False).head(top_n)


# ── LOAD ──────────────────────────────────────────────────────────────────────
df = load_data()
tfidf_mat, num_mat = build_tfidf(df)

ALL_GENRES    = sorted(["Action","Adventure","Animation","Biography","Comedy",
                         "Crime","Drama","Fantasy","Horror","Mystery","Sci-Fi","Thriller"])
ALL_DIRECTORS = ["Any"] + sorted(df["Director"].unique().tolist())
ALL_TITLES    = sorted(df["Title"].unique().tolist())
PLOT_BG       = "rgba(0,0,0,0)"
FONT_COLOR    = "#1a1a1a"
GRID_COLOR    = "rgba(0,0,0,0.07)"
RED           = "#E53935"


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Filters")
    st.markdown("**Search Mode**")
    mode = st.radio("", ["🎯 Find Similar Movies", "🔍 Browse by Preference"],
                    label_visibility="collapsed")
    st.markdown("**Genre**")
    genre_filter = st.multiselect("", ALL_GENRES, default=[], label_visibility="collapsed")
    st.markdown("**Year Range**")
    year_range   = st.slider("", int(df["Year"].min()), int(df["Year"].max()), (2008, 2016),
                              label_visibility="collapsed")
    st.markdown("**Min Rating**")
    rating_min   = st.slider("", 1.0, 9.0, 6.0, 0.1, label_visibility="collapsed")
    st.markdown("**Director**")
    director_sel = st.selectbox("", ALL_DIRECTORS, label_visibility="collapsed")
    st.markdown("**Number of Results**")
    top_n        = st.slider("", 3, 20, 8, label_visibility="collapsed")
    st.markdown("---")
    st.caption("Model: TF-IDF + Cosine Similarity  Features: Genre · Director · Cast · Description · Rating · Year · Votes")


# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown(
    '<span class="imdb-badge">IMDb</span>'
    '<span style="font-size:2.2rem;font-weight:800;vertical-align:middle">🎬 Movie Recommendation Engine</span>',
    unsafe_allow_html=True
)
st.markdown("<p style='color:#888;margin-top:0.2rem'>Content-based filtering using genre, cast, director &amp; numeric features</p>",
            unsafe_allow_html=True)

# Stats row
s1, s2, s3, s4 = st.columns(4)
with s1:
    st.markdown('<div class="stat-label">Total Movies</div><div class="stat-value">' + str(len(df)) + '</div>', unsafe_allow_html=True)
with s2:
    st.markdown('<div class="stat-label">Directors</div><div class="stat-value">' + str(df["Director"].nunique()) + '</div>', unsafe_allow_html=True)
with s3:
    st.markdown('<div class="stat-label">Genres</div><div class="stat-value">' + str(len(ALL_GENRES)) + '</div>', unsafe_allow_html=True)
with s4:
    st.markdown('<div class="stat-label">Year Range</div><div class="stat-value">' + str(int(df["Year"].min())) + ' – ' + str(int(df["Year"].max())) + '</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Recommendations", "📊 Data Explorer",
    "🧠 How It Works", "⭐ Rating-Based ML", "💰 Revenue Prediction"
])


# ════════════════════════════════════════════════════════
# TAB 1 — RECOMMENDATIONS
# ════════════════════════════════════════════════════════
with tab1:
    if "Find Similar" in mode:
        st.markdown("### Find Similar Movies")
        st.markdown("Pick a movie you enjoyed:")
        selected = st.selectbox("", ALL_TITLES, label_visibility="collapsed")

        if st.button("Get Recommendations 🚀"):
            seed = df[df["Title"] == selected].iloc[0]

            with st.expander("About: " + selected, expanded=True):
                ca, cb = st.columns([2, 1])
                with ca:
                    st.markdown(
                        "**Director:** " + seed["Director"] +
                        " | **Year:** " + str(int(seed["Year"])) +
                        " | **Genre:** " + seed["Genre"]
                    )
                    st.markdown("**Cast:** " + seed["Actors"])
                    st.write(seed["Description"])
                with cb:
                    st.markdown('<div class="seed-label">IMDb Rating</div><div class="seed-big">⭐ ' + str(seed["Rating"]) + '</div>', unsafe_allow_html=True)
                    st.markdown('<div class="seed-label">Runtime</div><div class="seed-mid">' + str(int(seed["Runtime_(Minutes)"])) + ' min</div>', unsafe_allow_html=True)
                    st.markdown('<div class="seed-label">Votes</div><div class="seed-mid">' + f"{int(seed['Votes']):,}" + '</div>', unsafe_allow_html=True)

            recs = get_recs(selected, df, tfidf_mat, num_mat,
                            genre_filter if genre_filter else None,
                            year_range, rating_min, director_sel, top_n)

            if recs is None or recs.empty:
                st.warning("No movies found — try relaxing your filters.")
            else:
                st.success("Found " + str(len(recs)) + " recommendations!")
                for rank, (_, row) in enumerate(recs.iterrows(), 1):
                    title  = row["Title"]
                    year   = int(row["Year"])
                    rat    = row["Rating"]
                    direc  = row["Director"]
                    rt     = int(row["Runtime_(Minutes)"])
                    desc   = str(row["Description"])[:220]
                    genres = [g.strip() for g in row["Genre"].split(",")]
                    badges = "".join('<span class="rec-badge">' + g + '</span>' for g in genres)
                    st.markdown(
                        '<div class="rec-card">'
                        '<div class="rec-rank">#' + str(rank) + '</div>'
                        '<div class="rec-title">' + title + '</div>'
                        '<div class="rec-meta">🎬 ' + direc + ' &nbsp;|&nbsp; 📅 ' + str(year) + ' &nbsp;|&nbsp; ⭐ ' + str(rat) + ' &nbsp;|&nbsp; ⏱ ' + str(rt) + ' min</div>'
                        '<div>' + badges + '</div>'
                        '<div class="rec-desc">' + desc + '…</div>'
                        '</div>',
                        unsafe_allow_html=True
                    )

    else:  # Browse by Preference
        st.markdown("### Browse by Preference")
        st.info("Set your genre, year, rating and director filters in the sidebar, then click Search.")
        if st.button("🔍 Search Movies"):
            results = browse_prefs(df,
                                   genre_filter if genre_filter else None,
                                   year_range, rating_min, director_sel, top_n)
            if results.empty:
                st.warning("No movies match — try widening your filters.")
            else:
                st.success("Found " + str(len(results)) + " movies!")
                for rank, (_, row) in enumerate(results.iterrows(), 1):
                    title  = row["Title"]
                    year   = int(row["Year"])
                    rat    = row["Rating"]
                    direc  = row["Director"]
                    desc   = str(row["Description"])[:220]
                    genres = [g.strip() for g in row["Genre"].split(",")]
                    badges = "".join('<span class="rec-badge">' + g + '</span>' for g in genres)
                    st.markdown(
                        '<div class="rec-card">'
                        '<div class="rec-rank">#' + str(rank) + '</div>'
                        '<div class="rec-title">' + title + '</div>'
                        '<div class="rec-meta">🎬 ' + direc + ' &nbsp;|&nbsp; 📅 ' + str(year) + ' &nbsp;|&nbsp; ⭐ ' + str(rat) + '</div>'
                        '<div>' + badges + '</div>'
                        '<div class="rec-desc">' + desc + '…</div>'
                        '</div>',
                        unsafe_allow_html=True
                    )


# ════════════════════════════════════════════════════════
# TAB 2 — DATA EXPLORER
# ════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Dataset Explorer")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x="Rating", nbins=30, title="Rating Distribution",
                           color_discrete_sequence=[RED])
        fig.update_layout(paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
                          font_color=FONT_COLOR, title_font_color=FONT_COLOR)
        fig.update_xaxes(gridcolor=GRID_COLOR)
        fig.update_yaxes(gridcolor=GRID_COLOR)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        gc = df["Main_Genre"].value_counts().reset_index()
        gc.columns = ["Genre","Count"]
        fig = px.bar(gc, x="Count", y="Genre", orientation="h",
                     color="Count", color_continuous_scale=["#ffcdd2", RED],
                     title="Movies per Genre")
        fig.update_layout(paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
                          font_color=FONT_COLOR, title_font_color=FONT_COLOR,
                          coloraxis_showscale=False, yaxis={"categoryorder":"total ascending"})
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
                      markers=True, color_discrete_sequence=[RED])
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

    st.markdown("### Raw Dataset")
    q    = st.text_input("🔎 Search title")
    show = df if not q else df[df["Title"].str.contains(q, case=False, na=False)]
    st.dataframe(
        show[["Title","Genre","Director","Year","Rating","Runtime_(Minutes)","Votes","Revenue(Crores)"]],
        use_container_width=True, height=400
    )


# ════════════════════════════════════════════════════════
# TAB 3 — HOW IT WORKS
# ════════════════════════════════════════════════════════
with tab3:
    st.markdown("### How the Recommendation Engine Works")
    st.markdown("""
The engine uses **Content-Based Filtering** with a hybrid scoring formula:

| Component | Weight | Details |
|-----------|--------|---------|
| **TF-IDF Text Similarity** | 70% | Genre + Director + Cast + Description → 5 000-feature TF-IDF matrix |
| **Numerical Similarity** | 30% | Rating · Year · Votes · Runtime → MinMax-normalised, cosine similarity |

**Combined score:**
```
final_score = 0.70 × cosine_sim(text) + 0.30 × cosine_sim(numerics)
```

**Why Content-Based?**
- ✅ No cold-start problem — works without any user history
- ✅ Fully explainable — every recommendation is traceable
- ✅ Rich IMDB metadata (genre, cast, director, description)
    """)

    col1, col2 = st.columns(2)
    with col1:
        ds = df.groupby("Director").agg(Avg_Rating=("Rating","mean"), Movies=("Title","count")).reset_index()
        ds = ds[ds["Movies"] >= 2].sort_values("Avg_Rating", ascending=False).head(15)
        fig = px.bar(ds, x="Avg_Rating", y="Director", orientation="h",
                     color="Avg_Rating", color_continuous_scale=["#ffcdd2", RED],
                     hover_data=["Movies"], title="Top 15 Directors by Avg Rating")
        fig.update_layout(paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
                          font_color=FONT_COLOR, title_font_color=FONT_COLOR,
                          coloraxis_showscale=False, yaxis={"categoryorder":"total ascending"})
        fig.update_xaxes(range=[7, 9], gridcolor=GRID_COLOR)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        ga = df.groupby("Main_Genre")["Rating"].mean().sort_values(ascending=False).reset_index()
        fig = px.bar(ga, x="Main_Genre", y="Rating",
                     color="Rating", color_continuous_scale=["#ffcdd2", RED],
                     title="Average Rating by Genre")
        fig.update_layout(paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
                          font_color=FONT_COLOR, title_font_color=FONT_COLOR,
                          coloraxis_showscale=False)
        fig.update_yaxes(range=[5, 8], gridcolor=GRID_COLOR)
        st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════
# TAB 4 — RATING-BASED ML (SVD + Ridge, 5-fold CV)
# ════════════════════════════════════════════════════════
with tab4:
    st.markdown("### Rating-Based ML Model")
    st.markdown("Runs a Ridge regression on top of SVD latent features using 5-fold CV to predict IMDb ratings.")

    if st.button("Run Accuracy Check 📊"):
        with st.spinner("Training SVD + Ridge model with 5-fold cross-validation…"):
            # Feature matrix
            feat_cols = ["Year","Runtime_(Minutes)","Votes","Metascore","Revenue(Crores)"]
            X_raw = df[feat_cols].fillna(0).values
            y     = df["Rating"].values

            # SVD latent features from TF-IDF
            svd   = TruncatedSVD(n_components=50, random_state=42)
            X_svd = svd.fit_transform(tfidf_mat)

            X = np.hstack([X_svd, MinMaxScaler().fit_transform(X_raw)])

            # Ridge + 5-fold CV predictions
            ridge     = Ridge(alpha=1.0)
            y_pred_cv = cross_val_predict(ridge, X, y, cv=5)

            rmse = round(np.sqrt(mean_squared_error(y, y_pred_cv)), 4)
            mae  = round(mean_absolute_error(y, y_pred_cv), 4)

        # Metrics row
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown('<div class="big-metric-label">RMSE ❓</div><div class="big-metric-value">' + str(rmse) + '</div>', unsafe_allow_html=True)
        with m2:
            st.markdown('<div class="big-metric-label">MAE ❓</div><div class="big-metric-value">' + str(mae) + '</div>', unsafe_allow_html=True)
        with m3:
            st.markdown('<div class="big-metric-label">Rating Scale</div><div class="big-metric-value">1 – 10</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Actual vs Predicted scatter (matches screenshot 2)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=y, y=y_pred_cv,
            mode="markers",
            marker=dict(color=RED, opacity=0.6, size=7),
            name="Movies"
        ))
        # Perfect-prediction line
        mn, mx = float(y.min()), float(y.max())
        fig.add_trace(go.Scatter(
            x=[mn, mx], y=[mn, mx],
            mode="lines",
            line=dict(color="#F5C518", dash="dash", width=2),
            name="Perfect Fit"
        ))
        fig.update_layout(
            title="Actual vs Predicted (5-fold CV)",
            xaxis_title="Actual Rating",
            yaxis_title="Predicted Rating",
            paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
            font_color=FONT_COLOR, title_font_color=FONT_COLOR,
            legend=dict(orientation="h", y=-0.15)
        )
        fig.update_xaxes(gridcolor=GRID_COLOR)
        fig.update_yaxes(gridcolor=GRID_COLOR)
        st.plotly_chart(fig, use_container_width=True)

        # Feature importance
        ridge.fit(X, y)
        svd_importance = np.abs(ridge.coef_[:50]).mean()
        num_importance = np.abs(ridge.coef_[50:]).mean()
        fi_df = pd.DataFrame({
            "Feature Group": ["SVD Latent (Text)", "Numerical Features"],
            "Avg |Coefficient|": [round(svd_importance,4), round(num_importance,4)]
        })
        fig2 = px.bar(fi_df, x="Feature Group", y="Avg |Coefficient|",
                      color="Avg |Coefficient|",
                      color_continuous_scale=["#ffcdd2", RED],
                      title="Feature Group Importance")
        fig2.update_layout(paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
                           font_color=FONT_COLOR, title_font_color=FONT_COLOR,
                           coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Click **Run Accuracy Check** to train the SVD + Ridge model and see RMSE, MAE and the Actual vs Predicted plot.")


# ════════════════════════════════════════════════════════
# TAB 5 — REVENUE PREDICTION
# ════════════════════════════════════════════════════════
with tab5:
    st.markdown("### Predict Revenue for a Movie")
    st.markdown("Select a movie:")

    movies_with_rev = df[df["Revenue(Crores)"] > 0].copy()
    title_opts      = sorted(movies_with_rev["Title"].unique().tolist())
    sel_movie       = st.selectbox("", title_opts, label_visibility="collapsed")

    if st.button("💰 Predict"):
        row = movies_with_rev[movies_with_rev["Title"] == sel_movie].iloc[0]

        genre_df      = df[df["Main_Genre"] == row["Main_Genre"]]
        genre_avg     = genre_df["Revenue(Crores)"].mean()
        rating_factor = row["Rating"] / df["Rating"].mean()
        votes_factor  = np.log1p(row["Votes"]) / np.log1p(df["Votes"].mean())
        predicted     = round(genre_avg * rating_factor * votes_factor * 1.15, 1)
        actual        = round(row["Revenue(Crores)"], 1)
        diff          = round(abs(predicted - actual), 1)
        over          = predicted >= actual

        # 3-col metrics (matches screenshot)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div class="big-metric-label">Predicted Revenue</div><div class="big-metric-value">Rs. ' + str(predicted) + ' Cr</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="big-metric-label">Actual Revenue</div><div class="big-metric-value">Rs. ' + str(actual) + ' Cr</div>', unsafe_allow_html=True)
        with c3:
            badge_cls = "background:#fde8e8;color:#c62828" if over else "background:#e8f5e9;color:#2e7d32"
            arrow     = "↑ Over by" if over else "↓ Under by"
            st.markdown(
                '<div class="big-metric-label">Difference</div>'
                '<div class="big-metric-value">Rs. ' + str(diff) + ' Cr</div>'
                '<span style="' + badge_cls + ';border-radius:20px;padding:3px 12px;font-size:0.82rem;font-weight:600;margin-top:6px;display:inline-block">'
                + arrow + ' ' + str(diff) + ' Cr</span>',
                unsafe_allow_html=True
            )

        # Movie info card
        votes_fmt = f"{int(row['Votes']):,}"
        rt        = int(row["Runtime_(Minutes)"])
        meta      = int(row["Metascore"]) if row["Metascore"] > 0 else "N/A"
        yr        = int(row["Year"])
        st.markdown(
            '<div class="rec-card" style="margin-top:1.5rem">'
            '<div style="font-size:1.4rem;font-weight:800;margin-bottom:0.5rem">' + sel_movie + '</div>'
            '<div style="font-size:0.9rem;color:#555;display:flex;gap:1rem;flex-wrap:wrap">'
            '<span>⭐ ' + str(row["Rating"]) + '</span>'
            '<span style="color:#ccc">|</span>'
            '<span>🎟 ' + votes_fmt + ' votes</span>'
            '<span style="color:#ccc">|</span>'
            '<span>Metascore: ' + str(meta) + '</span>'
            '<span style="color:#ccc">|</span>'
            '<span>⏱ ' + str(rt) + ' min</span>'
            '<span style="color:#ccc">|</span>'
            '<span>📅 ' + str(yr) + '</span>'
            '</div></div>',
            unsafe_allow_html=True
        )

        # Comparison chart
        comp = genre_df[["Title","Revenue(Crores)"]].dropna()
        comp = comp[comp["Revenue(Crores)"] > 0].sort_values("Revenue(Crores)", ascending=False).head(12).copy()
        comp["Type"] = "Actual"
        pred_row = pd.DataFrame([{"Title": sel_movie + " (Predicted)", "Revenue(Crores)": predicted, "Type": "Predicted"}])
        comp = pd.concat([comp, pred_row], ignore_index=True)

        fig = px.bar(comp, x="Title", y="Revenue(Crores)", color="Type",
                     color_discrete_map={"Actual": "#e0e0e0", "Predicted": RED},
                     title="Revenue Comparison — " + row["Main_Genre"] + " Movies")
        fig.update_layout(paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
                          font_color=FONT_COLOR, title_font_color=FONT_COLOR,
                          xaxis_tickangle=-35, legend_title_text="",
                          xaxis_title="", yaxis_title="Revenue (Crores)")
        fig.update_xaxes(gridcolor=GRID_COLOR)
        fig.update_yaxes(gridcolor=GRID_COLOR)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Select a movie and click **Predict** to see the revenue forecast.")

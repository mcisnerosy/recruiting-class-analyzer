# ─────────────────────────────────────────────────────────────────────────────
# app.py
# PURPOSE: Streamlit dashboard that lets a user pick any two Power 4 teams
# and a season, then shows the model's win probability for that matchup.
# Also shows team logos, program rankings, head-to-head history, and a
# log of all predictions made during the session.
#
# To run locally:  streamlit run app.py
# Deployed at:     https://recruiting-class-analyzer-lkeemy72oemz9iusy36uax.streamlit.app/
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from dotenv import load_dotenv

# Load .env so the API key is available locally (Streamlit Cloud uses st.secrets instead)
load_dotenv()

# ── API key helper ────────────────────────────────────────────────────────────
# Try Streamlit Cloud secrets first (production), fall back to .env (local dev).
# If neither exists, show an error and stop the app rather than crashing silently.
def get_api_key():
    key = None
    try:
        key = st.secrets["CFBD_API_KEY"]
    except Exception:
        pass
    if not key:
        key = os.getenv("CFBD_API_KEY")
    if not key:
        st.error("CFBD_API_KEY not found. Add it to Streamlit secrets or your .env file.")
        st.stop()
    return key

# ── Page configuration ────────────────────────────────────────────────────────
# Must be called before any other Streamlit commands.
# layout="centered" keeps the content in a readable column rather than full-width.
st.set_page_config(page_title="CFB Win Probability", page_icon="🏈", layout="centered")

# ── Custom CSS ────────────────────────────────────────────────────────────────
# Streamlit's default theme is light grey — we override it with a dark background
# and custom fonts (Bebas Neue for headers, Barlow Condensed for body text).
# unsafe_allow_html=True is required any time we inject raw HTML/CSS.
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Barlow+Condensed:wght@400;600&display=swap');
        @import url('https://fonts.googleapis.com/icon?family=Material+Icons');

        /* Dark background everywhere — main app, sidebar, header bar */
        .stApp { background-color: #111111; }
        section[data-testid="stSidebar"] { background-color: #111111; }
        [data-testid="stHeader"] { background-color: #111111; }
        [data-testid="stToolbar"] { background-color: #111111; }

        /* Typography overrides — Bebas Neue for display, Barlow Condensed for body */
        h1 { font-family: 'Bebas Neue', sans-serif !important; font-size: 3rem !important; color: #ffffff !important; letter-spacing: 2px; }
        h2, h3 { font-family: 'Bebas Neue', sans-serif !important; color: #ffffff !important; letter-spacing: 1px; }
        p, label, .stCaption { font-family: 'Barlow Condensed', sans-serif !important; color: #e0e0e0 !important; }
        .stMarkdown, .stText, [data-testid="stMetricLabel"], [data-testid="stMetricValue"] { font-family: 'Barlow Condensed', sans-serif !important; color: #e0e0e0 !important; }
        [data-testid="stHeader"] { background-color: #111111 !important; }

        /* Hide Streamlit's default hamburger menu and footer branding */
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }

        /* Win probability bar — a custom HTML bar chart using team colors */
        .color-bar-wrap { background: #2a2a2a; border-radius: 6px; height: 18px; width: 100%; margin-top: 6px; }
        .color-bar-fill { height: 18px; border-radius: 6px; transition: width 0.4s ease; }

        /* Mobile: stack columns vertically on small screens */
        @media (max-width: 640px) {
            h1 { font-size: 2rem !important; }
            [data-testid="column"] { min-width: 100% !important; }
        }
    </style>
""", unsafe_allow_html=True)

# ── Data loading functions ────────────────────────────────────────────────────
# @st.cache_data tells Streamlit to only run this function once and reuse the result.
# Without caching, the CSV would be re-read from disk on every user interaction.
@st.cache_data
def load_data():
    games    = pd.read_csv('games_with_features.csv')   # game-level data for training
    features = pd.read_csv('team_features.csv')          # per-team stats for lookups
    return games, features

# load_team_data pulls logos and brand colors from the CFBD API.
# This is separate from load_data() because it hits the network — we cache it
# so the API is only called once per session, not on every rerender.
@st.cache_data
def load_team_data():
    import cfbd
    cfg = cfbd.Configuration(access_token=get_api_key())
    logos  = {}  # maps team name → logo URL (dark version preferred)
    colors = {}  # maps team name → primary hex color code
    with cfbd.ApiClient(cfg) as client:
        teams = cfbd.TeamsApi(client).get_fbs_teams()
        for t in teams:
            if t.logos:
                # Prefer the dark-background logo — our app has a dark theme
                dark = next((l for l in t.logos if 'dark' in l), t.logos[0])
                logos[t.school] = dark
            if t.color:
                colors[t.school] = t.color
    return logos, colors

# Load logos and colors at startup — used throughout the app
LOGOS, COLORS = load_team_data()

# ── Model training ────────────────────────────────────────────────────────────
# @st.cache_resource is like @st.cache_data but for objects that shouldn't be
# serialized (like a trained sklearn model). It's only trained once per session.
@st.cache_resource
def train_model(games):
    cols = ['rank_diff', 'points_diff', 'neutral', 'ppa_return_diff', 'usage_return_diff']
    df = games.dropna(subset=cols)  # can't train on rows with missing features
    lr = LogisticRegression()
    lr.fit(df[cols], df['home_win'])  # train on the full historical dataset
    return lr

# Load data and train model — these run once and are cached
games_df, team_features = load_data()
model = train_model(games_df)

# ── Power 4 team list and year options ───────────────────────────────────────
# Only show Power 4 teams in the dropdowns — the model was trained on those only
POWER4 = {
    'Alabama', 'Arkansas', 'Auburn', 'Florida', 'Georgia', 'Kentucky',
    'LSU', 'Mississippi State', 'Missouri', 'Ole Miss', 'South Carolina',
    'Tennessee', 'Texas', 'Texas A&M', 'Vanderbilt',
    'Illinois', 'Indiana', 'Iowa', 'Maryland', 'Michigan', 'Michigan State',
    'Minnesota', 'Nebraska', 'Northwestern', 'Ohio State', 'Oregon',
    'Penn State', 'Purdue', 'Rutgers', 'UCLA', 'USC', 'Washington', 'Wisconsin',
    'Baylor', 'BYU', 'Cincinnati', 'Colorado', 'Houston', 'Iowa State',
    'Kansas', 'Kansas State', 'Oklahoma', 'Oklahoma State', 'TCU',
    'Texas Tech', 'UCF', 'Utah', 'West Virginia',
    'Boston College', 'Clemson', 'Duke', 'Florida State', 'Georgia Tech',
    'Louisville', 'Miami', 'NC State', 'North Carolina', 'Notre Dame',
    'Pittsburgh', 'Stanford', 'Syracuse', 'Virginia', 'Virginia Tech', 'Wake Forest',
}
# Intersect team_features teams with POWER4 to exclude any non-P4 teams in the CSV
TEAMS = sorted(t for t in team_features['team'].unique() if t in POWER4)
YEARS = sorted(team_features['year'].unique(), reverse=True)  # newest year first

# ── Utility functions ─────────────────────────────────────────────────────────
def safe(val):
    # Converts a value to float, treating NaN (missing) as 0.
    # Needed because pandas NaN silently breaks predict_proba — `or 0` does not catch it.
    return 0.0 if pd.isna(val) else float(val)

def fmt_rank(v):
    # Formats a recruiting rank as "#12" — returns "N/A" if missing
    return f"#{int(round(v))}" if pd.notna(v) else "N/A"

def fmt_pct(v):
    # Formats a decimal (0.67) as a percentage ("67.0%") — returns "N/A" if 0 or missing
    return f"{v:.1%}" if pd.notna(v) and v != 0 else "N/A"

def get_team_stats(team, year):
    # Look up a team's stats for a given year from team_features.
    # If the exact year isn't in the data, fall back to the nearest available year
    # (common for newly added or recently expanded teams).
    row = team_features[(team_features['team'] == team) & (team_features['year'] == year)]
    if row.empty:
        available = team_features[team_features['team'] == team]
        if available.empty:
            return None  # team not in dataset at all
        # Pick the row whose year is closest to what was requested
        row = available.iloc[(available['year'] - year).abs().argsort()].iloc[[0]]
    return row.iloc[0]

def get_last_season_record(team, year):
    # Count wins and losses from the previous season (year - 1) using games_with_features.csv.
    # We look for all games where the team appeared (as home or away).
    prev = games_df[
        (games_df['year'] == year - 1) &
        ((games_df['home_team'] == team) | (games_df['away_team'] == team))
    ]
    if prev.empty:
        return None, None
    # Count wins: home win when playing at home, or home_win == 0 when playing away
    wins   = ((prev['home_team'] == team) & (prev['home_win'] == 1)).sum() + \
             ((prev['away_team'] == team) & (prev['home_win'] == 0)).sum()
    losses = len(prev) - wins
    return int(wins), int(losses)

def get_head_to_head(home_team, away_team):
    # Count all historical matchups between two teams in the dataset (2014–2023).
    # We look for games where either team appeared as home OR away.
    h2h = games_df[
        ((games_df['home_team'] == home_team) & (games_df['away_team'] == away_team)) |
        ((games_df['home_team'] == away_team) & (games_df['away_team'] == home_team))
    ]
    if h2h.empty:
        return 0, 0, 0
    # Count wins from home_team's perspective regardless of which side they were on
    home_wins = (
        ((h2h['home_team'] == home_team) & (h2h['home_win'] == 1)) |
        ((h2h['away_team'] == home_team) & (h2h['home_win'] == 0))
    ).sum()
    away_wins = len(h2h) - home_wins
    return int(home_wins), int(away_wins), len(h2h)

def confidence_label(prob):
    # Translate raw probability into a human-readable confidence label.
    # A 52% vs 48% prediction is essentially a coin flip — we should say so.
    # A 70%+ prediction is a real edge — meaningful to call out.
    margin = abs(prob - 0.5)
    if margin < 0.08:
        return "🟡 Low Confidence — essentially a toss-up"
    elif margin < 0.18:
        return "🟠 Medium Confidence — slight edge"
    else:
        return "🟢 High Confidence — clear advantage"

# ── Session state: prediction history ─────────────────────────────────────────
# st.session_state persists data across Streamlit rerenders (which happen every
# time the user interacts with any widget). Without it, the history list would
# reset to empty every time the user changed a dropdown.
if 'history' not in st.session_state:
    st.session_state.history = []

# ── URL parameter handling (shareable links) ──────────────────────────────────
# st.query_params reads ?home=Alabama&away=Michigan&year=2023 from the URL.
# If those params exist (someone shared a link), pre-fill the dropdowns.
params       = st.query_params
default_home = params.get('home', 'Alabama')
default_away = params.get('away', 'Michigan')
default_year = int(params.get('year', YEARS[0]))
# Validate — if a shared URL has a team not in our list, fall back to defaults
if default_home not in TEAMS: default_home = 'Alabama'
if default_away not in TEAMS: default_away = 'Michigan'
if default_year not in YEARS: default_year = YEARS[0]

# ── Sidebar: program talent rankings ─────────────────────────────────────────
# Shows the top 10 and bottom 10 programs by 4-year average recruiting rank
# for the selected season — gives context before making a prediction.
with st.sidebar:
    st.markdown("## Program Rankings")
    sidebar_year = st.selectbox("Season", YEARS, key="sidebar_year")

    # Filter team_features to Power 4 teams for the selected year
    year_talent = team_features[
        (team_features['year'] == sidebar_year) &
        (team_features['team'].isin(POWER4))
    ].copy().dropna(subset=['avg_rank']).sort_values('avg_rank')  # sort best→worst (lower rank = better)

    st.markdown("### Top 10 Programs")
    for i, row in enumerate(year_talent.head(10).itertuples(), 1):
        # Show team logo inline next to the ranking number and name
        logo_html = f"<img src='{LOGOS[row.team]}' width='22' style='vertical-align:middle; margin-right:6px;'>" if row.team in LOGOS else ""
        st.markdown(
            f"{logo_html}**{i}. {row.team}** — Avg Rank #{int(round(row.avg_rank))}",
            unsafe_allow_html=True
        )

    st.divider()

    st.markdown("### Bottom 10 Programs")
    bottom = year_talent.tail(10).iloc[::-1]  # reverse so worst is at the top
    for i, row in enumerate(bottom.itertuples(), 1):
        logo_html = f"<img src='{LOGOS[row.team]}' width='22' style='vertical-align:middle; margin-right:6px;'>" if row.team in LOGOS else ""
        st.markdown(
            f"{logo_html}**{i}. {row.team}** — Avg Rank #{int(round(row.avg_rank))}",
            unsafe_allow_html=True
        )

# ── Page header ───────────────────────────────────────────────────────────────
st.title("🏈 College Football Win Probability")
st.caption("Predictions based on 4-year recruiting talent and returning production. Power 4 teams only.")
st.caption("Built by Marcos Cisneros")
st.divider()

# ── Team selectors ────────────────────────────────────────────────────────────
# Three columns: home team (wide) | "vs" label (narrow) | away team (wide)
col1, col2, col3 = st.columns([5, 1, 5])

with col1:
    st.subheader("Home Team")
    # label_visibility="collapsed" hides the duplicate "Home Team" label above the dropdown
    home_team = st.selectbox("Home Team", TEAMS, index=TEAMS.index(default_home), key="home", label_visibility="collapsed")
    if home_team in LOGOS:
        st.image(LOGOS[home_team], width=100)

with col2:
    st.markdown("<br><br><div style='text-align:center; font-size:1.4rem; font-weight:bold;'>vs</div>",
                unsafe_allow_html=True)

with col3:
    st.subheader("Away Team")
    away_team = st.selectbox("Away Team", TEAMS, index=TEAMS.index(default_away), key="away", label_visibility="collapsed")
    if away_team in LOGOS:
        st.image(LOGOS[away_team], width=100)

# Season selector and neutral site toggle on the same row
col_year, col_neutral = st.columns([3, 2])
with col_year:
    year = st.selectbox("Season", YEARS, index=YEARS.index(default_year))
with col_neutral:
    # Neutral site removes the home field advantage that's built into the model
    neutral = st.checkbox("Neutral site game", value=False)

st.divider()

# ── Build the prediction ──────────────────────────────────────────────────────
home_stats = get_team_stats(home_team, year)
away_stats = get_team_stats(away_team, year)

if home_stats is None or away_stats is None:
    st.error("Could not find stats for one or both teams in that year.")
    st.stop()

# Compute differential features — same formula used when training the model
rank_diff         = away_stats['avg_rank']   - home_stats['avg_rank']
points_diff       = home_stats['avg_points'] - away_stats['avg_points']
ppa_return_diff   = safe(home_stats['ppa_return'])   - safe(away_stats['ppa_return'])
usage_return_diff = safe(home_stats['usage_return'])  - safe(away_stats['usage_return'])
neutral_val       = int(neutral)

# Package features into a single-row DataFrame — predict_proba requires this shape
X = pd.DataFrame(
    [[rank_diff, points_diff, neutral_val, ppa_return_diff, usage_return_diff]],
    columns=['rank_diff', 'points_diff', 'neutral', 'ppa_return_diff', 'usage_return_diff']
)

# predict_proba returns [prob_away_win, prob_home_win] — take index [1] for home win
home_prob = model.predict_proba(X)[0][1]
away_prob = 1 - home_prob  # probabilities must sum to 1

# ── Display win probabilities ─────────────────────────────────────────────────
st.subheader("Win Probability")

col_h, col_a = st.columns(2)
# Look up each team's brand color — fall back to a neutral color if not found
home_color = COLORS.get(home_team, '#4a90d9')
away_color = COLORS.get(away_team, '#e05c5c')

with col_h:
    st.metric(home_team, f"{home_prob:.1%}")
    # Render a colored progress bar using inline HTML + CSS
    # The width is set proportionally to the win probability (e.g. 65% prob → 65% wide bar)
    st.markdown(
        f"<div class='color-bar-wrap'><div class='color-bar-fill' style='width:{home_prob*100:.1f}%;background:{home_color};'></div></div>",
        unsafe_allow_html=True
    )
with col_a:
    st.metric(away_team, f"{away_prob:.1%}")
    st.markdown(
        f"<div class='color-bar-wrap'><div class='color-bar-fill' style='width:{away_prob*100:.1f}%;background:{away_color};'></div></div>",
        unsafe_allow_html=True
    )

# Confidence label — tells the user how much to trust the prediction
st.info(confidence_label(home_prob))

# Upset alert — flags when the underdog has a meaningful chance (within 5%)
winner      = home_team if home_prob >= away_prob else away_team
winner_prob = max(home_prob, away_prob)
loser       = away_team if home_prob >= away_prob else home_team

if winner_prob < 0.55:
    st.warning(f"⚠️ **Upset Alert** — {loser} has a real shot. Don't sleep on this one.")

# Summary line — states the model's pick and the margin
if home_prob > away_prob:
    st.success(f"Model favors **{home_team}** by {abs(home_prob - away_prob):.1%}")
else:
    st.success(f"Model favors **{away_team}** by {abs(home_prob - away_prob):.1%}")

# ── Share link ────────────────────────────────────────────────────────────────
# Update the browser URL with the current selection so users can copy/share it.
# Anyone opening that URL will see the same matchup pre-filled in the dropdowns.
st.query_params.update({'home': home_team, 'away': away_team, 'year': str(year)})
share_url = f"?home={home_team.replace(' ', '+')}&away={away_team.replace(' ', '+')}&year={year}"
st.caption(f"Share this matchup: `{share_url}`")

# ── Log to prediction history ─────────────────────────────────────────────────
# Only add to history if this prediction is different from the last one
# (prevents duplicate entries when Streamlit rerenders without user input)
entry = {
    'Season':   year,
    'Home':     home_team,
    'Away':     away_team,
    'Predicted Winner': winner,
    'Win Prob': f"{winner_prob:.1%}",
    'Confidence': confidence_label(home_prob).split('—')[0].strip(),
}
if not st.session_state.history or st.session_state.history[-1] != entry:
    st.session_state.history.append(entry)

st.divider()

# ── Last season record ─────────────────────────────────────────────────────────
# Shows each team's win-loss record from the season before the selected year.
# Provides quick context on recent form without needing external data.
st.subheader("Last Season Record")

col_hr, col_ar = st.columns(2)
home_w, home_l = get_last_season_record(home_team, year)
away_w, away_l = get_last_season_record(away_team, year)

with col_hr:
    if home_w is not None:
        st.metric(home_team, f"{home_w}–{home_l}", help=f"{year - 1} regular season")
    else:
        st.metric(home_team, "N/A")

with col_ar:
    if away_w is not None:
        st.metric(away_team, f"{away_w}–{away_l}", help=f"{year - 1} regular season")
    else:
        st.metric(away_team, "N/A")

st.divider()

# ── Head-to-head history ──────────────────────────────────────────────────────
# Shows how many times these teams have played in the dataset (2014–2023)
# and which team has won more often.
st.subheader("Historical Head-to-Head (2014–2023)")

h_wins, a_wins, total = get_head_to_head(home_team, away_team)

if total == 0:
    st.write("These teams have not met in the dataset.")
else:
    col_hh1, col_hh2, col_hh3 = st.columns(3)
    col_hh1.metric(f"{home_team} wins", h_wins)
    col_hh2.metric("Games played", total)
    col_hh3.metric(f"{away_team} wins", a_wins)

st.divider()

# ── Why the model thinks this ─────────────────────────────────────────────────
# Shows the raw stats behind the prediction so users understand what's driving it.
# Recruiting rank and returning production are the two main signals in the model.
st.subheader("Why the Model Thinks This")

comparison = pd.DataFrame({
    "Stat": [
        "4-Year Avg Recruiting Rank",
        "4-Year Avg Recruiting Points",
        "Returning Production (% PPA)",
        "Returning Usage",
    ],
    home_team: [
        fmt_rank(home_stats['avg_rank']),
        f"{home_stats['avg_points']:.1f}",
        fmt_pct(home_stats['ppa_return']),
        fmt_pct(home_stats['usage_return']),
    ],
    away_team: [
        fmt_rank(away_stats['avg_rank']),
        f"{away_stats['avg_points']:.1f}",
        fmt_pct(away_stats['ppa_return']),
        fmt_pct(away_stats['usage_return']),
    ],
}).set_index("Stat")

st.table(comparison)

st.divider()

# ── Prediction history ─────────────────────────────────────────────────────────
# Shows all predictions made during this session, newest first ([::-1] reverses the list).
# The Clear History button resets session_state and triggers a full rerender.
if st.session_state.history:
    st.subheader("Prediction History")
    history_df = pd.DataFrame(st.session_state.history[::-1])
    st.dataframe(history_df, use_container_width=True, hide_index=True)
    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()

st.divider()

# ── Disclaimer ─────────────────────────────────────────────────────────────────
# States the model's known limitations upfront so users don't misuse it.
# Key caveat: the model doesn't know about injuries, coaching changes, or weather —
# it only sees historical recruiting and returning production.
st.info(
    "**About this model** \n\n"
    "Model design, analysis, and interpretation by **Marcos Cisneros**. "
    "Built with assistance from Claude (Anthropic) for code structure and debugging. \n\n"
    "Trained on 1,955 Power 4 regular season games (2014–2020). "
    "Test set accuracy: **64.1%** | AUC-ROC: **0.70**. "
    "For comparison, Vegas lines achieve ~71% accuracy on the same games. "
    "This model uses only recruiting history and returning production — it does not account for "
    "injuries, coaching, weather, or in-season performance. Treat predictions as a baseline, not a betting guide."
)

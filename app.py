import streamlit as st
import numpy as np
import pandas as pd
import pickle
import base64
import pathlib  # For robust file paths

# ============================================================
# LOAD MODELS + ENCODERS
# ============================================================

with open("model.pkl", "rb") as f:
    ball_model = pickle.load(f)

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

with open("X_test.pkl", "rb") as f:
    X_test = pickle.load(f)

with open("target_col.pkl", "rb") as f:
    target_col = pickle.load(f)

with open("win_model.pkl", "rb") as f:
    win_model = pickle.load(f)

with open("win_features.pkl", "rb") as f:
    win_features = pickle.load(f)

with open("win_scaler.pkl", "rb") as f:
    win_scaler = pickle.load(f)

target_classes = encoders[target_col].classes_

# --- Absolute path for team logos ---
BASE_DIR = pathlib.Path(__file__).parent
LOGO_DIR = BASE_DIR / "/Users/arpit/Desktop/IPL/logos"

# ============================================================
# CITY → VENUE MAPPING (AUTO SELECTION)
# ============================================================

decoded = X_test.copy()
decoded["venue"] = encoders["venue"].inverse_transform(decoded["venue"])
decoded["city"] = encoders["city"].inverse_transform(decoded["city"])

city_to_venue = (
    decoded.groupby("city")["venue"]
    .agg(lambda x: x.mode()[0])
    .to_dict()
)

cities = sorted(list(encoders["city"].classes_))
venues = sorted(list(encoders["venue"].classes_))
teams = sorted(list(encoders["batting_team"].classes_))

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def safe_encode(col, value):
    classes = list(encoders[col].classes_)
    if value not in classes:
        value = classes[0]
    return encoders[col].transform([value])[0]


TEAM_LOGO_MAP = {
    # Current Teams
    "Chennai Super Kings": "CSK",
    "Deccan Chargers": "DeccanCharger",
    "Delhi Capitals": "DelhiCapitals",
    "Delhi Daredevils": "DelhiDaredevils",
    "Gujarat Titans": "GT",
    "Gujarat Lions": "GujratLions",
    "Kolkata Knight Riders": "KKR",
    "Lucknow Super Giants": "LSG",
    "Mumbai Indians": "MI",
    "Punjab Kings": "PunjabKings",
    "Royal Challengers Bangalore": "RCB",
    "Rajasthan Royals": "RR",
    "Sunrisers Hyderabad": "SRH",

    # Old / Defunct
    "Kings XI Punjab": "KingsXIPunjab",
    "Kochi Tuskers Kerala": "KochiTuskers",
    "Pune Warriors": "PuneWarriors",
    "Rising Pune Supergiant": "RPS",
    "Rising Pune Supergiants": "RPS"
}


def load_team_logo(team):
    filename = TEAM_LOGO_MAP.get(team, team)
    logo_path = LOGO_DIR / f"{filename}.png"
    try:
        with open(logo_path, "rb") as f:
            return f"data:image/png;base64,{base64.b64encode(f.read()).decode()}"
    except:
        # simple transparent placeholder
        return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="


def get_players(team, role="batsman", role_col="batting_team"):
    """
    Get list of players based on their team and role (batsman or bowler).
    """
    try:
        team_id = encoders[role_col].transform([team])[0]
    except:
        # Return all players if team not found
        return sorted(encoders[role].classes_)

    # Filter X_test for matches involving that team in that role
    df = X_test[X_test[role_col] == team_id]

    if df.empty:
        # Return all players if no data for that team
        return sorted(encoders[role].classes_)

    # Get unique players for that role
    players = encoders[role].inverse_transform(df[role].unique())
    return sorted(list(set(players)))


def build_input(inning, over, ball, runs, wkts,
                batting_team, bowling_team, batsman, bowler,
                venue, city):
    balls_before = (over - 1) * 6 + ball - 1
    rr = runs * 6 / balls_before if balls_before > 0 else 0

    return pd.DataFrame([{
        "inning": inning,
        "over": over,
        "ball": ball,
        "runs_before": runs,
        "wkts_before": wkts,
        "balls_before": balls_before,
        "run_rate_before": rr,
        "is_powerplay": 1 if over <= 6 else 0,
        "is_middle": 1 if 7 <= over <= 15 else 0,
        "is_death": 1 if over >= 16 else 0,
        "season_int": 2024,  # Using a recent season
        "batting_team": safe_encode("batting_team", batting_team),
        "bowling_team": safe_encode("bowling_team", bowling_team),
        "batsman": safe_encode("batsman", batsman),
        "bowler": safe_encode("bowler", bowler),
        "venue": safe_encode("venue", venue),
        "city": safe_encode("city", city)
    }])


# ============================================================
# HELPER FOR WIN PROBABILITY MODEL (11 FEATURES)
# ============================================================

def build_win_row_from_state(
    runs, wkts, balls_before,
    inning, target,
    is_powerplay, is_middle, is_death
):
    """
    Build the 11-feature row expected by win_model / win_scaler
    using the current match state (automatic, Option B).
    Features (in order):
    1. runs_before
    2. wickets_in_hand
    3. run_rate
    4. runs_left
    5. balls_left
    6. rr_required
    7. is_powerplay
    8. is_middle
    9. is_death
    10. runs_last_12_balls
    11. wickets_last_12_balls
    """

    if inning != 2 or target is None:
        return None

    runs_before = runs
    # model uses wickets_in_hand, so convert from fallen wickets
    wickets_in_hand = max(0, 10 - wkts)

    # overall run rate so far
    run_rate = runs * 6 / balls_before if balls_before > 0 else 0

    runs_left = max(0, target - runs)
    balls_left = max(0, 120 - balls_before)
    rr_required = runs_left * 6 / balls_left if balls_left > 0 else 0

    # --- Approximate last 12 balls stats from overall rates ---
    recent_balls = min(12, balls_before)
    if recent_balls == 0 or balls_before == 0:
        runs_last_12 = 0
        wkts_last_12 = 0
    else:
        runs_per_ball = runs / balls_before
        wkts_per_ball = wkts / balls_before

        runs_last_12 = int(round(runs_per_ball * recent_balls))
        wkts_last_12 = int(round(wkts_per_ball * recent_balls))

        # Cap so we don't exceed totals
        runs_last_12 = min(runs_last_12, runs)
        wkts_last_12 = min(wkts_last_12, wkts)

    row = np.array([
        runs_before,
        wickets_in_hand,
        run_rate,
        runs_left,
        balls_left,
        rr_required,
        int(is_powerplay),
        int(is_middle),
        int(is_death),
        runs_last_12,
        wkts_last_12
    ], dtype=float).reshape(1, -1)

    return row


# ============================================================
# STREAMLIT PAGE CONFIG
# ============================================================

st.set_page_config(page_title="IPL Prediction Dashboard", layout="wide")

st.markdown("""
<style>
    button[data-baseweb="tab"] {
        font-size: 17px !important;
        font-weight: 600 !important;
        padding-top: 10px !important;
        padding-bottom: 10px !important;
        padding-left: 20px !important;
        padding-right: 20px !important;
    }

    h1 {
        text-align:center;
        color:#ffffff;
        font-size:50px;
    }
    .big-score {
        background: rgba(255,255,255,0.08);
        padding: 20px;
        border-radius: 20px;
        text-align: left;
        color:#fff;
        font-size: 28px;
        border: 1px solid rgba(255,255,255,0.2);
        display: flex;
        align-items: center;
        justify-content: start;
    }
    .metric-card {
        background: rgba(255,255,255,0.09);
        padding: 15px;
        border-radius: 15px;
        text-align:center;
        color:#fff;
        font-size:18px;
        border: 1px solid rgba(255,255,255,0.15);
    }
    .team-logo-score {
        width: 60px;
        height: 60px;
        margin-right: 15px;
        vertical-align: middle;
    }
    .score-text {
        vertical-align: middle;
        line-height: 1.2;
    }
    .ball-outcome {
        display: inline-block;
        width: 50px;
        height: 50px;
        line-height: 50px;
        text-align: center;
        border-radius: 50%;
        color: white;
        font-weight: bold;
        margin: 4px;
        font-size: 18px;
        border: 1px solid rgba(255,255,255,0.3);
    }
    .ball-outcome-run { background-color: #007bff; }
    .ball-outcome-four { background-color: #28a745; }
    .ball-outcome-six { background-color: #17a2b8; }
    .ball-outcome-wicket { background-color: #dc3545; }
    .ball-outcome-dot { background-color: #6c757d; }
</style>
""", unsafe_allow_html=True)

st.markdown(
    "<h1>IPL Prediction Dashboard</h1>",
    unsafe_allow_html=True
)

# ============================================================
# SIDEBAR INPUTS
# ============================================================

st.sidebar.header("Match Setup")

batting_team = st.sidebar.selectbox("Batting Team", teams, index=0)
bowling_team = st.sidebar.selectbox("Bowling Team", teams, index=1)

if batting_team == bowling_team:
    st.sidebar.error("Batting and Bowling teams cannot be the same.")
    st.stop()

bat_logo = load_team_logo(batting_team)
bowl_logo = load_team_logo(bowling_team)

colL, colR = st.sidebar.columns(2)
if bat_logo:
    colL.image(bat_logo, width=80)
if bowl_logo:
    colR.image(bowl_logo, width=80)

inning = st.sidebar.radio("Innings", [1, 2], horizontal=True, index=0)

city = st.sidebar.selectbox("City", cities)
default_venue = city_to_venue.get(city, venues[0])
venue = st.sidebar.selectbox(
    "Venue", venues,
    index=venues.index(default_venue) if default_venue in venues else 0
)

st.sidebar.markdown("---")
st.sidebar.subheader("Current Score")
col_runs, col_wkts = st.sidebar.columns(2)
runs = col_runs.number_input("Runs", min_value=0, value=60)
wkts = col_wkts.number_input("Wickets", min_value=0, max_value=10, value=2)

col_ov, col_ball = st.sidebar.columns(2)
over = col_ov.number_input("Current Over", min_value=1, max_value=20, value=10)
ball = col_ball.number_input("Current Ball", min_value=1, max_value=6, value=1)

balls_before = (over - 1) * 6 + ball - 1
run_rate_before = runs * 6 / balls_before if balls_before > 0 else 0

if inning == 2:
    target = st.sidebar.number_input("Target", min_value=1, value=150)
else:
    target = None

st.sidebar.markdown("---")
st.sidebar.subheader("Current Players")
batsmen = get_players(batting_team, role="batsman", role_col="batting_team")
bowlers = get_players(bowling_team, role="bowler", role_col="bowling_team")
batsman = st.sidebar.selectbox("Batsman", batsmen)
bowler = st.sidebar.selectbox("Bowler", bowlers)

is_powerplay = 1 if over <= 6 else 0
is_middle = 1 if 7 <= over <= 15 else 0
is_death = 1 if over >= 16 else 0

# ============================================================
# SCORECARD TOP
# ============================================================

st.markdown("## Match Situation")

c1, c2, c3 = st.columns([2, 1, 1])

completed_overs = balls_before // 6
completed_balls = balls_before % 6
display_overs = f"{completed_overs}.{completed_balls}"

score_html = f"""
<div class='big-score'>
    <img src='{bat_logo}' class='team-logo-score'>
    <span class='score-text'>
        <b style='font-size: 26px;'>{batting_team}</b><br>
        <span style='font-size: 36px; font-weight:bold;'>{runs} / {wkts}</span>
        <span style='font-size: 20px; color: #ddd;'>({display_overs} Overs)</span>
    </span>
</div>
"""
c1.markdown(score_html, unsafe_allow_html=True)

c2.markdown(
    f"<div class='metric-card'><b>Current RR</b><br>"
    f"<span style='font-size: 30px; font-weight:bold;'>{run_rate_before:.2f}</span></div>",
    unsafe_allow_html=True
)

if inning == 2 and target:
    runs_left = target - runs
    balls_left = 120 - balls_before
    rr_req = (runs_left * 6 / balls_left) if balls_left > 0 else 0
    c3.markdown(
        f"<div class='metric-card'><b>Required RR</b><br>"
        f"<span style='font-size: 30px; font-weight:bold;'>{rr_req:.2f}</span></div>",
        unsafe_allow_html=True
    )
else:
    phase = "Powerplay" if is_powerplay else "Middle Overs" if is_middle else "Death Overs"
    c3.markdown(
        f"<div class='metric-card'><b>Phase</b><br>"
        f"<span style='font-size: 30px; font-weight:bold;'>{phase}</span></div>",
        unsafe_allow_html=True
    )

st.markdown("---")

# ============================================================
# TABS: NEXT BALL • WIN PROB • OVER SIM • PROJECTION
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(
    ["Next Ball", "Win Probability", "Simulate Over", "Score Projection"]
)

# ============================================================
# TAB 1 — NEXT BALL
# ============================================================

with tab1:
    st.subheader("Next Ball Outcome Prediction")

    if st.button("Predict Next Ball Outcome"):
        if wkts == 10 or over > 20:
            st.error("Innings is over.")
        else:
            row = build_input(
                inning, over, ball, runs, wkts,
                batting_team, bowling_team, batsman, bowler, venue, city
            )
            probs = ball_model.predict_proba(row)[0]

            df = pd.DataFrame({
                "Outcome": target_classes,
                "Probability": probs
            }).sort_values("Probability", ascending=False)

            df["Probability"] = df["Probability"] * 100

            st.bar_chart(df.set_index("Outcome"), y="Probability")

            top = df.iloc[0]
            st.success(f"Most likely outcome: **{top.Outcome}** ({top.Probability:.1f}%)")

# ============================================================
# TAB 2 — WIN PROBABILITY
# ============================================================

with tab2:
    st.subheader("Win Probability (2nd Innings Only)")

    st.markdown(
        "Win probability uses the current score, target, phase of the innings, "
        "and an approximate summary of the last 12 balls (auto-computed)."
    )

    if st.button("Calculate Win Probability"):
        if inning == 1:
            st.warning("Win Probability is only available for the **2nd innings chase**.")
        elif target is None:
            st.error("Please set a Target in the sidebar for 2nd innings.")
        else:
            row = build_win_row_from_state(
                runs=runs,
                wkts=wkts,
                balls_before=balls_before,
                inning=inning,
                target=target,
                is_powerplay=is_powerplay,
                is_middle=is_middle,
                is_death=is_death
            )

            if row is None:
                st.error("Win Probability is only defined for 2nd innings with a target.")
            else:
                # Scale using the same scaler used during training
                row_scaled = win_scaler.transform(row)

                # Predict probability of class 1 (win)
                wp = win_model.predict_proba(row_scaled)[0][1] * 100
                lp = 100 - wp

                col_win, col_lose = st.columns(2)
                col_win.markdown(f"""
                <div class='metric-card' style='border-color: #28a745;'>
                    <b>{batting_team} (Win)</b><br>
                    <span style='font-size: 36px; font-weight:bold; color:#28a745;'>{wp:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)

                col_lose.markdown(f"""
                <div class='metric-card' style='border-color: #dc3545;'>
                    <b>{bowling_team} (Win)</b><br>
                    <span style='font-size: 36px; font-weight:bold; color:#dc3545;'>{lp:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)

# ============================================================
# TAB 3 — SIX-BALL SIMULATION
# ============================================================

with tab3:
    st.subheader("Simulate Next Over ")

    if st.button("Simulate Next Over"):
        if wkts == 10 or over > 20:
            st.error("Innings is over.")
        else:
            cur_o, cur_b = over, ball
            cur_r, cur_w = runs, wkts
            seq = []

            balls_to_sim = 7 - cur_b
            if over == 20:
                balls_to_sim = min(balls_to_sim, 7 - cur_b)

            for _ in range(balls_to_sim):
                if cur_w == 10 or cur_o > 20:
                    break

                row = build_input(
                    inning, cur_o, cur_b, cur_r, cur_w,
                    batting_team, bowling_team, batsman, bowler, venue, city
                )

                probs = ball_model.predict_proba(row)[0]
                outcome = np.random.choice(target_classes, p=probs)
                seq.append(outcome)

                if outcome.isdigit():
                    cur_r += int(outcome)
                elif outcome.upper().startswith("W"):
                    cur_w += 1

                cur_b += 1
                if cur_b > 6:
                    cur_b = 1
                    cur_o += 1

            st.markdown("<h4>Predicted Over Sequence:</h4>", unsafe_allow_html=True)
            ball_html = ""
            for outcome in seq:
                css_class = "ball-outcome"
                if outcome.upper().startswith("W"):
                    css_class += " ball-outcome-wicket"
                elif outcome == '6':
                    css_class += " ball-outcome-six"
                elif outcome == '4':
                    css_class += " ball-outcome-four"
                elif outcome == '0':
                    css_class += " ball-outcome-dot"
                elif outcome.isdigit():
                    css_class += " ball-outcome-run"

                ball_html += f"<div class='{css_class}'>{outcome}</div>"

            st.markdown(f"<div>{ball_html}</div>", unsafe_allow_html=True)

            sim_runs = sum(int(o) for o in seq if o.isdigit())
            sim_wkts = sum(1 for o in seq if o.upper().startswith("W"))

            st.info(f"Projected over result: **{sim_runs} runs** for **{sim_wkts} wickets**.")
            st.markdown(f"**Score after {balls_to_sim} balls: {cur_r}/{cur_w}**")

# ============================================================
# TAB 4 — SCORE PROJECTION
# ============================================================

with tab4:
    st.subheader("Project Score")

    sim_balls = st.slider("Balls to simulate:", 6, 60, 30, 6)

    if st.button(f"Project Next {sim_balls} Balls"):
        if wkts == 10 or over > 20:
            st.error("Innings is over.")
        else:
            temp_o, temp_b = over, ball
            temp_r, temp_w = runs, wkts
            proj_runs = 0

            for _ in range(sim_balls):
                if temp_w == 10 or temp_o > 20:
                    break

                row = build_input(
                    inning, temp_o, temp_b, temp_r, temp_w,
                    batting_team, bowling_team, batsman, bowler, venue, city
                )

                probs = ball_model.predict_proba(row)[0]
                out = np.random.choice(target_classes, p=probs)

                if out.isdigit():
                    runs_val = int(out)
                    temp_r += runs_val
                    proj_runs += runs_val
                elif out.upper().startswith("W"):
                    temp_w += 1

                temp_b += 1
                if temp_b > 6:
                    temp_b = 1
                    temp_o += 1

            st.info(f"Projected runs in next {sim_balls} balls: **{proj_runs}**")

            final_runs = temp_r
            final_wkts = temp_w

            final_balls_played = balls_before + sim_balls
            if temp_o > 20:
                final_overs_display = "20.0"
            else:
                final_ov = final_balls_played // 6
                final_b = final_balls_played % 6
                final_overs_display = f"{final_ov}.{final_b}"

            st.markdown(f"""
            <div class='metric-card' style='border-color: #ffc107;'>
                <b>Projected Score</b><br>
                <span style='font-size: 44px; font-weight:bold; color:#ffc107;'>
                    {final_runs} / {final_wkts}
                </span>
                <br>
                (After ~{final_overs_display} Overs)
            </div>
            """, unsafe_allow_html=True)

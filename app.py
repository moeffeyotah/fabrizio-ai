import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import plotly.graph_objects as go

# 1. Page Configuration
st.set_page_config(
    page_title="FABRIZIO AI 3.3.3", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# 2. Fabrizio Brand Styling 
st.markdown(
    """
    <style>
    /* Main Background and Text */
    .stApp { background-color: #0B0B0B; color: #E0E0E0; font-family: 'Inter', sans-serif; }
    
    /* Branding Colors */
    .brand-fab { color: #00FF87; font-weight: 900; letter-spacing: 2px; }
    .brand-ai { color: #FFD700; font-weight: 300; }
    
    /* Glassmorphism Containers */
    [data-testid="stVerticalBlock"] > div:has(div.element-container) {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(0, 255, 135, 0.2);
        border-radius: 15px;
        padding: 25px;
    }
    
    /* The "Here We Go" Button */
    .stButton>button {
        background-color: #00FF87 !important;
        color: #0B0B0B !important; 
        border: none !important;
        border-radius: 5px;
        font-weight: 900;
        width: 100%;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #FFD700 !important;
        transform: scale(1.02);
    }
    
    /* Slider Color Theme */
    div[data-baseweb="slider"] > div > div { background-color: #FFD700 !important; }
    
    header {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def load_assets():
    # Adding error handling for missing files
    try:
        custom_objects = {"mse": tf.keras.losses.MeanSquaredError()}
        model = tf.keras.models.load_model("champion_model.h5", custom_objects=custom_objects)
        scaler = joblib.load("scaler.pkl")
        data = pd.read_csv("player_stats.csv", encoding="latin1")
        data.columns = data.columns.str.strip().str.lower()
        data = data.fillna(0)
        return model, scaler, data
    except Exception as e:
        st.error(f"Asset Load Error: Ensure champion_model.h5, scaler.pkl, and player_stats.csv are in the directory.")
        return None, None, None

model, scaler, df = load_assets()

# --- TOP NAVIGATION ---
st.markdown(
    "<h1 style='text-align: center; margin-bottom: 0;'><span class='brand-fab'>FABRIZIO</span> <span class='brand-ai'>AI</span></h1>",
    unsafe_allow_html=True,
)
st.markdown("<p style='text-align: center; color: #666; font-size: 0.8em;'>PREDICTIVE SCOUTING ENGINE v3.0</p>", unsafe_allow_html=True)

_, search_col, _ = st.columns([1, 2, 1])
with search_col:
    selected_player = st.selectbox(
        "SEARCH TRANSFER MARKET", ["Search Player..."] + list(df["player"].unique()), key="fab_search"
    )

# --- DYNAMIC INITIALIZATION ---
if selected_player != "Search Player...":
    p_data = df[df["player"] == selected_player].iloc[0]
    display_name = selected_player.upper()
    init_vals = [
        (p_data["sprint_speed"] + p_data["acceleration"]) / 2,
        (p_data["finishing"] + p_data["shot_power"]) / 2,
        (p_data["short_pass"] + p_data["vision"]) / 2,
        (p_data["dribbling"] + p_data["ball_control"]) / 2,
        (p_data["stand_tackle"] + p_data["interceptions"]) / 2,
        (p_data["strength"] + p_data["stamina"]) / 2,
    ]
else:
    display_name = "MOSES EFFEYOTAH"
    init_vals = [45.0] * 6

# --- UI LAYOUT ---
col_dna, col_report = st.columns([1, 1.2], gap="large")

with col_dna:
    st.markdown("### 🧬 <span style='color:#FFD700;'>PLAYER STATS</span>", unsafe_allow_html=True)

    s1 = st.slider("PACE", 0, 100, int(init_vals[0]))
    s2 = st.slider("SHOOTING", 0, 100, int(init_vals[1]))
    s3 = st.slider("PASSING", 0, 100, int(init_vals[2]))
    s4 = st.slider("DRIBBLING", 0, 100, int(init_vals[3]))
    s5 = st.slider("DEFENDING", 0, 100, int(init_vals[4]))
    s6 = st.slider("PHYSICALITY", 0, 100, int(init_vals[5]))

    fig = go.Figure(
        go.Scatterpolar(
            r=[s1, s2, s3, s4, s5, s6],
            theta=["PACE", "SHOOTING", "PASSING", "DRIBBLING", "DEFENDING", "PHYSICALITY"],
            fill="toself",
            fillcolor="rgba(0, 255, 135, 0.15)",
            line=dict(color="#00FF87", width=3),
        )
    )
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 100], color="#555", gridcolor="#333"),
            angularaxis=dict(color="#888", gridcolor="#333")
        ),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=40, b=40, l=40, r=40),
    )
    st.plotly_chart(fig, use_container_width=True)

with col_report:
    st.markdown("### 📡 <span style='color:#FF3131;'>AI SCOUT VERDICT</span>", unsafe_allow_html=True)
    threshold = st.slider("TARGET SENSITIVITY", 0.0, 1.0, 0.85)

    with st.container():
        rep_col1, rep_col2 = st.columns([1, 1.5])
        with rep_col1:
            # Using your yellow/red palette for the scout icon if possible
            st.image("https://cdn-icons-png.flaticon.com/512/2591/2591458.png", width=140)
            st.markdown("<p style='text-align: center; color: #00FF87; font-size: 0.8em;'>LINK STABLE</p>", unsafe_allow_html=True)

        with rep_col2:
            st.markdown(f"<h2 style='margin-bottom:0;'>{display_name}</h2>", unsafe_allow_html=True)
            if st.button("📢 HERE WE GO!"):
                if selected_player != "Search Player...":
                    full_row = df[df["player"] == selected_player].iloc[0].copy()
                else:
                    full_row = df.quantile(0.25, numeric_only=True).copy()
                    full_row["age"] = 18

                # Map sliders to model input
                full_row["acceleration"], full_row["sprint_speed"] = s1, s1
                full_row["finishing"], full_row["shot_power"] = s2, s2
                full_row["short_pass"], full_row["vision"] = s3, s3
                full_row["dribbling"], full_row["ball_control"] = s4, s4
                full_row["stand_tackle"], full_row["interceptions"] = s5, s5
                full_row["strength"], full_row["stamina"] = s6, s6

                feat = full_row.drop(["player", "value", "country_code", "club_code"], errors="ignore")
                input_raw = pd.to_numeric(feat, errors="coerce").fillna(0).values.reshape(1, -1)
                st.write(f"Scaler expects {scaler.n_features_in_} features.")
                st.write(f"You provided {input_raw.shape[1]} features.")
                val_pred, cls_pred = model.predict(scaler.transform(input_raw))
                prob = float(cls_pred[0][0])

                st.markdown("---")
                status = "ELITE TARGET 🌟" if prob > threshold else "PROSPECT ASSET ⚽"
                status_color = "#00FF87" if prob > threshold else "#FFD700"

                st.markdown(f"<h3 style='color:{status_color}; text-align: center;'>{status}</h3>", unsafe_allow_html=True)
                
                final_val = val_pred[0][0] if s1 > 20 else val_pred[0][0] * 0.1
                st.metric("PREDICTED MARKET VALUE", f"${final_val:.2f}M")
                
                st.write(f"Neural Confidence: {prob*100:.1f}%")
                st.progress(prob)

# --- FOOTER BAR ---
st.markdown("<br><hr style='border: 1px solid #333;'>", unsafe_allow_html=True)
f1, f2, f3 = st.columns(3)
with f1: st.button("📑 DOSSIER")
with f2: st.button("🔄 COMPARE")
with f3: st.button("🗞️ LIVE FEED")

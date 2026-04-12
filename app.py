import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import plotly.graph_objects as go
import os

# 1. Page Configuration
st.set_page_config(
    page_title="FABRIZIO AI | Lab", layout="wide", initial_sidebar_state="collapsed"
)

# 2. Obsidian Lab Styling (Improved Contrast)
st.markdown(
    """
    <style>
    .stApp { background-color: #0B0B0B; color: #F1F5F9; font-family: 'Inter', sans-serif; }
    .brand-fab { color: #00FF87; font-weight: 900; }
    .brand-ai { color: #FF3131; font-weight: 300; }
    
    /* Make sliders and text readable against dark background */
    .stSlider label, .stSelectbox label { color: #FFFFFF !important; font-weight: bold; }
    
    [data-testid="stVerticalBlock"] > div:has(div.element-container) {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 25px;
    }
    .stButton>button {
        background-color: #1A1A1A !important;
        color: #00FF87 !important; 
        border: 2px solid #00FF87 !important;
        width: 100%;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #00FF87 !important;
        color: #000000 !important;
    }
    header {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def load_assets():
    # Fix: Use relative pathing and check if files exist to prevent crash
    model_path = "champion_model.h5"
    scaler_path = "scaler.pkl"
    csv_path = "player_stats.csv"
    
    if not os.path.exists(model_path):
        st.error(f"FATAL ERROR: {model_path} not found in repository.")
        st.stop()

    # Load with custom objects to handle specific loss functions
    custom_objects = {"mse": tf.keras.losses.MeanSquaredError()}
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    
    scaler = joblib.load(scaler_path)
    
    # Load Data
    data = pd.read_csv(csv_path, encoding="latin1")
    data.columns = data.columns.str.strip().str.lower()
    
    # Pre-process categorization
    data["country_code"] = data["country"].astype("category").cat.codes
    data["club_code"] = data["club"].astype("category").cat.codes
    data = data.fillna(0)
    
    return model, scaler, data

# Safe Asset Loading
try:
    model, scaler, df = load_assets()
except Exception as e:
    st.error(f"Initialization Failed: {e}")
    st.stop()

# --- TOP NAVIGATION ---
st.markdown(
    """
    <div style='text-align: center; margin-bottom: 20px;'>
        <h1 style='margin-bottom: 0;'><span class='brand-fab'>FABRIZIO</span> <span class='brand-ai'>AI</span></h1>
        <p style='color: #00FF87; font-weight: 800; letter-spacing: 2px; font-size: 0.9rem; margin-top: 0;'>
            DESIGNED & ENGINEERED BY MOSES MUDIAGA EFFEYOTAH
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

_, search_col, _ = st.columns([1, 2, 1])
with search_col:
    selected_player = st.selectbox(
        "SEARCH DATABASE", ["Search Player..."] + list(df["player"].unique()), key="fab_search"
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
    display_name = "NEW PROSPECT"
    init_vals = [45.0] * 6

# --- UI LAYOUT ---
col_dna, col_report = st.columns([1, 1.2], gap="large")

with col_dna:
    st.markdown("### 🧬 <span style='color:#00FF87;'>DNA VISUALIZATION</span>", unsafe_allow_html=True)

    s1 = st.slider("PACE", 0, 100, int(init_vals[0]))
    s2 = st.slider("SHOOTING", 0, 100, int(init_vals[1]))
    s3 = st.slider("PASSING", 0, 100, int(init_vals[2]))
    s4 = st.slider("DRIBBLING", 0, 100, int(init_vals[3]))
    s5 = st.slider("DEFENDING", 0, 100, int(init_vals[4]))
    s6 = st.slider("PHYSICALITY", 0, 100, int(init_vals[5]))

    fig = go.Figure(
        go.Scatterpolar(
            r=[s1, s2, s3, s4, s5, s6],
            theta=["PACE","SHOOTING","PASSING","DRIBBLING","DEFENDING","PHYSICALITY"],
            fill="toself",
            fillcolor="rgba(0, 255, 135, 0.1)",
            line=dict(color="#00FF87", width=2),
        )
    )
    fig.update_layout(
        polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(visible=True, range=[0, 100], color="#888")),
        showlegend=False, paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=30, b=30),
    )
    st.plotly_chart(fig, use_container_width=True)

with col_report:
    st.markdown("### 📡 <span style='color:#FF3131;'>OFFICIAL REPORT</span>", unsafe_allow_html=True)

    threshold = st.slider("SCOUT SENSITIVITY", 0.0, 1.0, 0.85)

    with st.container():
        rep_col1, rep_col2 = st.columns([1, 1.5])
        with rep_col1:
            st.image("https://cdn-icons-png.flaticon.com/512/2591/2591458.png", width=150)
            st.markdown("<p style='text-align: center; color: #00FF87;'>BIO-LINK READY</p>", unsafe_allow_html=True)

        with rep_col2:
            st.markdown(f"## {display_name}")
            if st.button("📢 CONFIRM: HERE WE GO!"):
         with st.spinner("Analyzing Market DNA..."):
        # Ensure the rest of your logic is also indented here
        if selected_player != "Search Player...":
            full_row = df[df["player"] == selected_player].iloc[0].copy()
        else:
            full_row = df.quantile(0.25, numeric_only=True).copy()
            full_row["age"] = 18

        # Inject slider values
        full_row["acceleration"], full_row["sprint_speed"] = s1, s1
        full_row["finishing"], full_row["shot_power"] = s2, s2
        full_row["short_pass"], full_row["vision"] = s3, s3
        full_row["dribbling"], full_row["ball_control"] = s4, s4
        full_row["stand_tackle"], full_row["interceptions"] = s5, s5
        full_row["strength"], full_row["stamina"] = s6, s6

        # Prepare for model
        feat = full_row.drop(["player", "value", "country", "club"], errors="ignore")
        input_raw = pd.to_numeric(feat, errors="coerce").fillna(0).values.reshape(1, -1)
        scaled_input = scaler.transform(input_raw)

        # PREDICTION FIX: Handling standard single-output models
        predictions = model.predict(scaled_input)
        
        # If your model has 2 outputs (Value, Class), use this:
        if isinstance(predictions, list):
            val_pred = float(predictions[0][0])
            prob = float(predictions[1][0][0])
        else:
            # If it's a single output (just value), we simulate probability based on value
            val_pred = float(predictions[0][0])
            prob = min(val_pred / 100, 1.0) # Example logic

        # --- DISPLAY RESULTS ---
        st.markdown("---")
        status = "ELITE TARGET 🌟" if prob > threshold else "STANDARD ASSET ⚽"
        status_color = "#00FF87" if prob > threshold else "#FF3131"

        st.markdown(f"<h3 style='color:{status_color}; text-align: center;'>{status}</h3>", unsafe_allow_html=True)
        st.metric("ESTIMATED MARKET VALUE", f"${val_pred:.1f}M")
        st.progress(prob)
        st.write(f"Scout Confidence: {prob*100:.1f}%")

# --- FOOTER ---
st.markdown("<br>", unsafe_allow_html=True)
f1, f2, f3 = st.columns(3)
f1.button("📑 DOSSIER")
f2.button("🔄 COMPARE")
f3.button("🗞️ FEED")

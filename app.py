import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import plotly.graph_objects as go
import google.generativeai as genai

# 1. Page Configuration
st.set_page_config(
    page_title="FABRIZIO AI 3.3.3", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# 2. LLM Setup
genai.configure(api_key="AIzaSyDWVxRdsnuBeVMGzdqkVRypVrcmX97nBWo")
llm_model = genai.GenerativeModel('gemini-pro')

# 3. Fabrizio Brand Styling 
st.markdown(
    """
    <style>
    .stApp { background-color: #0B0B0B; color: #E0E0E0; font-family: 'Inter', sans-serif; }
    
    /* Branding Colors: Fabrizio = Gold, AI = Red */
    .brand-fab { color: #FFD700; font-weight: 900; letter-spacing: 2px; }
    .brand-ai { color: #FF3131; font-weight: 900; }
    
    [data-testid="stVerticalBlock"] > div:has(div.element-container) {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 215, 0, 0.2);
        border-radius: 15px;
        padding: 25px;
    }
    
    .stButton>button {
        background-color: #FFD700 !important;
        color: #0B0B0B !important; 
        border: none !important;
        border-radius: 5px;
        font-weight: 900;
        width: 100%;
    }
    
    div[data-baseweb="slider"] > div > div { background-color: #FFD700 !important; }
    
    header {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

def get_smart_analysis(name, val, prob, status):
    prompt = f"""
    Act as Fabrizio Romano, the football transfer expert. 
    Analyze this AI Scout Report:
    Player: {name}
    Neural Value: ${val:.2f}M
    Elite Confidence: {prob*100:.1f}%
    Status: {status}
    
    Write a punchy, 2-sentence 'Here We Go' style update for a transfer dossier. 
    Be excited if it's Elite, and professional if it's a Prospect.
    """
    try:
        response = llm_model.generate_content(prompt)
        return response.text
    except:
        return "📡 Connection to Scouting Network established. Analysis pending..."

@st.cache_resource
def load_assets():
    try:
        custom_objects = {"mse": tf.keras.losses.MeanSquaredError()}
        model = tf.keras.models.load_model("champion_model.h5", custom_objects=custom_objects)
        scaler = joblib.load("scaler.pkl")
        data = pd.read_csv("player_stats.csv", encoding="latin1")
        data.columns = data.columns.str.strip().str.lower()
        data = data.fillna(0)
        return model, scaler, data
    except:
        st.error("Missing model files.")
        return None, None, None

model, scaler, df = load_assets()

# --- TOP NAVIGATION ---
st.markdown("<h1 style='text-align: center; margin-bottom: 0;'><span class='brand-fab'>FABRIZIO</span> <span class='brand-ai'>AI</span></h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666; font-size: 0.8em;'>INTELLIGENT NEURAL SCOUTING v3.3.3</p>", unsafe_allow_html=True)

_, search_col, _ = st.columns([1, 2, 1])
with search_col:
    selected_player = st.selectbox("SEARCH MARKET", ["Search Player..."] + list(df["player"].unique()), key="fab_search")

if selected_player != "Search Player...":
    p_data = df[df["player"] == selected_player].iloc[0]
    display_name = selected_player.upper()
    init_vals = [
        (p_data["sprint_speed"] + p_data["acceleration"]) / 2, 
        (p_data["finishing"] + p_data["shot_power"]) / 2, 
        (p_data["short_pass"] + p_data["vision"]) / 2, 
        (p_data["dribbling"] + p_data["ball_control"]) / 2, 
        (p_data["stand_tackle"] + p_data["interceptions"]) / 2, 
        (p_data["strength"] + p_data["stamina"]) / 2
    ]
else:
    display_name = "MOSES EFFEYOTAH"
    init_vals = [45.0] * 6

col_dna, col_report = st.columns([1, 1.2], gap="large")

with col_dna:
    st.markdown("### 🧬 <span style='color:#FFD700;'>PLAYER DNA</span>", unsafe_allow_html=True)
    s1 = st.slider("PACE", 0, 100, int(init_vals[0]))
    s2 = st.slider("SHOOTING", 0, 100, int(init_vals[1]))
    s3 = st.slider("PASSING", 0, 100, int(init_vals[2]))
    s4 = st.slider("DRIBBLING", 0, 100, int(init_vals[3]))
    s5 = st.slider("DEFENDING", 0, 100, int(init_vals[4]))
    s6 = st.slider("PHYSICALITY", 0, 100, int(init_vals[5]))

    fig = go.Figure(go.Scatterpolar(
        r=[s1, s2, s3, s4, s5, s6],
        theta=["PACE", "SHOOTING", "PASSING", "DRIBBLING", "DEFENDING", "PHYSICALITY"],
        fill="toself", fillcolor="rgba(255, 215, 0, 0.1)", line=dict(color="#FFD700", width=3)
    ))
    fig.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(visible=True, range=[0, 100], color="#555")), showlegend=False, paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width='stretch')

with col_report:
    st.markdown("### 📡 <span style='color:#FF3131;'>NEURAL VERDICT</span>", unsafe_allow_html=True)
    threshold = st.slider("SCOUT SENSITIVITY", 0.0, 1.0, 0.85)

    if st.button("📢 ANALYZE: HERE WE GO!"):
        if selected_player != "Search Player...":
            full_row = df[df["player"] == selected_player].iloc[0].copy()
        else:
            full_row = df.quantile(0.25, numeric_only=True).copy()
            full_row["age"] = 18

        # Inject Slider Stats
        slider_map = [s1, s2, s3, s4, s5, s6]
        attr_groups = [
            ["acceleration", "sprint_speed"], ["finishing", "shot_power"], 
            ["short_pass", "vision"], ["dribbling", "ball_control"], 
            ["stand_tackle", "interceptions"], ["strength", "stamina"]
        ]
        
        for idx, attr_list in enumerate(attr_groups):
            for col_name in attr_list:
                if col_name in full_row: 
                    full_row[col_name] = slider_map[idx]

        # Auto-Align features
        expected_features = scaler.feature_names_in_
        for col in expected_features:
            if col not in full_row: 
                full_row[col] = 0
        
        feat = full_row[expected_features]
        input_raw = pd.to_numeric(feat, errors="coerce").fillna(0).values.reshape(1, -1)
        
        # Scale and Predict
        input_scaled = scaler.transform(input_raw)
        val_pred, cls_pred = model.predict(input_scaled)
        prob = float(cls_pred[0][0])

        st.markdown("---")
        status = "ELITE TARGET 🌟" if prob > threshold else "PROSPECT ASSET ⚽"
        status_color = "#00FF87" if prob > threshold else "#FFD700"
        final_val = val_pred[0][0] if s1 > 20 else val_pred[0][0] * 0.1

        st.markdown(f"<h3 style='color:{status_color}; text-align: center;'>{status}</h3>", unsafe_allow_html=True)
        st.metric("PREDICTED VALUE", f"${final_val:.2f}M")
        
        with st.expander("🤖 SMART SCOUT ANALYSIS", expanded=True):
            with st.spinner("Consulting LLM..."):
                analysis = get_smart_analysis(display_name, final_val, prob, status)
                st.write(analysis)

        st.progress(prob)

# --- FOOTER BAR ---
st.markdown("<br><hr style='border: 1px solid #333;'>", unsafe_allow_html=True)
f1, f2, f3 = st.columns(3)
with f1: st.button("📑 DOSSIER")
with f2: st.button("🔄 COMPARE")
with f3: st.button("🗞️ LIVE FEED")

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import plotly.graph_objects as go
import os
from groq import Groq # Swapped from Google Gemini to Groq

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="FABRIZIO AI 3.3.3", 
    page_icon="⚽",
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# --- 2. SECURE LLM SETUP (GROQ LPU ENGINE) ---
# Hardcoded as a fallback for immediate use, but try to use st.secrets long-term!
groq_key = st.secrets.get(
    "GROQ_API_KEY", 
    os.getenv("GROQ_API_KEY", "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")
)
client = Groq(api_key=groq_key)

# --- 3. PREMIUM BRAND STYLING (GLASSMORPHISM & GLOW) ---
st.markdown(
    """
    <style>
    /* Main Background */
    .stApp { 
        background: radial-gradient(circle at 50% 0%, #1a1a1a 0%, #050505 100%);
        color: #E0E0E0; 
        font-family: 'Inter', sans-serif; 
    }
    
    /* Typography & Brand Colors */
    .brand-fab { color: #FFD700; font-weight: 900; letter-spacing: 3px; text-shadow: 0px 0px 15px rgba(255, 215, 0, 0.4); }
    .brand-ai { color: #FF3131; font-weight: 900; text-shadow: 0px 0px 15px rgba(255, 49, 49, 0.4); }
    .section-title { font-size: 1.2rem; font-weight: 800; letter-spacing: 1px; margin-bottom: 15px; color: #FFD700; }
    
    /* Glassmorphism Cards */
    [data-testid="stVerticalBlock"] > div:has(div.element-container) {
        background: linear-gradient(145deg, rgba(25, 25, 25, 0.6) 0%, rgba(10, 10, 10, 0.8) 100%);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 215, 0, 0.15);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        border-radius: 20px;
        padding: 25px;
    }
    
    /* Custom Metric Cards */
    .metric-box {
        background: rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-bottom: 15px;
    }
    .metric-value { font-size: 2.5rem; font-weight: 900; margin: 10px 0; }
    
    /* Expander (Drop Down) Hover Fix */
    [data-testid="stExpander"] details summary {
        background-color: rgba(20, 20, 20, 0.6) !important;
        border-radius: 8px;
        transition: background-color 0.3s ease, color 0.3s ease;
    }
    [data-testid="stExpander"] details summary:hover {
        background-color: rgba(45, 45, 45, 0.9) !important; /* Subtle dark gray, no white flash */
        color: #FFD700 !important; /* Text turns gold on hover */
    }
    [data-testid="stExpander"] details summary svg {
        fill: #FFD700 !important; /* Keep the drop-down arrow gold */
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #FFD700 0%, #FDB931 100%) !important;
        color: #0B0B0B !important; 
        border: none !important;
        border-radius: 8px;
        font-weight: 900;
        letter-spacing: 1px;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0px 4px 15px rgba(255, 215, 0, 0.3);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0px 6px 20px rgba(255, 215, 0, 0.5);
    }
    
    /* Sliders */
    div[data-baseweb="slider"] > div > div { background-color: #FFD700 !important; }
    
    /* Signature Footer */
    .signature-footer {
        text-align: center;
        color: #666;
        font-size: 0.85rem;
        letter-spacing: 1px;
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
    }
    .signature-footer strong { color: #FFD700; }
    
    header {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# --- 4. CORE FUNCTIONS (NOW POWERED BY GROQ) ---
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
        # Utilizing Groq's high-speed Llama 3.3 70B engine
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.7, 
            max_tokens=150,  
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"📡 Connection to Groq Network failed. Analysis pending... (Error: {str(e)})"


@st.cache_resource
def load_assets():
    try:
        custom_objects = {"mse": tf.keras.losses.MeanSquaredError()}
        model = tf.keras.models.load_model(
            "champion_model.h5", custom_objects=custom_objects
        )
        scaler = joblib.load("scaler.pkl")
        data = pd.read_csv("player_stats.csv", encoding="latin1")
        data.columns = data.columns.str.strip().str.lower()
        data = data.fillna(0)
        return model, scaler, data
    except Exception as e:
        st.error(f"Missing model files: {str(e)}")
        return None, None, None

model, scaler, df = load_assets()

# --- 5. TOP NAVIGATION & HEADER ---
st.markdown(
    "<h1 style='text-align: center; margin-bottom: -10px; font-size: 3.5rem;'><span class='brand-fab'>FABRIZIO</span> <span class='brand-ai'>AI</span></h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align: center; color: #888; font-size: 0.9em; letter-spacing: 2px; margin-bottom: 20px;'>INTELLIGENT NEURAL SCOUTING v3.3.3</p>",
    unsafe_allow_html=True,
)

# Search Bar
_, search_col, _ = st.columns([1, 2, 1])
with search_col:
    player_list = ["Search Player..."] + list(df["player"].unique()) if df is not None else ["Search Player..."]
    selected_player = st.selectbox(
        "MARKET DATABASE",
        player_list,
        key="fab_search",
        label_visibility="collapsed"
    )

# Logic initialization
if selected_player != "Search Player..." and df is not None:
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
    display_name = "TARGET ACQUISITION"
    init_vals = [50.0] * 6

# --- APP DOCUMENTATION ---
doc_col1, doc_col2 = st.columns(2)
with doc_col1:
    with st.expander("📖 About FABRIZIO AI"):
        st.write("""
        **FABRIZIO AI** is an advanced neural scouting platform designed to evaluate football talent using Deep Learning. 
        
        It utilizes a **TensorFlow Neural Network** to analyze a player's physical and technical DNA, outputting two distinct metrics:
        * **Market Valuation:** A predictive monetary value based on historical stats.
        * **AI Classification:** A binary threshold rating classifying the player as an *Elite Target* or a *Prospect Asset*.
        
        The system is augmented by a **Groq LPU LLM** acting as a "Fabrizio Romano" agent to synthesize the data into actionable transfer intelligence.
        """)
with doc_col2:
    with st.expander("🛠️ Scouting Guide"):
        st.write("""
        1. **Search the Market:** Select a known player from the central dropdown to instantly load their core stats.
        2. **Custom DNA Generation:** Alternatively, ignore the search and manually adjust the DNA sliders to build your ideal transfer target.
        3. **Set Threshold:** Adjust the *Scout Strictness* slider to determine how aggressively the AI filters for Elite players.
        4. **Execute:** Click **'ANALYZE: HERE WE GO!'** to feed the data through the neural network and generate the transfer dossier.
        """)

st.divider()

# --- 6. MAIN DASHBOARD LAYOUT ---
col_dna, col_report = st.columns([1, 1.2], gap="large")

with col_dna:
    st.markdown("<div class='section-title'>🧬 PLAYER DNA OVERVIEW</div>", unsafe_allow_html=True)
    
    # Sliders
    s1 = st.slider("⚡ PACE", 0, 100, int(init_vals[0]))
    s2 = st.slider("🎯 SHOOTING", 0, 100, int(init_vals[1]))
    s3 = st.slider("👁️ PASSING", 0, 100, int(init_vals[2]))
    s4 = st.slider("✨ DRIBBLING", 0, 100, int(init_vals[3]))
    s5 = st.slider("🛡️ DEFENDING", 0, 100, int(init_vals[4]))
    s6 = st.slider("💪 PHYSICALITY", 0, 100, int(init_vals[5]))

    # Upgraded Radar Chart
    fig = go.Figure(
        go.Scatterpolar(
            r=[s1, s2, s3, s4, s5, s6],
            theta=["PACE", "SHOOTING", "PASSING", "DRIBBLING", "DEFENDING", "PHYSICALITY"],
            fill="toself",
            fillcolor="rgba(255, 215, 0, 0.15)",
            line=dict(color="#FFD700", width=4),
            marker=dict(color="#FF3131", size=8)
        )
    )
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 100], color="#444", gridcolor="rgba(255,255,255,0.1)"),
            angularaxis=dict(color="#FFF", gridcolor="rgba(255,255,255,0.1)")
        ),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=30, r=30, t=30, b=30),
    )
    st.plotly_chart(fig, use_container_width=True)

with col_report:
    st.markdown("<div class='section-title'>📡 SCOUTING VERDICT</div>", unsafe_allow_html=True)
    
    threshold = st.slider("SCOUT STRICTNESS THRESHOLD", 0.0, 1.0, 0.85, help="Higher values require higher AI confidence to flag a player as ELITE.")

    if st.button("📢 ANALYZE: HERE WE GO!"):
        if df is not None and model is not None and scaler is not None:
            if selected_player != "Search Player...":
                full_row = df[df["player"] == selected_player].iloc[0].copy()
            else:
                full_row = df.quantile(0.25, numeric_only=True).copy()
                full_row["age"] = 18

            # Inject Slider Stats
            slider_map = [s1, s2, s3, s4, s5, s6]
            attr_groups = [
                ["acceleration", "sprint_speed"],
                ["finishing", "shot_power"],
                ["short_pass", "vision"],
                ["dribbling", "ball_control"],
                ["stand_tackle", "interceptions"],
                ["strength", "stamina"],
            ]

            for idx, attr_list in enumerate(attr_groups):
                for col_name in attr_list:
                    if col_name in full_row:
                        full_row[col_name] = slider_map[idx]

            expected_features = scaler.feature_names_in_
            for col in expected_features:
                if col not in full_row:
                    full_row[col] = 0

            feat = full_row[expected_features]
            input_raw = pd.to_numeric(feat, errors="coerce").fillna(0).values.reshape(1, -1)

            # Predict
            input_scaled = scaler.transform(input_raw)
            val_pred, cls_pred = model.predict(input_scaled)
            prob = float(cls_pred[0][0])

            status = "ELITE TARGET 🌟" if prob > threshold else "PROSPECT ASSET ⚽"
            status_color = "#00FF87" if prob > threshold else "#FFD700"
            final_val = val_pred[0][0] if s1 > 20 else val_pred[0][0] * 0.1

            st.markdown("<br>", unsafe_allow_html=True)
            
            # Rich Metrics Layout
            m1, m2 = st.columns(2)
            with m1:
                st.markdown(f"""
                <div class="metric-box">
                    <div style="color: #888; font-size: 0.9rem; letter-spacing: 1px;">MARKET VALUATION</div>
                    <div class="metric-value" style="color: #FFF;">${final_val:.2f}M</div>
                </div>
                """, unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class="metric-box">
                    <div style="color: #888; font-size: 0.9rem; letter-spacing: 1px;">AI CLASSIFICATION</div>
                    <div class="metric-value" style="color: {status_color}; font-size: 1.8rem;">{status}</div>
                </div>
                """, unsafe_allow_html=True)

            st.progress(prob, text=f"Neural Confidence Match: {prob*100:.1f}%")

            with st.expander("🤖 ROMANO INSIGHTS", expanded=True):
                with st.spinner("Connecting to Fabrizio's Network..."):
                    analysis = get_smart_analysis(display_name, final_val, prob, status)
                    st.info(analysis)
        else:
            st.error("System offline. Please ensure model files are loaded correctly.")

# --- 7. FOOTER & SIGNATURE ---
st.markdown("<br>", unsafe_allow_html=True)
f1, f2, f3, f4, f5 = st.columns([1, 2, 2, 2, 1])
with f2:
    st.button("📑 SAVE DOSSIER")
with f3:
    st.button("🔄 COMPARE TARGETS")
with f4:
    st.button("🗞️ LIVE SCOUT FEED")

# The Signature
st.markdown(
    """
    <div class='signature-footer'>
        Designed and Engineered by <strong>Moses Mudiaga Effeyotah</strong><br>
        School of Information Technology | Fanshawe College
    </div>
    """, 
    unsafe_allow_html=True
)

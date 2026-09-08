"""
KrishiDrishti — Officer Dashboard
SIH 2026 · Problem 26131 · Owner: Khushi (Officer Dashboard)

Aligned to Execution Handbook §08 + Common Integration Contract (§10):
  Overview KPIs · Map with risk markers · Case evidence · Expert actions
  CONFIRM → status VERIFIED + farmer advisory SENT
"""

import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
from datetime import datetime, timedelta
import re
import os

# Cases API client (live backend). Falls back to mock if API is down.
try:
    import api_client
except ImportError:
    api_client = None  # type: ignore

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="KrishiDrishti | Officer Dashboard",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS — polished dark agricultural theme
# ============================================================

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', system-ui, sans-serif;
    }

    .stApp {
        background:
            radial-gradient(ellipse 80% 50% at 90% -10%, rgba(34, 197, 94, 0.18), transparent 50%),
            radial-gradient(ellipse 60% 40% at 5% 100%, rgba(245, 158, 11, 0.10), transparent 45%),
            radial-gradient(circle at 50% 50%, rgba(15, 50, 35, 0.4), transparent 70%),
            linear-gradient(160deg, #040d0a 0%, #0a1f16 40%, #071812 100%);
        color: #f8fafc;
    }
    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        opacity: 0.07;
        background-image:
            linear-gradient(rgba(34,197,94,0.18) 1px, transparent 1px),
            linear-gradient(90deg, rgba(34,197,94,0.18) 1px, transparent 1px);
        background-size: 48px 48px;
        mask-image: linear-gradient(to bottom, black 0%, transparent 70%);
        z-index: 0;
    }

    /* Typography */
    h1, h2, h3, h4, h5, h6, p, span, label, div[data-testid="stMarkdownContainer"] {
        color: #f1f5f4 !important;
    }
    .main .block-container {
        padding-top: 1.4rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    .main-title {
        font-size: 28px;
        font-weight: 800;
        letter-spacing: -0.6px;
        margin-bottom: 2px;
        background: linear-gradient(90deg, #f0fdf4, #86efac);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .subtitle {
        color: #94a89c !important;
        font-size: 14px;
        margin-bottom: 18px;
        font-weight: 500;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(185deg, #030a08 0%, #071a12 50%, #04110c 100%) !important;
        border-right: 1px solid rgba(74, 222, 128, 0.18);
    }
    section[data-testid="stSidebar"] > div {
        padding-top: 1rem;
    }
    .brand-box {
        padding: 8px 4px 18px 4px;
        border-bottom: 1px solid rgba(148,163,184,0.12);
        margin-bottom: 14px;
    }
    .brand-name {
        font-size: 22px;
        font-weight: 800;
        color: #f0fdf4 !important;
        letter-spacing: -0.4px;
    }
    .brand-subtitle {
        font-size: 11px;
        color: #7d9588 !important;
        margin-top: 2px;
        font-weight: 500;
    }
    .tricolor-line {
        height: 3px;
        width: 120px;
        margin-top: 10px;
        background: linear-gradient(90deg, #f97316 0%, #f8fafc 50%, #22c55e 100%);
        border-radius: 20px;
        box-shadow: 0 0 12px rgba(34,197,94,0.35);
    }
    .user-profile-card {
        background: linear-gradient(145deg, rgba(12, 40, 28, 0.95), rgba(8, 28, 18, 0.9));
        border: 1px solid rgba(74, 222, 128, 0.22);
        border-radius: 14px;
        padding: 14px;
        margin-bottom: 14px;
        font-size: 12px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.25);
    }
    .sidebar-section {
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #6b8576 !important;
        margin: 16px 0 8px 0;
        font-weight: 700;
    }

    /* Metrics */
    div[data-testid="stMetric"] {
        background: linear-gradient(155deg, rgba(12, 36, 26, 0.98), rgba(6, 22, 15, 0.95));
        border: 1px solid rgba(74,222,128,0.18);
        padding: 18px 16px;
        border-radius: 16px;
        box-shadow:
            0 8px 24px rgba(0,0,0,0.28),
            inset 0 1px 0 rgba(255,255,255,0.04);
        transition: border-color 0.2s ease, transform 0.15s ease;
    }
    div[data-testid="stMetric"]:hover {
        border-color: rgba(74,222,128,0.4);
    }
    div[data-testid="stMetric"] label {
        color: #8aa396 !important;
        font-size: 11px !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 28px !important;
        font-weight: 800 !important;
        color: #f0fdf4 !important;
    }

    /* Cluster alert */
    .cluster-alert-banner {
        background: linear-gradient(135deg, rgba(127, 29, 29, 0.88) 0%, rgba(69, 10, 10, 0.94) 100%);
        border: 1.5px solid #f87171;
        border-radius: 16px;
        padding: 18px 22px;
        margin-bottom: 16px;
        box-shadow:
            0 12px 32px rgba(239, 68, 68, 0.22),
            inset 0 1px 0 rgba(255,255,255,0.06);
        position: relative;
        overflow: hidden;
    }
    .cluster-alert-banner::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #ef4444, #fbbf24, #ef4444);
    }
    .cluster-alert-banner.emerging {
        background: linear-gradient(135deg, rgba(120, 53, 15, 0.88) 0%, rgba(69, 26, 3, 0.94) 100%);
        border-color: #f59e0b;
        box-shadow: 0 12px 32px rgba(245, 158, 11, 0.18), inset 0 1px 0 rgba(255,255,255,0.06);
    }
    .cluster-alert-banner.emerging::before {
        background: linear-gradient(90deg, #f59e0b, #fbbf24, #f59e0b);
    }
    .cluster-alert-title {
        font-size: 16px;
        font-weight: 800;
        color: #fecaca !important;
        display: flex;
        align-items: center;
        gap: 10px;
        flex-wrap: wrap;
    }
    .cluster-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 800;
        background: rgba(0,0,0,0.25);
        border: 1px solid rgba(252,165,165,0.5);
        color: #fecaca !important;
        letter-spacing: 0.03em;
    }

    /* Case card */
    .case-card {
        background: linear-gradient(160deg, rgba(10, 30, 22, 0.98), rgba(5, 18, 12, 0.96));
        border: 1px solid rgba(74,222,128,0.16);
        padding: 22px 24px;
        border-radius: 18px;
        margin-bottom: 18px;
        box-shadow:
            0 12px 36px rgba(0,0,0,0.28),
            inset 0 1px 0 rgba(255,255,255,0.03);
    }
    .case-id-chip {
        display: inline-block;
        font-family: 'JetBrains Mono', ui-monospace, monospace;
        font-size: 11px;
        font-weight: 700;
        padding: 4px 11px;
        border-radius: 8px;
        background: rgba(34, 197, 94, 0.12);
        border: 1px solid rgba(74,222,128,0.32);
        color: #86efac !important;
        letter-spacing: 0.06em;
    }

    .risk-high { color: #fb7185 !important; font-weight: 800; }
    .risk-medium { color: #fbbf24 !important; font-weight: 800; }
    .risk-low { color: #4ade80 !important; font-weight: 800; }

    .risk-badge {
        display: inline-block;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 0.04em;
    }
    .badge-high {
        background: rgba(239,68,68,0.18);
        border: 1px solid rgba(248,113,113,0.45);
        color: #fca5a5 !important;
        box-shadow: 0 0 16px rgba(239,68,68,0.15);
    }
    .badge-medium {
        background: rgba(245,158,11,0.16);
        border: 1px solid rgba(251,191,36,0.4);
        color: #fcd34d !important;
    }
    .badge-low {
        background: rgba(34,197,94,0.14);
        border: 1px solid rgba(74,222,128,0.35);
        color: #86efac !important;
    }

    .status-chip {
        display: inline-block;
        padding: 4px 11px;
        border-radius: 8px;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 0.05em;
        margin-top: 6px;
    }
    .status-pending {
        background: rgba(251,191,36,0.14);
        border: 1px solid rgba(251,191,36,0.38);
        color: #fbbf24 !important;
    }
    .status-verified {
        background: rgba(34,197,94,0.14);
        border: 1px solid rgba(74,222,128,0.4);
        color: #4ade80 !important;
    }
    .status-rejected {
        background: rgba(148,163,184,0.12);
        border: 1px solid rgba(148,163,184,0.32);
        color: #cbd5e1 !important;
    }
    .status-visit {
        background: rgba(59,130,246,0.14);
        border: 1px solid rgba(96,165,250,0.4);
        color: #93c5fd !important;
    }

    /* Evidence boxes */
    .evidence-box {
        background: linear-gradient(160deg, rgba(4, 18, 12, 0.7), rgba(2, 10, 7, 0.85));
        border: 1px solid rgba(148,163,184,0.12);
        border-radius: 14px;
        padding: 16px;
        min-height: 158px;
        margin-bottom: 8px;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
        transition: border-color 0.2s ease;
    }
    .evidence-box:hover {
        border-color: rgba(74,222,128,0.28);
    }
    .evidence-title {
        font-size: 11px;
        font-weight: 700;
        margin-bottom: 10px;
        color: #7d9588 !important;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .evidence-value { font-size: 15px; font-weight: 750; line-height: 1.3; }
    .evidence-label { font-size: 11px; color: #6b8576 !important; margin-top: 4px; }
    .evidence-ok { color: #4ade80 !important; }
    .evidence-warn { color: #fbbf24 !important; }
    .evidence-bad { color: #fb7185 !important; }

    .fusion-strip {
        background: linear-gradient(135deg, rgba(8, 28, 18, 0.9), rgba(12, 40, 28, 0.85));
        border: 1px solid rgba(74,222,128,0.22);
        border-radius: 14px;
        padding: 16px 18px;
        margin: 12px 0 8px;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
    }

    /* WhatsApp */
    .whatsapp-bubble {
        background: linear-gradient(135deg, rgba(7,94,84,0.9), rgba(5,55,48,0.92));
        border-radius: 14px;
        padding: 16px;
        color: #f8fafc !important;
        border-left: 4px solid #25D366;
        box-shadow: 0 6px 20px rgba(0,0,0,0.2);
    }
    .whatsapp-label {
        color: #86efac !important;
        font-size: 11px;
        font-weight: 700;
        margin-bottom: 6px;
        letter-spacing: 0.06em;
    }
    .advisory-sent {
        background: rgba(34,197,94,0.12);
        border: 1px solid rgba(74,222,128,0.35);
        border-radius: 10px;
        padding: 8px 12px;
        margin-top: 8px;
        font-size: 12px;
        color: #86efac !important;
        font-weight: 600;
    }

    /* Health pills */
    .health-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 13px;
        border-radius: 20px;
        font-size: 11px;
        margin-right: 7px;
        margin-bottom: 7px;
        font-weight: 650;
        backdrop-filter: blur(6px);
    }
    .health-up {
        background: rgba(34,197,94,0.12);
        color: #86efac !important;
        border: 1px solid rgba(34,197,94,0.28);
    }
    .health-demo {
        background: rgba(251,191,36,0.10);
        color: #fcd34d !important;
        border: 1px solid rgba(251,191,36,0.28);
    }
    .health-standby {
        background: rgba(148,163,184,0.10);
        color: #94a3b8 !important;
        border: 1px solid rgba(148,163,184,0.28);
    }
    .health-down {
        background: rgba(248,113,113,0.12);
        color: #fca5a5 !important;
        border: 1px solid rgba(248,113,113,0.3);
    }
    /* Expert queue list */
    .expert-queue-item {
        background: linear-gradient(145deg, rgba(10, 28, 20, 0.9), rgba(6, 18, 12, 0.95));
        border: 1px solid rgba(74,222,128,0.14);
        border-radius: 12px;
        padding: 12px 14px;
        margin-bottom: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 12px;
    }
    .expert-queue-item.high {
        border-left: 3px solid #f87171;
    }
    .expert-queue-item.medium {
        border-left: 3px solid #fbbf24;
    }
    .expert-queue-item.low {
        border-left: 3px solid #4ade80;
    }
    .expert-hero {
        background: linear-gradient(135deg, rgba(15, 45, 32, 0.95), rgba(8, 24, 18, 0.98));
        border: 1px solid rgba(74,222,128,0.22);
        border-radius: 18px;
        padding: 20px 22px;
        margin-bottom: 18px;
        box-shadow: 0 10px 28px rgba(0,0,0,0.25), inset 0 1px 0 rgba(255,255,255,0.04);
    }
    .expert-hero-title {
        font-size: 15px;
        font-weight: 700;
        color: #86efac !important;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .expert-hero-body {
        font-size: 13.5px;
        color: #b7c9be !important;
        line-height: 1.5;
    }
    .triage-step {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 650;
        margin-right: 6px;
        margin-bottom: 6px;
        background: rgba(34,197,94,0.1);
        border: 1px solid rgba(74,222,128,0.25);
        color: #bbf7d0 !important;
    }

    /* Login */
    .login-page-wrap {
        min-height: 70vh;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 20px 0 40px;
    }
    .login-container {
        max-width: 520px;
        width: 100%;
        margin: 0 auto 8px;
        padding: 36px 36px 28px;
        background: linear-gradient(155deg, rgba(10, 32, 22, 0.97), rgba(5, 18, 12, 0.98));
        border: 1px solid rgba(74, 222, 128, 0.28);
        border-radius: 22px;
        box-shadow:
            0 24px 60px rgba(0, 0, 0, 0.45),
            0 0 0 1px rgba(255,255,255,0.03),
            inset 0 1px 0 rgba(255,255,255,0.05);
        text-align: center;
    }
    .login-icon {
        width: 64px;
        height: 64px;
        margin: 0 auto 14px;
        border-radius: 18px;
        background: linear-gradient(145deg, rgba(34,197,94,0.25), rgba(22,101,52,0.4));
        border: 1px solid rgba(74,222,128,0.35);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 32px;
        box-shadow: 0 8px 24px rgba(34,197,94,0.2);
    }
    .login-title {
        font-size: 24px;
        font-weight: 800;
        margin-bottom: 6px;
        color: #f0fdf4 !important;
        letter-spacing: -0.4px;
    }
    .login-sub {
        font-size: 13px;
        color: #8aa396 !important;
        margin-bottom: 8px;
        line-height: 1.45;
    }
    .login-badge {
        display: inline-block;
        margin-top: 10px;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        background: rgba(34,197,94,0.12);
        border: 1px solid rgba(74,222,128,0.3);
        color: #86efac !important;
    }

    .section-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(74,222,128,0.25), transparent);
        margin: 8px 0 18px;
        border: none;
    }
    .map-caption {
        font-size: 12px;
        color: #7d9588 !important;
        padding: 8px 12px;
        background: rgba(8, 24, 16, 0.5);
        border-radius: 8px;
        border: 1px solid rgba(74,222,128,0.1);
        margin-top: 8px;
    }

    /* —— BUTTONS — black / dark text on light-ish surfaces where needed —— */
    .stButton > button {
        border-radius: 10px !important;
        border: 1px solid rgba(74,222,128,0.35) !important;
        background: linear-gradient(180deg, rgba(22, 70, 42, 0.95), rgba(12, 45, 28, 0.98)) !important;
        color: #ecfdf5 !important;
        font-weight: 650 !important;
        font-size: 14px !important;
        padding: 0.45rem 1rem !important;
        transition: all 0.18s ease !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2) !important;
    }
    .stButton > button:hover {
        border-color: rgba(134, 239, 172, 0.65) !important;
        background: linear-gradient(180deg, rgba(28, 90, 52, 0.98), rgba(16, 58, 34, 1)) !important;
        color: #ffffff !important;
        box-shadow: 0 4px 14px rgba(34,197,94,0.25) !important;
    }
    .stButton > button p,
    .stButton > button span,
    .stButton > button div {
        color: #ecfdf5 !important;
    }

    /* Login form submit — high-contrast black text on mint button */
    div[data-testid="stFormSubmitButton"] > button,
    div[data-testid="stFormSubmitButton"] button {
        background: linear-gradient(180deg, #86efac 0%, #4ade80 45%, #22c55e 100%) !important;
        border: 1px solid #16a34a !important;
        color: #052e16 !important;
        font-weight: 800 !important;
        font-size: 15px !important;
        border-radius: 12px !important;
        padding: 0.65rem 1.2rem !important;
        box-shadow:
            0 6px 20px rgba(34, 197, 94, 0.35),
            inset 0 1px 0 rgba(255,255,255,0.35) !important;
        letter-spacing: 0.01em !important;
    }
    div[data-testid="stFormSubmitButton"] > button:hover,
    div[data-testid="stFormSubmitButton"] button:hover {
        background: linear-gradient(180deg, #a7f3d0 0%, #6ee7b7 45%, #34d399 100%) !important;
        color: #022c14 !important;
        border-color: #15803d !important;
        box-shadow: 0 8px 24px rgba(34, 197, 94, 0.45) !important;
    }
    div[data-testid="stFormSubmitButton"] > button p,
    div[data-testid="stFormSubmitButton"] > button span,
    div[data-testid="stFormSubmitButton"] > button div,
    div[data-testid="stFormSubmitButton"] button p,
    div[data-testid="stFormSubmitButton"] button span,
    div[data-testid="stFormSubmitButton"] button div {
        color: #052e16 !important;
        font-weight: 800 !important;
    }

    /* Inputs */
    .stTextInput > div > div > input,
    .stSelectbox > div > div,
    .stTextArea > div > div > textarea {
        border-radius: 10px !important;
        border-color: rgba(74,222,128,0.22) !important;
        background-color: rgba(4, 16, 11, 0.7) !important;
    }
    div[data-testid="stExpander"] {
        background: rgba(6, 20, 14, 0.55);
        border: 1px solid rgba(74,222,128,0.12);
        border-radius: 12px;
        margin-bottom: 8px;
    }

    /* Dataframe */
    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(74,222,128,0.12);
    }

    /* Radio horizontal */
    div[role="radiogroup"] label {
        background: rgba(8, 28, 18, 0.6) !important;
        border-radius: 8px !important;
        padding: 4px 8px !important;
    }

    /* Divider */
    hr {
        border-color: rgba(74,222,128,0.12) !important;
    }

    /* Footer strip */
    .app-footer {
        margin-top: 32px;
        padding: 14px 0;
        text-align: center;
        font-size: 11px;
        color: #5c7366 !important;
        border-top: 1px solid rgba(74,222,128,0.1);
        letter-spacing: 0.04em;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# MOCK BACKEND — Integration Contract (§10) shapes
# ============================================================

PROTOTYPE_WEIGHTS = {
    "Image AI": 0.40,
    "Satellite": 0.25,
    "Weather": 0.15,
    "Nearby reports": 0.20,
}

# Digras Wadi canonical coords for CASE-001 demo journey
DIGRAS_WADI = {
    "district": "Yavatmal",
    "taluka": "Pusad",
    "village": "Digras Wadi",
    "lat": 19.90,
    "lon": 77.57,
    "zone": "north-east",
}

MAHARASHTRA_LOCATIONS = [
    DIGRAS_WADI,
    {"district": "Yavatmal", "taluka": "Pusad", "village": "Shembalpimpri", "lat": 19.86, "lon": 77.52, "zone": "south"},
    {"district": "Yavatmal", "taluka": "Darwha", "village": "Shirpur", "lat": 20.22, "lon": 77.78, "zone": "north-west"},
    {"district": "Amravati", "taluka": "Achalpur", "village": "Paratwada", "lat": 21.27, "lon": 77.51, "zone": "east"},
    {"district": "Amravati", "taluka": "Achalpur", "village": "Chandur Bazar", "lat": 21.24, "lon": 77.60, "zone": "central"},
    {"district": "Amravati", "taluka": "Daryapur", "village": "Anjangaon", "lat": 21.05, "lon": 77.31, "zone": "south-east"},
    {"district": "Wardha", "taluka": "Hinganghat", "village": "Ajansara", "lat": 20.55, "lon": 78.83, "zone": "north"},
    {"district": "Nagpur", "taluka": "Kalmeshwar", "village": "Mohpa", "lat": 21.28, "lon": 78.85, "zone": "west"},
    {"district": "Nanded", "taluka": "Bhokar", "village": "Umri", "lat": 19.15, "lon": 77.65, "zone": "north-east"},
    {"district": "Nanded", "taluka": "Kandhar", "village": "Fulwal", "lat": 18.83, "lon": 77.15, "zone": "south-west"},
]

FARMER_NAMES = [
    "Ramesh Patil", "Suresh Jadhav", "Vandana Rathod", "Ganesh Deshmukh",
    "Kavita More", "Ashok Kale", "Meena Shinde", "Prakash Wagh", "Baban Shinde", "Sunita Gawande",
]

DISEASE_CLASSES = [
    "Cercospora Leaf Blight",
    "Bacterial Blight",
    "Leaf Spot (Fungal)",
    "Rust",
    "Healthy",
]

RISK_REASON_POOL = [
    "Disease detected from crop image",
    "Satellite vegetation anomaly detected",
    "Weather conditions indicate elevated risk",
    "Similar nearby reports found",
    "Historical hotspot in this village",
    "Pest / hopper vector activity reported nearby",
]


def _case_id(n: int) -> str:
    return f"CASE-{n:03d}"


def _field_id(n: int) -> str:
    return f"FIELD-{n:03d}"


def _generate_mock_case(seq: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    loc = MAHARASHTRA_LOCATIONS[seed % len(MAHARASHTRA_LOCATIONS)]

    confidence = round(float(rng.uniform(0.60, 0.98)), 2)
    prediction = DISEASE_CLASSES[seed % len(DISEASE_CLASSES)]
    disease_detected = prediction != "Healthy"

    ndvi_current = round(float(rng.uniform(0.32, 0.65)), 2)
    ndvi_historical = round(ndvi_current + float(rng.uniform(0.08, 0.25)), 2)
    anomaly_score = round(float(rng.uniform(0.35, 0.95)), 2)
    anomaly_detected = anomaly_score >= 0.45

    weather_risk = round(float(rng.uniform(0.35, 0.92)), 2)
    nearby_count = int(rng.integers(0, 5))
    nearby_norm = min(1.0, nearby_count / 4.0)

    image_signal = confidence if disease_detected else confidence * 0.25
    sat_signal = anomaly_score if anomaly_detected else anomaly_score * 0.3

    fused = (
        PROTOTYPE_WEIGHTS["Image AI"] * image_signal
        + PROTOTYPE_WEIGHTS["Satellite"] * sat_signal
        + PROTOTYPE_WEIGHTS["Weather"] * weather_risk
        + PROTOTYPE_WEIGHTS["Nearby reports"] * nearby_norm
    )
    risk_score_01 = round(float(min(0.98, max(0.12, fused))), 2)
    risk_score = int(round(risk_score_01 * 100))

    if risk_score >= 70:
        risk_level = "HIGH"
    elif risk_score >= 40:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    reasons = []
    if disease_detected:
        reasons.append("Disease detected from crop image")
    if anomaly_detected:
        reasons.append("Satellite vegetation anomaly detected")
    if weather_risk >= 0.6:
        reasons.append("Weather conditions indicate elevated risk")
    if nearby_count >= 2:
        reasons.append("Similar nearby reports found")
    if not reasons:
        reasons = list(rng.choice(RISK_REASON_POOL, size=2, replace=False))

    contrib = {
        "Image AI (40%)": round(PROTOTYPE_WEIGHTS["Image AI"] * image_signal * 100, 1),
        "Satellite (25%)": round(PROTOTYPE_WEIGHTS["Satellite"] * sat_signal * 100, 1),
        "Weather (15%)": round(PROTOTYPE_WEIGHTS["Weather"] * weather_risk * 100, 1),
        "Nearby reports (20%)": round(PROTOTYPE_WEIGHTS["Nearby reports"] * nearby_norm * 100, 1),
    }

    report_time = datetime.now() - timedelta(hours=int(rng.integers(1, 48)))
    # Lean soybean for one-crop prototype narrative; keep some variety for map density
    crop = "Soybean" if seed % 4 != 0 else "Cotton"
    crop_mr = "सोयाबीन" if crop == "Soybean" else "कापूस"
    lat = loc["lat"] + float(rng.uniform(-0.015, 0.015))
    lon = loc["lon"] + float(rng.uniform(-0.015, 0.015))

    farmer_report_text = (
        "पाने पिवळी पडत आहेत, काही ठिकाणी ठिपके दिसत आहेत."
        if seed % 2 == 0
        else "Leaves yellowing near north side; spots visible on lower canopy."
    )

    cid = _case_id(seq)
    fid = _field_id(seq)

    return {
        "case_id": cid,
        "field_id": fid,
        "satellite": {
            "anomaly_detected": anomaly_detected,
            "anomaly_score": anomaly_score,
            "centroid": {"lat": round(lat, 5), "lng": round(lon, 5)},
            "zone": loc["zone"],
            "ndvi_current": ndvi_current,
            "ndvi_historical": ndvi_historical,
        },
        "image_ai": {
            "crop": "soyabean" if crop == "Soybean" else "cotton",
            "condition": prediction,
            "confidence": confidence,
            "quality": "GOOD" if confidence > 0.60 else "POOR",
            "diseaseDetected": disease_detected,
        },
        "fusion": {
            "risk": risk_level,
            "score": risk_score_01,
            "expert_required": risk_level != "LOW",
            "reasons": reasons,
        },
        "farmer_name": FARMER_NAMES[seed % len(FARMER_NAMES)],
        "crop": crop,
        "district": loc["district"],
        "taluka": loc["taluka"],
        "village": loc["village"],
        "lat": lat,
        "lon": lon,
        "image_prediction": prediction,
        "image_confidence": confidence,
        "image_quality": "GOOD" if confidence > 0.60 else "POOR",
        "ndvi_current": ndvi_current,
        "ndvi_historical": ndvi_historical,
        "anomaly_score": anomaly_score,
        "anomaly_detected": anomaly_detected,
        "weather_risk": weather_risk,
        "nearby_reports": nearby_count,
        "farmer_report": {
            "received": True,
            "channel": "WhatsApp voice",
            "language": "mr" if seed % 2 == 0 else "en",
            "summary": farmer_report_text,
        },
        "risk_score": risk_score,
        "risk_level": risk_level,
        "reasons": reasons,
        "contribution": contrib,
        "needs_expert": risk_level != "LOW",
        "status": "PENDING",
        "advisory_sent": False,
        "reported_at": report_time,
        "log": [
            f"{report_time.strftime('%H:%M')} — Satellite anomaly flagged · centroid ({lat:.3f}, {lon:.3f})",
            f"{report_time.strftime('%H:%M')} — WhatsApp alert sent to farmer",
            f"{(report_time + timedelta(minutes=12)).strftime('%H:%M')} — Farmer voice report received",
            f"{(report_time + timedelta(minutes=18)).strftime('%H:%M')} — Crop image received via WhatsApp",
            f"{(report_time + timedelta(minutes=19)).strftime('%H:%M')} — Image AI → {prediction} ({confidence:.0%})",
            f"{(report_time + timedelta(minutes=20)).strftime('%H:%M')} — Fusion risk = {risk_level} ({risk_score_01})",
        ],
        "whatsapp_advisory_mr": (
            f"⚠️ सूचना: आपल्या {crop_mr} शेतात ({loc['village']}) असामान्य बदल व रोगाची लक्षणे आढळली आहेत. "
            f"अधिक माहितीसाठी कृषी सहाय्यकांशी संपर्क साधा। — KrishiDrishti"
        ),
        "whatsapp_advisory_en": (
            f"⚠️ Alert: Unusual stress and disease symptoms detected in your {crop} field ({loc['village']}). "
            f"Please contact your agriculture assistant. — KrishiDrishti"
        ),
    }


def _ensure_demo_images() -> tuple[str, str]:
    """
    Make sure demo satellite + leaf JPGs exist under assets/demo/.
    Creates folders + simple placeholder images if missing (no manual copy needed).
    Returns (satellite_path, leaf_path).
    """
    from pathlib import Path

    root = Path(__file__).resolve().parent
    demo_dir = root / "assets" / "demo"
    sat = demo_dir / "case001_satellite.jpg"
    leaf = demo_dir / "case001_leaf.jpg"

    try:
        demo_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return str(sat), str(leaf)

    if not sat.is_file():
        try:
            w, h = 800, 500
            img = Image.new("RGB", (w, h), (12, 40, 28))
            d = ImageDraw.Draw(img)
            for i in range(0, w, 40):
                d.line([(i, 0), (i, h)], fill=(20, 60, 40), width=1)
            for j in range(0, h, 40):
                d.line([(0, j), (w, j)], fill=(20, 60, 40), width=1)
            d.ellipse([80, 60, 520, 420], fill=(34, 120, 70), outline=(46, 160, 90))
            d.ellipse([380, 120, 700, 380], fill=(140, 70, 30), outline=(220, 100, 40))
            d.ellipse([480, 180, 640, 320], fill=(180, 50, 40), outline=(255, 80, 60))
            cx, cy = 560, 250
            d.ellipse([cx - 12, cy - 12, cx + 12, cy + 12], outline=(255, 255, 255), width=3)
            d.line([(cx - 28, cy), (cx + 28, cy)], fill=(255, 255, 200), width=2)
            d.line([(cx, cy - 28), (cx, cy + 28)], fill=(255, 255, 200), width=2)
            d.rectangle([16, 16, 480, 100], fill=(8, 24, 16))
            d.text((28, 28), "DEMO satellite preview · FIELD-001", fill=(180, 230, 190))
            d.text((28, 55), "Digras Wadi · NE stress zone · anomaly 0.78", fill=(250, 220, 120))
            d.text((28, 78), "Prototype only · not live GEE", fill=(140, 160, 150))
            img.save(sat, quality=90)
        except Exception:
            pass

    if not leaf.is_file():
        try:
            w, h = 640, 640
            img = Image.new("RGB", (w, h), (30, 50, 28))
            d = ImageDraw.Draw(img)
            d.ellipse([120, 80, 520, 560], fill=(55, 130, 55), outline=(40, 90, 40), width=4)
            d.ellipse([200, 140, 480, 500], fill=(70, 150, 60))
            d.line([(320, 120), (320, 520)], fill=(40, 90, 35), width=6)
            for x, y in [
                (250, 220), (300, 280), (360, 250), (280, 340),
                (340, 360), (390, 310), (260, 400), (350, 420),
            ]:
                d.ellipse([x - 18, y - 14, x + 22, y + 18], fill=(90, 50, 20), outline=(60, 30, 10))
                d.ellipse([x - 8, y - 6, x + 10, y + 8], fill=(40, 25, 10))
            d.rectangle([16, 16, 520, 90], fill=(15, 25, 15))
            d.text((28, 28), "DEMO farmer crop photo · WhatsApp", fill=(200, 230, 200))
            d.text((28, 55), "Soybean · CASE-001 · prototype only", fill=(250, 200, 120))
            img.save(leaf, quality=90)
        except Exception:
            pass

    return str(sat), str(leaf)


def _resolve_asset(path: str | None) -> str | None:
    """Resolve demo image path relative to project root (cwd) or this file."""
    if not path:
        return None
    from pathlib import Path

    # Ensure default demo assets exist when CASE-001 paths are requested
    if "case001_satellite" in path.replace("\\", "/") or "case001_leaf" in path.replace("\\", "/"):
        _ensure_demo_images()

    candidates = [
        Path(path),
        Path.cwd() / path,
        Path(__file__).resolve().parent / path,
    ]
    for p in candidates:
        try:
            if p.is_file():
                return str(p)
        except OSError:
            continue
    return None


def _normalize_case(case: dict) -> dict:
    """Ensure reported_at is datetime so existing UI .strftime calls work."""
    c = dict(case)
    ra = c.get("reported_at")
    if isinstance(ra, str):
        try:
            c["reported_at"] = datetime.fromisoformat(ra.replace("Z", ""))
        except ValueError:
            c["reported_at"] = datetime.now()
    elif not isinstance(ra, datetime):
        c["reported_at"] = datetime.now()
    # defaults so older partial payloads don't crash the UI
    c.setdefault("log", [])
    c.setdefault("reasons", [])
    c.setdefault("contribution", {})
    c.setdefault("advisory_sent", False)
    c.setdefault("farmer_report", {
        "received": False, "channel": "—", "language": "en", "summary": "—",
    })
    c.setdefault("satellite", {
        "anomaly_detected": c.get("anomaly_detected", False),
        "anomaly_score": c.get("anomaly_score", 0),
        "centroid": {"lat": c.get("lat", 0), "lng": c.get("lon", 0)},
        "zone": "—",
        "ndvi_current": c.get("ndvi_current", 0),
        "ndvi_historical": c.get("ndvi_historical", 0),
    })
    c.setdefault("image_ai", {
        "crop": str(c.get("crop", "")).lower(),
        "condition": c.get("image_prediction", "—"),
        "confidence": c.get("image_confidence", 0),
        "quality": c.get("image_quality", "—"),
        "diseaseDetected": c.get("image_prediction", "") != "Healthy",
    })
    c.setdefault("fusion", {
        "risk": c.get("risk_level", "LOW"),
        "score": (c.get("risk_score") or 0) / 100,
        "expert_required": c.get("needs_expert", False),
        "reasons": c.get("reasons", []),
    })
    c.setdefault("whatsapp_advisory_mr", "—")
    c.setdefault("whatsapp_advisory_en", "—")
    c.setdefault("weather_risk", 0.5)
    c.setdefault("nearby_reports", 0)
    c.setdefault("needs_expert", c.get("risk_level") != "LOW")
    c.setdefault("satellite_image", None)
    c.setdefault("crop_image", None)
    c.setdefault("images_source", "PENDING")
    return c


def _build_local_mock_cases(n: int = 10) -> list:
    cases = [_generate_mock_case(i + 1, i * 7 + 3) for i in range(n)]
    demo = cases[0]
    demo_lat = DIGRAS_WADI["lat"] + 0.008
    demo_lon = DIGRAS_WADI["lon"] - 0.006
    demo["case_id"] = "CASE-001"
    demo["field_id"] = "FIELD-001"
    demo["crop"] = "Soybean"
    demo["image_prediction"] = "Cercospora Leaf Blight"
    demo["image_ai"] = {
        "crop": "soyabean",
        "condition": "Cercospora Leaf Blight",
        "confidence": 0.87,
        "quality": "GOOD",
        "diseaseDetected": True,
    }
    demo["image_confidence"] = 0.87
    demo["image_quality"] = "GOOD"
    demo["anomaly_detected"] = True
    demo["anomaly_score"] = 0.78
    demo["ndvi_current"] = 0.41
    demo["ndvi_historical"] = 0.62
    demo["satellite"] = {
        "anomaly_detected": True,
        "anomaly_score": 0.78,
        "centroid": {"lat": round(demo_lat, 5), "lng": round(demo_lon, 5)},
        "zone": DIGRAS_WADI["zone"],
        "ndvi_current": 0.41,
        "ndvi_historical": 0.62,
    }
    demo["lat"] = demo_lat
    demo["lon"] = demo_lon
    demo["village"] = DIGRAS_WADI["village"]
    demo["taluka"] = DIGRAS_WADI["taluka"]
    demo["district"] = DIGRAS_WADI["district"]
    demo["farmer_name"] = "Ramesh Patil"
    demo["risk_level"] = "HIGH"
    demo["risk_score"] = 81
    demo["weather_risk"] = 0.72
    demo["nearby_reports"] = 3
    demo["needs_expert"] = True
    demo["fusion"] = {
        "risk": "HIGH",
        "score": 0.81,
        "expert_required": True,
        "reasons": [
            "Disease detected from crop image",
            "Satellite vegetation anomaly detected",
            "Weather conditions indicate elevated risk",
            "Similar nearby reports found",
        ],
    }
    demo["reasons"] = demo["fusion"]["reasons"]
    demo["contribution"] = {
        "Image AI (40%)": 34.8,
        "Satellite (25%)": 19.5,
        "Weather (15%)": 10.8,
        "Nearby reports (20%)": 15.0,
    }
    demo["farmer_report"] = {
        "received": True,
        "channel": "WhatsApp voice",
        "language": "mr",
        "summary": "पाने पिवळी पडत आहेत, काही ठिकाणी ठिपके दिसत आहेत. उत्तर-पूर्व कोपऱ्यात जास्त दिसते.",
    }
    demo["satellite_image"] = "assets/demo/case001_satellite.jpg"
    demo["crop_image"] = "assets/demo/case001_leaf.jpg"
    demo["images_source"] = "DEMO"
    return cases


def get_mock_cases(n: int = 10):
    """Local-only seed (used when Cases API is offline)."""
    if "cases" not in st.session_state or not st.session_state.cases:
        st.session_state.cases = _build_local_mock_cases(n)
        st.session_state.data_source = "MOCK"
    return st.session_state.cases


def load_cases(force_refresh: bool = False) -> list:
    """
    Prefer Cases API (LIVE). If unreachable, use in-browser mock.
    Set KRISHI_CASES_API env var to change base URL (default http://127.0.0.1:8000).
    """
    force_mock = os.getenv("KRISHI_FORCE_MOCK", "").lower() in ("1", "true", "yes")
    if force_mock:
        st.session_state.data_source = "MOCK"
        return get_mock_cases()

    if force_refresh:
        st.session_state.pop("cases", None)

    # Try live API first
    if api_client is not None:
        ok, result = api_client.try_live_cases()
        if ok and isinstance(result, list):
            st.session_state.cases = [_normalize_case(c) for c in result]
            st.session_state.data_source = "LIVE"
            st.session_state.api_error = None
            return st.session_state.cases
        st.session_state.api_error = str(result)

    # Fallback
    st.session_state.data_source = "MOCK"
    return get_mock_cases()


def update_case_status(case_id: str, new_status: str, advisory: bool = False, action: str | None = None):
    """
    If LIVE → POST /api/cases/{id}/action then refresh.
    If MOCK → mutate session_state only.
    action: CONFIRM | REJECT | FIELD_VISIT_REQUIRED (required for LIVE path)
    """
    # Map status → API action name when caller only passed status
    if action is None:
        action = {
            "VERIFIED": "CONFIRM",
            "REJECTED": "REJECT",
            "FIELD_VISIT_REQUIRED": "FIELD_VISIT_REQUIRED",
        }.get(new_status)

    source = st.session_state.get("data_source", "MOCK")
    if source == "LIVE" and api_client is not None and action:
        try:
            officer = None
            user = st.session_state.get("user") or {}
            officer = user.get("phone") or user.get("role")
            updated = api_client.post_action(case_id, action, officer_id=officer)
            # refresh full list so KPIs stay in sync
            ok, result = api_client.try_live_cases()
            if ok and isinstance(result, list):
                st.session_state.cases = [_normalize_case(c) for c in result]
            else:
                # at least patch this case
                for i, c in enumerate(st.session_state.cases):
                    if c["case_id"] == case_id:
                        st.session_state.cases[i] = _normalize_case(updated)
                        break
            st.session_state.last_action = f"{case_id} → {action} (LIVE)"
            if updated.get("advisory_sent"):
                st.session_state.last_action += " · advisory SENT"
            return
        except Exception as exc:
            st.session_state.api_error = str(exc)
            # fall through to local mutation so demo never hard-fails

    for case in st.session_state.cases:
        if case["case_id"] == case_id:
            case["status"] = new_status
            ts = datetime.now().strftime("%H:%M")
            case.setdefault("log", []).append(f"{ts} — Officer/Expert marked case as {new_status}")
            if advisory and new_status == "VERIFIED":
                case["advisory_sent"] = True
                case["log"].append(f"{ts} — Localized farmer advisory SENT via WhatsApp")
            st.session_state.last_action = f"{case_id} → {new_status}" + (
                " · advisory SENT" if case.get("advisory_sent") else ""
            )
            return


def simulate_new_case():
    if st.session_state.get("data_source") == "LIVE" and api_client is not None:
        try:
            created = api_client.ingest_demo_case()
            ok, result = api_client.try_live_cases()
            if ok and isinstance(result, list):
                st.session_state.cases = [_normalize_case(c) for c in result]
            else:
                st.session_state.cases.append(_normalize_case(created))
            return
        except Exception as exc:
            st.session_state.api_error = str(exc)

    existing_nums = []
    for c in st.session_state.cases:
        try:
            existing_nums.append(int(str(c["case_id"]).split("-")[-1]))
        except ValueError:
            pass
    new_n = (max(existing_nums) if existing_nums else 0) + 1
    st.session_state.cases.append(_generate_mock_case(new_n, new_n * 13 + 7))


def reset_cases():
    if st.session_state.get("data_source") == "LIVE" and api_client is not None:
        try:
            api_client.reset_demo()
            ok, result = api_client.try_live_cases()
            if ok and isinstance(result, list):
                st.session_state.cases = [_normalize_case(c) for c in result]
                return
        except Exception as exc:
            st.session_state.api_error = str(exc)
    for key in ("cases", "last_action"):
        st.session_state.pop(key, None)
    get_mock_cases()


def get_pipeline_health(active_view: str = "officer"):
    """Honest demo labels — do not claim production uptime.
    active_view: 'officer' | 'expert' controls which dashboard shows Active.
    """
    officer_state = "ACTIVE" if active_view == "officer" else "STANDBY"
    expert_state = "ACTIVE" if active_view == "expert" else "STANDBY"
    return {
        "Image AI": "DEMO",
        "Satellite / NDVI": "DEMO",
        "Risk Fusion": "DEMO",
        "WhatsApp": "DEMO",
        "Officer Dashboard": officer_state,
        "Expert Dashboard": expert_state,
    }


def _health_label(state) -> tuple:
    """Return (css_class, display_text) for pipeline status."""
    if state is True or state == "ACTIVE":
        return "health-up", "Active"
    if state == "STANDBY":
        return "health-standby", "Standby"
    if state == "DEMO":
        return "health-demo", "Demo"
    if state is False or state == "OFFLINE":
        return "health-down", "Offline"
    return "health-demo", str(state)


# ============================================================
# CLUSTER DETECTION
# ============================================================

def detect_disease_clusters(case_list, distance_threshold_deg=0.18, min_cluster_cases=2):
    active_cases = [c for c in case_list if c["risk_level"] in ("HIGH", "MEDIUM")]
    clusters = []
    visited = set()

    for base in active_cases:
        if base["case_id"] in visited:
            continue
        group = [base]
        visited.add(base["case_id"])

        for other in active_cases:
            if other["case_id"] in visited:
                continue
            dist = np.sqrt((base["lat"] - other["lat"]) ** 2 + (base["lon"] - other["lon"]) ** 2)
            same_region = base["taluka"] == other["taluka"] and base["district"] == other["district"]
            if dist <= distance_threshold_deg or same_region:
                group.append(other)
                visited.add(other["case_id"])

        if len(group) >= min_cluster_cases:
            high_count = sum(1 for c in group if c["risk_level"] == "HIGH")
            crops = list(set(c["crop"] for c in group))
            diseases = list(
                set(c["image_prediction"] for c in group if c["image_prediction"] != "Healthy")
            )
            villages = sorted(set(c["village"] for c in group))
            # Keep severity key stable for UI colour branching
            severity = "CRITICAL OUTBREAK" if high_count >= 2 else "EMERGING CONCENTRATION"
            clusters.append(
                {
                    "cluster_id": f"CLUST-{len(clusters) + 1:02d}",
                    "severity": severity,
                    "district": group[0]["district"],
                    "taluka": group[0]["taluka"],
                    "villages": villages,
                    "crops": crops,
                    "diseases": diseases if diseases else ["Unspecified Anomaly"],
                    "total_cases": len(group),
                    "high_risk_cases": high_count,
                    "center_lat": float(np.mean([c["lat"] for c in group])),
                    "center_lon": float(np.mean([c["lon"] for c in group])),
                    "cases": group,
                }
            )
    return clusters


# ============================================================
# LOGIN
# ============================================================

def render_login_portal():
    st.markdown(
        """
        <div class="login-page-wrap">
        <div class="login-container">
            <div class="login-icon">🌾</div>
            <div class="login-title">KrishiDrishti Portal</div>
            <div class="login-sub">
                Government of Maharashtra<br>
                Officer &amp; Expert Crop Surveillance · SIH 26131
            </div>
            <div class="tricolor-line" style="margin: 14px auto 6px; width: 140px;"></div>
            <div class="login-badge">Prototype Demo Access</div>
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _, c2, _ = st.columns([1, 2.1, 1])
    with c2:
        with st.form("login_form"):
            st.markdown("##### Sign in")
            phone = st.text_input(
                "📱 Mobile / Phone Number*",
                placeholder="e.g. 9823012345 or +91 98230 12345",
            )
            address = st.text_input(
                "📍 Official Station / District*",
                placeholder="e.g. SDAO Pusad, Yavatmal",
            )
            role_choice = st.selectbox(
                "👤 Portal Role*",
                [
                    "🛡️ Agricultural Officer (Command & Hotspot Intelligence)",
                    "🔬 Diagnostic Expert (Evidence-Based Triage)",
                ],
            )
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
            submit = st.form_submit_button(
                "Enter KrishiDrishti Dashboard",
                use_container_width=True,
            )

            if submit:
                clean_phone = re.sub(r"[\s\-+]", "", phone)
                if len(clean_phone) < 10 or not clean_phone.isdigit():
                    st.error("Please enter a valid 10-digit mobile number.")
                elif len(address.strip()) < 3:
                    st.error("Please enter your official station or workplace address.")
                else:
                    st.session_state.authenticated = True
                    st.session_state.user = {
                        "phone": phone.strip(),
                        "address": address.strip(),
                        "role": role_choice,
                        "login_time": datetime.now(),
                    }
                    st.success("Demo access granted. Loading dashboard…")
                    st.rerun()

        st.markdown(
            "<div style='text-align:center; margin: 18px 0 8px; color:#7d9588; font-size:12px; font-weight:600; letter-spacing:0.08em;'>QUICK DEMO PROFILES</div>",
            unsafe_allow_html=True,
        )
        q1, q2 = st.columns(2)
        if q1.button("🛡️ Officer · Yavatmal", use_container_width=True):
            st.session_state.authenticated = True
            st.session_state.user = {
                "phone": "+91 98221 44550",
                "address": "District Agriculture Office, Yavatmal (MH)",
                "role": "🛡️ Agricultural Officer (Command & Hotspot Intelligence)",
                "login_time": datetime.now(),
            }
            st.rerun()

        if q2.button("🔬 Expert · Pusad", use_container_width=True):
            st.session_state.authenticated = True
            st.session_state.user = {
                "phone": "+91 94230 88990",
                "address": "KVK Agronomy Research Center, Pusad",
                "role": "🔬 Diagnostic Expert (Evidence-Based Triage)",
                "login_time": datetime.now(),
            }
            st.rerun()

        st.markdown(
            """
            <div class="app-footer" style="border:none; margin-top:28px;">
                TARGET → VERIFY → FUSE → ACT &nbsp;·&nbsp; Prototype — not production
            </div>
            """,
            unsafe_allow_html=True,
        )


if "authenticated" not in st.session_state or not st.session_state.authenticated:
    render_login_portal()
    st.stop()

# ============================================================
# UI HELPERS
# ============================================================

def render_pipeline_health_strip(active_view: str = "officer"):
    health = get_pipeline_health(active_view)
    pills = ""
    for module, state in health.items():
        cls, label = _health_label(state)
        pills += f'<span class="health-pill {cls}">● {module} · {label}</span>'
    st.markdown(pills, unsafe_allow_html=True)


def risk_css_class(level: str) -> str:
    return {"HIGH": "risk-high", "MEDIUM": "risk-medium", "LOW": "risk-low"}.get(
        level, "risk-low"
    )


def render_risk_badge(level: str) -> str:
    cls = {
        "HIGH": "badge-high",
        "MEDIUM": "badge-medium",
        "LOW": "badge-low",
    }.get(level, "badge-low")
    return f'<span class="risk-badge {cls}">{level} RISK</span>'


def render_status_chip(status: str) -> str:
    mapping = {
        "PENDING": ("status-pending", "PENDING VERIFICATION"),
        "VERIFIED": ("status-verified", "VERIFIED"),
        "REJECTED": ("status-rejected", "REJECTED"),
        "FIELD_VISIT_REQUIRED": ("status-visit", "FIELD VISIT REQUIRED"),
        "CONFIRMED": ("status-verified", "VERIFIED"),
        "NEEDS_VISIT": ("status-visit", "FIELD VISIT REQUIRED"),
    }
    cls, label = mapping.get(status, ("status-pending", status))
    return f'<span class="status-chip {cls}">{label}</span>'


def render_case_card(case: dict, show_actions: bool = True, key_prefix: str = ""):
    # Header as one HTML block (avoid empty wrapper boxes)
    advisory_html = (
        '<div class="advisory-sent">✅ Farmer advisory · SENT</div>'
        if case.get("advisory_sent")
        else ""
    )
    st.markdown(
        f"""
        <div class="case-card" style="margin-bottom:8px;">
          <div style="display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;align-items:flex-start;">
            <div>
              <span class="case-id-chip">{case["case_id"]}</span>
              &nbsp;<span class="case-id-chip">{case["field_id"]}</span>
              <div style="margin-top:10px;font-size:20px;font-weight:800;color:#f0fdf4;">
                {case['farmer_name']} · {case['crop']}
              </div>
              <div style="margin-top:4px;font-size:12px;color:#8aa396;">
                📍 {case['village']}, {case['taluka']}, {case['district']}
                · Zone: {case['satellite']['zone']}
                · Reported {case['reported_at'].strftime('%d %b, %H:%M')}
              </div>
              <div style="margin-top:2px;font-size:11px;color:#7d9588;">
                🛰️ Centroid {case['satellite']['centroid']['lat']:.4f},
                {case['satellite']['centroid']['lng']:.4f}
                · field-level stress zone (not leaf-exact pin)
              </div>
            </div>
            <div style="text-align:right;">
              {render_risk_badge(case["risk_level"])}
              <div style="margin-top:6px;">{render_status_chip(case["status"])}</div>
              <div style="margin-top:6px;font-size:12px;color:#a7b8ae;">
                Fusion {case['fusion']['score']} · {case['risk_score']}/100
              </div>
              {advisory_html}
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # One complete HTML card per column (no open/close divs around widgets —
    # that pattern creates empty boxes in Streamlit).
    sat_label = "anomaly detected" if case["anomaly_detected"] else "no anomaly"
    sat_cls = "evidence-bad" if case["anomaly_detected"] else "evidence-ok"
    ai_cls = "evidence-bad" if case["image_ai"]["diseaseDetected"] else "evidence-ok"
    fr = case["farmer_report"]
    recv_cls = "evidence-ok" if fr["received"] else "evidence-warn"
    recv_txt = "received" if fr["received"] else "pending"
    w_cls = (
        "evidence-bad"
        if case["weather_risk"] >= 0.7
        else "evidence-warn"
        if case["weather_risk"] >= 0.45
        else "evidence-ok"
    )
    # escape-ish for HTML text from farmer summary
    fr_summary = (
        str(fr.get("summary", "—"))
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )

    e1, e2, e3, e4 = st.columns(4)

    with e1:
        st.markdown(
            f"""
            <div class="evidence-box">
              <div class="evidence-title">🛰️ Satellite</div>
              <div class="evidence-value {sat_cls}">{sat_label}</div>
              <div class="evidence-label" style="margin-top:8px;">Anomaly {case['anomaly_score']:.2f}</div>
              <div class="evidence-label">NDVI {case['ndvi_current']} vs hist {case['ndvi_historical']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(float(case["anomaly_score"]))

    with e2:
        st.markdown(
            f"""
            <div class="evidence-box">
              <div class="evidence-title">📷 Image AI</div>
              <div class="evidence-value {ai_cls}">{case['image_prediction']}</div>
              <div class="evidence-label" style="margin-top:8px;">Confidence {case['image_confidence']*100:.0f}%</div>
              <div class="evidence-label">Quality {case['image_quality']} · diseaseDetected={str(case['image_ai']['diseaseDetected']).lower()}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(float(case["image_confidence"]))

    with e3:
        st.markdown(
            f"""
            <div class="evidence-box">
              <div class="evidence-title">🗣️ Farmer report</div>
              <div class="evidence-value {recv_cls}">{recv_txt}</div>
              <div class="evidence-label" style="margin-top:8px;">{fr.get('channel','—')} · lang={fr.get('language','—')}</div>
              <div class="evidence-label" style="margin-top:6px; line-height:1.35;">{fr_summary}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with e4:
        st.markdown(
            f"""
            <div class="evidence-box">
              <div class="evidence-title">🌦️ Weather / context</div>
              <div class="evidence-value {w_cls}">risk signal · {case['weather_risk']:.2f}</div>
              <div class="evidence-label" style="margin-top:8px;">Weather {case['weather_risk']:.0%}</div>
              <div class="evidence-label">Nearby reports: {case['nearby_reports']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(float(case["weather_risk"]))

    st.markdown(
        f"""
        <div class="fusion-strip">
            <div class="evidence-title">⚖️ Fused risk · prototype weights</div>
            <div class="{risk_css_class(case['risk_level'])}" style="font-size:22px; margin-top:4px;">
                {case['risk_level']}&nbsp;&nbsp;·&nbsp;&nbsp;score {case['fusion']['score']}
                &nbsp;&nbsp;·&nbsp;&nbsp;expert_required={str(case['fusion']['expert_required']).lower()}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ----- Demo imagery (senior: demo pics — not live GEE/WhatsApp fetch) -----
    st.markdown("#### 🖼️ Field evidence imagery")
    src_tag = case.get("images_source") or "PENDING"
    if src_tag == "DEMO":
        st.caption("Demo pictures attached for SIH prototype · not live satellite/WhatsApp fetch")
    else:
        st.caption("Imagery pending from satellite / farmer photo pipeline")

    img_l, img_r = st.columns(2)
    sat_path = _resolve_asset(case.get("satellite_image"))
    leaf_path = _resolve_asset(case.get("crop_image"))

    with img_l:
        st.markdown("**🛰️ Satellite / field preview**")
        if sat_path:
            st.image(sat_path, use_container_width=True)
            st.caption(
                f"Zone {case['satellite'].get('zone', '—')} · "
                f"anomaly {case.get('anomaly_score', '—')} · "
                "field-level stress screening only"
            )
        else:
            st.markdown(
                """
                <div class="evidence-box" style="min-height:180px;display:flex;align-items:center;justify-content:center;">
                    <div style="text-align:center;color:#7d9588;">
                        🛰️<br>No satellite preview yet<br>
                        <span style="font-size:11px;">Demo asset or GEE image will appear here</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with img_r:
        st.markdown("**📷 Farmer crop photo**")
        if leaf_path:
            st.image(leaf_path, use_container_width=True)
            st.caption(
                f"{case.get('image_prediction', '—')} · "
                f"conf {case.get('image_confidence', 0):.0%} · "
                f"quality {case.get('image_quality', '—')}"
            )
        else:
            st.markdown(
                """
                <div class="evidence-box" style="min-height:180px;display:flex;align-items:center;justify-content:center;">
                    <div style="text-align:center;color:#7d9588;">
                        📷<br>No crop photo yet<br>
                        <span style="font-size:11px;">WhatsApp image / demo asset will appear here</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("#### Evidence breakdown")
    st.caption(
        "Prototype weights — Image AI 40% · Satellite 25% · Weather 15% · Nearby reports 20% "
        "(transparent demo weights, not scientifically validated)."
    )
    contrib_df = pd.DataFrame(
        {
            "Signal": list(case["contribution"].keys()),
            "Contribution": list(case["contribution"].values()),
        }
    )
    st.bar_chart(contrib_df.set_index("Signal"), height=180)

    st.markdown("**Diagnostic drivers**")
    for reason in case["reasons"]:
        st.markdown(f"• {reason}")

    with st.expander("📱 Farmer communication (WhatsApp)"):
        wa = (
            str(case.get("whatsapp_advisory_mr", "—"))
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        st.markdown(
            f"""
            <div class="whatsapp-bubble">
              <div class="whatsapp-label">KRISHIDRISHTI → FARMER WHATSAPP</div>
              <div style="margin-top:6px; line-height:1.45;">{wa}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if case.get("advisory_sent"):
            st.success("Advisory status: **SENT** after expert verification.")
        else:
            st.caption(
                "Advisory is staged. It is marked SENT only after expert CONFIRM."
            )

    with st.expander("🕒 Case audit timeline"):
        for entry in case["log"]:
            st.write(f"• {entry}")

    with st.expander("🧾 Integration JSON (case_id contract)"):
        payload = {
            "case_id": case["case_id"],
            "field_id": case["field_id"],
            "satellite": case["satellite"],
            "image_ai": case["image_ai"],
            "fusion": case["fusion"],
            "status": case["status"],
            "advisory_sent": case.get("advisory_sent", False),
        }
        st.json(payload)

    if show_actions and case["status"] == "PENDING":
        st.markdown("#### Expert verification")
        st.caption("CONFIRM · REJECT · FIELD VISIT REQUIRED")
        b1, b2, b3 = st.columns(3)
        if b1.button(
            "✅ CONFIRM",
            key=f"{key_prefix}confirm_{case['case_id']}",
            use_container_width=True,
        ):
            update_case_status(
                case["case_id"], "VERIFIED", advisory=True, action="CONFIRM"
            )
            st.toast(f"{case['case_id']} VERIFIED · farmer advisory SENT", icon="✅")
            st.rerun()
        if b2.button(
            "❌ REJECT",
            key=f"{key_prefix}reject_{case['case_id']}",
            use_container_width=True,
        ):
            update_case_status(case["case_id"], "REJECTED", action="REJECT")
            st.toast(f"{case['case_id']} REJECTED", icon="❌")
            st.rerun()
        if b3.button(
            "🚶 FIELD VISIT REQUIRED",
            key=f"{key_prefix}visit_{case['case_id']}",
            use_container_width=True,
        ):
            update_case_status(
                case["case_id"], "FIELD_VISIT_REQUIRED", action="FIELD_VISIT_REQUIRED"
            )
            st.toast(f"{case['case_id']} → FIELD VISIT REQUIRED", icon="📋")
            st.rerun()
    elif case["status"] != "PENDING":
        msg = f"Case resolution: **{case['status']}**"
        if case.get("advisory_sent"):
            msg += " · Farmer advisory: **SENT**"
        st.success(msg)


def render_footer(mode: str = "officer"):
    label = "Officer Command" if mode == "officer" else "Expert Diagnostics"
    st.markdown(
        f"""
        <div class="app-footer">
            KRISHIDRISHTI · SIH 26131 · {label} ·
            TARGET → VERIFY → FUSE → ACT · Prototype demo
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    """
    <div class="brand-box">
        <div class="brand-name">🌿 KrishiDrishti</div>
        <div class="brand-subtitle">Crop Surveillance · SIH 26131</div>
        <div class="tricolor-line"></div>
    </div>
    """,
    unsafe_allow_html=True,
)

user_info = st.session_state.get("user", {})
st.sidebar.markdown(
    f"""
    <div class="user-profile-card">
        <b style="color:#f0fdf4;">👤 Active session</b><br>
        <span style="color:#86efac; font-weight:600;">{user_info.get('role', 'User')}</span><br>
        <span style="color:#9cb0a5;">📞 {user_info.get('phone', 'N/A')}</span><br>
        <span style="color:#7d9588; font-size:11px;">📍 {user_info.get('address', 'Maharashtra')}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

if st.sidebar.button("🚪 Logout", use_container_width=True):
    st.session_state.authenticated = False
    st.session_state.user = None
    st.rerun()

st.sidebar.markdown(
    '<div class="sidebar-section">View navigation</div>', unsafe_allow_html=True
)

default_index = 0 if "Officer" in user_info.get("role", "") else 1
role = st.sidebar.radio(
    "Switch Portal Mode:",
    ["🛡️ Officer Command View", "🔬 Expert Diagnostic View"],
    index=default_index,
    label_visibility="collapsed",
)

st.sidebar.markdown(
    '<div class="sidebar-section">Pipeline actions</div>', unsafe_allow_html=True
)

if st.sidebar.button("📡 Ingest mock farmer report", use_container_width=True):
    simulate_new_case()
    st.sidebar.success("New case injected.")
    st.rerun()

if st.sidebar.button("🔄 Refresh data", use_container_width=True):
    load_cases(force_refresh=True)
    st.rerun()

if st.sidebar.button("♻️ Reset demo cases", use_container_width=True):
    reset_cases()
    st.rerun()

# active_view is resolved after role radio — sidebar status refreshed below load
_sidebar_status_slot = st.sidebar.empty()

if st.session_state.get("last_action"):
    st.sidebar.info(f"Last action: {st.session_state.last_action}")

# ============================================================
# LOAD DATA (LIVE Cases API → mock fallback)
# ============================================================

cases = load_cases()
clusters = detect_disease_clusters(cases)

# Resolve which portal is live for status pills
_active_view = "officer" if role == "🛡️ Officer Command View" else "expert"
_data_src = st.session_state.get("data_source", "MOCK")

with _sidebar_status_slot.container():
    st.markdown(
        '<div class="sidebar-section">System status</div>', unsafe_allow_html=True
    )
    if _data_src == "LIVE":
        st.markdown("🟢 **Cases API** · Live")
        base = os.getenv("KRISHI_CASES_API", "http://127.0.0.1:8000")
        st.caption(base)
    else:
        st.markdown("🟡 **Cases API** · Mock fallback")
        err = st.session_state.get("api_error")
        if err:
            st.caption(f"API: {err[:80]}")
    for mod, state in get_pipeline_health(_active_view).items():
        _, label = _health_label(state)
        if label == "Active":
            icon = "🟢"
        elif label == "Demo":
            icon = "🟡"
        elif label == "Standby":
            icon = "⚪"
        else:
            icon = "🔴"
        st.markdown(f"{icon} **{mod}** · {label}")

# ============================================================
# 1. OFFICER COMMAND VIEW
# ============================================================

if role == "🛡️ Officer Command View":
    st.markdown(
        '<div class="main-title">🛡️ Officer Command Center</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="subtitle">District surveillance · outbreak map · case verification · field action</div>',
        unsafe_allow_html=True,
    )

    render_pipeline_health_strip("officer")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Cluster alerts
    if clusters:
        for cl in clusters:
            villages_str = ", ".join(cl["villages"])
            diseases_str = ", ".join(cl["diseases"])
            crops_str = ", ".join(cl["crops"])
            is_critical = cl["severity"] == "CRITICAL OUTBREAK"
            banner_cls = "cluster-alert-banner" if is_critical else "cluster-alert-banner emerging"
            title_color = "#fecaca" if is_critical else "#fde68a"

            st.markdown(
                f"""
                <div class="{banner_cls}">
                    <div class="cluster-alert-title" style="color:{title_color} !important;">
                        🚨 {cl['severity']} — {cl['taluka'].upper()} · {cl['district']}
                        <span class="cluster-badge">{cl['total_cases']} active farms</span>
                    </div>
                    <div style="margin-top: 10px; font-size: 13.5px; line-height: 1.55; color:#f1f5f4;">
                        <b>Crop:</b> {crops_str} &nbsp;·&nbsp;
                        <b>Disease signal:</b> <span style="color:{title_color};">{diseases_str}</span> &nbsp;·&nbsp;
                        <b>High risk:</b> {cl['high_risk_cases']}<br>
                        <b>Villages:</b> {villages_str}
                    </div>
                    <div style="margin-top: 8px; font-size: 11.5px; color:#d4d4d8;">
                        Spatial clustering: nearby HIGH/MEDIUM farms in shared taluka or proximity.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            act_col1, act_col2, act_col3 = st.columns([1.5, 1.5, 1])
            with act_col1:
                if st.button(
                    f"📢 Broadcast precaution ({cl['taluka']})",
                    key=f"cl_broadcast_{cl['cluster_id']}",
                    use_container_width=True,
                ):
                    st.toast(
                        f"Precaution advisory queued for {villages_str} (demo toast)",
                        icon="📲",
                    )
            with act_col2:
                if st.button(
                    "🚜 Mobilize KVK field team",
                    key=f"cl_kvk_{cl['cluster_id']}",
                    use_container_width=True,
                ):
                    st.toast(
                        f"Inspection order logged for KVK · {cl['taluka']} (demo)",
                        icon="📝",
                    )
            with act_col3:
                st.caption(f"`{cl['cluster_id']}`")
    else:
        st.success("Normal surveillance — no high-risk disease clusters in view.")

    st.markdown("---")

    # Filters
    st.markdown("### 🗺️ Geographic filters")
    districts = ["All"] + sorted(set(c["district"] for c in cases))
    f1, f2, f3 = st.columns(3)

    sel_district = f1.selectbox("District", districts)
    filtered = (
        cases if sel_district == "All" else [c for c in cases if c["district"] == sel_district]
    )

    talukas = ["All"] + sorted(set(c["taluka"] for c in filtered))
    sel_taluka = f2.selectbox("Taluka", talukas)
    if sel_taluka != "All":
        filtered = [c for c in filtered if c["taluka"] == sel_taluka]

    villages = ["All"] + sorted(set(c["village"] for c in filtered))
    sel_village = f3.selectbox("Village", villages)
    if sel_village != "All":
        filtered = [c for c in filtered if c["village"] == sel_village]

    # Overview KPIs — handbook §08
    total_cases = len(filtered)
    high_risk = len([c for c in filtered if c["risk_level"] == "HIGH"])
    pending_verification = len([c for c in filtered if c["status"] == "PENDING"])
    verified_cases = len([c for c in filtered if c["status"] == "VERIFIED"])
    field_visit_cases = len(
        [c for c in filtered if c["status"] == "FIELD_VISIT_REQUIRED"]
    )

    st.markdown("---")
    st.markdown("### 📊 Overview")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total cases", total_cases)
    k2.metric("High-risk cases", high_risk)
    k3.metric("Pending verification", pending_verification)
    k4.metric("Verified cases", verified_cases)
    k5.metric("Field visit cases", field_visit_cases)

    # Map
    st.markdown("---")
    m_head_col, m_mode_col = st.columns([2, 2.6])
    with m_head_col:
        st.markdown("### 📍 Outbreak map")
        st.caption("District / field locations · risk markers · affected fields")
    with m_mode_col:
        map_mode = st.radio(
            "Map layers",
            ["🔥 Density heatmap", "📍 Farm points", "🗺️ Combined"],
            horizontal=True,
            label_visibility="collapsed",
        )

    if filtered:
        map_df = pd.DataFrame(
            [
                {
                    "lat": c["lat"],
                    "lon": c["lon"],
                    "farmer_name": c["farmer_name"],
                    "village": c["village"],
                    "taluka": c["taluka"],
                    "crop": c["crop"],
                    "image_prediction": c["image_prediction"],
                    "risk_score": c["risk_score"],
                    "risk_level": c["risk_level"],
                    "status": c["status"],
                    "case_id": c["case_id"],
                }
                for c in filtered
            ]
        )
        risk_color = {
            "HIGH": [248, 113, 113, 230],
            "MEDIUM": [251, 191, 36, 220],
            "LOW": [74, 222, 128, 210],
        }
        map_df["color"] = map_df["risk_level"].map(risk_color)
        map_df["radius"] = map_df["risk_score"].apply(lambda s: 650 + s * 24)

        center_lat = float(map_df["lat"].mean())
        center_lon = float(map_df["lon"].mean())
        deck_layers = []

        if "heatmap" in map_mode.lower() or "Combined" in map_mode:
            deck_layers.append(
                pdk.Layer(
                    "HeatmapLayer",
                    data=map_df,
                    get_position="[lon, lat]",
                    get_weight="risk_score",
                    radius_pixels=70,
                    intensity=1.9,
                    threshold=0.08,
                    color_range=[
                        [34, 197, 94, 70],
                        [234, 179, 8, 150],
                        [249, 115, 22, 200],
                        [239, 68, 68, 235],
                        [153, 27, 27, 255],
                    ],
                )
            )

        if "points" in map_mode.lower() or "Farm" in map_mode or "Combined" in map_mode:
            deck_layers.append(
                pdk.Layer(
                    "ScatterplotLayer",
                    data=map_df,
                    get_position="[lon, lat]",
                    get_fill_color="color",
                    get_radius="radius",
                    pickable=True,
                    opacity=0.88,
                    stroked=True,
                    get_line_color=[255, 255, 255],
                    line_width_min_pixels=1.5,
                )
            )

        if clusters and ("Combined" in map_mode or "points" in map_mode.lower() or "Farm" in map_mode):
            cluster_df = pd.DataFrame(
                [
                    {
                        "lat": c["center_lat"],
                        "lon": c["center_lon"],
                        "radius": 3000,
                        "color": [239, 68, 68, 55]
                        if c["severity"] == "CRITICAL OUTBREAK"
                        else [245, 158, 11, 48],
                        "outline": [239, 68, 68, 240]
                        if c["severity"] == "CRITICAL OUTBREAK"
                        else [245, 158, 11, 220],
                    }
                    for c in clusters
                ]
            )
            deck_layers.append(
                pdk.Layer(
                    "ScatterplotLayer",
                    data=cluster_df,
                    get_position="[lon, lat]",
                    get_fill_color="color",
                    get_radius="radius",
                    stroked=True,
                    get_line_color="outline",
                    line_width_min_pixels=2.5,
                    opacity=0.55,
                    pickable=False,
                )
            )

        st.pydeck_chart(
            pdk.Deck(
                map_style=None,
                initial_view_state=pdk.ViewState(
                    latitude=center_lat,
                    longitude=center_lon,
                    zoom=8.3,
                    pitch=35,
                ),
                layers=deck_layers,
                tooltip={
                    "text": (
                        "{case_id}\n"
                        "Farmer: {farmer_name}\n"
                        "Village: {village}, {taluka}\n"
                        "Crop: {crop}\n"
                        "Diagnosis: {image_prediction}\n"
                        "Risk: {risk_score}/100 ({risk_level})\n"
                        "Status: {status}"
                    )
                },
            ),
            use_container_width=True,
        )
        st.markdown(
            '<div class="map-caption">🔥 Heatmap = concentration &nbsp;·&nbsp; '
            "🔴 Pins sized by fused risk &nbsp;·&nbsp; "
            "Halo rings = detected clusters</div>",
            unsafe_allow_html=True,
        )
    else:
        st.warning("No cases match the selected filters.")

    # Priority queue
    st.markdown("---")
    st.markdown("### 🚨 Priority action queue")
    if filtered:
        priority_df = pd.DataFrame(
            [
                {
                    "Case ID": c["case_id"],
                    "Farmer": c["farmer_name"],
                    "Village": c["village"],
                    "Taluka": c["taluka"],
                    "Crop": c["crop"],
                    "Disease": c["image_prediction"],
                    "Risk": c["risk_level"],
                    "Score": c["risk_score"],
                    "Status": c["status"],
                    "Advisory": "SENT" if c.get("advisory_sent") else "—",
                }
                for c in sorted(filtered, key=lambda x: x["risk_score"], reverse=True)
            ]
        )
        st.dataframe(priority_df, use_container_width=True, hide_index=True)

    # Case detail
    st.markdown("---")
    st.markdown("### 🔎 Case detail — evidence & action")
    if filtered:
        case_labels = {
            f"{c['case_id']} · {c['farmer_name']} — {c['village']} "
            f"({c['crop']}: {c['image_prediction']} · {c['risk_level']} · {c['status']})": c
            for c in filtered
        }
        label_list = list(case_labels.keys())
        default_ix = 0
        for i, lab in enumerate(label_list):
            if lab.startswith("CASE-001"):
                default_ix = i
                break
        chosen_label = st.selectbox(
            "Select case",
            label_list,
            index=default_ix,
            key="officer_inspect_select",
        )
        selected_case = case_labels[chosen_label]
        render_case_card(
            selected_case,
            show_actions=(selected_case["status"] == "PENDING"),
            key_prefix="officer_",
        )

    render_footer("officer")

# ============================================================
# 2. EXPERT DIAGNOSTIC VIEW
# ============================================================

elif role == "🔬 Expert Diagnostic View":
    st.markdown(
        '<div class="main-title">🔬 Expert Diagnostic Workspace</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="subtitle">Evidence triage · confirm or escalate · trigger farmer advisory on verify</div>',
        unsafe_allow_html=True,
    )

    render_pipeline_health_strip("expert")

    st.markdown(
        """
        <div class="expert-hero">
            <div class="expert-hero-title">Triage workflow</div>
            <div class="expert-hero-body">
                Review multi-signal evidence for each pending case, then decide.
                Confirming a case marks it <b>VERIFIED</b> and stages the farmer advisory as <b>SENT</b>.
            </div>
            <div style="margin-top:12px;">
                <span class="triage-step">1 · Open case</span>
                <span class="triage-step">2 · Read evidence</span>
                <span class="triage-step">3 · CONFIRM / REJECT / FIELD VISIT</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    pending = [c for c in cases if c["status"] == "PENDING"]
    resolved = [c for c in cases if c["status"] != "PENDING"]
    high_pending = len([c for c in pending if c["risk_level"] == "HIGH"])
    expert_needed = len([c for c in pending if c.get("needs_expert")])

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Pending queue", len(pending))
    m2.metric("High-risk pending", high_pending)
    m3.metric("Expert-required", expert_needed)
    m4.metric("Verified today", len([c for c in resolved if c["status"] == "VERIFIED"]))
    m5.metric(
        "Field visits",
        len([c for c in resolved if c["status"] == "FIELD_VISIT_REQUIRED"]),
    )

    st.markdown("---")

    left_col, right_col = st.columns([1.15, 2.0])

    with left_col:
        st.markdown("### 📋 Review queue")
        sort_option = st.selectbox(
            "Sort by",
            [
                "Highest Risk Score First",
                "Most Recent Reports First",
                "Expert-required first",
                "By Crop (Soybean First)",
            ],
            key="expert_sort",
        )

        if sort_option == "Highest Risk Score First":
            pending = sorted(pending, key=lambda c: c["risk_score"], reverse=True)
        elif sort_option == "Most Recent Reports First":
            pending = sorted(pending, key=lambda c: c["reported_at"], reverse=True)
        elif sort_option == "Expert-required first":
            pending = sorted(
                pending, key=lambda c: (not c["needs_expert"], -c["risk_score"])
            )
        else:
            pending = sorted(
                pending, key=lambda c: (c["crop"] != "Soybean", -c["risk_score"])
            )

        if not pending:
            st.info("Queue clear. Ingest a mock report from the sidebar, or reset demo cases.")
            selected_expert_case = None
        else:
            # Visual queue strip
            queue_html = ""
            for c in pending[:8]:
                tier = c["risk_level"].lower()
                queue_html += (
                    f'<div class="expert-queue-item {tier}">'
                    f'<div><b style="color:#f0fdf4;">{c["case_id"]}</b><br>'
                    f'<span style="font-size:12px;color:#9cb0a5;">{c["farmer_name"]} · {c["crop"]}</span><br>'
                    f'<span style="font-size:11px;color:#7d9588;">{c["village"]} · {c["image_prediction"]}</span></div>'
                    f'<div style="text-align:right;">'
                    f'<span class="risk-badge badge-{tier}">{c["risk_level"]}</span><br>'
                    f'<span style="font-size:12px;color:#a7b8ae;">{c["risk_score"]}/100</span></div>'
                    f"</div>"
                )
            st.markdown(queue_html, unsafe_allow_html=True)
            if len(pending) > 8:
                st.caption(f"+ {len(pending) - 8} more in queue")

            case_labels = {
                f"{c['case_id']} · {c['farmer_name']} — {c['crop']} · {c['village']} · {c['risk_level']}": c
                for c in pending
            }
            elabels = list(case_labels.keys())
            eix = 0
            for i, lab in enumerate(elabels):
                if lab.startswith("CASE-001"):
                    eix = i
                    break
            chosen_label = st.selectbox(
                "Open case for decision",
                elabels,
                index=eix,
                key="expert_case_selector",
            )
            selected_expert_case = case_labels[chosen_label]

    with right_col:
        st.markdown("### 🧪 Evidence bench")
        if not pending or selected_expert_case is None:
            st.markdown(
                """
                <div class="expert-hero" style="min-height:220px; display:flex; align-items:center; justify-content:center;">
                    <div class="expert-hero-body" style="text-align:center;">
                        No pending case selected.<br>
                        When the queue has items, open one on the left to review evidence and act.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            c = selected_expert_case
            # Compact decision header
            st.markdown(
                f"""
                <div class="expert-hero" style="padding:14px 18px; margin-bottom:12px;">
                    <div style="display:flex; justify-content:space-between; flex-wrap:wrap; gap:8px; align-items:center;">
                        <div>
                            <span class="case-id-chip">{c['case_id']}</span>
                            &nbsp;<span class="case-id-chip">{c['field_id']}</span>
                            <div style="margin-top:8px; font-size:18px; font-weight:800; color:#f0fdf4;">
                                {c['farmer_name']} · {c['crop']}
                            </div>
                            <div style="font-size:12px; color:#8aa396; margin-top:2px;">
                                {c['village']}, {c['taluka']}, {c['district']} · {c['image_prediction']}
                            </div>
                        </div>
                        <div style="text-align:right;">
                            {render_risk_badge(c['risk_level'])}
                            <div style="margin-top:6px; font-size:13px; color:#a7b8ae;">
                                Fusion {c['fusion']['score']} · score {c['risk_score']}/100
                            </div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            render_case_card(
                selected_expert_case,
                show_actions=True,
                key_prefix="expert_selected_",
            )

    if resolved:
        st.markdown("---")
        st.markdown("### 📁 Resolved archive")
        r1, r2, r3 = st.columns(3)
        r1.metric("Verified", len([c for c in resolved if c["status"] == "VERIFIED"]))
        r2.metric("Rejected", len([c for c in resolved if c["status"] == "REJECTED"]))
        r3.metric(
            "Field visit",
            len([c for c in resolved if c["status"] == "FIELD_VISIT_REQUIRED"]),
        )
        with st.expander(f"View {len(resolved)} resolved / audited cases"):
            for case in resolved:
                render_case_card(
                    case, show_actions=False, key_prefix="expert_resolved_"
                )

    render_footer("expert")

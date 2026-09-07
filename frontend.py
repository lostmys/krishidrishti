import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
from datetime import datetime, timedelta
import re

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="KrishiDrishti | Agricultural Intelligence",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS & ENHANCED THEME
# ============================================================

st.markdown("""
<style>
    /* --------------------------------------------------------
       GLOBAL BACKGROUND & ATMOSPHERE
       -------------------------------------------------------- */
    .stApp {
        background:
            radial-gradient(circle at 85% 5%, rgba(34, 197, 94, 0.15), transparent 35%),
            radial-gradient(circle at 15% 90%, rgba(245, 158, 11, 0.10), transparent 30%),
            linear-gradient(135deg, #06130d 0%, #0a2118 45%, #061612 100%);
        color: #f8fafc;
    }

    /* Subtle agricultural grid overlay */
    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        opacity: 0.12;
        background-image:
            linear-gradient(rgba(34,197,94,0.12) 1px, transparent 1px),
            linear-gradient(90deg, rgba(34,197,94,0.12) 1px, transparent 1px);
        background-size: 60px 60px;
        mask-image: linear-gradient(to bottom, black, transparent 85%);
        z-index: 0;
    }

    /* Typography */
    h1, h2, h3, h4, h5, h6, p, span, label, div[data-testid="stMarkdownContainer"] {
        color: #f8fafc !important;
    }

    .main-title {
        font-size: 32px;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin-bottom: 2px;
    }

    .subtitle {
        color: #a7b8ae !important;
        font-size: 15px;
        margin-bottom: 16px;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #04110c 0%, #071c13 55%, #05140e 100%) !important;
        border-right: 1px solid rgba(34,197,94,0.22);
    }

    .brand-box {
        padding: 10px 6px 16px 6px;
        border-bottom: 1px solid rgba(148,163,184,0.15);
        margin-bottom: 16px;
    }

    .brand-name {
        font-size: 23px;
        font-weight: 800;
        color: #f8fafc !important;
    }

    .brand-subtitle {
        font-size: 11px;
        color: #9caea4 !important;
        margin-top: -2px;
    }

    .tricolor-line {
        height: 3px;
        width: 130px;
        margin-top: 8px;
        background: linear-gradient(90deg, #f97316 0%, #f8fafc 50%, #22c55e 100%);
        border-radius: 20px;
    }

    .user-profile-card {
        background: rgba(15, 38, 27, 0.85);
        border: 1px solid rgba(74, 222, 128, 0.25);
        border-radius: 12px;
        padding: 12px;
        margin-bottom: 16px;
        font-size: 12px;
    }

    /* Metric Cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(15, 37, 27, 0.95), rgba(7, 24, 17, 0.92));
        border: 1px solid rgba(74,222,128,0.22);
        padding: 16px;
        border-radius: 14px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.25);
    }

    div[data-testid="stMetric"] label {
        color: #9eb1a6 !important;
        font-size: 12px;
        font-weight: 600;
    }

    /* --------------------------------------------------------
       CLUSTER ALERT BANNER (OFFICER DASHBOARD)
       -------------------------------------------------------- */
    .cluster-alert-banner {
        background: linear-gradient(135deg, rgba(153, 27, 27, 0.92) 0%, rgba(88, 28, 28, 0.95) 100%);
        border: 2px solid #ef4444;
        border-radius: 14px;
        padding: 18px 20px;
        margin-bottom: 22px;
        box-shadow: 0 10px 30px rgba(239, 68, 68, 0.28);
        animation: pulse-border 2.5s infinite;
    }

    @keyframes pulse-border {
        0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.5); }
        70% { box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); }
        100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }

    .cluster-alert-title {
        font-size: 18px;
        font-weight: 800;
        color: #fecaca !important;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .cluster-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 800;
        background: rgba(239, 68, 68, 0.25);
        border: 1px solid #f87171;
        color: #fca5a5 !important;
    }

    /* Case Card */
    .case-card {
        background: linear-gradient(145deg, rgba(12,32,23,0.96), rgba(6,20,14,0.94));
        border: 1px solid rgba(74,222,128,0.20);
        padding: 20px;
        border-radius: 16px;
        margin-bottom: 16px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.20);
    }

    .risk-high { color: #fb7185 !important; font-weight: 800; }
    .risk-medium { color: #fbbf24 !important; font-weight: 800; }
    .risk-low { color: #4ade80 !important; font-weight: 800; }

    .risk-badge {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 800;
    }
    .badge-high { background: rgba(239,68,68,0.15); border: 1px solid rgba(248,113,113,0.4); color: #fb7185 !important; }
    .badge-medium { background: rgba(245,158,11,0.15); border: 1px solid rgba(251,191,36,0.4); color: #fbbf24 !important; }
    .badge-low { background: rgba(34,197,94,0.12); border: 1px solid rgba(74,222,128,0.35); color: #4ade80 !important; }

    .evidence-box {
        background: rgba(2,12,8,0.5);
        border: 1px solid rgba(148,163,184,0.14);
        border-radius: 12px;
        padding: 15px;
        min-height: 145px;
    }
    .evidence-title { font-size: 13px; font-weight: 700; margin-bottom: 8px; }
    .evidence-value { font-size: 17px; font-weight: 750; }
    .evidence-label { font-size: 11px; color: #899d92 !important; }

    /* WhatsApp advisory bubble */
    .whatsapp-bubble {
        background: linear-gradient(135deg, rgba(7,94,84,0.85), rgba(6,70,63,0.85));
        border-radius: 12px;
        padding: 14px;
        color: #f8fafc !important;
        border-left: 4px solid #25D366;
    }
    .whatsapp-label { color: #86efac !important; font-size: 11px; font-weight: 700; margin-bottom: 5px; }

    /* Health pills */
    .health-pill {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 11px;
        margin-right: 6px;
        margin-bottom: 6px;
        font-weight: 650;
    }
    .health-up { background: rgba(34,197,94,0.12); color: #86efac !important; border: 1px solid rgba(34,197,94,0.3); }
    .health-down { background: rgba(248,113,113,0.12); color: #fca5a5 !important; border: 1px solid rgba(248,113,113,0.3); }

    /* Login Portal Card */
    .login-container {
        max-width: 540px;
        margin: 40px auto;
        padding: 32px;
        background: linear-gradient(145deg, rgba(13, 35, 25, 0.95), rgba(7, 23, 16, 0.96));
        border: 1px solid rgba(74, 222, 128, 0.3);
        border-radius: 20px;
        box-shadow: 0 15px 45px rgba(0, 0, 0, 0.4);
    }

    .login-title {
        font-size: 26px;
        font-weight: 800;
        margin-bottom: 4px;
    }

    .login-sub {
        font-size: 13px;
        color: #9cb0a5 !important;
        margin-bottom: 24px;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 9px;
        border: 1px solid rgba(74,222,128,0.3);
        background: rgba(15,40,27,0.9);
        color: #f8fafc;
        font-weight: 650;
        transition: 0.2s ease;
    }
    .stButton > button:hover {
        border-color: rgba(74,222,128,0.6);
        background: rgba(22,65,39,0.95);
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# MOCK BACKEND DATA LAYER
# ============================================================

MAHARASHTRA_LOCATIONS = [
    {"district": "Yavatmal", "taluka": "Pusad", "village": "Digras Wadi", "lat": 19.90, "lon": 77.57},
    {"district": "Yavatmal", "taluka": "Pusad", "village": "Shembalpimpri", "lat": 19.86, "lon": 77.52},
    {"district": "Yavatmal", "taluka": "Darwha", "village": "Shirpur", "lat": 20.22, "lon": 77.78},
    {"district": "Amravati", "taluka": "Achalpur", "village": "Paratwada", "lat": 21.27, "lon": 77.51},
    {"district": "Amravati", "taluka": "Achalpur", "village": "Chandur Bazar", "lat": 21.24, "lon": 77.60},
    {"district": "Amravati", "taluka": "Daryapur", "village": "Anjangaon", "lat": 21.05, "lon": 77.31},
    {"district": "Wardha", "taluka": "Hinganghat", "village": "Ajansara", "lat": 20.55, "lon": 78.83},
    {"district": "Nagpur", "taluka": "Kalmeshwar", "village": "Mohpa", "lat": 21.28, "lon": 78.85},
    {"district": "Nanded", "taluka": "Bhokar", "village": "Umri", "lat": 19.15, "lon": 77.65},
    {"district": "Nanded", "taluka": "Kandhar", "village": "Fulwal", "lat": 18.83, "lon": 77.15},
]

FARMER_NAMES = [
    "Ramesh Patil", "Suresh Jadhav", "Vandana Rathod", "Ganesh Deshmukh",
    "Kavita More", "Ashok Kale", "Meena Shinde", "Prakash Wagh", "Baban Shinde", "Sunita Gawande"
]

DISEASE_CLASSES = [
    "Bacterial Blight", "Leaf Spot (Fungal)", "Curl Virus", "Rust", "Healthy"
]

RISK_REASON_POOL = [
    "Leaf lesions match bacterial/fungal profile",
    "Vegetation satellite anomaly detected (>25% drop)",
    "Consecutive high humidity (>85%) in forecast",
    "3 nearby outbreak reports in 5km radius",
    "Historical hotspot in this village",
    "Pest / Hopper vector activity reported nearby",
]

def _generate_mock_case(case_id, seed):
    rng = np.random.default_rng(seed)
    loc = MAHARASHTRA_LOCATIONS[seed % len(MAHARASHTRA_LOCATIONS)]
    confidence = round(float(rng.uniform(0.60, 0.98)), 2)
    prediction = DISEASE_CLASSES[seed % len(DISEASE_CLASSES)]
    ndvi_current = round(float(rng.uniform(0.32, 0.65)), 2)
    ndvi_historical = round(ndvi_current + float(rng.uniform(0.08, 0.25)), 2)
    anomaly_score = round(float(rng.uniform(0.35, 0.95)), 2)

    image_w = round(confidence * 40, 1)
    satellite_w = round(anomaly_score * 30, 1)
    weather_w = round(float(rng.uniform(6, 18)), 1)
    remaining = max(0, 100 - image_w - satellite_w - weather_w)
    reports_w = round(remaining, 1)

    risk_score = int(min(98, image_w + satellite_w + weather_w + reports_w))
    if risk_score >= 70:
        risk_level = "HIGH"
    elif risk_score >= 40:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    n_reasons = 2 if risk_level == "LOW" else 3 if risk_level == "MEDIUM" else 4
    reasons = list(rng.choice(RISK_REASON_POOL, size=n_reasons, replace=False))
    report_time = datetime.now() - timedelta(hours=int(rng.integers(1, 48)))

    crop = "Cotton" if seed % 2 == 0 else "Soybean"
    crop_mr = "कापूस" if crop == "Cotton" else "सोयाबीन"

    return {
        "case_id": case_id,
        "farmer_name": FARMER_NAMES[seed % len(FARMER_NAMES)],
        "crop": crop,
        "district": loc["district"],
        "taluka": loc["taluka"],
        "village": loc["village"],
        "lat": loc["lat"] + float(rng.uniform(-0.02, 0.02)),
        "lon": loc["lon"] + float(rng.uniform(-0.02, 0.02)),
        "field_id": f"F-{100 + seed}",
        "image_prediction": prediction,
        "image_confidence": confidence,
        "image_quality": "good" if confidence > 0.60 else "poor",
        "ndvi_current": ndvi_current,
        "ndvi_historical": ndvi_historical,
        "anomaly_score": anomaly_score,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "reasons": reasons,
        "contribution": {
            "Image evidence": image_w,
            "Satellite anomaly": satellite_w,
            "Weather risk": weather_w,
            "Spatial clustering": reports_w
        },
        "needs_expert": risk_level != "LOW",
        "status": "PENDING",
        "reported_at": report_time,
        "log": [
            f"{report_time.strftime('%H:%M')} - Leaf image received via WhatsApp",
            "Automated quality check passed",
            "Multi-modal fusion (Image AI + GEE Satellite + Geospatial) completed"
        ],
        "whatsapp_advisory_mr": (
            f"⚠️ सूचना: आपल्या {crop_mr} शेतात ({loc['village']}) असामान्य बदल व रोगाची लक्षणे आढळली आहेत. "
            f"अधिक माहितीसाठी कृषी सहाय्यकांशी संपर्क साधा."
        )
    }

def get_mock_cases(n=10):
    if "cases" not in st.session_state:
        st.session_state.cases = [_generate_mock_case(i, i * 7 + 3) for i in range(n)]
    return st.session_state.cases

def update_case_status(case_id, new_status):
    for case in st.session_state.cases:
        if case["case_id"] == case_id:
            case["status"] = new_status
            case["log"].append(
                f"{datetime.now().strftime('%H:%M')} - Officer/Expert marked case as {new_status}"
            )
    st.session_state.last_action = f"Case {case_id} → {new_status}"

def simulate_new_case():
    new_id = max(c["case_id"] for c in st.session_state.cases) + 1
    st.session_state.cases.append(_generate_mock_case(new_id, new_id * 13 + 7))

def get_pipeline_health():
    return {
        "Image AI Model": True,
        "Satellite GEE / NDVI": True,
        "Risk Fusion Engine": True,
        "Crop Registry": True,
        "WhatsApp Gateway": True,
    }

# ============================================================
# CLUSTER DETECTION ENGINE
# ============================================================

def detect_disease_clusters(case_list, distance_threshold_deg=0.18, min_cluster_cases=2):
    """
    Identifies geographic disease clusters from high and medium risk reports.
    Groups reports that share spatial proximity or the same taluka with active symptoms.
    """
    active_cases = [c for c in case_list if c["risk_level"] in ("HIGH", "MEDIUM")]
    clusters = []
    visited = set()

    for i, base in enumerate(active_cases):
        if base["case_id"] in visited:
            continue
        group = [base]
        visited.add(base["case_id"])

        for other in active_cases:
            if other["case_id"] in visited:
                continue
            dist = np.sqrt((base["lat"] - other["lat"])**2 + (base["lon"] - other["lon"])**2)
            same_region = (base["taluka"] == other["taluka"] and base["district"] == other["district"])
            if dist <= distance_threshold_deg or same_region:
                group.append(other)
                visited.add(other["case_id"])

        if len(group) >= min_cluster_cases:
            high_count = sum(1 for c in group if c["risk_level"] == "HIGH")
            avg_lat = float(np.mean([c["lat"] for c in group]))
            avg_lon = float(np.mean([c["lon"] for c in group]))
            crops = list(set(c["crop"] for c in group))
            diseases = list(set(c["image_prediction"] for c in group if c["image_prediction"] != "Healthy"))
            villages = sorted(set(c["village"] for c in group))
            severity = "CRITICAL OUTBREAK" if high_count >= 2 else "EMERGING CONCENTRATION"

            clusters.append({
                "cluster_id": f"CLUST-{len(clusters)+1:02d}",
                "severity": severity,
                "district": group[0]["district"],
                "taluka": group[0]["taluka"],
                "villages": villages,
                "crops": crops,
                "primary_crop": crops[0] if crops else "Cotton",
                "diseases": diseases if diseases else ["Unspecified Anomaly"],
                "total_cases": len(group),
                "high_risk_cases": high_count,
                "center_lat": avg_lat,
                "center_lon": avg_lon,
                "cases": group
            })

    return clusters

# ============================================================
# LOGIN & SESSION GATE
# ============================================================

def render_login_portal():
    st.markdown("""
        <div class="login-container">
            <div style="font-size:36px; margin-bottom: 8px;">🌾</div>
            <div class="login-title">KrishiDrishti Portal Login</div>
            <div class="login-sub">Government of Maharashtra · Crop Surveillance & Agricultural Intelligence</div>
            <div class="tricolor-line" style="margin-bottom: 24px; width: 100%;"></div>
        </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 2.2, 1])
    with c2:
        with st.form("login_form"):
            phone = st.text_input("📱 Mobile / Phone Number*", placeholder="e.g. 9823012345 or +91 98230 12345")
            address = st.text_input("📍 Official Station / Work Address / District*", placeholder="e.g. Sub-Divisional Agriculture Office, Pusad, Yavatmal")
            role_choice = st.selectbox(
                "👤 Select Your Portal Role*",
                [
                    "🛡️ Agricultural Officer (Command & Hotspot Intelligence)",
                    "🔬 AI Diagnostic Expert (Evidence-Based Triage)"
                ]
            )
            submit = st.form_submit_button("🚀 Enter KrishiDrishti Dashboard", use_container_width=True)

            if submit:
                clean_phone = re.sub(r'[\s\-+]', '', phone)
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
                        "login_time": datetime.now()
                    }
                    st.success("Authentication successful! Loading dashboard...")
                    st.rerun()

        st.markdown("---")
        st.caption("⚡ **Demo Quick-Access Profiles:**")
        q1, q2 = st.columns(2)
        if q1.button("🛡️ Officer (Yavatmal)", use_container_width=True):
            st.session_state.authenticated = True
            st.session_state.user = {
                "phone": "+91 98221 44550",
                "address": "District Agriculture Office, Yavatmal (MH)",
                "role": "🛡️ Agricultural Officer (Command & Hotspot Intelligence)",
                "login_time": datetime.now()
            }
            st.rerun()

        if q2.button("🔬 AI Expert (Pusad)", use_container_width=True):
            st.session_state.authenticated = True
            st.session_state.user = {
                "phone": "+91 94230 88990",
                "address": "KVK Agronomy Research Center, Pusad",
                "role": "🔬 AI Diagnostic Expert (Evidence-Based Triage)",
                "login_time": datetime.now()
            }
            st.rerun()

# If not authenticated, render login and stop further rendering
if "authenticated" not in st.session_state or not st.session_state.authenticated:
    render_login_portal()
    st.stop()

# ============================================================
# UI RENDER HELPERS
# ============================================================

def render_pipeline_health_strip():
    health = get_pipeline_health()
    pills = ""
    for module, ok in health.items():
        cls = "health-up" if ok else "health-down"
        status = "Online" if ok else "Offline"
        pills += f'<span class="health-pill {cls}">● {module} · {status}</span>'
    st.markdown(pills, unsafe_allow_html=True)

def risk_css_class(level):
    return {"HIGH": "risk-high", "MEDIUM": "risk-medium", "LOW": "risk-low"}.get(level, "risk-low")

def render_risk_badge(level):
    cls = {"HIGH": "badge-high", "MEDIUM": "badge-medium", "LOW": "badge-low"}.get(level, "badge-low")
    return f'<span class="risk-badge {cls}">{level} RISK</span>'

def render_case_card(case, show_actions=True, key_prefix=""):
    st.markdown('<div class="case-card">', unsafe_allow_html=True)
    top_l, top_r = st.columns([4, 1])

    with top_l:
        st.markdown(f"### 👨‍🌾 {case['farmer_name']} · {case['crop']}")
        st.caption(
            f"📍 {case['village']}, {case['taluka']}, {case['district']} · "
            f"Field {case['field_id']} · Reported {case['reported_at'].strftime('%d %b, %H:%M')}"
        )

    with top_r:
        st.markdown(render_risk_badge(case["risk_level"]), unsafe_allow_html=True)
        st.caption(f"Score: **{case['risk_score']}/100**")
        st.caption(f"Status: `{case['status']}`")

    st.markdown("---")

    e1, e2, e3 = st.columns(3)
    with e1:
        st.markdown('<div class="evidence-box">', unsafe_allow_html=True)
        st.markdown('<div class="evidence-title">📷 IMAGE AI SIGNAL</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="evidence-value">{case["image_prediction"]}</div>', unsafe_allow_html=True)
        st.progress(case["image_confidence"], text=f'Confidence: {case["image_confidence"]*100:.0f}%')
        st.markdown(f'<div class="evidence-label">Image quality: {case["image_quality"].title()}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with e2:
        st.markdown('<div class="evidence-box">', unsafe_allow_html=True)
        st.markdown('<div class="evidence-title">🛰️ SATELLITE / NDVI ANOMALY</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="evidence-value">{case["ndvi_current"]} <span style="font-size:12px;color:#8da197">vs {case["ndvi_historical"]} hist.</span></div>',
            unsafe_allow_html=True
        )
        st.progress(case["anomaly_score"], text=f'Drop signal: {case["anomaly_score"]*100:.0f}%')
        st.markdown('<div class="evidence-label">Macro vegetation screening anomaly</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with e3:
        st.markdown('<div class="evidence-box">', unsafe_allow_html=True)
        st.markdown('<div class="evidence-title">⚖️ FUSED RISK SCORE</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="{risk_css_class(case["risk_level"])}" style="font-size:24px;">{case["risk_score"]}/100</div>', unsafe_allow_html=True)
        st.markdown(f"**{case['risk_level']} RISK TIER**")
        st.caption("Multi-modal decision score")
        st.markdown('</div>', unsafe_allow_html=True)

    # Signal contributions & reasons
    st.markdown("#### ⚖️ Evidence breakdown")
    contrib_df = pd.DataFrame({
        "Signal": list(case["contribution"].keys()),
        "Contribution": list(case["contribution"].values())
    })
    st.bar_chart(contrib_df.set_index("Signal"), height=180)

    st.markdown("**Diagnostic Drivers:**")
    for reason in case["reasons"]:
        st.markdown(f"• {reason}")

    # WhatsApp Advisory Expander
    with st.expander("📱 Farmer Communication (Zero-App Delivery)"):
        st.markdown('<div class="whatsapp-bubble">', unsafe_allow_html=True)
        st.markdown('<div class="whatsapp-label">KRISHIDRISHTI → FARMER WHATSAPP</div>', unsafe_allow_html=True)
        st.markdown(case["whatsapp_advisory_mr"])
        st.markdown('</div>', unsafe_allow_html=True)
        st.caption("Sent automatically in Marathi / Hindi to the farmer's registered number.")

    # Timeline Expander
    with st.expander("🕒 Case Audit Timeline"):
        for entry in case["log"]:
            st.write(f"• {entry}")

    # Expert / Officer Actions
    if show_actions and case["status"] == "PENDING":
        st.markdown("#### 👨‍🔬 Expert Verification Decision")
        b1, b2, b3 = st.columns(3)
        if b1.button("✅ Confirm Diagnosis", key=f"{key_prefix}confirm_{case['case_id']}"):
            update_case_status(case["case_id"], "CONFIRMED")
            st.rerun()
        if b2.button("❌ Reject / Retake Photo", key=f"{key_prefix}reject_{case['case_id']}"):
            update_case_status(case["case_id"], "REJECTED")
            st.rerun()
        if b3.button("🚶 Order Field Visit", key=f"{key_prefix}visit_{case['case_id']}"):
            update_case_status(case["case_id"], "NEEDS_VISIT")
            st.rerun()
    elif case["status"] != "PENDING":
        st.success(f"Case resolution: **{case['status']}**")

    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("""
    <div class="brand-box">
        <div class="brand-name">🌿 KrishiDrishti</div>
        <div class="brand-subtitle">Agricultural Intelligence Platform</div>
        <div class="tricolor-line"></div>
    </div>
""", unsafe_allow_html=True)

# Logged in user profile display
user_info = st.session_state.get("user", {})
st.sidebar.markdown(f"""
    <div class="user-profile-card">
        <b>👤 Active User</b><br>
        <span style="color:#86efac;">{user_info.get('role', 'User')}</span><br>
        📞 {user_info.get('phone', 'N/A')}<br>
        📍 <small>{user_info.get('address', 'Maharashtra')}</small>
    </div>
""", unsafe_allow_html=True)

if st.sidebar.button("🚪 Logout", use_container_width=True):
    st.session_state.authenticated = False
    st.session_state.user = None
    st.rerun()

st.sidebar.markdown('<div class="sidebar-section">View Navigation</div>', unsafe_allow_html=True)

default_index = 0 if "Officer" in user_info.get("role", "") else 1
role = st.sidebar.radio(
    "Switch Portal Mode:",
    ["🛡️ Officer Command View", "🔬 AI Expert Diagnostic View"],
    index=default_index,
    label_visibility="collapsed"
)

st.sidebar.markdown('<div class="sidebar-section">Live Pipeline Actions</div>', unsafe_allow_html=True)

if st.sidebar.button("📡 Ingest Mock Farmer Report", use_container_width=True):
    simulate_new_case()
    st.sidebar.success("New field case injected into pipeline.")
    st.rerun()

if st.sidebar.button("🔄 Refresh Dashboard Data", use_container_width=True):
    st.rerun()

st.sidebar.markdown('<div class="sidebar-section">System Diagnostics</div>', unsafe_allow_html=True)
for mod, ok in get_pipeline_health().items():
    st.sidebar.markdown(f"**{mod}**: {'🟢 Ready' if ok else '🔴 Offline'}")

# ============================================================
# LOAD CASES & DETECT CLUSTERS
# ============================================================

cases = get_mock_cases()
clusters = detect_disease_clusters(cases)

# ============================================================
# 1. OFFICER COMMAND VIEW
# ============================================================

if role == "🛡️ Officer Command View":
    st.markdown('<div class="main-title">🛡️ Agricultural Officer Command Center</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">District → Taluka → Village Outbreak Surveillance & Automated Cluster Detection</div>', unsafe_allow_html=True)

    render_pipeline_health_strip()

    # --------------------------------------------------------
    # AUTOMATED CLUSTER FORMATION ALERT TO OFFICER
    # --------------------------------------------------------
    if clusters:
        for cl in clusters:
            villages_str = ", ".join(cl["villages"])
            diseases_str = ", ".join(cl["diseases"])
            crops_str = ", ".join(cl["crops"])
            border_col = "#ef4444" if cl["severity"] == "CRITICAL OUTBREAK" else "#f59e0b"

            st.markdown(f"""
                <div class="cluster-alert-banner" style="border-color:{border_col};">
                    <div class="cluster-alert-title">
                        🚨 {cl['severity']} DETECTED — {cl['taluka'].upper()} TALUKA ({cl['district']})
                        <span class="cluster-badge">{cl['total_cases']} Active Farms</span>
                    </div>
                    <div style="margin-top: 8px; font-size: 14px; line-height: 1.5;">
                        <b>Crop:</b> {crops_str} &nbsp;|&nbsp;
                        <b>Dominant Symptom / Disease:</b> <span style="color:#fca5a5;">{diseases_str}</span> &nbsp;|&nbsp;
                        <b>High Risk Cases:</b> {cl['high_risk_cases']}<br>
                        <b>Affected Villages:</b> {villages_str}
                    </div>
                    <div style="margin-top: 10px; font-size: 12px; color:#e2e8f0;">
                        <b>Automated Protocol:</b> Spatial clustering criteria satisfied (&gt;1 nearby farms with high-risk NDVI drop and verified disease symptoms).
                    </div>
                </div>
            """, unsafe_allow_html=True)

            act_col1, act_col2, act_col3 = st.columns([1.5, 1.5, 1])
            with act_col1:
                if st.button(f"📢 Broadcast WhatsApp Precaution Alert ({cl['taluka']})", key=f"cl_broadcast_{cl['cluster_id']}"):
                    st.toast(f"✅ Advisory dispatched to all registered farmers in {villages_str} via WhatsApp!", icon="📲")
            with act_col2:
                if st.button(f"🚜 Mobilize KVK Field Inspection Team", key=f"cl_kvk_{cl['cluster_id']}"):
                    st.toast(f"📋 Inspection order logged for KVK officers in {cl['taluka']}!", icon="📝")
            with act_col3:
                st.caption(f"Cluster ID: `{cl['cluster_id']}`")

    else:
        st.success("✅ **Normal Surveillance**: No active high-risk disease clusters detected across your monitored jurisdiction.")

    st.markdown("---")

    # --------------------------------------------------------
    # GEOGRAPHIC FILTER CONTROLS
    # --------------------------------------------------------
    st.markdown("### 🗺️ Geographic Surveillance Filters")
    districts = ["All"] + sorted(set(c["district"] for c in cases))
    f1, f2, f3 = st.columns(3)

    sel_district = f1.selectbox("District", districts)
    filtered = cases if sel_district == "All" else [c for c in cases if c["district"] == sel_district]

    talukas = ["All"] + sorted(set(c["taluka"] for c in filtered))
    sel_taluka = f2.selectbox("Taluka", talukas)
    if sel_taluka != "All":
        filtered = [c for c in filtered if c["taluka"] == sel_taluka]

    villages = ["All"] + sorted(set(c["village"] for c in filtered))
    sel_village = f3.selectbox("Village", villages)
    if sel_village != "All":
        filtered = [c for c in filtered if c["village"] == sel_village]

    # Command KPI row
    high = len([c for c in filtered if c["risk_level"] == "HIGH"])
    medium = len([c for c in filtered if c["risk_level"] == "MEDIUM"])
    low = len([c for c in filtered if c["risk_level"] == "LOW"])
    confirmed = len([c for c in filtered if c["status"] == "CONFIRMED"])

    st.markdown("---")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Cases Monitored", len(filtered))
    k2.metric("🔴 High Risk", high)
    k3.metric("🟡 Medium Risk", medium)
    k4.metric("🟢 Low Risk", low)
    k5.metric("Confirmed Outbreaks", confirmed)

    # --------------------------------------------------------
    # PYDECK MAP WITH DENSITY HEATMAP & SCATTERPLOT
    # --------------------------------------------------------
    st.markdown("---")
    m_head_col, m_mode_col = st.columns([2, 2.5])
    with m_head_col:
        st.markdown("### 📍 Disease Outbreak & Density Map")
    with m_mode_col:
        map_mode = st.radio(
            "Map Layer Mode:",
            ["🔥 Density Heatmap", "📍 Farm Points (Scatter)", "🗺️ Combined (Heatmap + Points)"],
            horizontal=True
        )

    if filtered:
        map_df = pd.DataFrame(filtered)
        risk_color = {
            "HIGH": [248, 113, 113, 220],
            "MEDIUM": [251, 191, 36, 220],
            "LOW": [74, 222, 128, 220]
        }
        map_df["color"] = map_df["risk_level"].map(risk_color)
        map_df["radius"] = map_df["risk_score"].apply(lambda s: 600 + s * 22)

        center_lat = float(map_df["lat"].mean())
        center_lon = float(map_df["lon"].mean())

        deck_layers = []

        # Heatmap Layer
        if "Heatmap" in map_mode or "Combined" in map_mode:
            heatmap_layer = pdk.Layer(
                "HeatmapLayer",
                data=map_df,
                get_position="[lon, lat]",
                get_weight="risk_score",
                radius_pixels=65,
                intensity=1.8,
                threshold=0.08,
                color_range=[
                    [34, 197, 94, 80],    # Low risk (Green)
                    [234, 179, 8, 160],   # Moderate (Yellow)
                    [249, 115, 22, 210],  # High (Orange)
                    [239, 68, 68, 240],   # Critical (Red)
                    [185, 28, 28, 255]    # Outbreak core (Dark Red)
                ]
            )
            deck_layers.append(heatmap_layer)

        # Scatterplot Farm Pins Layer
        if "Scatter" in map_mode or "Combined" in map_mode:
            scatter_layer = pdk.Layer(
                "ScatterplotLayer",
                data=map_df,
                get_position="[lon, lat]",
                get_fill_color="color",
                get_radius="radius",
                pickable=True,
                opacity=0.85,
                stroked=True,
                get_line_color=[255, 255, 255],
                line_width_min_pixels=1.5
            )
            deck_layers.append(scatter_layer)

        # Cluster Halo Layer if clusters detected
        if clusters and ("Combined" in map_mode or "Scatter" in map_mode):
            cluster_df = pd.DataFrame([
                {
                    "lat": c["center_lat"],
                    "lon": c["center_lon"],
                    "radius": 2800,
                    "color": [239, 68, 68, 60] if c["severity"] == "CRITICAL OUTBREAK" else [245, 158, 11, 50],
                    "outline": [239, 68, 68, 240] if c["severity"] == "CRITICAL OUTBREAK" else [245, 158, 11, 220]
                }
                for c in clusters
            ])
            cluster_layer = pdk.Layer(
                "ScatterplotLayer",
                data=cluster_df,
                get_position="[lon, lat]",
                get_fill_color="color",
                get_radius="radius",
                stroked=True,
                get_line_color="outline",
                line_width_min_pixels=2.5,
                opacity=0.6,
                pickable=False
            )
            deck_layers.append(cluster_layer)

        st.pydeck_chart(
            pdk.Deck(
                map_style=None,
                initial_view_state=pdk.ViewState(
                    latitude=center_lat,
                    longitude=center_lon,
                    zoom=8.2,
                    pitch=30
                ),
                layers=deck_layers,
                tooltip={
                    "text": (
                        "Farmer: {farmer_name}\n"
                        "Village: {village}, {taluka}\n"
                        "Crop: {crop}\n"
                        "Diagnosis: {image_prediction}\n"
                        "Risk Score: {risk_score}/100 ({risk_level})\n"
                        "Status: {status}"
                    )
                }
            )
        )
        st.caption("🔥 **Heatmap**: Red/orange hot zones indicate dense disease concentration · 🔴 **Farm Pins**: Bubble size proportional to fused risk score.")
    else:
        st.warning("No cases match the selected filters.")

    # --------------------------------------------------------
    # PRIORITY FIELD INSPECTIONS
    # --------------------------------------------------------
    st.markdown("---")
    st.markdown("### 🚨 Priority Field Action Queue")
    if filtered:
        priority_df = pd.DataFrame(filtered).sort_values("risk_score", ascending=False)
        display_df = priority_df[["farmer_name", "village", "taluka", "crop", "image_prediction", "risk_level", "risk_score", "status"]].copy()
        display_df.columns = ["Farmer Name", "Village", "Taluka", "Crop", "Detected Disease", "Risk Tier", "Risk Score", "Audit Status"]
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    # --------------------------------------------------------
    # CASE EVIDENCE INSPECTION
    # --------------------------------------------------------
    st.markdown("---")
    st.markdown("### 🔎 Inspect Individual Field Evidence")
    if filtered:
        case_labels = {
            f"{c['farmer_name']} — {c['village']} ({c['crop']}: {c['image_prediction']} · {c['risk_level']})": c
            for c in filtered
        }
        chosen_label = st.selectbox("Select specific farm to inspect:", list(case_labels.keys()), key="officer_inspect_select")
        selected_case = case_labels[chosen_label]
        render_case_card(selected_case, show_actions=(selected_case["status"] == "PENDING"), key_prefix="officer_")

# ============================================================
# 2. AI EXPERT DIAGNOSTIC VIEW
# ============================================================

elif role == "🔬 AI Expert Diagnostic View":
    st.markdown('<div class="main-title">🔬 AI Expert Diagnostic Panel</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Multi-Signal Evidence Triage — Verify Leaf AI, Satellite Anomaly, and Environmental Risk</div>', unsafe_allow_html=True)

    render_pipeline_health_strip()

    pending = [c for c in cases if c["status"] == "PENDING"]
    resolved = [c for c in cases if c["status"] != "PENDING"]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Pending Triage", len(pending))
    m2.metric("Confirmed Diagnoses", len([c for c in resolved if c["status"] == "CONFIRMED"]))
    m3.metric("Rejected / Reshoot", len([c for c in resolved if c["status"] == "REJECTED"]))
    m4.metric("Field Visits Dispatched", len([c for c in resolved if c["status"] == "NEEDS_VISIT"]))

    st.markdown("---")

    q1, q2 = st.columns([2.5, 1])
    with q1:
        sort_option = st.selectbox(
            "Sort diagnostic queue by:",
            ["Highest Risk Score First", "Most Recent Reports First", "By Crop (Cotton First)"]
        )

    if sort_option == "Highest Risk Score First":
        pending = sorted(pending, key=lambda c: c["risk_score"], reverse=True)
    elif sort_option == "Most Recent Reports First":
        pending = sorted(pending, key=lambda c: c["reported_at"], reverse=True)
    else:
        pending = sorted(pending, key=lambda c: (c["crop"] != "Cotton", -c["risk_score"]))

    st.markdown("### 📋 Cases Requiring Expert Review")
    if not pending:
        st.info("Queue is all clear! Simulate a new farmer report from the sidebar.")

    for case in pending:
        render_case_card(case, show_actions=True, key_prefix="expert_queue_")

    if resolved:
        st.markdown("---") 
        with st.expander(f"📁 View {len(resolved)} Resolved / Audited Cases"):
            for case in resolved:
                render_case_card(case, show_actions=False, key_prefix="expert_resolved_")
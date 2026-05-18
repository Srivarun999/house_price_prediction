import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hyderabad House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1a1a2e;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #555;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .price-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        color: white;
        margin: 1.5rem 0;
    }
    .price-label {
        font-size: 1rem;
        opacity: 0.85;
        margin-bottom: 0.4rem;
    }
    .price-value {
        font-size: 3rem;
        font-weight: 900;
        letter-spacing: -1px;
    }
    .price-range {
        font-size: 0.9rem;
        opacity: 0.8;
        margin-top: 0.5rem;
    }
    .metric-row {
        display: flex;
        justify-content: space-around;
        gap: 1rem;
        margin-top: 1rem;
    }
    .metric-box {
        background: rgba(255,255,255,0.15);
        border-radius: 10px;
        padding: 0.6rem 1.2rem;
        text-align: center;
    }
    .metric-box .val { font-size: 1.3rem; font-weight: 700; }
    .metric-box .lbl { font-size: 0.7rem; opacity: 0.8; }
    .section-header {
        font-size: 1rem;
        font-weight: 700;
        color: #333;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin: 1.5rem 0 0.5rem 0;
        padding-left: 0.2rem;
        border-left: 3px solid #667eea;
        padding-left: 0.6rem;
    }
    .info-pill {
        background: #f0f4ff;
        border-radius: 20px;
        padding: 0.25rem 0.75rem;
        font-size: 0.82rem;
        color: #555;
        display: inline-block;
        margin: 0.2rem;
    }
    div[data-testid="stButton"] button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.7rem 2rem;
        border-radius: 10px;
        font-size: 1.1rem;
        font-weight: 700;
        width: 100%;
        cursor: pointer;
        transition: opacity 0.2s;
    }
    div[data-testid="stButton"] button:hover { opacity: 0.9; }
</style>
""", unsafe_allow_html=True)

# ─── Load model ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load("house_price_model.joblib")
    with open("model_meta.json") as f:
        meta = json.load(f)
    return model, meta

model, meta = load_model()

# ─── Header ─────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🏠 Hyderabad House Price Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Powered by Machine Learning · Trained on 6,000+ Hyderabad listings</div>', unsafe_allow_html=True)

# ─── Inputs ─────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📍 Location & Property</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    locality = st.selectbox("Locality", meta["localities"])
    property_type = st.selectbox("Property Type", ["Apartment", "Independent House", "Villa"])
with col2:
    bhk = st.selectbox("BHK", [1, 2, 3, 4, 5])
    size = st.number_input("Size (sq ft)", min_value=300, max_value=10000, value=1800, step=50)

st.markdown('<div class="section-header">🏗️ Building Details</div>', unsafe_allow_html=True)

col3, col4 = st.columns(2)
with col3:
    floor_no = st.slider("Floor Number", 0, 30, 3)
    total_floors = st.slider("Total Floors in Building", 1, 40, 10)
    age = st.slider("Age of Property (years)", 0, 30, 5)
with col4:
    furnished = st.selectbox("Furnished Status", ["Furnished", "Semi-furnished", "Unfurnished"])
    facing = st.selectbox("Facing", ["North", "East", "West", "South"])
    availability = st.selectbox("Availability", ["Ready_to_Move", "Under_Construction"])

st.markdown('<div class="section-header">✨ Amenities & Facilities</div>', unsafe_allow_html=True)

col5, col6 = st.columns(2)
with col5:
    parking = st.checkbox("Parking Space", value=True)
    security = st.checkbox("24×7 Security", value=True)
    pool = st.checkbox("Swimming Pool", value=False)
with col6:
    gym = st.checkbox("Gym", value=False)
    clubhouse = st.checkbox("Clubhouse", value=False)
    transport = st.selectbox("Public Transport Access", ["High", "Medium", "Low"])

col7, col8 = st.columns(2)
with col7:
    schools = st.slider("Nearby Schools", 1, 10, 5)
with col8:
    hospitals = st.slider("Nearby Hospitals", 1, 10, 5)

# ─── Predict ────────────────────────────────────────────────────────────────
if st.button("🔍 Predict House Price"):

    # Feature engineering (must match training)
    amenity_count = 1 + int(pool) + int(gym) + int(clubhouse)
    floor_ratio = floor_no / (total_floors + 1) if total_floors > 0 else 0
    transport_score = {"Low": 1, "Medium": 2, "High": 3}[transport]
    furnished_score = {"Unfurnished": 0, "Semi-furnished": 1, "Furnished": 2}[furnished]
    property_type_enc = {"Apartment": 0, "Independent House": 1, "Villa": 2}[property_type]
    owner_type_enc = 0  # default Owner
    facing_enc = {"North": 3, "East": 2, "West": 1, "South": 0}[facing]
    availability_bin = 1 if availability == "Ready_to_Move" else 0
    loc_rank = meta["loc_price_rank"].get(locality, 7.5)

    row = {
        "BHK": bhk,
        "Size_in_SqFt": size,
        "Age_of_Property": age,
        "Floor_No": floor_no,
        "Total_Floors": total_floors,
        "Nearby_Schools": schools,
        "Nearby_Hospitals": hospitals,
        "loc_rank": loc_rank,
        "amenity_count": amenity_count,
        "has_pool": int(pool),
        "has_gym": int(gym),
        "has_clubhouse": int(clubhouse),
        "floor_ratio": floor_ratio,
        "transport_score": transport_score,
        "parking_bin": int(parking),
        "security_bin": int(security),
        "availability_bin": availability_bin,
        "furnished_score": furnished_score,
        "property_type_enc": property_type_enc,
        "owner_type_enc": owner_type_enc,
        "facing_enc": facing_enc,
    }

    # Locality one-hot
    for c in meta["loc_cols"]:
        loc_name = c.replace("loc_", "")
        row[c] = 1 if loc_name == locality else 0

    X_input = pd.DataFrame([row])[meta["features"]]
    prediction = model.predict(X_input)[0]

    # Confidence band: ±15% for synthetic data uncertainty
    low = prediction * 0.85
    high = prediction * 1.15
    price_per_sqft_est = (prediction / size) * 100000  # in ₹ per sqft

    # Display result
    st.markdown(f"""
    <div class="price-card">
        <div class="price-label">Estimated Property Value</div>
        <div class="price-value">₹ {prediction:.1f} Lakhs</div>
        <div class="price-range">Expected Range: ₹ {low:.1f}L – ₹ {high:.1f}L</div>
        <div class="metric-row">
            <div class="metric-box">
                <div class="val">₹ {price_per_sqft_est:,.0f}</div>
                <div class="lbl">per sq ft</div>
            </div>
            <div class="metric-box">
                <div class="val">₹ {prediction/100:.2f} Cr</div>
                <div class="lbl">in crores</div>
            </div>
            <div class="metric-box">
                <div class="val">{size:,}</div>
                <div class="lbl">sq ft</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Summary tags
    st.markdown("**Property Summary:**")
    tags = [
        f"📍 {locality}",
        f"🏢 {property_type}",
        f"🛏️ {bhk} BHK",
        f"📐 {size:,} sq ft",
        f"🏗️ Floor {floor_no}/{total_floors}",
        f"🪑 {furnished}",
        f"📅 {age} years old",
    ]
    if parking: tags.append("🚗 Parking")
    if security: tags.append("🔒 Security")
    if pool: tags.append("🏊 Pool")
    if gym: tags.append("💪 Gym")
    if clubhouse: tags.append("🏛️ Clubhouse")

    pills_html = "".join([f'<span class="info-pill">{t}</span>' for t in tags])
    st.markdown(pills_html, unsafe_allow_html=True)

    # Locality benchmark
    avg = meta["locality_avg_price"].get(locality, meta["price_mean"])
    delta = prediction - avg
    direction = "above" if delta > 0 else "below"
    st.info(f"📊 Average price in **{locality}** is ₹ {avg:.1f} Lakhs. Your estimate is ₹ {abs(delta):.1f}L {direction} the locality average.")

# ─── Footer ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center><small>Predictions are estimates based on ML model trained on Hyderabad property data. "
    "Consult a registered broker for official valuations.</small></center>",
    unsafe_allow_html=True
)

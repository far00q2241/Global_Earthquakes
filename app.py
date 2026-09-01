import streamlit as st
import pandas as pd
import joblib

# ----------------------------
# Load Model
# ----------------------------
model = joblib.load("global_earthquake_model.pkl")

st.set_page_config(page_title="Global Earthquake Magnitude Prediction", page_icon="🌍")

st.title("🌍 Global Earthquake Magnitude Prediction")
st.write("Enter earthquake details below to predict the earthquake magnitude class.")

# ----------------------------
# Numerical Inputs
# ----------------------------
year = st.number_input("Year", 2000, 2035, 2024)
decade = (year // 10) * 10

latitude = st.number_input("Latitude", value=20.0)
longitude = st.number_input("Longitude", value=78.0)
depth = st.number_input("Depth (km)", value=10.0)

nst = st.number_input("NST", value=20)
gap = st.number_input("Gap", value=50.0)
dmin = st.number_input("Dmin", value=0.5)
rms = st.number_input("RMS", value=0.8)

horizontalError = st.number_input("Horizontal Error", value=0.5)
depthError = st.number_input("Depth Error", value=1.0)
magError = st.number_input("Magnitude Error", value=0.1)

# Frequency Encoded Features
place_freq = st.number_input("Place Frequency", value=100)
region_freq = st.number_input("Region Frequency", value=200)

# ----------------------------
# Categorical Inputs
# ----------------------------
magType = st.selectbox(
    "Magnitude Type",
    ["ml", "ml(texnet)", "mw", "mwb", "mwc", "mwp", "mwr", "mww"]
)

net = st.selectbox(
    "Network",
    ["ci", "hv", "iscgem", "nc", "nn", "ok", "pr", "se", "tx", "us", "uu"]
)

# ----------------------------
# Create Input Dictionary
# ----------------------------
input_data = {
    "year": year,
    "decade": decade,
    "latitude": latitude,
    "longitude": longitude,
    "depth": depth,
    "nst": nst,
    "gap": gap,
    "dmin": dmin,
    "rms": rms,
    "horizontalError": horizontalError,
    "depthError": depthError,
    "magError": magError,
    "place_freq": place_freq,
    "region_freq": region_freq,

    "magType_ml": 0,
    "magType_ml(texnet)": 0,
    "magType_mw": 0,
    "magType_mwb": 0,
    "magType_mwc": 0,
    "magType_mwp": 0,
    "magType_mwr": 0,
    "magType_mww": 0,

    "net_ci": 0,
    "net_hv": 0,
    "net_iscgem": 0,
    "net_nc": 0,
    "net_nn": 0,
    "net_ok": 0,
    "net_pr": 0,
    "net_se": 0,
    "net_tx": 0,
    "net_us": 0,
    "net_uu": 0,
}

# One-Hot Encoding
input_data[f"magType_{magType}"] = 1
input_data[f"net_{net}"] = 1

# Convert to DataFrame
X_new = pd.DataFrame([input_data])

# Ensure Column Order Matches Training
feature_order = [
    'year', 'decade', 'latitude', 'longitude', 'depth', 'nst', 'gap',
    'dmin', 'rms', 'horizontalError', 'depthError', 'magError',
    'place_freq', 'region_freq',
    'magType_ml', 'magType_ml(texnet)', 'magType_mw', 'magType_mwb',
    'magType_mwc', 'magType_mwp', 'magType_mwr', 'magType_mww',
    'net_ci', 'net_hv', 'net_iscgem', 'net_nc', 'net_nn', 'net_ok',
    'net_pr', 'net_se', 'net_tx', 'net_us', 'net_uu'
]

X_new = X_new[feature_order]

# ----------------------------
# Prediction
# ----------------------------
if st.button("Predict Magnitude Class"):

    prediction = model.predict(X_new)[0]
    probabilities = model.predict_proba(X_new)[0]

    st.success(f"### Predicted Magnitude Class: {prediction}")

    st.subheader("Prediction Probabilities")

    prob_df = pd.DataFrame({
        "Magnitude Class": range(len(probabilities)),
        "Probability": probabilities
    })

    st.dataframe(prob_df, use_container_width=True)
    st.bar_chart(prob_df.set_index("Magnitude Class"))

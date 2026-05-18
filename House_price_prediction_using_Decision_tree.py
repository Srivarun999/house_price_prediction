import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.set_page_config(page_title="Hyderabad House Price Prediction", layout="wide")
st.title("Hyderabad House Price Prediction")

@st.cache_data
def load_data():
    df = pd.read_csv("hyderabad_house_price_dataset_1000.csv")
    return df

df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df)

required_columns = [
    "locality",
    "area_sqft",
    "bedrooms",
    "bathrooms",
    "parking",
    "distance_school_km",
    "distance_metro_km",
    "age_house_years",
    "price_inr"
]

missing_cols = [col for col in required_columns if col not in df.columns]
if missing_cols:
    st.error(f"Missing columns in dataset: {missing_cols}")
    st.stop()

data = df.copy()

encoder = LabelEncoder()
data["locality_encoded"] = encoder.fit_transform(data["locality"])

X = data[
    [
        "locality_encoded",
        "area_sqft",
        "bedrooms",
        "bathrooms",
        "parking",
        "distance_school_km",
        "distance_metro_km",
        "age_house_years",
    ]
]
y = data["price_inr"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("Model Performance")
c1, c2, c3, c4 = st.columns(4)
c1.metric("MSE", f"{mse:,.2f}")
c2.metric("RMSE", f"{rmse:,.2f}")
c3.metric("MAE", f"{mae:,.2f}")
c4.metric("R² Score", f"{r2:.4f}")

st.subheader("Predict House Price")

locality = st.selectbox("Locality", sorted(df["locality"].unique()))
area_sqft = st.number_input("Area (sq ft)", min_value=500, max_value=10000, value=1500)
bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
parking = st.number_input("Parking", min_value=0, max_value=5, value=1)
distance_school_km = st.number_input("Distance to School (km)", min_value=0.0, max_value=20.0, value=2.0)
distance_metro_km = st.number_input("Distance to Metro (km)", min_value=0.0, max_value=20.0, value=3.0)
age_house_years = st.number_input("Age of House (years)", min_value=0, max_value=100, value=5)

if st.button("Predict Price"):
    locality_encoded = encoder.transform([locality])[0]

    input_df = pd.DataFrame([{
        "locality_encoded": locality_encoded,
        "area_sqft": area_sqft,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "parking": parking,
        "distance_school_km": distance_school_km,
        "distance_metro_km": distance_metro_km,
        "age_house_years": age_house_years
    }])

    prediction = model.predict(input_df)[0]
    st.success(f"Estimated House Price: ₹{prediction:,.0f}")

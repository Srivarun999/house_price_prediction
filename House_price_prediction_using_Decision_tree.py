import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.set_page_config(page_title="House Price Prediction", layout="wide")
st.title("House Price Prediction App")

@st.cache_data
def load_data():
    df = pd.read_csv("hyderabad_house_price_dataset_1000.csv")
    return df

df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# Preprocessing
data = df.copy()
encoder = LabelEncoder()
data["locality"] = encoder.fit_transform(data["locality"])

X = data[["locality", "areasqft", "bedrooms", "bathrooms"]]
y = data["priceinr"]

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
col1, col2, col3, col4 = st.columns(4)
col1.metric("MSE", f"{mse:,.2f}")
col2.metric("RMSE", f"{rmse:,.2f}")
col3.metric("MAE", f"{mae:,.2f}")
col4.metric("R² Score", f"{r2:.4f}")

st.subheader("Predict House Price")

locality_name = st.selectbox("Select Locality", df["locality"].unique())
area = st.number_input("Area (sq ft)", min_value=500, max_value=10000, value=1500)
bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)

if st.button("Predict Price"):
    locality_encoded = encoder.transform([locality_name])[0]
    input_data = pd.DataFrame([{
        "locality": locality_encoded,
        "areasqft": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms
    }])
    prediction = model.predict(input_data)[0]
    st.success(f"Estimated House Price: ₹{prediction:,.0f}")

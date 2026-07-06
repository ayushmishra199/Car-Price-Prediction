import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


df = pd.read_csv("car_data.csv")



if "Unnamed: 1" in df.columns:
    df.drop("Unnamed: 1", axis=1, inplace=True)

df.dropna(inplace=True)

df = df[df["Selling_Price"] > 0]
df = df[df["Present_Price"] > 0]



car_names = sorted(df["Car_Name"].unique())
companies = sorted(df["company"].unique())
fuel_types = sorted(df["Fuel_Type"].unique())
seller_types = sorted(df["Seller_Type"].unique())
transmissions = sorted(df["Transmission"].unique())



car_encoder = LabelEncoder()
company_encoder = LabelEncoder()
fuel_encoder = LabelEncoder()
seller_encoder = LabelEncoder()
trans_encoder = LabelEncoder()

df["Car_Name"] = car_encoder.fit_transform(df["Car_Name"])
df["company"] = company_encoder.fit_transform(df["company"])
df["Fuel_Type"] = fuel_encoder.fit_transform(df["Fuel_Type"])
df["Seller_Type"] = seller_encoder.fit_transform(df["Seller_Type"])
df["Transmission"] = trans_encoder.fit_transform(df["Transmission"])



X = df.drop("Selling_Price", axis=1)

y = df["Selling_Price"]



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



model = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)



y_pred = model.predict(X_test)

score = r2_score(y_test, y_pred)



st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 Car Price Prediction")

st.success(f"R² Score : {score:.2f}")

st.subheader("Enter Car Details")

car_name = st.selectbox(
    "Car Name",
    car_names
)

company = st.selectbox(
    "Company",
    companies
)

year = st.number_input(
    "Year",
    min_value=1990,
    max_value=2025,
    value=2018
)

present_price = st.number_input(
    "Present Price (Lakhs)",
    min_value=0.0,
    value=5.0
)

kms_driven = st.number_input(
    "Kilometers Driven",
    min_value=0,
    value=50000
)

fuel_type = st.selectbox(
    "Fuel Type",
    fuel_types
)

seller_type = st.selectbox(
    "Seller Type",
    seller_types
)

transmission = st.selectbox(
    "Transmission",
    transmissions
)

owner = st.selectbox(
    "Previous Owners",
    [0, 1, 2, 3, 4]
)


if st.button("Predict Price"):

    car_encoded = car_encoder.transform(
        [car_name]
    )[0]

    company_encoded = company_encoder.transform(
        [company]
    )[0]

    fuel_encoded = fuel_encoder.transform(
        [fuel_type]
    )[0]

    seller_encoded = seller_encoder.transform(
        [seller_type]
    )[0]

    trans_encoded = trans_encoder.transform(
        [transmission]
    )[0]

    prediction = model.predict([[
        car_encoded,
        company_encoded,
        year,
        present_price,
        kms_driven,
        fuel_encoded,
        seller_encoded,
        trans_encoded,
        owner
    ]])

    st.success(
        f"Estimated Selling Price: ₹ {prediction[0]:.2f} Lakhs"
    )
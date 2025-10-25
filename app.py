import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Urban Air Quality App", layout="wide")
st.title(" 🌍 Global Air Quality Data Explorer & Predictor")


# 1️⃣ Load Data

st.header("📥 Load Data")

uploaded_file = st.file_uploader("Upload your air quality CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    
    df = pd.read_csv("http://localhost:8888/edit/air_quality_global.csv", delimiter=';')

    st.info("Loaded pre-cleaned dataset from /data folder.")

st.write("### Data Preview")
st.dataframe(df.head())


# 2️⃣ EDA

st.header("🔍 Global air quality Data Analysis")

st.write("### Summary Statistics")
st.write(df.describe())

numeric_cols = df.select_dtypes(include='number').columns.tolist()
if len(numeric_cols) > 1:
    st.write("### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm")
    st.pyplot(fig)


# 3️⃣ Visualization

st.header("📊 Visualizations")

if {'latitude', 'longitude', 'pm25_ugm3'}.issubset(df.columns):
    fig = px.scatter_geo(df, lat='latitude', lon='longitude',
                         color='pm25_ugm3', hover_name='city',
                         title="Global PM2.5 Concentration",
                         color_continuous_scale='Reds')
    st.plotly_chart(fig, use_container_width=True)

if {'no2_ugm3', 'pm25_ugm3'}.issubset(df.columns):
    fig = px.scatter(df, x='no2_ugm3', y='pm25_ugm3',
                     color='country', title="PM2.5 vs NO₂")
    st.plotly_chart(fig, use_container_width=True)


# 4️⃣ Predictive Modeling

st.header("🤖 Simple Predictive Modeling")

if len(numeric_cols) > 1:
    target_col = st.selectbox("Select target variable", numeric_cols, index=0)
    features = [col for col in numeric_cols if col != target_col]

    X = df[features]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.write("### Model Evaluation")
    st.write(f"**MAE:** {mae:.2f}")
    st.write(f"**MSE:** {mse:.2f}")
    st.write(f"**R²:** {r2:.2f}")

    fig = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual', 'y': 'Predicted'},
                     title="Actual vs Predicted")
    st.plotly_chart(fig, use_container_width=True)

st.success("✅ Dashboard Complete!")

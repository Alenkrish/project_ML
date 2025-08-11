import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import os
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(page_title="🏆 FIFA World Cup Predictor", layout="wide")


MODEL_PATH = "model.pkl"
DATA_PATH = "data/Data.csv"


if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    st.error("Model file not found.")
    st.stop()


if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
else:
    st.error("Dataset file not found.")
    st.stop()


menu = st.sidebar.radio(
    "Navigation",
    ["Home", "Data Exploration", "Visualizations", "Prediction", "Model Performance", "About"]
)


if menu == "Home":
    st.title("🏆 FIFA World Cup Match Outcome Predictor")
    st.write("""
    This app predicts **Win**, **Lose**, or **Draw** for the Home Team based on FIFA World Cup match details.

    ### Features in the dataset:
    - Year, Stage, Stadium, City
    - Home & Away Team names, initials, goals
    - Attendance, half-time scores
    - MatchID & RoundID

    **Target:** Match outcome (Home Win / Away Win / Draw)
    """)


elif menu == "Data Exploration":
    st.subheader("Dataset Overview")
    st.write("Shape of dataset:", df.shape)
    st.write("Column names:", list(df.columns))
    st.dataframe(df.head())

    st.subheader("Data Types")
    st.write(df.dtypes)

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    st.subheader("Interactive Filtering")
    filter_col = st.selectbox("Select column to filter", options=df.columns)
    unique_vals = df[filter_col].unique().tolist()
    selected_vals = st.multiselect("Select values", options=unique_vals, default=unique_vals)
    filtered_df = df[df[filter_col].isin(selected_vals)]
    st.write("Filtered rows:", filtered_df.shape[0])
    st.dataframe(filtered_df)


elif menu == "Visualizations":
    st.subheader("Data Visualizations")

    if "Outcome" in df.columns:
        
        st.markdown("### 1. Match Outcome Distribution")
        fig1 = px.histogram(df, x="Outcome", color="Outcome", title="Match Outcomes")
        st.plotly_chart(fig1, use_container_width=True)

   
    if "Stage" in df.columns and "Attendance" in df.columns:
        st.markdown("### 2. Attendance by Stage")
        fig2 = px.box(df, x="Stage", y="Attendance", color="Stage", points="all")
        st.plotly_chart(fig2, use_container_width=True)

    
    st.markdown("### 3. Correlation Heatmap")
    numeric_df = df.select_dtypes(include=np.number)
    if not numeric_df.empty:
        fig3 = px.imshow(numeric_df.corr(), text_auto=True, aspect="auto", color_continuous_scale="RdBu_r")
        st.plotly_chart(fig3, use_container_width=True)


elif menu == "Prediction":
    st.subheader("Predict FIFA Match Outcome")

    year = st.number_input("Year", min_value=1930, max_value=2050, value=2018)
    stage = st.text_input("Stage", value="Final")
    stadium = st.text_input("Stadium", value="Luzhniki Stadium")
    city = st.text_input("City", value="Moscow")
    home_team_name = st.text_input("Home Team Name", value="France")
    home_team_goals = st.number_input("Home Team Goals (so far)", min_value=0, max_value=20, value=0)
    away_team_goals = st.number_input("Away Team Goals (so far)", min_value=0, max_value=20, value=0)
    away_team_name = st.text_input("Away Team Name", value="Croatia")
    attendance = st.number_input("Attendance", min_value=0, max_value=200000, value=78011)
    half_time_home_goals = st.number_input("Half-time Home Goals", min_value=0, max_value=20, value=0)
    half_time_away_goals = st.number_input("Half-time Away Goals", min_value=0, max_value=20, value=0)
    round_id = st.number_input("RoundID", min_value=0, value=1046)
    match_id = st.number_input("MatchID", min_value=0, value=300331522)
    home_team_initials = st.text_input("Home Team Initials", value="FRA")
    away_team_initials = st.text_input("Away Team Initials", value="CRO")

    input_data = pd.DataFrame({
        'Year': [year],
        'Stage': [stage],
        'Stadium': [stadium],
        'City': [city],
        'Home Team Name': [home_team_name],
        'Home Team Goals': [home_team_goals],
        'Away Team Goals': [away_team_goals],
        'Away Team Name': [away_team_name],
        'Attendance': [attendance],
        'Half-time Home Goals': [half_time_home_goals],
        'Half-time Away Goals': [half_time_away_goals],
        'RoundID': [round_id],
        'MatchID': [match_id],
        'Home Team Initials': [home_team_initials],
        'Away Team Initials': [away_team_initials]
    })

    if st.button("Predict Outcome"):
        prediction = model.predict(input_data)[0]
        st.success(f"Predicted Match Outcome: **{prediction}**")


elif menu == "Model Performance":
    st.subheader("Model Evaluation")
    if "Outcome" not in df.columns:
        st.warning("No 'Outcome' column in dataset for evaluation.")
    else:
        X = df.drop(columns=["Outcome"])
        y = df["Outcome"]

        y_pred = model.predict(X)
        acc = accuracy_score(y, y_pred)
        st.write(f"Accuracy: {acc:.4f}")

        cm = confusion_matrix(y, y_pred)
        st.write("Confusion Matrix:")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)

        st.write("Classification Report:")
        st.dataframe(pd.DataFrame(classification_report(y, y_pred, output_dict=True)).transpose())


elif menu == "About":
    st.subheader("About This Project")
    st.write("""
    - **Objective:** Predict FIFA World Cup match outcomes.
    - **Dataset:** FIFA match history dataset.
    - **Model:** Trained ML model saved with joblib.
    - **Tech:** Python, Pandas, NumPy, Scikit-learn, Streamlit, Plotly, Seaborn.
    - **Developer:** N. Kishnapriyan
    """)

import streamlit as st
import pandas as pd
import joblib

model = joblib.load('model.pkl')

st.title("🏆 World Cup Match Outcome Predictor")
st.write("Predict whether the Home Team will Win, Lose, or Draw based on match details.")

st.sidebar.header("Match Details")

year = st.sidebar.number_input("Year", min_value=1930, max_value=2050, value=2018)
stage = st.sidebar.text_input("Stage", value="Final")
stadium = st.sidebar.text_input("Stadium", value="Luzhniki Stadium")
city = st.sidebar.text_input("City", value="Moscow")
home_team_name = st.sidebar.text_input("Home Team Name", value="France")
home_team_goals = st.sidebar.number_input("Home Team Goals (so far)", min_value=0, max_value=20, value=0)
away_team_goals = st.sidebar.number_input("Away Team Goals (so far)", min_value=0, max_value=20, value=0)
away_team_name = st.sidebar.text_input("Away Team Name", value="Croatia")
attendance = st.sidebar.number_input("Attendance", min_value=0, max_value=200000, value=78011)
half_time_home_goals = st.sidebar.number_input("Half-time Home Goals", min_value=0, max_value=20, value=0)
half_time_away_goals = st.sidebar.number_input("Half-time Away Goals", min_value=0, max_value=20, value=0)
round_id = st.sidebar.number_input("RoundID", min_value=0, value=1046)
match_id = st.sidebar.number_input("MatchID", min_value=0, value=300331522)
home_team_initials = st.sidebar.text_input("Home Team Initials", value="FRA")
away_team_initials = st.sidebar.text_input("Away Team Initials", value="CRO")


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


if st.sidebar.button("Predict Outcome"):
    prediction = model.predict(input_data)[0]
    st.subheader(f"Predicted Match Outcome: **{prediction}**")

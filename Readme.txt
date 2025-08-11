# Project ML — World Cup Match Outcome Predictor

This project is a Machine Learning application that predicts the outcome of FIFA World Cup matches  
(Home Win / Draw / Away Win) based on historical match data.

It includes:
- **modeltraining_cleaned.ipynb** — Jupyter Notebook for training the model
- **model.pkl** — Serialized trained model
- **app.py** — Streamlit web app for predictions
- **requirements.txt** — Exact Python dependencies for the project

## Dataset
The dataset used (`Data.csv`) contains historical FIFA World Cup match data with columns such as:
- Year
- Stage
- Stadium
- City
- Home Team Name, Home Team Goals
- Away Team Name, Away Team Goals
- Attendance, Half-time goals
- Other match metadata

## How it Works
1. **Preprocessing** — Categorical and numerical features are preprocessed using Scikit-learn pipelines.
2. **Model Training** — A Random Forest Classifier is trained to predict the match result.
3. **Saving Model** — The trained pipeline is saved as `model.pkl`.
4. **Prediction App** — A Streamlit app allows users to input match details and get predicted outcomes.

https://github.com/Alenkrish/project_ML

# Core Pkgs
from pandas.core.arrays import categorical
import streamlit as st

# EDA Pkgs
import pandas as pd
import numpy as np

# Utils
import os
import joblib

# Data Viz Pkgs
import matplotlib.pyplot as plt
import matplotlib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


def display():
    st.title('Stroke Prediction Application')


def load_PreTrainedModelDetails():
    PreTrainedModelDetails = joblib.load(
        'model/stroke_pred_classification.joblib')
    return PreTrainedModelDetails


def prediction(input_df):

    PreTrainedModelDetails = load_PreTrainedModelDetails()

    # Random Forest Classifier
    RandomForestClassifier = PreTrainedModelDetails.get('model')

    # PreFitted Encoder
    PreFittedEncoder = PreTrainedModelDetails.get('encoder')

    # PreFitted Scaler
    PreFittedScaler = PreTrainedModelDetails.get('scaler')

    numeric_cols = PreTrainedModelDetails.get('numeric_cols')

    categorical_cols = PreTrainedModelDetails.get('categorical_cols')

    encoded_cols = PreTrainedModelDetails.get('encoded_cols')

    input_df[encoded_cols] = PreFittedEncoder.transform(
        input_df[categorical_cols])
    input_df[numeric_cols] = PreFittedScaler.transform(input_df[numeric_cols])

    inputs_for_prediction = input_df[numeric_cols+encoded_cols]

    prediction = RandomForestClassifier.predict(inputs_for_prediction)

    if prediction == 0:
        st.success("Patient is not at risk of getting a stroke.")
    else:
        st.warning("Patient is at risk of getting a stroke.")

    # st.write("Accuracy of the prediction : {}".format(accuracy))


def get_user_input():
    form = st.form(key='user input form')
    gender = form.radio("gender", ['Male', 'Female', 'Other'], key='gender')
    age = form.number_input("age", 1, 120, key='age')
    hypertension = form.radio(
        "hypertension", ['Yes', 'No'], key='hypertension')
    heart_disease = form.radio(
        "heart_disease", ['Yes', 'No'], key='heart_disease')
    ever_married = form.radio(
        "ever_married", ['Married', 'Unmarried'], key='ever_married')
    work_type = form.radio("work_type", ['Private Sector', 'Government Sector', 'Never Worked',
                                         'Self-employed', 'Children'], key='work_type')
    residence_type = form.radio(
        "residence_type", ['Urban', 'Rural'], key='residence_type')
    avg_glucose_level = form.number_input(
        "avg_glucose_level", 40.0, 400.0, key='avg_glucose_level')
    bmi = form.number_input("bmi", 10.00, 120.00, key='bmi')
    smoking_status = form.radio("smoking_status", [
        'Never Smoked', 'Formerly Smoked', 'Smokes', 'Unknown'], key='smoking_status')

    submitButton = form.form_submit_button(label='Predict Stroke Condition')

    if submitButton:
        SingleUserInput = {
            'gender': gender,
            'age': age,
            'hypertension': hypertension,
            'heart_disease': heart_disease,
            'ever_married': ever_married,
            'work_type': work_type,
            'residence_type': residence_type,
            'avg_glucose_level': avg_glucose_level,
            'bmi': bmi,
            'smoking_status': smoking_status

        }

        input_df = pd.DataFrame([SingleUserInput])

        return input_df


def main():
    display()
    input_details_df = get_user_input()

    if input_details_df is not None:
        prediction(input_details_df)


if __name__ == '__main__':
    main()

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
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Background
import base64

# Creating the separation
col1, col2 = st.columns([2, 1])


def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
        f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"avif"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
        unsafe_allow_html=True
    )


add_bg_from_local('images/hos2.avif')


def display():
    col1.title('Stroke Prediction Application')


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

    numerical_cols = PreTrainedModelDetails.get('numerical_cols')

    categorical_cols = PreTrainedModelDetails.get('categorical_cols')

    label_gender = LabelEncoder()
    label_married = LabelEncoder()
    label_work = LabelEncoder()
    label_residence = LabelEncoder()
    label_smoking = LabelEncoder()

    categorical_cols[0] = label_gender.fit_transform(
        categorical_cols['gender'])
    categorical_cols[1] = label_work.fit_transform(
        categorical_cols['work_type'])
    categorical_cols[2] = label_residence.fit_transform(
        categorical_cols['Residence_type'])
    categorical_cols[3] = label_smoking.fit_transform(
        categorical_cols['smoking_status'])
    categorical_cols[4] = label_married.fit_transform(
        categorical_cols['ever_married'])

    # encoded_cols = PreTrainedModelDetails.get('encoded_cols')

    input_df[categorical_cols] = PreFittedEncoder.transform(
        input_df[categorical_cols])
    input_df[numerical_cols] = PreFittedScaler.transform(
        input_df[numerical_cols])

    inputs_for_prediction = input_df[numerical_cols+categorical_cols]

    prediction = RandomForestClassifier.predict(inputs_for_prediction)

    if prediction == 0:
        st.success("Patient is not at risk of getting a stroke.")
    else:
        st.warning("Patient is at risk of getting a stroke.")

    # st.write("Accuracy of the prediction : {}".format(accuracy))


def get_user_input():
    form = col1.form(key='user input form')
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
    Residence_type = form.radio(
        "Residence_type", ['Urban', 'Rural'], key='Residence_type')
    avg_glucose_level = form.number_input(
        "avg_glucose_level", 40.0, 400.0, key='avg_glucose_level')
    bmi = form.number_input("bmi", 10.00, 120.00, key='bmi')
    smoking_status = form.radio("smoking_status", [
        'Never Smoked', 'Formerly Smoked', 'Smokes', 'Unknown'], key='smoking_status')

    submitButton = form.form_submit_button(label='Predict Stroke Condition')

    # Getting inputs from the form.
    if submitButton:
        SingleUserInput = {
            'gender': gender,
            'age': age,
            'hypertension': hypertension,
            'heart_disease': heart_disease,
            'ever_married': ever_married,
            'work_type': work_type,
            'Residence_type': Residence_type,
            'avg_glucose_level': avg_glucose_level,
            'bmi': bmi,
            'smoking_status': smoking_status

        }

        user_input = pd.DataFrame([SingleUserInput])

        return user_input


def main():
    display()
    input_details_df = get_user_input()

    if input_details_df is not None:
        prediction(input_details_df)


# Sidebar Configurations
st.sidebar.header("Introduction")

with st.sidebar:
    st.write("A stroke, sometimes called a brain attack, occurs when something blocks blood supply to part of the brain or when a blood vessel in the brain bursts. In either case, parts of the brain become damaged or die. A stroke can cause lasting brain damage, long-term disability, or even death.")

st.sidebar.image("images/stroke9.jfif", use_column_width=True)


if __name__ == '__main__':
    main()

# Image grid
col2.image("images/stroke1.jpg")
col2.image("images/stroke3.jpg")
col2.image("images/stroke4.jpg")
col2.image("images/stroke2.jpg")
col2.image("images/stroke5.jfif")
col2.image("images/stroke6.jfif")
col2.image("images/stroke7.jfif")
col2.image("images/stroke8.jfif")

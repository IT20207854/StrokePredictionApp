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
from sklearn.preprocessing import OneHotEncoder

# Background
import base64

# Creating the separation
col1, col2 = st.columns([2, 1])

# Adding background image


def display():
    col1.title('Stroke Prediction Application')


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


def load_PreTrainedModelDetails():
    PreTrainedModelDetails = joblib.load(
        "model/stroke_pred_classification.joblib")
    return PreTrainedModelDetails


def prediction(input_df):
    PreTrainedModelDetails = load_PreTrainedModelDetails()

    # Random Forest Classifier
    DecisionTreeClassifier = PreTrainedModelDetails.get('model')

    prediction = DecisionTreeClassifier.predict(input_df)

    st.subheader('Prediction')

    if prediction == 0:
        st.success("Patient is not at risk of getting a stroke.")
    else:
        st.warning("Patient is at risk of getting a stroke.")


def get_user_input():
    form = col1.form(key='user input form')

    gender = form.radio("Gender", ['Male', 'Female', 'Other'], key='gender')

    age = form.number_input("Age", 1, 120, key='age')

    hypertension = form.radio(
        "Hypertension", ['Yes', 'No'], key='hypertension')

    heart_disease = form.radio(
        "Heart Disease", ['Yes', 'No'], key='heart_disease')

    ever_married = form.radio(
        "Ever Married", ['Married', 'Unmarried'], key='ever_married')

    work_type = form.radio("Work Type", [
        'Private Sector', 'Government Sector', 'Never Worked', 'Self-employed', 'Children'], key='work_type')

    Residence_type = form.radio(
        "Residence Type", ['Urban', 'Rural'], key='Residence_type')

    avg_glucose_level = form.number_input(
        "Avg. Glucose Level", 40.0, 400.0, key='avg_glucose_level')

    bmi = form.number_input("BMI", 10.00, 120.00, key='bmi')

    smoking_status = form.radio("Smoking Status", [
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

        user_input = pd.DataFrame(SingleUserInput, index=[0])

        return user_input


def data_preprocessor(df):
    df.gender = df.gender.map({'Male': 0, 'Female': 1})
    df.hypertension = df.hypertension.map({'Yes': 1, 'No': 0})
    df.heart_disease = df.heart_disease.map({'Yes': 1, 'No': 0})
    df.ever_married = df.ever_married.map({'Married': 1, 'Unmarried': 0})
    df.work_type = df.work_type.map(
        {'Private Sector': 2, 'Government Sector': 0, 'Never Worked': 1, 'Self-employed': 3, 'Children': 4})
    df.Residence_type = df.Residence_type.map({'Urban': 1, 'Rural': 0})
    df.smoking_status = df.smoking_status.map(
        {'Never Smoked': 2, 'Formerly Smoked': 1, 'Smokes': 3, 'Unknown': 0})
    return df


def main():
    display()
    # Sidebar Configurations
    st.sidebar.header("Introduction")

    with st.sidebar:
        st.write("A stroke, sometimes called a brain attack, occurs when something blocks blood supply to part of the brain or when a blood vessel in the brain bursts. In either case, parts of the brain become damaged or die. A stroke can cause lasting brain damage, long-term disability, or even death.")

    st.sidebar.image("images/stroke9.jfif", use_column_width=True)

    # Image grid
    col2.image("images/stroke1.jpg")
    col2.image("images/stroke3.jpg")
    col2.image("images/stroke4.jpg")
    col2.image("images/stroke2.jpg")
    col2.image("images/stroke5.jfif")
    col2.image("images/stroke6.jfif")
    col2.image("images/stroke7.jfif")
    col2.image("images/stroke8.jfif")

    user_input_df = get_user_input()
    processed_user_input = data_preprocessor(user_input_df)

    if user_input_df is not None:
        prediction(processed_user_input)


if __name__ == '__main__':
    main()

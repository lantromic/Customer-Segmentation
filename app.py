import streamlit as st
import numpy as np
import pandas as pd
import joblib

model = joblib.load('customer_segmentation_rf.pkl')
pipeline = joblib.load('customer_segmentation_pipeline.pkl')

Gender = st.selectbox(
        "Gender",
        ["Male","Female"]
    )

Ever_Married = st.selectbox(
        "Ever Married?",
        ["Yes","No"]
    )

age = st.slider("Age", 1, 150, 50)

Graduated = st.selectbox(
        "Graduated?",
        ["Yes","No"]
    )

Profession = st.selectbox(
        "Profession?",
        ['Healthcare','Engineer','Lawyer','Entertainment','Artist','Executive','Doctor','Homemaker','Marketing']
    )

Work_Experience = st.slider("Work Experience(in years)", 0, 14, 0)

Spending = st.selectbox(
        "Spending",
        ['Low','Average','High']
    )

family_size = st.slider("Family Size", 1, 9, 2)

var_1 = "Cat_6"

if st.button("Predict"):
    new_data = pd.DataFrame({'Gender':[Gender],'Ever_Married':[Ever_Married],'Age':[age],'Graduated':[Graduated],'Profession':[Profession],'Work_Experience':[Work_Experience],'Spending_Score':[Spending],'Family_Size':[family_size],'Var_1':[var_1]})
    new_data = new_data.rename(columns={'Var_1':'Category'})

    pipelined_data = pipeline.transform(new_data)
    predictions = model.predict(pipelined_data)

    st.header("this customer belongs to category " + str(predictions[0]))
    
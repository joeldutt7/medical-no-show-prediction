import streamlit as st
import joblib
import numpy as np

st.title("Medical no-Show Prediction")

st.write("This app predicts whether a patient will miss their appointment.")
st.markdown("---")

#Load model
model=joblib.load("final_no_show_model.pkl")

columns=joblib.load("columns.pkl")

if len(columns)==31:
    columns=list(columns)
columns.append("appointment_shift_afternoon")

st.subheader("Enter Patient Details")

age=st.number_input("Age",0,100,25)
sms_received=st.selectbox("SMS Received",[0,1])
hypertension=st.selectbox("Hypertension",[0,1])
diabetes=st.selectbox("Diabetes",[0,1])

st.write(columns)

if st.button("Predict"):
    input_data={col:0 for col in columns}

    input_data['appointment_time']=0
    input_data['age']=age
    input_data['sms_received']=sms_received
    input_data['hipertension']=hypertension
    input_data['diabetes']=diabetes
    input_data['appointment_shift_afternoon']=0

    missing=[col for col in columns if col not in input_data]
    extra=[col for col in input_data if col not in columns]

    st.write("length:", len(input_data))
    st.write("Missing:",missing)
    st.write("Extra:",extra)

    features=np.array([[input_data[col] for col in columns]])

    prediction=model.predict(features)

    if prediction[0] ==1:
        st.error("High Risk: Patient may miss the appointment")
    else:
        st.success("Patient likely to attend")

st.markdown("### what this means")
st.write("The model uses patient history and conditions to predict no-shows.")

import os
st.write(os.listdir())
st.write(os.listdir(".."))

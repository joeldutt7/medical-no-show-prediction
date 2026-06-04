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
alcoholism=st.selectbox("Alcoholism",[0,1])
scholarship=st.selectbox("Scholarship",[0,1])
gender=st.selectbox("Gender",["Female","Male"])
disability=st.selectbox(
    "Disability",
    ["None","Unknown","Intellectual","Motor"]
)
specialty=st.selectbox(
    "Specialty",
    [
        "Assist",
        "ENF",
        "Occupational Therapy",
        "Pedagogo",
        "Physiotherapy",
        "Psychotherapy",
        "Sem Especialidade",
        "Speech Therapy"
    ]
)
appointment_shift=st.selectbox(
    "Appointment Shift",
    ["Morning","Afternoon"]
)

if st.button("Predict"):
    input_data={col:0 for col in columns}

    input_data["age"]=age
    input_data["sms_received"]=sms_received
    input_data["hipertension"]=hypertension
    input_data["diabetes"]=diabetes
    input_data["alcoholism"]=alcoholism
    input_data["scholarship"]=scholarship
    
    if gender=="Female":
        input_data["gender_F"]=1
    else:
        input_data["gender_M"]=1

    if disability=="Unknown":
        input_data["disability_Unknown"]=1
    elif disability=="Intellectual":
        input_data["disability_intellectual"]=1
    elif disability=="Motor":
        input_data["disability_motor"]=1

    if specialty=="Assist":
        input_data["specialty_assist"]=1
    elif specialty=="ENF":
        input_data["specialty_enf"]=1
    elif specialty=="Occupational Therapy":
        input_data["specialty_occupational therapy"]=1
    elif specialty=="Pedagogo":
        input_data["specialty_pedagogo"]=1
    elif specialty=="Physiotherapy":
        input_data["specialty_physiotherapy"]=1
    elif specialty=="Psychotherapy":
        input_data["specialty_psychotherapy"]=1
    elif specialty=="Sem Especialidade":
        input_data["specialty_sem especialidade"]=1
    elif specialty=="Speech Therapy":
        input_data["specialty_speech therapy"]=1

    if appointment_shift=="Morning":
        input_data["appointment_shift_morning"]=1
    else:
        input_data["appointment_shift_afternoon"]=1

    missing=[col for col in columns if col not in input_data]
    extra=[col for col in input_data if col not in columns]

    features=np.array([[input_data[col] for col in columns]])

    prediction=model.predict(features)

    if prediction[0] ==1:
        st.error("Prediction: Patient is likely to miss the appointment.")
    else:
        st.success("Prediction: patient is likely to attend the appointment.")

st.markdown("### what this means")
st.write("The model uses patient history and conditions to predict no-shows.")

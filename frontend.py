import streamlit as st
from joblib import load

import numpy as np
import pandas as pd

def main():
    st.title("Stroke Prediction App")
    st.markdown("""
    <h2 style='text-align: center;'>Please enter prediction info</h2>
    """, unsafe_allow_html=True)

    genders = ["Female", "Male", "Other"]
    yn = ["Yes", "No"]
    work_types = ["Govt_job","Never_worked","Private","Self-employed","children"]
    Residence_types = ["Rural","Urban"]
    smoking_statuses = ["never smoked", "formerly smoked", "smokes", "Unknown"]
    
    gender = st.selectbox("Gender:", genders) 
    age = st.text_input("Age: ", "0")
    hypertension = st.selectbox("Hypertension",yn)
    heart_disease = st.selectbox("Heart Disease",yn)
    ever_married = st.selectbox("Ever Married",yn)
    work_type = st.selectbox("Work Type",work_types)
    Residence_type = st.selectbox("Residence Type",Residence_types)
    avg_glucose_level = st.text_input("Average Glucose Level: ", "0")
    bmi = st.text_input("BMI: ", "0")
    smoking_status = st.selectbox("Smoking Status",smoking_statuses)

    # st.markdown(f"""<h4 style='text-align: center;'>
    #     Here is the info recieved:<br/>
    #     Gender: {gender}<br/>
    #     Age: {age}<br/>
    #     hypertension: {hypertension}<br/>
    #     heart_disease: {heart_disease}<br/>
    #     ever_married: {ever_married}<br/>
    #     work_type: {work_type}<br/>
    #     Residence_type: {Residence_type}<br/>
    #     avg_glucose_level: {avg_glucose_level}<br/>
    #     bmi: {bmi}<br/>
    #     smoking_status: {smoking_status}<br/>
    # </h4>""", unsafe_allow_html=True)

    gender = genders.index(gender)
    age = int(age)
    hypertension = yn.index(hypertension)
    heart_disease = yn.index(heart_disease)
    ever_married = yn.index(ever_married)

    work_type = work_types.index(work_type)
    w0 = 0 if work_type != 0 else 1
    w1 = 0 if work_type != 1 else 1
    w2 = 0 if work_type != 2 else 1
    w3 = 0 if work_type != 3 else 1
    w4 = 0 if work_type != 4 else 1

    Residence_type = Residence_types.index(Residence_type)
    r0 = 0 if Residence_type != 0 else 1
    r1 = 0 if Residence_type != 1 else 1

    avg_glucose_level = float(avg_glucose_level)
    bmi = float(bmi)
    smoking_status = smoking_statuses.index(smoking_status)

    if st.button("Predict"):
        rf_model = load('models/rf_model.joblib')
        cls_model = load('models/cls_model.joblib')
        svc_model = load('models/svc_model.joblib')
        mlp_model = load('models/mlp_model.joblib')
        gnb_model = load('models/gnb_model.joblib')
        knn_model = load('models/knn_model.joblib')
        stacking_classifier_model = load('models/stacking_classifier_model.joblib')

        scaler = load('scaler/scaler.joblib')

        model_list = [rf_model, cls_model, svc_model, mlp_model, gnb_model, knn_model, stacking_classifier_model]
        model_names = ['Random Forest', 'Logistic Regression', 'SVC', 'MLP', 'Gaussian Naive Bayes', 'KNN', 'Stacking Classifier']

        data = np.array([gender, age, hypertension, heart_disease, ever_married, avg_glucose_level, bmi, smoking_status, w0, w1, w2, w3, w4, r0, r1])
        df = pd.DataFrame(data.reshape(1, -1), columns=['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'avg_glucose_level', 'bmi', 'smoking_status', 'work_type_0.0', 'work_type_1.0', 'work_type_2.0', 'work_type_3.0', 'work_type_4.0', 'Residence_type_0.0', 'Residence_type_1.0'])

        df[['age', 'avg_glucose_level', 'bmi']] = scaler.transform(df[['age', 'avg_glucose_level', 'bmi']])
        for i in range(len(model_list)):
            prediction = model_list[i].predict(df)
            pred = "Stroke" if prediction[0] == 1 else "No Stroke"
            col = "red" if prediction[0] == 1 else "green"
            st.markdown(f"""<h2 style='text-align: center;'>{model_names[i]} Prediction: <span style='color: {col}'>{pred}</span></h2>""", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
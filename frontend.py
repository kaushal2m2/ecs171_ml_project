import streamlit as st
from joblib import load

from sklearn import metrics

import numpy as np
import pandas as pd

def pre_process_df(df):
    # df = df.drop(columns=['id'])
    # df['bmi'] = df['bmi'].fillna(df['bmi'].mean())

    cats = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    encoder = load('encoder/encoder.joblib')
    df[cats] = encoder.fit_transform(df[cats])
    df = pd.get_dummies(df, columns = ["work_type", "Residence_type"], dtype=int)

    num = ["age", "avg_glucose_level", "bmi"]
    scaler = load('scaler/scaler.joblib')
    df[num] = scaler.fit_transform(df[num])

    X = df.drop('stroke', axis=1)
    y = df['stroke']

    return X, y


def main():
    st.title("Stroke Prediction App")
    st.markdown("""
    <h2 style='text-align: center;'>Please enter prediction info</h2>
    """, unsafe_allow_html=True)

    # user input
    genders = ["Female", "Male", "Other"]
    yn = ["Yes", "No"]
    work_types = ["Govt_job","Never_worked","Private","Self-employed","children"]
    Residence_types = ["Rural","Urban"]
    smoking_statuses = ["Unknown", "formerly smoked", "never smoked", "smokes"]
    
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
    fl = False

    # file input
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df)
        X, y = pre_process_df(df)
        fl = True

    if st.button("Predict"):
        rf_model = load('models/rf_model.joblib')
        cls_model = load('models/cls_model.joblib')
        svc_model = load('models/svc_model.joblib')
        mlp_model = load('models/mlp_model.joblib')
        gnb_model = load('models/gnb_model.joblib')
        knn_model = load('models/knn_model.joblib')
        stacking_classifier_model = load('models/stacking_classifier_model.joblib')

        model_list = [cls_model, rf_model, svc_model, mlp_model, gnb_model, knn_model, stacking_classifier_model]
        model_names = ['Logistic Regression (preferred)', 'Random Forest', 'SVC', 'MLP', 'Gaussian Naive Bayes', 'KNN', 'Stacking Classifier']

        if fl:
            for i in range(len(model_list)):
                prediction = model_list[i].predict(X)
                confusion_matrix = metrics.confusion_matrix(y, prediction)
                acc = np.mean(prediction == y)
                acc = round(acc, 3)
                recall = round(confusion_matrix[1][1] / (confusion_matrix[1][0] + confusion_matrix[1][1]), 3)
                st.markdown(f"""<h2 style='text-align: center;'>{model_names[i]}</h2>
                                <h3 style='text-align: center; color: gray'>Accuracy: {acc}</h3>
                                <h3 style='text-align: center; color: gray'>Recall: {recall}</h3>""", unsafe_allow_html=True)
        else:
            scaler = load('scaler/scaler.joblib')

            data = np.array([gender, age, hypertension, heart_disease, ever_married, avg_glucose_level, bmi, smoking_status, w0, w1, w2, w3, w4, r0, r1])
            df = pd.DataFrame(data.reshape(1, -1), columns=['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'avg_glucose_level', 'bmi', 'smoking_status', 'work_type_0.0', 'work_type_1.0', 'work_type_2.0', 'work_type_3.0', 'work_type_4.0', 'Residence_type_0.0', 'Residence_type_1.0'])

            df[['age', 'avg_glucose_level', 'bmi']] = scaler.transform(df[['age', 'avg_glucose_level', 'bmi']])
            for i in range(len(model_list)):
                prediction = model_list[i].predict(df)
                pred = "Stroke" if prediction[0] == 1 else "No Stroke"
                col = "red" if prediction[0] == 1 else "green"
                st.markdown(f"""<h2 style='text-align: center;'>{model_names[i]}</h2>
                                <h3 style='text-align: center; color: gray'>Prediction: <span style='color: {col}'>{pred}</span></h3>
                                <br/>
                                <br/>
                                """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
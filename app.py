# -*- coding: utf-8 -*-
"""
Created on Tuesday Oct 19 02:20:31 2021
@author: Dhananjay Kumar
"""
import numpy as np
import pickle
import pandas as pd
import streamlit as st

pickle_in = open("C:/Users/dhananjay.kumar01/test/test/apps/dt_model.pkl", "rb")
classifier=pickle.load(pickle_in)

def main():
    """Simple ML App"""

    st.sidebar.title('Model Selection Panel')
    condition = st.sidebar.selectbox("Select the Model for prediction", ("Decision Tree", "Logistic Regression", "Randon Forest"))
    if condition == 'Decision Tree':
        pickle_in = open("C:/Users/dhananjay.kumar01/test/test/apps/dt_model.pkl", "rb")
        classifier = pickle.load(pickle_in)
    elif condition == 'Logistic Regression':
        pickle_in = open("C:/Users/dhananjay.kumar01/test/test/apps/lr_model.pkl", "rb")
        classifier = pickle.load(pickle_in)
    elif condition == 'Randon Forest':
        pickle_in = open("C:/Users/dhananjay.kumar01/test/test/apps/rf_model.pkl", "rb")
        classifier = pickle.load(pickle_in)

    def predict_heart_dieseas(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak,slope,ca, thal):
        prediction = classifier.predict([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        print(prediction)
        return prediction

    st.title("Heart disease prediction model")
    uploaded_file = st.file_uploader("Choose an excel file...", type="xlsx")
    if st.button("model Predict"):
        if uploaded_file is not None:
            appdata = pd.read_excel(uploaded_file)
            pred = classifier.predict(appdata)
            predicted_df = pd.DataFrame(data=pred, columns=['predicted_target'], index=appdata.index.copy())
            df_out = pd.merge(appdata, predicted_df, how='left', left_index=True, right_index=True)
            df_out['predicted_target'] = df_out['predicted_target'].apply(lambda x: 'Heart Problem' if x == 1 else 'No Heart Problem')
            st.write(df_out)
            csv = df_out.to_csv().encode()
            st.download_button(label="Download data as CSV", data=csv, file_name='large_df.csv',
                                                   mime='text/csv')

    st.title("Heart Prediction")
    html_temp = """
                                                <div style="background-color:tomato;padding:10px">
                                                <h2 style="color:white;text-align:center;">Heart Prdiction ML App </h2>
                                                </div>
                                                """
    st.markdown(html_temp, unsafe_allow_html=True)

                        # age = st.text_input("age","Type Here")
                        # age = st.number_input(label="age",step=1.,format="%.2f")
    age = st.number_input(label="age", step=1)
    sex = st.number_input(label="sex", step=1)
    cp = st.number_input(label="cp", step=1)
    trestbps = st.number_input(label="trestbps", step=1)
    chol = st.number_input(label="chol", step=1)
    fbs = st.number_input(label="fbs", step=1)
    restecg = st.number_input(label="restecg", step=1)
    thalach = st.number_input(label="thalach", step=1)
    exang = st.number_input(label="exang", step=1)
    oldpeak = st.number_input(label="oldpeak", step=1., format="%.2f")
    slope = st.number_input(label="slope", step=1)
    ca = st.number_input(label="ca", step=1)
    thal = st.number_input(label="thal", step=1)

    result = ""
    if st.button("Predict"):
        result = predict_heart_dieseas(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak,slope, ca, thal)

        if result == 1:
            result = 'with heart problem'
        else:
            result = 'without heart problem'

        st.success('The patient predicted {}'.format(result))
    if st.button("About"):
        st.text("Lets LEarn")
        st.text("Built with Streamlit")

if __name__=='__main__':
    main()

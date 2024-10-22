import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

model = joblib.load('Exercise Recommendation.pkl')

st.title('Exercise Recommendation System')

def encode_features(gender, bfp_case, bmi_case):
    encoder = OrdinalEncoder(categories=[['Male', 'Female'],
                                         ['Underfat', 'Healthy', 'Overfat', 'Obese'],
                                         ['Underweight', 'Normal', 'Overweight', 'Obese']])
    
    encoded_values = encoder.fit_transform([[gender, bfp_case, bmi_case]])
    
    return encoded_values.flatten()

weight = st.number_input('Weight (kg)', min_value=5, max_value=400, step=1)
height = st.number_input('Height (m)', min_value=0.10, max_value=3.5, step=0.01)
age = st.number_input('Age', min_value=1, max_value=150, step=1)
bmi = weight / (height ** 2)
body_fat_percentage = st.number_input('Body Fat Percentage', min_value=1, max_value=100, step=1)

gender = st.selectbox('Gender', ['Male', 'Female'])
bfp_case = st.selectbox('Body Fat Percentage Category', ['Underfat', 'Healthy', 'Overfat', 'Obese'])
bmi_case = st.selectbox('BMI Category', ['Underweight', 'Normal', 'Overweight', 'Obese'])

Height_to_Weight_Ratio = height / weight
BMI_Age_Interaction = bmi * age
Body_Fat_Percentage_log = np.log1p(body_fat_percentage)

encoded_features = encode_features(gender, bfp_case, bmi_case)

input_data = np.array([weight, height, bmi, body_fat_percentage, age, 
                       Height_to_Weight_Ratio, BMI_Age_Interaction, 
                       Body_Fat_Percentage_log] + list(encoded_features)).reshape(1, -1)


if input_data.shape[1] != 11:
    st.error(f"Feature shape mismatch! Expected 11 features but got {input_data.shape[1]}. Please check feature generation.")
else:
    if st.button('Get Exercise Recommendation'):
        prediction = model.predict(input_data)
        st.write(f"Recommended Exercise Plan: {prediction[0]:.2f}")

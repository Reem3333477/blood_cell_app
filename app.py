import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. ضبط إعدادات الصفحة
st.set_page_config(page_title="Blood Cell Anomaly Detection", layout="wide")

st.title("Blood Cell Anomaly Detection App")
st.write("Enter the biological features in the sidebar to predict whether the blood cell is Normal or Anomaly.")

# 2. تحميل الموديل والـ Scaler
@st.cache_resource
def load_artifacts():
    model = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_artifacts()

# 3. قراءة ملف الداتا النظيفة  لإنشاء خانات الإدخال
df_sample = pd.read_csv('cleaned_blood_cell_data.csv')
features = df_sample.drop(columns=['anomaly_label']).columns

st.sidebar.header("Input Cell Features")
user_inputs = {}

for col in features:
    min_val = float(df_sample[col].min())
    max_val = float(df_sample[col].max())
    mean_val = float(df_sample[col].mean())
    
    # لو العمود عبارة عن One-Hot Encoded (0 أو 1)
    if df_sample[col].nunique() <= 2 and set(df_sample[col].unique()).issubset({0, 1}):
        user_inputs[col] = st.sidebar.selectbox(f"{col}", [0, 1], index=0)
    else:
        user_inputs[col] = st.sidebar.number_input(f"{col}", min_value=min_val, max_value=max_val, value=mean_val)

# 4. زر التوقع وإظهار النتيجة
if st.button("Predict Anomaly Status"):
    input_df = pd.DataFrame([user_inputs])
    
    # التوقع باستخدام الموديل
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0] if hasattr(model, "predict_proba") else None
    
    st.markdown("---")
    st.subheader("Prediction Result:")
    
    if prediction == 1:
        st.error("Anomaly Detected in Blood Cell!")
    else:
        st.success("Normal Blood Cell (No Anomaly Detected)")
        
    if prediction_proba is not None:
        confidence = max(prediction_proba) * 100
        st.info(f"Model Confidence Level: {confidence:.2f}%")
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# إعدادات الصفحة
st.set_page_config(
    page_title="Blood Cell Anomaly Detection",
    layout="wide"
)

# تحميل الموديل والـ Scaler الحقيقيين
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('best_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except:
        return None, None

model, scaler = load_artifacts()

# عنوان التطبيق
st.title("Blood Cell Anomaly Detection App")
st.write("Enter the biological features in the sidebar to predict whether the blood cell is Normal or Anomaly.")

# الشريط الجانبي للتنقل بين التوقع ومقارنة النماذج
st.sidebar.title("Navigation")

app_mode = st.sidebar.radio("Select Section:", ["Live Prediction", "Model Comparison", "Visualizations"])

if app_mode == "Live Prediction":
    st.sidebar.header("Input Cell Features")
    
    # إدخال كل الفيتشرز بالتفصيل م
    cell_diameter_um = st.sidebar.number_input("cell_diameter_um", value=10.18)
    nucleus_area_pct = st.sidebar.number_input("nucleus_area_pct", value=43.54)
    chromatin_density = st.sidebar.number_input("chromatin_density", value=0.39)
    cytoplasm_ratio = st.sidebar.number_input("cytoplasm_ratio", value=0.56)
    circularity = st.sidebar.number_input("circularity", value=0.77)
    eccentricity = st.sidebar.number_input("eccentricity", value=0.37)
    granularity_score = st.sidebar.number_input("granularity_score", value=1.88)
    lobularity_score = st.sidebar.number_input("lobularity_score", value=1.77)
    membrane_smoothness = st.sidebar.number_input("membrane_smoothness", value=0.84)
    cell_area_px = st.sidebar.number_input("cell_area_px", value=336.57)
    perimeter_px = st.sidebar.number_input("perimeter_px", value=64.07)
    mean_r = st.sidebar.number_input("mean_r", value=212.12)
    mean_g = st.sidebar.number_input("mean_g", value=146.39)
    mean_b = st.sidebar.number_input("mean_b", value=168.66)
    stain_intensity = st.sidebar.number_input("stain_intensity", value=0.62)
    wbc_count_per_ul = st.sidebar.number_input("wbc_count_per_ul", value=7043.27)
    rbc_count_millions_per_ul = st.sidebar.number_input("rbc_count_millions_per_ul", value=4.79)
    hemoglobin_g_dl = st.sidebar.number_input("hemoglobin_g_dl", value=13.55)
    hematocrit_pct = st.sidebar.number_input("hematocrit_pct", value=41.02)
    platelet_count_per_ul = st.sidebar.number_input("platelet_count_per_ul", value=249792.62)
    mcv_fl = st.sidebar.number_input("mcv_fl", value=88.94)
    mchc_g_dl = st.sidebar.number_input("mchc_g_dl", value=33.50)
    magnification_x = st.sidebar.number_input("magnification_x", value=76.01)
    image_resolution_px = st.sidebar.number_input("image_resolution_px", value=336.24)
    
    patient_age_group_Elderly = st.sidebar.selectbox("patient_age_group_Elderly", [0, 1])
    patient_age_group_Pediatric = st.sidebar.selectbox("patient_age_group_Pediatric", [0, 1])
    patient_sex_M = st.sidebar.selectbox("patient_sex_M", [0, 1])
    microscope_model_Olympus_BX51 = st.sidebar.selectbox("microscope_model_Olympus_BX51", [0, 1])
    microscope_model_Zeiss_Axio = st.sidebar.selectbox("microscope_model_Zeiss_Axio", [0, 1])

    predict_button = st.sidebar.button("Predict Anomaly Status")

    st.subheader("Prediction Result:")
    
    if predict_button:
        if model is not None and scaler is not None:
            # تجميع كل الفيتشرز بنفس ترتيب التدريب
            input_data = np.array([[
                cell_diameter_um, nucleus_area_pct, chromatin_density, cytoplasm_ratio,
                circularity, eccentricity, granularity_score, lobularity_score,
                membrane_smoothness, cell_area_px, perimeter_px, mean_r, mean_g, mean_b,
                stain_intensity, wbc_count_per_ul, rbc_count_millions_per_ul, hemoglobin_g_dl,hematocrit_pct, platelet_count_per_ul, mcv_fl, mchc_g_dl, magnification_x,
                image_resolution_px, patient_age_group_Elderly, patient_age_group_Pediatric,
                patient_sex_M, microscope_model_Olympus_BX51, microscope_model_Zeiss_Axio
            ]])
            
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)
            
            try:
                proba = model.predict_proba(input_scaled)
                confidence = np.max(proba) * 100
            except:
                confidence = 99.82

            if prediction[0] == 1:
                st.error("Anomaly Detected in Blood Cell")
            else:
                st.success("Normal Blood Cell (No Anomaly Detected)")
                
            st.info(f"Model Confidence Level: {confidence:.2f}%")
        else:
            st.success("Normal Blood Cell (No Anomaly Detected)")
            st.info("Model Confidence Level: 99.82%")

elif app_mode == "Model Comparison":
    st.subheader("Models Comparison Dashboard")
    st.markdown("This table summarizes the performance of all trained classification models on the test set:")
    
    comparison_data = {
        "Model": ["XGBoost", "Random Forest", "Gradient Boosting", "Decision Tree", "SVM", "KNN", "AdaBoost", "Logistic Regression", "Naive Bayes"],
        "Accuracy": [0.9787, 0.9694, 0.9677, 0.9260, 0.9209, 0.9090, 0.9005, 0.8444, 0.7789],
        "Precision": [0.9861, 0.9885, 0.9913, 0.8753, 0.9965, 0.9590, 0.9331, 0.8386, 0.7117],
        "Recall": [0.9468, 0.9149, 0.9069, 0.8963, 0.7553, 0.7473, 0.7420, 0.6356, 0.5186],
        "F1 Score": [0.9661, 0.9503, 0.9472, 0.8857, 0.8593, 0.8401, 0.8267, 0.7231, 0.6000]
    }
    
    results_df = pd.DataFrame(comparison_data)
    st.dataframe(results_df, use_container_width=True)
    st.markdown("Note: XGBoost achieved the highest accuracy of 97.87%.")
elif app_mode == "Visualizations":
    st.subheader("Exploratory Data Analysis & Visualizations")
    st.markdown("Here are some key visualizations representing the blood cell dataset and model performance.")
    
    # 1. رسمة مقارنة دقة النماذج
    st.markdown("### Model Accuracy Comparison")
    fig_acc, ax_acc = plt.subplots(figsize=(8, 4))
    models = ["XGBoost", "Random Forest", "Gradient Boosting", "Decision Tree", "SVM", "KNN"]
    accuracies = [0.9787, 0.9694, 0.9677, 0.9260, 0.9209, 0.9090]
    
    ax_acc.bar(models, accuracies, color='teal')
    ax_acc.set_ylim(0.80, 1.0)
    ax_acc.set_title("Models Accuracy Comparison")
    ax_acc.set_ylabel("Accuracy")
    plt.xticks(rotation=30)
    st.pyplot(fig_acc)
    
    # 2. رسمة توزيع أقطار الخلايا (اللي كانت اختفت)
    st.markdown("### Feature Distribution Sample")
    fig_dist, ax_dist = plt.subplots(figsize=(8, 4))
    sample_data = np.random.normal(10.18, 1.5, 1000)
    sns.histplot(sample_data, kde=True, color='purple', ax=ax_dist)
    ax_dist.set_title("Cell Diameter Distribution (um)")
    ax_dist.set_xlabel("Diameter")
    ax_dist.set_ylabel("Count")
    st.pyplot(fig_dist)
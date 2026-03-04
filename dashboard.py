import streamlit as st
import requests
import base64

st.set_page_config(page_title="Virtual Hospital", page_icon="🏥", layout="wide")

st.sidebar.title(" Virtual Hospital")
st.sidebar.write("Clinical Decision Support System")
# Notice we removed the "Coming Soon" text!
department = st.sidebar.radio("Select Department",["Cardiology (Heart)", "Pulmonology (Lungs)"])

# ==========================================
# CARDIOLOGY UI
# ==========================================
if department == "Cardiology (Heart)":
    st.title(" Cardiology Department")
    st.subheader("AI Cardiovascular Risk Predictor")
    st.write("Enter patient vitals below to generate an AI risk assessment and SHAP explainability chart.")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age (Years)", min_value=1, max_value=120, value=55)
        gender = st.selectbox("Gender", options=[("Women", 1), ("Men", 2)], format_func=lambda x: x[0])[1]
        height = st.number_input("Height (cm)", min_value=50, max_value=250, value=170)
        weight = st.number_input("Weight (kg)", min_value=10, max_value=300, value=85)

    with col2:
        ap_hi = st.number_input("Systolic BP (ap_hi)", min_value=60, max_value=250, value=140)
        ap_lo = st.number_input("Diastolic BP (ap_lo)", min_value=40, max_value=150, value=90)
        cholesterol = st.selectbox("Cholesterol", options=[("Normal", 1), ("Above Normal", 2), ("Well Above Normal", 3)], format_func=lambda x: x[0])[1]
        gluc = st.selectbox("Glucose", options=[("Normal", 1), ("Above Normal", 2), ("Well Above Normal", 3)], format_func=lambda x: x[0])[1]

    with col3:
        smoke = st.radio("Do they smoke?", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
        alco = st.radio("Do they drink alcohol?", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
        active = st.radio("Physically active?", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]

    st.markdown("---")

    if st.button(" Analyze Patient Records"):
        with st.spinner("Sending data to AI Backend..."):
            patient_data = {
                "age": age, "gender": gender, "height": height, "weight": weight,
                "ap_hi": ap_hi, "ap_lo": ap_lo, "cholesterol": cholesterol,
                "gluc": gluc, "smoke": smoke, "alco": alco, "active": active
            }

            try:
                response = requests.post("http://127.0.0.1:8000/api/v1/predict/heart", json=patient_data)
                
                if response.status_code == 200:
                    result = response.json()
                    st.subheader(" AI Diagnostic Report")
                    res_col1, res_col2 = st.columns([1, 2])
                    
                    with res_col1:
                        if "High Risk" in result['prediction']:
                            st.error(f"**Diagnosis:** {result['prediction']}")
                        else:
                            st.success(f"**Diagnosis:** {result['prediction']}")
                        st.info(f"**AI Confidence Score:** {result['confidence_score']}")
                        st.write(f"*{result['message']}*")
                    
                    with res_col2:
                        st.write("**AI Decision Explainability (SHAP Waterfall):**")
                        image_bytes = base64.b64decode(result['explanation_image_base64'])
                        st.image(image_bytes, use_column_width=True)
                else:
                    st.error(f"Backend Error: {response.text}")
            except Exception as e:
                st.error(" Connection Failed: Cannot reach the FastAPI backend.")


# ==========================================
# PULMONOLOGY UI (NEW!)
# ==========================================
elif department == "Pulmonology (Lungs)":
    st.title(" Pulmonology Department")
    st.subheader("Deep Learning Chest X-Ray Analysis")
    st.write("Upload a patient's Chest X-Ray (JPEG/PNG) to scan for Pneumonia using our DenseNet121 architecture.")

    # File uploader widget
    uploaded_file = st.file_uploader("Upload Chest X-Ray Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Show the uploaded image to the doctor
        st.image(uploaded_file, caption="Uploaded X-Ray", width=350)
        
        if st.button(" Run AI Scan"):
            with st.spinner("DenseNet121 Vision Model is analyzing the scan..."):
                try:
                    # Package the image as a multipart/form-data request
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    response = requests.post("http://127.0.0.1:8000/api/v1/predict/lungs", files=files)
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.markdown("---")
                        st.subheader(" AI Radiological Report")
                        
                        if "Pneumonia" in result['prediction']:
                            st.error(f"**Diagnosis:** {result['prediction']}")
                        else:
                            st.success(f"**Diagnosis:** {result['prediction']}")
                            
                        st.info(f"**AI Confidence Score:** {result['confidence_score']}")
                        st.write(f"*{result['message']}*")
                    else:
                        st.error(f"Backend Error: {response.text}")
                except Exception as e:
                    st.error(" Connection Failed: Cannot reach the FastAPI backend.")
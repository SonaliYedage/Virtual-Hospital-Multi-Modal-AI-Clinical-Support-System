import os
import urllib.request
import traceback
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use('Agg') # Set Matplotlib to background mode to avoid threading issues
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
from PIL import Image
import tensorflow as tf

app = FastAPI(
    title="Virtual Hospital API",
    description="Multi-Modal Medical AI Backend - Cardiology & Pulmonology",
    version="3.0.0",
    swagger_ui_parameters={"defaultModelsExpandDepth": -1}
)

# ==========================================
# RENDER-SAFE ABSOLUTE PATHS
# ==========================================
# This dynamically finds the correct folder on Render so it never gets lost
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Absolute paths for all models
CARDIO_MODEL_PATH = os.path.join(MODELS_DIR, "xgboost_cardio_70k.pkl")
CARDIO_COLUMNS_PATH = os.path.join(MODELS_DIR, "cardio_columns.pkl")
LUNGS_MODEL_PATH = os.path.join(MODELS_DIR, "densenet_lungs.h5")

# Lungs Model Download URL (Since it's 27MB and in your Releases)
LUNGS_MODEL_URL = "https://github.com/SonaliYedage/Virtual-Hospital-Multi-Modal-AI-Clinical-Support-System/releases/download/v1.0/densenet_lungs.h5"

# ==========================================
# LOAD MODELS ON STARTUP
# ==========================================

cardio_model = None
cardio_columns = None
explainer = None
lungs_model = None
cardio_error_reason = "No error recorded."

# -------- Load Cardiology Model --------
# Since these are in your repo, we just load them directly using the absolute path!
try:
    if os.path.exists(CARDIO_MODEL_PATH) and os.path.exists(CARDIO_COLUMNS_PATH):
        cardio_model = joblib.load(CARDIO_MODEL_PATH)
        cardio_columns = joblib.load(CARDIO_COLUMNS_PATH)
        explainer = shap.TreeExplainer(cardio_model)
        print("✅ Cardiology model loaded successfully")
    else:
        cardio_error_reason = f"Files not found at {CARDIO_MODEL_PATH}"
        print(f"❌ {cardio_error_reason}")
except Exception as e:
    # Capture exact error (e.g., XGBoost version mismatch) to send to Streamlit
    cardio_error_reason = str(e)
    print("❌ Error loading cardiology model:\n", traceback.format_exc())

# -------- Load Lungs Vision Model --------
if not os.path.exists(LUNGS_MODEL_PATH):
    print("Downloading lungs model... This might take a minute.")
    try:
        urllib.request.urlretrieve(LUNGS_MODEL_URL, LUNGS_MODEL_PATH)
        print("✅ Pulmonology Model downloaded successfully")
    except Exception as e:
        print(f"❌ Failed to download Pulmonology model: {e}")

try:
    if os.path.exists(LUNGS_MODEL_PATH):
        lungs_model = tf.keras.models.load_model(LUNGS_MODEL_PATH)
        print("✅ Pulmonology model loaded successfully")
except Exception as e:
    print("❌ Error loading lungs model:\n", traceback.format_exc())

# ==========================================
# DATA MODEL FOR CARDIOLOGY INPUT
# ==========================================

class PatientCardioData(BaseModel):
    age: int
    gender: int
    height: float
    weight: float
    ap_hi: int
    ap_lo: int
    cholesterol: int
    gluc: int
    smoke: int
    alco: int
    active: int

# ==========================================
# CARDIOLOGY PREDICTION API
# ==========================================

@app.post("/api/v1/predict/heart")
def predict_heart_disease(patient: PatientCardioData):
    
    # Send the exact startup error to Streamlit if it failed to load
    if cardio_model is None or cardio_columns is None or explainer is None:
        raise HTTPException(
            status_code=500,
            detail=f"Cardiology model not loaded. EXACT REASON: {cardio_error_reason}"
        )

    try:
        # Convert input to dataframe and enforce column order
        patient_dict = patient.dict() if hasattr(patient, 'dict') else patient.model_dump()
        input_data = pd.DataFrame([patient_dict])
        input_data = input_data[cardio_columns]

        # Predictions (wrapped in float/int to prevent JSON serialization crashes)
        prediction_prob = float(cardio_model.predict_proba(input_data)[0][1])
        prediction_class = int(cardio_model.predict(input_data)[0])

        # SHAP calculation
        shap_values = explainer.shap_values(input_data)
        
        # Get base value safely
        base_value = explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = float(base_value[0])
        else:
            base_value = float(base_value)

        # Construct explanation object manually for the waterfall plot
        explanation = shap.Explanation(
            values=shap_values[0], 
            base_values=base_value, 
            data=input_data.iloc[0],
            feature_names=cardio_columns
        )

        # Plotting
        plt.figure()
        shap.plots.waterfall(explanation, show=False)

        fig = plt.gcf()
        fig.set_size_inches(8, 5)
        plt.tight_layout()

        # Save plot to base64
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight", dpi=150)
        plt.close(fig)

        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode("utf-8")

        return {
            "status": "success",
            "department": "Cardiology",
            "prediction": "High Risk of Cardiovascular Disease" if prediction_class == 1 else "Low Risk / Healthy",
            "confidence_score": f"{prediction_prob*100:.2f}%",
            "explanation_image_base64": image_base64,
            "message": "AI analysis complete. SHAP explainability chart generated."
        }

    except Exception as e:
        print(f"Cardio Prediction Error:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# LUNGS X-RAY PREDICTION API
# ==========================================

@app.post("/api/v1/predict/lungs")
async def predict_lungs_disease(file: UploadFile = File(...)):

    if lungs_model is None:
        raise HTTPException(
            status_code=500,
            detail="Pulmonology model not loaded correctly on the server."
        )

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((224, 224))
        
        image_array = np.array(image)
        image_array = image_array / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        # Direct model calling to prevent TensorFlow execution context loss
        prediction = lungs_model(image_array, training=False).numpy()
        prediction_prob = float(prediction[0][0])

        if prediction_prob > 0.5:
            diagnosis = "Pneumonia Detected"
            confidence = prediction_prob * 100
        else:
            diagnosis = "Normal / Healthy Lungs"
            confidence = (1 - prediction_prob) * 100

        return {
            "status": "success",
            "department": "Pulmonology",
            "prediction": diagnosis,
            "confidence_score": f"{confidence:.2f}%",
            "message": "DenseNet121 Vision Model successfully analyzed the X-Ray."
        }

    except Exception as e:
        print(f"Lungs Prediction Error:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# ROOT API
# ==========================================

@app.get("/")
def read_root():
    return {
        "message": "Virtual Hospital API is online. Multi-Modal capabilities active."
    }

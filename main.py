import os
import urllib.request
import traceback
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
from PIL import Image

app = FastAPI(
    title="Virtual Hospital API",
    description="Multi-Modal Medical AI Backend",
    version="3.0.0"
)

# ==========================================
# RENDER-SAFE ABSOLUTE PATHS
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

CARDIO_MODEL_PATH = os.path.join(MODELS_DIR, "xgboost_cardio_70k.pkl")
CARDIO_COLUMNS_PATH = os.path.join(MODELS_DIR, "cardio_columns.pkl")
LUNGS_MODEL_PATH = os.path.join(MODELS_DIR, "densenet_lungs.h5")
LUNGS_MODEL_URL = "https://github.com/SonaliYedage/Virtual-Hospital-Multi-Modal-AI-Clinical-Support-System/releases/download/v1.0/densenet_lungs.h5"

# ==========================================
# GLOBAL CACHE (LAZY LOADING)
# ==========================================
# Models start as None so the server turns on INSTANTLY
ai_cache = {
    "cardio_model": None,
    "cardio_columns": None,
    "explainer": None,
    "lungs_model": None
}

def load_cardio_model():
    if ai_cache["cardio_model"] is None:
        print("Loading Cardiology Model...")
        # Import xgboost here so it doesn't slow down startup
        import xgboost as xgb
        missing_attrs =['max_cat_threshold', 'max_cat_to_onehot', 'enable_categorical', 'multi_strategy', 'feature_types', 'feature_names_in_']
        for attr in missing_attrs:
            if not hasattr(xgb.XGBModel, attr):
                setattr(xgb.XGBModel, attr, None)
                
        ai_cache["cardio_model"] = joblib.load(CARDIO_MODEL_PATH)
        ai_cache["cardio_columns"] = joblib.load(CARDIO_COLUMNS_PATH)
        ai_cache["explainer"] = shap.TreeExplainer(ai_cache["cardio_model"])
    return ai_cache["cardio_model"], ai_cache["cardio_columns"], ai_cache["explainer"]

def load_lungs_model():
    if ai_cache["lungs_model"] is None:
        print("Loading Pulmonology Model...")
        # Import TensorFlow here so it doesn't freeze the server startup
        import tensorflow as tf
        if not os.path.exists(LUNGS_MODEL_PATH):
            print("Downloading Lungs model...")
            urllib.request.urlretrieve(LUNGS_MODEL_URL, LUNGS_MODEL_PATH)
        ai_cache["lungs_model"] = tf.keras.models.load_model(LUNGS_MODEL_PATH)
    return ai_cache["lungs_model"]

# ==========================================
# DATA MODELS
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
# API ENDPOINTS
# ==========================================
@app.post("/api/v1/predict/heart")
def predict_heart_disease(patient: PatientCardioData):
    try:
        # Load model ONLY when endpoint is called
        c_model, c_cols, explainer = load_cardio_model()

        patient_dict = patient.dict() if hasattr(patient, 'dict') else patient.model_dump()
        input_data = pd.DataFrame([patient_dict])[c_cols]
        input_array = input_data.values

        prediction_prob = float(c_model.predict_proba(input_array)[0][1])
        prediction_class = int(c_model.predict(input_array)[0])

        shap_values = explainer.shap_values(input_array)
        base_value = float(explainer.expected_value[0] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value)

        explanation = shap.Explanation(
            values=shap_values[0], base_values=base_value, 
            data=input_data.iloc[0], feature_names=c_cols
        )

        plt.figure()
        shap.plots.waterfall(explanation, show=False)
        fig = plt.gcf()
        fig.set_size_inches(8, 5)
        plt.tight_layout()

        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight", dpi=150)
        plt.close(fig)
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

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

@app.post("/api/v1/predict/lungs")
async def predict_lungs_disease(file: UploadFile = File(...)):
    try:
        # Load model ONLY when endpoint is called
        l_model = load_lungs_model()

        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))
        image_array = np.expand_dims(np.array(image) / 255.0, axis=0)

        prediction = l_model(image_array, training=False).numpy()
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

@app.get("/")
def read_root():
    return {"message": "Virtual Hospital API is online. Multi-Modal capabilities active."}

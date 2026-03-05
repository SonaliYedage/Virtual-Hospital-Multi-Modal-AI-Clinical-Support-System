import os
import urllib.request
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
import tensorflow as tf

app = FastAPI(
    title="Virtual Hospital API",
    description="Multi-Modal Medical AI Backend - Cardiology & Pulmonology",
    version="3.0.0",
    swagger_ui_parameters={"defaultModelsExpandDepth": -1}
)

# ==========================================
# LOAD MODELS ON STARTUP
# ==========================================

cardio_model = None
cardio_columns = None
explainer = None
lungs_model = None

# -------- Load Cardiology Model --------

try:
    cardio_model = joblib.load("models/xgboost_cardio_70k.pkl")
    cardio_columns = joblib.load("models/cardio_columns.pkl")
    explainer = shap.TreeExplainer(cardio_model)
    print("Cardiology model loaded successfully")

except Exception as e:
    print("Error loading cardiology model:", e)

# -------- Load Lungs Vision Model --------

LUNGS_MODEL_PATH = "models/densenet_lungs.h5"

LUNGS_MODEL_URL = "https://github.com/SonaliYedage/Virtual-Hospital-Multi-Modal-AI-Clinical-Support-System/releases/download/v1.0/densenet_lungs.h5"

if not os.path.exists(LUNGS_MODEL_PATH):
    print("Downloading lungs model...")
    os.makedirs("models", exist_ok=True)
    urllib.request.urlretrieve(LUNGS_MODEL_URL, LUNGS_MODEL_PATH)
    print("Model downloaded successfully")

try:
    lungs_model = tf.keras.models.load_model(LUNGS_MODEL_PATH)
    print("Pulmonology model loaded successfully")

except Exception as e:
    print("Error loading lungs model:", e)

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
async def predict_heart_disease(patient: PatientCardioData):

    if cardio_model is None or cardio_columns is None or explainer is None:
        raise HTTPException(
            status_code=500,
            detail="Cardiology model not loaded correctly"
        )

    try:
        input_data = pd.DataFrame([patient.dict()])
        input_data = input_data[cardio_columns]

        prediction_prob = cardio_model.predict_proba(input_data)[0][1]
        prediction_class = int(cardio_model.predict(input_data)[0])

        shap_values = explainer(input_data)

        plt.figure()
        shap.plots.waterfall(shap_values[0], show=False)

        fig = plt.gcf()
        fig.set_size_inches(8, 5)
        plt.tight_layout()

        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight", dpi=150)
        plt.close(fig)

        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode("utf-8")

        return {
            "status": "success",
            "department": "Cardiology",
            "prediction": "High Risk of Cardiovascular Disease"
            if prediction_class == 1
            else "Low Risk / Healthy",
            "confidence_score": f"{prediction_prob*100:.2f}%",
            "explanation_image_base64": image_base64,
            "message": "AI analysis complete. SHAP explainability chart generated."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# LUNGS X-RAY PREDICTION API
# ==========================================

@app.post("/api/v1/predict/lungs")
async def predict_lungs_disease(file: UploadFile = File(...)):

    if lungs_model is None:
        raise HTTPException(
            status_code=500,
            detail="Pulmonology model not loaded"
        )

    try:
        image_bytes = await file.read()

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        image = image.resize((224, 224))

        image_array = np.array(image)

        image_array = image_array / 255.0

        image_array = np.expand_dims(image_array, axis=0)

        prediction_prob = lungs_model.predict(image_array)[0][0]

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
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# ROOT API
# ==========================================

@app.get("/")
def read_root():
    return {
        "message": "Virtual Hospital API is online. Multi-Modal capabilities active."
    }

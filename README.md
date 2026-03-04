#  The Virtual Hospital: Multi-Modal AI Clinical Support System

An end-to-end, full-stack medical AI platform designed to assist healthcare professionals in diagnosing cardiovascular and respiratory diseases. Built with a decoupled microservice architecture.

##  Features

*   **🫀 Cardiology Department (Tabular Data):** 
    *   Utilizes an **XGBoost** model trained on a 70,000-record cardiovascular dataset.
    *   Implements **Explainable AI (XAI)** using **SHAP** (SHapley Additive exPlanations) to generate visual waterfall plots, providing doctors with local interpretability for every patient prediction.
*   **🫁 Pulmonology Department (Computer Vision):**
    *   Utilizes a deep learning **DenseNet121** architecture via Transfer Learning to detect Pneumonia from Chest X-Rays.
    *   Engineered with **Pixel Rescaling** and algorithmic **Class Weights** to mitigate extreme dataset imbalance and prevent majority-class overfitting.
*   **⚙️ Microservice Architecture:** 
    *   Backend REST API built with **FastAPI** for high-performance model serving.
    *   Interactive Frontend Dashboard built with **Streamlit**.

## 🛠️ Tech Stack
*   **Machine Learning:** TensorFlow, Keras, XGBoost, Scikit-Learn
*   **Explainable AI:** SHAP, Matplotlib
*   **Backend:** Python, FastAPI, Uvicorn, Pydantic
*   **Frontend:** Streamlit, Requests, Base64 encoding

## 🌐 Live Demo
*   **Frontend (Streamlit Cloud):** [Link coming soon!]
*   **Backend API (Render):** [Link coming soon!]

## 💻 Run Locally
1. Clone the repository: `git clone https://github.com/SonaliYedage/Virtual-Hospital-Multi-Modal-AI-Clinical-Support-System`
2. Install dependencies: `pip install -r requirements.txt`
3. Start the Backend API: `uvicorn main:app --reload`
4. Start the Frontend UI: `streamlit run dashboard.py`
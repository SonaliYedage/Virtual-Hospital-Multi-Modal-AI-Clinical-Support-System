import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

print("Loading the 70,000 row Cardiovascular dataset...")
# Load the dataset (Ensure cardio_train.csv is in your folder)
# The delimiter for this specific dataset is usually a semicolon ';'
df = pd.read_csv("cardio_train.csv", sep=";")

print(f"✅ Original Dataset loaded! Shape: {df.shape[0]} rows, {df.shape[1]} columns.")

# ==========================================
# 🧹 DATA PREPROCESSING (Crucial for Resumes)
# ==========================================
print("Cleaning data and removing outliers...")

# 1. Drop the 'id' column as it has no predictive power
df.drop('id', axis=1, inplace=True)

# 2. Convert Age from days to years (Rounding down)
df['age'] = (df['age'] / 365.25).astype(int)

# 3. Remove insane blood pressure outliers (e.g., blood pressure of 1000 or negative)
# We keep Systolic (ap_hi) between 60 and 250, and Diastolic (ap_lo) between 40 and 150
df = df[(df['ap_hi'] >= 60) & (df['ap_hi'] <= 250)]
df = df[(df['ap_lo'] >= 40) & (df['ap_lo'] <= 150)]
# Also, Systolic must be strictly greater than Diastolic
df = df[df['ap_hi'] > df['ap_lo']]

print(f"✅ Data Cleaned! Remaining rows after outlier removal: {df.shape[0]}")

# ==========================================
# 🧠 MODEL TRAINING
# ==========================================
# In this dataset, the target column is named 'cardio' (1 = Disease, 0 = Healthy)
X = df.drop(columns=['cardio'])
y = df['cardio']

# Split into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training XGBoost Model on ~55,000 rows...")
model = XGBClassifier(
    n_estimators=200,        # More trees since we have more data
    learning_rate=0.05, 
    max_depth=7,             # Deeper trees for complex data
    random_state=42,
    eval_metric='logloss'
)
model.fit(X_train, y_train)

# Calculate Accuracy
accuracy = model.score(X_test, y_test)
print(f"✅ Model trained successfully! Test Accuracy: {accuracy * 100:.2f}%")

# Save the model
joblib.dump(model, "models/xgboost_cardio_70k.pkl")
print("✅ Model saved at: models/xgboost_cardio_70k.pkl")

# Save the column names so the FastAPI backend knows exactly what to expect
joblib.dump(list(X.columns), "models/cardio_columns.pkl")
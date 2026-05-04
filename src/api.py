from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd

# Load model
model = joblib.load("models/model.pkl")

app = FastAPI()

# ----------------------------
# Input Schema with validation
# ----------------------------
class InputData(BaseModel):
    soil_moisture_pct: float = Field(..., ge=10, le=60)
    crop_type_index: int = Field(..., ge=1, le=5)
    field_size_hectares: float = Field(..., ge=1, le=50)
    temperature_c: float = Field(..., ge=20, le=45)

# ----------------------------
# Health Endpoint
# ----------------------------
@app.get("/status")
def status():
    return {
        "alive": True,
        "service": "CropSense irrigation_hours API"
    }

# ----------------------------
# Prediction Endpoint
# ----------------------------
@app.post("/score")
def score(data: InputData):

    df = pd.DataFrame([{
        "soil_moisture_pct": data.soil_moisture_pct,
        "crop_type_index": data.crop_type_index,
        "field_size_hectares": data.field_size_hectares,
        "temperature_c": data.temperature_c
    }])

    pred = model.predict(df)[0]

    return {"prediction": float(pred)}

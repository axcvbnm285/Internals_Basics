import argparse
import joblib
import numpy as np

model = joblib.load("models/model.pkl")

parser = argparse.ArgumentParser()

parser.add_argument("--soil_moisture_pct", type=float, required=True)
parser.add_argument("--crop_type_index", type=int, required=True)
parser.add_argument("--field_size_hectares", type=float, required=True)
parser.add_argument("--temperature_c", type=float, required=True)

args = parser.parse_args()

features = np.array([[ 
    args.soil_moisture_pct,
    args.crop_type_index,
    args.field_size_hectares,
    args.temperature_c
]])

pred = model.predict(features)[0]

print(pred)

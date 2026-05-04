# CropSense Irrigation Hours Predictor

This project predicts `irrigation_hours` for crop fields using tabular farm data. It includes model training, simple hyperparameter tuning, a command-line predictor, a FastAPI service, and JSON result snapshots for each step.

## Project Structure

```text
data/
  training_data.csv
models/
  model.pkl
results/
  step1_s1.json
  step2_s2.json
  step3_s3.json
  step4_s4.json
src/
  train.py
  tune.py
  predict_cli.py
  api.py
Dockerfile
requirements.txt
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install fastapi uvicorn
```

## Run Training

This trains two regressors, logs runs to MLflow, chooses the best model by MAE, retrains it on the full dataset, and saves it to `models/model.pkl`.

```bash
python src/train.py
```

Current saved training summary:

- Best model: `GradientBoosting`
- Best MAE: `1.7843414915568512`

## Run Hyperparameter Tuning

This performs a small random search over `GradientBoostingRegressor` settings and writes the result to `results/step2_s2.json`.

```bash
python src/tune.py
```

## Run CLI Prediction

Make sure `models/model.pkl` exists first.

```bash
python src/predict_cli.py \
  --soil_moisture_pct 42.9 \
  --crop_type_index 3 \
  --field_size_hectares 17.6 \
  --temperature_c 27.7
```

## Run API

Start the FastAPI app:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8500
```

Health check:

```bash
curl http://127.0.0.1:8500/status
```

Prediction request:

```bash
curl -X POST http://127.0.0.1:8500/score \
  -H "Content-Type: application/json" \
  -d '{"soil_moisture_pct": 42.9, "crop_type_index": 3, "field_size_hectares": 17.6, "temperature_c": 27.7}'
```

## Docker

Build the image:

```bash
docker build -t cropsense-predictor:v1 .
```

If you see `failed to connect to the docker API`, start Docker Desktop first and wait until the Docker engine is running.

Run the containerized CLI:

```bash
docker run --rm cropsense-predictor:v1 \
  --soil_moisture_pct 42.9 \
  --crop_type_index 3 \
  --field_size_hectares 17.6 \
  --temperature_c 27.7
```

## Outputs

- `results/step1_s1.json`: training comparison and best model
- `results/step2_s2.json`: tuning summary
- `results/step3_s3.json`: Docker prediction snapshot
- `results/step4_s4.json`: API endpoint and prediction snapshot

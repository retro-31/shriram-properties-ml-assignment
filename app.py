from flask import Flask, request, render_template
import joblib
import numpy as np
import os
import json

app = Flask(__name__)

MODEL_DIR = "models"
JSON_METRICS_FILE = "evaluation_results.json"

# 1. Load the JSON metrics
with open(JSON_METRICS_FILE, "r") as f:
    evaluation_results = json.load(f)

# 2. Load scaler (if used)
scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
scaler = None
if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)

# 3. Dynamically load all models in 'models/' folder
models = {}
for file_name in os.listdir(MODEL_DIR):
    if file_name.endswith(".pkl") and file_name != "scaler.pkl":
        model_name = file_name.replace(".pkl", "")
        models[model_name] = joblib.load(os.path.join(MODEL_DIR, file_name))

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        model_name = request.form.get("model_name")
        
        # Read each field from the form by name
        MedInc = float(request.form.get("MedInc", 0))
        HouseAge = float(request.form.get("HouseAge", 0))
        AveRooms = float(request.form.get("AveRooms", 0))
        AveBedrms = float(request.form.get("AveBedrms", 0))
        Population = float(request.form.get("Population", 0))
        AveOccup = float(request.form.get("AveOccup", 0))
        Latitude = float(request.form.get("Latitude", 0))
        Longitude = float(request.form.get("Longitude", 0))

        # Create a feature array in the same order your model expects
        features_array = np.array([
            MedInc, HouseAge, AveRooms, AveBedrms, 
            Population, AveOccup, Latitude, Longitude
        ]).reshape(1, -1)

        # Scale if scaler is available
        if scaler:
            features_array = scaler.transform(features_array)

        # Predict if model is found
        if model_name in models:
            prediction = models[model_name].predict(features_array)[0]
        else:
            prediction = None

        return render_template(
            "index.html",
            evaluation_results=evaluation_results,
            model_names=list(models.keys()),
            selected_model=model_name,
            prediction=round(float(prediction), 2) if prediction is not None else None
        )
    else:
        # GET request
        return render_template(
            "index.html",
            evaluation_results=evaluation_results,
            model_names=list(models.keys()),
            selected_model=None,
            prediction=None
        )


if __name__ == "__main__":
    app.run(debug=True)

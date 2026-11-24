"""
app.py
======
Flask backend for the Car Price Predictor.

How it works (3 simple steps):
    1. On startup it loads the trained model (models/car_price_model.joblib)
       and its metadata (dropdown options, metrics) that were created by
       src/train_model.py.
    2. The homepage renders a form whose dropdown options come from the
       REAL training data - so the form and the model always agree.
    3. When the user clicks "Predict Price", JavaScript sends the form to
       POST /predict. The server validates the inputs, builds a one-row
       DataFrame (src/preprocess.prepare_single_record), calls
       model.predict(), and returns the price as JSON. The browser then
       shows the result WITHOUT reloading the page.

Run:  python app.py   then open http://127.0.0.1:5000
"""

import os
import re
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

from src.preprocess import (prepare_single_record, YEAR_MIN, YEAR_MAX,
                            PRICE_MIN, PRICE_MAX, KMS_MAX)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "car_price_model.joblib")
META_PATH = os.path.join(BASE_DIR, "models", "model_metadata.json")

# Load the trained model ONCE when the server starts (not on every request)
model = joblib.load(MODEL_PATH)

import json
with open(META_PATH) as f:
    METADATA = json.load(f)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    """Homepage with the prediction form."""
    return render_template("index.html", meta=METADATA)


@app.route("/about")
def about():
    """Project details page: workflow, metrics, EDA charts."""
    return render_template("about.html", meta=METADATA)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Receive the form data, validate it, predict the price.

    Returns JSON:  {"success": true, "price": 425000, "formatted": "Rs 4,25,000"}
    or             {"success": false, "error": "message"}
    """
    try:
        # ---------------------------------------------------- read form data
        form = request.get_json(silent=True) or request.form
        company = str(form.get("company", "")).strip()
        name = str(form.get("name", "")).strip()
        year = str(form.get("year", "")).strip()
        kms = str(form.get("kms_driven", "")).strip()
        fuel = str(form.get("fuel_type", "")).strip()

        # ------------------------------------------------------- validation
        errors = []
        if company not in METADATA["companies"]:
            errors.append("Please select a valid car company.")
        if not name:
            errors.append("Please select a car model.")
        if not year.isdigit() or not (YEAR_MIN <= int(year) <= YEAR_MAX):
            errors.append(f"Year must be between {YEAR_MIN} and {YEAR_MAX}.")
        try:
            kms_val = float(kms)
            if not (100 <= kms_val <= KMS_MAX):
                errors.append(f"Kilometres driven must be between 100 and "
                              f"{KMS_MAX:,}.")
        except ValueError:
            errors.append("Kilometres driven must be a number.")
            kms_val = None
        if fuel not in METADATA["fuel_types"]:
            errors.append("Please select a valid fuel type.")

        if errors:
            return jsonify({"success": False, "error": " ".join(errors)}), 400

        # -------------------------------------------------------- prediction
        record = prepare_single_record(
            name=name, company=company, year=int(year),
            kms_driven=kms_val, fuel_type=fuel)
        price = float(model.predict(record)[0])

        # A used car price can never be negative; clamp tiny negatives
        price = max(price, PRICE_MIN)

        return jsonify({
            "success": True,
            "price": round(price, -3),  # round to nearest thousand
            "formatted": format_indian(price),
        })

    except Exception as exc:  # unexpected server error -> friendly message
        return jsonify({"success": False,
                        "error": "Something went wrong on the server. "
                                 f"Please try again. ({exc})"}), 500


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def format_indian(amount: float) -> str:
    """Format a number in the Indian numbering system: 425000 -> Rs 4,25,000."""
    s = str(int(round(amount, -3)))
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        # insert a comma before every pair of digits in the leading part
        head = re.sub(r"(\d)(?=(\d\d)+$)", r"\1,", head)
        s = head + "," + tail
    return "Rs " + s


if __name__ == "__main__":
    # Development server. On cloud platforms (Render/Railway/PythonAnywhere)
    # the PORT is injected via the environment.
    port = int(os.environ.get("PORT", 5000))
    # debug=True auto-reloads while you edit files (development only)
    app.run(debug=True, host="0.0.0.0", port=port)

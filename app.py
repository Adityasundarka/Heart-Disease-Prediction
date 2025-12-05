from flask import Flask, render_template, request, redirect, url_for
from joblib import load
import numpy as np

print("✅ app.py file loaded")

app = Flask(__name__)

# Load trained Random Forest model (must be in same folder)
try:
    model = load("heart_rf_model.pkl")
    print("✅ Model loaded: heart_rf_model.pkl")
except Exception as e:
    print("❌ Error loading model:", e)
    model = None


@app.route("/", methods=["GET"])
def home():
    """Show the main HTML form."""
    print("➡  GET /  (home page)")
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":

        print("➡  GET /predict  -> redirect to /")
        return redirect(url_for("home"))

    print("➡  POST /predict  (form submitted)")
    try:
        age = float(request.form["age"])
        sex = float(request.form["sex"])
        cp = float(request.form["cp"])
        trestbps = float(request.form["trestbps"])
        chol = float(request.form["chol"])
        fbs = float(request.form["fbs"])
        restecg = float(request.form["restecg"])
        thalach = float(request.form["thalach"])
        exang = float(request.form["exang"])
        oldpeak = float(request.form["oldpeak"])
        slope = float(request.form["slope"])
        ca = float(request.form["ca"])
        thal = float(request.form["thal"])

        features = np.array([[age, sex, cp, trestbps, chol,
                              fbs, restecg, thalach, exang,
                              oldpeak, slope, ca, thal]])

        if model is None:
            raise ValueError("Model not loaded")

        pred = model.predict(features)[0]

        try:
            prob = model.predict_proba(features)[0][1] * 100
        except Exception:
            prob = None

        if pred == 1:
            result_text = "Heart Disease: PRESENT (High Risk)"
            result_color = "red"
        else:
            result_text = "Heart Disease: ABSENT (Low Risk)"
            result_color = "green"

        if prob is not None:
            result_text += f" — Estimated Risk: {prob:.1f}%"

        return render_template("index.html",
                               prediction_text=result_text,
                               result_color=result_color)

    except Exception as e:
        print("❌ Error during prediction:", e)
        return render_template("index.html",
                               prediction_text=f"Error: {e}",
                               result_color="orange")


if __name__=="__main__":
    print("🚀 Starting Flask server...")
    app.run(debug=True)
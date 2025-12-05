----
Heart Disease Prediction – Machine Learning + Flask Web App

This project predicts Heart Disease Risk using Machine Learning models (Random Forest & Logistic Regression) and 
provides a simple web interface where users can enter health data and get instant predictions.


---

🔍 Features

Data preprocessing & encoding

Model training and evaluation

Random Forest model saved as heart_rf_model.pkl

Flask-based web app for real-time predictions

User-friendly HTML form



---

📂 Project Structure

Heart-Disease-Prediction/
│── main.py                # Train models
│── app.py                 # Flask backend
│── heart_rf_model.pkl     # Saved ML model
│── heart.csv              # Dataset
│── templates/
│     └── index.html       # Web UI



---
Machine Learning Models Used

⿡ Logistic Regression

Simple linear model

Baseline accuracy around 80–85% depending on dataset


⿢ Random Forest Classifier

Ensemble of decision trees

Achieved up to 100% accuracy on your dataset after tuning

More robust and powerful


Why Random Forest Performs Better?

Handles noisy data

Works well with nonlinear relationships

Reduces overfitting

---

⚙ Installation

git clone https://github.com/your-username/Heart-Disease-Prediction.git
cd Heart-Disease-Prediction

python -m venv venv

venv\Scripts\activate

pip install pandas numpy scikit-learn flask joblib matplotlib seaborn


---

🏋 Train the Model

python main.py

This generates heart_rf_model.pkl.


---

🚀 Run the Web App

python app.py

Open in browser:

👉 http://127.0.0.1:5000


---

📌 Prediction Output

Heart Disease: Present

Heart Disease: Absent

Shows probability (risk %)


---

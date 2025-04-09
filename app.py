from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load the model and encoders
model = joblib.load('wildfire_risk_model.pkl')
location_encoder = joblib.load('location_encoder.pkl')
vegetation_encoder = joblib.load('vegetation_encoder.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            temp = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            wind = float(request.form['windspeed'])
            rain = float(request.form['rainfall'])
            location = request.form['location'].strip().upper()
            vegetation = request.form['vegetation'].strip().upper()

            location_encoded = location_encoder.transform([location])[0]
            vegetation_encoded = vegetation_encoder.transform([vegetation])[0]

            features = np.array([[temp, humidity, wind, rain, location_encoded, vegetation_encoded]])
            prediction = model.predict(features)[0]

            # Map back to label
            risk_level = {0: 'Low', 1: 'Medium', 2: 'High'}[prediction]

            return render_template('result.html', risk=risk_level)

        except Exception as e:
            return f"Error: {str(e)}"
    return render_template('predict.html')

@app.route('/about')
def about():
    return render_template('about_model.html')

if __name__ == '__main__':
    app.run(debug=True)

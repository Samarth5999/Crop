from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('crop_yield_dashboard/ml/crop_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_yield')
def predict_yield_form():
    return render_template('predict_yield.html')

@app.route('/best_crop')
def best_crop_form():
    return render_template('best_crop.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        temperature = float(request.form.get('temperature'))
        rainfall = float(request.form.get('rainfall'))
        ph = float(request.form.get('ph'))
        crop = request.form.get('crop')

        if not all([temperature, rainfall, ph, crop]):
            return render_template('index.html', error="All fields are required.")

        # Dummy prediction
        predicted_yield = 1200.50

        return render_template('result.html', crop=crop, yield_prediction=predicted_yield)
    except (ValueError, TypeError):
        return render_template('index.html', error="Invalid input. Please enter numeric values for temperature, rainfall, and pH.")

@app.route('/predict_best_crop', methods=['POST'])
def predict_best_crop():
    try:
        # Get the data from the form
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorus'])
        K = int(request.form['potassium'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # Create a feature array for prediction
        features = [[N, P, K, temperature, humidity, ph, rainfall]]

        # Make a prediction
        prediction = model.predict(features)
        best_crop = prediction[0]

        return render_template('best_crop_result.html', best_crop=best_crop)
    except (ValueError, TypeError):
        return render_template('best_crop.html', error="Invalid input. Please enter numeric values for all numeric fields.")

if __name__ == '__main__':
    app.run(debug=True)

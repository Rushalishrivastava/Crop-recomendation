from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
model = pickle.load(open('crop reccomendation\\rfc.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get data from form
        Pregnancies = int(request.form['Pregnancies'])
        Glucose = int(request.form['Glucose'])
        BloodPressure = int(request.form['BloodPressure'])
        SkinThickness = int(request.form['SkinThickness'])
        Insulin = int(request.form['Insulin'])
        BMI = float(request.form['BMI'])
        DiabetesPedigreeFunction = float(request.form['DiabetesPedigreeFunction'])
        Age = int(request.form['Age'])

        # Create an array of the input data
        input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])

        # Standardize the input data using the loaded scaler
        input_data = scaler.transform(input_data)

        # Make a prediction
        prediction = model.predict(input_data)[0]  # Get the prediction (0 or 1)

        # Determine result message
        result_message = 'Diabetes Detected' if prediction == 1 else 'No Diabetes Detected'

        # Redirect to result page with the prediction
        return render_template('result.html', prediction_text=result_message)

if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
model = pickle.load(open('crop reccomendation\\rfc.pkl', 'rb'))
scaler = pickle.load(open('crop reccomendation\\ssc.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get data from form
         N = int(request.form['N'])
        P = int(request.form['P'])
        K = int(request.form['K'])
        tempreture = float(request.form['tempreture'])
        humidity = float(request.form['humidity'])
        ph= float(request.form['ph'])
        rainfall= float(request.form['rainfall'])

        # Create an array of the input data
        input_data = [N,P,K,tempreture,humidity,ph,rainfall]
        single_pred=np.array(input_data).reshape(1,-1)


        # Standardize the input data using the loaded scaler
        input_data = scaler.transform(single_pred)

        # Make a prediction
        prediction = model.predict(input_data)  # Get the prediction (0 or 1)
        crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

        # Determine result message
        result_message = '' if prediction[0] in crop_dict else "sorry, we could not find"'

        # Redirect to result page with the prediction
        return render_template('result.html', prediction_text=result_message)

if __name__ == "__main__":
    app.run(debug=True)

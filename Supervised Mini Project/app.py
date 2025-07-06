from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    gender = int(request.form['gender'])
    race = int(request.form['race'])
    parental_education = int(request.form['parental_education'])
    lunch = int(request.form['lunch'])
    prep_course = int(request.form['prep_course'])

    # Combine into input array
    input_data = np.array([[gender, race, parental_education, lunch, prep_course]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    result = "✅ PASS" if prediction == 1 else "❌ FAIL"
    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)

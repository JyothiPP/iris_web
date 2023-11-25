import pandas as pd
import pickle
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('svm_model.pickle', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
        SL = float(request.form['SL'])
        SW= float(request.form['SW'])
        PL= float(request.form['PL'])
        PW= float(request.form['PW'])
    # Create a DataFrame with the user's input
        input_data = pd.DataFrame({
            'SL': [SL],
            'SW': [SW],
            'PL': [PL],
            'PW': [PW]
        })
        
        # Make the prediction
        prediction = model.predict(input_data)[0]

        return render_template('result.html',prediction=prediction)

    
if __name__ == '__main__':
    app.run(debug=True)
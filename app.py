# app.py
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, render_template
import warnings
warnings.filterwarnings('ignore')

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
try:
    model = joblib.load(r"C:\Users\Sanjoli\OneDrive\Desktop\Financial_Forecasting_Project\model.joblib") #Replace with your file directory
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route("/")
def home():
    """Render the home page."""
    return render_template('index.html')

@app.route('/predict')
def predict():
    """Render the prediction page."""
    return render_template('predict.html')

@app.route('/submit', methods=["POST"])
def submit():
    """Handle form submission and make predictions."""
    try:
        if model is None:
            return render_template("result.html", 
                                 result="Model not loaded. Please check the model file.",
                                 prediction="Error")
        
        # Reading the inputs given by the user
        input_features = []
        
        # Get form data in the correct order
        age = int(request.form.get('age'))
        workclass = int(request.form.get('workclass'))
        fnlwgt = int(request.form.get('fnlwgt', 0))
        education = int(request.form.get('education'))
        education_num = int(request.form.get('education_num', 0))
        marital_status = int(request.form.get('marital_status'))
        occupation = int(request.form.get('occupation'))
        relationship = int(request.form.get('relationship'))
        race = int(request.form.get('race'))
        sex = int(request.form.get('sex'))
        capital_gain = int(request.form.get('capital_gain', 0))
        capital_loss = int(request.form.get('capital_loss', 0))
        hours_per_week = int(request.form.get('hours_per_week'))
        native_country = int(request.form.get('native_country'))
        
        # Create feature array
        input_features = [age, workclass, fnlwgt, education, education_num, 
                         marital_status, occupation, relationship, race, sex,
                         capital_gain, capital_loss, hours_per_week, native_country]
        
        # Convert to numpy array and reshape for prediction
        input_array = np.array(input_features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(input_array)[0]
        
        # Determine the result message based on the prediction value
        if prediction == 1:
            result = "Your income is predicted to be more than $50,000. Yes, you are ready for investment. Invest wisely!"
            prediction_text = ">50K"
        else:
            result = "Your income is predicted to be less than $50,000. Better to invest your money to learn skills."
            prediction_text = "<=50K"
            
        return render_template("result.html", result=result, prediction=prediction_text)
        
    except Exception as e:
        # Handle exceptions and print error for debugging
        print(f"Error during prediction: {e}")
        return render_template("result.html", 
                             result="An error occurred during prediction. Please check your inputs and try again.",
                             prediction="Error")

@app.route('/about')
def about():
    """Render the about page."""
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True, port=4000, host='127.0.0.1')
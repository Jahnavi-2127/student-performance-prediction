from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the model 'brain'
# We use an absolute path check to ensure it loads correctly in VS Code
model_path = 'student_model.pkl'

if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    model = None
    print("CRITICAL ERROR: student_model.pkl not found. Please run model_train.py first!")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "Model file missing. Please train the model first."

    # 1. Extract values from the HTML form
    try:
        # Collects: hours, attendance, prev_score
        input_features = [float(x) for x in request.form.values()]
        final_features = [np.array(input_features)]
        
        # 2. Generate Prediction
        prediction = model.predict(final_features)
        score = round(prediction[0], 2)

        # 3. Logic for Risk Level (Fixed to prevent UnboundLocalError)
        # We initialize with default values just in case
        status = "Unknown"
        color = "#6c757d" # Gray
        remark = "No data available."

        if score >= 85:
            status = "Low Risk"
            color = "#28a745"  # Green
            remark = "Excellent Performance! You are well-prepared for your exams."
        elif score >= 65:
            status = "Moderate Risk"
            color = "#fd7e14"  # Orange
            remark = "Good standing. Increasing your study hours could secure an 'A' grade."
        elif score >= 40:
            status = "At Risk"
            color = "#ffc107"  # Yellow
            remark = "Average performance. Focus on consistent attendance and revision."
        else:
            status = "High Risk"
            color = "#dc3545"  # Red
            remark = "Warning: Predicted score is low. Immediate intervention and extra classes recommended."

        # 4. Send all data to the results page
        return render_template('index.html', 
                               prediction_text=score, 
                               remark=remark, 
                               color=color,
                               status=status)
    
    except Exception as e:
        return f"An error occurred during prediction: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)

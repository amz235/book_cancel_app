import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, render_template

# Initialize Flask app
app = Flask(__name__)


# Load the pre-trained pipeline (which includes both preprocessing and model)
def load_pipeline():
    with open('./predict_pipeline.pickle', 'rb') as f:
        pipeline = pickle.load(f)
    return pipeline


# Predict the cancellation outcome using the pipeline
def predict_cancellation(input_data):
    pipeline = load_pipeline()  # Load the entire pipeline (including model and preprocessor)

    # Convert input_data to a DataFrame with the correct columns (6 features)
    selected_features = ['type_of_meal_plan',
                         'room_type_reserved',
                         'market_segment_type',
                         'lead_time',
                         'avg_price_per_room',
                         'no_of_special_requests']
    input_df = pd.DataFrame([input_data], columns=selected_features)

    # Use the pipeline to make predictions
    prediction = pipeline.predict(input_df)  # Prediction as 0 or 1
    probability = pipeline.predict_proba(input_df)[:, 1]  # Probability of the positive class (1)
    return prediction[0], probability[0]


# Define the route for displaying the form and handling predictions
@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    prob = None

    if request.method == 'POST':
        # Get form data
        sample_input = {
            'type_of_meal_plan': request.form.get('type_of_meal_plan'),
            'room_type_reserved': request.form.get('room_type_reserved'),
            'market_segment_type': request.form.get('market_segment_type'),
            'lead_time': float(request.form.get('lead_time')),
            'avg_price_per_room': float(request.form.get('avg_price_per_room')),
            'no_of_special_requests': int(request.form.get('no_of_special_requests'))
        }

        # Predict the cancellation outcome
        result, prob = predict_cancellation(sample_input)

    return render_template('index.html', result=result, prob=prob)


if __name__ == '__main__':
    app.run(debug=True)
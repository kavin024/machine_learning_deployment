from flask import Flask, request, jsonify
import joblib
import pandas as pd
# Instantiate the Flask application
app = Flask(__name__)
# Load the pre-trained logistic regression model
model = joblib.load('logistic_regression_model.joblib')
print("Logistic regression model loaded successfully.")
# Retrieve and store the exact column names of the features (X.columns)
# X is already defined from previous steps
model_features = X.columns
print("Model features (column names) stored successfully.")

# Re-declare the 'categorical_features' list used during training
categorical_features = ['metadata/ministry_department', 'metadata/tags/1', 'metadata/tags/2']
print("Categorical features re-declared successfully.")


# Load the pre-trained logistic regression model
model = joblib.load('logistic_regression_model.joblib')
print("Logistic regression model loaded successfully.")

# Retrieve and store the exact column names of the features (X.columns)
# X is already defined from previous steps
model_features = X.columns
print("Model features (column names) stored successfully.")

# Re-declare the 'categorical_features' list used during training
categorical_features = ['metadata/ministry_department', 'metadata/tags/1', 'metadata/tags/2']
print("Categorical features re-declared successfully.")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No JSON data provided or JSON is malformed'}), 400

    print(f"Received data for prediction: {data}")

    # 1. Convert the received data into a pandas DataFrame.
    input_df = pd.DataFrame([data])
    # print("\nInput DataFrame created:")
    # print(input_df)

    # 2. Apply one-hot encoding to the 'categorical_features' within 'input_df'.
    # Handle cases where input_df[categorical_features] might be empty if input data lacks these keys
    input_encoded = pd.get_dummies(input_df.get(categorical_features, pd.DataFrame()), columns=categorical_features, drop_first=True)
    # print("\nOne-hot encoded input:")
    # print(input_encoded)

    # 3. Align the columns of 'input_encoded' with the 'model_features'.
    # Ensure all model_features are present, filling with 0 if not.
    input_aligned = input_encoded.reindex(columns=model_features, fill_value=0)
    # print("\nAligned input features (X for prediction):")
    # print(input_aligned)

    # 4. Make a prediction using the loaded model.
    prediction_proba = model.predict_proba(input_aligned)
    prediction = model.predict(input_aligned)

    # Convert prediction to a Python list/int for JSON serialization
    predicted_class = int(prediction[0])
    # Convert probabilities to a list for JSON serialization
    probabilities = prediction_proba[0].tolist()

    # 5. Return the prediction as a JSON response.
    return jsonify({
        'prediction': predicted_class,
        'probability_class_0': probabilities[0],
        'probability_class_1': probabilities[1]
    }), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

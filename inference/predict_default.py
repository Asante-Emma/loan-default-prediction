import joblib
import pandas as pd

# Load model, scaler, and feature columns
model = joblib.load('../models/final_random_forest_model.joblib')
encoder = joblib.load('../models/one_hot_encoder.joblib')
scaler = joblib.load('../models/standard_scaler.joblib')
feature_columns = joblib.load('../models/feature_columns.joblib')

# Define the preprocess_input function
def preprocess_input(data):
    df = pd.DataFrame([data])

    # Select categorical and numeric columns
    categorical_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
    numeric_cols = [
        'person_age', 'person_income', 'person_emp_length',
        'loan_amnt', 'loan_int_rate', 'loan_percent_income',
        'cb_person_cred_hist_length'
    ]

    # Encode categorical variables using saved encoder
    cat_encoded = encoder.transform(df[categorical_cols])
    cat_encoded_df = pd.DataFrame(cat_encoded, columns=encoder.get_feature_names_out(categorical_cols))

    # Concatenate with numeric
    numeric_df = df[numeric_cols].reset_index(drop=True)
    full_df = pd.concat([numeric_df, cat_encoded_df], axis=1)

    # Align columns with training feature columns
    for col in feature_columns:
        if col not in full_df.columns:
            full_df[col] = 0  # add missing columns as zero

    full_df = full_df[feature_columns]  # ensure correct order

    # Scale numeric features
    full_df[numeric_cols] = scaler.transform(full_df[numeric_cols])

    return full_df

# Define main prediction function
def predict_default(input_data):
    X_input = preprocess_input(input_data)
    prediction = model.predict(X_input)[0]
    probability = model.predict_proba(X_input)[0][1]

    return {
        "prediction": "Default" if prediction == 1 else "Non-Default",
        "probability_of_default": round(probability * 100, 2)
    }

if __name__ == "__main__":
    input_data = {
        'person_age': 35,
        'person_income': 60000,
        'person_emp_length': 5,
        'loan_amnt': 15000,
        'loan_int_rate': 10.5,
        'loan_percent_income': 0.25,
        'cb_person_cred_hist_length': 4,
        'person_home_ownership': 'RENT',
        'loan_intent': 'PERSONAL',
        'loan_grade': 'C',
        'cb_person_default_on_file': 'N'
    }

    result = predict_default(input_data)
    print(f"Prediction: {result['prediction']}")
    print(f"Probability of Default: {result['probability_of_default']}%")
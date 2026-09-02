import joblib as jb
import pandas as pd

# Load the saved model, scaler, and encoder
saved_model = jb.load('churn_model.joblib')

# Take input from the user for each required feature
input_values = {}
for feature in saved_model['input_fields']:
    value = input(f'Enter the {feature.lower()}: ')
    input_values.update({feature: value})

# Convert numeric fields to float
for field in saved_model['numeric_fields']:
    input_values[field] = float(input_values[field])

# Create a single-row DataFrame from the input
input_record = pd.DataFrame(input_values, index=[0])

# Scale numeric fields
input_record[saved_model['numeric_fields']] = saved_model['scaler'].transform(
    input_record[saved_model['numeric_fields']]
)

# Encode categorical fields
input_record.loc[:, saved_model['encoder'].get_feature_names_out()] = saved_model['encoder'].transform(
    input_record[saved_model['categorical_fields']]
).toarray()

# Make prediction
if saved_model['model'].predict(input_record[saved_model['model_inputs']]) == [1]:
    prediction = 'churn'
else:
    prediction = 'wont_churn'

print(f'Predicted status: {prediction}')

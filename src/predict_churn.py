import pandas as pd
import joblib

# 1. Load the saved machine-learning model
model = joblib.load("customer_churn_model.joblib")

# 2. Load the test dataset
test = pd.read_csv("data/test.csv")

# 3. Preserve customer IDs
customer_ids = test["CustomerID"]

# 4. Remove CustomerID from the model inputs
X_test = test.drop(columns=["CustomerID"])

# 5. Predict churn probabilities
churn_probabilities = model.predict_proba(X_test)[:, 1]

# 6. Apply our selected threshold
threshold = 0.60

predictions = (churn_probabilities >= threshold).astype(int)

# 7. Create a results table
results = pd.DataFrame({
    "CustomerID": customer_ids,
    "ChurnProbability": churn_probabilities,
    "PredictedChurn": predictions
})

# 8. Save predictions to a CSV file
results.to_csv("churn_predictions.csv", index=False)

# 9. Display a summary
print("Predictions completed successfully!")

print("\nTotal customers:", len(results))

print("\nPredicted churn outcomes:")
print(results["PredictedChurn"].value_counts())

print("\nFirst five predictions:")
print(results.head())

print("\nSaved results to churn_predictions.csv")


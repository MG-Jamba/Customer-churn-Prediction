import pandas as pd

# Load the predictions
results = pd.read_csv("churn_predictions.csv")

# Categorise customers by churn probability
def classify_risk(probability):
    if probability >= 0.70:
        return "High Risk"
    elif probability >= 0.40:
        return "Medium Risk"
    else:
        return "Low Risk"


# Create a new risk category column
results["RiskCategory"] = results["ChurnProbability"].apply(
    classify_risk
)

# Count customers in each risk category
risk_summary = results["RiskCategory"].value_counts()

print("CUSTOMER CHURN RISK SUMMARY")
print("---------------------------")
print(risk_summary)

# Calculate percentages
risk_percentages = (
    results["RiskCategory"].value_counts(normalize=True) * 100
)

print("\nRISK CATEGORY PERCENTAGES")
print("-------------------------")
print(risk_percentages.round(2))

# Save the results
results.to_csv("customer_risk_summary.csv", index=False)

print("\nRisk summary saved successfully!")
print("File: customer_risk_summary.csv")
# Customer Churn Prediction and Risk Analytics

## Project Overview

This project uses machine learning to identify customers who may be at risk of leaving a business. It combines Python, machine learning, and Microsoft Power BI to transform customer data into actionable retention insights.

The goal is to help businesses identify potentially at-risk customers and prioritise customer-retention initiatives.

## Business Problem

Customer churn can reduce revenue, weaken customer relationships, and increase the cost of acquiring new customers.

Businesses need a data-driven way to identify customers who may be likely to leave so that they can investigate potential causes and develop targeted retention strategies.

This project demonstrates how predictive analytics can support that process.

## Tools and Technologies

- **Python:** Data processing and model development.
- **Pandas:** Data manipulation and analysis.
- **Scikit-learn:** Machine learning pipeline and model evaluation.
- **Logistic Regression:** Customer churn classification.
- **Matplotlib:** Visualisation of model threshold comparisons.
- **Power BI:** Interactive dashboard and customer risk analysis.
- **VS Code:** Development environment.

## Project Workflow

1. Inspected and explored the customer dataset.
2. Examined the distribution of the churn target variable.
3. Prepared numerical and categorical features for machine learning.
4. Built a preprocessing and classification pipeline.
5. Trained a Logistic Regression model with class balancing.
6. Evaluated the model using a validation dataset.
7. Compared different classification thresholds.
8. Trained and saved the final model.
9. Generated churn probabilities and predictions for 104,480 customers.
10. Grouped customers into low-, medium-, and high-risk categories.
11. Developed a Power BI dashboard to communicate the findings.

## Machine Learning Approach

The model uses Logistic Regression with class balancing to address the imbalance between customers who churn and those who do not.

The preprocessing pipeline includes:

- Median imputation and standardisation for numerical features.
- Most-frequent-value imputation and one-hot encoding for categorical features.
- Exclusion of CustomerID from the model's input features.

The classification threshold was compared at 0.30, 0.40, 0.50, 0.60, and 0.70.

A threshold of 0.60 was selected provisionally because it produced the highest F1 score among the tested thresholds. The best threshold for a real business would also depend on retention campaign costs and the consequences of missed churn cases.

## Customer Risk Segmentation

The model's predicted churn probabilities were grouped into three project-defined categories:

- **High Risk:** Probability greater than or equal to 0.70.
- **Medium Risk:** Probability greater than or equal to 0.40 and below 0.70.
- **Low Risk:** Probability below 0.40.

### Prediction Results

| Risk Category |   Customers |  Percentage |
| ------------- | ----------: | ----------: |
| Low Risk      |      47,979 |      45.92% |
| Medium Risk   |      42,108 |      40.30% |
| High Risk     |      14,393 |      13.78% |
| **Total**     | **104,480** | **100.00%** |

These figures represent model-generated predictions, not confirmed future churn outcomes.

## Power BI Dashboard

The dashboard presents customer risk information through:

- Total customers analysed.
- High-risk customer count.
- High-risk customer percentage.
- Customer risk distribution.
- Customer-level prediction details.
- An interactive risk-category slicer.
- Business insights to support retention planning.

## Power BI Dashboard

![Customer Churn Prediction Dashboard](screenshots/customer-churn-dashboard.png)

## Key Business Insight

The model classified 14,393 customers, or 13.78% of the analysed test population, as high risk. A further 42,108 customers, or 40.30%, were classified as medium risk.

These groups could be prioritised for further investigation and appropriately designed retention campaigns. Before using the predictions for operational decisions, the model should be evaluated against actual churn outcomes.

## Model Evaluation and Limitations

On a stratified validation split, the model achieved the following results at a classification threshold of 0.60:

- **Accuracy:** Approximately 76%.
- **Precision for churn:** Approximately 38%.
- **Recall for churn:** Approximately 54%.
- **F1 score for churn:** Approximately 45%.

The model's performance demonstrates the trade-off between identifying potential churners and avoiding false alarms.

The test dataset does not contain actual churn labels. Therefore, the generated test predictions cannot be treated as confirmed outcomes, and the validation results should not be confused with performance on the test dataset.

Further work could include testing additional models, tuning the classification threshold using business costs, and evaluating predictions once actual churn outcomes become available.

## Repository Structure

```text
customer-churn-prediction/
├── data/
├── src/
├── customer_churn_model.joblib
├── churn_predictions.csv
├── customer_risk_summary.csv
├── threshold_comparison.png
├── Customer_Churn_Prediction_Dashboard.pbix
└── README.md
```

## Conclusion

This project demonstrates an end-to-end predictive analytics workflow, from customer data preparation and machine learning to customer risk segmentation and interactive business reporting.

It showcases practical skills in Python, supervised machine learning, model evaluation, data analysis, and Power BI dashboard development.

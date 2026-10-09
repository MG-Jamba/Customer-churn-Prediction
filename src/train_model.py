import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# 1. Load the training dataset
train = pd.read_csv("data/train.csv")

# 2. Separate the target from the input features
X = train.drop(columns=["Churn", "CustomerID"])
y = train["Churn"]

# 3. Identify numerical and categorical columns
numerical_columns = X.select_dtypes(include=["number"]).columns.tolist()
categorical_columns = X.select_dtypes(exclude=["number"]).columns.tolist()

print("Numerical columns:", len(numerical_columns))
print("Categorical columns:", len(categorical_columns))

# 4. Prepare numerical data
numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# 5. Prepare categorical data
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# 6. Apply the appropriate preparation to each column type
preprocessor = ColumnTransformer([
    ("numerical", numerical_pipeline, numerical_columns),
    ("categorical", categorical_pipeline, categorical_columns)
])

# 7. Split the data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining rows:", len(X_train))
print("Validation rows:", len(X_valid))

# 8. Build the machine-learning pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
))
])

# 9. Train the model
print("\nTraining the model...")
model.fit(X_train, y_train)

# 10. Calculate churn probabilities
churn_probabilities = model.predict_proba(X_valid)[:, 1]

# 11. Evaluate different decision thresholds
for threshold in [0.30, 0.40, 0.50, 0.60, 0.70]:

    predictions = (churn_probabilities >= threshold).astype(int)

    print(f"\n{'=' * 40}")
    print(f"THRESHOLD: {threshold:.2f}")
    print("=" * 40)

    print("\nCONFUSION MATRIX")
    print(confusion_matrix(y_valid, predictions))

    print("\nCLASSIFICATION REPORT")
    print(
        classification_report(
            y_valid,
            predictions,
            zero_division=0
        )
    )



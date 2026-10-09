import pandas as pd

# Load the training and test datasets
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# Display the dataset sizes
print("TRAINING DATA")
print("Shape:", train.shape)

print("\nTEST DATA")
print("Shape:", test.shape)

# Display column names
print("\nTRAINING COLUMNS:")
print(train.columns.tolist())

print("\nTEST COLUMNS:")
print(test.columns.tolist())

# Display the first five rows
print("\nFIRST FIVE TRAINING ROWS:")
print(train.head())

# Check for missing values
print("\nMISSING VALUES IN TRAINING DATA:")
print(train.isnull().sum())

# Check the distribution of churn
print("\nCHURN DISTRIBUTION:")
print(train["Churn"].value_counts())

# Check churn percentages
print("\nCHURN PERCENTAGES:")
print(train["Churn"].value_counts(normalize=True) * 100)

# Check data types
print("\nDATA TYPES:")
print(train.dtypes)
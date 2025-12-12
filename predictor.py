
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Load the dataset
file_path = "insurance_data_with_features.xlsx"
df = pd.read_excel(file_path, sheet_name="Sheet1", engine="openpyxl")

# Target variable
target = "Converted"

# Drop identifier and date columns that are not useful for prediction
id_cols = ["Proposal Number", "Application Number", "Policy Number", "Agent Code"]
date_cols = ["Proposal Created Date", "Application Created Date", "Application Submit Date"]
drop_cols = id_cols + date_cols

# Separate features and target
X = df.drop(columns=drop_cols + [target])
y = df[target]

# Handle missing values:
# - For numeric columns, fill missing with median
# - For categorical columns, fill missing with mode
for col in X.columns:
    if X[col].dtype in ["int64", "float64"]:
        X[col] = X[col].fillna(X[col].median())
    else:
        X[col] = X[col].fillna(X[col].mode()[0])

# Identify categorical and numeric columns
categorical_cols = [col for col in X.columns if X[col].dtype == "object"]
numeric_cols = [col for col in X.columns if X[col].dtype in ["int64", "float64"]]

# Preprocessing pipeline:
# - One-hot encode categorical variables
# - Standard scale numeric variables
categorical_transformer = OneHotEncoder(handle_unknown="ignore")
numeric_transformer = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)

# Split data into training and test sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Fit the preprocessor on training data and transform both train and test sets
pipeline = Pipeline(steps=[("preprocessor", preprocessor)])
pipeline.fit(X_train)

X_train_processed = pipeline.transform(X_train)
X_test_processed = pipeline.transform(X_test)

print("Training data shape after preprocessing:", X_train_processed.shape)
print("Test data shape after preprocessing:", X_test_processed

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import os

# Load the original dataset
data_path = os.path.join(os.path.dirname(__file__), 'Datasets.csv')
df = pd.read_csv(data_path)
df['results'] = df['Label'].astype(int)

# Define features
feature_cols = [
    'Age', 'Followers', 'NAME_CONTRACT_TYPE', 'GENDER',
    'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
    'NAME_INCOME_TYPE', 'NAME_FAMILY_STATUS'
]

# Prepare training data
X = df[feature_cols]
y = df['results']

# Define preprocessing
categorical_cols = ['NAME_CONTRACT_TYPE', 'GENDER', 'NAME_INCOME_TYPE', 'NAME_FAMILY_STATUS']
numeric_cols = ['Age', 'Followers', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE']

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first')),
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols),
    ],
    remainder='drop'
)

# Train models
models = [
    ('random_forest', Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42))
    ])),
    ('gradient_boosting', Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=3, random_state=42))
    ])),
    ('logistic', Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
    ])),
]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)

# Train all models and find best one
best_model = None
best_score = 0.0
model_scores = {}

for name, model in models:
    model.fit(X_train, y_train)
    # Transform test data and score
    X_test_transformed = model.named_steps['preprocessor'].transform(X_test)
    score = model.named_steps['classifier'].score(X_test_transformed, y_test)
    model_scores[name] = score
    if score > best_score:
        best_score = score
        best_model = model

print(f"Best model: {max(model_scores, key=model_scores.get)} with accuracy: {best_score:.4f}")

# Generate test dataset with diverse scenarios
test_data = []

# Scenario 1: High-risk profiles (young, high income, cash loans)
test_data.extend([
    {'Age': 25, 'Followers': 500, 'NAME_CONTRACT_TYPE': 'Cash loans', 'GENDER': 'M',
     'AMT_INCOME_TOTAL': 300000, 'AMT_CREDIT': 1000000, 'AMT_ANNUITY': 50000, 'AMT_GOODS_PRICE': 900000,
     'NAME_INCOME_TYPE': 'Working', 'NAME_FAMILY_STATUS': 'Single / not married', 'Expected_Risk': 'High'},
    {'Age': 28, 'Followers': 800, 'NAME_CONTRACT_TYPE': 'Cash loans', 'GENDER': 'F',
     'AMT_INCOME_TOTAL': 250000, 'AMT_CREDIT': 800000, 'AMT_ANNUITY': 40000, 'AMT_GOODS_PRICE': 750000,
     'NAME_INCOME_TYPE': 'Commercial associate', 'NAME_FAMILY_STATUS': 'Married', 'Expected_Risk': 'High'},
])

# Scenario 2: Low-risk profiles (older, stable income, revolving loans)
test_data.extend([
    {'Age': 55, 'Followers': 2000, 'NAME_CONTRACT_TYPE': 'Revolving loans', 'GENDER': 'M',
     'AMT_INCOME_TOTAL': 150000, 'AMT_CREDIT': 200000, 'AMT_ANNUITY': 10000, 'AMT_GOODS_PRICE': 180000,
     'NAME_INCOME_TYPE': 'State servant', 'NAME_FAMILY_STATUS': 'Married', 'Expected_Risk': 'Low'},
    {'Age': 60, 'Followers': 1500, 'NAME_CONTRACT_TYPE': 'Revolving loans', 'GENDER': 'F',
     'AMT_INCOME_TOTAL': 120000, 'AMT_CREDIT': 150000, 'AMT_ANNUITY': 8000, 'AMT_GOODS_PRICE': 140000,
     'NAME_INCOME_TYPE': 'Pensioner', 'NAME_FAMILY_STATUS': 'Widow', 'Expected_Risk': 'Low'},
])

# Scenario 3: Medium-risk profiles (middle-aged, moderate income)
test_data.extend([
    {'Age': 40, 'Followers': 1200, 'NAME_CONTRACT_TYPE': 'Cash loans', 'GENDER': 'M',
     'AMT_INCOME_TOTAL': 180000, 'AMT_CREDIT': 400000, 'AMT_ANNUITY': 20000, 'AMT_GOODS_PRICE': 360000,
     'NAME_INCOME_TYPE': 'Working', 'NAME_FAMILY_STATUS': 'Married', 'Expected_Risk': 'Medium'},
    {'Age': 45, 'Followers': 900, 'NAME_CONTRACT_TYPE': 'Cash loans', 'GENDER': 'F',
     'AMT_INCOME_TOTAL': 160000, 'AMT_CREDIT': 350000, 'AMT_ANNUITY': 18000, 'AMT_GOODS_PRICE': 320000,
     'NAME_INCOME_TYPE': 'Working', 'NAME_FAMILY_STATUS': 'Civil marriage', 'Expected_Risk': 'Medium'},
])

# Scenario 4: Edge cases (very young, very old, extreme values)
test_data.extend([
    {'Age': 21, 'Followers': 100, 'NAME_CONTRACT_TYPE': 'Cash loans', 'GENDER': 'M',
     'AMT_INCOME_TOTAL': 50000, 'AMT_CREDIT': 200000, 'AMT_ANNUITY': 15000, 'AMT_GOODS_PRICE': 180000,
     'NAME_INCOME_TYPE': 'Working', 'NAME_FAMILY_STATUS': 'Single / not married', 'Expected_Risk': 'High'},
    {'Age': 70, 'Followers': 3000, 'NAME_CONTRACT_TYPE': 'Revolving loans', 'GENDER': 'F',
     'AMT_INCOME_TOTAL': 80000, 'AMT_CREDIT': 100000, 'AMT_ANNUITY': 5000, 'AMT_GOODS_PRICE': 90000,
     'NAME_INCOME_TYPE': 'Pensioner', 'NAME_FAMILY_STATUS': 'Widow', 'Expected_Risk': 'Low'},
])

# Scenario 5: Similar profiles with different outcomes (to test consistency)
test_data.extend([
    {'Age': 35, 'Followers': 1000, 'NAME_CONTRACT_TYPE': 'Cash loans', 'GENDER': 'M',
     'AMT_INCOME_TOTAL': 200000, 'AMT_CREDIT': 500000, 'AMT_ANNUITY': 25000, 'AMT_GOODS_PRICE': 450000,
     'NAME_INCOME_TYPE': 'Working', 'NAME_FAMILY_STATUS': 'Married', 'Expected_Risk': 'Medium'},
    {'Age': 35, 'Followers': 1000, 'NAME_CONTRACT_TYPE': 'Cash loans', 'GENDER': 'M',
     'AMT_INCOME_TOTAL': 200000, 'AMT_CREDIT': 500000, 'AMT_ANNUITY': 25000, 'AMT_GOODS_PRICE': 450000,
     'NAME_INCOME_TYPE': 'Working', 'NAME_FAMILY_STATUS': 'Married', 'Expected_Risk': 'Medium'},
])

# Convert to DataFrame
test_df = pd.DataFrame(test_data)

# Make predictions with all models
predictions = {}
for name, model in models:
    preds = model.predict(test_df[feature_cols])
    predictions[name] = preds

# Add predictions to test dataframe
test_df['Random_Forest_Prediction'] = predictions['random_forest']
test_df['Gradient_Boosting_Prediction'] = predictions['gradient_boosting']
test_df['Logistic_Regression_Prediction'] = predictions['logistic']

# Calculate consensus prediction (majority vote)
def majority_vote(row):
    votes = [row['Random_Forest_Prediction'], row['Gradient_Boosting_Prediction'], row['Logistic_Regression_Prediction']]
    return 1 if sum(votes) >= 2 else 0

test_df['Consensus_Prediction'] = test_df.apply(majority_vote, axis=1)

# Add prediction confidence (agreement level)
def prediction_confidence(row):
    votes = [row['Random_Forest_Prediction'], row['Gradient_Boosting_Prediction'], row['Logistic_Regression_Prediction']]
    agreement = sum(votes)
    if agreement == 0 or agreement == 3:
        return 'High'
    else:
        return 'Low'

test_df['Prediction_Confidence'] = test_df.apply(prediction_confidence, axis=1)

# Add analysis columns
def analyze_prediction(row):
    consensus = row['Consensus_Prediction']
    expected = row['Expected_Risk']

    if expected == 'High' and consensus == 1:
        return 'Correct - High risk detected'
    elif expected == 'High' and consensus == 0:
        return 'False Negative - High risk missed'
    elif expected == 'Low' and consensus == 0:
        return 'Correct - Low risk cleared'
    elif expected == 'Low' and consensus == 1:
        return 'False Positive - Low risk flagged'
    elif expected == 'Medium':
        return 'Medium risk - requires review'
    else:
        return 'Uncertain'

test_df['Analysis'] = test_df.apply(analyze_prediction, axis=1)

# Save to CSV
output_path = os.path.join(os.path.dirname(__file__), 'Test_Dataset_With_Predictions.csv')
test_df.to_csv(output_path, index=False)

print(f"Test dataset with predictions saved to: {output_path}")
print(f"Dataset contains {len(test_df)} test cases")
print("\nModel accuracies on training data:")
for name, score in model_scores.items():
    print(f"{name}: {score:.4f}")

print("\nTest dataset summary:")
print(test_df.groupby(['Expected_Risk', 'Consensus_Prediction']).size().unstack(fill_value=0))
print("\nPrediction confidence distribution:")
print(test_df['Prediction_Confidence'].value_counts())
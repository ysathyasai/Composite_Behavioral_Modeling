import pandas as pd
import os

# Load the test dataset
data_path = os.path.join(os.path.dirname(__file__), 'Test_Dataset_With_Predictions.csv')
df = pd.read_csv(data_path)

print("=== MODEL ERROR ANALYSIS REPORT ===\n")

print("1. OVERALL MODEL PERFORMANCE:")
print("-" * 40)
total_cases = len(df)
consensus_correct = 0
false_positives = 0
false_negatives = 0

for _, row in df.iterrows():
    analysis = row['Analysis']
    if 'Correct' in analysis:
        consensus_correct += 1
    elif 'False Positive' in analysis:
        false_positives += 1
    elif 'False Negative' in analysis:
        false_negatives += 1

accuracy = consensus_correct / total_cases * 100
print(f"Total test cases: {total_cases}")
print(f"Consensus accuracy: {accuracy:.1f}%")
print(f"False positives: {false_positives}")
print(f"False negatives: {false_negatives}")
print()

print("2. PERFORMANCE BY RISK CATEGORY:")
print("-" * 40)
risk_performance = df.groupby('Expected_Risk').agg({
    'Consensus_Prediction': lambda x: (x == 1).sum() if 'High' in x.name else (x == 0).sum(),
    'Expected_Risk': 'count'
}).rename(columns={'Consensus_Prediction': 'Correct_Predictions', 'Expected_Risk': 'Total'})

for risk_level in ['High', 'Medium', 'Low']:
    if risk_level in risk_performance.index:
        correct = risk_performance.loc[risk_level, 'Correct_Predictions']
        total = risk_performance.loc[risk_level, 'Total']
        percentage = correct / total * 100 if total > 0 else 0
        print(f"{risk_level} Risk: {correct}/{total} correct ({percentage:.1f}%)")

print()

print("3. MODEL AGREEMENT ANALYSIS:")
print("-" * 40)
confidence_dist = df['Prediction_Confidence'].value_counts()
for confidence, count in confidence_dist.items():
    percentage = count / total_cases * 100
    print(f"{confidence} confidence predictions: {count} ({percentage:.1f}%)")

print()

print("4. DETAILED ERROR CASES:")
print("-" * 40)
error_cases = df[~df['Analysis'].str.contains('Correct')]
if len(error_cases) > 0:
    for idx, row in error_cases.iterrows():
        print(f"Case {idx+1}: {row['Analysis']}")
        print(f"  Profile: {row['Age']}yo, {row['GENDER']}, {row['NAME_INCOME_TYPE']}, {row['AMT_INCOME_TOTAL']/1000:.0f}k income")
        print(f"  Predictions: RF={row['Random_Forest_Prediction']}, GB={row['Gradient_Boosting_Prediction']}, LR={row['Logistic_Regression_Prediction']}")
        print()
else:
    print("No error cases found in test dataset!")

print("5. RECOMMENDATIONS:")
print("-" * 40)
if false_negatives > false_positives:
    print("- Model is missing high-risk cases (false negatives)")
    print("- Consider adjusting classification threshold or feature weights")
elif false_positives > false_negatives:
    print("- Model is overly cautious (false positives)")
    print("- Consider relaxing classification criteria")
else:
    print("- Balanced error distribution")

if confidence_dist.get('Low', 0) > confidence_dist.get('High', 0):
    print("- Models show disagreement on many predictions")
    print("- Consider ensemble methods or additional features")

print("- Review edge cases (very young/old applicants)")
print("- Validate with larger, more diverse test set")
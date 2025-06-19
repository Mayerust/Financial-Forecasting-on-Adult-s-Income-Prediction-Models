
# model_training.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

# Activity 1.1: Load and Explore Dataset
print("Loading Adult Census Income Dataset...")
df = pd.read_csv('adult.csv')

print(f"Dataset Shape: {df.shape}")
print("\nDataset Info:")
print(df.info())

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Description:")
print(df.describe())

# Activity 2.1: Handle Missing Values
print("\nChecking for missing values:")
print(df.isnull().sum())

# Check for '?' values which represent missing data
print("\nChecking for '?' values:")
for col in df.columns:
    if df[col].dtype == 'object':
        print(f"{col}: {(df[col] == '?').sum()}")

# Replace '?' with 'Unknown' in workclass and occupation
df['workclass'] = df['workclass'].replace('?', 'Unknown')
df['occupation'] = df['occupation'].replace('?', 'Unknown')
df['native.country'] = df['native.country'].replace('?', 'Unknown')

# Activity 2.2: Handle Categorical Values
print("\nCategorical columns analysis:")
categorical_cols = ['workclass', 'education', 'marital.status', 'occupation', 
                   'relationship', 'race', 'sex', 'native.country', 'income']

for col in categorical_cols:
    print(f"\n{col} value counts:")
    print(df[col].value_counts())

# Label Encoding
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])
    
print("\nAfter Label Encoding:")
print(df.head())

# Activity 2.3: Outlier Detection and Treatment
plt.figure(figsize=(15, 10))
numeric_cols = ['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']

for i, col in enumerate(numeric_cols, 1):
    plt.subplot(2, 3, i)
    plt.boxplot(df[col])
    plt.title(f'Boxplot of {col}')
    plt.ylabel(col)

plt.tight_layout()
plt.savefig('outlier_analysis.png')
plt.show()

print("Outlier analysis completed. Age outliers are considered normal for adult income prediction.")

# Exploratory Data Analysis (EDA)
print("\n=== EXPLORATORY DATA ANALYSIS ===")

# Univariate Analysis
plt.figure(figsize=(20, 15))

# Age distribution
plt.subplot(3, 3, 1)
plt.hist(df['age'], bins=30, edgecolor='black', alpha=0.7)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')

# Income distribution
plt.subplot(3, 3, 2)
income_counts = df['income'].value_counts()
plt.pie(income_counts.values, labels=['<=50K', '>50K'], autopct='%1.1f%%')
plt.title('Income Distribution')

# Education distribution
plt.subplot(3, 3, 3)
education_counts = df['education'].value_counts()
plt.bar(range(len(education_counts)), education_counts.values)
plt.title('Education Level Distribution')
plt.xlabel('Education Level (Encoded)')
plt.ylabel('Count')

# Hours per week distribution
plt.subplot(3, 3, 4)
plt.hist(df['hours.per.week'], bins=30, edgecolor='black', alpha=0.7)
plt.title('Hours per Week Distribution')
plt.xlabel('Hours per Week')
plt.ylabel('Frequency')

# Bivariate Analysis
# Gender vs Income
plt.subplot(3, 3, 5)
gender_income = pd.crosstab(df['sex'], df['income'])
gender_income.plot(kind='bar', ax=plt.gca())
plt.title('Gender vs Income')
plt.xlabel('Gender (0=Female, 1=Male)')
plt.ylabel('Count')

# Marital Status vs Income
plt.subplot(3, 3, 6)
marital_income = pd.crosstab(df['marital.status'], df['income'])
marital_income.plot(kind='bar', ax=plt.gca())
plt.title('Marital Status vs Income')
plt.xlabel('Marital Status (Encoded)')
plt.ylabel('Count')

# Correlation Heatmap
plt.subplot(3, 3, 7)
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Heatmap')

plt.tight_layout()
plt.savefig('eda_analysis.png')
plt.show()

# Data Balancing with SMOTE
print("\n=== DATA BALANCING ===")
print("Original class distribution:")
print(df['income'].value_counts())

# Prepare features and target
X = df.drop('income', axis=1)
y = df['income']

# Apply SMOTE for balancing
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)

print("After SMOTE balancing:")
print(pd.Series(y_balanced).value_counts())

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, test_size=0.2, random_state=100, stratify=y_balanced
)

print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# Model Building and Evaluation
print("\n=== MODEL BUILDING AND EVALUATION ===")

models = {}
results = {}

# 1. Logistic Regression
print("\n1. Training Logistic Regression...")
model1 = LogisticRegression(random_state=42, max_iter=1000)
model1.fit(X_train, y_train)

y_pred_train1 = model1.predict(X_train)
y_pred_test1 = model1.predict(X_test)

train_acc1 = accuracy_score(y_train, y_pred_train1)
test_acc1 = accuracy_score(y_test, y_pred_test1)
cv_score1 = cross_val_score(model1, X_balanced, y_balanced, cv=5).mean()

models['Logistic Regression'] = model1
results['Logistic Regression'] = {
    'train_accuracy': train_acc1,
    'test_accuracy': test_acc1,
    'cv_score': cv_score1
}

print(f"Train Accuracy: {train_acc1:.4f}")
print(f"Test Accuracy: {test_acc1:.4f}")
print(f"Cross Validation Score: {cv_score1:.4f}")

# 2. Decision Tree Classifier
print("\n2. Training Decision Tree Classifier...")
model2 = DecisionTreeClassifier(criterion="gini", max_depth=5, splitter="best", random_state=42)
model2.fit(X_train, y_train)

y_pred_train2 = model2.predict(X_train)
y_pred_test2 = model2.predict(X_test)

train_acc2 = accuracy_score(y_train, y_pred_train2)
test_acc2 = accuracy_score(y_test, y_pred_test2)
cv_score2 = cross_val_score(model2, X_balanced, y_balanced, cv=5).mean()

models['Decision Tree'] = model2
results['Decision Tree'] = {
    'train_accuracy': train_acc2,
    'test_accuracy': test_acc2,
    'cv_score': cv_score2
}

print(f"Train Accuracy: {train_acc2:.4f}")
print(f"Test Accuracy: {test_acc2:.4f}")
print(f"Cross Validation Score: {cv_score2:.4f}")

# Calculate additional metrics for Decision Tree
precision2 = precision_score(y_test, y_pred_test2)
recall2 = recall_score(y_test, y_pred_test2)
f1_2 = f1_score(y_test, y_pred_test2)

print(f"Precision: {precision2:.4f}")
print(f"Recall: {recall2:.4f}")
print(f"F1 Score: {f1_2:.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_test2))

# 3. Hyperparameter Tuning for Decision Tree
print("\n3. Hyperparameter Tuning for Decision Tree...")
param_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": list(range(3, 8)),
    "splitter": ["best", "random"]
}

grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

best_dt = grid_search.best_estimator_
models['Best Decision Tree'] = best_dt

# 4. Random Forest Classifier
print("\n4. Training Random Forest Classifier...")
model3 = RandomForestClassifier(criterion="gini", max_depth=7, n_estimators=100, random_state=42)
model3.fit(X_train, y_train)

y_pred_train3 = model3.predict(X_train)
y_pred_test3 = model3.predict(X_test)

train_acc3 = accuracy_score(y_train, y_pred_train3)
test_acc3 = accuracy_score(y_test, y_pred_test3)
cv_score3 = cross_val_score(model3, X_balanced, y_balanced, cv=5).mean()

models['Random Forest'] = model3
results['Random Forest'] = {
    'train_accuracy': train_acc3,
    'test_accuracy': test_acc3,
    'cv_score': cv_score3
}
print(f"Train Accuracy: {train_acc3:.4f}")
print(f"Test Accuracy: {test_acc3:.4f}")
print(f"Cross Validation Score: {cv_score3:.4f}")

# 5. AdaBoost Classifier
print("\n5. Training AdaBoost Classifier...")
model4 = AdaBoostClassifier(n_estimators=50, learning_rate=1.0, random_state=42)
model4.fit(X_train, y_train)

y_pred_train4 = model4.predict(X_train)
y_pred_test4 = model4.predict(X_test)

train_acc4 = accuracy_score(y_train, y_pred_train4)
test_acc4 = accuracy_score(y_test, y_pred_test4)
cv_score4 = cross_val_score(model4, X_balanced, y_balanced, cv=5).mean()

models['AdaBoost'] = model4
results['AdaBoost'] = {
    'train_accuracy': train_acc4,
    'test_accuracy': test_acc4,
    'cv_score': cv_score4
}

print(f"Train Accuracy: {train_acc4:.4f}")
print(f"Test Accuracy: {test_acc4:.4f}")
print(f"Cross Validation Score: {cv_score4:.4f}")

# Model Comparison and Selection
print("\n=== MODEL COMPARISON ===")
comparison_df = pd.DataFrame(results).T
print(comparison_df)

# Select best model based on test accuracy and cross-validation score
best_model_name = comparison_df['test_accuracy'].idxmax()
best_model = models[best_model_name]

print(f"\nBest Model: {best_model_name}")
print(f"Test Accuracy: {comparison_df.loc[best_model_name, 'test_accuracy']:.4f}")

# Save the best model
joblib.dump(best_model, 'model.joblib')
print(f"\nBest model ({best_model_name}) saved as 'model.joblib'")

# Test predictions
print("\n=== TESTING PREDICTIONS ===")

# Get the feature columns from your training data
feature_columns = X_train.columns.tolist()

# Example: 14 features per sample (replace with your actual values)
test_samples = [
    [40, 4, 11, 2, 6, 0, 4, 1, 0, 0, 0, 0, 40, 39],      # First sample, 14 values
    [74, 6, 10, 4, 9, 2, 4, 0, 0, 0, 3683, 0, 20, 39],   # Second sample, 14 values
    [90, 8, 11, 6, 14, 1, 4, 0, 0, 0, 4356, 0, 40, 39]   # Third sample, 14 values
]

for i, sample in enumerate(test_samples):
    # Create a DataFrame with the correct column names
    sample_df = pd.DataFrame([sample], columns=feature_columns)
    prediction = best_model.predict(sample_df)[0]
    result = ">50K" if prediction == 1 else "<=50K"
    print(f"Sample {i+1} prediction: {result}")
    
print("\nModel training completed successfully!")
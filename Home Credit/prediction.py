import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LinearRegression

# Step 1: Load the Training Dataset
file_path = 'train_base.csv'
base_data = pd.read_csv(file_path)

# Prepare the base_data by dropping 'case_id', converting 'date_decision' to datetime, and extracting 'day_of_week'
base_data.drop(['case_id'], axis=1, inplace=True)
base_data['date_decision'] = pd.to_datetime(base_data['date_decision'])
base_data['day_of_week'] = base_data['date_decision'].dt.dayofweek
base_data.drop(['date_decision'], axis=1, inplace=True)

# Define target variable 'y' and features 'X'
y = base_data['target']
X = base_data.drop(['target'], axis=1)

# Step 2: Create Preprocessing Pipelines
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object', 'bool']).columns

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)])

# Step 3: Define the Model Pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(solver='liblinear', random_state=42))])

# Split data, train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# Prediction on the test split (for demonstration, replace with actual test data loading for submission)
test_predictions = model.predict_proba(X_test)[:, 1]

# Load the Test Dataset for Submission
file_path = 'test_base.csv'
test_data = pd.read_csv(file_path)

# Assuming 'case_id' needs to be retained for submission
case_ids = test_data['case_id']

# Prepare the test_data in the same way as the training data
# NOTE: Replace 'X_test' with 'test_data' in your actual submission workflow
test_data_processed = test_data.drop(['case_id'], axis=1)  # Adjust based on your actual test data

# Directly use the trained model to predict the test dataset
# Ensure the test data is processed identically to the training data before this step
test_predictions = model.predict_proba(test_data_processed)[:, 1]

# Create a submission DataFrame
submission = pd.DataFrame({
    'case_id': case_ids,
    'score': test_predictions
})

# Save the DataFrame to a CSV file, without the index
submission_file_path = 'submission.csv'
submission.to_csv(submission_file_path, index=False)

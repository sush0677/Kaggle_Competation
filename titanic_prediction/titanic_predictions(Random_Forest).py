import pandas as pd

from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

# Load the training data
file_path = 'train.csv'
train_data = pd.read_csv(file_path)

# Data Preprocessing
# Handling missing values
imputer = SimpleImputer(strategy='mean')
train_data['Age'] = imputer.fit_transform(train_data[['Age']])
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)

# Converting categorical variables
label_enc_sex = LabelEncoder()
train_data['Sex'] = label_enc_sex.fit_transform(train_data['Sex'])

label_enc_embarked = LabelEncoder()
train_data['Embarked'] = label_enc_embarked.fit_transform(train_data['Embarked'])

# Dropping less useful columns
train_data.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1, inplace=True)

# Creating a new feature for family size
train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch']

# Splitting the dataset into features (X) and target variable (y)
X = train_data.drop('Survived', axis=1)
y = train_data['Survived']

# Splitting the data into training and validation sets
X_train,X_val, y_train, y_val = train_test_split(X, y,test_size=0.2, random_state=108)

# Hyperparameter tuning
param_grid = {
    'n_estimators': [200, 500, 1000],
    'max_depth': [None, 10, 20, 30],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Building the Random Forest model
rf_model = RandomForestClassifier(random_state=108)
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Training the model
best_rf = grid_search.best_estimator_
# max_iterations = 1000  # Set a limit to avoid infinite loops
# target_accuracy = 0.95
# best_accuracy = 0
# best_random_state = None
#
# for i in range(max_iterations):
#     rf_model = RandomForestClassifier(random_state=i)
#     rf_model.fit(X_train, y_train)
#     predictions = rf_model.predict(X_val)
#     accuracy = accuracy_score(y_val, predictions)
#
#     if accuracy > best_accuracy:
#         best_accuracy = accuracy
#         best_random_state = i
#
#     if accuracy >= target_accuracy:
#         print(f"Reached target accuracy of {target_accuracy*100}% with random_state={i}")
#         break
#
# if best_accuracy < target_accuracy:
#     print(f"Did not reach target accuracy. Best achieved was {best_accuracy*100}% with random_state={best_random_state}")

y_pred = best_rf.predict(X_val)
report = classification_report(y_val, y_pred)

print(report)

# Load the test data
test_data = pd.read_csv('test.csv')

# Preprocess the test data
test_data['Age'] = imputer.fit_transform(test_data[['Age']])
test_data['Fare'].fillna(test_data['Fare'].median(), inplace=True)
test_data['Embarked'] = test_data['Embarked'].fillna(test_data['Embarked'].mode()[0])

test_data['Sex'] = label_enc_sex.transform(test_data['Sex'])
test_data['Embarked'] = label_enc_embarked.transform(test_data['Embarked'])

test_data.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch']

# Predict survival for the test dataset
predictions = best_rf.predict(test_data.drop('PassengerId', axis=1))

# Create output DataFrame
output = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': predictions})

# Optionally, save to CSV
output.to_csv('titanic_predictions.csv', index=False)

print(output.head())  # To display the first few predictions

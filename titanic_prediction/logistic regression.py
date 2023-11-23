import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
file_path = 'train.csv'  # Replace with your file path
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
# Splitting the dataset into training and test sets
X = train_data.drop('Survived', axis=1)
y = train_data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=150)

# Normalizing the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Training the Logistic Regression model
log_reg = LogisticRegression(random_state=150)
log_reg.fit(X_train, y_train)

# Predictions on the test set
y_pred = log_reg.predict(X_test)

# Evaluating the model's performance
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print the evaluation metrics
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)



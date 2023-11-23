import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Load the Titanic dataset
url = 'train.csv'
train_data = pd.read_csv(url)


# Preprocessing
def preprocess_data(data):
  # Drop unnecessary columns
  data = data.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1)

  # Fill missing values
  data['Age'].fillna(data['Age'].median(), inplace=True)
  data['Fare'].fillna(data['Fare'].median(), inplace=True)
  data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

  # Convert categorical variables to numerical
  le = LabelEncoder()
  data['Sex'] = le.fit_transform(data['Sex'])
  data['Embarked'] = le.fit_transform(data['Embarked'])
  return data

titanic_data = preprocess_data(train_data)

# Define features and target variable
X = titanic_data.drop('Survived', axis=1)
y = titanic_data['Survived']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the Gradient Boosting Classifier model
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
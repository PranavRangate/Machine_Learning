from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import pandas as pd

# Load your dataset
data = pd.read_csv('d.csv')
X = data.drop('Loan', axis=1)
y = data['Loan']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the k-NN function
def knn_func(train_x, train_label, test_x, k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_x, train_label)
    predictions = knn.predict(test_x)
    return predictions, knn

# Train the model and get predictions
y_pred, model = knn_func(X_train, y_train, X_test, k=3)

# Check accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Function to check loan eligibility
def check_loan_eligibility(features, model):
    scaled_features = scaler.transform([features])
    loan_prediction = model.predict(scaled_features)
    return loan_prediction[0]

# Example usage to check loan eligibility
new_customer = [30, 15, 2]
eligibility = check_loan_eligibility(new_customer, model)

print(f"Loan eligibility (0 = Not Eligible, 1 = Eligible): {eligibility}")

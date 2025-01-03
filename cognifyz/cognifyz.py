# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
  # Update this path if needed
data = pd.read_csv('Dataset .csv')

# Task 1: Predict Restaurant Ratings
# Step 1: Preprocess the dataset
data_cleaned = data.copy()
data_cleaned.fillna(method='ffill', inplace=True)

# Encode categorical variables
categorical_columns = ['Restaurant Name', 'City', 'Cuisines', 'Currency', 
                       'Rating color', 'Rating text', 'Has Table booking', 
                       'Has Online delivery', 'Is delivering now', 'Switch to order menu']

label_encoders = {}
for column in categorical_columns:
    le = LabelEncoder()
    data_cleaned[column] = le.fit_transform(data_cleaned[column])
    label_encoders[column] = le

# Select features and target
X = data_cleaned[['Price range', 'Votes', 'Has Table booking', 'City', 'Cuisines']]
y = data_cleaned['Aggregate rating']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict and evaluate
y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Task 1 - Predict Restaurant Ratings:\nMSE: {mse}\nR-squared: {r2}\n")

# Task 2: Restaurant Recommendation System
# Step 1: Preprocess and prepare the dataset for recommendation
recommendation_features = data_cleaned[['Cuisines', 'Price range', 'City']]

# Fit Nearest Neighbors model
recommendation_model = NearestNeighbors(n_neighbors=5, metric='euclidean')
recommendation_model.fit(recommendation_features)

# Provide sample user preference and find recommendations
sample_preference = [[1000, 2, 50]]  # Replace with actual preferences
recommendations = recommendation_model.kneighbors(sample_preference, return_distance=False)

print("Task 2 - Restaurant Recommendation System:\nRecommended Restaurant Indices:", recommendations[0])

# Task 3: Cuisine Classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Select features and target
X_classification = data_cleaned[['Price range', 'Votes', 'City', 'Has Table booking']]
y_classification = data_cleaned['Cuisines']

# Split the data
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_classification, y_classification, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train_cls, y_train_cls)

# Predict and evaluate
y_pred_cls = classifier.predict(X_test_cls)
accuracy = accuracy_score(y_test_cls, y_pred_cls)
report = classification_report(y_test_cls, y_pred_cls)
print(f"Task 3 - Cuisine Classification:\nAccuracy: {accuracy}\n{report}\n")

# Task 4: Location-based Analysis
# Visualize restaurant distribution
data_cleaned['City'] = label_encoders['City'].inverse_transform(data_cleaned['City'])
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data_cleaned, x='Longitude', y='Latitude', hue='City')
plt.title("Restaurant Distribution by Location")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

# Group by city and analyze
city_stats = data_cleaned.groupby('City').agg({
    'Aggregate rating': 'mean',
    'Cuisines': 'nunique',
    'Price range': 'mean'
})
print("Task 4 - Location-based Analysis:\n", city_stats)

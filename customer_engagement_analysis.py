# Customer Engagement Analysis using E-commerce Dataset

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score

# Step 2: Load Dataset
# Load the dataset
data = pd.read_csv('ecommerce_customer_data.csv')
print("Dataset Preview:")
print(data.head())

# Step 3: Data Cleaning
# Drop irrelevant columns (if any exist)
data = data.drop(['CustomerID', 'ProductID'], axis=1, errors='ignore')

# Handle missing values
# Only apply median imputation on numeric columns
numeric_data = data.select_dtypes(include=[np.number])  # Select only numeric columns
data[numeric_data.columns] = numeric_data.fillna(numeric_data.median())

# Handle missing values in categorical columns (e.g., Gender, Product Category)
data['Gender'].fillna(data['Gender'].mode()[0], inplace=True)
data['Product Category'].fillna(data['Product Category'].mode()[0], inplace=True)

# Handle missing values in the 'Purchase Date' column (if necessary)
# You can either drop rows with missing dates or fill them with a placeholder
# For this example, let's drop rows with missing 'Purchase Date'
data.dropna(subset=['Purchase Date'], inplace=True)

# Step 4: Exploratory Data Analysis
# Check correlations (removing non-numeric columns)
numeric_data = data.select_dtypes(include=[np.number])  # Only numeric columns for correlation
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Distribution of Quantity (as a proxy for engagement)
sns.histplot(data['Quantity'], bins=30, kde=True)
plt.title('Quantity Distribution (Proxy for Engagement)')
plt.xlabel('Quantity')
plt.show()

# Step 5: Feature Engineering
# Add Behavioral Features (e.g., time spent on the website, feedback ratings)
# Assuming 'SessionTime' represents time spent on the website and 'FeedbackRating' is the customer rating for the product
# Create an engagement score based on multiple features: Quantity, Product Price, Session Time, and Feedback Rating
# If 'SessionTime' or 'FeedbackRating' columns are missing, you can add or create placeholders or logic as needed
if 'SessionTime' in data.columns and 'FeedbackRating' in data.columns:
    data['EngagementScore'] = data['Quantity'] * data['Product Price'] * data['SessionTime'] * data['FeedbackRating']
else:
    data['EngagementScore'] = data['Quantity'] * data['Product Price']  # Default engagement score without those columns

# Add 'HighEngagement' binary target (based on EngagementScore)
data['HighEngagement'] = (data['EngagementScore'] > data['EngagementScore'].median()).astype(int)

# Additional Features: Recency of Activity (Days since last purchase)
# Assuming 'Purchase Date' is in datetime format
data['PurchaseDate'] = pd.to_datetime(data['Purchase Date'])
data['DaysSinceLastPurchase'] = (data['PurchaseDate'].max() - data['PurchaseDate']).dt.days

# Drop unnecessary columns
X = data.drop(['HighEngagement', 'EngagementScore', 'Purchase Date'], axis=1)
y = data['HighEngagement']

# Normalize numeric features
scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X.select_dtypes(include=[np.number])), columns=X.select_dtypes(include=[np.number]).columns)

# Step 6: Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Model Evaluation
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))
print("Accuracy:", accuracy_score(y_test, y_pred_rf))

# Feature Importance
importance_rf = pd.DataFrame({'Feature': X.columns, 'Importance': rf_model.feature_importances_})
importance_rf = importance_rf.sort_values(by='Importance', ascending=False)

# Visualize Feature Importance
sns.barplot(x='Importance', y='Feature', data=importance_rf)
plt.title('Feature Importance')
plt.show()

# Step 8: Save Results
data.to_csv('processed_ecommerce_data.csv', index=False)
with open('rf_model_summary.txt', 'w') as f:
    f.write("Random Forest Accuracy: {:.2f}\n".format(accuracy_score(y_test, y_pred_rf)))
    f.write("\nFeature Importances:\n")
    f.write(importance_rf.to_string(index=False))

print("\nAnalysis Complete! Processed data and results saved.")

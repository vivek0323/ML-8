#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# Data
data = {
    'Age': [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
    'Income': [55000, 48000, 45000, 42000, 37000, 33000, 29000, 25000, 22000, 19000, 16000],
    'Student': [0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    'Credit rating': ['fair', 'good', 'excellent', 'fair', 'fair', 'fair', 'poor', 'poor', 'poor', 'fair', 'poor'],
    'Buys computer': ['yes', 'no', 'yes', 'no', 'yes', 'no', 'no', 'no', 'no', 'no', 'no']
}

# Create DataFrame
df = pd.DataFrame(data)

# Calculate prior probabilities
prior_probabilities = df['Buys computer'].value_counts(normalize=True)

# Print prior probabilities
print(prior_probabilities)


# In[2]:


import pandas as pd
from scipy.stats import norm

# Assuming 'df' is the DataFrame with the provided data
# You can adapt this code based on the specific features you want to analyze

# Separate the data by class
class_yes = df[df['Buys computer'] == 'yes']
class_no = df[df['Buys computer'] == 'no']

# Calculate class-conditional densities for the 'Age' feature
density_age_yes = norm.pdf(class_yes['Age'], loc=class_yes['Age'].mean(), scale=class_yes['Age'].std())
density_age_no = norm.pdf(class_no['Age'], loc=class_no['Age'].mean(), scale=class_no['Age'].std())

# Calculate class-conditional densities for the 'Income' feature
density_income_yes = norm.pdf(class_yes['Income'], loc=class_yes['Income'].mean(), scale=class_yes['Income'].std())
density_income_no = norm.pdf(class_no['Income'], loc=class_no['Income'].mean(), scale=class_no['Income'].std())

# Print the densities
print("Class-conditional densities for 'Age' and 'Buys computer = yes':\n", density_age_yes)
print("Class-conditional densities for 'Age' and 'Buys computer = no':\n", density_age_no)
print("Class-conditional densities for 'Income' and 'Buys computer = yes':\n", density_income_yes)
print("Class-conditional densities for 'Income' and 'Buys computer = no':\n", density_income_no)


# In[1]:


import pandas as pd
from scipy.stats import chi2_contingency

# Data
data = {
    'Age': [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
    'Income': [55000, 48000, 45000, 42000, 37000, 33000, 29000, 25000, 22000, 19000, 16000],
    'Student': [0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    'Credit rating': ['fair', 'good', 'excellent', 'fair', 'fair', 'fair', 'poor', 'poor', 'poor', 'fair', 'poor'],
    'Buys computer': ['yes', 'no', 'yes', 'no', 'yes', 'no', 'no', 'no', 'no', 'no', 'no']
}

# Create DataFrame
df = pd.DataFrame(data)

# Create a contingency table
contingency_table = pd.crosstab(df['Credit rating'], df['Buys computer'])

# Test for independence
c_statistic, p_value, _, _ = chi2_contingency(contingency_table)

# Print results
print("Chi-square statistic:", c_statistic)
print("P-value:", p_value)


# In[2]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Data
data = {
    'Age': [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
    'Income': [55000, 48000, 45000, 42000, 37000, 33000, 29000, 25000, 22000, 19000, 16000],
    'Student': [0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    'Credit rating': ['fair', 'good', 'excellent', 'fair', 'fair', 'fair', 'poor', 'poor', 'poor', 'fair', 'poor'],
    'Buys computer': ['yes', 'no', 'yes', 'no', 'yes', 'no', 'no', 'no', 'no', 'no', 'no']
}

# Create DataFrame
df = pd.DataFrame(data)

# Convert categorical variables to numerical using one-hot encoding
df_encoded = pd.get_dummies(df, columns=['Credit rating'])

# Features and target variable
X = df_encoded.drop('Buys computer', axis=1)
y = df_encoded['Buys computer']

# Split the data into training and testing sets
Tr_X, Te_X, Tr_y, Te_y = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Gaussian Naive Bayes model
nb_model = GaussianNB()
nb_model.fit(Tr_X, Tr_y)

# Make predictions on the test set
predictions = nb_model.predict(Te_X)

# Evaluate the model
accuracy = accuracy_score(Te_y, predictions)
classification_rep = classification_report(Te_y, predictions)

# Print results
print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_rep)


# In[3]:


pip install pandas scikit-learn


# In[3]:


import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Data provided
data = {
    'age': ['<=30', '<=30', '31...40', '>40', '>40', '>40', '31...40', '<=30', '<=30', '>40', '<=30', '31...40', '31...40', '>40'],
    'income': ['high', 'high', 'high', 'medium', 'low', 'low', 'low', 'medium', 'low', 'medium', 'medium', 'medium', 'high', 'medium'],
    'student': ['no', 'no', 'no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no'],
    'credit_rating': ['fair', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'excellent'],
    'buys_computer': ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']
}

# Create a DataFrame
df = pd.DataFrame(data)

# Convert categorical variables to numerical using LabelEncoder
le = LabelEncoder()
df['age'] = le.fit_transform(df['age'])
df['income'] = le.fit_transform(df['income'])
df['student'] = le.fit_transform(df['student'])
df['credit_rating'] = le.fit_transform(df['credit_rating'])
df['buys_computer'] = le.fit_transform(df['buys_computer'])

# Split the data into features (X) and target variable (y)
X = df.drop('buys_computer', axis=1)
y = df['buys_computer']

# Split the data into training and testing sets
Tr_X, Ts_X, Tr_y, Ts_y = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the Naive Bayes (NB) classifier
model = GaussianNB()
model.fit(Tr_X, Tr_y)

# Print accuracy on the training set
accuracy_train = metrics.accuracy_score(Tr_y, model.predict(Tr_X))
print(f"Accuracy on training set: {accuracy_train}")

# Now you can use the trained model to make predictions on the test set (Ts_X)
predictions = model.predict(Ts_X)

# Print accuracy on the test set
accuracy_test = metrics.accuracy_score(Ts_y, predictions)
print(f"Accuracy on test set: {accuracy_test}")


# In[10]:


print(train_data.columns)


# In[19]:


import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the data
train_data = pd.read_excel('training (2) (1).xlsx')
test_data = pd.read_excel('testing (2) (1).xlsx')

# Convert the 'input' column to numeric, handling non-numeric values by replacing them with NaN
train_data['input'] = pd.to_numeric(train_data['input'], errors='coerce')

# Replace NaN values in the 'input' column with a default value (e.g., 0)
train_data['input'] = train_data['input'].fillna(0)

# Drop rows with NaN values in 'output' and 'Classification' columns
train_data = train_data.dropna(subset=['output', 'Classification'])

# Separate features and labels in the training dataset
train_features = train_data[['input', 'output']]
train_labels = train_data['Classification']

# Ensure that 'train_labels' is not empty
if not train_labels.empty:
    # Split the training data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(train_features, train_labels, test_size=0.2)

    # Train the Naive Bayes classifier
    model = GaussianNB()
    model.fit(X_train, y_train)

    # Make predictions on the validation set
    y_pred = model.predict(X_val)

    # Evaluate the performance of the model
    accuracy = accuracy_score(y_val, y_pred)
    print("Accuracy:", accuracy)
else:
    print("No samples left after preprocessing.")


# In[ ]:





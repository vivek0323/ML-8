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


# In[ ]:





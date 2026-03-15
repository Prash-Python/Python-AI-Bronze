import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

# 1. Dataset Construction
data = {
    'Age': [28, 45, 35, 50, 30, 42, 26, 48, 38, 55],
    'AnnualIncome': [6.5, 12, 8, 15, 7, 10, 5.5, 14, 9, 16],
    'CreditScore': [720, 680, 750, 640, 710, 660, 730, 650, 700, 620],
    'LoanAmount': [5, 10, 6, 12, 5, 9, 4, 11, 7, 13],
    'LoanTerm': [5, 10, 7, 15, 5, 10, 4, 12, 8, 15],
    'EmploymentType': ['Salaried', 'Self-Employed', 'Salaried', 'Self-Employed', 'Salaried', 
                       'Salaried', 'Salaried', 'Self-Employed', 'Salaried', 'Self-Employed'],
    'Default': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
}
df = pd.DataFrame(data)

# 2. Feature Engineering: Debt-to-Income (DTI) Ratio
# This normalizes the Loan Amount against Income
df['DTI_Ratio'] = df['LoanAmount'] / df['AnnualIncome']

# 3. Encoding & Scaling
le = LabelEncoder()
df['EmploymentType'] = le.fit_transform(df['EmploymentType'])

X = df.drop('Default', axis=1)
y = df['Default']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Training KNN (k=3)
knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn.fit(X_scaled, y)

print("Model trained successfully on financial features.")
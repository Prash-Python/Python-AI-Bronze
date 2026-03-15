import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# 1. Prepare the dataset
data = {
    'Experience': [2, 5, 1, 8, 4, 10, 3, 6, 7, 2],
    'Training_Hours': [40, 60, 20, 80, 50, 90, 30, 70, 75, 25],
    'Working_Hours': [38, 42, 35, 45, 40, 48, 37, 44, 46, 36],
    'Projects': [3, 6, 2, 8, 5, 9, 4, 7, 7, 3],
    'Productivity_Score': [62, 78, 55, 88, 72, 92, 65, 82, 85, 60]
}

df = pd.DataFrame(data)

# 2. Define Features (X) and Target (y)
X = df[['Experience', 'Training_Hours', 'Working_Hours', 'Projects']]
y = df['Productivity_Score']

# 3. Initialize and Train the Model
model = LinearRegression()
model.fit(X, y)

# 4. Extract Results
intercept = model.intercept_
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])

print(f"Base Productivity (Intercept): {intercept:.2f}")
print("\nImpact of each factor (Coefficients):")
print(coefficients)

# 5. Example Prediction: 5 years exp, 50 training hours, 40 work hours, 5 projects
new_employee = np.array([[5, 50, 40, 5]])
prediction = model.predict(new_employee)
print(f"\nPredicted Productivity Score: {prediction[0]:.2f}")
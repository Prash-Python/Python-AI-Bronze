import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. Generate Sample Data
# Area (sq ft), Number of Bedrooms
X = np.array([
    [1500, 3], [2000, 4], [1200, 2], [2500, 4], [1800, 3],
    [2200, 4], [1400, 2], [2800, 5], [1600, 3], [2100, 3]
])
# House Price (in $1000s)
y = np.array([300, 400, 250, 500, 360, 440, 280, 580, 320, 410])

# 2. Fit the Linear Regression Model
model = LinearRegression()
model.fit(X, y)

# 3. Create a Grid for the Surface Plot
# We create a mesh of points to draw the "Plane"
x_range = np.linspace(X[:,0].min(), X[:,0].max(), 10)
y_range = np.linspace(X[:,1].min(), X[:,1].max(), 10)
x_mesh, y_mesh = np.meshgrid(x_range, y_range)

# Predict prices over the grid
# .ravel() flattens the grid for the model, .reshape() puts it back for plotting
predict_mesh = model.predict(np.c_[x_mesh.ravel(), y_mesh.ravel()])
z_mesh = predict_mesh.reshape(x_mesh.shape)

# 4. 3D Visualization
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the actual data points (Scatter)
ax.scatter(X[:,0], X[:,1], y, color='red', marker='o', s=50, label='Actual Data')

# Plot the Regression Plane
surface = ax.plot_surface(x_mesh, y_mesh, z_mesh, alpha=0.5, cmap='viridis')

# Labels and Titles
ax.set_xlabel('Area (sq ft)')
ax.set_ylabel('No. of Bedrooms')
ax.set_zlabel('Price ($1000s)')
ax.set_title('3D Regression Plane: House Price Prediction')

plt.colorbar(surface, ax=ax, shrink=0.5, aspect=5)
plt.legend()
plt.show()
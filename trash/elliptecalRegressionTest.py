import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from convexhull import addNoise

# Generate synthetic elliptical data with noise

np.random.seed(42)
theta = np.linspace(0, 2 * np.pi, 2000)
a, b = 600, 500  # Semi-major and semi-minor axes
x = a * np.cos(theta)
y = b * np.sin(theta)
x +=0
"""
x_noisy, y_noisy = addNoise(x_ellipse, y_ellipse, 40, 2, 2)

# Add Gaussian noise to simulate real-world data
noise_x = np.random.normal(0, 0.5, x_ellipse.shape)
noise_y = np.random.normal(0, 0.5, y_ellipse.shape)
x_noisy = x_ellipse + noise_x
y_noisy = y_ellipse + noise_y

"""

x_noisy, y_noisy = addNoise(x, y, 50, 5, 5)

# Combine noisy data into a single array
data = np.vstack((x_noisy, y_noisy)).T

# Define the cost function for fitting an ellipse
def ellipse_cost(params, data):
    A, B, C, D, E, F = params
    x, y = data[:, 0], data[:, 1]
    # Compute the value of the ellipse equation for each point
    cost = (A * x**2 + B * x * y + C * y**2 + D * x + E * y + F)**2
    back_end = (int)(cost.size * -.2)
    #back_end = -9
    return np.sum(np.sort(cost)[:back_end])

# Initial guess for parameters
initial_params = [1, 0, 1, 0, 0, -1]

# Fit the ellipse to noisy data using optimization
result = minimize(ellipse_cost, initial_params, args=(data,))
fitted_params = result.x

# Extract fitted parameters
A, B, C, D, E, F = fitted_params

# Generate fitted ellipse points for visualization
def generate_fitted_ellipse(A, B, C, D, E, F, num_points=500):
    """
    Generate points on an ellipse using parametric equations for better performance.

    Parameters:
        A, B, C, D, E, F: Coefficients of the general quadratic ellipse equation.
        num_points: Number of points to generate on the ellipse.

    Returns:
        np.array: Array of shape (num_points, 2) containing (x, y) coordinates of the ellipse.
    """
    # Generate theta values for parametric representation
    theta = np.linspace(0, 2 * np.pi, num_points)

    # Calculate the center of the ellipse using partial derivatives
    x_center = (2 * C * D - B * E) / (B**2 - 4 * A * C)
    y_center = (2 * A * E - B * D) / (B**2 - 4 * A * C)
    print(x_center)
    print(y_center)
    # Approximate semi-major and semi-minor axes
    semi_major = np.sqrt(abs(F / A))
    semi_minor = np.sqrt(abs(F / C))

    # Generate points parametrically
    x = x_center + semi_major * np.cos(theta)
    y = y_center + semi_minor * np.sin(theta)


    return plt.plot(x, y, label="Optimized Ellipse", color="red", linewidth=2)




def calculate_ellipse_params(a, b, h=0, k=0, theta=0):
    """
    Calculate the parameters A, B, C, D, E, F for the general quadratic equation of an ellipse.

    Parameters:
        a (float): Semi-major axis length.
        b (float): Semi-minor axis length.
        h (float): x-coordinate of the center (default is 0).
        k (float): y-coordinate of the center (default is 0).
        theta (float): Rotation angle in radians (default is 0).

    Returns:
        tuple: Coefficients (A, B, C, D, E, F) for the general quadratic equation.
    """
    # Calculate coefficients
    A = a**2 * np.sin(theta)**2 + b**2 * np.cos(theta)**2
    B = 2 * (a**2 - b**2) * np.sin(theta) * np.cos(theta)
    C = a**2 * np.cos(theta)**2 + b**2 * np.sin(theta)**2
    D = -2 * A * h - B * k
    E = -B * h - 2 * C * k
    F = A * h**2 + B * h * k + C * k**2 - a**2 * b**2

    return A, B, C, D, E, F

print('cost: '+ str(ellipse_cost(fitted_params, data)))
# Plot the noisy data and fitted ellipse
plt.figure(figsize=(8, 6))
plt.scatter(x_noisy, y_noisy, label="Noisy Data", alpha=0.6)
generate_fitted_ellipse(A, B, C, D, E, F)
plt.xlabel("X")
plt.ylabel("Y")
plt.legend(["Fitted Ellipse", "Noisy Data"])
plt.title("Elliptical Regression with Noise")
#plt.axis("equal")
plt.show()


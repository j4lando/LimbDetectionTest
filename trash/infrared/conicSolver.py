import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def fit_conic(points):
    # Ensure we have 5 points
    if len(points) != 5:
        raise ValueError("Exactly 5 points are required to determine a unique conic section")

    # Create the matrix M
    M = np.zeros((5, 6))
    for i, (x, y) in enumerate(points):
        M[i] = [x**2, x*y, y**2, x, y, 1]

    # Solve the system of equations
    _, _, V = np.linalg.svd(M)
    coeffs = V[-1]

    # Normalize coefficients
    coeffs = coeffs / coeffs[0]

    return coeffs

def ellipse_cost(params, data):
    A, B, C, D, E, F = params
    x, y = data[:, 0], data[:, 1]
    # Compute the value of the ellipse equation for each point
    cost = (A * x**2 + B * x * y + C * y**2 + D * x + E * y + F)**2
    back_end = (int)(cost.size * -.2)
    #back_end = -9
    return np.sum(cost)
    return np.sum(np.sort(cost)[:back_end])

def minimze_conic(data):
    # Initial guess for parameters
    initial_params = [1, 0, 1, 0, 0, -1]

    # Fit the ellipse to noisy data using optimization
    result = minimize(ellipse_cost, initial_params, args=(data,))
    fitted_params = result.x

    # Extract fitted parameters
    return fitted_params

# Example usage
def generate_conic_points(a, b, h =0, k =1000):
    theta = np.linspace(0, 2 * np.pi, 6)
    x = a * np.cos(theta) + h
    y = b * np.sin(theta) + k
    
    A = 1 / (a**2)
    C = 1 / (b**2)
    B = 0
    D = -2 * h * A
    E = -2 * k * C
    F = (h**2 * A) + (k**2 * C) - 1


    return np.column_stack((x, y)), [A, B, C, D, E, F]

def generate_fitted_ellipse(fitted_params, num_points=500):
    A, B, C, D, E, F = fitted_params
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

    A = 1 / (semi_major**2)
    C = 1 / (semi_minor**2)
    B = 0
    D = -2 * x_center * A
    E = -2 * y_center * C
    F = (x_center**2 * A) + (y_center**2 * C) - 1


    return np.column_stack((x, y)), [A, B, C, D, E, F]
# Example usage

random_points, example_coeffs = generate_conic_points(40, 50)
generated_points, generated_coeffs = generate_fitted_ellipse(example_coeffs)
print(random_points)
coeffs = minimze_conic(random_points)

print("Coefficients [A, B, C, D, E, F]:")
print(example_coeffs)
print(generated_coeffs)
print(coeffs)

# Determine the type of conic section
discriminant = coeffs[1]**2 - 4*coeffs[0]*coeffs[2]
if abs(discriminant) < 1e-6:
    print("The conic section is a parabola")
elif discriminant < 0:
    print("The conic section is an ellipse")
else:
    print("The conic section is a hyperbola")


points, trash = generate_fitted_ellipse(coeffs)
plt.scatter(random_points[:, 0], random_points[:, 1], color='blue', label='Points')
#plt.plot(points[:, 0], points[:, 1], color='red', label='Line')
#plt.plot(generated_points[:, 0], generated_points[:, 1], color='green', label='Correct_Line')
plt.title('Numpy Array of Points with Line')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.grid(True)
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import cv2

def ellipse_cost(params, data):
    A, B, C, D, E, F = params
    x, y = data[:, 0], data[:, 1]
    # Compute the value of the ellipse equation for each point
    cost = (A * x**2 + B * x * y + C * y**2 + D * x + E * y + F)**2
    back_end = (int)(cost.size * -.8)
    #back_end = -9
    return np.sum(np.sort(cost)[:back_end])

# Initial guess for parameters
def fit(data):
    initial_params = [1, 0, 1, 0, 0, -1]

    # Fit the ellipse to noisy data using optimization
    result = minimize(ellipse_cost, initial_params, args=(data,))
    fitted_params = result.x

    # Extract fitted parameters
    A, B, C, D, E, F = fitted_params
    return fitted_params

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

    # Approximate semi-major and semi-minor axes
    semi_major = np.sqrt(abs(F / A))
    semi_minor = np.sqrt(abs(F / C))

    # Generate points parametrically
    x = x_center + semi_major * np.cos(theta)
    y = y_center + semi_minor * np.sin(theta)


    return plt.plot(x, y, label="Optimized Ellipse", color="red", linewidth=2)

def plot(img):
    points = np.argwhere(img == 255)
    x_coords, y_coords = points[:, 1], points[:, 0]  # Note: Flip row and column for plotting
    x_coords -=200
    y_coords -=200
    data = np.vstack((x_coords, y_coords)).T
    A, B, C, D, E, F = fit(data)

    # Plot the noisy data and fitted ellipse
    plt.figure(figsize=(8, 6))
    plt.imshow(img, cmap='gray')
    plt.scatter(x_coords, y_coords, color='green', s=.5)
    generate_fitted_ellipse(A, B, C, D, E, F)
    plt.xlabel("X")
    plt.ylabel("Y")
    #plt.axis("equal")
    plt.show()


image = cv2.imread('test_images/earth_night.jpg', 0 )
threshold = 10


binary_mask = np.where(image < threshold, 255, 0).astype(np.uint8)
# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(binary_mask, (3,3), 0)

# Apply Laplacian of Gaussian
log = cv2.Laplacian(blurred, cv2.CV_64F)

# Perform Canny edge detection
canny = cv2.Canny(binary_mask, 100, 200)
# Find zero crossings
zero_crossings = np.zeros_like(log)
zero_crossings[:-1, :-1] = np.logical_and(log[:-1, :-1] < 0, log[1:, 1:] > 0) | \
                           np.logical_and(log[:-1, :-1] > 0, log[1:, 1:] < 0)

# Apply thresholding to get strong edges
threshold = 0.1 * np.max(np.abs(log))
edges = np.zeros_like(zero_crossings, dtype=np.uint8)
edges[np.abs(log) > threshold] = 255

plot(canny)
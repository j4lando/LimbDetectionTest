import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import cv2

# Ellipse cost function
def ellipse_cost(params, data):
    A, B, C, D, E, F = params
    x, y = data[:, 0], data[:, 1]
    cost = (A * x**2 + B * x * y + C * y**2 + D * x + E * y + F)**2
    valid_cost = np.sort(cost)[:-10]  # Exclude top 10 largest values
    return np.sum(valid_cost)

# Fit ellipse parameters
def fit(data):
    initial_params = [1, 0, 1, 0, 0, -1]
    result = minimize(ellipse_cost, initial_params, args=(data,))
    return result.x

# Generate fitted ellipse points for visualization
def generate_fitted_ellipse(A, B, C, D, E, F, num_points=500):
    theta = np.linspace(0, 2 * np.pi, num_points)
    x_center = (2 * C * D - B * E) / (B**2 - 4 * A * C)
    y_center = (2 * A * E - B * D) / (B**2 - 4 * A * C)
    semi_major = np.sqrt(abs(F / A))
    semi_minor = np.sqrt(abs(F / C))
    x = x_center + semi_major * np.cos(theta)
    y = y_center + semi_minor * np.sin(theta)
    plt.plot(x, y, label="Fitted Ellipse", color="red", linewidth=2)

# Plotting function
def plot(img):
    points = np.argwhere(img == 255)
    if len(points) == 0:
        print("No points detected in the image.")
        return
    x_coords, y_coords = points[:, 1], points[:, 0]
    data = np.vstack((x_coords, y_coords)).T
    A, B, C, D, E, F = fit(data)
    plt.figure(figsize=(8, 6))
    plt.scatter(x_coords, y_coords, color='green', s=0.5)
    generate_fitted_ellipse(A, B, C, D, E, F)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Fitted Ellipse on Edge Points")
    plt.axis("equal")
    plt.show()

# Preprocessing and edge detection
image = cv2.imread('test_images/earth.jpg', cv2.IMREAD_GRAYSCALE)
threshold_value = 10
binary_mask = np.where(image < threshold_value, 255, 0).astype(np.uint8)
blurred = cv2.GaussianBlur(binary_mask, (3, 3), 0)
laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
threshold = 0.1 * np.max(np.abs(laplacian))
edges = np.zeros_like(laplacian, dtype=np.uint8)
edges[np.abs(laplacian) > threshold] = 255

# Plot edges and fit ellipse
plot(edges)

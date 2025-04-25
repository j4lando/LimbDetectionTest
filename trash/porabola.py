import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.linear_model import RANSACRegressor, LinearRegression

# Define the parabola function for fitting
def parabola_fit(x, a, b, c):
    return a * x**2 + b * x + c

# Define the points
x = np.linspace(-2, 2, 80)

# Compute the corresponding y-values for the parabola y = x^2 - 1
y = x**2 - 1


new_random_x = np.random.uniform(-3, 3, 10)

# Compute the corresponding y-values
new_random_y = np.random.uniform(-3, 3, 10)

# Append these new random points to x and y
x_noisy = np.append(x, new_random_x)
y_noisy = np.append(y, new_random_y)
# Fit a parabola (2nd degree polynomial)



# Reshape x_noisy for RANSAC
x_noisy_reshaped = x_noisy.reshape(-1, 1)

# Create a RANSAC regressor with a linear model
ransac = RANSACRegressor(LinearRegression(), min_samples=0.5, residual_threshold=1.0)

# Fit the RANSAC model to the noisy data
ransac.fit(x_noisy_reshaped, y_noisy)

# Get inlier mask and outlier mask
inlier_mask = ransac.inlier_mask_

# Fit the parabola to the inliers only using curve fitting
params, _ = curve_fit(parabola_fit, x_noisy[inlier_mask], y_noisy[inlier_mask])
a_fit, b_fit, c_fit = params

# Generate fitted parabola points using the fitted parameters
x_fit = np.linspace(-3.5, 3.5, 100)
y_fit_fitted = parabola_fit(x_fit, a_fit, b_fit, c_fit)

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(x_noisy, y_noisy, color='red', alpha=0.6, label='Noisy Points')
plt.scatter(x_noisy[inlier_mask], y_noisy[inlier_mask], color='blue', label='Inliers')
plt.plot(x_fit, y_fit_fitted, color='blue', linewidth=2.0,
         label=f'Fitted Parabola: $y = {a_fit:.2f}x^2 + {b_fit:.2f}x + {c_fit:.2f}$')
plt.title('Parabola of Best Fit Ignoring Outliers')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()
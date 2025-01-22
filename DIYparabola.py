import numpy as np
import matplotlib.pyplot as plt

def parabola_fit(x, a, b, c):
    return a * x**2 + b * x + c
# Define the quadratic regression function
def quadratic_regression(X, Y):
    """
    Perform quadratic regression to find coefficients for y = ax^2 + bx + c.
    
    Parameters:
        X (array-like): Independent variable values.
        Y (array-like): Dependent variable values.
        
    Returns:
        coefficients (numpy array): Coefficients [c, b, a] for the quadratic equation.
    """
    # Convert X and Y to numpy arrays
    X = np.array(X)
    Y = np.array(Y)
    
    # Create the design matrix with columns [1, X, X^2]
    X_design = np.column_stack((np.ones(len(X)), X, X**2))
    
    # Calculate coefficients using the normal equation: (X^T * X)^(-1) * X^T * Y
    coefficients = np.linalg.inv(X_design.T @ X_design) @ X_design.T @ Y
    
    return coefficients

def addNoise(x,y, num):
    new_random_x = np.random.uniform(-3, 3, num)

    # Compute the corresponding y-values
    new_random_y = np.random.uniform(-3, 3, num)

    # Append these new random points to x and y
    x_noisy = np.append(x, new_random_x)
    y_noisy = np.append(y, new_random_y)
    return x_noisy, y_noisy

def removePoint(x,y, c):
    # create vecor of R^2 terms
    errors = (y - parabola_fit(x, c[2], c[1], c[0])) ** 2
    x = np.delete(x, np.argmax(errors))
    y = np.delete(y, np.argmax(errors))
    return x, y, (y - parabola_fit(x, c[2], c[1], c[0])) ** 2
"""
# Example data
X = np.linspace(-2, 2, 20)

# Compute the corresponding y-values for the parabola y = x^2 - 1
Y = X**2 - 1

#addnoise
X, Y = addNoise(X, Y, 6)
# Perform quadratic regression
coefficients = quadratic_regression(X, Y)
X_fit = np.linspace(min(X), max(X), 100)
Y_fit = coefficients[0] + coefficients[1] * X_fit + coefficients[2] * X_fit**2
plt.figure(figsize=(12, 4))
plt.subplot(131), plt.scatter(X, Y, color='blue', label='Original Data')
plt.subplot(131), plt.plot(X_fit, Y_fit, color='red', label='Quadratic Fit') 
for i in range(0, 100):
    coefficients = quadratic_regression(X, Y)
    X, Y, errors = removePoint(X, Y, coefficients)
    if(np.sum(errors) < .5):
        print(np.sum(errors))
        break
"""

def error_elimination(X,Y):
    for i in range(0, 100):
        coefficients = quadratic_regression(X, Y)
        X, Y, errors = removePoint(X, Y, coefficients)
        if(np.sum(errors) < .5):
            #print(np.sum(errors))
            break
    return np.column_stack((X, Y))

"""
# Print the coefficients
print(f"Coefficients: c = {coefficients[0]:.4f}, b = {coefficients[1]:.4f}, a = {coefficients[2]:.4f}")

# Generate predictions using the fitted model
X_fit = np.linspace(min(X), max(X), 100)
Y_fit = coefficients[0] + coefficients[1] * X_fit + coefficients[2] * X_fit**2

# Plot the original data and the quadratic fit
plt.subplot(132), plt.scatter(X, Y, color='blue', label='Original Data')
plt.subplot(132), plt.plot(X_fit, Y_fit, color='red', label='Quadratic Fit') 
plt.show()
"""

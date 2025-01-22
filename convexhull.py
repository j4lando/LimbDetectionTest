import numpy as np
import matplotlib.pyplot as plt
from DIYparabola import error_elimination

def convex_hull(points):
    """
    Compute the convex hull of a set of 2D points using the Graham scan algorithm.
    
    Args:
    points: numpy array of shape (n, 2) representing n points in 2D space
    
    Returns:
    numpy array of shape (m, 2) representing the m points of the convex hull
    """
    def orientation(p, q, r):
        """
        Determine the orientation of triplet (p, q, r).
        Returns:
         0 --> p, q and r are collinear
         1 --> Clockwise
        -1 --> Counterclockwise
        """
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0:
            return 0
        return 1 if val > 0 else -1

    # Find the bottommost point (and leftmost if there are multiple)
    start = min(points, key=lambda p: (p[1], p[0]))
    
    # Sort points based on polar angle with respect to start point
    sorted_points = sorted(points, key=lambda p: (np.arctan2(p[1] - start[1], p[0] - start[0]), 
                                                  np.linalg.norm(p - start)))
    
    # Initialize the stack with first three points
    stack = [start, sorted_points[0], sorted_points[1]]
    
    # Process remaining points
    for i in range(2, len(sorted_points)):
        while len(stack) > 1 and orientation(stack[-2], stack[-1], sorted_points[i]) != -1:
            stack.pop()
        stack.append(sorted_points[i])
    
    return np.array(stack)

def addNoise(x,y, num, x_c, y_c):
    x_range = np.max(abs(x)) * x_c
    y_range = np.max(abs(y)) * y_c
    new_random_x = np.random.uniform(-x_range, x_range, num)

    # Compute the corresponding y-values
    new_random_y = np.random.uniform(-y_range, y_range, num)

    # Append these new random points to x and y
    x_noisy = np.append(x, new_random_x)
    y_noisy = np.append(y, new_random_y)
    return x_noisy, y_noisy

def calculateError(points, hull):
    c = quadratic_regression(points)
    errors = (hull[:,1] - parabola_fit(hull[:,0], c[2], c[1], c[0])) ** 2
    errors = errors[:, np.newaxis]
    
    # Add a column with the index of each row
    indexed_errors = np.hstack((errors, np.arange(len(errors)).reshape(-1, 1)))
    # Sort the rows based on the value in the first column
    sorted_errors = indexed_errors[indexed_errors[:, 0].argsort()[::-1]]
    
    return sorted_errors
"""
def convergeError(points, hull):
    errors = calculateError(points, hull)
    if(np.sum(errors[:,0]) < .5):
            return hull

    for gap in sorted_errors:
        bad_point = points[int(gap[1])]
        t = errors[:,0]
        if(np.sum(errors[:,0]) < .5):
            print('middle')
            return hull
        check = len(np.argwhere(hull == bad_point))
        if(len(np.argwhere(hull == bad_point)) > 0):
            x = points[:,0]
            temp_points = np.delete(points, np.argwhere(x == bad_point[0]), axis= 0)
            temp_hull = convex_hull(temp_points)
            if(len(temp_hull) > len(hull)):
                hull = converge(temp_points, temp_hull)
                return hull
            else:
                print('fail')
    return hull


"""
def converge_try(points, hull):
    errors = calculateError(points, hull)
    if(np.sum(errors[:,0]) < .5):
            return hull
    
    for gap in errors:
        x_worst_hull = hull[int(gap[1]), 0]

        x = points[:,0]
        temp_points = np.delete(points, np.argwhere(x == x_worst_hull), axis= 0)
        temp_hull = convex_hull(temp_points)
        if(True or len(temp_hull) >= len(hull)):
            hull = converge_try(temp_points, temp_hull)
            return hull
    
    return hull

def converge(points, hull):
    for p in hull:
        x = points[:,0]
        temp_points = np.delete(points, np.argwhere(x == p[0]), axis= 0)
        temp_hull = convex_hull(temp_points)
        if(len(temp_hull) > len(hull)):
            hull = converge(temp_points, temp_hull)
            return hull
        
    return hull

def quadratic_regression(points):
    """
    Perform quadratic regression to find coefficients for y = ax^2 + bx + c.
    
    Parameters:
        X (array-like): Independent variable values.
        Y (array-like): Dependent variable values.
        
    Returns:
        coefficients (numpy array): Coefficients [c, b, a] for the quadratic equation.
    """
    # Convert X and Y to numpy arrays
    
    X = points[:,0]
    Y = points[:,1]

    # Create the design matrix with columns [1, X, X^2]
    X_design = np.column_stack((np.ones(len(X)), X, X**2))
    
    # Calculate coefficients using the normal equation: (X^T * X)^(-1) * X^T * Y
    coefficients = np.linalg.inv(X_design.T @ X_design) @ X_design.T @ Y
    
    return coefficients   

def parabola_fit(x, a, b, c):
    return a * x**2 + b * x + c

def removePoint(points):
    # create vecor of R^2 terms
    c = quadratic_regression(points)
    errors = (points[1,:] - parabola_fit(points[0,:], c[2], c[1], c[0])) ** 2
    x = np.delete(x, np.argmax(errors))
    y = np.delete(y, np.argmax(errors))
    return x, y, (y - parabola_fit(x, c[2], c[1], c[0])) ** 2


#find line for all 
def step(coef):
    x = np.linspace(-6, 6, 20)
    y = coef[2]*x**2 + coef[1]*x  + coef[0]
    x, y = addNoise(x, y, 10)
    points = np.column_stack((x, y))

    hull = convex_hull(points)
    converge_hull = converge(points, hull)
    converge_error_hull = converge_try(points, hull)
    quadratic_error_remove = error_elimination(x, y)

    return converge_hull, converge_error_hull, quadratic_error_remove

# truth -> [c,b,a]
def simulation(truth, iterations):
    error = np.array([0.0,0.0,0.0])
    for i in range(0, iterations):
        sims = step(truth)
        for j, points in enumerate(sims):
            coefficients = quadratic_regression(points)
            if(all(abs(coefficients - truth) <= .05)):
                error[j] += 1

    return error / iterations


def show():
    x = np.linspace(-6, 6, 20)
    y = .1*x**2  - 1
    x, y = addNoise(x, y, 8)
    points = np.column_stack((x, y))

    hull = convex_hull(points)
    converge_hull = converge(points, hull)
    converge_error_hull = converge_try(points, hull)
    coefficients = quadratic_regression(converge_error_hull)
    quadratic_error_remove = error_elimination(x, y)


    X_fit = np.linspace(-6, 6, 80)
    Y_fit = coefficients[0] + coefficients[1] * X_fit + coefficients[2] * X_fit**2

    print(coefficients)
    # Plot the original data and the quadratic fit


    plt.figure(figsize=(12, 4))
    coefficients = quadratic_regression(quadratic_error_remove)
    X_q = np.linspace(min(x), max(x), 100)
    Y_q = coefficients[0] + coefficients[1] * X_q + coefficients[2] * X_q**2
    print(coefficients)
    # Plot the original data and the quadratic fit
    plt.subplot(131), plt.scatter(x, y, color='blue', label='Original Data'), plt.title("quadratic fit")
    plt.subplot(131), plt.plot(X_q, Y_q, color='red', label='Quadratic Fit') 
    plt.subplot(132), plt.scatter(points[:, 0], points[:, 1], label="Points", color="blue"), plt.title("converge hull")
    plt.subplot(132), plt.plot(np.append(converge_hull[:, 0], converge_hull[0, 0]), np.append(converge_hull[:, 1], converge_hull[0, 1]), label="Convex Hull", color="red")
    plt.subplot(133), plt.scatter(points[:, 0], points[:, 1], label="Points", color="green")
    plt.subplot(133), plt.scatter(converge_error_hull[:, 0], converge_error_hull[:, 1], label="Points", color="blue"), plt.title("combination hull")
    plt.subplot(133), plt.plot(X_fit, Y_fit, color='red', label='Quadratic Fit') 
    plt.show()

#show()
#error = simulation(np.array([-1,0,.1]), 1000)
#print(error)
#print(simulation(np.array([-1,0,.1]), 100))
import numpy as np
import matplotlib.pyplot as plt



# Define the coordinates of the five points
points = np.array([
    [9, 2],
    [2, 3],
    [6, 1],
    [8, 8],
    [2, 8],
    [1,1]
])
#def BMC(points):
# Set up the matrix and vector for the system of equations
M = []
b = []

for x, y in points:
    # Each row corresponds to one point substituted into the conic equation
    M.append([x**2, x*y, y**2, x, y, 1])
    b.append(0)  # Right-hand side is zero since all points satisfy the conic equation

M = np.array(M)
b = np.array(b)

# Solve for the coefficients (A, B, C, D, E, F)
# Since this is a homogeneous system (b = 0), we find a non-trivial solution
# using Singular Value Decomposition (SVD)
_, _, Vt = np.linalg.svd(M)
coefficients = Vt[-1]  # The last row of Vt gives the solution

# Extract coefficients
A, B, C, D, E, F = coefficients

print("Coefficients of the conic section:")
print(f"A = {A}, B = {B}, C = {C}, D = {D}, E = {E}, F = {F}")


x = np.linspace(0, 10, 500)
y = np.linspace(0, 10, 500)
X, Y = np.meshgrid(x, y)

# Evaluate the conic equation at each point in the grid
Z = A * X**2 + B * X * Y + C * Y**2 + D * X + E * Y + F

# Plot the conic section
plt.contour(X, Y, Z, levels=[0], colors='blue')
plt.scatter(points[:, 0], points[:, 1], color='blue', label='Points')
#plt.plot(points[:, 0], points[:, 1], color='red', label='Line')
#plt.plot(generated_points[:, 0], generated_points[:, 1], color='green', label='Correct_Line')
plt.title('Numpy Array of Points with Line')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.grid(True)
plt.show()

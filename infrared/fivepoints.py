import cv2
import numpy as np
import matplotlib.pyplot as plt

def run_convolution_on_image(img, x_m):
    # Extract a vertical slice of the image at x_m
    slice_img = img[:, int(x_m):int(x_m)+1]

    # Apply Sobel operator for vertical edge detection
    canny = cv2.Canny(slice_img, 100,200)

    #print(canny)
    # Threshold to detect edges (white pixels)
    #_, edge = cv2.threshold(np.abs(sobel_x), 50, 255, cv2.THRESH_BINARY)
    return canny

def coordinates_of_white_pixels_on_edge(edge, x_m):
    # Find coordinates of white pixels (non-zero values)
    coords = np.column_stack(np.where(edge > 0))
    coords[:, [0, 1]] = coords[:, [1, 0]]
    coords[:,0] = x_m
    return coords if len(coords) > 0 else None


def findpoints(x_0, x_n, points, img, left = False, right = False):
    # Base case: If we already have more than 5 points, return the points
    if (points is not None and len(points) > 3) or np.abs(x_0 - x_n) < 10:
        print('done')
        print(points)
        return points
    else:
        # Calculate the midpoint of the current range
        x_m = (x_0 + x_n) / 2

        # Apply Sobel operator to detect edges at x_m
        edge = run_convolution_on_image(img, x_m)  # Vertical Sobel convolution
        new_points = coordinates_of_white_pixels_on_edge(edge, x_m)  # Extract white pixel coordinates

        
        # Optimize search by skipping known blank areas
        if new_points is not None:
            if points is None:
                points =  new_points
                print('yay')
                print(points)
            else:
                points = np.concatenate((points, new_points), axis=0)
                print(points)
            # Search both left and right of the midpoint if edges are found
            p1 = findpoints(x_0, x_m, points, img, False, True)
            p2 = findpoints(x_m, x_n, points, img, True, False)
            points = np.concatenate((points, p1, p2), axis=0)
        elif left:
            # Continue searching only in the left half if no edges are found on the right
            p1 = findpoints(x_0, x_m, points, img, True, False)
            points = np.concatenate((points, p1), axis=0)
        elif right:
            # Continue searching only in the right half if no edges are found on the left
            p2 = findpoints(x_m, x_n, points, img,False, True)
            points = np.concatenate((points, p2), axis=0)
        else:
            # Default case: Search both halves if no specific direction is specified
            p1 =findpoints(x_0, x_m, points, img)
            p2 =findpoints(x_m, x_n, points, img)
            points = np.concatenate((points, p1, p2), axis=0)
        
        
        return points


img = cv2.imread('test_images/earth.jpg', 0)
binary_mask = np.where(img < 10, 255, 0).astype(np.uint8)



height, width = img.shape
print(f"Image dimensions: {width}x{height}")

points = findpoints(0, width, None, binary_mask)
print("points:")
print(points)


# Display the image using matplotlib
canny = cv2.Canny(binary_mask, 100, 200)

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


x = np.linspace(0, 1600, 500)
y = np.linspace(0, 1600, 500)
X, Y = np.meshgrid(x, y)

# Evaluate the conic equation at each point in the grid
Z = A * X**2 + B * X * Y + C * Y**2 + D * X + E * Y + F

# Plot the conic section

plt.figure(figsize=(10, 8))
plt.contour(X, Y, Z, levels=[0], colors='blue')
#plt.imshow(canny, cmap='gray')
plt.imshow(img, cmap='gray')
plt.scatter(points[:, 0], points[:, 1], color='blue')
plt.axis('off')
plt.title('Loaded Image')
plt.show()
#points = findpoints(0, len(img), [], img)
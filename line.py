import numpy as np
import cv2

def mark_longest_white_line(image_path, output_path):
    # Read the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Error: Unable to read the image '{image_path}'.")

    # Ensure the image is binary
    _, binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_length = 0
    longest_line = None

    for contour in contours:
        # Fit a line to the contour
        [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)

        if vx == 0:
            # Handle vertical line case
            length = binary.shape[0]  # Use image height for vertical lines
            longest_line = ((int(x), 0), (int(x), binary.shape[0] - 1).item())
        else:
            # Calculate the endpoints of the line
            lefty = int((-x * vy / vx) + y)
            righty = int(((binary.shape[1] - x) * vy / vx + y).item())

            # Calculate the length of the line using Euclidean distance
            length = np.sqrt(float((binary.shape[1] - 1)**2 + (righty - lefty)**2))
            longest_line = ((0, lefty), (binary.shape[1] - 1, righty))

        if length > max_length:
            max_length = length

    # Read the original image in color
    color_img = cv2.imread(image_path)
    if color_img is None:
        raise ValueError(f"Error: Unable to read the color image '{image_path}'.")

    # Draw the longest line in red if found
    if longest_line:
        cv2.line(color_img, longest_line[0], longest_line[1], (0, 0, 255), 2)
        # Save the output image
        cv2.imwrite(output_path, color_img)
        return f"Output image saved as '{output_path}'."
    else:
        return "No white lines found in the image."

# Test the function with a sample image
try:
    result = mark_longest_white_line('test_images/earth.jpg', 'test_images/output_image.png')
    print(result)
except Exception as e:
    print(f"Error: {e}")

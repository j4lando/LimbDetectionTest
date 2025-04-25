import cv2
import numpy as np
import matplotlib.pyplot as plt

def threshold_image(image_path, threshold):
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise FileNotFoundError(f"Unable to read the image at {image_path}")
    
    # Create a binary mask based on the threshold
    binary_mask = np.where(img < threshold, 255, 0).astype(np.uint8)
    
    return img, binary_mask

# Example usage
image_path = "test_images/earth_side.jpg"  # Replace with your image path
threshold_value = 10  # Adjust this value as needed

try:
    # Get the original and thresholded images
    original_image, thresholded_image = threshold_image(image_path, threshold_value)

    # Display the images using matplotlib
    plt.figure(figsize=(10, 5))

    # Original image
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_image, cmap="gray")
    plt.axis("off")

    # Thresholded image
    plt.subplot(1, 2, 2)
    plt.title("Thresholded Image")
    plt.imshow(thresholded_image, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please make sure the image file exists and the path is correct.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

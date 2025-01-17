import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def load_images_from_folder(folder_path):
    """
    Load all images from a specified folder into a list.
    
    Args:
        folder_path (str): Path to the folder containing images.
    
    Returns:
        images (list): List of loaded images as NumPy arrays.
    """
    images = []
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')  # Supported image formats
    
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(valid_extensions):  # Check for valid image extensions
            file_path = os.path.join(folder_path, file_name)
            image = cv2.imread(file_path)  # Read the image using OpenCV
            if image is not None:
                images.append(image)
    
    return images

# Example usage
folder_path = "test_images"  # Replace with your folder path
image_list = load_images_from_folder(folder_path)

# Print the number of loaded images
print(f"Loaded {len(image_list)} images from the folder.")


# Function to perform channel mixing
def channel_mix(image):
    original = image
    flat_mono_mix = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # R+G+B as grayscale
    red_channel = image[:, :, 2]
    two_r_minus_b = 2 * image[:, :, 2] - image[:, :, 0]  # 2*R - B
    r_plus_g_minus_b = image[:, :, 2] + image[:, :, 1] - image[:, :, 0]  # R + G - B

    return original, flat_mono_mix, red_channel, two_r_minus_b, r_plus_g_minus_b

# Process each image
results = []
for image in image_list:
    results.append(channel_mix(image))

# Display the results using matplotlib
fig, axes = plt.subplots(len(results), 5, figsize=(15, 10))

for i, (original, flat_mono_mix, red_channel, two_r_minus_b, r_plus_g_minus_b) in enumerate(results):
    axes[i, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[i, 0].set_title("Original")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(flat_mono_mix, cmap="gray")
    axes[i, 1].set_title("Flat Mono Mix")
    axes[i, 1].axis("off")

    axes[i, 2].imshow(red_channel, cmap="Reds")
    axes[i, 2].set_title("Red Channel")
    axes[i, 2].axis("off")

    axes[i, 3].imshow(two_r_minus_b, cmap="gray")
    axes[i, 3].set_title("2*R - B")
    axes[i, 3].axis("off")

    axes[i, 4].imshow(r_plus_g_minus_b, cmap="gray")
    axes[i, 4].set_title("R + G - B")
    axes[i, 4].axis("off")

plt.tight_layout()
plt.show()

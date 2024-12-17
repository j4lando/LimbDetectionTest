import matplotlib
matplotlib.use('TkAgg')  # Use an interactive backend

import numpy as np
import matplotlib.pyplot as plt
import cv2


# Step 1: Read the image and convert it to grayscale
def load_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image not found or invalid path.")
    return img

# Step 2: Perform FFT on the image
def fft_image(img):
    f = np.fft.fft2(img)  # Compute the 2D FFT
    fshift = np.fft.fftshift(f)  # Shift the zero-frequency component to the center
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)  # Magnitude Spectrum
    return fshift, magnitude_spectrum

# Step 3: Perform Inverse FFT
def ifft_image(fshift):
    f_ishift = np.fft.ifftshift(fshift)  # Undo the shift
    img_reconstructed = np.fft.ifft2(f_ishift)  # Compute the Inverse FFT
    img_reconstructed = np.abs(img_reconstructed)  # Take the magnitude
    return img_reconstructed

# Step 4: Display images
def display_images(original, magnitude_spectrum, reconstructed):
    plt.figure(figsize=(12, 6))

    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    # Magnitude Spectrum
    plt.subplot(1, 3, 2)
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('FFT Magnitude Spectrum')
    plt.axis('off')

    # Reconstructed image after IFFT
    plt.subplot(1, 3, 3)
    plt.imshow(reconstructed, cmap='gray')
    plt.title('Reconstructed Image (IFFT)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# Main Function
def main():
    image_path = 'test_images/women_with_hat.jpg'  # Replace with the path to your image
    original_img = load_image(image_path)

    # Perform FFT and IFFT
    fshift, magnitude_spectrum = fft_image(original_img)
    reconstructed_img = ifft_image(fshift)

    # Display results
    display_images(original_img, magnitude_spectrum, reconstructed_img)

if __name__ == "__main__":
    main()

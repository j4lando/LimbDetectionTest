import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.signal import fftconvolve

# Step 1: Create a Laplacian of Gaussian (LoG) kernel
def create_log_kernel(size=15, sigma=2.0):
    """Generates a Laplacian of Gaussian (LoG) kernel."""
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    gaussian = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    gaussian /= 2 * np.pi * sigma**2
    laplacian = (xx**2 + yy**2 - 2 * sigma**2) * gaussian / (sigma**4)
    log_kernel = laplacian - np.mean(laplacian)  # Ensure the kernel sums to zero
    return log_kernel

# Step 2: Perform FFT-based convolution
def fft_convolution(image, kernel):
    """Performs FFT-based convolution of an image with a kernel."""
    # Pad the kernel to match the image size
    padded_kernel = np.zeros_like(image)
    k_size_x, k_size_y = kernel.shape
    padded_kernel[:k_size_x, :k_size_y] = kernel
    
    # Center the kernel
    padded_kernel = np.fft.ifftshift(padded_kernel)
    
    # FFT of image and kernel
    fft_image = np.fft.fft2(image)
    fft_kernel = np.fft.fft2(padded_kernel)
    
    # Element-wise multiplication in frequency domain
    fft_result = fft_image * fft_kernel
    
    # Perform Inverse FFT
    result = np.fft.ifft2(fft_result)
    result = np.real(result)  # Take the real part
    return result

# Step 3: Display images
def display_results(original, log_kernel, filtered):
    plt.figure(figsize=(12, 6))

    # Original Image
    plt.subplot(1, 3, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    # Laplacian of Gaussian Kernel
    plt.subplot(1, 3, 2)
    plt.imshow(log_kernel, cmap='gray')
    plt.title('Laplacian of Gaussian Kernel')
    plt.axis('off')

    # Filtered Image (Edge Detection)
    plt.subplot(1, 3, 3)
    plt.imshow(filtered, cmap='gray')
    plt.title('Filtered Image (LoG via FFT)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# Main Function
def main():
    image_path = 'test_images/earth.jpg'  # Replace with the path to your image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError("Image not found or invalid path.")

    # Create a Laplacian of Gaussian kernel
    log_kernel = create_log_kernel(size=15, sigma=2.0)

    # Perform FFT-based convolution
    filtered_image = fft_convolution(image, log_kernel)

    # Display results
    display_results(image, log_kernel, filtered_image)

if __name__ == "__main__":
    main()

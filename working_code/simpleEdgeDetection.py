import cv2
import numpy as np
import matplotlib.pyplot as plt


# Read the image in grayscale
#path = 'test_images/BlackMarble2016_00066.png'
print('hello')
path = ['test_images/earth_side.jpg', 'test_images/earth_night.jpg',
        'test_images/piCam_530km_glare.png','test_images/piCam_530km.png',
        'test_images/BlackMarble2016_00066.png', 'test_images/earth.jpg']
image_list = []
canny_list = []
LoG_list = []
sobel_list = []
for p in path:
    img = cv2.imread(p, cv2.IMREAD_COLOR)
    image = cv2.imread(p, 0 )
    threshold = 10


    #binary_mask = np.where(image < threshold, 255, 0).astype(np.uint8)
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image, (3,3), 0)

    # Apply Laplacian of Gaussian
    log = cv2.Laplacian(blurred, cv2.CV_64F)

    # Perform Canny edge detection
    canny = cv2.Canny(image, 100, 200)

    # Find zero crossings
    zero_crossings = np.zeros_like(log)
    zero_crossings[:-1, :-1] = np.logical_and(log[:-1, :-1] < 0, log[1:, 1:] > 0) | \
                            np.logical_and(log[:-1, :-1] > 0, log[1:, 1:] < 0)

    # Apply thresholding to get strong edges
    threshold = 0.1 * np.max(np.abs(log))
    edges = np.zeros_like(zero_crossings, dtype=np.uint8)
    edges[np.abs(log) > threshold] = 255

    #coords = np.column_stack(np.where(canny[:, int(800):int(800)+1] > 0))
    #print(coords)
    #Sobel gaussian blus edge detection

    # Sobel edges (x and y gradients)
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(sobel_x**2 + sobel_y**2)

    # Thresholding (adjust based on image)
    _, sobel_edges = cv2.threshold(gradient.astype(np.uint8), 150, 255, cv2.THRESH_BINARY)

    image_list.append(image)
    canny_list.append(canny)
    LoG_list.append(edges)
    sobel_list.append(sobel_edges)


plt.figure(figsize=(14, 8))
len = len(image_list)
for i in range(len):
    image = image_list[i]
    canny = canny_list[i]
    edges = LoG_list[i]
    sobel_edges = sobel_list[i]

    plt.subplot(len, 4, i*4 +1), plt.imshow(image, cmap='gray'), plt.title('Original Image')
    plt.subplot(len, 4, i*4 +2), plt.imshow(canny, cmap='gray'), plt.title('Canny')
    plt.subplot(len, 4, i*4 +3), plt.imshow(edges, cmap='gray'), plt.title('Laplacian of Gaussian')
    plt.subplot(len, 4, i*4 +4), plt.imshow(sobel_edges, cmap='gray'), plt.title('Sobel')

plt.tight_layout()
plt.show()

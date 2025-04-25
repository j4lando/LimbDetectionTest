import numpy as np
import cv2

def extract_largest_value(edge: np.ndarray, x_m: int):
    largest_number = np.max(edge)
    print(largest_number)
    return np.array([x_m, largest_number])

example_array = np.array([[103.8], [2.0]])
x_m_value = 10
result = extract_largest_value(example_array, x_m_value)
print(result)  # Output: [10, 103]

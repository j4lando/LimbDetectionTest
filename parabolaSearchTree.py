import numpy as np

# Create a numpy array of random values
random_values = np.random.rand(10, 1)  # 10 rows, 1 column
# Add a column with the index of each row
print(np.shape(random_values))
print(random_values)
indexed_array = np.hstack((random_values, np.arange(len(random_values)).reshape(-1, 1)))
# Sort the rows based on the value in the first column
sorted_array = indexed_array[indexed_array[:, 0].argsort()]

print(sorted_array)
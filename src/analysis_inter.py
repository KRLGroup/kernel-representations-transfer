import os
import pickle
import clize
import numpy as np
from itertools import combinations

from tqdm import tqdm

from dataset import load_dataset


def calculate_min_distances_sum(X, task_y_data, labels=None):
    # Calculate pairwise squared Euclidean distances between task Y data and data in tasks X
    distances_y_to_x = np.zeros((task_y_data.shape[0], X.shape[0]))
    for i in range(task_y_data.shape[0]):
        for j in range(X.shape[0]):
            distances_y_to_x[i, j] = np.sum((task_y_data[i] - X[j]) ** 2)

    # Find the minimum distances for each data point in task Y
    min_distances = np.min(distances_y_to_x, axis=1)

    # Calculate and return the sum of the minimum distances
    min_distances_sum = np.sum(min_distances)
    return min_distances_sum


def main(dataset1, dataset2, kernel, out_path):
    # config, automata, formulas, _, kernel_reprs = ds
    kernels_1 = load_dataset(os.path.join("datasets", dataset1), kernel=kernel)[4]
    kernels_2 = load_dataset(os.path.join("datasets", dataset2), kernel=kernel)[4]

    kernels_1 = np.concatenate(kernels_1)

    sums = []
    for i, k2 in enumerate(tqdm(kernels_2)):
        s = calculate_min_distances_sum(kernels_1, k2)
        sums.append(s)
    
    mean = np.mean(sums)
    std = np.std(sums)

    print(f"Sum of min distances: {mean} +- {std}")

    with open(out_path, "wb") as f:
        pickle.dump(sums, f)



if __name__ == '__main__':
    clize.run(main)


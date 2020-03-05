import os
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
# from itertools import zip_longest

# def grouper(iterable, n, fillvalue=None):
#     "Collect data into fixed-length chunks or blocks"
#     # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
#     args = [iter(iterable)] * n
#     return zip_longest(*args, fillvalue=fillvalue)

def load_points_from_file(filename):
    """Loads 2d points from a csv called filename
    Args:
        filename : Path to .csv file
    Returns:
        (xs, ys) where xs and ys are a numpy array of the co-ordinates.
    """
    points = pd.read_csv(filename, header=None)
    return points[0].values, points[1].values

def view_data_segments(xs, ys):
    """Visualises the input file with each segment plotted in a different colour.
    Args:
        xs : List/array-like of x co-ordinates.
        ys : List/array-like of y co-ordinates.
    Returns:
        None
    """
    assert len(xs) == len(ys)
    assert len(xs) % 20 == 0
    len_data = len(xs)
    num_segments = len_data // 20
    colour = np.concatenate([[i] * 20 for i in range(num_segments)])
    plt.scatter(xs, ys, c=colour)

def least_squares_linear(x, y):
    # extend the first column with 1s
    ones = np.ones(x.shape)
    x_e = np.column_stack((ones, x))
    v = np.linalg.inv(x_e.T.dot(x_e)).dot(x_e.T).dot(y)
    return v[0], v[1]

def reconstruct_linear_line(xs, ys, a, b):
    x_1r = xs.min()
    x_2r = xs.max()
    y_1r = a + b * x_1r
    y_2r = a + b * x_2r

    return x_1r, x_2r, y_1r, y_2r

def square_error(y, y_hat):
    return np.sum((y - y_hat) ** 2)

def main():
    # Grabs filename from command line argument and saves points to variables
    csv_file = sys.argv[1]
    x_coordinates, y_coordinates = load_points_from_file(csv_file)
    total_reconstructed_error = 0

    # Splits x and y coordinate lists into equal length segments
    x_segments = [x_coordinates[i:i + 20] for i in range(0, len(x_coordinates), 20)]
    y_segments = [y_coordinates[i:i + 20] for i in range(0, len(y_coordinates), 20)]

    # Logical statement for optional '--plot' command line argument
    if len(sys.argv) == 3 and sys.argv[2] == '--plot':
        view_data_segments(x_coordinates, y_coordinates)

        # Plots reconstructed line segments, of length 20 data points
        for i, j in zip(x_segments, y_segments):
            a_1, b_1 = least_squares_linear(i, j)

            # Calculates error for given segment and adds to total reconstructed error
            y_hat = a_1 + b_1 * i
            error = square_error(j, y_hat)
            print(error)
            total_reconstructed_error += error

            line_data = reconstruct_linear_line(i, j, a_1, b_1)
            plt.plot([line_data[0], line_data[1]], [line_data[2], line_data[3]], 'r-', lw=4)

        print(total_reconstructed_error)
        plt.show()
        pass
    else:
        pass

main()

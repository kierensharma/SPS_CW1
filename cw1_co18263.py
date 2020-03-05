import os
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def load_points_from_file(filename):
    """Loads 2d points from a csv called filename
    Args:
        filename : Path to .csv file
    Returns:
        (xs, ys) where xs and ys are a numpy array of the co-ordinates.
    """
    points = pd.read_csv(filename, header=None)
    return points[0].values, points[1].values

def view_data_segments(xs, ys, a, b):
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
    fig, ax = plt.subplots()
    ax.scatter(xs, ys, c=colour)

    # Plots reconstructed line
    xs_1r = xs.min()
    xs_2r = ys.max()
    ys_1r = a + b * xs_1r
    ys_2r = a + b * xs_2r

    ax.plot([xs_1r, xs_2r], [ys_1r, ys_2r], 'r-', lw=4)

    plt.show()

def least_squares(x, y):
    # extend the first column with 1s
    ones = np.ones(x.shape)
    x_e = np.column_stack((ones, x))
    v = np.linalg.inv(x_e.T.dot(x_e)).dot(x_e.T).dot(y)
    return v[0], v[1]

# def reconstructed_linear_line(xs, ys, a, b):
#     xs_1r = xs.min()
#     xs_2r = xy.max()
#     ys_1r = a + b * xs_1r
#     ys_2r = a + b * xs_2r
#
#     return xs_1r, xs_2r, ys_1r, ys_2r

# Grabs filename from command line argument and saves points to variables
csv_file = sys.argv[1]
x_coordinates, y_coordinates = load_points_from_file(csv_file)

# Logical statement for optional '--plot' command line argument
if len(sys.argv) == 3 and sys.argv[2] == '--plot':
    a_1, b_1 = least_squares(x_coordinates, y_coordinates)
    view_data_segments(x_coordinates, y_coordinates, a_1, b_1)
    pass
else:
    pass

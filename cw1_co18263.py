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
    # Extend the first column with 1s
    ones = np.ones(x.shape)
    x_e = np.column_stack((ones, x))
    A = np.linalg.inv(x_e.T.dot(x_e)).dot(x_e.T).dot(y)

    # Calculates error in regression line
    y_hat = A[0] + A[1] * x
    error = np.sum((y - y_hat) ** 2)

    return A[0], A[1], error

def reconstruct_linear_line(x, y, a, b):
    x_1r = x.min()
    x_2r = x.max()
    y_1r = a + b * x_1r
    y_2r = a + b * x_2r

    return x_1r, x_2r, y_1r, y_2r

def least_squares_quadratic(x, y):
    # extend the first column with 1s
    ones = np.ones(x.shape)
    x_squared = np.square(x)

    x_e = np.column_stack((ones, x, x_squared))
    A = np.linalg.inv(x_e.T.dot(x_e)).dot(x_e.T).dot(y)

    # calculates error in regression line
    y_hat = A[0] + A[1] * x + A[2] * np.square(x)
    error = np.sum((y - y_hat) ** 2)

    return A[0], A[1], A[2], error

def reconstruct_quadratic_line(x, y, a, b1, b2):
    y_r = a + b1 * x + b2 *np.square(x)

    return y_r

def main():
    # Grabs filename from command line argument and saves points to variables
    csv_file = sys.argv[1]
    x_coordinates, y_coordinates = load_points_from_file(csv_file)
    error_list = []
    total_reconstructed_linear_error = 0
    total_reconstructed_quadratic_error = 0

    # Splits x and y coordinate lists into equal length segments
    x_segments = [x_coordinates[i:i + 20] for i in range(0, len(x_coordinates), 20)]
    y_segments = [y_coordinates[i:i + 20] for i in range(0, len(y_coordinates), 20)]

    # Plots reconstructed line segments, of length 20 data points
    for i, j in zip(x_segments, y_segments):
        a_1, b_1, error = least_squares_linear(i, j)

        # Adds error to total reconstructed error for linear function
        total_reconstructed_linear_error += error

    error_list.append(total_reconstructed_linear_error)

    for i, j in zip(x_segments, y_segments):
        a_1, b_1, b_2, error = least_squares_quadratic(i, j)

        # Adds error to total reconstructed error for quadratic function
        total_reconstructed_quadratic_error += error

    error_list.append(total_reconstructed_quadratic_error)

    # Finds smallest value in error list for all line types to determine function
    smallest_error = min(error_list)
    print(smallest_error)
    function_type = error_list.index(smallest_error)

    # Logical statement for optional '--plot' command line argument
    if len(sys.argv) == 3 and sys.argv[2] == '--plot':
        view_data_segments(x_coordinates, y_coordinates)

        if function_type == 0:
            for i, j in zip(x_segments, y_segments):
                a_1, b_1, error = least_squares_linear(i, j)

                line_data = reconstruct_linear_line(i, j, a_1, b_1)
                plt.plot([line_data[0], line_data[1]], [line_data[2], line_data[3]], 'r-', lw=4)

        elif function_type == 1:
            for i, j in zip(x_segments, y_segments):
                a_1, b_1, b_2, error = least_squares_quadratic(i, j)

                new_y = reconstruct_quadratic_line(i, j, a_1, b_1, b_2)
                plt.plot(i, new_y, 'r-', lw=4)

        plt.show()
        pass

    else:
        pass

main()

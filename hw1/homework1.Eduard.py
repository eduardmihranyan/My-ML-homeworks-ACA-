#!/usr/bin/env python3
"""
Homework 1 of Machine Learning Course

Implement fit_linear_regression function.
Implement plot_fitted_line.

Example Usage:
    python3 homework1.py --user arsen

Make sure to test before submiting:
    python3 homework1.py test
"""
from __future__ import print_function
import argparse
import getpass
import hashlib
import matplotlib.pyplot as plt
import numpy as np


def parse_args(*argument_array):
    parser = argparse.ArgumentParser()
    parser.set_defaults(function=main)
    parser.add_argument('--user', default=getpass.getuser(),
                        help='Override system username with something else to '
                             'be include in the output file.')
    subs = parser.add_subparsers()
    test_parser = subs.add_parser('test')
    test_parser.set_defaults(function=test_function_signatures)
    args = parser.parse_args(*argument_array)
    return args


def test_function_signatures(args):
    beta = fit_linear_regression(np.array([[1, 2], [1, 3]]), np.array([2, 3]))
    # Make sure that fit_linear_regression returns a np.array
    assert type(beta) == np.ndarray
    # Make sure that fit_linear_regression returns beta of correct length
    assert beta.shape == (2,)
    # Make sure that fit_linear_regression returns beta of correct value
    assert np.abs(beta[0]) < 1e-10
    assert np.abs(beta[1] - 1) < 1e-10


def fit_linear_regression(data_matrix, response_vector):
    X = data_matrix
    Y = response_vector
    beta= np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
    return beta


def plot_fitted_line(b0, b1, X, Y, username):
    """
    :param b0: Intersept of line to plot
    :param b1: Slope of the line to plot
    :param X: An array of length N containing x coordinates of points
    :param Y: An array of length N containing y coordinates of points
    """
    # TODO: Write the plotting code.
     
    plt.plot(X,Y,'bo')
    plt.plot(X, b0+b1*X, color='red')
    plt.title('user: Eduard'.format(username))
    plt.savefig('homework1.Eduard.png'.format(username), dpi=320)


def main(args):
    username_based_variance = 1 + 0.2 * (int(
        hashlib.sha1(args.user.encode('utf-8')).hexdigest(), 16) % 10)
    # Generate synthetic data
    beta_0 = 15
    beta_1 = 0.2
    # Generate 100 points uniform randomly in [-20, 80] interval
    X1 = np.random.uniform(-20, 80, 100)
    # Generate corresponding y_i normally distributed around b_0 + b_1 * x line
    Y = [np.random.normal(beta_0 + beta_1 * x,
                          np.sqrt(username_based_variance)) for x in X1]
    # Convert X1 into a matrix
    X = np.array([(1, x) for x in X1])
    # Get estimates for beta_0 and beta_1
    b0_hat, b1_hat = fit_linear_regression(X, Y)
    plot_fitted_line(b0_hat, b1_hat, X1, Y, args.user)


if __name__ == '__main__':
    args = parse_args()
    args.function(args)

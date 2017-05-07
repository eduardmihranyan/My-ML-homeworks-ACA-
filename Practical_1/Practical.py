#!/usr/bin/env python3
"""
Run regression on apartment data.
"""
from __future__ import print_function
import argparse
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import getpass


def parse_args(*argument_array):
    parser = argparse.ArgumentParser()
    parser.set_defaults(function=main)
    parser.add_argument('--user', default=getpass.getuser(),
                        help='Override system username with something else to '
                             'be include in the output file.')
    subs = parser.add_subparsers()
    test_parser = subs.add_parser('test')
    test_parser.set_defaults(function=test_function_signatures)
    parser.add_argument('--csv', default='yerevan_april_9.csv.gz',
                        help='CSV file with the apartment data.')
    args = parser.parse_args(*argument_array)
    return args


def featurize(apartment):
    """
    :param apartment: Apartment DataFrame row (a dictionary like object)
    :return: (x, y) tuple, where x is a numpy vector, and y is a number
    """
    dist=['Center' 'Arabkir' 'Shengavit' 'Avan' 'Malatia-Sebastia' 'Qanaqer-Zeytun'
    'Nor Norq' 'Achapnyak' 'Davtashen' 'Erebuni' 'Norq Marash' 'Nubarashen'
    'Vahagni district']
    build=['panel' 'monolit' 'other' 'stone']

    fi= np.ones(26)
    fi[1]=apartment["area"]
    fi[2]=apartment["num_rooms"]
    fi[3]=apartment["ceiling_height"]
    fi[4]=apartment["num_bathrooms"]
    fi[5]=fi[6]=fi[7]=0
    if apartment["condition"]=="good":
        fi[5] = 1
    elif apartment["condition"]=="newly repaired":
        fi[6]=1
    else:
        fi[7]=1
    for i in range(len(dist)):
        if apartment["district"]==dist[i]:
            fi[i+8]=1
        else:
            fi[i+8]=0
    for i in range(len(build)):
        if apartment["building_type"]==build[i]:
            fi[i+21]=1
        else:
            fi[i+21]=0
    fi[25]=apartment["max_floor"]

    return fi,apartment["price"]



def poly_featurize(apartment, degree=2):
    """
    :param apartment: Apartment DataFrame row (a dictionary like object)
    :return: (x, y) tuple, where x is a numpy vector, and y is a number
    """
    x, y = featurize(apartment)
    poly_x = x
    return poly_x, y


def fit_ridge_regression(X, Y, l=0.1):
    """
    :param X: A numpy matrix, where each row is a data element (X)
    :param Y: A numpy vector of responses for each of the rows (y)
    :param l: ridge variable
    :return: A vector containing the hyperplane equation (beta)
    """
    D = X.shape[1]  # dimension + 1
    beta = np.linalg.inv(X.T.dot(X) + l * np.identity(D)).dot(X.T).dot(Y)
    beta = np.reshape(beta, beta.shape[0])
    return beta


def cross_validate(X, Y, fitter, folds=5):
    """
    :param X: A numpy matrix, where each row is a data element (X)
    :param Y: A numpy vector of responses for each of the rows (y)
    :param fitter: A function that takes X, Y as parameters and returns beta
    :param folds: number of cross validation folds (parts)
    :return: list of corss-validation scores
    """
    scores = []
    for i in range(folds):
        # TODO: train on the rest
        # TODO: Add corresponding score to scores
        test_X = X[i::folds]
        test_Y = Y[i::folds]
        train_X = np.array([X[j] for j in range(len(X)) if (j % folds) != i])
        train_Y = np.array([Y[j] for j in range(len(Y)) if (j % folds) != i])
        beta = fitter(train_X, train_Y)
        err = test_Y - test_X.dot(beta.reshape(len(beta),1))


        scores.append(np.sqrt(err.T.dot(err) / len(test_Y)))
    return scores


def my_featurize(apartment):
    """
    This is the function we will use for scoring your implmentation.
    :param apartment: apartment row
    :return: (x, y) pair where x is feature vector, y is the response variable.

    """
    return featurize(apartment)


def my_beta():
    """
    :return: beta_hat that you estimate.
    """
    return np.array([-1.40715056e+01, 1.45043185e+00, -1.43152008e+01, 7.15961455e+01,
   1.98853809e+01,  -7.28438671e+00,   8.33862711e+00,  -1.51257458e+01,
   0.00000000e+00,  -1.40715056e+01,  -1.40715056e+01,  -1.40715056e+01,
  -1.40715056e+01,  -1.40715056e+01,  -1.40715056e+01,  -1.40715056e+01,
  -1.40715056e+01,  -1.40715056e+01,  -1.40715056e+01,  -1.40715056e+01,
  -1.40715056e+01,   0.00000000e+00,  -1.40715056e+01,  -1.40715056e+01,
  -1.40715056e+01,  -1.27997209e-02])


def main(args):
    df = pd.read_csv(args.csv)
    df["price"]/=1000
    # TODO: Convert `df` into features (X) and responses (Y) using featurize
    X=np.concatenate([[featurize(x[1])[0] for x in df.iterrows()]],axis=0)
    Y=np.array([[featurize(x[1])[1] for x in df.iterrows()]]).T
    beta = fit_ridge_regression(X, Y, l=0.1)

    # TODO you should probably create another function to pass to `cross_validate`
    scores = cross_validate(X, Y, fit_ridge_regression)
    #print(scores)
    #print(np.mean(scores))




def test_function_signatures(args):
    apt = pd.Series({'price': 65000.0, 'condition': 'good', 'district': 'Center', 'max_floor': 9, 'street': 'Vardanants St', 'num_rooms': 3, 'region': 'Yerevan', 'area': 80.0, 'url': 'http://www.myrealty.am/en/item/24032/3-senyakanoc-bnakaran-vacharq-Yerevan-Center', 'num_bathrooms': 1, 'building_type': 'panel', 'floor': 4, 'ceiling_height': 2.7999999999999998})  # noqa
    x, y = my_featurize(apt)
    beta = my_beta()

    assert type(y) == float
    assert len(x.shape) == 1
    assert x.shape == beta.shape

if __name__ == '__main__':
    args = parse_args()
    args.function(args)

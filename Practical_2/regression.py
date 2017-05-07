import numpy as np


def fit_ridge_regression(X, Y, l):
    """
    :param X: data matrix (2 dimensional np.array)
    :param Y: response variables (1 dimensional np.array)
    :param l: regularization parameter lambda
    :return: value of beta (1 dimensional np.array)
    """
    # TODO: Implement fit_ridge_regression (same as previous homeworks)
    D = X.shape[1]
    beta = np.linalg.inv(X.T.dot(X) + l * np.identity(D)).dot(X.T).dot(Y)
    return beta


def gradient_descent(X, Y, epsilon=1e-6, l=1, step_size=1e-4, max_steps=1000):
    """
    Implement gradient descent using full value of the gradient.
    :param X: data matrix (2 dimensional np.array)
    :param Y: response variables (1 dimensional np.array)
    :param l: regularization parameter lambda
    :param epsilon: approximation strength
    :param max_steps: maximum number of iterations before algorithm will
        terminate.
    :return: value of beta (1 dimensional np.array)
    """
    X_norm=np.copy(X)
    X_mean = np.mean(X,axis=0)
    X_std=np.std(X,axis=0)
    #print(X_std)
    for i in range(1,X.shape[1]):
        X_norm[:,i]=(X_norm[:,i]-X_mean[i])/X_std[i]
    X_var=np.var(X,axis=0)


    beta = np.zeros(X.shape[1])
    for s in range(max_steps):
        # TODO: Implement iterations.
        grad_beta= normalized_gradient(X_norm, Y, beta, l)
        if np.linalg.norm(step_size*grad_beta)<epsilon:

            break
        else:
            beta= beta - step_size*grad_beta
        pass
    beta[0] = beta[0] - sum(X_mean[j] * beta[j] / X_std[j] for j in range(1, X.shape[1]))
    for i in range(1, X.shape[1]):
        beta[i] = beta[i] / X_std[i]

    return beta


def normalized_gradient(X, Y, beta, l):
    """
    :param X: data matrix (2 dimensional np.array)
    :param Y: response variables (1 dimensional np.array)
    :param beta: value of beta (1 dimensional np.array)
    :param l: regularization parameter lambda
    :return: normalized gradient, i.e. gradient normalized according to data
    """
    # TODO: Implement

    return (-2)*((X.T.dot( Y-(X.dot(np.array(beta))))-l*beta )/(X.shape[0])  )


def stochastic_gradient_descent(X, Y, epsilon=0.0001, l=1, step_size=0.01,
                                max_steps=1000):
    """
    Implement gradient descent using stochastic approximation of the gradient.
    :param X: data matrix (2 dimensional np.array)
    :param Y: response variables (1 dimensional np.array)
    :param l: regularization parameter lambda
    :param epsilon: approximation strength
    :param max_steps: maximum number of iterations before algorithm will
        terminate.
    :return: value of beta (1 dimensional np.array)
    """
    X_norm=np.copy(X)
    X_mean = np.mean(X,axis=0)
    X_std=np.std(X,axis=0)
    #print(X_std)
    for i in range(1,X.shape[1]):
        X_norm[:,i]=(X_norm[:,i]-X_mean[i])/X_std[i]
    X_var=np.var(X,axis=0)

    beta = np.ones(X.shape[1])
    for s in range(max_steps):
        # TODO: Implement iterations.
        i=s%X.shape[0]
        grad_stoch= normalized_gradient(X_norm[i].reshape((1,X.shape[1])),Y[i],beta,l)
        if np.linalg.norm(step_size * grad_stoch) < epsilon:
            break
        else:
            beta=beta-step_size*grad_stoch
        pass

    beta[0] = beta[0] - sum(X_mean[j] * beta[j] / X_std[j] for j in range(1, X.shape[1]))
    for i in range(1, X.shape[1]):
        beta[i] = beta[i] / X_std[i]

    return beta

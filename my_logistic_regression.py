# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    my_logistic_regression.py                          :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dsindres <dsindres@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/09/04 11:35:04 by dsindres          #+#    #+#              #
#    Updated: 2025/09/24 15:18:31 by dsindres         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np

class MyLogisticRegression():
    def __init__(self, thetas, alpha=0.001, max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter
        if not isinstance(thetas, np.ndarray):
            self.thetas = np.array(thetas, dtype=float)
        else:
            self.thetas = thetas.astype(float)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit_(self, X, Y):
        m = len(Y)
        X_ = np.c_[np.ones((m, 1)), X]

        for _ in range(self.max_iter):
            y_hat = self.sigmoid(X_.dot(self.thetas))
            gradient = (1 / m) * X_.T.dot(y_hat - Y)
            self.thetas -= self.alpha * gradient
        return self.thetas

    def standardize(self, X, mean=None, std=None):
        if not isinstance(X, np.ndarray):
            return None
        if X.size == 0:
            return None

        if mean is None or std is None:
            mean = X.mean(axis=0)
            std = X.std(axis=0)

        self.mean_ = mean
        self.std_ = std

        new_X = (X - mean) / std
        return new_X
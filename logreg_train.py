# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    logreg_train.py                                    :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dsindres <dsindres@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/09/24 13:46:08 by dsindres          #+#    #+#              #
#    Updated: 2025/09/24 14:49:55 by dsindres         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import sys
import pandas as pd
import numpy as np
from my_logistic_regression import MyLogisticRegression as MyLR

def main():
    data = pd.read_csv(sys.argv[1])

    # Features choisies et NaN remplacés par la moyenne
    X = data[["Astronomy", "Herbology"]].copy().fillna(data[["Astronomy","Herbology"]].mean())
    X = np.array(X)

    # Standardisation
    X_norm = (X - X.mean(axis=0)) / X.std(axis=0)

    houses = data["Hogwarts House"].unique()
    all_thetas = {}

    # One-vs-all pour chaque maison
    for house in houses:
        Y_binary = (data["Hogwarts House"] == house).astype(int).values.reshape(-1, 1)
        thetas_init = np.zeros((X_norm.shape[1]+1, 1))
        model = MyLR(thetas=thetas_init, alpha=0.01, max_iter=50000)
        trained_thetas = model.fit_(X_norm, Y_binary)
        all_thetas[house] = trained_thetas.flatten()

    # Construire CSV
    dict_thetas = {}
    for house, thetas in all_thetas.items():
        for i, theta in enumerate(thetas):
            dict_thetas[f"{house}_theta_{i}"] = [float(theta)]

    df_thetas = pd.DataFrame(dict_thetas)
    df_thetas.to_csv("thetas.csv", index=False)
    print("Training finished. Thetas saved to 'thetas.csv'.")

if __name__ == "__main__":
    main()


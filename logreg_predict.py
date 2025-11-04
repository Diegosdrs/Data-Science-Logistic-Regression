# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    logreg_predict.py                                  :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dsindres <dsindres@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/09/24 14:36:51 by dsindres          #+#    #+#              #
#    Updated: 2025/11/04 13:36:42 by dsindres         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from my_logistic_regression import MyLogisticRegression as MyLR

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def main():
    if len(sys.argv) < 2:
        print("Erreur: mauvais nombre d'argument")
        sys.exit(1)
    X_test = pd.read_csv(sys.argv[1])
    X_features = X_test[["Astronomy", "Herbology"]].copy().fillna(X_test[["Astronomy","Herbology"]].mean())
    X = np.array(X_features)

    # Standardisation (centrée-réduite) du test
    X_norm = (X - X.mean(axis=0)) / X.std(axis=0)

    file_theta = Path("./thetas.csv")
    if not file_theta.exists():
        print("Erreur: pas de fichier thetas trouve")
        sys.exit(1)

    thetas_df = pd.read_csv("thetas.csv")

    # Récupérer les maisons
    houses = [col.split("_theta_0")[0] for col in thetas_df.columns if "_theta_0" in col]
    probs = np.zeros((X_norm.shape[0], len(houses)))

    for j, house in enumerate(houses):
        theta_cols = [col for col in thetas_df.columns if col.startswith(house)]
        theta_values = thetas_df[theta_cols].values.flatten().reshape(-1, 1)

        X_ = np.c_[np.ones((X_norm.shape[0], 1)), X_norm]
        y_hat = sigmoid(X_.dot(theta_values))
        probs[:, j] = y_hat.flatten()

    # Maison avec probabilité maximale
    predicted_indices = np.argmax(probs, axis=1)
    predicted_houses = [houses[i] for i in predicted_indices]

    df_pred = pd.DataFrame({
        "Index": np.arange(len(predicted_houses)),
        "Hogwarts House": predicted_houses
    })
    df_pred.to_csv("houses.csv", index=False)
    print("Predictions saved to 'houses.csv'.")

if __name__ == "__main__":
    main()


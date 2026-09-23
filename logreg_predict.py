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
    if len(sys.argv) < 3:
        print("Erreur: mauvais nombre d'argument")
        print("Usage: python3 logreg_predict.py <dataset_test.csv> <thetas.csv>")
        sys.exit(1)

    file_theta = Path(sys.argv[2])
    if not file_theta.exists():
        print("Erreur: pas de fichier thetas trouve")
        sys.exit(1)

    thetas_df = pd.read_csv(sys.argv[2])

    features = ["Arithmancy", "Astronomy", "Herbology", "Defense Against the Dark Arts",
                "Divination", "Muggle Studies", "Ancient Runes", "History of Magic",
                "Transfiguration", "Potions", "Care of Magical Creatures", "Charms", "Flying"]

    X_test = pd.read_csv(sys.argv[1])
    X_features = X_test[features].copy().fillna(X_test[features].mean())
    X = np.array(X_features)

    # Standardisation avec la moyenne/ecart-type sauvegardes par logreg_train.py 
    mean_train = np.array([thetas_df[f"mean_{f}"].iloc[0] for f in features])
    std_train = np.array([thetas_df[f"std_{f}"].iloc[0] for f in features])

    scaler = MyLR(thetas=np.zeros((X.shape[1] + 1, 1)))
    X_norm = scaler.standardize(X, mean=mean_train, std=std_train)

    # recuperer les maisons
    houses = [col.split("_theta_0")[0] for col in thetas_df.columns if "_theta_0" in col]
    probs = np.zeros((X_norm.shape[0], len(houses)))

    for j, house in enumerate(houses):
        theta_cols = [col for col in thetas_df.columns if col.startswith(f"{house}_theta")]
        theta_values = thetas_df[theta_cols].values.flatten().reshape(-1, 1)

        X_ = np.c_[np.ones((X_norm.shape[0], 1)), X_norm]
        # trouver la prediction
        y_hat = sigmoid(X_.dot(theta_values))
        probs[:, j] = y_hat.flatten()

    # trouver la proba maaaaax
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
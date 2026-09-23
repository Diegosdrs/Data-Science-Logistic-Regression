# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    logreg_train.py                                    :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dsindres <dsindres@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/09/24 13:46:08 by dsindres          #+#    #+#              #
#    Updated: 2025/11/04 13:30:03 by dsindres         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import sys
import pandas as pd
import numpy as np
from my_logistic_regression import MyLogisticRegression as MyLR

def main():
    if len(sys.argv) < 2:
        print("Erreur: mauvais nombre d'argument")
        sys.exit(1)
        
    data = pd.read_csv(sys.argv[1])

    features = ["Arithmancy", "Astronomy", "Herbology", "Defense Against the Dark Arts",
                "Divination", "Muggle Studies", "Ancient Runes", "History of Magic",
                "Transfiguration", "Potions", "Care of Magical Creatures", "Charms", "Flying"]

    # Features choisies et NaN remplacés par la moyenne
    X = data[features].copy().fillna(data[features].mean())
    X = np.array(X)

    # calcul moyenne et ecart-type pour predict
    scaler = MyLR(thetas=np.zeros((X.shape[1] + 1, 1)))
    X_norm = scaler.standardize(X)
    mean_train = scaler.mean_
    std_train = scaler.std_

    houses = data["Hogwarts House"].unique()
    all_thetas = {}

    # One-vs-all pour chaque maison
    for house in houses:
        Y_binary = (data["Hogwarts House"] == house).astype(int).values.reshape(-1, 1) # 1 si appartient sinon 0
        thetas_init = np.zeros((X_norm.shape[1]+1, 1))
        model = MyLR(thetas=thetas_init, alpha=0.01, max_iter=50000)
        trained_thetas = model.fit_(X_norm, Y_binary)
        all_thetas[house] = trained_thetas.flatten()

    # construire CSV
    dict_thetas = {}
    for house, thetas in all_thetas.items():
        for i, theta in enumerate(thetas):
            dict_thetas[f"{house}_theta_{i}"] = [float(theta)]

    # on sauvegarde la moyenne et l'ecart-type du train, pour que
    # logreg_predict standardisse le jeu de test de la meme facon
    for i, feature in enumerate(features):
        dict_thetas[f"mean_{feature}"] = [float(mean_train[i])]
        dict_thetas[f"std_{feature}"] = [float(std_train[i])]

    df_thetas = pd.DataFrame(dict_thetas)
    df_thetas.to_csv("thetas.csv", index=False)
    print("Training finished. Thetas saved to 'thetas.csv'.")

if __name__ == "__main__":
    main()
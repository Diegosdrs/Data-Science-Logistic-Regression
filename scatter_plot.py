# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    scatter_plot.py                                    :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dsindres <dsindres@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/09/24 13:14:28 by dsindres          #+#    #+#              #
#    Updated: 2025/11/04 14:04:25 by dsindres         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

def plot_scatter(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if 'Index' in numeric_cols:
        numeric_cols.remove('Index')
    
    n_features = len(numeric_cols)

    # On calcule la correlation de TOUTES les paires possibles, puis on les
    # trie par correlation absolue decroissante. Ainsi, les paires les plus
    # "similaires" (fortement correlees, positivement ou negativement) sont
    # toujours affichees en premier, peu importe l'ordre des colonnes dans
    # le dataset (avant, seules les 20 premieres paires rencontrees dans
    # l'ordre du CSV etaient montrees, ce qui pouvait cacher la vraie reponse)
    pairs = []
    for i in range(n_features):
        for j in range(i + 1, n_features):
            data_clean = df[[numeric_cols[i], numeric_cols[j]]].dropna()
            corr = data_clean[numeric_cols[i]].corr(data_clean[numeric_cols[j]])
            pairs.append((abs(corr), corr, numeric_cols[i], numeric_cols[j]))

    pairs.sort(key=lambda p: p[0], reverse=True)

    fig, axes = plt.subplots(4, 5, figsize=(18, 12))
    axes = axes.flatten()

    for plot_idx, (_, corr, col_i, col_j) in enumerate(pairs[:len(axes)]):
        ax = axes[plot_idx]

        # Données sans NaN
        data_clean = df[[col_i, col_j]].dropna()

        # Scatter plot
        ax.scatter(data_clean[col_i], data_clean[col_j], alpha=0.6)
        ax.set_xlabel(col_i)
        ax.set_ylabel(col_j)
        ax.set_title(f'{col_i} vs {col_j}')
        ax.grid(True, alpha=0.3)

        # Calcule et affiche la corrélation
        ax.text(0.05, 0.95, f'Corr: {corr:.3f}', transform=ax.transAxes,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # Cache les axes inutilisés
    for idx in range(len(pairs), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.suptitle('Scatter plots - Recherche de features similaires', y=1.02)
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scatter_plot.py <dataset.csv>")
        sys.exit(1)
    
    filename = sys.argv[1]
    df = pd.read_csv(filename)
    plot_scatter(df)
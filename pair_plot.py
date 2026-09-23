# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    pair_plot.py                                       :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dsindres <dsindres@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/09/24 13:25:27 by dsindres          #+#    #+#              #
#    Updated: 2025/11/04 13:55:32 by dsindres         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

def plot_pairplot(df):
    house_col = "Hogwarts House"
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if 'Index' in numeric_cols:
        numeric_cols.remove('Index')
    
    plot_data = df[numeric_cols + [house_col]].copy()
    
    plot_data = plot_data.dropna()
    
    #plt.figure(figsize=(15, 15))
    
    if HAS_SEABORN:
        sns.pairplot(plot_data, hue=house_col, diag_kind='hist', plot_kws={'alpha':0.6})
        plt.suptitle('Pair Plot - Analyse des relations entre features', y=1.02)
        plt.savefig('pair_plot.png', dpi=150)
        print("Pair plot sauvegarde dans 'pair_plot.png'")
        plt.show()
    else:
        n_features = len(numeric_cols)
        houses = plot_data[house_col].unique()
        colors = ['red', 'green', 'blue', 'orange']

        fig, axes = plt.subplots(n_features, n_features, figsize=(28, 28))

        for i in range(n_features):
            for j in range(n_features):
                ax = axes[i, j]

                if i == j:
                    # diagonale : histogramme empile par maison
                    for k, house in enumerate(houses):
                        house_data = plot_data[plot_data[house_col] == house][numeric_cols[i]]
                        ax.hist(house_data, bins=15, alpha=0.5, color=colors[k % len(colors)])
                else:
                    # hors diagonale : nuage de points colore par maison,
                    for k, house in enumerate(houses):
                        house_data = plot_data[plot_data[house_col] == house]
                        ax.scatter(house_data[numeric_cols[j]], house_data[numeric_cols[i]],
                                   alpha=0.4, s=3, color=colors[k % len(colors)])

                # on enleve les graduations internes pas lisible a cette taille
                ax.set_xticks([])
                ax.set_yticks([])
                # afficher colonnes et lignes que sur les premieres 
                if i == n_features - 1:
                    ax.set_xlabel(numeric_cols[j], rotation=45, ha='right', fontsize=8)
                if j == 0:
                    ax.set_ylabel(numeric_cols[i], rotation=0, ha='right', fontsize=8)

        # legende 
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[k % len(colors)],
                   markersize=8, label=house) for k, house in enumerate(houses)]
        fig.legend(handles=handles, loc='upper right', fontsize=12)

        plt.suptitle('Pair Plot - Analyse des relations entre features', y=1.0)
        plt.tight_layout()
        plt.savefig('pair_plot.png', dpi=150)
        print("Pair plot sauvegarde dans 'pair_plot.png'")
        plt.show()

    
    print("Analyse pour la régression logistique:")
    print("Choisis les features avec:")
    print("- Faible corrélation entre elles (évite la multicolinéarité)")
    print("- Bonne séparation des classes dans le pair plot")
    print("- Distributions différentes selon les maisons")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python pair_plot.py <dataset.csv>")
        sys.exit(1)
    
    filename = sys.argv[1]
    df = pd.read_csv(filename)
    plot_pairplot(df)

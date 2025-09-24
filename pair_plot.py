# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    pair_plot.py                                       :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dsindres <dsindres@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/09/24 13:25:27 by dsindres          #+#    #+#              #
#    Updated: 2025/09/24 15:17:14 by dsindres         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys

def plot_pairplot(df):
    """
    Crée un pair plot pour analyser toutes les relations entre features
    """
    house_col = "Hogwarts House"
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if 'Index' in numeric_cols:
        numeric_cols.remove('Index')
    
    plot_data = df[numeric_cols + [house_col]].copy()
    
    plot_data = plot_data.dropna()
    
    plt.figure(figsize=(15, 15))
    
    try:
        sns.pairplot(plot_data, hue=house_col, diag_kind='hist', plot_kws={'alpha':0.6})
        plt.suptitle('Pair Plot - Analyse des relations entre features', y=1.02)
        plt.show()
    except ImportError:
        n_features = len(numeric_cols)
        fig, axes = plt.subplots(n_features, n_features, figsize=(15, 15))
        
        for i in range(n_features):
            for j in range(n_features):
                ax = axes[i, j]
                
                if i == j:
                    ax.hist(plot_data[numeric_cols[i]].dropna(), bins=20, alpha=0.7)
                    ax.set_title(numeric_cols[i])
                else:
                    clean_data = plot_data[[numeric_cols[j], numeric_cols[i]]].dropna()
                    ax.scatter(clean_data[numeric_cols[j]], clean_data[numeric_cols[i]], alpha=0.6)
                    
                if i == n_features - 1:
                    ax.set_xlabel(numeric_cols[j])
                if j == 0:
                    ax.set_ylabel(numeric_cols[i])
        
        plt.tight_layout()
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
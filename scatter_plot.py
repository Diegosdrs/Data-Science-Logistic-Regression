# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    scatter_plot.py                                    :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dsindres <dsindres@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/09/24 13:14:28 by dsindres          #+#    #+#              #
#    Updated: 2025/09/24 15:17:40 by dsindres         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

def plot_scatter(df):
    """
    Affiche des scatter plots pour trouver les deux features similaires
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if 'Index' in numeric_cols:
        numeric_cols.remove('Index')
    
    n_features = len(numeric_cols)
    fig, axes = plt.subplots(4, 5, figsize=(18, 12))
    axes = axes.flatten()
    
    # Compare chaque paire de features
    plot_idx = 0
    for i in range(n_features):
        for j in range(i+1, n_features):
            if plot_idx >= len(axes):
                break
                
            ax = axes[plot_idx]
            
            # Données sans NaN
            data_clean = df[[numeric_cols[i], numeric_cols[j]]].dropna()
            
            # Scatter plot
            ax.scatter(data_clean[numeric_cols[i]], data_clean[numeric_cols[j]], alpha=0.6)
            ax.set_xlabel(numeric_cols[i])
            ax.set_ylabel(numeric_cols[j])
            ax.set_title(f'{numeric_cols[i]} vs {numeric_cols[j]}')
            ax.grid(True, alpha=0.3)
            
            # Calcule et affiche la corrélation
            corr = data_clean[numeric_cols[i]].corr(data_clean[numeric_cols[j]])
            ax.text(0.05, 0.95, f'Corr: {corr:.3f}', transform=ax.transAxes, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            plot_idx += 1
    
    # Cache les axes inutilisés
    for idx in range(plot_idx, len(axes)):
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
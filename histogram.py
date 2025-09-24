# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    histogram.py                                       :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dsindres <dsindres@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/09/24 12:05:00 by dsindres          #+#    #+#              #
#    Updated: 2025/09/24 15:16:40 by dsindres         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

def load_data(filename):
    """Charge les données du fichier CSV"""
    try:
        df = pd.read_csv(filename)
        return df
    except Exception as e:
        print(f"Erreur lors du chargement du fichier: {e}")
        sys.exit(1)

def plot_histogram(df):
    """
    Affiche des histogrammes pour chaque cours, groupés par maison
    pour répondre à: Quel cours a une distribution homogène entre maisons?
    """
    
    # Trouve les colonnes de cours (exclut la maison et autres infos)
    house_col = "Hogwarts House"  # Adapte selon ton CSV
    course_cols = [col for col in df.columns if col not in [house_col, "Index", "First Name", "Last Name", "Birthday", "Best Hand"]]
    
    houses = df[house_col].unique()
    colors = ['red', 'green', 'blue', 'yellow']
    
    fig, axes = plt.subplots(4, 4, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, course in enumerate(course_cols):
        ax = axes[i]
        
        for j, house in enumerate(houses):
            if pd.isna(house):
                continue
                
            house_data = df[df[house_col] == house][course].dropna()
            
            # Histogramme avec transparence pour voir les chevauchements
            ax.hist(house_data, bins=20, alpha=0.6, label=house, color=colors[j % len(colors)])
        
        ax.set_title(f'{course}')
        ax.set_xlabel('Notes')
        ax.set_ylabel('Fréquence')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle('Distribution des notes par cours et par maison', y=1.02)
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python histogram.py <dataset.csv>")
        sys.exit(1)
    
    filename = sys.argv[1]
    df = load_data(filename)
    plot_histogram(df)
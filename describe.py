# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    describe.py                                        :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dsindres <dsindres@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/09/08 13:19:35 by dsindres          #+#    #+#              #
#    Updated: 2025/10/29 11:06:53 by dsindres         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
import pandas as pd
from pathlib import Path
import sys


def display_quartiles(data, features, count):
    q1 = []
    med = []
    q3 = []

    # Convertir les données en numpy array numérique
    data_num = np.array([[float(x) if x != "" else np.nan for x in row] for row in data])
    
    for idx_col, col in enumerate(features):
        column_data = data_num[:, idx_col]
        
        # enlever les NaN
        valid_data = column_data[~np.isnan(column_data)]
        
        # Trier
        sorted_col = np.sort(valid_data)
        n = len(sorted_col)
        
        if n == 0:
            q1.append(None)
            med.append(None) 
            q3.append(None)
            continue
            
        # Q1 = 25ème percentile
        q1_pos = (n - 1) * 0.25
        if q1_pos == int(q1_pos):
            q1_value = sorted_col[int(q1_pos)]
        else:
            lower = int(q1_pos)
            upper = lower + 1
            weight = q1_pos - lower
            q1_value = sorted_col[lower] * (1 - weight) + sorted_col[upper] * weight
        
        # Médiane = 50ème percentile  
        med_pos = (n - 1) * 0.5
        if med_pos == int(med_pos):
            med_value = sorted_col[int(med_pos)]
        else:
            lower = int(med_pos)
            upper = lower + 1
            weight = med_pos - lower
            med_value = sorted_col[lower] * (1 - weight) + sorted_col[upper] * weight
            
        # Q3 = 75ème percentile
        q3_pos = (n - 1) * 0.75
        if q3_pos == int(q3_pos):
            q3_value = sorted_col[int(q3_pos)]
        else:
            lower = int(q3_pos)
            upper = lower + 1  
            weight = q3_pos - lower
            q3_value = sorted_col[lower] * (1 - weight) + sorted_col[upper] * weight
            
        q1.append(q1_value)
        med.append(med_value)
        q3.append(q3_value)
    
    return q1, med, q3
            

def display_max(data, features):
    li = []
    for idx_col, col in enumerate(features):
        maxi = None
        for row in data:
            value = row[idx_col]
            if value == "" or pd.isna(value):
                continue
            try:
                value = float(value)
                if maxi is None or value > maxi:
                    maxi = value
            except ValueError:
                continue
        li.append(maxi)
    return li


def display_min(data, features):
    li = []
    for idx_col, col in enumerate(features):
        mini = None
        for row in data:
            value = row[idx_col]
            if value == "" or pd.isna(value):
                continue
            try:
                value = float(value)
                if mini is None or value < mini:
                    mini = value
            except ValueError:
                continue
        li.append(mini)
    return li


def display_std(data, features, mean, count):
    li = []
    for idx_col, col in enumerate(features):
        res_1 = 0
        res_sum = 0
        res_final = 0
        for idx_row, row in enumerate(data):
            value = row[idx_col]
            if value == "" or pd.isna(value):
                continue
            try:
                value = float(value)
                res_1 = (value - mean[idx_col])**2
                res_sum += res_1
            except ValueError:
                continue
        # n-1 (et non n) pour matcher l'ecart-type "echantillon" utilise
        # par defaut par pandas.describe() (correction de Bessel)
        ddof = count[idx_col] - 1 if count[idx_col] > 1 else 1
        res_final = (res_sum / ddof) ** 0.5
        li.append(res_final)
    return li

def display_mean(data, features):
    li = []
    for idx_col, col in enumerate(features):
        count_len = 0
        count_sum = 0.0
        for idx_row, row in enumerate(data):
            value = row[idx_col]
            if value == "" or pd.isna(value):
                continue
            try:
                value = float(value)
                count_len += 1
                count_sum += value
            except ValueError:
                continue
        res = count_sum / count_len if count_len > 0 else None
        li.append(res)
    return li


def display_count(data, features):
    li = []
    for idx_col, col in enumerate(features):
        count = 0
        for idx_row, row in enumerate(data):
            value = row[idx_col]
            if value == "" or pd.isna(value):
                continue
            try:
                value = float(value)
                count += 1
            except ValueError:
                continue
        li.append(count)
    return li


def display_missing(data, features, count):
    li = []
    total = len(data)
    for idx_col, col in enumerate(features):
        li.append(total - count[idx_col])
    return li


def display_variance(std):
    return [s ** 2 for s in std]


def display_range(mini, maxi):
    return [ma - mi for mi, ma in zip(mini, maxi)]


def display_iqr(q1, q3):
    return [q3_val - q1_val for q1_val, q3_val in zip(q1, q3)]


def display_skewness(data, features, mean, count):
    li = []
    for idx_col, col in enumerate(features):
        m2_sum = 0.0
        m3_sum = 0.0
        for idx_row, row in enumerate(data):
            value = row[idx_col]
            if value == "" or pd.isna(value):
                continue
            try:
                value = float(value)
                diff = value - mean[idx_col]
                m2_sum += diff ** 2
                m3_sum += diff ** 3
            except ValueError:
                continue

        n = count[idx_col]
        if n == 0:
            li.append(None)
            continue

        m2 = m2_sum / n
        m3 = m3_sum / n

        if m2 == 0:
            li.append(0.0)
            continue

        li.append(m3 / (m2 ** 1.5))
    return li


def data_display(data, features):
    print("               ", end="")
    for ft in features:
        displayed = trunc(str(ft))
        print(f"{displayed}{add_space(displayed)}", end="")

    print("")
        
    count = display_count(data, features)    
    print(f"Count{add_space('count')}", end="")
    for value in count:
        displayed = trunc(str(value))
        print(f"{displayed}{add_space(displayed)}", end="")

    print("")
    mean = display_mean(data, features)
    print(f"Mean{add_space('mean')}", end="")
    for value in mean:
        mean_formated = f"{value:.4f}"
        print(f"{mean_formated}{add_space(mean_formated)}", end="")    
        
    print("")
    std = display_std(data, features, mean, count)
    print(f"Std{add_space('std')}", end="")
    for value in std:
        std_formated = f"{value:.4f}"
        print(f"{std_formated}{add_space(str(std_formated))}", end="")

    print("")
    mini = display_min(data, features)
    print(f"Min{add_space('min')}", end="")
    for value in mini:
        min_formated = f"{value:.4f}"
        print(f"{min_formated}{add_space(str(min_formated))}", end="")
        
    print("")
    maxi = display_max(data, features)
    print(f"Max{add_space('max')}", end="")
    for value in maxi:
        max_formated = f"{value:.4f}"
        print(f"{max_formated}{add_space(str(max_formated))}", end="")

    print("")
    q1, med, q3 = display_quartiles(data, features, count)
    print(f"25%{add_space('25%')}", end="")
    for value in q1:
        q1_formated = f"{float(value):.4f}"
        print(f"{q1_formated}{add_space(str(q1_formated))}", end="")

    print("")
    print(f"50%{add_space('50%')}", end="")
    for value in med:
        med_formated = f"{float(value):.4f}"
        print(f"{med_formated}{add_space(str(med_formated))}", end="")

    print("")
    print(f"75%{add_space('75%')}", end="")
    for value in q3:
        q3_formated = f"{float(value):.4f}"
        print(f"{q3_formated}{add_space(str(q3_formated))}", end="")

    print("")
    missing = display_missing(data, features, count)
    print(f"Missing{add_space('missing')}", end="")
    for value in missing:
        displayed = trunc(str(value))
        print(f"{displayed}{add_space(displayed)}", end="")

    print("")
    variance = display_variance(std)
    print(f"Variance{add_space('variance')}", end="")
    for value in variance:
        var_formated = f"{value:.4f}"
        print(f"{var_formated}{add_space(str(var_formated))}", end="")

    print("")
    range_ = display_range(mini, maxi)
    print(f"Range{add_space('range')}", end="")
    for value in range_:
        range_formated = f"{value:.4f}"
        print(f"{range_formated}{add_space(str(range_formated))}", end="")

    print("")
    iqr = display_iqr(q1, q3)
    print(f"IQR{add_space('iqr')}", end="")
    for value in iqr:
        iqr_formated = f"{value:.4f}"
        print(f"{iqr_formated}{add_space(str(iqr_formated))}", end="")

    print("")
    skewness = display_skewness(data, features, mean, count)
    print(f"Skewness{add_space('skewness')}", end="")
    for value in skewness:
        skew_formated = f"{value:.4f}"
        print(f"{skew_formated}{add_space(str(skew_formated))}", end="")

    print("")


def trunc(text: str):
    return text[:10]
        

def add_space(text: str) -> str:
    nbr_of_space = 15 - len(text)
    if nbr_of_space < 1:
        nbr_of_space = 1
    return " " * nbr_of_space

    
def supp_index(new_data, numeric_data_features):
    for idx, value in enumerate(numeric_data_features):
        if value == "Index" or value == "index":
            new_data = np.delete(new_data, idx, axis=1)
            del numeric_data_features[idx]
            break
    
    return new_data, numeric_data_features
    

def numerical_data(data : np.ndarray , features):
    new_data = []
    numeric_data_features = []

    first_line_data = data[0]
    for idx_col, value in enumerate(first_line_data):
        try:
            float(value)
            numeric_data_features.append(features[idx_col])
            new_data.append(data[:, idx_col])
        except ValueError:
            continue
    
    new_data = np.array(new_data, dtype=float).T

    #print(f"{new_data[:5]}")
    
    return (supp_index(new_data, numeric_data_features))
    

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Erreur: mauvais nombre d'argument")
        sys.exit(1)

    filename = sys.argv[1]
    file_path = Path(filename)

    if not file_path.exists() or not file_path.is_file():
        print(f"Erreur: fichier introuvable")
        sys.exit(1)

    with open(filename, "r") as f:
        first_line = f.readline().strip()
    features = first_line.split(",")


    data = np.array(pd.read_csv(filename))
    data_model, features_model = numerical_data(data, features)
    data_display(data_model, features_model)
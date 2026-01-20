import numpy as np
import csv


# Retourne dataset avec y adapté a sigmoid (pas de one hot)
def create_set_sigmoid(csv_path) :
    
    X_list = []
    y_list = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)

        for row in reader:
            filtered_row = row[2:] # Nouveau : je skippe les deux premieres colonnes (id et y)
            X_list.append(filtered_row)

            # Je crée les one hot pour y
            if (row[1] == 'M'):
                y_one_hot_line = [1]
            else :
                y_one_hot_line = [0]
            y_list.append(y_one_hot_line)

        X_list = [[float(x) for x in row] for row in X_list]
    X_new = np.array(X_list)
    y_new = np.array(y_list)

    X = X_new
    y = y_new

    # On normalise (standardisation) pour avoir une moyenne de 0 et un ecart type de 1 : 
    mean = np.mean(X, axis=0)        # moyenne de chaque colonne : Retourne une linge avec la moyenne pour chaque colone (ex : [12, 41, 22, 455])
    std = np.std(X, axis=0)          # écart-type de chaque colonne
    X_norm = (X - mean) / (std + 1e-8)  # Ajoute 1e-8 pour éviter division par zéro, on soustrait chaque ligne de X par les valeurs contenues dans la liste mean. Même principe avec la division pour std
    X = X_norm

    return X, y

# Retourne dataset avec y adapté a softmax (avec one hot)
def create_set_softmax(csv_path) :
    
    X_list = []
    y_list = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)

        for row in reader:
            filtered_row = row[2:] # Nouveau : je skippe les deux premieres colonnes (id et y)
            X_list.append(filtered_row)

            # Je crée les one hot pour y
            if (row[1] == 'M'):
                y_one_hot_line = [1, 0]
            else :
                y_one_hot_line = [0, 1]
            y_list.append(y_one_hot_line)

        X_list = [[float(x) for x in row] for row in X_list]
    X_new = np.array(X_list)
    y_new = np.array(y_list)

    X = X_new
    y = y_new

    # On normalise (standardisation) pour avoir une moyenne de 0 et un ecart type de 1 : 
    mean = np.mean(X, axis=0)        # moyenne de chaque colonne : Retourne une linge avec la moyenne pour chaque colone (ex : [12, 41, 22, 455])
    std = np.std(X, axis=0)          # écart-type de chaque colonne
    X_norm = (X - mean) / (std + 1e-8)  # Ajoute 1e-8 pour éviter division par zéro, on soustrait chaque ligne de X par les valeurs contenues dans la liste mean. Même principe avec la division pour std
    X = X_norm

    return X, y


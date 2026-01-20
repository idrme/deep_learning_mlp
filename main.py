from layers import *
from sequential import *
from loss_optimize import *
import numpy as np
import csv
from set import *
import os

import argparse

# Ajoute les couches à une liste. Si dans les parametres su programme on choisis par ex : 10 15 alors il y aura deux couches cachées + une couche entrée + une couche sortie
def getLayersFromArg(layer):
    if (layer is None):
        # 2 couches cachées par défaut
        return (
        [
            Dense(30, 10), # C'est la premiere couche cachée et non la couche d'entrée, la sortie de la couche d'entrée est X (a voir)
            ReLU(),
            Dense(10, 10), # Couche cachée 2
            ReLU(),
            Dense(10, 2), # Couche de sortie
            Softmax()
        ])
    else:
        # Arreter si on souhaite avoir au minimum 2 couches cachées
        # if (len(args.layer) < 2):
        #     print("You need at least 2 hidden layers")

        for elt in layer:
            if (elt <= 0):
                raise Exception("A layer must contains at least 1 neuron")

        # print("Toutes les couches sont créées à partir du parametre --layer")
        layers_final = []
        new_dense = Dense(30, layer[0])
        new_activation = ReLU()
        last_nb_neurons = layer[0]

        layers_final.append(new_dense)
        layers_final.append(new_activation)

        # Si il n'y a qu'une seule couche cachée
        if (len(layer) == 1):
            new_dense = Dense(last_nb_neurons, 2) # 2 de sortie pour softmax
            new_activation = Softmax()
            layers_final.append(new_dense)
            layers_final.append(new_activation)

        # Je fais une boucle for sur le nombre d'élements dans la variable args.layer, en partant a l'index 1
        for i in range(1, len(layer)):

            # Vérifie si derniere couche
            if (i == len(layer) - 1):
                new_dense = Dense(last_nb_neurons, 2) # 2 de sortie pour softmax
                new_activation = Softmax()
            else :
                new_dense = Dense(last_nb_neurons, layer[i])
                last_nb_neurons = layer[i]
                new_activation = ReLU()
            
            # Ajoute la couche dense et d'activation à la liste
            layers_final.append(new_dense)
            layers_final.append(new_activation)

        # model = Sequential(layers_final)
        return (layers_final)

    return

def main():

    # Supprime le fichier d'historique si il existe
    if os.path.exists("history.txt"):
        os.remove("history.txt")

    # Arguments du programme : python3 main.py train_set.csv validation_set.csv...
    parser = argparse.ArgumentParser(description="Parse MLP")
    parser.add_argument("file_train_set", nargs="?", default="train_set.csv", help="Train set (.csv)")
    parser.add_argument("file_validation_set", nargs="?", default="validation_set.csv", help="Validation set (.csv)")
    parser.add_argument("--epochs", type=int, default=70, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--layer", type=int, nargs="+")
    parser.add_argument("--adam", action="store_true", help="Activate the Adam Optimizer")
    args = parser.parse_args()

    # Crée les couches à partir de l'argument du programme --layer 
    try :
        model = Sequential(getLayersFromArg(args.layer))
    except Exception as err:
        print("A layer must contains at least 1 neuron")
        return

    # Récupère les entrées et sorties pour chaque dataset. Si aucun n'est spécifié en argument alors charge le chemin par défaut
    try:
        if args.file_train_set is None:
            X, y = create_set_softmax("train_set.csv")
        else :
            X, y = create_set_softmax(args.file_train_set)

        if args.file_validation_set is None:
            X_validation, y_validation = create_set_softmax("validation_set.csv")
        else :
            X_validation, y_validation = create_set_softmax(args.file_validation_set)
    except :
        print("Error when opening file")
        return

    # On affiche la forme des inputs du dataset
    print(f"x_train shape : {X.shape}")
    print(f"x_valid shape : {X_validation.shape}")

    # Choix entre ADAM ou SGD 
    if (args.adam):
        # ADAM / CCE
        model.compile(
            loss=CategoricalCrossEntropy(),
            optimizer=Adam(getLayersFromArg(args.layer), lr=args.learning_rate)
        )
    else :
        # SGD / CCE
        model.compile(
            loss=CategoricalCrossEntropy(),
            optimizer=SGD(lr=args.learning_rate)
        )

    # Lance l'entrainement du modèle
    model.fit(X, y, X_validation, y_validation, epochs=args.epochs, batch_size=args.batch_size)

    # Exporte les poids/biais et topologie du réseau de neurones
    with open("export.pkl", "wb") as f:
        pickle.dump(model.layers, f)
        print(f"saving model './export.pkl' to disk...")

if __name__ == "__main__":
    main()
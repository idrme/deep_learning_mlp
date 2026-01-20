import pickle
from layers import *
from sequential import *
from loss_optimize import *
import numpy as np

from set import *
import argparse


def main():
    # Arguments du programme : python3 predict.py validation_set.csv export.kpl (arguments are optionnals)
    parser = argparse.ArgumentParser(description="Parse MLP")
    parser.add_argument("file_validation_set", nargs="?", default="validation_set.csv", help="Validation set (.csv)")
    parser.add_argument("file_export", nargs="?", default="export.pkl", help="Topology file / parameters file")
    args = parser.parse_args()

    try :
        # Récupère la liste des couches denses et d'activation (fonctionne)
        with open(args.file_export, "rb") as f:
            layers_trained = pickle.load(f)

        # Charge le dataset pour sigmoid
        X, y = create_set_sigmoid(args.file_validation_set) # M'assurer que le fichier a bien été ouvert !!!!!
    except :
        print("Error when opening file")
        return

    # Je modifie la derniere couche de sortie et fonction d'activation en sigmoide
    layers_trained[-1] = Sigmoid()

    # Je calcule les poids qui remplaceront les poids dans la dernière couche
    w_sigmoid = (layers_trained[-2].W[:, 0] - layers_trained[-2].W[:, 1]).reshape(-1, 1) # Prend toute les valeurs dans la premiere colone (poids entre L -1 et premier neurone derniere couche), puis toutes les valeurs de la deuxieme colone (poids entre L - 1 et deuxieme neurone)
    b_sigmoid = layers_trained[-2].b[0][0] - layers_trained[-2].b[0][1]
    b_sigmoid = np.array([[b_sigmoid]])

    # Je modifie les poids derniere couche de sorti
    layers_trained[-2].W = w_sigmoid
    layers_trained[-2].b = b_sigmoid

    # J'insert les couches dans le modèle
    model = Sequential(layers_trained)

    # Je chosis le BCE et SGD
    model.compile(
        loss=BinaryCrossEntropy(),
        optimizer=SGD(lr=0.1)
    )

    # Je fais du forward propagation pour avoir les résutlats et les afficher
    model.fit_predict(X, y, 4, shuffle=False)


if __name__ == "__main__":
    main()
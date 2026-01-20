import pandas as pd
import sys

def main():
    try :
        # Vérifie si il y a deux arguments et si les arguments sont corrects
        if (len(sys.argv) != 3):
            raise Exception("Wrong number of arguments : pyton3 split.py <path_csv.csv> <validation set size (between 0 and 1)>")
        if (float(sys.argv[2]) <= 0 or float(sys.argv[2]) >= 1):
            raise Exception("Choose a validation set size greater than 0 and lower than 1")

        # Récupère les données d'un csv
        data = pd.read_csv(sys.argv[1])

        # Stocke la taille d'un validation set
        validation_set_size = float(sys.argv[2])

        # Récupère de façon aléatoire 20% des lignes dans le fichier csv et les met dans validation_set
        validation_set = data.sample(frac=validation_set_size)

        # Récupère le reste des lignes en supprimant les lignes contenues dans validation_set (grâce a .index)
        train_set = data.drop(validation_set.index)

        # Sauvegarde les nouveaux datasets dans deux fichier .csv
        validation_set.to_csv("validation_set.csv", index=False) # index=False permet de ne pas ajouter l'id au début de chaque ligne dans le fichier sauvegardé
        train_set.to_csv("train_set.csv", index=False) # index=False permet de ne pas ajouter l'id au début de chaque ligne dans le fichier sauvegardé

    except Exception as err :
        print(f"Erreur try catch : {err}")

if __name__ == "__main__":
    main()
import matplotlib.pyplot as plt
import numpy as np

def showGraphLoss(epochs, loss_history, loss_history_validation) :

    try:
        epochs_graph = range(0, epochs)
        train_loss = loss_history
        val_loss = loss_history_validation

        # Création du graphe
        plt.figure()
        plt.plot(epochs_graph, train_loss, label="training loss")
        plt.plot(epochs_graph, val_loss, label="validation loss")

        # Labels et titre
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Loss train vs loss validation")

        # Limites de l'axe Y
        train_loss_max = np.max(train_loss)
        val_loss_max = np.max(val_loss)
        plt.ylim(0, max(train_loss_max, val_loss_max))

        # Légende
        plt.legend()

        # Affichage
        plt.show()

    except KeyboardInterrupt:
        print("Ctrl +C : fermeture de la fenetre et du programme")
        plt.close('all') # ferme la fenetre image de matplotlib proprement



def showGraphAccuracy(epochs, accuracy_history, accuracy_history_validation) :

    try :
        epochs_graph = range(0, epochs) 
        train_accuracy= accuracy_history
        val_accuracy = accuracy_history_validation

        # Création du graphe
        plt.figure()
        plt.plot(epochs_graph, train_accuracy, label="training accuracy")
        plt.plot(epochs_graph, val_accuracy, label="validation accuracy")

        # Labels et titre
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Accuracy train vs accuracy validation")

        # Limites de l'axe Y
        train_accuracy_max = np.max(train_accuracy)
        val_accuracy_max = np.max(val_accuracy)
        train_accuracy_min = np.min(train_accuracy)
        val_accuracy_min = np.min(val_accuracy)
        plt.ylim(min(train_accuracy_min, val_accuracy_min), max(train_accuracy_max, val_accuracy_max))

        # Légende
        plt.legend()

        # Affichage
        plt.show()

    except KeyboardInterrupt:
        print("Ctrl +C : fermeture de la fenetre et du programme")
        plt.close('all') # ferme la fenetre image de matplotlib proprement

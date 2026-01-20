import numpy as np
from loss_optimize import *
from layers import *
import matplotlib.pyplot as plt

from graphs import *
import pickle

class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def compile(self, loss, optimizer):
        self.loss_fn = loss
        self.optimizer = optimizer

        # Si Sigmoid / BCE ou Softmax / CCE : on ignore la dérivée de la fonction Sigmoid ou Softmax
        lastLayer = self.layers[-1]
        lastLayer.isLast = True
        if (isinstance(self.loss_fn, BinaryCrossEntropy) and isinstance(lastLayer, Sigmoid)):
            lastLayer.ignoreLastBackward = True
        elif (isinstance(self.loss_fn, CategoricalCrossEntropy) and isinstance(lastLayer, Softmax)):
            lastLayer.ignoreLastBackward = True

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            # Appelle la bonne fonction de back propagation. Si BCE et Sigmoid alors on appelle backward_last_layer, sinon la fonction normale
            if (layer.ignoreLastBackward):
                grad = layer.backward_last_layer(grad)
            else :
                grad = layer.backward(grad)

    # Uniquement pour sigmoide lors des tests inférence
    def fit_predict(self, X_validation, y_validation, batch_size=2, shuffle=True):
        """Forward propagation pour phase d'inférence (production avec poids déja entrainés)"""
        n_samples_validation = X_validation.shape[0]
        if shuffle:
            indices_validation = np.random.permutation(n_samples_validation)
            X_validation = X_validation[indices_validation]
            y_validation = y_validation[indices_validation]

        loss_value = 0.0

        total_correct_validation = 0
        total_samples_validation = 0

        for i in range(0, n_samples_validation, batch_size):
            X_batch_validation = X_validation[i:i + batch_size]
            y_batch_validation = y_validation[i:i + batch_size]

            # Forward
            y_pred_validation = self.forward(X_batch_validation)


            # Affiche pour chaque ligne du CSV (dataset) le résultat de la prédiction (forward puis sortie sigmoide)
            for y_predicted_solo in y_pred_validation :
                if (y_predicted_solo >= 0.5):
                    print("M")
                else : 
                    print("B")

            # Loss
            loss_validation = self.loss_fn.forward(y_pred_validation, y_batch_validation)
            loss_value += loss_validation * X_batch_validation.shape[0] # On multiplie par le nombre d'exemples dans le batch, par la suite on divise par le nombre d'exemples total car le epoch loss doit etre la moyenne

            # Calcule l'accuracy pour le validation set
            # predicted_index_validation = np.argmax(y_pred_validation, axis=1) # Contient une matrice de (4, 2) si il y a 4 batchs et 2 neurones de sortie softmax. Renvoi une matrice avec chaque ligne contenant l'indice de la plus haute valeur de sortie softmax
            # reality_index_validation = np.argmax(y_batch_validation, axis=1) # Pareil mais renvoi un tableau contenant l'indice du one-hot valant 1
            # total_correct_validation += np.sum(reality_index_validation == predicted_index_validation) # Crée un tableau contenant par ex [1, 0, 1, 1] avec 1 = true et 0 = false puis additionne tous les éléments du tableau
            total_samples_validation += y_batch_validation.shape[0] # Ajoute le nombre d'éléments dans ce batch au nombre total d'éléments pour ce dataset train

        # Calcul final en pourcentage de l'accuracy
        accuracy_validation = total_correct_validation / total_samples_validation
        loss_value /= n_samples_validation
        
        # Affichage des résutlats pour la prediction
        # print(f"Accuracy prediction: {accuracy_validation:.2f}")
        print(f"Loss prediction: {loss_value:.4f}")

    def fit(self, X, y, X_validation, y_validation, epochs=100, batch_size=32, shuffle=True):
        n_samples = X.shape[0]
        n_samples_validation = X_validation.shape[0]

        # Conserve l'historique des loss a chaque epoch pour affichage dans le graph dans le futur
        loss_history = []
        loss_history_validation = []

        # Conserve l'historique des accuracy (precision) a chaque epoch pour affichage dans le graph dans le futur
        accuracy_history = []
        accuracy_history_validation = []

        # Conserve l'historique des precision a chaque epoch pour affichage dans le graph dans le futur
        precision_history = []
        precision_history_validation = []

        # Variables pour early stopping
        best_loss_val = float("inf")
        patience_early_stopping = 3 # Si la loss du set validation est supérieur 3 fois d'affilé
        counter_early_stopping = 0

        for epoch in range(epochs):

            # Contient le nombre de prédictions correctes et le nombre d'exemples pour un epoch (accuracy)
            total_correct = 0
            total_samples = 0

            total_true_positive = 0
            total_false_positive = 0


            # Forward, back propagation et descente de gradient pour train set    
            if shuffle:
                indices = np.random.permutation(n_samples)
                X = X[indices]
                y = y[indices]

            epoch_loss = 0.0

            for i in range(0, n_samples, batch_size):
                X_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]

                # Forward
                y_pred = self.forward(X_batch)

                # Loss
                loss = self.loss_fn.forward(y_pred, y_batch)
                epoch_loss += loss * X_batch.shape[0]

                # Calcule l'accuracy pour le train set

                # predicted_index et reality_index sont de forme (4, 1) si 4 échantillons avec l'indice obtenu
                # total_correct compare deux indices et crée une nouvelle liste qui sera additionné pour foremr un nombre final 
                predicted_index = np.argmax(y_pred, axis=1) # Contient une matrice de (4, 2) si il y a 4 batchs et 2 neurones de sortie softmax. Renvoi une matrice avec chaque ligne contenant l'indice de la plus haute valeur de sortie softmax
                reality_index = np.argmax(y_batch, axis=1) # Pareil mais renvoi un tableau contenant l'indice du one-hot valant 1
                total_correct += np.sum(predicted_index == reality_index) # Crée un tableau contenant par ex [1, 0, 1, 1] avec 1 = true et 0 = false puis additionne tous les éléments du tableau
                total_samples += y_batch.shape[0] # Ajoute le nombre d'éléments dans ce batch au nombre total d'éléments pour ce dataset train

                # Calcule la precision (crée un tableau de 0 et de 1 puis fait la somme) pour tp et fp
                total_true_positive += sum((predicted_index == 0) & (reality_index == 0)) # On prend tous les positifs (1er neurone >= 0.5) corrects
                total_false_positive += sum((predicted_index == 0) & (reality_index == 1)) # On prend tous les positifs (1er neurone >= 0.5) incorrects (considéré vrai alors qu'en réalité c'est faux)




                # Backward
                grad = self.loss_fn.backward()
                self.backward(grad)

                # Update
                for layer in self.layers:
                    self.optimizer.step(layer.params(), layer.grads(), layer)


            # Forward propagation validation set

            epoch_loss_validation = 0.0
            total_correct_validation = 0
            total_samples_validation = 0
            total_true_positive_validation = 0
            total_false_positive_validation = 0

            for i in range(0, n_samples_validation, batch_size):
                X_batch_validation = X_validation[i:i + batch_size]
                y_batch_validation = y_validation[i:i + batch_size]

                # Forward
                y_pred_validation = self.forward(X_batch_validation)

                # Loss
                loss_validation = self.loss_fn.forward(y_pred_validation, y_batch_validation)
                epoch_loss_validation += loss_validation * X_batch_validation.shape[0] # On multiplie par le nombre d'exemples dans le batch, par la suite on divise par le nombre d'exemples total car le epoch loss doit etre la moyenne

                # Calcule l'accuracy pour le validation set
                predicted_index_validation = np.argmax(y_pred_validation, axis=1) # Contient une matrice de (4, 2) si il y a 4 batchs et 2 neurones de sortie softmax. Renvoi une matrice avec chaque ligne contenant l'indice de la plus haute valeur de sortie softmax
                reality_index_validation = np.argmax(y_batch_validation, axis=1) # Pareil mais renvoi un tableau contenant l'indice du one-hot valant 1
                total_correct_validation += np.sum(reality_index_validation == predicted_index_validation) # Crée un tableau contenant par ex [1, 0, 1, 1] avec 1 = true et 0 = false puis additionne tous les éléments du tableau
                total_samples_validation += y_batch_validation.shape[0] # Ajoute le nombre d'éléments dans ce batch au nombre total d'éléments pour ce dataset train

                total_true_positive_validation += sum((predicted_index_validation == 0) & (reality_index_validation == 0)) # On prend tous les positifs (1er neurone >= 0.5) corrects (ON NE PREND PAS LES NEGATIFS CORRECTS)
                total_false_positive_validation += sum((predicted_index_validation == 0) & (reality_index_validation == 1)) # On prend tous les positifs (1er neurone >= 0.5) incorrects (considéré vrai alors qu'en réalité c'est faux)



            # Calcul final en pourcentage de l'accuracy
            accuracy_train = total_correct / total_samples
            accuracy_validation = total_correct_validation / total_samples_validation

            # Insertion dans la liste des résultats accuracy pour utilisation dans le futur pour afficher graphs
            accuracy_history.append(accuracy_train)
            accuracy_history_validation.append(accuracy_validation)

            # Calcul des résutlats de la fonction cout a chaque epoch dans le terminal
            epoch_loss /= n_samples
            epoch_loss_validation /= n_samples_validation

            # Insertion dans la liste des résultats loss pour utilisation dans le futur pour afficher graphs
            loss_history.append(epoch_loss)
            loss_history_validation.append(epoch_loss_validation)

            # Calcul final de la precision + append
            precision_train = total_true_positive / (total_true_positive + total_false_positive + 1e-8)
            precision_validation = total_true_positive_validation / (total_true_positive_validation + total_false_positive_validation + 1e-8)
            precision_history.append(precision_train)
            precision_history_validation.append(precision_validation)


            
            # Affichage des résutlats de la fonction cout a chaque epoch dans le terminal
            print(f"epoch {epoch + 1}/{epochs} - loss: {epoch_loss:.4f} - val_loss: {epoch_loss_validation:.4f}")

            # Sauvegarde l'historique
            with open("history.txt", "a") as f:
                f.write(f"epoch {epoch + 1}/{epochs} - loss: {epoch_loss:.4f} - val_loss: {epoch_loss_validation:.4f} - accuracy_train: {accuracy_train:.2f} - accuracy_validation: {accuracy_validation:.2f} - precision_train : {precision_train:.2f} - precision_validation : {precision_validation:.2f}\n")

            # Early stopping : arrete l'entrainement si la loss du validation augmente trop
            if (epoch_loss_validation < best_loss_val):
                best_loss_val = epoch_loss_validation
                counter_early_stopping = 0
            else :
                counter_early_stopping += 1
            if counter_early_stopping >= patience_early_stopping:
                print(f"Early stopping at epoch {epoch + 1}/{epochs}")
                epochs = epoch + 1 # TEST : je modifie le nombre total d'epochs réalisés
                break

        # Affichage du graph
        showGraphLoss(epochs, loss_history, loss_history_validation)
        showGraphAccuracy(epochs, accuracy_history, accuracy_history_validation)


        # print(loss_history)
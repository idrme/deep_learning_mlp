import numpy as np
from layers import *

class BinaryCrossEntropy:
    def forward(self, y_pred, y_true):
        self.y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7) # Permet d'éviter d'avoir 0 si j'utilise la fonction log car log(0) = -infini
        self.y_true = y_true
        return -np.mean( # On prend bien la moyenne des loss des batchs pour retourner une valeur loss unique
            y_true * np.log(self.y_pred) +
            (1 - y_true) * np.log(1 - self.y_pred)
        )

    def backward(self):
        return (self.y_pred - self.y_true) / self.y_true.shape[0] # Vecteur de même taille que la sortie du reseau (4, 1) dans l'exemple
        # la division par self.y_true.shape[0] permet d'éviter de diviser a chaque couche les gradients par le nombre d'éléménts dans le batch


class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, params, grads, layer=None):
        # Fait une itération : la premiere recupere les poids et la deuxieme recupere les biais
        for p, g in zip(params, grads): # IL N'Y A QUUNE SEULE ITERATION ICI SI JAI BIEN COMPRIS CAR 1 SEULE COUCHE
            p -= self.lr * g



class CategoricalCrossEntropy:
    def forward(self, y_pred, y_true):
        self.y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7) # Permet d'éviter d'avoir 0 si j'utilise la fonction log car log(0) = -infini
        self.y_true = y_true
        return -np.mean(np.sum(y_true * np.log(self.y_pred), axis=1))

    def backward(self):
        # return -self.y_true / self.y_pred / self.y_true.shape[0]
        return (self.y_pred - self.y_true) / self.y_true.shape[0]


# Explications : 
# Pour RMSProp (lr) : Si une valeur du gradient est grande alors son learning rate diminue. Si petit il augmente
# Pour Momentum (gradient) : modifie le gradient en se basant sur les anciens gradients : renforce les gradients qui vont dans la meme direction et annule les oscillations
# Donc pour la même position d'un poids dans le gradient : si les anciennes valeurs du gradient a cet emplacement sont positives alors ce poids est augmenté tandis que si il varie d'un positif vers le négatif souvent alors il s'annule
class Adam:
    def __init__(self, layer, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8): ##### !!!!!! MODIFIER LE lr a 0.001 !!!!!

        # Hyperparametres
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # Chaque indice représente une couche. self.mW[1] représente la deuxieme couche par ex. mW[0] représente la première couche Dense (1e couche cachée, apres la couche d'entrée)
        self.mW = [] 
        self.mb = [] 
        self.vW = [] 
        self.vb = []
        self.t = 1

        # Je crée les matrices pour mW, mb, vW et vb puis je les ajoute à la liste
        # Je fais une itération sur toutes les couches Dense (je dois bien sauter la fct d'activation)
        # Puis je récupère .myshape : je crée une matrice numpy et je l'ajoute aux m et v
        for elt in layer:

            if (isinstance(elt, Dense)):
                self.mW.append(np.zeros((elt.myshape[0], elt.myshape[1])))
                self.mb.append(np.zeros((1, elt.myshape[1])))
                self.vW.append(np.zeros((elt.myshape[0], elt.myshape[1])))
                self.vb.append(np.zeros((1, elt.myshape[1])))
            else :
                self.mW.append(np.zeros((0, 0)))
                self.mb.append(np.zeros((0, 0)))
                self.vW.append(np.zeros((0, 0)))
                self.vb.append(np.zeros((0, 0)))

    def step(self, params, grads, layer=None):
        i = 0
        for p, g in zip(params, grads):

            if (isinstance(layer, Dense)):
                # 1ere itération : les poids
                if (i == 0):
                    
                    # Les gradients précédents ont plus d'impact que le gradient actuel calculé. Si beta = 0.9 alors les anciens comptent pour 0.9 et le nouveau pour 0.1
                    self.mW[layer.layer_id] = self.beta1 * self.mW[layer.layer_id] + (1 - self.beta1) * g # Formule du SGD Momentum
                    self.vW[layer.layer_id] = self.beta2 * self.vW[layer.layer_id] + (1 - self.beta2) * (g ** 2) # Formule du RMSProp

                    # Correction du biais : permet de modifier la valeur de m au début 
                    # Car si beta est 0.9 alors : m = (1 - 0.9) * le gradient. Ce qui modifie le gradient en le rendant beaucoup trop petit
                    # Cela permet donc d'augmenter la valeur du gradient au début afin d'aller plus vite et de réduire le nombre d'epochs
                    m_hat_W = self.mW[layer.layer_id] / (1 - self.beta1 ** self.t)
                    v_hat_W = self.vW[layer.layer_id] / (1 - self.beta2 ** self.t)

                    # Modification des poids
                    p -= self.lr * m_hat_W / (np.sqrt(v_hat_W) + self.epsilon) # epsilon empeche la division par 0

                # 2ème itération : les biais
                else :
                    self.mb[layer.layer_id] = self.beta1 * self.mb[layer.layer_id] + (1 - self.beta1) * g
                    self.vb[layer.layer_id] = self.beta2 * self.vb[layer.layer_id] + (1 - self.beta2) * (g ** 2)

                    # Correction du biais : permet de modifier la valeur de m au début 
                    # Car si beta est 0.9 alors : m = (1 - 0.9) * le gradient. Ce qui modifie le gradient en le rendant beaucoup trop petit
                    # Cela permet donc d'augmenter la valeur du gradient au début afin d'aller plus vite et de réduire le nombre d'epochs
                    m_hat_b = self.mb[layer.layer_id] / (1 - self.beta1 ** self.t)
                    v_hat_b = self.vb[layer.layer_id] / (1 - self.beta2 ** self.t)

                    # Modification des biais
                    p -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon) # epsilon empeche la division par 0

            # Modifie le parametre t a la fin de chaque batch, donc a la fin de la dernière couche
            if (layer.layer_id == Layer.counter - 1):
                self.t += 1

            i += 1

import numpy as np

class Layer:

    counter = 0

    def __init__(self):
        self.isLast = False # Ajouté par moi : dernière couche : utilisé pour obtenir le delta de la dernière couche de façon optimisée
        self.ignoreLastBackward = False # Si true alors cette couche d'activation ne fera rien de special
        self.layer_id = Layer.counter
        Layer.counter += 1

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

    def params(self):
        return []

    def grads(self):
        return []

class Dense(Layer):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.W = np.random.randn(in_features, out_features) * 0.01
        self.b = np.zeros((1, out_features)) # biais initialisés à 0

        # heUniform : met les poids dans un intervalle dépendant du nombre de neurones d'entrées
        # Cela permet d'éviter le vanishing gradient (poids qui deviennent très proches de 0 et donc les poids des couches précédentes ne sont pas mis a jour )
        # Cela permet aussi d'éviter le exploding gradient : lorsque les gradeints sont sup a 1
        # En gros : limite la taille des poids
        np.random.seed(123) # Permet de toujours générer les meme nombres aléatoires a chaque epoch
        limit = np.sqrt(6 / in_features)
        myshape = (in_features, out_features)
        self.myshape = myshape
        self.W = np.random.uniform(-limit, limit, size=myshape)

    def forward(self, x):
        self.x = x # Prend (4, 16) pour derniere couche dense
        return x @ self.W + self.b # retourne (4, 1) pour la derniere couche dense

    def backward(self, grad): # Retourne les deltas (qui devront par la suite aussi etre modifié par la dérivée de la fonction d'activation) si j'ai bien compris
        self.dW = self.x.T @ grad # Retourne la forme (16, 1) pour la dernière couche dense
        self.db = np.sum(grad, axis=0, keepdims=True) # Additionne tous les exemples dans le batch pour chaque neurone. Prend (1, 1) pour derniere couche dense. (si par ex il y avait 2 neurones dans la couche alors la forme serait (1, 2))
        return grad @ self.W.T # Retourne la forme (4, 16) pour la derniere couche dense

    def params(self):
        return [self.W, self.b]

    def grads(self):
        return [self.dW, self.db]
    


class ReLU(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        self.mask = x > 0 # conserve dans un tableau True ou False
        return x * self.mask

    def backward(self, grad):
        return grad * self.mask # Si par exemple grad et self.mask sont (4, 2) alors il n'y a aucune modification de la forme. Les éléments sont multipliés un a un grace a la multiplication de hadamard
    

class Sigmoid(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x)) # Retourne toujours la même forme que x
        return self.out

    def backward(self, grad):
        return grad * self.out * (1 - self.out) # Garde la meme forme, modifie juste le contenu pour inclure la multiplication avec la dérivée de la fonction d'activation
    
    # Appelé uniquement si dernière couche avec BCE
    def backward_last_layer(self, grad): 
        return grad
    

class Softmax(Layer):

    def __init__(self):
        super().__init__()

        # Si 2 neurones en sortie alors la forme peut etre : (4, 2) avec 4 exemples en entrée et en sortie

    def forward(self, x):
        x_shifted = x - np.max(x, axis=1, keepdims=True) # Je soustrais le maximum de toutes les entrées dans softmax pour éviter un overflow
        exp_x = np.exp(x_shifted) # Chaque élément est appliqué a exponentiel
        self.out = exp_x / np.sum(exp_x, axis=1, keepdims=True) # divise chaque element par la somme de tous les elements exp.  exemple : [7.389/11.107,2.718/11.107,1/11.107]
        return self.out # Retourne une matrice (4, 2) par ex avec chaque entrée de chaque exemple remplacé par la valeur softmax finale

    def backward(self, grad):
        # Non implémenté car pas besoin étant donné qu'on l'utilise en tant que dernière activation avec CCE : optimisation
        return grad #

    # Appelé uniquement si dernière couche avec CGE
    def backward_last_layer(self, grad): 
        return grad

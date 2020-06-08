from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

class LogisticRegressionGD(object):
    '''Logistic Regression Classifier using gradient descent.
    Parameter
    ---------
    eta : float
        Lernrate (zwischen 0.0 und 1.0)
    n_iter : int
        Durchläufe der Trainingsdatenmenge
    random_state : int
        Zufallszahlgenerator für zufällige Gewichtung initialisieren

    Attributes
    ----------
    w_: 1d-array
        Gewichtungen nach Anpassung
    cost_ : list
        Summe der quadrierten Werte der Strafffunktion pro Epoche
    '''

    def __init__(self, eta=0.05, n_iter=100, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    def fit(self, X, y):
        """ Fit-Trainingsdaten
        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Trainingsvektoren, n_samples ist die Anzahl der Exemplare und n_features
            ist die Anzahl der Merkmale.
        y : {array-like}, shape = [n_samples]
            Zielwerte

        Rückgabewerte
        -------------
        self : object
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y-output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            #Wir berechnen nun den Wert der Strafffunktion
            #der logistischen Regression, nicht mehr die
            #Summe der quadrierten Werte der Strafffunktion
            cost = (-y.dot(np.log(output)) -
                    ((1 - y).dot(np.log(1-output))))
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        '''Nettoeingabe berechnen'''
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, z):
        '''Logistische Aktivierungsfunktion berechnen'''
        return 1./ (1. + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        """Klassenbezeichnung zurückgeben"""
        return np.where(self.net_input(X) >= 0.0, 1, 0)

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    # Markierungen und Farben einstellen
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # Plotten der Entscheidungsgrenze
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 0].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, \
                                     resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx1.min(), xx2.max())

    # Plotten aller Exemplare
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=colors[idx],
                    marker=markers[idx], label=cl,
                    edgecolor='black')

    # Exemplare der testdatenmenge hervorheben
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='',
                    edgecolor='black',
                    alpha=1.0, linewidths=1, marker='o',
                    s=100, label='Testdaten')

iris = datasets.load_iris()
X = iris.data[:, [2,3]]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

SC = StandardScaler()
SC.fit(X_train)
X_train_std = SC.transform(X_train)
X_test_std = SC.transform(X_test)

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
X_train_01_subset = X_train[(y_train == 0) | (y_train == 1)]
y_train_01_subset = y_train[(y_train == 0) | (y_train == 1)]
print(y_train_01_subset)
lrgd = LogisticRegressionGD(eta=0.05, n_iter=1000, random_state=1)
lrgd.fit(X_train_01_subset, y_train_01_subset)
#Plotten aller Exemplare

plot_decision_regions(X=X_train_01_subset,
                       y=y_train_01_subset,
                       classifier=lrgd)

plt.xlabel('Länge des Blütenblatts [standardisiert]')
plt.ylabel('Breite des Blütenblatts [standardisiert]')
plt.ylim(-1., 2.5)
plt.legend(loc='upper left')
plt.show()

lr = LogisticRegression(C=100.0, random_state=1)
lr.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_combined, classifier=lr,
                      test_idx=range(105, 150))
plt.xlabel('Länge des Blütenblatts [standardisiert]')
plt.ylabel('Breite des Blütenblatts [standardisiert]')
plt.legend(loc='upper left')
plt.show()

print(lr.predict_proba(X_test_std[:3,:]))
print(lr.predict_proba(X_test_std[:3,:]).argmax(axis=1))
print(lr.predict(X_test_std[:3, :]))
print(lr.predict(X_test_std[0,:].reshape(1,-1)))

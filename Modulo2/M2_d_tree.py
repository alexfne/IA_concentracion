# -*- coding: utf-8 -*-
"""Módulo 2 - decision_tree.ipynb
Original file is located at
    https://colab.research.google.com/drive/1ULVWUYfqQ1DBGmIoFvlrfpiZaBXBoxQw

### Importar librerias
Se importan las bibliotecas necesarias como numpy, pandas y algunas funciones específicas de scikit-learn.
"""

# Importando bibliotecas necesarias
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

"""### Leer los datos

Se lee un archivo CSV y se muestran las primeras 10 filas para tener una visión preliminar.
"""

# Leer los datos
'''
filename = input(''Ingresa la ruta donde se encuentra el archivo + /filename.csv
                 => '')
'''
# Cargar el archivo CSV, omitiendo la primera fila y sin encabezado
data = pd.read_csv("iris.csv", skiprows=1, header=None)

# Leer los títulos/nombres de las columnas
col_names = list(data.columns)
data.head(10)

"""### Clase de nodos

Define la estructura básica de un nodo en el árbol de decisión. Puede ser un nodo de decisión (con un índice de característica y un umbral para tomar decisiones) o un nodo hoja (con un valor específico o etiqueta).
"""

# Definición de la clase Nodo para el árbol de decisión
class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        ''' constructor '''
        # NODO DE DECISION
        # Condition
        self.feature_index = feature_index
        self.threshold = threshold
        # Accesar a las ramas
        self.left = left
        self.right = right
        # Guarda la informacion de la separacion de los datos
        self.info_gain = info_gain

        # Propiedad para nodos hoja
        self.value = value

"""### Clase de Arbo

Esto es esencialmente el corazón del algoritmo. Aquí se definen las funciones para:

- Construir el árbol de decisión de forma recursiva.
- Encontrar la mejor división en un conjunto de datos dado.
- Realizar divisiones basadas en características y umbrales.
- Calcular ganancia de información utilizando entropía o índice de Gini.
- Hacer predicciones en nuevos datos.
"""

# Definición de la clase del Árbol de Decisión
class DecisionTreeClassifier():
    def __init__(self, min_samples_split=2, max_depth=2):
        ''' constructor '''

        # INICIALIZA LA RUTA DEL ARBOL
        self.root = None

        # CONDICIONES DE PARO
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    def build_tree(self, dataset, curr_depth=0):
        ''' recursive function to build the tree '''

        X, Y = dataset[:,:-1], dataset[:,-1]
        num_samples, num_features = np.shape(X)

        # Separa hasta que las condiciones se cumplan
        if num_samples>=self.min_samples_split and curr_depth<=self.max_depth:
            # Encuentra la mejor fragmento
            best_split = self.get_best_split(dataset, num_samples, num_features)
            # Checa si la informacion es positiva
            if best_split["info_gain"]>0:
                # Izquierda
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth+1)
                # Derecha
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth+1)
                # Regresar decisiones para el NODO
                return Node(best_split["feature_index"], best_split["threshold"],
                            left_subtree, right_subtree, best_split["info_gain"])

        # Computar NODO hoja
        leaf_value = self.calculate_leaf_value(Y)
        # Regresa NODO hoja
        return Node(value=leaf_value)

    def get_best_split(self, dataset, num_samples, num_features):
        ''' function to find the best split '''

        # Diccionario para guardar la mejor fragmento
        best_split = {}
        max_info_gain = -float("inf")

        # Loop
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            # loop over all the feature values present in the data
            for threshold in possible_thresholds:
                # Obten la fragmento actual
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                # Checa si los hijos no son NULOS
                if len(dataset_left)>0 and len(dataset_right)>0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    # Computa informacion de ganancia
                    curr_info_gain = self.information_gain(y, left_y, right_y, "gini")
                    # Actualiza la mejor fragmento si es necesario
                    if curr_info_gain>max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["info_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain

        # Regresa la mejor fragmento
        return best_split

    def split(self, dataset, feature_index, threshold):
        ''' function to split the data '''

        dataset_left = np.array([row for row in dataset if row[feature_index]<=threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index]>threshold])
        return dataset_left, dataset_right

    def information_gain(self, parent, l_child, r_child, mode="entropy"):
        ''' function to compute information gain '''

        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        if mode=="gini":
            gain = self.gini_index(parent) - (weight_l*self.gini_index(l_child) + weight_r*self.gini_index(r_child))
        else:
            gain = self.entropy(parent) - (weight_l*self.entropy(l_child) + weight_r*self.entropy(r_child))
        return gain

    def entropy(self, y):
        ''' function to compute entropy '''

        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy

    def gini_index(self, y):
        ''' function to compute gini index '''

        class_labels = np.unique(y)
        gini = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            gini += p_cls**2
        return 1 - gini

    def calculate_leaf_value(self, Y):
        ''' function to compute leaf node '''

        Y = list(Y)
        return max(Y, key=Y.count)

    def print_tree(self, tree=None, indent=" "):
        ''' function to print the tree '''

        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)

        else:
            print("X_"+str(tree.feature_index), "<=", tree.threshold, "?", tree.info_gain)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)

    def fit(self, X, Y):
        ''' function to train the tree '''

        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset)

    def predict(self, X):
        ''' function to predict new dataset '''

        preditions = [self.make_prediction(x, self.root) for x in X]
        return preditions

    def make_prediction(self, x, tree):
        ''' function to predict a single data point '''

        if tree.value!=None: return tree.value
        feature_val = x[tree.feature_index]
        if feature_val<=tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)

"""### Train-Test

Se divide el conjunto de datos Iris en entrenamiento y prueba usando scikit-learn.
"""

# Separar los datos en características y etiquetas
X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values.reshape(-1,1)

# Dividir los datos en conjuntos de entrenamiento y prueba
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=41)

"""### Fit el modelo

Se realiza una validación cruzada en varias profundidades de árbol para encontrar la profundidad óptima. Esto se visualiza en un gráfico que muestra la precisión media en función de la profundidad del árbol.
"""

# Inicializar y entrenar el clasificador del árbol de decisión
classifier = DecisionTreeClassifier(min_samples_split=3, max_depth=3)
classifier.fit(X_train,Y_train)
classifier.print_tree()

"""### Testear el modelo"""

# Predecir usando el conjunto de prueba y calcular la precisión
Y_pred = classifier.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(Y_test, Y_pred)

"""### Métricas de validación

Se calculan y visualizan diversas métricas de evaluación como la matriz de confusión, el informe de clasificación y, en caso de clasificación binaria, el AUC y la curva ROC.
"""

# Definir k-fold cross-validation
# Sección para validar el modelo utilizando validación cruzada y visualizar el rendimiento
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Listas para almacenar resultados
depths = list(range(1, 10))
mean_accuracies = []

# Probar diferentes profundidades del árbol
# Realizar validación cruzada en diferentes profundidades del árbol
for depth in depths:
    fold_accuracies = []
    for train_index, val_index in kf.split(X):
        X_train_fold, X_val_fold = X[train_index], X[val_index]
        Y_train_fold, Y_val_fold = Y[train_index], Y[val_index]

        classifier = DecisionTreeClassifier(min_samples_split=3, max_depth=depth)
        classifier.fit(X_train_fold, Y_train_fold)
        Y_pred_fold = classifier.predict(X_val_fold)

        fold_accuracies.append(accuracy_score(Y_val_fold, Y_pred_fold))

    mean_accuracies.append(np.mean(fold_accuracies))

# Gráfico de rendimiento en función de la profundidad del árbol
plt.plot(depths, mean_accuracies)
plt.xlabel('Depth of Tree')
plt.ylabel('Mean Accuracy')
plt.title('Performance vs Depth of Tree')
plt.show()

"""### Metricas de evaluacion

1. Matriz de Confusión (confusion_matrix).
2. Reporte de Clasificación (classification_report), que incluye precisión, sensibilidad, valor-F1, entre otros.
3. Área Bajo la Curva (AUC) del Receiver Operating Characteristic (ROC).

"""

from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve

# 1. Matriz de Confusión
conf_matrix = confusion_matrix(Y_test, Y_pred)
print("Matriz de Confusión:")
print(conf_matrix)

# 2. Reporte de Clasificación
class_report = classification_report(Y_test, Y_pred)
print("\nReporte de Clasificación:")
print(class_report)

# AUC-ROC para Clasificación binaria
if len(np.unique(Y_test)) == 2:  # Si es binario
    # 3. AUC del ROC
    auc_roc = roc_auc_score(Y_test, Y_pred)
    print("\nAUC-ROC:", auc_roc)

    # Curva ROC
    fpr, tpr, _ = roc_curve(Y_test, Y_pred)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', label='Curva ROC (área = %0.2f)' % auc_roc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()

"""### Curva de aprendizaje"""

from sklearn.model_selection import learning_curve
from sklearn.tree import DecisionTreeClassifier


train_sizes, train_scores, val_scores = learning_curve(
    DecisionTreeClassifier(min_samples_split=3, max_depth=3),
    X, Y,
    cv=5,
    scoring='accuracy',
    train_sizes=np.linspace(0.1, 1.0, 10)
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')

plt.plot(train_sizes, val_mean, color='green', linestyle='--', marker='s', markersize=5, label='validation accuracy')
plt.fill_between(train_sizes, val_mean + val_std, val_mean - val_std, alpha=0.15, color='green')

plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

"""### Visualización del árbol"""

from sklearn.tree import export_graphviz
import graphviz
'''
# Entrena un árbol de decisión usando sklearn
from sklearn.tree import DecisionTreeClassifier as SkDecisionTreeClassifier

clf = SkDecisionTreeClassifier(max_depth=3)
clf.fit(X_train, Y_train)

dot_data = export_graphviz(clf, out_file=None,
                           feature_names=col_names[:-1],
                           class_names=np.unique(Y).astype(str),
                           filled=True, rounded=True,
                           special_characters=True)
graph = graphviz.Source(dot_data)
graph.view()
'''
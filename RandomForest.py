# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 16:38:37 2022

@author: Mario
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier


#Definicion de los hiperparámetros a tener en cuenta de RandomForest
#   criterion: Utilizamos 'entropy' por ser la estudiada en clase
#   random_state: Para obtener siempre los mismo resultados
#   n_estimators: Número de árboles utilizados. Tan alto como el tiempo de ejecucion
#nos permita
#   max_features: Máximo número de variables aleatorias usadas para crear  
#árboles. Como opciones tenemos Auto, sqrt (o log que son similares)
#   min_sample_leaf: Mínimo número de ejemplos requeridos para llegar a un nodo hoja.
#Por defecto es 1.
#   max_depth: Máxima profundidad del árbol, al aumentar la profundidad el train aumenta pq aumenta el sesgo,
#pero el test disminuye pq aumenta la varianza
#   min_sample_split: By increasing the value of the min_sample_split, we can reduce the number of splits 
#that happen in the decision tree and therefore prevent the model from overfitting
#   max_terminal_nodes tb restringe el tamaño del árbol
def RandomForest(X_train,X_test,y_train,y_test,h_models):
    parameters = {'criterion' : ['entropy'], 'random_state' : [10], 'n_estimators' : [10,100,200],
                   'max_features' : ['auto','sqrt'], 'min_samples_leaf' : [1,10,50], 'bootstrap':True}
    
    
    
    cont = 0
    for n_estimators in parameters['n_estimators']:
        for min_samples_leaf in parameters['min_samples_leaf']:
            for max_features in parameters['max_features']:
                print(n_estimators)
                RF = RandomForestClassifier(criterion='entropy',
                                    
                                        n_estimators = n_estimators, 
                                        max_features=max_features,
                                        min_samples_leaf=min_samples_leaf,
                                        bootstrap=True)
                RF.fit(X_train, y_train)
                h_models.append(RF)
                print(RF.score(X_train,y_train))
                cont = cont + 1
    
    

    
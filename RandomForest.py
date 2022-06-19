# -*- coding: utf-8 -*-
"""
Edited on Sat Jun 18 21:55:37 2022

@author: Mario
@co-author: Brian
"""
import pandas as pd
from prettytable import PrettyTable
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate

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
def RandomForest(X_train,y_train,scoring=['f1_micro','roc_auc_ovr'],
    parameters= {'criterion' : ['entropy'], 'random_state' : [10],
        'n_estimators' : [10,100,200],'max_features' : ['sqrt'],
        'min_samples_leaf' : [1,10,50], 'bootstrap':True}):
    maximo = cont = 0
    rf_models_table = PrettyTable()
    rf_models_table.field_names = ["Parámetros","Media F1","Media AUC_ROC"]
    max_p = {}
    for ne in parameters['n_estimators']:
        for msl in parameters['min_samples_leaf']:
            for mf in parameters['max_features']:
                par = "n_estimator="+ str(ne) + " min_samples_leaf=" + str(msl) +" \nmax_features=" + str(mf)
                RF = RandomForestClassifier(
                        criterion='entropy',
                        n_estimators = ne,
                        max_features=mf,
                        min_samples_leaf=msl,
                        bootstrap=True,
                        random_state = 42,
                        n_jobs=2)
                cv_results = cross_validate(RF,X_train,y_train,cv=5,scoring=scoring)
                rf_models_table.add_row([par,cv_results['test_f1_micro'].mean(),cv_results['test_roc_auc_ovr'].mean()])
                if maximo < cv_results['test_roc_auc_ovr'].mean():
                    maximo = cv_results['test_roc_auc_ovr'].mean()
                    max_p  = {'n_estimators':ne,'min_samples_leaf':msl,'max_features':mf,'criterio':'entropy'}
                    bestRF = RF

    print(rf_models_table)
    return bestRF, max_p

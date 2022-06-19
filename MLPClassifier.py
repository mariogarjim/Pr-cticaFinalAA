# -*- coding: utf-8 -*-
"""
Edited on Sat Jun 19 10:30:37 2022

@author: Brian
@co-author: Mario
"""
import pandas as pd
from prettytable import PrettyTable
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate

def MultiLayerPerceptron(X_train,y_train,scoring=['f1_micro','roc_auc_ovr'],
        parameters= {'batch_size' : ['auto'], 'learning_rate':['constant','adaptive'],
            'early_stopping':True, 'max_iter':[300], 'hidden_layer_sizes':[50,75,100]
            } ):
    maximo = cont = 0
    mlp_models_table = PrettyTable()
    mlp_models_table.field_names = ["Parámetros","Media F1","Media AUC_ROC"]
    max_p = {}
    for bs in parameters['batch_size']:
        for lr in parameters['learning_rate']:
            for mt in parameters['max_iter']:
                for hls in parameters['hidden_layer_sizes']:
                    par = "batch="+ str(bs) + " lr=" + str(lr) +" \nT=" + str(mt) + " hls=" + str(hls)
                    MLP = MLPClassifier(
                            activation='tanh',
                            hidden_layer_sizes=hls,
                            max_iter=mt,
                            learning_rate=lr,
                            early_stopping=parameters['early_stopping'],
                            batch_size=bs,
                            random_state = 42
                            )
                    cv_results = cross_validate(MLP,X_train,y_train,cv=5,scoring=scoring)
                    mlp_models_table.add_row([par,cv_results['test_f1_micro'].mean(),cv_results['test_roc_auc_ovr'].mean()])
                    if maximo < cv_results['test_roc_auc_ovr'].mean():
                        maximo = cv_results['test_roc_auc_ovr'].mean()
                        max_p  = {
                                'batch_size':bs,'learning_rate':lr,
                                'max_iter':mt,'hidden_layer_sizes':hls}
                        bestMLP = MLP

    print(mlp_models_table)
    return bestMLP, max_p

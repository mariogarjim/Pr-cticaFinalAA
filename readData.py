# Definimos las funciones de lectura de datos
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import learning_curve 

def LearningCurves(bestRF,X_train,y_train):
    #Curvas de aprendizaje
    train_sizes, train_scores, test_scores = learning_curve(bestRF,X_train,y_train,cv=5,
                                                            train_sizes=np.linspace(0.01, 1.0, 20), scoring='f1_micro')

    train_error = 1 - train_scores
    test_error = 1 - test_scores

    train_mean = np.mean(train_error, axis=1)

    test_mean = np.mean(test_error, axis=1)

    
    plt.subplots(1)
    plt.plot(train_sizes, train_mean,  color="blue",  label="Ein")
    plt.plot(train_sizes, test_mean, '--', color="red", label="Eout")
    
    plt.title("Curvas de Aprendizaje")
    plt.xlabel("Tamaño del conjunto de entrenamiento"), plt.ylabel("Error Esperado"), plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

def Cmatrix(y_train,y_pred,labels=[0,1,2,3,4,5,6,7,8,9],title="Modelo Desconocido"):
    lrcm = confusion_matrix(y_train,y_pred,labels=labels)
    df_lrcm = pd.DataFrame(lrcm, columns = labels)
    plt.figure(figsize = (10,7))
    sns.heatmap(df_lrcm,annot=True,fmt="d",cbar=False,linewidths=.5, cmap="YlGnBu")
    plt.title("Matriz de confusión " + title)
    plt.show()

    return lrcm

def Percentages(lrcm,y_train,title="Modelo Desconocido"):
    lrcm = np.array(lrcm,dtype=np.float)
    porcentajesLR = []
    cont = 1
    for i in range(1,10):
        value = round(lrcm[cont,cont]/len(np.where(y_train==i)[0])*100,2)
        porcentajesLR.append(value)
        lrcm[cont,cont] = value
        cont = cont + 1

    lrcm[0,0] = round(float(lrcm[0,0]) / float(len(np.where(y_train==0)[0])) * 100,2)
    print("############## Porcentajes de Classificación " + title + " #########")
    for c,p in enumerate(porcentajesLR):
        print("Clase: {} {:.2f}%".format(c,round(p,2)))
    print("##########################################################")

def readAll(user="All",targets=False,path="./datos/"):
    onlyfiles = [f for f in listdir(path) if  isfile(join(path,f))]
    onlyfiles.sort()
    data = []
    searching_for = "datapoints"
    h = 0;
    if targets:
        searching_for = "targets"
        h = None;
    if user=="All":
        cont = 0
        for c,f in enumerate(onlyfiles):
            if targets: c=c-1
            if f.find(searching_for)!=-1 and f.find("a")==0:
                #print(c,"Adding:",f)
                temp = pd.read_csv(path+f,header=h,delimiter=" ")
                if targets:
                    cont = cont + 1
                    temp = temp.replace([1],cont);
                data.append(temp)
                f = "b" + f[1:]
                #print(c,"Adding:", f)
                temp = pd.read_csv(path+f,header=h,delimiter=" ")
                if targets:
                    temp = temp.replace([1],cont);
                data.append(temp)
                #print("-----")
    else:
        cont = 1
        for f in onlyfiles:
            if f.find(searching_for)!=-1 and f.find(user)==0:
                # print("Adding:",f)
                temp = pd.read_csv(path+f,header=h,delimiter=" ")
                if targets:
                    temp = temp.replace([1],cont);
                    cont = cont + 1
                data.append(temp)

    data = pd.concat(data,ignore_index=True)
    return data

def load_affirm_data(user="All",path="./datos/"):
    if user!="All":
        return pd.read_csv(path+user+"_affirmative_datapoints.txt",header=0,delimiter=" ")
    else:
        a = pd.read_csv(path+"a_affirmative_datapoints.txt",header=0,delimiter=" ")
        b = pd.read_csv(path+"b_affirmative_datapoints.txt",header=0,delimiter=" ")
        return pd.concat([a,b],ignore_index=True)

def load_affirm_target(user="All",path="./datos/"):
    if user!="All":
        return pd.read_csv(path+user+"_affirmative_targets.txt",header=None,delimiter=" ")
    else:
        a = pd.read_csv(path+"a_affirmative_targets.txt",header=None,delimiter=" ")
        b = pd.read_csv(path+"b_affirmative_targets.txt",header=None,delimiter=" ")
        return pd.concat([a,b],ignore_index=True)

def load_cond_data(user="All",path="./datos/"):
    if user!="All":
        return pd.read_csv(path+user+"_conditional_datapoints.txt",header=0,delimiter=" ")
    else:
        a = pd.read_csv(path+"a_conditional_datapoints.txt",header=0,delimiter=" ")
        b = pd.read_csv(path+"b_conditional_datapoints.txt",header=0,delimiter=" ")
        return pd.concat([a,b],ignore_index=True)

def load_cond_target(user="All",path="./datos/"):
    if user!="All":
        return pd.read_csv(path+user+"_conditional_targets.txt",header=None,delimiter=" ")
    else:
        a = pd.read_csv(path+"a_conditional_targets.txt",header=None,delimiter=" ")
        b = pd.read_csv(path+"b_conditional_targets.txt",header=None,delimiter=" ")
        return pd.concat([a,b],ignore_index=True)


def load_doubtq_data(user="All",path="./datos/"):
    if user!="All":
        return pd.read_csv(path+user+"_doubt_question_datapoints.txt",header=0,delimiter=" ")
    else:
        a = pd.read_csv(path+"a_doubt_question_datapoints.txt",header=0,delimiter=" ")
        b = pd.read_csv(path+"b_doubt_question_datapoints.txt",header=0,delimiter=" ")
        return pd.concat([a,b],ignore_index=True)

def load_doubtq_target(user="All",path="./datos/"):
    if user!="All":
        return pd.read_csv(path+user+"_doubt_question_targets.txt",header=None,delimiter=" ")
    else:
        a = pd.read_csv(path+"a_doubt_question_targets.txt",header=None,delimiter=" ")
        b = pd.read_csv(path+"b_doubt_question_targets.txt",header=None,delimiter=" ")
        return pd.concat([a,b],ignore_index=True)


def load_emphasis_data(user="All",path="./datos/"):
    if user!="All":
        return pd.read_csv(path+user+"_emphasis_datapoints.txt",header=0,delimiter=" ")
    else:
        a = pd.read_csv(path+"a_emphasis_datapoints.txt",header=0,delimiter=" ")
        b = pd.read_csv(path+"b_emphasis_datapoints.txt",header=0,delimiter=" ")
        return pd.concat([a,b],ignore_index=True)

def load_emphasis_target(user="All",path="./datos/"):
    if user!="All":
        return pd.read_csv(path+user+"_emphasis_targets.txt",header=None,delimiter=" ")
    else:
        a = pd.read_csv(path+"a_emphasis_targets.txt",header=None,delimiter=" ")
        b = pd.read_csv(path+"b_emphasis_targets.txt",header=None,delimiter=" ")
        return pd.concat([a,b],ignore_index=True)


def load_neg_data(user="All",path="./datos/"):
    if user!="All":
        return pd.read_csv(path+user+"_negative_datapoints.txt",header=0,delimiter=" ")
    else:
        a = pd.read_csv(path+"a_negative_datapoints.txt",header=0,delimiter=" ")
        b = pd.read_csv(path+"b_negative_datapoints.txt",header=0,delimiter=" ")
        return pd.concat([a,b],ignore_index=True)

def load_neg_target(user="All",path="./datos/"):
    if user!="All":
        return pd.read_csv(path+user+"_negative_targets.txt",header=None,delimiter=" ")
    else:
        a = pd.read_csv(path+"a_negative_targets.txt",header=None,delimiter=" ")
        b = pd.read_csv(path+"b_negative_targets.txt",header=None,delimiter=" ")
        return pd.concat([a,b],ignore_index=True)


def load_rel_data(user="All",path="./datos/"):
    if user!="All":
        return pd.read_csv(path+user+"_relative_datapoints.txt",header=0,delimiter=" ")
    else:
        a = pd.read_csv(path+"a_relative_datapoints.txt",header=0,delimiter=" ")
        b = pd.read_csv(path+"b_relative_datapoints.txt",header=0,delimiter=" ")
        return pd.concat([a,b],ignore_index=True)

def load_rel_target(user="All",path="./datos/"):
    if user!="All":
        return pd.read_csv(path+user+"_relative_targets.txt",header=None,delimiter=" ")
    else:
        a = pd.read_csv(path+"a_relative_targets.txt",header=None,delimiter=" ")
        b = pd.read_csv(path+"b_relative_targets.txt",header=None,delimiter=" ")
        return pd.concat([a,b],ignore_index=True)


def load_topics_data(user="All",path="./datos/"):
    if user!="All":
        return pd.read_csv(path+user+"_topics_datapoints.txt",header=0,delimiter=" ")
    else:
        a = pd.read_csv(path+"a_topics_datapoints.txt",header=0,delimiter=" ")
        b = pd.read_csv(path+"b_topics_datapoints.txt",header=0,delimiter=" ")
        return pd.concat([a,b],ignore_index=True)

def load_topics_target(user="All",path="./datos/"):
    if user!="All":
        return pd.read_csv(path+user+"_topics_targets.txt",header=None,delimiter=" ")
    else:
        a = pd.read_csv(path+"a_topics_targets.txt",header=None,delimiter=" ")
        b = pd.read_csv(path+"b_topics_targets.txt",header=None,delimiter=" ")
        return pd.concat([a,b],ignore_index=True)


def load_wh_data(user="All",path="./datos/"):
    if user!="All":
        return pd.read_csv(path+user+"_wh_question_datapoints.txt",header=0,delimiter=" ")
    else:
        a = pd.read_csv(path+"a_wh_question_datapoints.txt",header=0,delimiter=" ")
        b = pd.read_csv(path+"b_wh_question_datapoints.txt",header=0,delimiter=" ")
        return pd.concat([a,b],ignore_index=True)

def load_wh_target(user="All",path="./datos/"):
    if user!="All":
        return pd.read_csv(path+user+"_wh_question_targets.txt",header=None,delimiter=" ")
    else:
        a = pd.read_csv(path+"a_wh_question_targets.txt",header=None,delimiter=" ")
        b = pd.read_csv(path+"b_wh_question_targets.txt",header=None,delimiter=" ")
        return pd.concat([a,b],ignore_index=True)


def load_yn_data(user="All",path="./datos/"):
    if user!="All":
        return pd.read_csv(path+user+"_yn_question_datapoints.txt",header=0,delimiter=" ")
    else:
        a = pd.read_csv(path+"a_yn_question_datapoints.txt",header=0,delimiter=" ")
        b = pd.read_csv(path+"b_yn_question_datapoints.txt",header=0,delimiter=" ")
        return pd.concat([a,b],ignore_index=True)

def load_yn_target(user="All",path="./datos/"):
    if user!="All":
        return pd.read_csv(path+user+"_yn_question_targets.txt",header=None,delimiter=" ")
    else:
        a = pd.read_csv(path+"a_yn_question_targets.txt",header=None,delimiter=" ")
        b = pd.read_csv(path+"b_yn_question_targets.txt",header=None,delimiter=" ")
        return pd.concat([a,b],ignore_index=True)

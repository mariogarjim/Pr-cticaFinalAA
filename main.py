import readData as rd
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression



#Leemos los datos de los distintos archivos
ydata_table = []

affirdd = rd.load_affirm_data()
affirdt = rd.load_affirm_target()
ydata_table.append(["Affirmación",affirdt.value_counts()])

condd = rd.load_cond_data()
condt = rd.load_cond_target()
ydata_table.append(["Condicional",condt.value_counts()])
condt = condt.replace([1],2)

doubtqd = rd.load_doubtq_data()
doubtqt = rd.load_doubtq_target()
ydata_table.append(["Duda",doubtqt.value_counts()])
doubtqt = doubtqt.replace([1],3)

emphd = rd.load_emphasis_data()
empht = rd.load_emphasis_target()
ydata_table.append(["Énfasis",empht.value_counts()])
empht = empht.replace([1],4)

negd = rd.load_neg_data()
negt = rd.load_neg_target()
ydata_table.append(["Negación",negt.value_counts()])
negt = negt.replace([1],5)


reld = rd.load_rel_data()
relt = rd.load_rel_target()
ydata_table.append(["Relativa", relt.value_counts()])
relt = relt.replace([1],6)

topicsd = rd.load_topics_data()
topicst = rd.load_topics_target()
ydata_table.append(["Tópicos",topicst.value_counts()])
topicst = topicst.replace([1],7)

whd = rd.load_wh_data()
wht = rd.load_wh_target()
ydata_table.append(["Preguntas abiertas",wht.value_counts()])
wht = wht.replace([1],8)

ynd = rd.load_yn_data()
ynt = rd.load_yn_target()
ynt = ynt.replace([1],9)

print("###################### Tabla de distribución #####################")
table = PrettyTable()
table.field_names = ["Clase", "Negativo", "Positivo"]
for data in ydata_table:
    table.add_row([data[0],data[1][0],data[1][1]])

print(table)
print("################### FIN Tabla de distribución #####################")



#Concatenamos los ejemplos en una misma tabla
data = pd.concat([affirdd,condd,doubtqd,emphd,negd,reld,topicsd,whd,ynd], ignore_index=True)
print("############# Verificamos correcta lectura de datos ###########")
print(data)
print("###############################################################")

print("############ Quitamos la etiqueta temporal de cada dato #######")
data.drop('0.0',axis=1,inplace=True)
print("###############################################################")

print("################ Buscamos datos faltantes ###################")
print("Número de datos faltantes:\n", data.isnull().sum())
print("#############################################################")

#Concatenamos las etiquetas en una tabla
y = pd.concat([affirdt,condt,doubtqt,empht,negt,relt,topicst,wht,ynt], ignore_index=True)

#Dividimos en train y test
X_train,X_test,y_train,y_test = train_test_split(data.iloc[:,1:],y,stratify=y)


# Verificamos la distribución de datos obtenidas en training
y_train = np.ravel(y_train)
print("################# Distribución Training Entre Casos Positivos ###############")
sns.histplot(y_train[np.where(y_train>0)],kde=True)
plt.show()
string = input("Press any key to continue")
print("#####################################################################")
print("######################### Distribución Training Total ########################")
sns.histplot(y_train[np.where(y_train>=0)],kde=True)
plt.show()
string = input("Press any key to continue")
print("#####################################################################")

#Escalado
scaler = StandardScaler()
scaler.fit(X_train)
X_train = pd.DataFrame(scaler.transform(X_train))
X_test = pd.DataFrame(scaler.transform(X_test))

y_train = np.ravel(y_train)

parametersLR = {'penalty':['l1'], 'solver': ['saga'], 'C': [0.1,0.25,0.5,1,10],
                 'max_iter':[100]}

print("################ Búscando Hiperparámetros RL ###################")
h_models = []
total = 1
for i in parametersLR.keys():
    total = total * len(parametersLR[i])

cont = 0
for C in parametersLR['C']:
    for t in parametersLR['max_iter']:
        for p in parametersLR['penalty']:
            for s in parametersLR['solver']:
                RL =  LogisticRegression(C=C,penalty=p,solver=s, multi_class='multinomial')
                RL.fit(X_train,y_train)
                h_models.append(RL)
                print(RL.score(X_train,y_train))
            cont = cont + 1
            # print(cont/total*'=>',end='')
print("################ FIN Búsqueda RL ###################")

print("################ Error Baseline Moda ###################")
baseline0 = DummyClassifier(strategy="most_frequent")
baseline0.fit(X_train,y_train)
print(f"Error del baseline0: {1-baseline0.score(X_train,y_train)}")
print("#####################################################################")


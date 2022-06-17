import readData as rd
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
<<<<<<< HEAD
from sklearn.linear_model import LogisticRegression
    
=======
>>>>>>> 5593cd7053b05a4511741e57692cf9a49086f530

#Leemos los datos de los distintos archivos
affirdd = rd.load_affirm_data()
affirdt = rd.load_affirm_target()

condd = rd.load_cond_data()
condt = rd.load_cond_target()
condt = condt.replace([1],2)

doubtqd = rd.load_doubtq_data()
doubtqt = rd.load_doubtq_target()
doubtqt = doubtqt.replace([1],3)

emphd = rd.load_emphasis_data()
empht = rd.load_emphasis_target()
empht = empht.replace([1],4)

negd = rd.load_neg_data()
negt = rd.load_neg_target()
negt = negt.replace([1],5)


reld = rd.load_rel_data()
relt = rd.load_rel_target()
relt = relt.replace([1],6)

topicsd = rd.load_topics_data()
topicst = rd.load_topics_target()
topicst = topicst.replace([1],7)

whd = rd.load_wh_data()
wht = rd.load_wh_target()
wht = wht.replace([1],8)

ynd = rd.load_yn_data()
ynt = rd.load_yn_target()
ynt = ynt.replace([1],9)

#Concatenamos los ejemplos en una misma tabla
data = pd.concat([affirdd,condd,doubtqd,emphd,negd,reld,topicsd,whd,ynd], ignore_index=True)
print(data)

#test = rd.readAll()
#print(test)

#Quitamos la columna de tiempo de los frames
data.drop('0.0',axis=1,inplace=True)

#Concatenamos las etiquetas en una tabla
y = pd.concat([affirdt,condt,doubtqt,empht,negt,relt,topicst,wht,ynt], ignore_index=True)
# print(y)
#ytest = rd.readAll(targets=True)
# print(ytest)

#Dividimos en train y test
X_train,X_test,y_train,y_test = train_test_split(data.iloc[:,1:],y,stratify=y)

#Escalado
scaler = StandardScaler()
scaler.fit(X_train)
X_train = pd.DataFrame(scaler.transform(X_train))
X_test = pd.DataFrame(scaler.transform(X_test))


#Regresión Logística
#Parámetros interesantes: penalty (regularización), C (inversa de la regularazación),
#                         max_iter, solver --> Para multiclass: ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’



parametersLR = [{'penalty':['l1'], 'solver': ['saga'], 'C': [0.1,0.25,0.5,1,10], 
                 'max_iter':[100]}]

for C in [0.1,0.25,0.5,1,10]:
    RL =  LogisticRegression(C=C,penalty='l1',solver='saga', multi_class='multinomial')
    RL.fit(X_train,y_train)
    print(RL.score(X_train,y_train))
    







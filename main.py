import readData as rd
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Leemos los datos de los distintos archivos
affirdd = rd.load_affirm_data()
affirdt = rd.load_affirm_target()


condd = rd.load_cond_data()
condt = rd.load_cond_target()

doubtqd = rd.load_doubtq_data()
doubtqt = rd.load_doubtq_target()

emphd = rd.load_emphasis_data()
empht = rd.load_emphasis_target()

negd = rd.load_neg_data()
negt = rd.load_neg_target()

reld = rd.load_rel_data()
relt = rd.load_rel_target()

topicsd = rd.load_topics_data()
topicst = rd.load_topics_target()

whd = rd.load_wh_data()
wht = rd.load_wh_target()

ynd = rd.load_yn_data()
ynt = rd.load_yn_target()

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








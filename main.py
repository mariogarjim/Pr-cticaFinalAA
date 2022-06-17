import readData as rd
import pandas as pd


#Leemos los datos de los distintos archivos
affirdd = rd.load_affirm_data()
#affirdd.head()
affirdt = rd.load_affirm_target()
affirdt.head()


#assert(affirdd.shape[0] == affirdt.shape[0])

condd = rd.load_cond_data()
condd.head()
condt = rd.load_cond_target()
condt.head()

doubtqd = rd.load_doubtq_data()
doubtqd.head()
doubtqt = rd.load_doubtq_target()
doubtqt.head()

emphd = rd.load_emphasis_data()
emphd.head()
empht = rd.load_emphasis_target()
empht.head()

negd = rd.load_neg_data()
negd.head()
negt = rd.load_neg_target()
negt.head()

reld = rd.load_rel_data()
reld.head()
relt = rd.load_rel_target()
relt.head()

topicsd = rd.load_topics_data()
topicsd.head()
topicst = rd.load_topics_target()
topicst.head()

whd = rd.load_wh_data()
whd.head()
wht = rd.load_wh_target()
wht.head()

ynd = rd.load_yn_data()
ynd.head()
ynt = rd.load_yn_target()
ynt.head()

#Concatenamos los ejemplos
data = pd.concat([affirdd,condd,doubtqd,emphd,negd,reld,topicsd,whd,ynd], ignore_index=True)

#Quitamos la columna temporal
data.drop('0.0',axis=1)      

#Concatenamos las etiquetas
y = pd.concat([affirdt,condt,doubtqt,empht,negt,relt,topicst,wht,ynt], ignore_index=True)




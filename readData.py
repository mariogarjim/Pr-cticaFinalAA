# Definimos las funciones de lectura de datos
import pandas as pd
from os import listdir
from os.path import isfile, join

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
                # print(c,"Adding:",f)
                temp = pd.read_csv(path+f,header=h,delimiter=" ")
                if targets:
                    cont = cont + 1
                    temp = temp.replace([1],cont);
                data.append(temp)
                f = "b" + f[1:]
                # print(c,"Adding:", f)
                if targets:
                    temp = temp.replace([1],cont);
                data.append(temp)
                # print("-----:")
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

def load_affirm_data(user="b",path="./datos/"):
    return pd.read_csv(path+user+"_affirmative_datapoints.txt",header=0,delimiter=" ")
def load_affirm_target(user="b",path="./datos/"):
    return pd.read_csv(path+user+"_affirmative_targets.txt",header=None,delimiter=" ")

def load_cond_data(user="b",path="./datos/"):
    return pd.read_csv(path+user+"_conditional_datapoints.txt",header=0,delimiter=" ")
def load_cond_target(user="b",path="./datos/"):
    return pd.read_csv(path+user+"_conditional_targets.txt",header=None,delimiter=" ")

def load_doubtq_data(user="b",path="./datos/"):
    return pd.read_csv(path+user+"_doubt_question_datapoints.txt",header=0,delimiter=" ")
def load_doubtq_target(user="b",path="./datos/"):
    return pd.read_csv(path+user+"_doubt_question_targets.txt",header=None,delimiter=" ")

def load_emphasis_data(user="b",path="./datos/"):
    return pd.read_csv(path+user+"_emphasis_datapoints.txt",header=0,delimiter=" ")
def load_emphasis_target(user="b",path="./datos/"):
    return pd.read_csv(path+user+"_emphasis_targets.txt",header=None,delimiter=" ")

def load_neg_data(user="b",path="./datos/"):
    return pd.read_csv(path+user+"_negative_datapoints.txt",header=0,delimiter=" ")
def load_neg_target(user="b",path="./datos/"):
    return pd.read_csv(path+user+"_negative_targets.txt",header=None,delimiter=" ")

def load_rel_data(user="b",path="./datos/"):
    return pd.read_csv(path+user+"_relative_datapoints.txt",header=0,delimiter=" ")
def load_rel_target(user="b",path="./datos/"):
    return pd.read_csv(path+user+"_relative_targets.txt",header=None,delimiter=" ")

def load_topics_data(user="b",path="./datos/"):
    return pd.read_csv(path+user+"_topics_datapoints.txt",header=0,delimiter=" ")
def load_topics_target(user="b",path="./datos/"):
    return pd.read_csv(path+user+"_topics_targets.txt",header=None,delimiter=" ")

def load_wh_data(user="b",path="./datos/"):
    return pd.read_csv(path+user+"_wh_question_datapoints.txt",header=0,delimiter=" ")
def load_wh_target(user="b",path="./datos/"):
    return pd.read_csv(path+user+"_wh_question_targets.txt",header=None,delimiter=" ")

def load_yn_data(user="b",path="./datos/"):
    return pd.read_csv(path+user+"_yn_question_datapoints.txt",header=0,delimiter=" ")
def load_yn_target(user="b",path="./datos/"):
    return pd.read_csv(path+user+"_yn_question_targets.txt",header=None,delimiter=" ")

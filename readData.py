# Definimos las funciones de lectura de datos
import pandas as pd

def load_affirm_data(user="b",path="./datos/"):
    return pd.read_csv(path+user+"_affirmative_datapoints.txt",header=0,delimiter=" ")
def load_affirm_target(user="b",path="./datos/"):
    return pd.read_csv(path+user+"_affirmative_targets.txt",header=0,delimiter=" ")

def load_cond_data(user="b",path="./datos/"):
    return pd.read_csv(path+user+"_conditional_datapoints.txt",header=0,delimiter=" ")
def load_cond_target(user="b",path="./datos/"):
    return pd.read_csv(path+user+"_conditional_targets.txt",header=0,delimiter=" ")

def load_doubtq_data(user="b",path="./datos/"):
    return pd.read_csv(path+user+"_doubt_question_datapoints.txt",header=0,delimiter=" ")
def load_doubtq_target(user="b",path="./datos/"):
    return pd.read_csv(path+user+"_doubt_question_targets.txt",header=0,delimiter=" ")

def load_emphasis_data(user="b",path="./datos/"):
    return pd.read_csv(path+user+"_emphasis_datapoints.txt",header=0,delimiter=" ")
def load_emphasis_target(user="b",path="./datos/"):
    return pd.read_csv(path+user+"_emphasis_targets.txt",header=0,delimiter=" ")

def load_neg_data(user="b",path="./datos/"):
    return pd.read_csv(path+user+"_negative_datapoints.txt",header=0,delimiter=" ")
def load_neg_target(user="b",path="./datos/"):
    return pd.read_csv(path+user+"_negative_targets.txt",header=0,delimiter=" ")

def load_rel_data(user="b",path="./datos/"):
    return pd.read_csv(path+user+"_relative_datapoints.txt",header=0,delimiter=" ")
def load_rel_target(user="b",path="./datos/"):
    return pd.read_csv(path+user+"_relative_targets.txt",header=0,delimiter=" ")

def load_topics_data(user="b",path="./datos/"):
    return pd.read_csv(path+user+"_topics_datapoints.txt",header=0,delimiter=" ")
def load_topics_target(user="b",path="./datos/"):
    return pd.read_csv(path+user+"_topics_targets.txt",header=0,delimiter=" ")

def load_wh_data(user="b",path="./datos/"):
    return pd.read_csv(path+user+"_wh_question_datapoints.txt",header=0,delimiter=" ")
def load_wh_target(user="b",path="./datos/"):
    return pd.read_csv(path+user+"_wh_question_targets.txt",header=0,delimiter=" ")

def load_yn_data(user="b",path="./datos/"):
    return pd.read_csv(path+user+"_yn_question_datapoints.txt",header=0,delimiter=" ")
def load_yn_target(user="b",path="./datos/"):
    return pd.read_csv(path+user+"_yn_question_targets.txt",header=0,delimiter=" ")


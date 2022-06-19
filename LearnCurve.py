from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt
import numpy as np

def learning_curve_data(clf,k,Xtrain,ytrain,test,ytest,Ein_Loss=None):
  batch_size = int(Xtrain.shape[0]/k)
  x = []
  y = []
  yt = []
  for i in range(0,Xtrain.shape[0],batch_size):
    dt = Xtrain[0:i+batch_size]
    lb = ytrain[0:i+batch_size]
    clf.fit(dt,lb)
    x.append(i+batch_size)
    if(Ein_Loss!=None):
      y.append(Ein_Loss(lb,clf.predict(dt)))
      yt.append(Ein_Loss(ytest,clf.predict(test)))
    else:
      y.append(1-clf.score(dt,lb))
      yt.append(1-clf.score(test,ytest))

  return x,y,yt

def plot_slrcurve(x,y,yt):
  x_new = np.linspace(0,30000,300)
  a_BSpline = make_interp_spline(x,y)
  y_new = a_BSpline(x_new)
  plt.plot(x_new,y_new,c="blue",label='E_in')
  a_BSpline = make_interp_spline(x,yt)
  y_t = a_BSpline(x_new)
  plt.plot(x_new,y_t,c="red",label='E_out')
  plt.legend()
  plt.xlabel("Tamaño del conjunto de entrenamiento")
  plt.ylabel("Error esperado")

def plot_slrcurve(x,y,yt):
  x_new = np.linspace(0,30000,300)
  a_BSpline = make_interp_spline(x,y)
  y_new = a_BSpline(x_new)
  plt.plot(x_new,y_new,c="blue",label='E_in')
  a_BSpline = make_interp_spline(x,yt)
  y_t = a_BSpline(x_new)
  plt.plot(x_new,y_t,c="red",label='E_out')
  plt.legend()
  plt.xlabel("Tamaño del conjunto de entrenamiento")
  plt.ylabel("Error esperado")

def plot_lrcurve(x,y,yt):
  plt.plot(x,y,c="blue",label="E_in")
  plt.plot(x,yt,c="red",label="E_out")
  plt.legend()
  plt.xlabel("Tamaño del conjunto de entrenamiento")
  plt.ylabel("Error esperado")


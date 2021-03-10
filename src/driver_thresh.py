import numpy as np, pandas as pd, matplotlib.pyplot as plt 

df=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None)


x=df.iloc[0:100,0].values
y=df.iloc[0:100,4].values
y=np.where(y=="Iris-setosa",1,-1)

from ML import Threshold
t=Threshold()
t.find_cutoff(x,y)

print(t.max_thresh)
z=df.iloc[100:150,0].values

print(t.predict_label(z))

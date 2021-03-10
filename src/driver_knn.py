import numpy as np, pandas as pd, matplotlib.pyplot as plt 

df=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None)


#get even numbered elems for training data
x=df.iloc[0:100:2,[0,1,2,3]].values
y=df.iloc[0:100:2,4].values
y=np.where(y=="Iris-setosa",1,-1)

queries=df.iloc[1:101:2,[0,1,2,3]].values

from ML import KNN
k=KNN(x,y)
print("Query Classes",k.predict(3,queries))
print("Query Means", k.predict(3,queries,"e","r"))
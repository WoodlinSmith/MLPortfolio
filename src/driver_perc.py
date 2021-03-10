import numpy as np, pandas as pd, matplotlib.pyplot as plt 

df=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None)


x=df.iloc[0:100,[0,2]].values
y=df.iloc[0:100,4].values

y=np.where(y=='Iris-setosa',-1,1)

from ML import Perceptron
p=Perceptron(0.1,10)
p.fit(x,y)

print("Errors:",p.errors)
print("Weights:", p.weights)

p.net_input(x)

p.predict(x)

from ML import plot_decision_regions
plot_decision_regions(x,y,p)

import numpy as np, pandas as pd, matplotlib.pyplot as plt 

df=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None)


x=df.iloc[0:100,[0,2]].values
y=df.iloc[0:100,4].values

y=np.where(y=='Iris-setosa',-1,1)

from ML import SupportVectorMachine
svm=SupportVectorMachine()
svm.fit(x,y)

print("Weights:", svm.weights)



svm.predict(x)

from ML import plot_decision_regions
plot_decision_regions(x,y,svm)
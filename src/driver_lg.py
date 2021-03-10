import numpy as np, pandas as pd, matplotlib.pyplot as plt 

df=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None)


x=df.iloc[0:50,1].values
y=df.iloc[0:50,0].values

from ML import LinearRegression
lg=LinearRegression()
lg.fit(x,y)

print("Weights:", lg.weights)

print("Predicted value when X=3.5: "+str(lg.predict(3.5)))
print("Actual value: 5.1")

print("R^2 value: ")
print(lg.calc_rsquared(x))

plt.plot(x,lg.X.dot(lg.weights))
plt.scatter(x,y)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

x = np.array([1,2,3])
y = np.array([3,6,10])
plt.plot(x,y,"r.")

X = np.c_[np.ones((3,1)),x]
theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
xtest = np.array([[0],[2],[4]])
xtestb = np.c_[np.ones((3,1)),xtest]
ypred = xtestb.dot(theta)

plt.plot(xtest , ypred , "b-")
#plt.show()
#================================================
from sklearn.linear_model import LinearRegression
x = np.array([1,2,3]).reshape(-1,1)
y = np.array([3,5,7]).reshape(-1,1)

model = LinearRegression()
model.fit(x,y)

theta0 , theta1 = model.intercept_ , model.coef_

xtest = np.array([[0],[2],[4]]).reshape(-1,1)
ypred = model.predict(xtest)

plt.figure(2)
plt.plot(xtest , ypred , 'r.')
plt.show()



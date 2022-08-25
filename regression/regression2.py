import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as met

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

x = np.array([10,20,30,40,50,60,70,80,90,100])
y = np.array([18,41,61,79,70,120,141,150,120,200])
x.reshape(-1,1)
y.reshape(-1,1)

df = pd.DataFrame(x,y)


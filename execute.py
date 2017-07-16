import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

#obtain dataset
dataframe = pd.read_fwf('data.txt')
x = dataframe[['Brain']]
y = dataframe[['Body']]

#train the model
reg = linear_model.LinearRegression()
reg.fit(x, y)

#results
plt.scatter(x, y)
plt.plot(x, reg.predict(x))
plt.show()

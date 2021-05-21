import pandas as pd
import numpy as np
import matplotlib.pyplot as plt #

dataset = pd.read_csv("./data.csv")
x = dataset.iloc[:,1:2].values
y = dataset.iloc[:, -1].values

# fitting LR to the dataset - creating draft (1st) model
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(x,y) # training the model

# fitting polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
PR = PolynomialFeatures(degree = 5) 
x_poly = PR.fit_transform(x) # changing x model to polynomial fit.transform

LR2 = LinearRegression() # include fit with PolyReg into our linear regression model
LR2.fit(x_poly, y) # LR2 is a linear regression model taken from the x_poly values

"""TO MAKE POYNOMIAL CURVE SMOOTHER"""

# gives a rg from lower bound and uper bound then the increment -> vector
x_grid = np.arange(min(x), max(x), 0.1) # will contain all the levels plus the incremented steps in between 
# we need reshape to make the vector into a matrix with all increments from 1 to 10 incrementing on 0.1
x_grid = x_grid.reshape(len(x_grid),1)

# replace 'x' below with 'x_grid'

""""""
# visualizing the linear regression results and polynomial results
# real values
plt.scatter(x, y, color = "red")
plt.plot(x, LR.predict(x), color = "blue") # (x prediction point, y predicted values)
# add new 'PR.fit_transform(x)' instead of x -> will create new 'x_poly' // its for better practice
plt.plot(x_grid, LR2.predict(PR.fit_transform(x_grid)), color = "orange") # we have to include the polynomial features added to x_poly
plt.title("Linear")
plt.show()

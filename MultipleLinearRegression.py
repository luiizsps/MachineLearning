import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("FuelConsumptionCo2.csv")
df = df[['CO2EMISSIONS', 'ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']]

adf = np.random.rand(len(df)) < 0.8
train = df[adf]
test = df[~adf]

train_x = np.asanyarray(train[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])

from sklearn import linear_model

regre = linear_model.LinearRegression()
regre.fit(train_x, train_y)

print('Coeficientes:', regre.coef_)
print('Intersecção', regre.intercept_)

test_x = np.asanyarray(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])

print('Score: ', regre.score(test_x, test_y))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from sklearn.metrics import r2_score

df = pd.read_csv("FuelConsumptionCo2.csv")
cdf = df[["CYLINDERS", "CO2EMISSIONS"]]
plt.scatter(df.CYLINDERS, df.CO2EMISSIONS)
plt.show()

msk = np.random.rand(len(cdf)) < 0.8
train = cdf[msk]
test = cdf[~msk]

regression = lm.LinearRegression()
train_x = np.asanyarray(train[['CYLINDERS']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])

# calcula coeficientes da reta
regression.fit(train_x, train_y)
coeficient = regression.coef_
interception = regression.intercept_
print('Equação da reta: ',coeficient[0][0],'x + ',interception[0])

plt.scatter(df.CYLINDERS, df.CO2EMISSIONS, color="blue")

# plota reta vermelha no grafico através da previvão de resultados 
plt.plot(train_x, coeficient[0][0]*train_x + interception[0], 'k-')
plt.xlabel("Number of Cylinders")
plt.ylabel("Co2 Emission")
plt.show()

test_x = np.asanyarray(test[['CYLINDERS']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])

# Prevê resultados de emissão baseado no vetor de valores inserido
test_y_ = regression.predict(test_x)

# Calcula erro médio absoluto, soma dos quadrados residuais e score  
print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("Relative squared error: %.2f" % r2_score(test_y , test_y_))


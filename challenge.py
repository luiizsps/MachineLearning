import pandas as pd

path = r"C:\Users\lsps1\Desktop\Python\Challenge\dataset.csv" 
df = pd.read_csv(path)
df.head(4)

# divide o dataset em df_x e df_Y.
df_X = df.drop('Churn', axis=1)
df_Y = df['Churn']

# Exibe a distribuição dos dados de df_Y.
df_Y.value_counts().plot.pie(autopct='%.1f')
df_Y.value_counts()

# Agora dividimos o dataset em conjuntos de treino e teste.
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(df_X, df_Y, test_size = 0.35, random_state=5, stratify=df_Y)

# Testando se a proporção continua a mesma na base de treino
print(y_train.value_counts() / y_train.shape[0])

# testando se o conjunto de teste possui a mesma proporção da base de treino
print(y_test.value_counts() / y_test.shape[0])

# Dividindo o conjunto de teste em teste e validação
x_test, x_validation, y_test, y_validation = train_test_split(x_test, y_test, test_size=0.5, random_state=2, stratify=y_test)

# verificando se o conjunto de teste possue a mesma proporção
print(y_test.value_counts() / y_test.shape[0])

# verificando se o conjunto de validação possue a mesma proporção
print(y_validation.value_counts() / y_validation.shape[0])

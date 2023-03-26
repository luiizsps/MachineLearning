import pandas as pd

path = r"C:\Users\lsps1\Desktop\Python\Challenge\dataset.csv" 
df = pd.read_csv(path)
df.head(4)

# Exibe a distribuição dos dados das variáveis depedendentes
# f['Churn'].value_counts().plot.pie(autopct='%.1f')
df['Churn'].value_counts()


# ------ PREPROCESSING THE DATA ------ #

# Transforma os dados categóricos do alvo y em dados numéricos
df['Churn'] = [0 if x == 'No' else 1 for x in df['Churn']]
print(df['Churn'])
print(df['Churn'].shape)

# Divide o dataset em variáveis independentes e dependentes
df_x = df.drop('Churn', axis=1)
df_y = df['Churn']

print(df_x.head(5))

# Função que analisa os dados categóricos
def catergorias_unicas(df):
  for nome_coluna in df.columns:
    if df[nome_coluna].dtype == 'object':
      num_categorias_unicas = len(df[nome_coluna].unique())
      print(f"{nome_coluna} tem {num_categorias_unicas} categorias únicas")

catergorias_unicas(df_x)

# Transforma os dados categóricos das variáveis indepedentens x em dados numéricos
to_dummy_list = ['gender', 'Partner', 'Dependents', 'PhoneService','MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']

def to_dummy(df, to_dummy_list):
  for x in to_dummy_list:
    dummies = pd.get_dummies(df[x], prefix=x, dummy_na=False)
    df = df.drop(x, axis=1)
    df = pd.concat([df, dummies], axis=1)
  return df

df_x = to_dummy(df_x, to_dummy_list)
print(df_x.head(5))

# Verificando se nosso dataset possui dados faltando
print(df_x.isnull().sum().sort_values(ascending=False).head())
# caso haja dados faltando, podemos utilizar a função Imputer do sklearn.preprocessing para substituir estas posições


# Usando o algoritmo TukeyIQR para identificar outliers
import numpy as np
def find_ouliers_turkey(x):
  q1 = np.percentile(x,25)
  q3 = np.percentile(x,75)
  iqr = q3 - q1
  floor = q1 - 1.5*iqr
  ceiling = q3 + 1.5*iqr
  outlier_indicies = list(x.index[(x < floor) | (x > ceiling)])
  outlier_values = list(x[outlier_indicies])
  
  return outlier_indicies, outlier_values

outlier_indicies, outlier_values = find_ouliers_turkey(df_x['MonthlyCharges'])
print(outlier_values)

outlier_indicies, outlier_values = find_ouliers_turkey(df_x['tenure'])
print(outlier_values)




'''
import matplotlib.pyplot as plt
import seaborn 
seaborn.heatmap(df.corr(),
            annot = True,
            fmt = '.2f',
            cmap='Blues')
plt.show()

# Agora dividimos o dataset em conjuntos de treino e teste.
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(df_X, df_Y, test_size = 0.35, random_state=5, stratify=df_Y)

# Testando se a proporção continua a mesma na base de treino
print(y_train.value_counts() / y_train.shape[0])

# testando se o conjunto de teste possui a mesma proporção da base de treino
print(y_test.value_counts() / y_test.shape[0])

# Dividindo o conjunto de teste em teste e validação
x_test, x_validation, y_test, y_validation = train_test_split(x_test, y_test, test_size=0.5, random_state=2, stratify=y_test)

# verificando se o conjunto de teste possui a mesma proporção
print(y_test.value_counts() / y_test.shape[0])

# verificando se o conjunto de validação possui a mesma proporção
print(y_validation.value_counts() / y_validation.shape[0])

from imblearn.under_sampling import RandomUnderSampler
'''

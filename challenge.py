#!/usr/bin/env python
# coding: utf-8

# In[218]:


import pandas as pd

path = r"C:\Users\lsps1\Desktop\Python\Challenge\dataset.csv" 
df = pd.read_csv(path)
df.head(4)


# In[219]:
import matplotlib.pyplot as plt
# Exibe a distribuição dos dados das variáveis depedendentes
df['Churn'].value_counts().plot.pie(autopct='%.1f')
df['Churn'].value_counts()
plt.show()


# In[220]:


# Transforma os dados categóricos do alvo y em dados numéricos
df['Churn'] = [0 if x == 'No' else 1 for x in df['Churn']]
df['Churn']


# In[221]:


# Divide o dataset
df_x = df.drop('Churn', axis=1)
df_y = df['Churn']

df_x.head(5)


# In[222]:


# Função que analisa os dados categóricos
def catergorias_unicas(df):
  for nome_coluna in df.columns:
    if df[nome_coluna].dtype == 'object':
      num_categorias_unicas = len(df[nome_coluna].unique())
      print(f"{nome_coluna} tem {num_categorias_unicas} categorias únicas")

catergorias_unicas(df_x)


# In[223]:


x = df_x['TotalCharges']
x = x.replace({" ": 0})
x = pd.to_numeric(x)

x


# In[224]:


df_x = df_x.drop('TotalCharges', axis=1)
df_x['TotalCharges'] = x

df_x['TotalCharges']


# In[225]:


# Transforma os dados categóricos das variáveis indepedentens x em dados numéricos
to_dummy_list = ['gender', 'Partner', 'Dependents', 'PhoneService','MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']

def to_dummy(df, to_dummy_list):
  for x in to_dummy_list:
    dummies = pd.get_dummies(df[x], prefix=x, dummy_na=False)
    df = df.drop(x, axis=1)
    df = pd.concat([df, dummies], axis=1)
  return df

df_x = to_dummy(df_x, to_dummy_list)
df_x.head(5)


# In[226]:


# Verificando se nosso dataset possui dados faltando
print(df.isnull().sum().sort_values(ascending=False).head())


# In[227]:


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

outlier_indicies, outlier_values = find_ouliers_turkey(df_x['TotalCharges'])
print(outlier_values)


# In[228]:


outlier_indicies, outlier_values = find_ouliers_turkey(df_x['tenure'])
print(outlier_values)
'''
# ------ FEATURE EXPLORATION ------ #
from itertools import combinations
from sklearn.preprocessing import PolynomialFeatures

# Cria interações entre as features
def add_interactions(df):
  # define nome das novas features
  combos = list(combinations(list(df.columns), 2))
  col_names = list(df.columns) + ['_'.join(x) for x in combos]

  # encontra interações
  poly = PolynomialFeatures(include_bias=False, interaction_only=True)
  df = poly.fit_transform(df)
  df = pd.DataFrame(df)
  df.columns = col_names
  
  # remove interações que possuem apenas zeros
  noint_indicies = [i for i, x in enumerate(list((df == 0).all())) if x]
  df = df.drop(df.columns[noint_indicies], axis=1)
  
  return df

df_x = add_interactions(df_x)
print(df_x, df_x.shape)
'''
# In[229]:


# Agora dividimos o dataset em conjuntos de treino, teste e validação.
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size = 0.35, random_state=5, stratify=df_y)
x_test, x_validation, y_test, y_validation = train_test_split(x_test, y_test, test_size=0.5, random_state=2, stratify=y_test)


# In[230]:

import sklearn.feature_selection

select = sklearn.feature_selection.SelectKBest(k=20)
selected_features = select.fit(x_train, y_train)
indicies_selected = selected_features.get_support(indices=True)
colnames_selected = [df_x.columns[i] for i in indicies_selected]

x_train = x_train[colnames_selected]
x_test = x_test[colnames_selected]
x_validation = x_validation[colnames_selected]

print(colnames_selected)


# In[231]:


x_train['Churn'] = y_train

import seaborn as sb

sb.heatmap(x_train.corr(),
            cmap='Blues')
plt.figure(figsize=(12,8))
plt.show()


# In[232]:


x_train = x_train.drop('Churn', axis=1)
x_train


# In[233]:


from sklearn.preprocessing import MinMaxScaler

# Normalizando os dados
scaler = MinMaxScaler()
scaler.fit(x_train)


# In[234]:


x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_validation = scaler.transform(x_validation)

x_train.shape


# In[235]:


from imblearn.over_sampling import RandomOverSampler

# lidando com o desbalanceamento de dados no conjunto de treino
res = RandomOverSampler(random_state=42)
x_train_res, y_train_res = res.fit_resample(x_train, y_train)

x_train_res.shape
# In[236]:


# Criando modelo
import tensorflow as tf

model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(16, input_dim=20, activation='relu'),
  tf.keras.layers.Dense(16, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid')
])


# In[237]:


model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[tf.keras.metrics.BinaryAccuracy()],
)


# In[238]:


model.fit(
    x_train_res, 
    y_train_res,
    batch_size=32,
    epochs=100,
    validation_data=(x_validation, y_validation),
)

# Retorna acuracia e loss para conjunto de teste
model.evaluate(x_test, y_test)

# Prevê churn baseado no conjunto de teste
predictions = model.predict(x_test)

# Tranforma predições em 0 e 1
prediction_list = []
for i in predictions:
    if i>0.5:
        prediction_list.append(1)
    else:
        prediction_list.append(0)

'''
# cria dataframe
data = {'actual_churn':y_test, 'predicted_churn':prediction_list}
df_pred = pd.DataFrame(data)
df_pred.head(5)
'''

from sklearn.metrics import confusion_matrix, classification_report

#print classification_report
print(classification_report(y_test, prediction_list))

# ploting the confusion metrix plot
conf_mat = tf.math.confusion_matrix(labels=y_test,predictions=prediction_list)
plt.figure(figsize = (14,6))
sb.heatmap(conf_mat, annot=True,fmt='d')
plt.xlabel('Predicted_churn')
plt.ylabel('True_churn')
plt.show()

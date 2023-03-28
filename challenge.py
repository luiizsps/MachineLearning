#!/usr/bin/env python
# coding: utf-8

# In[565]:


import pandas as pd
 
path = r"C:\Users\lsps1\Desktop\Python\Challenge\dataset.csv" 
df = pd.read_csv(path)
df.head(4)


# In[566]:


# Exibe a distribuição dos dados das variáveis depedendentes
df['Churn'].value_counts().plot.pie(autopct='%.1f')
df['Churn'].value_counts()


# In[567]:


# Transforma os dados categóricos do alvo y em dados numéricos
df['Churn'] = [0 if x == 'No' else 1 for x in df['Churn']]
df['Churn']


# In[568]:


# Divide o dataset
df_x = df.drop('Churn', axis=1)
df_y = df['Churn']

df_x.head(5)


# In[569]:


# Função que analisa os dados categóricos
def catergorias_unicas(df):
  for nome_coluna in df.columns:
    if df[nome_coluna].dtype == 'object':
      num_categorias_unicas = len(df[nome_coluna].unique())
      print(f"{nome_coluna} tem {num_categorias_unicas} categorias únicas")

catergorias_unicas(df_x)


# In[570]:


x = df_x['TotalCharges']
x = x.replace({" ": 0})
x = pd.to_numeric(x)

x


# In[571]:


df_x = df_x.drop('TotalCharges', axis=1)
df_x['TotalCharges'] = x

df_x['TotalCharges']


# In[572]:


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


# In[573]:


# Verificando se nosso dataset possui dados faltando
print(df.isnull().sum().sort_values(ascending=False).head())


# In[574]:


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


# In[575]:


outlier_indicies, outlier_values = find_ouliers_turkey(df_x['tenure'])
print(outlier_values)


# In[576]:


# Agora dividimos o dataset em conjuntos de treino, teste e validação.
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size = 0.30, random_state=5, stratify=df_y)
x_test, x_validation, y_test, y_validation = train_test_split(x_test, y_test, test_size=0.5, random_state=2, stratify=y_test)


# In[577]:


import sklearn.feature_selection

select = sklearn.feature_selection.SelectKBest(k=10)
selected_features = select.fit(x_train, y_train)
indicies_selected = selected_features.get_support(indices=True)
colnames_selected = [df_x.columns[i] for i in indicies_selected]

x_train = x_train[colnames_selected]
x_test = x_test[colnames_selected]
x_validation = x_validation[colnames_selected]

print(colnames_selected)


# In[578]:


x_train['Churn'] = y_train

import matplotlib.pyplot as plt
import seaborn 

seaborn.heatmap(x_train.corr(),
            annot = True,
            fmt = '.2f',
            cmap='Blues')
plt.show()


# In[579]:


x_train = x_train.drop('Churn', axis=1)
x_train


# In[580]:


from sklearn.preprocessing import MinMaxScaler

# Normalizando os dados
scaler = MinMaxScaler()
scaler.fit(x_train)


# In[581]:


x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
x_validation = scaler.transform(x_validation)

x_train[0]


# In[582]:


from imblearn.under_sampling import RandomUnderSampler

# lidando com o desbalanceamento de dados no conjunto de treino
res = RandomUnderSampler(random_state=42)
x_train_res, y_train_res = res.fit_resample(x_train, y_train)


# In[583]:


from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier


# In[584]:


def create_model(optimizer='adam'):
    ann = Sequential()
    ann.add(Dense(units=6, input_dim=10, activation='relu', kernel_initializer='he_normal'))
    ann.add(Dense(units=6, activation='relu', kernel_initializer='he_normal'))
    ann.add(Dense(units=1, activation='sigmoid'))
    ann.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    return ann


# In[585]:


# criando modelo
model = KerasClassifier(build_fn=create_model, verbose=2)


# In[586]:


# define os parâmetros do grid search
optimizer = ['SGD', 'Adam']
batch_size = [32, 64, 128]
epochs = [40, 50]
param_grid = dict(optimizer=optimizer, batch_size=batch_size, epochs=epochs)

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=None, cv=5)
grid_result = grid.fit(x_train, y_train)


# In[587]:


# exibe resultados do grid search
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))



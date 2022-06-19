import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# lê arquivo csv com dados de população por sexo em todos os paises do mundo de 1950 a 2019
df = pd.read_csv("WPP2019_PopulationBySingleAgeSex_1950-2019.csv") 

pop_male = np.zeros(70)
pop_fem = np.zeros(70)
i = 0
k = 0
loop = np.arange(1950, 2020, 1)

# encontra a posição de onde inicia os dados sobre o Brasil
while df.loc[i,"Location"] != 'Brazil':
    i+=1
# loop for para incremendar os anos 1950-2019 
for j in loop:
    # while para armazenar os dados em vetores por ano, visto que tem mais de um dado por ano
    while int(df.loc[i,"Time"]) == j:
        pop_male[k] += float(df.loc[i, "PopMale"])
        pop_fem[k] += float(df.loc[i, "PopFemale"])
        i+=1
    k+=1

plt.plot(loop, pop_male, color= "blue")
plt.plot(loop, pop_fem, color="pink")
plt.ylabel("População 1:1000")
plt.xlabel("Ano")

plt.title("População Brasileira Masculina/Feminina 1950-2019")
plt.show()